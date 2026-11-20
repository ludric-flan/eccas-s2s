"""
Robustness of the monthly cycle (fixes of 19/11/2026).

* a partial calibration run must not erase what the other models wrote;
* a tie between the raw and the calibrated score is not an improvement;
* the CHIRPS file of the year in progress is refreshed when UCSB publishes more;
* several C3S variables can be requested at once, with their usual aliases.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eccas_s2s.operations import download_chirps
from eccas_s2s.operations.calibrate_hindcast import COMPARISON_KEYS
from eccas_s2s.operations.download_c3s import canonical_variables
from eccas_s2s.registry import merge_rows, registry_dir
from eccas_s2s.validate.comparison import compare_maps, recommend


def _rows(model, gain, n_periods=2):
    return pd.DataFrame([{"system": "c3s", "model": model, "variable": "precip",
                          "scale": "season", "product": "terciles", "metric": "rpss",
                          "method": "ngr", "period": f"season_m{k}", "median_gain": gain,
                          "fraction_improved": 0.8, "fraction_repaired": 0.1,
                          "fraction_broken": 0.0, "deciding": True}
                         for k in range(n_periods)])


def test_partial_run_keeps_the_other_models(tmp_path):
    """Incident of 17/11/2026: a run on NCEP alone left a 13-line register."""
    path = tmp_path / "comparison_raw_calibrated.csv"
    pd.concat([_rows("ecmwf", 0.02), _rows("ncep", 0.01)]).to_csv(path, index=False)

    merged = merge_rows(_rows("ncep", 0.05), path, COMPARISON_KEYS)

    assert sorted(merged["model"].unique()) == ["ecmwf", "ncep"]
    assert len(merged) == 4                                  # ncep replaced, not duplicated
    assert (merged[merged.model == "ncep"]["median_gain"] == 0.05).all()
    register = recommend(merged[merged["deciding"]])
    assert sorted(register["model"]) == ["ecmwf", "ncep"]


def test_registry_is_one_folder_per_cycle(tmp_path):
    cfg = SimpleNamespace(cycle_id="202611", path_of=lambda key: tmp_path)
    assert registry_dir(cfg) == tmp_path / "registry" / "202611"


def test_a_tie_is_not_an_improvement():
    """Rounding noise between equal scores must not count as a gain."""
    rng = np.random.default_rng(0)
    raw = xr.DataArray(rng.uniform(-0.2, 0.2, (20, 30)), dims=("latitude", "longitude"),
                       coords={"latitude": np.linspace(-5, 5, 20),
                               "longitude": np.linspace(10, 25, 30)})
    noisy = raw + rng.uniform(-1e-9, 1e-9, raw.shape)      # same score, float noise
    mask = xr.ones_like(raw, dtype=bool)
    row = compare_maps(noisy, raw, "rpss", mask)
    assert row["fraction_improved"] == 0.0
    better = compare_maps(raw + 0.01, raw, "rpss", mask)
    assert better["fraction_improved"] == 1.0
    # a method whose only "gain" is noise is never recommended
    table = pd.DataFrame([{**row, "system": "c3s", "model": "m", "variable": "precip",
                           "scale": "season", "median_gain": 1e-9, "fraction_improved": 0.9}])
    assert recommend(table)["method"].iat[0] == "raw"


def _chirps_file(path, last_day, stamp=None):
    time = pd.date_range(f"{pd.Timestamp(last_day).year}-01-01", last_day, freq="D")
    ds = xr.Dataset({"precip": (("time", "latitude", "longitude"),
                                np.zeros((len(time), 2, 2), "float32"))},
                    coords={"time": time, "latitude": [0.125, 0.375],
                            "longitude": [10.125, 10.375]})
    if stamp:
        ds.attrs["source_last_modified"] = stamp
    ds.to_netcdf(path)
    return path


class _Ctx:
    log = logging.getLogger("test")

    def __init__(self):
        self.warnings = []

    def warn(self, msg):
        self.warnings.append(msg)


@pytest.fixture
def ucsb(monkeypatch):
    """The stamp UCSB would answer, set by each test."""
    state = {"stamp": "Fri, 13 Nov 2026 22:38:11 GMT"}
    monkeypatch.setattr(download_chirps, "remote_stamp", lambda url, timeout=60: state["stamp"])
    return state


def test_complete_year_is_never_downloaded_again(tmp_path, ucsb):
    f = _chirps_file(tmp_path / "y2025.nc", "2025-12-31")
    assert download_chirps._needs_download(_Ctx(), 2025, f, "url", refresh=False)[0] is False


def test_year_in_progress_is_refreshed_when_ucsb_publishes(tmp_path, ucsb):
    f = _chirps_file(tmp_path / "y2026.nc", "2026-09-30", stamp="Tue, 13 Oct 2026 10:00:00 GMT")
    needed, stamp = download_chirps._needs_download(_Ctx(), 2026, f, "url", refresh=False)
    assert needed and stamp == ucsb["stamp"]


def test_year_in_progress_kept_when_nothing_new(tmp_path, ucsb):
    f = _chirps_file(tmp_path / "y2026.nc", "2026-10-31", stamp=ucsb["stamp"])
    assert download_chirps._needs_download(_Ctx(), 2026, f, "url", refresh=False)[0] is False
    # --refresh overrides
    assert download_chirps._needs_download(_Ctx(), 2026, f, "url", refresh=True)[0] is True


def test_file_without_stamp_is_refreshed_once(tmp_path, ucsb):
    f = _chirps_file(tmp_path / "y2026.nc", "2026-10-31")
    assert download_chirps._needs_download(_Ctx(), 2026, f, "url", refresh=False)[0] is True


def test_default_chirps_year_is_the_last_elapsed_month():
    assert download_chirps.default_years(SimpleNamespace(init_date=pd.Timestamp("2026-11-01"))) \
        == (2026, 2026)
    assert download_chirps.default_years(SimpleNamespace(init_date=pd.Timestamp("2027-01-01"))) \
        == (2026, 2026)


def test_variables_list_and_aliases():
    assert canonical_variables(["precip", "tmoy"]) == ["precip", "t2m"]
    assert canonical_variables("pluie") == ["precip"]
    assert canonical_variables(["t2m", "tmoy", "TMAX"]) == ["t2m", "tmax"]
    with pytest.raises(ValueError):
        canonical_variables(["precip", "vent"])


# ---------------------------------------------------------------- QC des deux flux C3S
from pathlib import Path  # noqa: E402

from eccas_s2s.io.c3s_qc import (effective_horizons, inspect_c3s_file,  # noqa: E402
                                 monthly_horizons, stream_of)

RAW_09 = Path("/home/ludric/Downloads/SVM/Previsions_S2S/DATA_OSF/raw/c3s/202609")
UKMO_MONTHLY_FC = RAW_09 / "c3s_ukmo_610_PRCP_forecast_2026_09_monthly.grib"


def test_stream_is_read_from_the_file_name():
    assert stream_of("c3s_ncep_2_PRCP_hindcast_1993_2016_11_monthly.grib") == "monthly"
    assert stream_of("c3s_ecmwf_51_TMAX_forecast_2026_11.grib") == "daily"


def test_horizons_are_split_by_stream():
    qc = pd.DataFrame([
        {"centre": "ukmo", "stream": "daily", "kind": "forecast", "horizon_all_years_days": 215,
         "horizon_any_year_days": 215, "members": 2},
        {"centre": "ukmo", "stream": "daily", "kind": "hindcast", "horizon_all_years_days": 215,
         "horizon_any_year_days": 215, "members": 7},
        {"centre": "ukmo", "stream": "monthly", "kind": "forecast",
         "horizon_all_years_months": 6, "members": 62, "members_min": 62},
        {"centre": "ukmo", "stream": "monthly", "kind": "hindcast",
         "horizon_all_years_months": 5, "members": 28, "members_min": 26},
    ])
    daily = effective_horizons(qc).iloc[0]
    assert (daily["usable_days"], daily["members_hindcast"]) == (215, 7)
    monthly = monthly_horizons(qc).iloc[0]
    assert (monthly["usable_months"], monthly["members_hindcast"],
            monthly["members_hindcast_min"]) == (5, 28, 26)


@pytest.mark.skipif(not UKMO_MONTHLY_FC.exists(), reason="fichier C3S mensuel absent")
def test_monthly_file_is_controlled_with_its_lagged_members():
    """UKMO: the monthly stream carries the lagged starts the daily one does not."""
    row = inspect_c3s_file(UKMO_MONTHLY_FC)
    assert row["stream"] == "monthly" and row["lagged_ensemble"]
    assert row["members"] > 10 and row["start_dates"] > 1
    assert row["horizon_all_years_months"] >= 6 and row["first_step_month"] == 0


# ------------------------------------------------- sélecteur et opérations échelle par échelle
CYCLE_202611 = Path(__file__).resolve().parents[1] / "config" / "cycle_202611.yaml"


@pytest.mark.skipif(not CYCLE_202611.exists(), reason="configuration du cycle 202611 absente")
def test_multi_scale_selection_works_scale_by_scale():
    """Incident of 20/11/2026: the month pass rejected "all-seasons" as unknown."""
    from eccas_s2s.core.periods import SelectionError
    from eccas_s2s.settings import load_cycle

    cfg = load_cycle(str(CYCLE_202611))
    both = ["all-months", "all-seasons"]
    months = cfg.periods_for(215, ("month",), selection=both)
    seasons = cfg.periods_for(215, ("season",), selection=both)
    assert {p.scale for p in months} == {"month"} and len(months) == 6
    assert {p.scale for p in seasons} == {"season"} and len(seasons) == 4
    assert cfg.periods_for(215, ("decade",), selection=both) == []
    with pytest.raises(SelectionError):                 # a typo still stops the step
        cfg.periods_for(215, ("month",), selection=["all-saisons"])
