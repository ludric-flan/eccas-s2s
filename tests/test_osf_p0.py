"""Tests of the OSF phase P0 foundations: settings, provenance, periods, daily totals, agro calendars."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from eccas_s2s.agro.calendars import (ALREADY_STARTED, BEYOND_HORIZON, FEASIBLE,
                                      feasibility, load_agro_products)
from eccas_s2s.core.daily import aggregate_periods, daily_precip_from_accumulated, to_year_dim
from eccas_s2s.core.periods import Period, build_periods, periods_table
from eccas_s2s.provenance import RunContext, archive_run
from eccas_s2s.settings import ConfigError, load_cycle

REPO = Path(__file__).resolve().parents[1]
CYCLE = REPO / "config" / "cycle_202609.yaml"
LEGACY_GRIB = Path("/home/ludric/Downloads/SVM/Previsions_S2S/DOWNLOAD_C3S_DATA/grib_files/"
                   "ecmwf_seasonal_forecast_2026_08_01.grib")


# ---------------------------------------------------------------- settings
@pytest.fixture
def tmp_cycle(tmp_path):
    """The real cycle config with all paths redirected into tmp_path."""
    raw = yaml.safe_load(CYCLE.read_text(encoding="utf-8"))
    for key in ("data_root", "output_root", "archive_root"):
        raw["paths"][key] = str(tmp_path / key)
    raw["includes"] = {k: str(REPO / "config" / v) for k, v in raw["includes"].items()}
    path = tmp_path / "cycle.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_cycle(path)


def test_load_real_cycle():
    cfg = load_cycle(CYCLE)
    assert cfg.init_date == pd.Timestamp("2026-09-01")
    assert cfg.cycle_id == "202609"
    assert cfg.scales == ["decade", "month", "season"]
    assert cfg.reference_period("c3s_fit") == (1993, 2016)
    assert cfg.reference_period("obs_normal") == (1991, 2020)
    assert cfg.cv_scheme == "loyo"
    assert cfg.c3s_models["ukmo"].system == "610"
    assert cfg.c3s_models["dwd"].max_lead_days == 181
    assert len(cfg.c3s_hindcast_years) == 24
    assert "zones" in cfg.domains and "products" in cfg.agro_calendars
    assert len(cfg.sha256) == 4          # cycle file + 3 includes


def test_invalid_init_day_rejected(tmp_path):
    raw = yaml.safe_load(CYCLE.read_text(encoding="utf-8"))
    raw["cycle"]["init_date"] = "2026-09-05"
    raw.pop("includes")
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="1er du mois"):
        load_cycle(p)


def test_unknown_scale_rejected(tmp_path):
    raw = yaml.safe_load(CYCLE.read_text(encoding="utf-8"))
    raw["scales"] = ["decade", "week"]
    raw.pop("includes")
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="week"):
        load_cycle(p)


# --------------------------------------------------------------- provenance
def test_run_context_writes_manifest_and_archive(tmp_cycle, tmp_path):
    out = tmp_path / "product.nc"
    xr.Dataset({"x": ("t", [1.0, 2.0])}).to_netcdf(out)
    with RunContext(tmp_cycle, step="unit") as ctx:
        ctx.record_parameter("alpha", 1)
        ctx.record_output(out, role="test")
    m = json.loads((ctx.run_dir / "manifest.json").read_text())
    assert m["status"] == "success"
    assert m["parameters"]["alpha"] == 1
    assert m["outputs"][0]["sha256_mode"] == "full"
    assert m["cycle"]["init_date"] == "2026-09-01"
    assert "commit" in m["code"]["git"]
    assert (ctx.run_dir / "run.log").read_text().count("unit") >= 2

    dest = archive_run(ctx, files=[out])
    assert dest.parts[-3:-1] == ("2026", "09")
    assert (dest / "manifest.json").exists() and (dest / "product.nc").exists()
    with pytest.raises(FileExistsError):
        archive_run(ctx)


def test_run_context_records_failure(tmp_cycle):
    with pytest.raises(RuntimeError):
        with RunContext(tmp_cycle, step="boom") as ctx:
            raise RuntimeError("panne simulée")
    m = json.loads((ctx.run_dir / "manifest.json").read_text())
    assert m["status"] == "failed" and "panne simulée" in m["error"]


# ------------------------------------------------------------------ periods
def test_periods_september_init_184_days():
    periods = build_periods(pd.Timestamp("2026-09-01"), horizon_days=184)
    tab = periods_table(periods, 2026)
    # 184 days from 2026-09-01 -> last day 2027-03-03: Sep..Feb complete
    assert list(tab.query("scale == 'month'")["label"]) == [
        "2026-09", "2026-10", "2026-11", "2026-12", "2027-01", "2027-02"]
    assert list(tab.query("scale == 'season'")["label"]) == [
        "SON 2026", "OND 2026", "NDJ 2026-27", "DJF 2026-27"]
    assert (tab["scale"] == "decade").sum() == 18
    first = tab.iloc[0]
    assert (first["start"], first["end"], first["first_lead_day"]) == (
        pd.Timestamp("2026-09-01").date(), pd.Timestamp("2026-09-10").date(), 0)


def test_effective_common_horizon_181_keeps_djf():
    """DWD delivers 181 days (to 2027-02-28): all periods up to DJF must remain."""
    tab = periods_table(build_periods(pd.Timestamp("2026-09-01"), horizon_days=181), 2026)
    assert tab.query("scale == 'season'")["label"].iloc[-1] == "DJF 2026-27"
    assert (tab["scale"] == "decade").sum() == 18


def test_decade_d3_and_leap_years():
    feb_d3 = Period("decade", init_month=9, month_offset=5, decade=3)
    assert feb_d3.dates(2026) == (pd.Timestamp("2027-02-21"), pd.Timestamp("2027-02-28"))
    assert feb_d3.dates(1995) == (pd.Timestamp("1996-02-21"), pd.Timestamp("1996-02-29"))
    assert feb_d3.n_days(1995) == 9 and feb_d3.n_days(2026) == 8
    djf = Period("season", 9, 3, n_months=3)
    assert djf.label(2026) == "DJF 2026-27"
    # same period key, different lead days in a leap hindcast year
    assert djf.day_slice(1995)[1] == djf.day_slice(2026)[1] + 1


def test_labels_fr():
    # noms de mois explicites, comme sur les cartes de la chaîne de référence
    assert Period("decade", 9, 0, decade=1).label_fr(2026) == "1ʳᵉ décade de Septembre 2026"
    assert Period("decade", 9, 1, decade=2).label_fr(2026) == "2ᵉ décade de Octobre 2026"
    assert Period("month", 9, 2).label_fr(2026) == "Novembre 2026"
    assert Period("season", 9, 1, n_months=3).label_fr(2026) == "OND 2026"
    assert Period("month", 9, 2).label_fr(2026, with_dates=True) == \
        "Novembre 2026 (01 nov. – 30 nov. 2026)"


# ---------------------------------------------------------- daily totals
def _synthetic_accumulated(n_days=40, years=(2000, 2001)):
    """Accumulated tp (m): rain of lead day i is (i + 1) mm."""
    rain_mm = np.arange(1, n_days + 1, dtype=float)
    acc_m = np.cumsum(rain_mm) / 1000.0
    data = np.broadcast_to(acc_m[None, :, None, None], (len(years), n_days, 2, 2)).copy()
    step = pd.to_timedelta(np.arange(1, n_days + 1), unit="D")
    time = pd.to_datetime([f"{y}-09-01" for y in years])
    return xr.DataArray(data, dims=("time", "step", "latitude", "longitude"),
                        coords={"time": time, "step": step,
                                "valid_time": ("step", pd.Timestamp("2000-09-01") + step)})


def test_daily_attribution_is_day_of_rain():
    daily = to_year_dim(daily_precip_from_accumulated(_synthetic_accumulated()))
    assert int(daily["lead_day"][0]) == 0
    np.testing.assert_allclose(daily.isel(year=0, latitude=0, longitude=0).values[:3], [1, 2, 3])


def test_negative_increments_clipped_and_counted():
    tp = _synthetic_accumulated()
    tp[0, 5, 0, 0] = tp[0, 4, 0, 0] - 0.0001    # artificial decrease
    daily = daily_precip_from_accumulated(tp)
    assert daily.attrs["n_negative_clipped"] >= 1
    assert float(daily.min()) >= 0


def test_non_contiguous_steps_rejected():
    tp = _synthetic_accumulated().isel(step=[0, 1, 3])
    with pytest.raises(ValueError, match="contiguës"):
        daily_precip_from_accumulated(tp)


def test_aggregate_decade_sum():
    daily = to_year_dim(daily_precip_from_accumulated(_synthetic_accumulated()))
    periods = build_periods(pd.Timestamp("2000-09-01"), horizon_days=40, scales=("decade",))
    tot = aggregate_periods(daily, periods, how="sum")
    # D1 = lead days 0..9 = 1+...+10 = 55 mm
    assert float(tot.sel(period="decade_m0_d1").isel(year=0, latitude=0, longitude=0)) == 55.0
    assert tot.attrs["units"] == "mm"


def test_aggregate_missing_days_give_nan():
    daily = to_year_dim(daily_precip_from_accumulated(_synthetic_accumulated(n_days=15)))
    p = Period("decade", 9, 0, decade=2)            # lead days 10..19, only 10..14 exist
    tot = aggregate_periods(daily, [p], how="sum")
    assert bool(tot.isnull().all())


@pytest.mark.skipif(not LEGACY_GRIB.exists(), reason="fichier GRIB legacy absent")
def test_equivalence_with_legacy_convention_on_real_grib():
    """New decade D1 (1-10 Aug) == legacy sum over valid_time 2-11 Aug: only the dates move."""
    from eccas_s2s.core.processing import decumulate_precip_to_mm
    from eccas_s2s.io.c3s_read import load_c3s_precip_daily, open_c3s_grib

    new = load_c3s_precip_daily(LEGACY_GRIB).isel(number=slice(0, 3))
    d1 = aggregate_periods(new, [Period("decade", 8, 0, decade=1)], how="sum")

    ds = open_c3s_grib(LEGACY_GRIB).isel(number=slice(0, 3))
    legacy = decumulate_precip_to_mm(ds)["tp_daily"].swap_dims({"step": "valid_time"})
    legacy_d1 = legacy.sel(valid_time=slice("2026-08-02", "2026-08-11")).sum("valid_time")
    np.testing.assert_allclose(d1.isel(year=0, period=0).values, legacy_d1.values, rtol=1e-5, atol=1e-3)


# ------------------------------------------------------------ agro calendars
def test_agro_feasibility_september_init():
    cfg = load_cycle(CYCLE)
    products = load_agro_products(cfg.agro_calendars)
    init = pd.Timestamp("2026-09-01")
    status = {n: feasibility(p, init, 184)["status"] for n, p in products.items()}
    assert status["ceeac_son"] == FEASIBLE
    assert status["ceeac_ndj"] == FEASIBLE
    assert status["ceeac_djf"] == FEASIBLE
    assert status["south_onset"] == ALREADY_STARTED
    assert status["equatorial_son"] == ALREADY_STARTED
    assert status["north_cessation"] == ALREADY_STARTED
    assert status["south_cessation"] == BEYOND_HORIZON
    assert status["north_onset"] == BEYOND_HORIZON
    # recommended init months of v10 make their own product feasible (215-day models)
    for name, p in products.items():
        rec = pd.Timestamp(2026, p.recommended_init, 1)
        assert feasibility(p, rec, 215)["status"] == FEASIBLE, name
