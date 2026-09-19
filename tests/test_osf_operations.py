"""Tests of the operational layer: C3S file QC helpers and the download step (dry run, no network)."""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from eccas_s2s.io.c3s_qc import effective_horizons, parse_c3s_filename
from eccas_s2s.operations import download_c3s

REPO = Path(__file__).resolve().parents[1]
CYCLE = REPO / "config" / "cycle_202609.yaml"


@pytest.fixture
def tmp_cycle_file(tmp_path):
    raw = yaml.safe_load(CYCLE.read_text(encoding="utf-8"))
    for key in ("data_root", "output_root", "archive_root"):
        raw["paths"][key] = str(tmp_path / key)
    raw["includes"] = {k: str(REPO / "config" / v) for k, v in raw["includes"].items()}
    path = tmp_path / "cycle.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def test_parse_c3s_filename():
    info = parse_c3s_filename("c3s_meteo_france_9_PRCP_hindcast_1993_2016_09.grib")
    assert info == {"centre": "meteo_france", "system": "9", "var": "PRCP", "kind": "hindcast"}
    with pytest.raises(ValueError):
        parse_c3s_filename("ecmwf_hindcast_1993_2016_09_01.grib")


def test_effective_horizon_is_min_of_forecast_and_hindcast():
    qc = pd.DataFrame([
        {"centre": "dwd", "kind": "forecast", "horizon_all_years_days": 181,
         "horizon_any_year_days": 181, "members": 50},
        {"centre": "dwd", "kind": "hindcast", "horizon_all_years_days": 181,
         "horizon_any_year_days": 182, "members": 30},
    ])
    h = effective_horizons(qc).iloc[0]
    assert (h["usable_days"], h["members_forecast"], h["members_hindcast"]) == (181, 50, 30)


def test_download_dry_run_records_requests(tmp_cycle_file):
    ctx = download_c3s.run(str(tmp_cycle_file), "precip", models=["ecmwf", "dwd"], dry_run=True)
    m = json.loads((ctx.run_dir / "manifest.json").read_text())
    assert m["status"] == "success" and m["outputs"] == []
    req = m["parameters"]["request.dwd.hindcast"]
    assert req["system"] == "22" and len(req["year"]) == 24
    assert req["leadtime_hour"] == "24..4344 (pas 24 h)"     # 181 days
    assert req["area"] == [30.0, -10.0, -25.0, 40.0]


def test_download_rejects_unknown_model(tmp_cycle_file):
    with pytest.raises(ValueError, match="absents"):
        download_c3s.run(str(tmp_cycle_file), "precip", models=["xyz"], dry_run=True)
