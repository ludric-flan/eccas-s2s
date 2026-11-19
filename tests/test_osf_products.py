"""Tests of the dry-season mask (products/masks)."""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eccas_s2s.products.masks import apply_mask, dry_mask, dry_mask_settings, forecast_value
from eccas_s2s.settings import load_cycle

CYCLE = Path(__file__).resolve().parents[1] / "config" / "cycle_202609.yaml"


@pytest.fixture
def thresholds():
    return load_cycle(CYCLE).thresholds


def _field(values, periods=None, scales=None):
    values = np.asarray(values, dtype=float)
    if periods is None:
        return xr.DataArray(values, dims=("longitude",), coords={"longitude": np.arange(values.size)})
    return xr.DataArray(values, dims=("period", "longitude"),
                        coords={"period": periods, "scale": ("period", scales),
                                "longitude": np.arange(values.shape[1])})


def test_default_is_iri_15_percent_with_30_mm_floor(thresholds):
    s = dry_mask_settings(thresholds)
    assert s["method"] == "iri" and s["iri_share_pct"] == 15 and s["iri_floor_mm"] == 30


def test_iri_mask_season(thresholds):
    s = dry_mask_settings(thresholds)
    fc = _field([1.0, 1.0, 1.0, 1.0, np.nan])
    clim = _field([100.0, 200.0, 20.0, 60.0, 100.0])      # normal of the season (mm)
    annual = _field([1000.0, 1000.0, 50.0, 300.0, 100.0])  # annual normal (mm)
    m = dry_mask(fc, s, clim, annual_mean=annual, period_days=92)
    # 10 % < 15 % dry ; 20 % wet ; 40 % of 50 mm but 20 mm < 30 mm floor -> dry ;
    # 20 % and 60 mm -> wet ; missing forecast cell never flagged
    assert m.values.tolist() == [True, False, True, False, False]


def test_iri_thresholds_scale_with_period(thresholds):
    s = dry_mask_settings(thresholds)
    annual = _field([1200.0, 1200.0, 1200.0])
    fc = _field([1.0, 1.0, 1.0])
    # a month: 5 % of 1200 = 60 mm, floor 10 mm
    month = dry_mask(fc, s, _field([59.0, 61.0, 61.0]), annual_mean=annual, period_days=31)
    assert month.values.tolist() == [True, False, False]
    # a 90- or 92-day season reads exactly 15 % (180 mm) thanks to whole months
    for days in (90, 92):
        season = dry_mask(fc, s, _field([179.0, 181.0, 400.0]), annual_mean=annual, period_days=days)
        assert season.values.tolist() == [True, False, False]


def test_relative_mask(thresholds):
    s = dry_mask_settings(thresholds, method="relative")
    fc = _field([1.0, 20.0, 100.0, np.nan])
    clim = _field([10.0, 100.0, 100.0, 100.0])
    annual = _field([1000.0, 1000.0, 100.0, 1000.0])
    m = dry_mask(fc, s, clim, annual_mean=annual, period_days=365.25 / 4)
    # limit = 0.15 x annual/4: 37.5, 37.5, 3.75, 37.5 -> 10 dry ; 100 wet ; 100 wet ; NaN not flagged
    assert m.values.tolist() == [True, False, False, False]


def test_absolute_mask_by_scale_along_period(thresholds):
    s = dry_mask_settings(thresholds, method="absolute")
    fc = _field([[4.0, 6.0], [10.0, 20.0], [40.0, 50.0]], periods=["d", "m", "s"],
                scales=["decade", "month", "season"])
    m = dry_mask(fc, s)
    # thresholds: decade 5, month 15, season 45
    assert m.values.tolist() == [[True, False], [True, False], [True, False]]


def test_runtime_overrides(thresholds):
    s = dry_mask_settings(thresholds, method="absolute", absolute_mm={"season": 100})
    assert s["absolute_mm"]["season"] == 100 and s["absolute_mm"]["decade"] == 5
    s2 = dry_mask_settings(thresholds, iri_share_pct=30)
    m = dry_mask(_field([1.0]), s2, _field([250.0]), annual_mean=_field([1000.0]), period_days=91)
    assert bool(m[0])


def test_invalid_settings(thresholds):
    with pytest.raises(ValueError):
        dry_mask_settings(thresholds, method="foo")
    with pytest.raises(ValueError):
        dry_mask_settings(thresholds, relative_fraction=1.5)
    with pytest.raises(ValueError):
        dry_mask_settings(thresholds, iri_share_pct=150)
    with pytest.raises(ValueError, match="climatologie annuelle"):
        dry_mask(_field([1.0]), dry_mask_settings(thresholds), _field([1.0]))


def test_forecast_value_and_apply():
    ens = xr.DataArray([[1.0, 2.0], [3.0, 10.0], [5.0, 30.0]], dims=("number", "longitude"))
    assert forecast_value(ens, "mean").values.tolist() == [3.0, 14.0]
    assert forecast_value(ens, "median").values.tolist() == [3.0, 10.0]
    out = apply_mask(xr.DataArray([7.0, 8.0], dims="longitude"), xr.DataArray([True, False], dims="longitude"))
    assert np.isnan(out[0]) and out[1] == 8.0
