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


def test_default_is_relative_15_percent(thresholds):
    s = dry_mask_settings(thresholds)
    assert s["method"] == "relative" and s["relative_fraction"] == 0.15


def test_relative_mask(thresholds):
    s = dry_mask_settings(thresholds)
    fc = _field([1.0, 20.0, 100.0, np.nan])
    clim = _field([10.0, 100.0, 100.0, 100.0])
    m = dry_mask(fc, s, clim)
    # 1 < 1.5 dry ; 20 >= 15 not dry ; 100 not dry ; NaN not flagged
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
    s2 = dry_mask_settings(thresholds, relative_fraction=0.3)
    m = dry_mask(_field([20.0]), s2, _field([100.0]))
    assert bool(m[0])


def test_invalid_settings(thresholds):
    with pytest.raises(ValueError):
        dry_mask_settings(thresholds, method="foo")
    with pytest.raises(ValueError):
        dry_mask_settings(thresholds, relative_fraction=1.5)
    with pytest.raises(ValueError, match="moyenne climatologique"):
        dry_mask(_field([1.0]), dry_mask_settings(thresholds))


def test_forecast_value_and_apply():
    ens = xr.DataArray([[1.0, 2.0], [3.0, 10.0], [5.0, 30.0]], dims=("number", "longitude"))
    assert forecast_value(ens, "mean").values.tolist() == [3.0, 14.0]
    assert forecast_value(ens, "median").values.tolist() == [3.0, 10.0]
    out = apply_mask(xr.DataArray([7.0, 8.0], dims="longitude"), xr.DataArray([True, False], dims="longitude"))
    assert np.isnan(out[0]) and out[1] == 8.0
