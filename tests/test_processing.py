"""
Phase-0 regression tests: lock the existing tercile / exceedance / decumulation math
and prove the package imports cleanly after the migration into eccas_s2s.
"""
import numpy as np
import xarray as xr
import pytest


def test_package_imports():
    import eccas_s2s
    assert eccas_s2s.__version__
    # the migrated modules and the model contract must import
    import eccas_s2s.config            # noqa: F401
    import eccas_s2s.core.data         # noqa: F401
    import eccas_s2s.core.processing   # noqa: F401
    from eccas_s2s.models import ForecastModel  # noqa: F401


def test_decumulate_precip_to_mm():
    from eccas_s2s.core.processing import decumulate_precip_to_mm

    # cumulative precip in metres at 24/48/72 h -> [2, 5, 9] mm cumulative
    step = np.array([np.timedelta64(h, "h") for h in (24, 48, 72)])
    tp = xr.DataArray([0.002, 0.005, 0.009], dims=["step"], coords={"step": step})
    ds = xr.Dataset({"tp": tp})

    out = decumulate_precip_to_mm(ds, clip_negative=True)
    daily = out["tp_daily"].values

    # first step (non-zero) kept as-is, rest are upper-labelled differences
    np.testing.assert_allclose(daily, [2.0, 3.0, 4.0])
    assert out["tp_daily"].attrs["units"] == "mm"


def test_exceedance_probabilities():
    from eccas_s2s.core.processing import compute_exceedance_probabilities

    members = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    fc = xr.DataArray(
        members.reshape(5, 1, 1),
        dims=["number", "latitude", "longitude"],
        coords={"number": np.arange(5), "latitude": [0.0], "longitude": [0.0]},
    )
    probs = compute_exceedance_probabilities(fc, thresholds_mm=[25, 50])
    # P(>=25) = 3/5, P(>=50) = 1/5
    assert float(probs[25].values.ravel()[0]) == pytest.approx(0.6)
    assert float(probs[50].values.ravel()[0]) == pytest.approx(0.2)


def test_tercile_probabilities_sum_to_one():
    from eccas_s2s.core.processing import compute_all_probabilities

    rng = np.random.default_rng(0)
    # hindcast: (number, time, lat, lon); forecast: (number, lat, lon)
    hind = xr.DataArray(
        rng.gamma(2.0, 40.0, size=(6, 8, 2, 2)),
        dims=["number", "time", "latitude", "longitude"],
        coords={"number": np.arange(6), "time": np.arange(8),
                "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )
    fc = xr.DataArray(
        rng.gamma(2.0, 40.0, size=(6, 2, 2)),
        dims=["number", "latitude", "longitude"],
        coords={"number": np.arange(6),
                "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )
    probs, thresholds = compute_all_probabilities(hind, fc)
    total = (probs["prob_BN"] + probs["prob_NN"] + probs["prob_AN"]).values
    np.testing.assert_allclose(total, np.ones_like(total), atol=1e-6)
