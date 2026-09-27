"""
Raw products must reproduce the reference chain (run_forecast_v2).

Each test re-implements the reference formula inline — from
`s2s_processing_v2.py` and `s2s_spi.py` — and checks our module against it.
Re-deriving these conventions from memory is what produced two calculation
errors in phase P3; these tests are there so it cannot happen silently again.
"""
import numpy as np
import pytest
import xarray as xr

from eccas_s2s.products.raw_products import (anomaly, anomaly_threshold_probabilities,
                                             cumul_to_spi, exceedance_probabilities,
                                             fit_gamma, model_climatology,
                                             quintile_and_median_probabilities, spi_products,
                                             tercile_probabilities)


def _hindcast_forecast(n_years=24, n_members=12, n_cells=4, seed=0):
    rng = np.random.default_rng(seed)
    years, members = np.arange(1993, 1993 + n_years), np.arange(n_members)
    cells = np.arange(n_cells) * 1.0
    hind = xr.DataArray(rng.gamma(3.0, 60.0, size=(n_years, n_members, n_cells)),
                        dims=("year", "number", "longitude"),
                        coords={"year": years, "number": members, "longitude": cells})
    fcst = xr.DataArray(rng.gamma(3.0, 70.0, size=(n_members, n_cells)),
                        dims=("number", "longitude"),
                        coords={"number": members, "longitude": cells})
    return hind, fcst


def test_climatology_pools_members_and_years_like_the_reference():
    hind, _ = _hindcast_forecast()
    clim = model_climatology(hind)
    ref = hind.stack(sample=("year", "number"))
    for name, q in (("tercile_lower", 1 / 3), ("tercile_upper", 2 / 3),
                    ("quintile_lower", 0.20), ("quintile_upper", 0.80), ("median", 0.50)):
        expected = ref.quantile(q, dim="sample").drop_vars("quantile")
        np.testing.assert_allclose(clim[name].values, expected.values)
    np.testing.assert_allclose(clim["clim_mean"].values, hind.mean(("year", "number")).values)


def test_dry_mask_is_the_one_millimetre_rule():
    hind, _ = _hindcast_forecast()
    dry = hind * 0.0 + 0.5                       # half a millimetre everywhere
    assert bool(model_climatology(dry)["dry_mask"].all())
    assert not bool(model_climatology(hind)["dry_mask"].any())


def test_tercile_probabilities_follow_the_reference_comparisons():
    """BN strictly below, AN strictly above, NN the closed interval — they sum to one."""
    hind, fcst = _hindcast_forecast()
    clim = model_climatology(hind)
    prob = tercile_probabilities(fcst, clim)
    n = fcst.sizes["number"]
    np.testing.assert_allclose(
        prob.sel(category="BN").values,
        ((fcst < clim["tercile_lower"]).sum("number") / n).values)
    np.testing.assert_allclose(
        prob.sel(category="AN").values,
        ((fcst > clim["tercile_upper"]).sum("number") / n).values)
    np.testing.assert_allclose(prob.sum("category").values, 1.0, atol=1e-12)


def test_quintile_and_median_use_strict_comparisons():
    hind, fcst = _hindcast_forecast()
    clim = model_climatology(hind)
    ds = quintile_and_median_probabilities(fcst, clim)
    n = fcst.sizes["number"]
    np.testing.assert_allclose(ds["prob_low20"].values,
                               ((fcst < clim["quintile_lower"]).sum("number") / n).values)
    np.testing.assert_allclose(ds["prob_exceed_median"].values,
                               ((fcst > clim["median"]).sum("number") / n).values)


def test_exceedance_includes_the_threshold_itself():
    """`>=` and not `>`: "at least 200 mm" includes exactly 200 mm."""
    fcst = xr.DataArray([[100.0, 200.0, 300.0]], dims=("longitude", "number"),
                        coords={"longitude": [0.0], "number": [0, 1, 2]})
    prob = exceedance_probabilities(fcst, [200.0])
    assert float(prob.sel(threshold=200.0)) == pytest.approx(2 / 3)


def test_anomaly_is_against_the_model_climatology():
    hind, fcst = _hindcast_forecast()
    clim = model_climatology(hind)
    anom = anomaly(fcst, clim["clim_mean"])
    np.testing.assert_allclose(anom.values, (fcst - hind.mean(("year", "number"))).values)


def test_anomaly_threshold_probabilities_partition_the_members():
    hind, fcst = _hindcast_forecast()
    clim = model_climatology(hind)
    ds = anomaly_threshold_probabilities(anomaly(fcst, clim["clim_mean"]), 50.0)
    total = ds["prob_athr_above"] + ds["prob_athr_normal"] + ds["prob_athr_below"]
    np.testing.assert_allclose(total.values, 1.0, atol=1e-12)


# ------------------------------------------------------------------------ SPI
def test_gamma_fit_matches_the_method_of_moments_on_wet_samples():
    rng = np.random.default_rng(3)
    samples = rng.gamma(2.5, 40.0, size=(500, 2))
    samples[:50, 0] = 0.0                                  # some dry years
    alpha, theta, p0, valid = fit_gamma(samples)
    wet = np.where(samples > 0.1, samples, np.nan)
    m, v = np.nanmean(wet, axis=0), np.nanvar(wet, axis=0, ddof=1)
    np.testing.assert_allclose(alpha, m ** 2 / v)
    np.testing.assert_allclose(theta, v / m)
    assert p0[0] == pytest.approx(50 / 500) and p0[1] == pytest.approx(0.0)
    assert valid.all()


def test_spi_of_the_median_is_near_zero_and_classes_sum_to_one():
    hind, _ = _hindcast_forecast(n_years=40, n_members=20, seed=5)
    median = hind.stack(sample=("year", "number")).median("sample")
    fcst = median.expand_dims(number=np.arange(3)).transpose("number", "longitude")
    ds = spi_products(hind, fcst)
    assert abs(float(ds["spi"].mean())) < 0.15
    total = ds["prob_spi_dry"] + ds["prob_spi_normal"] + ds["prob_spi_wet"]
    np.testing.assert_allclose(total.values, 1.0, atol=1e-12)


def test_spi_is_negative_for_a_dry_forecast_and_positive_for_a_wet_one():
    hind, _ = _hindcast_forecast(n_years=40, n_members=20, seed=7)
    clim = model_climatology(hind)
    dry = (clim["quintile_lower"] * 0.5).expand_dims(number=[0]).transpose("number", "longitude")
    wet = (clim["quintile_upper"] * 2.0).expand_dims(number=[0]).transpose("number", "longitude")
    assert float(spi_products(hind, dry)["spi"].mean()) < -0.8
    assert float(spi_products(hind, wet)["spi"].mean()) > 0.8


def test_spi_is_clamped_and_finite_on_a_zero_forecast():
    hind, _ = _hindcast_forecast(n_years=40, n_members=20, seed=11)
    pooled = hind.stack(sample=("year", "number")).transpose("sample", ...)
    alpha, theta, p0, valid = fit_gamma(pooled.values)
    spi = cumul_to_spi(np.zeros((1, hind.sizes["longitude"])), alpha, theta, p0, valid)
    assert np.all(np.isfinite(spi)) and float(spi.min()) >= -3.0


def test_verification_probabilities_use_the_same_comparisons_as_the_products():
    """
    The verification counts members exactly as the products do.

    Only the thresholds differ, and deliberately: leave-one-year-out in
    verification (decision D10), whole hindcast in operations.
    """
    from eccas_s2s.validate.pairs import tercile_probabilities as verif_probs

    hind, _ = _hindcast_forecast(n_years=24, n_members=12, seed=13)
    # thresholds of the whole hindcast on both sides: the two must then agree
    clim = model_climatology(hind)
    product = tercile_probabilities(hind.isel(year=0), clim)
    q33, q67 = clim["tercile_lower"], clim["tercile_upper"]
    ens = hind.isel(year=0)
    manual_bn = (ens < q33).mean("number")
    manual_an = (ens > q67).mean("number")
    np.testing.assert_allclose(product.sel(category="BN").values, manual_bn.values)
    np.testing.assert_allclose(product.sel(category="AN").values, manual_an.values)
    # the verification function uses the same comparisons (its thresholds are LOYO)
    loyo = verif_probs(hind)
    np.testing.assert_allclose(loyo.sum("category").values, 1.0, atol=1e-12)
