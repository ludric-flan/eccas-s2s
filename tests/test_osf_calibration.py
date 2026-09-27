"""Tests of the calibration methods (phase P3): closed forms, LOYO, and what each one fixes."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eccas_s2s.calibrate.base import (EnsembleDistribution, NormalDistribution,
                                      loyo_mean_and_var, to_ensemble)
from eccas_s2s.calibrate.bias import BiasCorrection, bias_calibrators
from eccas_s2s.calibrate.qmap import QuantileMapping


def _hindcast(n_years=24, n_members=10, n_cells=5, seed=0, bias=40.0, scale=0.5):
    """
    A model that is too wet (+bias), too flat (variability scaled by `scale`)
    and correlated with the truth: the three defects the corrections address.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(1993, 1993 + n_years)
    truth = rng.gamma(4.0, 60.0, size=(n_years, n_cells))
    obs = truth + rng.normal(scale=15.0, size=truth.shape)
    signal = truth.mean() + (truth - truth.mean()) * scale + bias
    members = signal[:, None, :] + rng.normal(scale=25.0, size=(n_years, n_members, n_cells))
    dims = ("year", "number", "longitude")
    coords = {"year": years, "number": np.arange(n_members), "longitude": np.arange(n_cells) * 1.0}
    return (xr.DataArray(members, dims=dims, coords=coords),
            xr.DataArray(obs, dims=("year", "longitude"),
                         coords={"year": years, "longitude": coords["longitude"]}))


# --------------------------------------------------------------- LOYO helpers
def test_loyo_mean_and_var_match_the_explicit_loop():
    """The closed form must equal the year-by-year computation it replaces."""
    da = xr.DataArray(np.array([[1.0, 2.0], [3.0, 5.0], [8.0, 13.0], [21.0, 34.0]]),
                      dims=("year", "longitude"),
                      coords={"year": [1, 2, 3, 4], "longitude": [0.0, 1.0]})
    mean, var = loyo_mean_and_var(da)
    for y in da["year"].values:
        others = da.sel(year=[v for v in da["year"].values if v != y])
        assert float(mean.sel(year=y)[0]) == pytest.approx(float(others[:, 0].mean()))
        assert float(var.sel(year=y)[0]) == pytest.approx(float(others[:, 0].var(ddof=1)))


def test_to_ensemble_adds_a_member_axis_for_nmme():
    da = xr.DataArray([1.0, 2.0], dims="year", coords={"year": [1, 2]})
    assert to_ensemble(da).sizes["number"] == 1


# ------------------------------------------------------------ bias correction
def test_mean_correction_removes_the_bias_and_keeps_the_spread():
    ens, obs = _hindcast()
    out = BiasCorrection("mean", "precip").fit_predict_loyo(ens, obs)
    raw_bias = float((ens.mean("number") - obs).mean())
    new_bias = float((out.mean() - obs).mean())
    assert raw_bias > 30 and abs(new_bias) < 3
    # the ensemble spread is untouched: this correction moves the whole ensemble
    assert float(out.spread().mean()) == pytest.approx(float(ens.std("number", ddof=1).mean()),
                                                       rel=1e-6)


def test_scaling_restores_the_interannual_variability():
    ens, obs = _hindcast(scale=0.4)
    sd_obs = float(obs.std("year", ddof=1).mean())
    sd_raw = float(ens.mean("number").std("year", ddof=1).mean())
    out = BiasCorrection("scaling", "precip").fit_predict_loyo(ens, obs)
    sd_cal = float(out.mean().std("year", ddof=1).mean())
    assert sd_raw < 0.7 * sd_obs                      # the model is too flat
    assert abs(sd_cal - sd_obs) < 0.25 * sd_obs       # the correction restores it


def test_ratio_correction_never_produces_negative_rain():
    ens, obs = _hindcast(bias=120.0)
    out = BiasCorrection("ratio", "precip").fit_predict_loyo(ens, obs)
    assert float(out.members.min()) >= 0.0


def test_loyo_prediction_does_not_use_its_own_year():
    """Changing one year must leave that year's own calibrated value untouched."""
    ens, obs = _hindcast(seed=3)
    cal = BiasCorrection("mean", "precip")
    before = cal.fit_predict_loyo(ens, obs).mean().sel(year=2000)
    obs2 = obs.copy()
    obs2.loc[dict(year=2005)] = obs2.sel(year=2005) + 500.0     # another year is altered
    after = cal.fit_predict_loyo(ens, obs2).mean().sel(year=2000)
    assert not np.allclose(before, after)      # the fit of 2000 uses 2005: it must move
    # ... but its own year never enters its own fit:
    obs3 = obs.copy()
    obs3.loc[dict(year=2000)] = obs3.sel(year=2000) + 500.0
    same = cal.fit_predict_loyo(ens, obs3).mean().sel(year=2000)
    np.testing.assert_allclose(before.values, same.values, rtol=1e-10)


def test_bias_calibrators_skip_the_ratio_for_temperature():
    assert [c.method for c in bias_calibrators("t2m")] == ["mean", "scaling"]
    assert "ratio" in [c.method for c in bias_calibrators("precip")]


# ----------------------------------------------------------- quantile mapping
def test_eqm_matches_the_observed_distribution():
    ens, obs = _hindcast(bias=60.0, scale=0.6)
    out = QuantileMapping(False, "precip").fit_predict_loyo(ens, obs)
    for q in (0.2, 0.5, 0.8):
        cal = float(out.members.quantile(q))
        ref = float(obs.quantile(q))
        raw = float(ens.quantile(q))
        assert abs(cal - ref) < abs(raw - ref)        # closer to the observed quantile


def test_eqm_on_a_grid_gives_each_cell_its_own_climatology():
    """
    Regression test of a silent bug: on a grid the mapping used to write into a
    copy of an uninitialised array, and returned memory garbage — which looked
    plausible on a single point (contiguous) and was wrong everywhere else.
    """
    rng = np.random.default_rng(21)
    years = np.arange(1993, 2017)
    lat, lon = np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0])
    # each cell has its own climate: 100 mm in the north-west, 900 in the south-east
    level = np.array([[100.0, 300.0], [500.0, 700.0], [800.0, 900.0]])
    obs = xr.DataArray(level[None] * (0.6 + 0.8 * rng.random((24, 3, 2))),
                       dims=("year", "latitude", "longitude"),
                       coords={"year": years, "latitude": lat, "longitude": lon})
    ens = xr.DataArray((level[None, None] * 2.5) * (0.6 + 0.8 * rng.random((24, 6, 3, 2))),
                       dims=("year", "number", "latitude", "longitude"),
                       coords={"year": years, "number": np.arange(6),
                               "latitude": lat, "longitude": lon})
    out = QuantileMapping(False, "precip").fit_predict_loyo(ens, obs).members
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            cal = float(out.sel(latitude=la, longitude=lo).mean())
            ref = float(obs.sel(latitude=la, longitude=lo).mean())
            assert abs(cal - ref) < 0.2 * ref, (la, lo, cal, ref)


def test_qdm_keeps_an_extreme_forecast_extreme():
    """EQM caps a forecast at the observed sample; QDM carries its departure through."""
    ens, obs = _hindcast(bias=0.0, scale=1.0, seed=7)
    extreme = ens.copy()
    extreme.loc[dict(year=2000)] = ens.max() * 2.0
    eqm = QuantileMapping(False, "precip").fit_predict_loyo(extreme, obs)
    qdm = QuantileMapping(True, "precip").fit_predict_loyo(extreme, obs)
    assert float(qdm.mean().sel(year=2000).mean()) > float(eqm.mean().sel(year=2000).mean())


# ------------------------------------------------------- predictive interface
def test_tercile_probabilities_sum_to_one_and_follow_the_thresholds():
    ens, obs = _hindcast()
    out = BiasCorrection("mean", "precip").fit_predict_loyo(ens, obs)
    q33 = obs.quantile(1 / 3, dim="year").drop_vars("quantile")
    q67 = obs.quantile(2 / 3, dim="year").drop_vars("quantile")
    prob = out.tercile_probs(q33, q67)
    np.testing.assert_allclose(prob.sum("category").values, 1.0, atol=1e-9)
    assert set(prob["category"].values) == {"BN", "NN", "AN"}


def test_normal_distribution_probabilities_and_quantiles():
    mu = xr.DataArray([10.0, 20.0], dims="longitude", coords={"longitude": [0.0, 1.0]})
    sigma = xr.full_like(mu, 2.0)
    dist = NormalDistribution(mu, sigma)
    assert float(dist.prob_below(mu)[0]) == pytest.approx(0.5, abs=1e-6)
    assert float(dist.quantile(0.5)[0]) == pytest.approx(10.0, abs=1e-6)
    assert float(dist.prob_below(mu + 2 * sigma)[0]) == pytest.approx(0.9772, abs=1e-3)


# --------------------------------------------------------- logistic family
def test_logistic_fit_recovers_known_coefficients():
    """A synthetic logistic sample must give back the coefficients that made it."""
    from eccas_s2s.calibrate.logistic import _sigmoid, logistic_fit

    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    X = np.stack([np.ones(n), x], -1)
    p = _sigmoid(-0.4 + 1.3 * x)
    y = (rng.random(n) < p).astype(float)
    beta = logistic_fit(X, y)
    assert beta[0] == pytest.approx(-0.4, abs=0.12)
    assert beta[1] == pytest.approx(1.3, abs=0.15)


def test_logistic_shrinks_a_useless_model_towards_climatology():
    """
    With a pure-noise model the calibration pulls the probabilities back to 1/3.

    Not exactly to 1/3: on 23 training years a logistic fit still finds spurious
    slopes, and no method can know from such a sample that the signal is empty.
    What must hold is that the calibrated probabilities are **markedly less
    scattered** than the raw member counting — that is the overconfidence being
    removed.
    """
    from eccas_s2s.calibrate.logistic import TercileLogistic

    rng = np.random.default_rng(1)
    years = np.arange(1993, 2017)
    cells = np.arange(3) * 1.0
    noise = xr.DataArray(rng.normal(size=(24, 8, 3)), dims=("year", "number", "longitude"),
                         coords={"year": years, "number": np.arange(8), "longitude": cells})
    obs = xr.DataArray(rng.gamma(4.0, 50.0, size=(24, 3)), dims=("year", "longitude"),
                       coords={"year": years, "longitude": cells})
    from eccas_s2s.validate.pairs import tercile_probabilities

    dist = TercileLogistic("precip").fit_predict_loyo(noise, obs)
    q33 = obs.quantile(1 / 3, dim="year").drop_vars("quantile")
    q67 = obs.quantile(2 / 3, dim="year").drop_vars("quantile")
    prob = dist.tercile_probs(q33, q67)
    raw = tercile_probabilities(noise)
    assert float(abs(prob - 1 / 3).mean()) < 0.7 * float(abs(raw - 1 / 3).mean())
    assert float(prob.std()) < 0.8 * float(raw.std())


def test_logistic_follows_a_perfect_model():
    """With a model that is the truth, the probability of the observed side goes up."""
    from eccas_s2s.calibrate.logistic import TercileLogistic

    ens, obs = _hindcast(seed=5, bias=0.0, scale=1.0)
    perfect = obs.expand_dims(number=[0]).transpose("year", "number", "longitude")
    dist = TercileLogistic("precip").fit_predict_loyo(perfect, obs)
    q33 = obs.quantile(1 / 3, dim="year").drop_vars("quantile")
    prob_bn = dist.prob_below(q33)
    wet = obs > obs.quantile(2 / 3, dim="year").drop_vars("quantile")
    dry = obs < q33
    assert float(prob_bn.where(dry).mean()) > float(prob_bn.where(wet).mean()) + 0.3


def test_elr_probabilities_increase_with_the_threshold():
    """The extended form is coherent by construction: P(y<q) never decreases with q."""
    from eccas_s2s.calibrate.logistic import ExtendedLogistic

    ens, obs = _hindcast(seed=2)
    dist = ExtendedLogistic("precip").fit_predict_loyo(ens, obs)
    probs = [float(dist.prob_below(obs.quantile(q, dim="year").drop_vars("quantile")).mean())
             for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert all(b >= a - 1e-6 for a, b in zip(probs, probs[1:])), probs
    assert probs[0] < 0.35 and probs[-1] > 0.65


def test_elr_quantiles_are_ordered_and_positive_for_rainfall():
    from eccas_s2s.calibrate.logistic import ExtendedLogistic

    ens, obs = _hindcast(seed=4)
    dist = ExtendedLogistic("precip").fit_predict_loyo(ens, obs)
    q20, q50, q80 = (float(dist.quantile(p).mean()) for p in (0.2, 0.5, 0.8))
    assert 0.0 <= q20 < q50 < q80


# ------------------------------------------------------------------- NGR/EMOS
def test_crps_normal_matches_a_monte_carlo_estimate():
    """The closed form must equal the definition E|X-y| - ½E|X-X'| on a large sample."""
    from eccas_s2s.calibrate.ngr import crps_normal

    rng = np.random.default_rng(0)
    mu, sigma, y = 2.0, 1.5, 3.2
    draws = rng.normal(mu, sigma, size=400_000)
    mc = np.abs(draws - y).mean() - 0.5 * np.abs(draws[:200_000] - draws[200_000:]).mean()
    assert crps_normal(y, mu, sigma) == pytest.approx(mc, abs=0.01)


def test_emos_recovers_the_parameters_of_a_synthetic_case():
    """Fitting data generated by y = a + b·x̄ + N(0, c² + d²s²) must find a, b, c, d."""
    from eccas_s2s.calibrate.ngr import _fit_emos

    rng = np.random.default_rng(3)
    n = 3000
    x = rng.normal(10.0, 3.0, size=(1, n))
    s2 = rng.gamma(4.0, 0.5, size=(1, n))
    a0, b0, c0, d0 = 2.0, 0.8, 1.0, 1.2
    y = a0 + b0 * x + rng.normal(size=(1, n)) * np.sqrt(c0 ** 2 + d0 ** 2 * s2)
    a, b, c, d = _fit_emos(x, s2, y, n_iter=600, lr=0.05)
    assert float(a[0]) == pytest.approx(a0, abs=0.5)
    assert float(b[0]) == pytest.approx(b0, abs=0.06)
    assert float(c[0]) == pytest.approx(c0, abs=0.4)
    assert float(d[0]) == pytest.approx(d0, abs=0.4)


def test_ngr_improves_the_crps_of_a_biased_underdispersed_ensemble():
    """The point of EMOS: a better CRPS than the raw ensemble, on left-out years."""
    from eccas_s2s.calibrate.ngr import NGR, crps_normal

    ens, obs = _hindcast(seed=11, bias=50.0, scale=0.6)
    dist = NGR("precip", True, n_iter=400).fit_predict_loyo(ens, obs)
    # CRPS of the calibrated law (in the square-root space it was fitted in)
    t = dist.transform
    cal = crps_normal(t.forward(obs).values, dist.mu.values, dist.sigma.values).mean()
    # CRPS of the raw ensemble, same space, empirical form
    e = t.forward(ens).transpose("year", "longitude", "number").values
    y = t.forward(obs).values[..., None]
    raw = (np.abs(e - y).mean(-1)
           - 0.5 * np.abs(e[..., :, None] - e[..., None, :]).mean((-1, -2))).mean()
    assert cal < raw


def test_ngr_widens_an_underdispersed_ensemble():
    """A too-narrow ensemble must come out with a wider, honest predictive spread."""
    from eccas_s2s.calibrate.ngr import NGR

    ens, obs = _hindcast(seed=13, bias=0.0, scale=1.0)
    narrow = ens.mean("number") + (ens - ens.mean("number")) * 0.15
    dist = NGR("t2m", True, n_iter=300).fit_predict_loyo(narrow, obs)
    assert float(dist.sigma.mean()) > float(narrow.std("number", ddof=1).mean())


def test_ngr_without_members_uses_the_constant_spread_form():
    from eccas_s2s.calibrate.ngr import ngr_calibrators

    cals = ngr_calibrators("precip", has_members=False)
    assert len(cals) == 1 and cals[0].name == "ngr_const"


def test_ngr_tercile_probabilities_are_proper():
    from eccas_s2s.calibrate.ngr import NGR

    ens, obs = _hindcast(seed=17)
    dist = NGR("precip", True, n_iter=200).fit_predict_loyo(ens, obs)
    q33 = obs.quantile(1 / 3, dim="year").drop_vars("quantile")
    q67 = obs.quantile(2 / 3, dim="year").drop_vars("quantile")
    prob = dist.tercile_probs(q33, q67)
    np.testing.assert_allclose(prob.sum("category").values, 1.0, atol=1e-9)
    assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0


# ------------------------------------------------------- choosing the method
def _tables():
    """One period, one model: a raw score and three candidate methods."""

    keys = dict(system="c3s", model="ecmwf", variable="precip", scale="season",
                period="season_m0")
    raw = pd.DataFrame([{**keys, "pearson_median": 0.30, "rpss_median": -0.05,
                         "msess_median": -1.2, "groc_median": 0.62, "label_fr": "SON 2026"}])
    cal = pd.DataFrame([
        {**keys, "method": "bias_mean", "pearson_median": 0.30, "rpss_median": -0.02,
         "msess_median": -0.2},
        {**keys, "method": "elr", "pearson_median": 0.30, "rpss_median": 0.06,
         "msess_median": -0.3},
        {**keys, "method": "ngr", "pearson_median": 0.29, "rpss_median": 0.04,
         "msess_median": 0.05},
    ])
    return raw, cal


def test_raw_baseline_is_the_forecast_untouched():
    """The `raw` method returns the interpolated members as they are."""
    from eccas_s2s.calibrate.base import RawForecast

    ens, obs = _hindcast()
    out = RawForecast("precip").fit_predict_loyo(ens, obs)
    np.testing.assert_allclose(out.members.values, ens.values)


def test_compare_uses_the_raw_rows_of_the_same_table():
    """Without an explicit raw table, the `raw` method of the table is the reference."""
    from eccas_s2s.calibrate.selection import compare

    raw, cal = _tables()
    same_table = pd.concat([cal, raw.assign(method="raw")], ignore_index=True)
    comp = compare(same_table)
    assert "raw" not in set(comp["method"])                    # the reference is not a candidate
    assert float(comp.set_index("method").loc["elr", "gain_prob"]) == pytest.approx(0.11)


def test_selection_keeps_only_methods_that_beat_raw_and_climatology():
    from eccas_s2s.calibrate.selection import compare

    raw, cal = _tables()
    comp = compare(cal, raw)
    by_method = comp.set_index("method")
    # bias_mean improves on raw but is still worse than climatology (RPSS < 0)
    assert bool(by_method.loc["bias_mean", "beats_raw"])
    assert not bool(by_method.loc["bias_mean", "beats_climatology"])
    assert not bool(by_method.loc["bias_mean", "eligible_method"])
    assert bool(by_method.loc["elr", "eligible_method"])
    assert float(by_method.loc["elr", "gain_prob"]) == pytest.approx(0.11)


def test_selection_picks_the_best_eligible_method():
    from eccas_s2s.calibrate.selection import compare, recommend

    raw, cal = _tables()
    rec = recommend(compare(cal, raw))
    assert len(rec) == 1
    assert rec.iloc[0]["method"] == "elr"
    assert rec.iloc[0]["n_eligible"] == 2


def test_selection_falls_back_on_raw_and_says_why():
    from eccas_s2s.calibrate.selection import compare, recommend

    raw, cal = _tables()
    cal["rpss_median"] = [-0.04, -0.03, -0.02]          # none beats climatology
    rec = recommend(compare(cal, raw))
    assert rec.iloc[0]["method"] == "raw"
    assert "climatologie" in rec.iloc[0]["reason"]


def test_selection_uses_msess_for_the_value_product():
    """A value map is judged on the MSESS, not on the RPSS of the categories."""
    from eccas_s2s.calibrate.selection import compare, recommend

    raw, cal = _tables()
    comp = compare(cal, raw)
    rec = recommend(comp, product="deterministic")
    assert rec.iloc[0]["method"] == "ngr"          # the only one above climatology in MSESS
    assert rec.iloc[0]["product"] == "deterministic"


def test_correlation_is_reported_but_does_not_gate_the_choice():
    """
    A cross-validated regression inherits the negative bias of LOYO correlation;
    using it as a gate would reject a method for an artefact of the validation.
    """
    from eccas_s2s.calibrate.selection import compare, recommend

    raw, cal = _tables()
    cal.loc[cal.method == "elr", "pearson_median"] = 0.10      # correlation apparently lost
    comp = compare(cal, raw)
    assert float(comp.set_index("method").loc["elr", "gain_det"]) == pytest.approx(-0.20)
    assert bool(comp.set_index("method").loc["elr", "eligible_method"])
    assert recommend(comp).iloc[0]["method"] == "elr"


def test_nmme_is_capped_at_the_longest_c3s_horizon():
    """NMME publishes twice as far ahead; beyond C3S it could not be combined."""
    from pathlib import Path
    from eccas_s2s.core.periods import build_periods
    from eccas_s2s.operations.skill_raw import horizon_days
    from eccas_s2s.settings import load_cycle

    cfg = load_cycle(Path(__file__).resolve().parents[1] / "config" / "cycle_202609.yaml")
    c3s_max = max(int(m.max_lead_days) for m in cfg.c3s_models.values())
    assert horizon_days(cfg, "nmme", "CFSv2") == c3s_max
    assert horizon_days(cfg, "c3s", "dwd") == int(cfg.c3s_models["dwd"].max_lead_days)
    n_nmme = len(build_periods(cfg.init_date, horizon_days(cfg, "nmme", "CFSv2"),
                               scales=("month", "season")))
    n_c3s = len(build_periods(cfg.init_date, c3s_max, scales=("month", "season")))
    assert n_nmme == n_c3s == 12


def test_raw_baseline_counts_members_against_the_model_climatology():
    """
    The baseline must reproduce the probabilities of phase P2.

    Counting biased members against the *observed* thresholds measures the bias:
    a model 28 % too dry announces "below normal" every year and scores an RPSS
    of -0.85, which would make any calibration look miraculous.
    """
    from eccas_s2s.calibrate.base import RawForecast
    from eccas_s2s.validate.pairs import tercile_probabilities

    ens, obs = _hindcast(bias=200.0)                       # a very dry-biased model
    dist = RawForecast("precip").fit_predict_loyo(ens, obs)
    q33 = obs.quantile(1 / 3, dim="year").drop_vars("quantile")
    q67 = obs.quantile(2 / 3, dim="year").drop_vars("quantile")
    prob = dist.tercile_probs(q33, q67)
    np.testing.assert_allclose(prob.values, tercile_probabilities(ens).values)
    # against the observed thresholds the same ensemble would be degenerate
    naive = (ens < q33).mean("number")
    assert float(naive.mean()) < 0.05 and float(prob.sel(category="BN").mean()) > 0.2


# ------------------------------------------------------------- housekeeping
def test_housekeeping_removes_what_the_scores_no_longer_justify(tmp_path):
    """
    A summary row and a map survive their score: the netCDF tree is the authority.

    This is what left NMME with 22 periods in the table and 18 in the figures
    after its horizon was capped at the C3S one.
    """
    import pandas as pd
    import yaml
    from eccas_s2s.operations.housekeeping import prune, scored_periods
    from eccas_s2s.operations.skill_raw import skill_dir
    from eccas_s2s.settings import load_cycle
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((repo / "config" / "cycle_202609.yaml").read_text(encoding="utf-8"))
    for key in ("data_root", "output_root", "archive_root"):
        raw["paths"][key] = str(tmp_path / key)
    raw["includes"] = {k: str(repo / "config" / v) for k, v in raw["includes"].items()}
    cfg_path = tmp_path / "cycle.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    cfg = load_cycle(cfg_path)
    root = skill_dir(cfg, "raw")

    # scores for two periods only
    kept = ["month_m0", "month_m1"]
    folder = root / "netcdf" / "nmme_CFSv2" / "month" / "precip"
    folder.mkdir(parents=True)
    xr.Dataset({"pearson": ("period", [0.3, 0.2])}, coords={"period": kept}).to_netcdf(
        folder / "pearson.nc")
    assert scored_periods(root) == {("nmme_CFSv2/month/precip", p) for p in kept}

    # figures and a summary that also carry a third, obsolete period
    figs = root / "figures" / "nmme_CFSv2" / "month" / "precip" / "pearson"
    figs.mkdir(parents=True)
    for name in ("month_m0.png", "month_m1.png", "month_m7.png", "month_m7_AN.png"):
        (figs / name).write_bytes(b"")
    rows = [{"system": "nmme", "model": "CFSv2", "variable": "precip", "scale": "month",
             "period": p, "eligible": True} for p in kept + ["month_m7"]]
    pd.DataFrame(rows).to_csv(root / "skill_raw_summary.csv", index=False)

    prune(str(cfg_path), kinds=("raw",))
    assert sorted(p.name for p in figs.iterdir()) == ["month_m0.png", "month_m1.png"]
    table = pd.read_csv(root / "skill_raw_summary.csv")
    assert sorted(table["period"]) == kept
    assert (cfg.path_of("output_root") / "registry" / "models_eligibility.csv").exists()


def test_raw_baseline_of_a_member_less_system_has_no_probability():
    """
    NMME is an ensemble mean: its "raw probability" would be 0 or 1 (D22).

    The baseline is then absent rather than degenerate, and the register asks the
    calibrated probabilities to beat climatology alone.
    """
    from eccas_s2s.calibrate.base import RawForecast

    ens, obs = _hindcast()
    mean_only = ens.mean("number")
    dist = RawForecast("precip").fit_predict_loyo(mean_only, obs)
    q33 = obs.quantile(1 / 3, dim="year").drop_vars("quantile")
    q67 = obs.quantile(2 / 3, dim="year").drop_vars("quantile")
    assert np.isnan(dist.tercile_probs(q33, q67).values).all()
    assert np.isnan(dist.prob_below(q33).values).all()
    assert np.isfinite(dist.mean().values).all()          # the value forecast stays


def test_selection_lets_a_member_less_system_be_judged_on_climatology_alone():
    from eccas_s2s.calibrate.selection import compare, recommend

    raw, cal = _tables()
    raw["rpss_median"] = np.nan                            # no raw probability
    comp = compare(cal, raw)
    assert bool(comp.set_index("method").loc["elr", "beats_raw"])
    assert recommend(comp).iloc[0]["method"] == "elr"
