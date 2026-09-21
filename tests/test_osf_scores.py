"""Gridpoint scores: properties, and cross-check against the R `verification` package."""
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eccas_s2s.validate.cv import loyo_mean, loyo_quantile
from eccas_s2s.validate.scores import (brier_scores, deterministic_scores, roc_area,
                                       rps_scores, tercile_skill)

RSCRIPT = shutil.which("Rscript")


def _pair(n=24, seed=0, skill=0.8):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=n)
    fc = skill * obs + np.sqrt(1 - skill ** 2) * rng.normal(size=n)
    years = np.arange(1993, 1993 + n)
    mk = lambda a: xr.DataArray(a[:, None], dims=("year", "longitude"),
                                coords={"year": years, "longitude": [0.0]})
    return mk(obs), mk(fc)


def test_deterministic_scores_perfect_and_climatology():
    obs, fc = _pair()
    perfect = deterministic_scores(obs, obs).isel(longitude=0)
    assert float(perfect.bias) == pytest.approx(0) and float(perfect.rmse) == pytest.approx(0)
    assert float(perfect.pearson) == pytest.approx(1) and float(perfect.msess) == pytest.approx(1)
    clim = deterministic_scores(xr.zeros_like(obs) + float(obs.mean()), obs).isel(longitude=0)
    assert float(clim.msess) == pytest.approx(0, abs=1e-9)
    s = deterministic_scores(fc, obs).isel(longitude=0)
    assert 0.6 < float(s.pearson) < 0.95 and float(s.n_years) == 24


def test_spearman_is_rank_based():
    obs, _ = _pair()
    fc = np.exp(obs)                # same ranks, different values
    s = deterministic_scores(fc, obs).isel(longitude=0)
    assert float(s.spearman) == pytest.approx(1.0)
    assert float(s.pearson) < 1.0


def _tercile_case(n=24, seed=1):
    rng = np.random.default_rng(seed)
    obs_cat = rng.integers(0, 3, size=n)
    prob = np.full((n, 3), 0.2)
    prob[np.arange(n), obs_cat] = 0.6          # forecast leans to the right category
    years = np.arange(1993, 1993 + n)
    p = xr.DataArray(prob[:, :, None], dims=("year", "category", "longitude"),
                     coords={"year": years, "category": ["BN", "NN", "AN"], "longitude": [0.0]})
    o = xr.DataArray(obs_cat[:, None], dims=("year", "longitude"),
                     coords={"year": years, "longitude": [0.0]})
    return p, o


def test_rpss_between_perfect_and_climatology():
    p, o = _tercile_case()
    good = rps_scores(p, o).isel(longitude=0)
    assert 0 < float(good.rpss) < 1
    clim = xr.full_like(p, 1 / 3)
    assert float(rps_scores(clim, o).isel(longitude=0).rpss) == pytest.approx(0, abs=1e-12)
    perfect = xr.zeros_like(p)
    for k in range(3):
        perfect[:, k, :] = (o == k).astype(float)
    assert float(rps_scores(perfect, o).isel(longitude=0).rpss) == pytest.approx(1)


def test_roc_area_extremes():
    years = np.arange(10)
    ev = xr.DataArray(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float), dims="year",
                      coords={"year": years})
    perfect = xr.DataArray(np.linspace(1, 0.55, 10), dims="year", coords={"year": years})
    assert float(roc_area(perfect, ev)) == pytest.approx(1.0)
    assert float(roc_area(1 - perfect, ev)) == pytest.approx(0.0)
    flat = xr.full_like(perfect, 0.5)
    assert float(roc_area(flat, ev)) == pytest.approx(0.5)


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript absent")
def test_scores_match_r_verification_package(tmp_path):
    """RPS/RPSS, Brier/BSS and ROC area must match the R `verification` package."""
    p, o = _tercile_case(seed=3)
    df = pd.DataFrame({"pBN": p.isel(category=0, longitude=0).values,
                       "pNN": p.isel(category=1, longitude=0).values,
                       "pAN": p.isel(category=2, longitude=0).values,
                       "obs_cat": o.isel(longitude=0).values})
    csv = tmp_path / "pairs.csv"
    df.to_csv(csv, index=False)
    rcode = f'''
    suppressPackageStartupMessages(library(verification))
    d <- read.csv("{csv}")
    P <- as.matrix(d[, c("pBN","pNN","pAN")])
    r <- rps(obs = d$obs_cat + 1L, pred = P, baseline = rep(1/3, 3))
    # fine thresholds: verification::brier bins the probabilities by default
    b <- brier(obs = as.integer(d$obs_cat == 0), pred = d$pBN, baseline = rep(1/3, nrow(d)),
               thresholds = seq(0, 1, 1e-4))
    a <- roc.area(obs = as.integer(d$obs_cat == 2), pred = d$pAN)
    cat(sprintf("%.10f %.10f %.10f %.10f %.10f", r$rps, r$rpss, b$bs, b$ss, a$A))
    '''
    out = subprocess.run([RSCRIPT, "-e", rcode], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    r_rps, r_rpss, r_bs, r_bss, r_roc = (float(x) for x in out.stdout.split()[-5:])

    py = tercile_skill(p, o).isel(longitude=0)
    assert float(py.rps) == pytest.approx(r_rps, abs=1e-8)
    assert float(py.rpss) == pytest.approx(r_rpss, abs=1e-8)
    bn = brier_scores(p.isel(category=0), (o == 0).astype(float), 1 / 3).isel(longitude=0)
    # 1e-4 bins in R leave a residual of the same size; our Brier is the exact one
    assert float(bn.bs) == pytest.approx(r_bs, abs=1e-4)
    assert float(bn.bss) == pytest.approx(r_bss, abs=1e-3)
    assert float(py.roc_area.sel(category="AN")) == pytest.approx(r_roc, abs=1e-8)


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript absent")
def test_r_brier_bins_probabilities_by_default(tmp_path):
    """Documents a difference with the reference chain: R's brier() bins by default."""
    rcode = ('suppressPackageStartupMessages(library(verification));'
             'set.seed(7); n <- 24; p <- round(runif(n), 3); o <- rbinom(n, 1, 0.4);'
             'b1 <- brier(o, p, baseline = rep(1/3, n));'
             'b2 <- brier(o, p, baseline = rep(1/3, n), thresholds = seq(0, 1, 1e-4));'
             'cat(sprintf("%.8f %.8f %.8f", mean((p - o)^2), b1$bs, b2$bs))')
    out = subprocess.run([RSCRIPT, "-e", rcode], capture_output=True, text=True, timeout=300)
    exact, binned, fine = (float(x) for x in out.stdout.split()[-3:])
    assert binned != pytest.approx(exact, abs=1e-6)      # default bins shift the score
    assert fine == pytest.approx(exact, abs=1e-6)        # fine thresholds recover it


def test_loyo_helpers():
    years = np.arange(1993, 1997)
    da = xr.DataArray(np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0], [10.0, 12.0]]),
                      dims=("year", "number"), coords={"year": years})
    m = loyo_mean(da)
    assert float(m.sel(year=1993)) == pytest.approx((3 + 4 + 11) / 3)   # members averaged first
    q = loyo_quantile(da, 0.5, dims=["number"])
    assert float(q.sel(year=1996)) == pytest.approx(3.0)                # median of 1..5 without 1996


# ----------------------------------------------------------------- pairs
def _ensemble(n_years=24, n_members=10, seed=5):
    rng = np.random.default_rng(seed)
    years = np.arange(1993, 1993 + n_years)
    truth = rng.normal(size=n_years)
    members = truth[:, None] + rng.normal(scale=0.8, size=(n_years, n_members)) + 5.0   # biased
    obs = truth + rng.normal(scale=0.3, size=n_years)
    fc = xr.DataArray(members[:, :, None], dims=("year", "number", "longitude"),
                      coords={"year": years, "number": np.arange(n_members), "longitude": [0.0]})
    ob = xr.DataArray(obs[:, None], dims=("year", "longitude"),
                      coords={"year": years, "longitude": [0.0]})
    return fc, ob


def test_pairs_probabilities_and_categories():
    from eccas_s2s.validate.pairs import build_pairs
    fc, ob = _ensemble()
    ds = build_pairs(fc, ob)
    assert ds.attrs["n_years"] == 24 and ds.attrs["n_members"] == 10
    # probabilities sum to 1 and each category is observed about a third of the time
    np.testing.assert_allclose(ds.prob.sum("category").values, 1.0, atol=1e-12)
    counts = [int((ds.obs_cat == k).sum()) for k in range(3)]
    assert all(5 <= c <= 11 for c in counts), counts
    # a 5 degC bias does not stop the model probabilities from being informative
    from eccas_s2s.validate.scores import tercile_skill
    sk = tercile_skill(ds.prob, ds.obs_cat).isel(longitude=0)
    assert float(sk.rpss) > 0.1


def test_pairs_thresholds_exclude_the_verified_year():
    from eccas_s2s.validate.pairs import observed_categories
    years = np.arange(2000, 2010)
    values = np.arange(10.0)                      # strictly increasing
    ob = xr.DataArray(values, dims="year", coords={"year": years})
    cat, q = observed_categories(ob)
    # the last year is the largest value; its thresholds come from the 9 others
    assert float(q.sel(year=2009, quantile=2 / 3)) == pytest.approx(np.quantile(values[:-1], 2 / 3,
                                                                                method="weibull"))
    assert int(cat.sel(year=2009)) == 2


def test_pairs_without_members_flagged():
    from eccas_s2s.validate.pairs import build_pairs
    fc, ob = _ensemble()
    ds = build_pairs(fc.mean("number"), ob)
    assert ds.attrs["has_members"] == 0 and ds.attrs["n_members"] == 0
    # ensemble mean only (NMME): no probability at all, a deterministic category
    assert "prob" not in ds
    assert set(np.unique(ds.fcst_cat.values)) <= {0.0, 1.0, 2.0}


# ------------------------------------------------------------- R bridge
@pytest.mark.skipif(RSCRIPT is None, reason="Rscript absent")
def test_r_bridge_end_to_end(tmp_path):
    """The R zone scores run and agree with the Python ones on the same pairs."""
    from eccas_s2s.validate.pairs import build_pairs
    from eccas_s2s.validate.r_bridge import check_packages, pairs_to_frame, run_zone_scores
    from eccas_s2s.validate.scores import deterministic_scores, tercile_skill

    info = check_packages()
    assert info["verification"] is True

    fc, ob = _ensemble(seed=11)
    ds = build_pairs(fc, ob).isel(longitude=0)
    ds = ds.expand_dims(period=["season_m0"]) if "period" not in ds.dims else ds
    frame = pairs_to_frame(ds)
    assert len(frame) == 24 and set(["pBN", "pNN", "pAN", "obs_cat"]).issubset(frame.columns)

    tables = run_zone_scores(frame, tmp_path, "test_zone")
    assert set(tables) >= {"deterministic_scores", "tercile_scores", "category_scores",
                           "reliability_bins"}
    det_r = tables["deterministic_scores"].iloc[0]
    ter_r = tables["tercile_scores"].iloc[0]

    det_py = deterministic_scores(ds.ensmean, ds.obs).isel(period=0)
    sk_py = tercile_skill(ds.prob, ds.obs_cat).isel(period=0)
    assert det_r["bias"] == pytest.approx(float(det_py.bias), abs=1e-5)
    assert det_r["rmse"] == pytest.approx(float(det_py.rmse), abs=1e-5)
    assert det_r["acc"] == pytest.approx(float(det_py.acc), abs=1e-5)
    assert det_r["spearman"] == pytest.approx(float(det_py.spearman), abs=1e-5)
    assert ter_r["rpss"] == pytest.approx(float(sk_py.rpss), abs=1e-5)
    roc_r = tables["category_scores"].set_index("category").loc["AN", "roc_area"]
    assert roc_r == pytest.approx(float(sk_py.roc_area.sel(category="AN")), abs=1e-5)
    # the reliability bins cover the ten classes for each category
    assert len(tables["reliability_bins"]) == 30


# ------------------------------------------------------------------ zones
def _grid(nlat=6, nlon=4):
    lat = np.arange(-2.5, -2.5 + nlat, 1.0)
    lon = np.arange(10.5, 10.5 + nlon, 1.0)
    return xr.DataArray(np.zeros((nlat, nlon)), dims=("latitude", "longitude"),
                        coords={"latitude": lat, "longitude": lon})


def test_zone_masks_and_fraction_positive():
    from eccas_s2s.validate.zones import DOMAIN, fraction_positive, zone_masks
    domains = {"domain": {"extent": [10.0, 15.0, -3.0, 3.0]},   # wide enough for every cell
               "zones": {"north": {"lat": [0.0, 3.0]}, "south": {"lat": [-3.0, 0.0]}}}
    ref = _grid()
    valid = ref.notnull()
    valid[0, 0] = False
    masks = zone_masks(ref, domains, valid=valid)
    assert set(masks) == {DOMAIN, "north", "south"}
    assert int(masks[DOMAIN].sum()) == ref.size - 1          # the invalid cell is excluded
    assert int(masks["north"].sum()) + int(masks["south"].sum()) == int(masks[DOMAIN].sum())

    score = xr.full_like(ref, 0.5)
    score[:3, :] = -0.5
    frac = fraction_positive(score, masks[DOMAIN])
    assert 0.45 < frac < 0.55                                 # about half the (weighted) area


def test_fraction_positive_ignores_nan():
    from eccas_s2s.validate.zones import fraction_positive
    ref = _grid()
    mask = ref.notnull()
    score = xr.full_like(ref, np.nan)
    score[0, :] = 1.0
    assert fraction_positive(score, mask) == pytest.approx(1.0)


# ------------------------------------------------- pooled pairs and diagrams
def _grid_ensemble(n_years=24, n_members=8, ny=6, nx=7, seed=7):
    """A small hindcast on a grid, with a shared signal so the skill is not zero."""
    rng = np.random.default_rng(seed)
    years = np.arange(1993, 1993 + n_years)
    lat = np.arange(-2.5, -2.5 + ny, 1.0)
    lon = np.arange(10.5, 10.5 + nx, 1.0)
    signal = rng.normal(size=(n_years, 1, ny, nx))
    members = signal + rng.normal(scale=0.9, size=(n_years, n_members, ny, nx))
    obs = signal[:, 0] + rng.normal(scale=0.9, size=(n_years, ny, nx))
    fc = xr.DataArray(members, dims=("year", "number", "latitude", "longitude"),
                      coords={"year": years, "number": np.arange(n_members),
                              "latitude": lat, "longitude": lon})
    ob = xr.DataArray(obs, dims=("year", "latitude", "longitude"),
                      coords={"year": years, "latitude": lat, "longitude": lon})
    return (fc.expand_dims(period=["SON"]).transpose("year", "number", "period",
                                                     "latitude", "longitude"),
            ob.expand_dims(period=["SON"]).transpose("year", "period", "latitude", "longitude"))


def _mask_of(ob):
    return xr.DataArray(np.ones((ob.sizes["latitude"], ob.sizes["longitude"]), bool),
                        dims=("latitude", "longitude"),
                        coords={"latitude": ob["latitude"], "longitude": ob["longitude"]})


def test_pooled_frame_pools_years_and_gridpoints():
    from eccas_s2s.validate.pairs import build_pairs
    from eccas_s2s.validate.pooled import pooled_frame
    fc, ob = _grid_ensemble()
    ds = build_pairs(fc, ob)
    frame = pooled_frame(ds, _mask_of(ob))
    assert len(frame) == 24 * 6 * 7                      # every year x every cell
    assert {"pBN", "pNN", "pAN", "obs_cat", "year"} <= set(frame.columns)
    assert frame[["pBN", "pNN", "pAN"]].sum(axis=1).round(6).eq(1.0).all()
    # the pooled sample is what makes ten probability bins usable
    assert frame["pAN"].round(3).nunique() > 5


def test_pooled_frame_thins_large_zones():
    from eccas_s2s.validate.pairs import build_pairs
    from eccas_s2s.validate.pooled import pooled_frame
    fc, ob = _grid_ensemble()
    ds = build_pairs(fc, ob)
    frame = pooled_frame(ds, _mask_of(ob), max_pixels=10)
    assert frame["cell"].nunique() <= 10 and len(frame) == 24 * frame["cell"].nunique()


def test_pooled_frame_without_members_has_no_probability():
    from eccas_s2s.validate.pairs import build_pairs
    from eccas_s2s.validate.pooled import pooled_frame
    fc, ob = _grid_ensemble()
    ds = build_pairs(fc.mean("number"), ob)
    frame = pooled_frame(ds, _mask_of(ob))
    assert "pBN" not in frame.columns and "fcst_cat" in frame.columns


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript absent")
def test_zone_diagrams_draw_two_figures(tmp_path):
    """One reliability figure and one ROC figure per period, three categories each."""
    from eccas_s2s.validate.pairs import build_pairs
    from eccas_s2s.validate.pooled import pooled_frame
    from eccas_s2s.validate.r_bridge import run_zone_diagrams
    from eccas_s2s.validate.scores import roc_area

    fc, ob = _grid_ensemble()
    ds = build_pairs(fc, ob)
    frame = pooled_frame(ds, _mask_of(ob))
    figures = run_zone_diagrams(frame, tmp_path, "test|precip|domain", n_boot=30)
    assert [f.name for f in figures] == ["reliability_SON.png", "roc_SON.png"]
    scores = pd.read_csv(tmp_path / "diagram_scores.csv")
    assert list(scores["category"]) == ["BN", "NN", "AN"]
    # the R area of the pooled sample is the pooled version of the Python map
    py = float(roc_area(ds["prob"].sel(category="AN"),
                        (ds["obs_cat"] == 2).astype(float)).mean())
    assert abs(float(scores.set_index("category").loc["AN", "roc_area"]) - py) < 0.1


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript absent")
def test_zone_diagrams_skip_systems_without_members(tmp_path):
    from eccas_s2s.validate.pairs import build_pairs
    from eccas_s2s.validate.pooled import pooled_frame
    from eccas_s2s.validate.r_bridge import run_zone_diagrams
    fc, ob = _grid_ensemble()
    frame = pooled_frame(build_pairs(fc.mean("number"), ob), _mask_of(ob))
    assert run_zone_diagrams(frame, tmp_path, "nmme|precip|domain", n_boot=30) == []


def test_summary_rows_without_probabilities():
    """A member-less system is judged on the deterministic criterion alone."""
    from eccas_s2s.operations.skill_raw import summary_rows
    from eccas_s2s.validate.pairs import build_pairs
    from eccas_s2s.validate.scores import deterministic_scores

    fc, ob = _grid_ensemble()
    pairs = build_pairs(fc.mean("number"), ob)
    maps = deterministic_scores(pairs["ensmean"], pairs["obs"]).assign_coords(
        label=("period", ["SON 2026"]), scale=("period", ["season"]))
    maps.attrs.update(n_years=24, n_members=0)
    rows = summary_rows(maps, {"domain": _mask_of(ob)}, "nmme", "CFSv2", "precip",
                        {"SON": 0})
    assert len(rows) == 1 and rows[0]["probabilistic"] is False
    assert "rpss_domain_median" not in rows[0] and "frac_rpss_positive" not in rows[0]
    assert rows[0]["eligible"] == (rows[0]["frac_pearson_positive"] >= 0.05)
