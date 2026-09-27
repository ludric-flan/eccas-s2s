"""
Each product is judged by the metric that fits it (register of recommended methods).

The catalogue follows the verification column of the CAPC-AC method note: a value
on its error, a tercile map on the RPSS, an event on the Brier skill score and its
discrimination.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eccas_s2s.calibrate.base import EnsembleDistribution
from eccas_s2s.products.catalogue import catalogue, deciding_metric
from eccas_s2s.validate.product_scores import score_products

THRESHOLDS = {"exceedance_mm": {"season": [50, 200], "month": [10, 50]},
              "anomaly_mm": {"season": 60, "month": 30}}


def _case(n_years=24, n_members=12, n_cells=6, skill=0.8, seed=0):
    rng = np.random.default_rng(seed)
    years = np.arange(1993, 1993 + n_years)
    cells = np.arange(n_cells) * 1.0
    truth = rng.gamma(4.0, 60.0, size=(n_years, n_cells))
    signal = skill * (truth - truth.mean()) + truth.mean()
    members = signal[:, None, :] + rng.normal(scale=40.0, size=(n_years, n_members, n_cells))
    obs = xr.DataArray(truth, dims=("year", "longitude"),
                       coords={"year": years, "longitude": cells})
    ens = xr.DataArray(np.clip(members, 0, None), dims=("year", "number", "longitude"),
                       coords={"year": years, "number": np.arange(n_members), "longitude": cells})
    return EnsembleDistribution(ens), obs


def test_catalogue_covers_the_products_of_the_reference_chain():
    names = {p.name for p in catalogue("precip", "season", THRESHOLDS)}
    assert {"cumul", "anomalie", "terciles", "categorie_BN", "categorie_AN", "quintile_bas",
            "quintile_haut", "depassement_median", "spi_classes", "spi"} <= names
    assert "depassement_200mm" in names and "depassement_50mm" in names
    # the millimetre thresholds follow the scale: a 50 mm dekad is not a 50 mm season
    monthly = {p.name for p in catalogue("precip", "month", THRESHOLDS)}
    assert "depassement_10mm" in monthly and "depassement_200mm" not in monthly


def test_temperature_has_no_rainfall_products():
    names = {p.name for p in catalogue("t2m", "season", THRESHOLDS)}
    assert "spi" not in names and not any(n.startswith("depassement_") and n.endswith("mm")
                                          for n in names)
    assert {"cumul", "anomalie", "terciles"} <= names


def test_each_product_is_decided_by_its_own_metric():
    assert deciding_metric("cumul") == "msess"
    assert deciding_metric("anomalie") == "acc"
    assert deciding_metric("terciles") == "rpss"
    assert deciding_metric("categorie_AN") == "bss"
    assert deciding_metric("depassement_200mm", thresholds=THRESHOLDS) == "bss"
    assert deciding_metric("spi_classes", thresholds=THRESHOLDS) == "rpss"


def test_scores_are_produced_for_every_product_with_one_deciding_metric():
    dist, obs = _case()
    table = score_products(dist, obs, "precip", "season", THRESHOLDS)
    products = {p.name for p in catalogue("precip", "season", THRESHOLDS)}
    assert set(table["product"]) == products
    for name, group in table.groupby("product"):
        assert group["deciding"].sum() == 1, name
    assert table["median"].notna().all()


def test_a_skilful_forecast_scores_positively_on_every_family():
    dist, obs = _case(skill=0.95, seed=3)
    t = score_products(dist, obs, "precip", "season", THRESHOLDS)
    dec = t[t["deciding"]].set_index("product")["median"]
    assert dec["terciles"] > 0            # RPSS
    assert dec["categorie_AN"] > 0        # BSS
    assert dec["cumul"] > 0               # MSESS
    assert dec["anomalie"] > 0.5          # ACC


def test_a_useless_forecast_scores_around_zero_or_below():
    rng = np.random.default_rng(9)
    _, obs = _case(seed=4)
    noise = xr.DataArray(rng.gamma(4.0, 60.0, size=(24, 12, 6)),
                         dims=("year", "number", "longitude"),
                         coords={"year": obs["year"], "number": np.arange(12),
                                 "longitude": obs["longitude"]})
    t = score_products(EnsembleDistribution(noise), obs, "precip", "season", THRESHOLDS)
    dec = t[t["deciding"]].set_index("product")["median"]
    assert dec["terciles"] < 0.1 and dec["anomalie"] < 0.3


def test_roc_is_reported_relative_to_the_no_skill_value():
    """
    Discrimination is stored as `area - 0.5`: positive means better than chance,
    like every other metric of the register, and 0.5 is a perfect separation.
    """
    dist, obs = _case(skill=0.95, seed=5)
    good = score_products(dist, obs, "precip", "season", THRESHOLDS)
    roc = good[(good["product"] == "categorie_AN")
               & (good["metric"] == "roc_skill")]["median"].iloc[0]
    assert 0.0 < roc <= 0.5

    rng = np.random.default_rng(31)
    noise = xr.DataArray(rng.gamma(4.0, 60.0, size=(24, 12, 6)),
                         dims=("year", "number", "longitude"),
                         coords={"year": obs["year"], "number": np.arange(12),
                                 "longitude": obs["longitude"]})
    bad = score_products(EnsembleDistribution(noise), obs, "precip", "season", THRESHOLDS)
    roc_bad = bad[(bad["product"] == "categorie_AN")
                  & (bad["metric"] == "roc_skill")]["median"].iloc[0]
    assert abs(roc_bad) < 0.25


def test_fair_scores_correct_the_ensemble_size_penalty():
    """
    The same system scored with 20 and with 51 members must not be ranked by size.

    A probability counted on m members carries a sampling error that inflates the
    RPS and the Brier score; our hindcasts hold 20 to 31 members, so the plain
    RPSS systematically favours the bigger ensembles. The debiased version
    removes that penalty — the two ensembles then agree far more closely.
    """
    import numpy as np
    import xarray as xr

    from eccas_s2s.validate.scores import brier_scores, rps_scores

    rng = np.random.default_rng(3)
    years = 40
    truth = rng.integers(0, 3, years)
    obs = xr.DataArray(truth, dims="year", coords={"year": range(years)})

    def sampled(m):
        p = np.full((years, 3), 0.2)
        p[np.arange(years), truth] = 0.6          # same underlying skill
        counts = np.array([rng.multinomial(m, row) / m for row in p])
        return xr.DataArray(counts.T, dims=("category", "year"),
                            coords={"category": ["BN", "NN", "AN"], "year": range(years)})

    small, big = sampled(20), sampled(51)
    plain = [float(rps_scores(p, obs, n_members=m)["rpss"]) for p, m in ((small, 20), (big, 51))]
    fair = [float(rps_scores(p, obs, n_members=m)["rpss_fair"]) for p, m in ((small, 20), (big, 51))]
    assert fair[0] > plain[0] and fair[1] > plain[1]        # the correction lifts both
    assert (fair[0] - fair[1]) > (plain[0] - plain[1])      # and lifts the small one more

    event = (obs == 2).astype(float)
    bs = brier_scores(small.sel(category="AN"), event, float(event.mean()), n_members=20)
    assert float(bs["bs_fair"]) < float(bs["bs"]) and float(bs["bss_fair"]) > float(bs["bss"])


def test_quantile_products_use_the_model_climatology_but_millimetres_do_not():
    """
    A biased model keeps its tercile skill and loses its millimetre skill.

    That contrast is the point of phase P2: a product defined by a position in
    the distribution stays informative under a bias, a product defined in
    millimetres does not — and only the second one needs the calibration to be
    usable at all.
    """
    import numpy as np
    import xarray as xr

    from eccas_s2s.calibrate.base import RawEnsemble
    from eccas_s2s.validate.product_scores import ObservationContext, score_products

    rng = np.random.default_rng(11)
    years, members, ny, nx = 24, 15, 4, 4
    coords = {"year": list(range(1993, 1993 + years)),
              "latitude": np.linspace(-2, 2, ny), "longitude": np.linspace(10, 13, nx)}
    signal = rng.normal(0, 1, (years, ny, nx))
    obs = xr.DataArray(np.clip(200 + 60 * signal + rng.normal(0, 20, (years, ny, nx)), 1, None),
                       dims=("year", "latitude", "longitude"), coords=coords)
    # the same signal, shifted 40 % dry: the information is intact, the values are not
    members_da = xr.DataArray(
        np.clip(0.6 * (200 + 60 * signal)[:, None] + rng.normal(0, 30, (years, members, ny, nx)),
                0.1, None),
        dims=("year", "number", "latitude", "longitude"),
        coords={**coords, "number": range(members)})

    ctx = ObservationContext(obs, "precip", "season", {"exceedance_mm": {"season": [300]}})
    table = score_products(RawEnsemble(members_da), ctx, mode="deciding", n_members=members)
    deciding = table[table["deciding"]].set_index("product")["median"]

    assert deciding["terciles"] > 0            # quantile product: the bias is divided out
    assert deciding["cumul"] < 0               # value product: the bias is measured
    assert deciding["depassement_300mm"] < 0   # absolute threshold: so is this one


def test_admission_rests_on_discrimination_not_on_the_deciding_metric():
    """
    The register answers "can this model be used at all?", not "is it good raw".

    A model that discriminates but is biased stays in the workflow — that is what
    the calibration repairs; a model that discriminates nowhere is dropped,
    because no calibration creates information that is not there.
    """
    import pandas as pd

    from eccas_s2s.operations.skill_raw import eligibility_table, models_eligibility

    base = {"system": "c3s", "variable": "precip", "stream": "daily", "scale": "season",
            "product": "cumul", "product_label": "cumul (mm)", "period": "season_m0"}
    rows = [
        # biaisé mais informatif : la métrique décisive échoue, la discrimination passe
        {**base, "model": "biaise", "metric": "msess", "deciding": True,
         "discrimination": False, "fraction_positive": 0.01},
        {**base, "model": "biaise", "metric": "acc", "deciding": False,
         "discrimination": True, "fraction_positive": 0.80},
        # aucun signal : ni l'un ni l'autre
        {**base, "model": "muet", "metric": "msess", "deciding": True,
         "discrimination": False, "fraction_positive": 0.00},
        {**base, "model": "muet", "metric": "acc", "deciding": False,
         "discrimination": True, "fraction_positive": 0.01},
        # bon des deux côtés
        {**base, "model": "bon", "metric": "msess", "deciding": True,
         "discrimination": False, "fraction_positive": 0.60},
        {**base, "model": "bon", "metric": "acc", "deciding": False,
         "discrimination": True, "fraction_positive": 0.85},
    ]
    reg = eligibility_table(pd.DataFrame(rows), min_fraction=0.05).set_index("model")
    assert reg.loc["biaise", "status"] == "exploitable — à calibrer"
    assert reg.loc["muet", "status"] == "non exploitable"
    assert reg.loc["bon", "status"] == "exploitable — utilisable brut"
    assert bool(reg.loc["biaise", "eligible"]) and not bool(reg.loc["muet", "eligible"])

    models = models_eligibility(reg.reset_index()).set_index("model")
    assert bool(models.loc["biaise", "eligible"]) and not bool(models.loc["muet", "eligible"])


def test_a_roc_area_has_no_skill_at_one_half():
    """Counting the cells where a GROC is "above zero" would count the whole map."""
    import numpy as np
    import xarray as xr

    from eccas_s2s.core.geo import fraction_above
    from eccas_s2s.validate.product_scores import NO_SKILL_VALUE

    assert NO_SKILL_VALUE["groc"] == 0.5 and NO_SKILL_VALUE["roc_area"] == 0.5
    groc = xr.DataArray(np.array([[0.45, 0.48], [0.55, 0.60]]),
                        dims=("latitude", "longitude"),
                        coords={"latitude": [0.0, 1.0], "longitude": [10.0, 11.0]})
    mask = xr.full_like(groc, True, dtype=bool)
    assert fraction_above(groc, mask, 0.0) == 1.0                    # sans intérêt
    assert abs(fraction_above(groc, mask, NO_SKILL_VALUE["groc"]) - 0.5) < 0.01


def test_methods_are_attached_to_the_products_they_can_answer():
    """A tercile logistic cannot return a rainfall total, and is not tried on one."""
    from eccas_s2s.calibrate.products import method_product_table, methods_for, why_not
    from eccas_s2s.operations.calibrate_hindcast import calibrators_for
    from eccas_s2s.products.catalogue import catalogue

    available = calibrators_for("precip", has_members=True)
    products = {p.name: p for p in catalogue("precip", "season",
                                             {"exceedance_mm": {"season": [300]}})}

    def names(product):
        return [getattr(c, "name", str(c)) for c in methods_for(products[product], available)]

    # la logistique par catégorie ne sert que la famille des terciles
    assert "logistic" in names("terciles") and "logistic" in names("categorie_BN")
    assert "logistic" not in names("depassement_300mm")
    assert "logistic" not in names("cumul") and "logistic" not in names("spi_classes")
    # l'ELR décrit la distribution par ses seuils : pas pour une valeur
    assert "elr" in names("depassement_300mm") and "elr" not in names("cumul")
    # le brut est toujours présent : c'est la référence du gain
    assert all("raw" in names(p) for p in products)
    # et les corrections de distribution servent partout
    for method in ("bias_mean", "eqm", "qdm", "ngr"):
        assert method in names("cumul") and method in names("terciles")

    table = method_product_table(list(products.values()), available)
    assert {r["method"] for r in table} == {getattr(c, "name") for c in available}
    assert why_not(products["cumul"], "logistic")
    assert not why_not(products["terciles"], "logistic")


def test_gain_is_measured_at_every_grid_point():
    """
    A calibration that helps one half and hurts the other is not "neutral".

    The median gain of such a method is near zero; what separates it from a
    method that changes nothing is the repaired and broken areas, which is why
    the comparison keeps all four numbers.
    """
    import numpy as np
    import xarray as xr

    from eccas_s2s.validate.comparison import compare_maps, gain_map, recommend

    coords = {"latitude": np.linspace(-4, 4, 4), "longitude": np.linspace(10, 16, 4)}
    dims = ("latitude", "longitude")
    raw = xr.DataArray(np.full((4, 4), -0.10), dims=dims, coords=coords)
    mask = xr.full_like(raw, True, dtype=bool)

    # moitié réparée (+0,20), moitié cassée (−0,20) : médiane nulle
    mixed = raw.copy()
    mixed[:2, :] = 0.10
    mixed[2:, :] = -0.30
    row = compare_maps(mixed, raw, "rpss", mask, method="mixte", product="terciles")
    assert abs(row["median_gain"]) < 1e-9
    assert abs(row["fraction_improved"] - 0.5) < 0.05
    assert row["fraction_repaired"] > 0.4 and row["fraction_broken"] == 0.0

    # une méthode franchement meilleure partout
    good = raw + 0.15
    row_good = compare_maps(good, raw, "rpss", mask, method="bonne", product="terciles")
    assert row_good["fraction_improved"] == 1.0 and row_good["fraction_repaired"] > 0.9

    # une erreur s'améliore quand elle diminue
    assert float(gain_map(raw + 1.0, raw, "rmse").mean()) == -1.0
    assert float(gain_map(raw - 1.0, raw, "rmse").mean()) == 1.0

    table = pd.DataFrame([row, row_good])
    reco = recommend(table, min_improved=0.5).iloc[0]
    assert reco["method"] == "bonne"
    # sans candidat, la recommandation est le brut
    only_bad = recommend(pd.DataFrame([{**row, "median_gain": -0.2,
                                        "fraction_improved": 0.1}])).iloc[0]
    assert only_bad["method"] == "raw" and only_bad["reason"]


def test_a_family_diagram_uses_one_method_for_all_its_classes():
    """
    Three curves fitted by three methods could not share a set of axes.

    The family therefore takes the method recommended for the majority of its
    products, ties going to the reference product of the family.
    """
    import pandas as pd

    from eccas_s2s.operations.skill_diagrams import family_method
    from eccas_s2s.products.catalogue import families

    products = families("precip", "season", {"exceedance_mm": {"season": [300, 500]}})
    base = {"system": "c3s", "model": "ecmwf", "variable": "precip", "scale": "season"}
    register = pd.DataFrame([
        {**base, "product": "depassement_300mm", "method": "eqm"},
        {**base, "product": "depassement_500mm", "method": "eqm"},
        {**base, "product": "terciles", "method": "ngr"},
        {**base, "product": "quintile_bas", "method": "qdm"},
        {**base, "product": "depassement_median", "method": "ngr"},
        {**base, "product": "quintile_haut", "method": "ngr"},
    ])
    assert family_method(register, **base, products=products["seuils_cumul"]) == "eqm"
    assert family_method(register, **base, products=products["terciles"]) == "ngr"
    # majorité nette dans la famille des percentiles
    assert family_method(register, **base, products=products["seuils_percentiles"]) == "ngr"
    # registre absent ou produit inconnu : pas de méthode, donc pas de diagramme calibré
    assert family_method(pd.DataFrame(), **base, products=products["terciles"]) == ""
    assert family_method(register, **{**base, "model": "dwd"},
                         products=products["terciles"]) == ""
