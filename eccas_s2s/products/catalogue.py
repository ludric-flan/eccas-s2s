"""
Catalogue of products, and the metric that decides for each one (E5, E8).

The CAPC-AC method note states the rule this module encodes: *each indicator is a
different operational question, so it has its own method of calibration and its
own verification*. The consequence for the register of recommended methods is
that a single criterion cannot serve every product — a rainfall total is judged
on its error, a tercile map on its ranked probability score, a threshold map on
its Brier score and its discrimination.

So, for each product of the reference chain (``run_forecast_v2``), this catalogue
says three things: what quantity is read from the predictive distribution, which
metric **decides** the recommended method, and which metrics are reported beside
it for the record.

===========================  =================  ==========  ===========================
Produit                      Quantité            Décide      Rapporté aussi
===========================  =================  ==========  ===========================
``cumul``                    valeur              MSESS       RMSE, MAE, biais, Pearson
``anomalie``                 valeur              ACC         Pearson, RMSE
``terciles``                 3 catégories        RPSS        GROC, aire ROC par catégorie
``categorie_BN/AN``          événement           BSS         ROC (aire − 0,5)
``quintile_bas/haut``        événement           BSS         ROC (aire − 0,5)
``depassement_median``       événement           BSS         ROC (aire − 0,5)
``depassement_<seuil>mm``    événement           BSS         ROC (aire − 0,5)
``spi_classes``              3 catégories        RPSS        GROC
``spi``                      valeur              MSESS       Pearson
===========================  =================  ==========  ===========================

Two points worth remembering when reading the register:

* a **value** product is judged on the MSESS and an **anomaly** on the ACC, and
  the two disagree on purpose: a method can improve the amplitude of the error
  (MSESS) while leaving the year-to-year signal untouched (ACC), and the reverse
  happens with a strong shrinkage;
* an **event** is judged on the Brier skill score against the observed base
  rate, because a rare event is easy to "win" on accuracy alone; its
  discrimination is reported as ``roc_skill`` — the ROC area **minus 0.5**, so
  that a positive number means "useful" for every metric of the register.
"""
from __future__ import annotations

from dataclasses import dataclass, field

VALUE = "value"
CATEGORIES = "categories"
EVENT = "event"


@dataclass(frozen=True)
class Product:
    """One product of the catalogue and the way it is judged."""

    name: str
    kind: str
    metric: str
    reported: tuple = ()
    label: str = ""
    #: for an event: how the observed occurrence is defined
    event: str = ""
    #: extra information (threshold in mm, quantile, category)
    parameter: float | str | None = None
    variables: tuple = ("precip", "t2m", "tmax", "tmin")


def catalogue(variable: str, scale: str, thresholds: dict) -> list[Product]:
    """
    The products to score for a variable and a scale.

    ``thresholds`` is the ``precip`` block of ``config/thresholds.yaml``; the
    millimetre thresholds depend on the scale (a 50 mm dekad and a 50 mm season
    are not the same event), and they only apply to rainfall.
    """
    items = [
        Product("cumul", VALUE, "msess", ("rmse", "mae", "bias", "pearson"),
                "cumul (mm)" if variable == "precip" else "valeur (°C)"),
        Product("anomalie", VALUE, "acc", ("pearson", "rmse"), "anomalie"),
        Product("terciles", CATEGORIES, "rpss", ("groc",), "terciles BN / NN / AN"),
        Product("categorie_BN", EVENT, "bss", ("roc_skill",), "catégorie déficitaire",
                event="obs < q33"),
        Product("categorie_AN", EVENT, "bss", ("roc_skill",), "catégorie excédentaire",
                event="obs > q67"),
        Product("quintile_bas", EVENT, "bss", ("roc_skill",), "P20 inférieur",
                event="obs < P20", parameter=0.20),
        Product("quintile_haut", EVENT, "bss", ("roc_skill",), "P80 supérieur",
                event="obs > P80", parameter=0.80),
        Product("depassement_median", EVENT, "bss", ("roc_skill",), "au-dessus de la médiane",
                event="obs > médiane", parameter=0.50),
    ]
    if variable == "precip":
        for mm in thresholds.get("exceedance_mm", {}).get(scale, []):
            items.append(Product(f"depassement_{int(mm)}mm", EVENT, "bss", ("roc_skill",),
                                 f"cumul ≥ {int(mm)} mm", event=f"obs >= {mm} mm",
                                 parameter=float(mm), variables=("precip",)))
        items += [
            Product("spi_classes", CATEGORIES, "rpss", ("groc",), "classes SPI (sec/normal/humide)",
                    variables=("precip",)),
            Product("spi", VALUE, "msess", ("pearson",), "SPI déterministe",
                    variables=("precip",)),
        ]
    return [p for p in items if variable in p.variables]


def deciding_metric(product: str, variable: str = "precip", scale: str = "season",
                    thresholds: dict | None = None) -> str:
    """The metric that decides the recommended method for a product."""
    for p in catalogue(variable, scale, thresholds or {}):
        if p.name == product:
            return p.metric
    raise KeyError(f"produit inconnu : {product}")


#: families of probabilistic products drawn on **one** reliability figure and
#: **one** ROC figure: classes that answer the same question belong on the same
#: axes, where they can be compared. The tercile categories (``categorie_BN`` and
#: ``categorie_AN``) are the BN and AN curves of the ``terciles`` family, so they
#: get no figure of their own.
FAMILY_LABEL = {"terciles": "terciles", "classes_spi": "classes SPI",
                "seuils_percentiles": "seuils de percentile",
                "seuils_cumul": "seuils de cumul"}


def families(variable: str, scale: str, thresholds: dict) -> dict:
    """
    Diagram families: ``{family name: [products]}``, empty families dropped.

    A family gathers the classes that share a set of axes — the three terciles,
    the three SPI classes, the percentile thresholds (P20, median, P80), the
    millimetre thresholds of the scale. Grouping them is not only economical: a
    forecaster reads the overconfidence of "below normal" against that of "above
    normal" on the same figure, and sees at a glance whether the 300 mm threshold
    behaves like the 500 mm one.
    """
    items = catalogue(variable, scale, thresholds)
    by_name = {p.name: p for p in items}
    groups = {
        "terciles": [by_name[n] for n in ("terciles",) if n in by_name],
        "classes_spi": [by_name[n] for n in ("spi_classes",) if n in by_name],
        "seuils_percentiles": [by_name[n] for n in
                               ("quintile_bas", "depassement_median", "quintile_haut")
                               if n in by_name],
        "seuils_cumul": [p for p in items
                         if p.name.startswith("depassement_") and p.name.endswith("mm")],
    }
    return {k: v for k, v in groups.items() if v}
