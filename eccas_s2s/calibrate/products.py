"""
Which calibration method may serve which product (workflow E5, phase P3).

The CAPC-AC method note states the rule: *each indicator is a different
operational question, so it has its own method*. That cuts both ways — a method
is not tried on a product it cannot answer, and a product is not judged on a
method that was never meant for it.

Three things decide whether a method may serve a product:

**What the method produces.** A tercile logistic regression produces three
category probabilities and nothing else: it cannot return a rainfall total, and
it cannot answer "more than 200 mm". An extended logistic regression (ELR), an
EMOS/NGR law or a corrected ensemble produce a **full distribution**, so they can
answer every question — a value through the mean, an event through the
probability beyond its threshold.

**What the product asks of the distribution.** A value product reads the mean; a
three-category product reads two thresholds; an event reads one. A method that
only corrects the **mean** of the ensemble (a bias correction) can still serve a
probabilistic product, because the members carry the spread — but it will not
repair an overconfident spread, which is what EMOS and the logistic family are
for. Both are therefore tried, and the comparison decides.

**Whether the product is defined by a quantile or by a value.** A product defined
in millimetres is the one that most needs the distribution to sit on the
observation scale: quantile mapping (EQM, QDM) and EMOS are the natural
candidates, a simple mean correction rarely suffices. The register does not
prejudge this — it measures it — but the mapping keeps the candidates honest.

The table below is deliberately declarative: it is read, discussed and amended as
a table, not buried in the calibration code.

==========================  ====================================================
Produit                     Méthodes candidates
==========================  ====================================================
``cumul``, ``anomalie``     corrections de biais, EQM, QDM, NGR/EMOS
``spi``                     corrections de biais, EQM, QDM, NGR/EMOS
``terciles``                toutes, y compris la logistique par catégorie
``categorie_BN/AN``         toutes, y compris la logistique par catégorie
``spi_classes``             toutes sauf la logistique par catégorie
``quintile_*``,             corrections de biais, EQM, QDM, ELR, NGR/EMOS
``depassement_*``
==========================  ====================================================
"""
from __future__ import annotations

from eccas_s2s.products.catalogue import CATEGORIES, EVENT, VALUE

#: the baseline is always kept: every gain is measured against it
BASELINE = "raw"
#: methods that return a **value** as well as probabilities (a full distribution)
DISTRIBUTION_METHODS = ("bias_mean", "bias_ratio", "bias_scaling", "eqm", "qdm",
                        "ngr", "ngr_const", "elr")
#: methods that return **only** the three tercile categories
CATEGORICAL_ONLY = ("logistic",)
#: products those categorical methods can serve — the tercile family and nothing else
TERCILE_PRODUCTS = ("terciles", "categorie_BN", "categorie_AN")


def methods_for(product, available) -> list:
    """
    Methods worth fitting for one product, among those ``available``.

    ``available`` is the list of calibrators built for the variable
    (:func:`eccas_s2s.operations.calibrate_hindcast.calibrators_for`), so a method
    that does not apply to the variable — a ratio correction in °C, a quantile
    mapping without members — never reaches this function.

    The baseline (``raw``) is always first: the register compares every method
    with it, product by product and grid point by grid point.
    """
    out = []
    for cal in available:
        name = getattr(cal, "name", str(cal))
        if name == BASELINE:
            out.append(cal)
        elif name in CATEGORICAL_ONLY:
            if product.name in TERCILE_PRODUCTS:
                out.append(cal)
        elif name == "elr" and product.kind == VALUE:
            # an extended logistic regression describes the distribution by its
            # thresholds; reading a value from it is possible but is not what it
            # is for, and a bias correction answers that question directly
            continue
        else:
            out.append(cal)
    return out


def method_product_table(products, available) -> list[dict]:
    """Flat table ``(product, method, applicable)`` — the mapping as data, for the record."""
    rows = []
    for product in products:
        allowed = {getattr(c, "name", str(c)) for c in methods_for(product, available)}
        for cal in available:
            name = getattr(cal, "name", str(cal))
            rows.append({"product": product.name, "kind": product.kind,
                         "deciding_metric": product.metric, "method": name,
                         "applicable": name in allowed})
    return rows


def why_not(product, method: str) -> str:
    """One sentence saying why a method is not tried on a product (empty if it is)."""
    if method in CATEGORICAL_ONLY and product.name not in TERCILE_PRODUCTS:
        return ("la régression logistique par catégorie ne produit que les trois catégories "
                "des terciles : elle ne peut pas répondre à ce produit")
    if method == "elr" and product.kind == VALUE:
        return ("la régression logistique étendue décrit la distribution par ses seuils ; "
                "une valeur se corrige plus directement par un biais ou un quantile mapping")
    return ""


#: reminder of what each kind reads from the distribution, used in the documentation
READS_FROM_DISTRIBUTION = {VALUE: "la moyenne", CATEGORIES: "deux seuils",
                           EVENT: "un seuil"}
