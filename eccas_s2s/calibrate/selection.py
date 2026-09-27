"""
Choosing the recommended method (workflow E5, "méthode recommandée").

Phase P3 produces several calibrated versions of the same forecast. Publishing
all of them would be honest but unusable; publishing the best-scoring one
without a guard would publish noise. The rule the chain applies is the one of the
CAPC-AC method note and of the workflow:

    a method is **recommended** for a model, variable, scale and period only if
    it beats the **raw** forecast *and* the **climatology**, on the criterion
    that matches the product.

Two criteria, because a calibration does two different jobs:

* the **deterministic** criterion is the correlation — a calibration cannot
  create correlation (it is a monotone transformation of the ensemble mean for
  most methods), so this criterion mainly protects against a method that
  destroys it;
* the **probabilistic** criterion is the RPSS, which is the one that moves: it
  is negative for most raw hindcasts of phase P2 (overconfidence) and it is
  exactly what a probabilistic calibration is meant to repair.

Beating climatology is read directly on the sign of the skill scores, which are
built against climatology: RPSS > 0 and MSESS > 0 *are* the statement "better
than climatology". Beating the raw forecast is read by comparing the two tables.

The result is a register — ``registry/calibration_methods.csv`` — that the
products (P7) and the multi-model weighting (P6) read: for each cell of the
matrix, the method to use, or ``raw`` when no calibration earned its place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: the criterion depends on the product, as the CAPC-AC note asks.
#:
#: * a **value** (a rainfall total, a temperature) is judged on the MSESS: it is
#:   built against climatology, so its sign is already the statement "better than
#:   climatology";
#: * a **probability** (terciles, thresholds, events) is judged on the RPSS.
#:
#: The correlation is reported but does **not** gate the choice. On 24 years a
#: cross-validated regression that shrinks towards the climatology inherits the
#: known negative bias of leave-one-out correlation (the left-out year's own
#: value is missing from the mean it is compared with), of order -1/(n-1) and
#: more when the shrinkage is strong. Using it as a gate would reject a method
#: for an artefact of the validation, not for a defect of the forecast.
CRITERION = {"deterministic": "msess_median", "probabilistic": "rpss_median"}
DETERMINISTIC = "pearson_median"
PROBABILISTIC = "rpss_median"
MARGIN = 0.01
#: floating-point slack: 0.29 - 0.30 is -0.010000000000000009, and a rule read
#: as "do not lose more than 0.01" must not flip on that.
TOL = 1e-9

KEYS = ["system", "model", "variable", "scale", "period"]


RAW = "raw"


def compare(calibrated: pd.DataFrame, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Put every calibrated method next to the **raw baseline of the same grid**.

    The reference is the ``raw`` method of the same table — the forecast as
    delivered, interpolated onto the observation grid and scored identically.
    Using the scores of phase P2 instead would compare two grids: they are
    computed at 1°, where a smoothed observation gives higher correlations, and
    every calibration would look worse than it is (a pure offset, which cannot
    change a correlation, appeared to lose 0.03 of it). ``raw`` may still be
    passed explicitly for a deliberate cross-grid comparison.
    """
    if raw is None:
        raw = calibrated[calibrated["method"] == RAW]
        calibrated = calibrated[calibrated["method"] != RAW]
    raw_cols = KEYS + [DETERMINISTIC, PROBABILISTIC, "msess_median", "groc_median", "label_fr"]
    raw_small = raw[[c for c in raw_cols if c in raw.columns]].copy()
    raw_small = raw_small.rename(columns={DETERMINISTIC: "raw_det", PROBABILISTIC: "raw_prob",
                                          "msess_median": "raw_msess",
                                          "groc_median": "raw_groc"})
    merged = calibrated.merge(raw_small, on=KEYS, how="left", suffixes=("", "_rawtab"))
    merged["gain_det"] = merged[DETERMINISTIC] - merged["raw_det"]
    merged["gain_prob"] = merged[PROBABILISTIC] - merged["raw_prob"]
    merged["gain_msess"] = merged["msess_median"] - merged["raw_msess"]
    # a system delivered as an ensemble mean (NMME) has no raw probability at
    # all: there is nothing to beat, only climatology to pass (decision D22)
    no_raw_prob = merged["raw_prob"].isna()
    merged["beats_raw"] = (merged["gain_prob"] >= MARGIN - TOL) | no_raw_prob
    merged["beats_climatology"] = merged[PROBABILISTIC] > 0
    merged["beats_raw_value"] = merged["gain_msess"] >= MARGIN - TOL
    merged["beats_climatology_value"] = merged["msess_median"] > 0
    merged["eligible_method"] = merged["beats_raw"] & merged["beats_climatology"]
    merged["eligible_value"] = merged["beats_raw_value"] & merged["beats_climatology_value"]
    return merged


def recommend(comparison: pd.DataFrame, product: str = "probabilistic") -> pd.DataFrame:
    """
    One recommended method per model, variable, scale and period.

    ``product`` picks the criterion: ``probabilistic`` (RPSS) for the category
    and threshold maps, ``deterministic`` (MSESS) for the value maps. Among the
    eligible methods the best score wins; when none is eligible the
    recommendation is ``raw``, and the register says why — that is a result, not
    a failure: it says this model is better used as it comes than transformed.
    """
    score = CRITERION[product]
    flag = "eligible_method" if product == "probabilistic" else "eligible_value"
    beats_clim = ("beats_climatology" if product == "probabilistic"
                  else "beats_climatology_value")
    beats_raw = "beats_raw" if product == "probabilistic" else "beats_raw_value"
    raw_col = "raw_prob" if product == "probabilistic" else "raw_msess"
    gain_col = "gain_prob" if product == "probabilistic" else "gain_msess"
    rows = []
    for key, group in comparison.groupby(KEYS, dropna=False):
        eligible = group[group[flag]]
        base = {**dict(zip(KEYS, key)), "product": product}
        if len(eligible):
            best = eligible.sort_values(score, ascending=False).iloc[0]
            rows.append({**base, "method": best["method"], "score": best[score],
                         "score_raw": best[raw_col], "gain": best[gain_col],
                         "gain_correlation": best["gain_det"],
                         "n_eligible": int(len(eligible)), "reason": "meilleur score éligible"})
        else:
            best_try = group.sort_values(score, ascending=False).iloc[0] if len(group) else None
            reason = "aucune méthode ne bat le brut et la climatologie"
            if best_try is not None and not bool(best_try[beats_clim]):
                reason = "aucune méthode ne bat la climatologie"
            elif best_try is not None and not bool(best_try[beats_raw]):
                reason = "aucune méthode ne bat la prévision brute"
            rows.append({**base, "method": "raw",
                         "score": best_try[raw_col] if best_try is not None else np.nan,
                         "score_raw": best_try[raw_col] if best_try is not None else np.nan,
                         "gain": 0.0, "gain_correlation": 0.0, "n_eligible": 0, "reason": reason})
    return pd.DataFrame(rows).sort_values(KEYS)


def summarise(recommendation: pd.DataFrame) -> pd.DataFrame:
    """How often each method is recommended, per variable and scale — for the report."""
    counts = (recommendation.groupby(["variable", "scale", "method"]).size()
              .rename("periods").reset_index())
    total = counts.groupby(["variable", "scale"])["periods"].transform("sum")
    counts["share"] = (counts["periods"] / total).round(3)
    return counts.sort_values(["variable", "scale", "periods"], ascending=[True, True, False])
