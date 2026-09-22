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

#: what "better" means, per criterion; the margin avoids crowning a method for
#: a difference that 24 years cannot resolve.
DETERMINISTIC = "pearson_median"
PROBABILISTIC = "rpss_median"
MARGIN = 0.01
#: floating-point slack: 0.29 - 0.30 is -0.010000000000000009, and a rule read
#: as "do not lose more than 0.01" must not flip on that.
TOL = 1e-9

KEYS = ["system", "model", "variable", "scale", "period"]


def compare(raw: pd.DataFrame, calibrated: pd.DataFrame) -> pd.DataFrame:
    """
    Put every calibrated method next to the raw scores of the same period.

    Returns one row per method with the two gains (calibrated − raw) and the
    flags that decide whether the method may be recommended.
    """
    raw_cols = KEYS + [DETERMINISTIC, PROBABILISTIC, "msess_median", "groc_median", "label_fr"]
    raw_small = raw[[c for c in raw_cols if c in raw.columns]].copy()
    raw_small = raw_small.rename(columns={DETERMINISTIC: "raw_det", PROBABILISTIC: "raw_prob",
                                          "msess_median": "raw_msess",
                                          "groc_median": "raw_groc"})
    merged = calibrated.merge(raw_small, on=KEYS, how="left", suffixes=("", "_rawtab"))
    merged["gain_det"] = merged[DETERMINISTIC] - merged["raw_det"]
    merged["gain_prob"] = merged[PROBABILISTIC] - merged["raw_prob"]
    merged["beats_raw"] = merged["gain_prob"] >= MARGIN - TOL
    merged["beats_climatology"] = merged[PROBABILISTIC] > 0
    merged["keeps_correlation"] = merged["gain_det"] >= -MARGIN - TOL
    merged["eligible_method"] = (merged["beats_raw"] & merged["beats_climatology"]
                                 & merged["keeps_correlation"])
    return merged


def recommend(comparison: pd.DataFrame) -> pd.DataFrame:
    """
    One recommended method per model, variable, scale and period.

    Among the eligible methods the best RPSS wins; when none is eligible the
    recommendation is ``raw``, and the register says why — that is a result, not
    a failure: it tells the forecaster that this model, on this period, is better
    used as it comes than transformed.
    """
    rows = []
    for key, group in comparison.groupby(KEYS, dropna=False):
        eligible = group[group["eligible_method"]]
        if len(eligible):
            best = eligible.sort_values(PROBABILISTIC, ascending=False).iloc[0]
            rows.append({**dict(zip(KEYS, key)), "method": best["method"],
                         "rpss": best[PROBABILISTIC], "rpss_raw": best["raw_prob"],
                         "gain_prob": best["gain_prob"], "gain_det": best["gain_det"],
                         "n_eligible": int(len(eligible)), "reason": "meilleur RPSS éligible"})
        else:
            raw_prob = group["raw_prob"].iloc[0] if "raw_prob" in group else np.nan
            best_try = (group.sort_values(PROBABILISTIC, ascending=False).iloc[0]
                        if len(group) else None)
            reason = "aucune méthode ne bat le brut et la climatologie"
            if best_try is not None and not bool(best_try["beats_climatology"]):
                reason = "aucune méthode ne bat la climatologie (RPSS ≤ 0)"
            elif best_try is not None and not bool(best_try["beats_raw"]):
                reason = "aucune méthode ne bat la prévision brute"
            rows.append({**dict(zip(KEYS, key)), "method": "raw", "rpss": raw_prob,
                         "rpss_raw": raw_prob, "gain_prob": 0.0, "gain_det": 0.0,
                         "n_eligible": 0, "reason": reason})
    return pd.DataFrame(rows).sort_values(KEYS)


def summarise(recommendation: pd.DataFrame) -> pd.DataFrame:
    """How often each method is recommended, per variable and scale — for the report."""
    counts = (recommendation.groupby(["variable", "scale", "method"]).size()
              .rename("periods").reset_index())
    total = counts.groupby(["variable", "scale"])["periods"].transform("sum")
    counts["share"] = (counts["periods"] / total).round(3)
    return counts.sort_values(["variable", "scale", "periods"], ascending=[True, True, False])
