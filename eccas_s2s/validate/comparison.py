"""
Raw against calibrated, grid point by grid point (workflow E5, phase P3).

Phase P2 says what the raw products are worth; phase P3 says whether a
calibration improves them. The comparison has to be **spatial**: a calibration
that lifts the equatorial belt and ruins the Sahel can leave the domain median
unchanged, and a method chosen on that median would be the wrong method for both
regions. So the gain is computed **on every grid point**, kept as a map, and only
then summarised.

For one product, one metric and one period, the gain of a method is

    gain(x) = metric_calibrated(x) − metric_raw(x)

at each grid point *x*, and the summary carries four numbers:

``median_gain``
    the gain of the typical grid point of the mask;
``fraction_improved``
    the share of the mask where the gain is positive — a method must help
    *somewhere broad*, not merely on average;
``fraction_repaired``
    the share where the raw metric was below its no-skill value and the
    calibrated one is above: the product becomes usable there, which is the
    operational question;
``fraction_broken``
    the reverse, where a usable raw product stops being usable. A method that
    repairs 30 % and breaks 20 % is not the same as one that repairs 30 % and
    breaks nothing, and the median gain alone hides the difference.

The metric is compared with itself: the same function, the same observation
context, the same leave-one-year-out thresholds, the same periods and the same
mask are used for the raw and the calibrated distribution
(:mod:`eccas_s2s.validate.product_scores`). The difference measured can therefore
only come from the calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.validate.product_scores import LOWER_IS_BETTER, NO_SKILL_VALUE


def no_skill_value(metric: str) -> float:
    """Value the metric takes without skill (0, or 0.5 for an area under the ROC curve)."""
    base = metric.split("_")[0]
    return float(NO_SKILL_VALUE.get(metric, NO_SKILL_VALUE.get(base, 0.0)))


def better(metric: str, a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """``a`` better than ``b``, in the direction that metric improves."""
    return (a < b) if metric.split("_")[0] in LOWER_IS_BETTER else (a > b)


def gain_map(calibrated: xr.DataArray, raw: xr.DataArray, metric: str) -> xr.DataArray:
    """
    Gain of a calibration over the raw forecast, per grid point.

    An error (RMSE, MAE, |bias|) improves when it *decreases*, so its gain is
    the reduction; every skill score improves when it increases. The sign of a
    gain is therefore always "positive is better", whatever the metric.
    """
    diff = (raw - calibrated) if metric.split("_")[0] in LOWER_IS_BETTER else (calibrated - raw)
    diff.name = "gain"
    diff.attrs = {"long_name": f"gain de la calibration sur {metric}",
                  "metric": metric, "convention": "positif = la calibration améliore",
                  "no_skill_value": no_skill_value(metric)}
    return diff


def compare_maps(calibrated: xr.DataArray, raw: xr.DataArray, metric: str,
                 mask: xr.DataArray, method: str = "", product: str = "",
                 period: str = "") -> dict:
    """
    Summary of one comparison: median gain and the three shares over the mask.

    Everything is weighted by the cosine of latitude (through
    :func:`eccas_s2s.core.geo.fraction_above`), so a degree square near the
    equator does not count as much as one at 20°N.
    """
    from eccas_s2s.core.geo import fraction_above

    gain = gain_map(calibrated, raw, metric)
    threshold = no_skill_value(metric)
    raw_ok = better(metric, raw, xr.full_like(raw, threshold))
    cal_ok = better(metric, calibrated, xr.full_like(calibrated, threshold))
    valid = mask & raw.notnull() & calibrated.notnull()

    return {
        "product": product, "metric": metric, "method": method, "period": period,
        "median_raw": float(raw.where(valid).median()),
        "median_calibrated": float(calibrated.where(valid).median()),
        "median_gain": float(gain.where(valid).median()),
        "fraction_improved": fraction_above(gain, valid, 0.0),
        "fraction_repaired": fraction_above((~raw_ok & cal_ok).astype(float), valid, 0.5),
        "fraction_broken": fraction_above((raw_ok & ~cal_ok).astype(float), valid, 0.5),
        "n_cells": int(valid.sum()),
    }


def recommend(table: pd.DataFrame, min_improved: float = 0.5,
              min_gain: float = 0.0) -> pd.DataFrame:
    """
    Recommended method per product, from the comparison table.

    A method is a candidate when it improves the **typical** grid point
    (``median_gain`` above ``min_gain``) **and** improves a broad part of the
    mask (``fraction_improved`` at least ``min_improved``): the two conditions
    together exclude a method that wins big in one corner and loses everywhere
    else. Among the candidates, the recommended one has the largest median gain;
    ties are broken by the repaired area, because making a product usable where
    it was not is worth more than improving one that already was.

    When no method qualifies, the recommendation is ``raw`` — the product is
    published uncalibrated, and the register says so.
    """
    keys = [k for k in ("system", "model", "variable", "scale", "product", "metric")
            if k in table.columns]
    rows = []
    for key, group in table.groupby(keys, dropna=False):
        candidates = group[(group["median_gain"] > min_gain)
                           & (group["fraction_improved"] >= min_improved)]
        record = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        if candidates.empty:
            record.update({"method": "raw", "median_gain": 0.0, "fraction_improved": np.nan,
                           "fraction_repaired": 0.0, "fraction_broken": 0.0,
                           "reason": "aucune méthode ne bat le brut sur la maille typique"})
        else:
            best = candidates.sort_values(["median_gain", "fraction_repaired"],
                                          ascending=False).iloc[0]
            record.update({"method": best["method"], "median_gain": float(best["median_gain"]),
                           "fraction_improved": float(best["fraction_improved"]),
                           "fraction_repaired": float(best["fraction_repaired"]),
                           "fraction_broken": float(best["fraction_broken"]),
                           "reason": ""})
        record["n_methods_tried"] = int(len(group))
        record["n_methods_beating_raw"] = int(len(candidates))
        rows.append(record)
    return pd.DataFrame(rows).sort_values(keys)
