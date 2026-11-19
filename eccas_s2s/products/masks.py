"""
Dry-season mask applied to precipitation products (step E8, decision of 2026-09-19).

A grid cell is masked ("dry") for a period when that period brings so little
rain **there** that a tercile or a percentile has nothing to say. Three methods,
chosen in ``thresholds.yaml`` (``precip.dry_mask``) and overridable at run time:

``iri`` (default since 12/11/2026, shared with the ENSO–CEEAC study)
    the IRI "dry season" criterion: for a 3-month season, the cell's
    climatological rainfall is below ``iri_share_pct`` (15 %) of its
    climatological **annual** total — "In regions and seasons where precipitation
    is sufficiently rare (<15% of the annual climatological precipitation in that
    season), no forecast is issued" (Wilks & Godfrey 2002, J. Climate 15, p. 1370)
    — **or** below ``iri_floor_mm`` (30 mm). The floor marks the deserts, where a
    season can carry 40 % of an annual total of 10 mm and pass the share test.
    For another period length both thresholds are scaled by the length of the
    period in units of 3 months (1/3 for a month; days/91.3 for a dekad).
``relative``
    the cell's climatological mean for the period is below ``relative_fraction``
    of its **pro-rata share of the year** — that is, of what the period would
    receive if the annual rainfall of that cell were spread evenly over the
    calendar. Everything is read cell by cell: the period climatology, the
    annual climatology and the ratio are three fields on the grid, and no
    regional average enters anywhere.
``absolute``
    forecast < ``absolute_mm[scale]`` (mm), one threshold per time scale.

Why the relative method reads the **climatology** and not the forecast
--------------------------------------------------------------------
Until 29/09/2026 the relative method compared the *forecast* with the cell's own
climatological mean. That criterion cannot mark a desert: over northern Chad the
October-December normal is 0,1 mm, so a forecast of 6,8 mm is fifty times the
normal — the cell is anything but relatively dry, and it stayed coloured while
being the driest place of the domain. The criterion answered "is the forecast
collapsing against the local normal?", a question about the **anomaly**, when
the mask has to answer "does this period bring any rain here at all?", a
question about the **climate**. The ratio to the pro-rata share answers the
second, works unchanged for a dekad, a month or a season — the share it is
compared with carries the length of the period — and stays a property of the
cell, never of its neighbours.

Which forecast value and which climatology to pass (see :func:`forecast_value`):

* **raw products**: the ensemble mean; the seasonality is read on the observed
  1991–2020 normal, because it describes the climate of the place and not the
  climate of a model;
* **calibrated products**: the median of the calibrated distribution, same
  observed normal.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

METHODS = ("iri", "relative", "absolute")


def dry_mask_settings(thresholds: dict, method: str | None = None,
                      relative_fraction: float | None = None,
                      absolute_mm: dict | None = None,
                      iri_share_pct: float | None = None,
                      iri_floor_mm: float | None = None) -> dict:
    """Configuration of the mask, with optional run-time overrides."""
    cfg = dict(thresholds["precip"]["dry_mask"])
    cfg.setdefault("iri_share_pct", 15.0)
    cfg.setdefault("iri_floor_mm", 30.0)
    if method is not None:
        cfg["method"] = method
    if relative_fraction is not None:
        cfg["relative_fraction"] = relative_fraction
    if absolute_mm is not None:
        cfg["absolute_mm"] = {**cfg["absolute_mm"], **absolute_mm}
    if iri_share_pct is not None:
        cfg["iri_share_pct"] = iri_share_pct
    if iri_floor_mm is not None:
        cfg["iri_floor_mm"] = iri_floor_mm
    if cfg["method"] not in METHODS:
        raise ValueError(f"méthode de masque sec inconnue {cfg['method']!r} ; valeurs : {METHODS}")
    if not 0 < float(cfg["relative_fraction"]) < 1:
        raise ValueError("relative_fraction doit être compris entre 0 et 1")
    if not 0 < float(cfg["iri_share_pct"]) < 100:
        raise ValueError("iri_share_pct doit être compris entre 0 et 100 (%)")
    if float(cfg["iri_floor_mm"]) < 0:
        raise ValueError("iri_floor_mm doit être positif ou nul")
    return cfg


def forecast_value(ensemble: xr.DataArray, statistic: str = "mean", member_dim: str = "number") -> xr.DataArray:
    """Representative forecast value: ensemble ``mean`` (raw) or ``median`` (calibrated)."""
    if statistic == "mean":
        return ensemble.mean(member_dim)
    if statistic == "median":
        return ensemble.median(member_dim)
    raise ValueError("statistic doit valoir 'mean' ou 'median'")


def _threshold_by_scale(forecast: xr.DataArray, absolute_mm: dict, scale: str | None) -> xr.DataArray | float:
    if "period" in forecast.dims and "scale" in forecast.coords:
        scales = forecast["scale"].values
        missing = sorted(set(scales) - set(absolute_mm))
        if missing:
            raise KeyError(f"absolute_mm sans seuil pour les échelles {missing}")
        return xr.DataArray([float(absolute_mm[s]) for s in scales], dims="period",
                            coords={"period": forecast["period"]})
    if scale is None:
        raise ValueError("indiquer scale= (decade, month, season) pour la méthode absolute")
    return float(absolute_mm[scale])


DAYS_IN_YEAR = 365.25


def three_month_units(period_days: float) -> float:
    """
    Length of the period in units of 3 months, the length the IRI criterion is
    defined for: whole months for a month or a season (1/3, 1), days/91.3 for a
    dekad. Rounding to whole months keeps a 90- or 92-day season at exactly 15 %.
    """
    days = float(period_days)
    if days >= 28:
        return round(days / (DAYS_IN_YEAR / 12)) / 3.0
    return days / (DAYS_IN_YEAR / 4)


def seasonal_share(climatology_mean: xr.DataArray, annual_mean: xr.DataArray) -> xr.DataArray:
    """
    Share of the annual rainfall the period carries, cell by cell (%).

    A product in its own right — it shows where a single season *is* the year,
    and therefore where a forecast for that season matters most — and the
    quantity the relative mask is built on.
    """
    share = 100.0 * climatology_mean / annual_mean.where(annual_mean > 0)
    share.name = "seasonal_share"
    share.attrs = {"long_name": "share of the annual rainfall falling in the period",
                   "units": "%", "reference": "climatologie observée 1991-2020"}
    return share


def dry_mask(forecast: xr.DataArray, settings: dict, climatology_mean: xr.DataArray | None = None,
             scale: str | None = None, annual_mean: xr.DataArray | None = None,
             period_days: float | None = None) -> xr.DataArray:
    """
    Boolean mask, True where the cell is dry (products are hidden there).

    Parameters
    ----------
    forecast : representative forecast value (see :func:`forecast_value`); may
        carry a ``period`` dimension with a ``scale`` coordinate. Only its
        missing cells are read by the relative method — the mask is a statement
        about the climate of the cell, not about this particular forecast.
    settings : output of :func:`dry_mask_settings`.
    climatology_mean : climatological mean of the **period**, required for
        ``relative``; same grid as ``forecast``.
    annual_mean : climatological **annual** total of the cell, required for
        ``relative``.
    period_days : length of the period in days, required for ``relative``: it is
        what turns the annual total into the share the period is entitled to.
    scale : time scale when ``forecast`` has no ``period``/``scale`` coordinate.

    Cells where the forecast is NaN (outside the domain) are not flagged dry.
    """
    method = settings["method"]
    if method == "iri":
        if climatology_mean is None or annual_mean is None or period_days is None:
            raise ValueError("la méthode iri demande la climatologie de la période, "
                             "la climatologie annuelle et la durée de la période")
        units = three_month_units(period_days)
        share_limit = float(settings["iri_share_pct"]) / 100.0 * annual_mean * units
        floor = float(settings["iri_floor_mm"]) * units
        dry = (climatology_mean < share_limit) | (climatology_mean < floor)
        mask = dry.where(forecast.notnull(), False).astype(bool)
    elif method == "relative":
        if climatology_mean is None or annual_mean is None or period_days is None:
            raise ValueError("la méthode relative demande la climatologie de la période, "
                             "la climatologie annuelle et la durée de la période")
        # the share the period would receive if the rain of the cell were spread
        # evenly over the year: a quarter of it for a season, a thirtieth for a
        # dekad. Computed cell by cell, like everything else here.
        prorata = annual_mean * (float(period_days) / DAYS_IN_YEAR)
        limit = float(settings["relative_fraction"]) * prorata
        dry = climatology_mean < limit
        mask = dry.where(forecast.notnull(), False).astype(bool)
    else:
        limit = _threshold_by_scale(forecast, settings["absolute_mm"], scale)
        mask = (forecast < limit).where(forecast.notnull(), False).astype(bool)
    mask.name = "dry_mask"
    mask.attrs = {"long_name": "dry-season mask (True = masked)", "method": method,
                  "relative_fraction": settings.get("relative_fraction"),
                  "period_days": period_days if period_days is not None else "",
                  "iri_share_pct": settings.get("iri_share_pct"),
                  "iri_floor_mm": settings.get("iri_floor_mm"),
                  "criterion": {"iri": "climatologie de la période < 15 % de la climatologie annuelle "
                                       "OU < 30 mm (pour 3 mois ; seuils × durée/3 mois) — IRI, Wilks & Godfrey 2002",
                                "relative": "climatologie de la période < fraction × part pro-rata "
                                            "de la climatologie annuelle de la maille"}.get(
                                    method, "prévision < seuil absolu de l'échelle"),
                  "absolute_mm": str(settings.get("absolute_mm"))}
    return mask


def apply_mask(field: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Set masked cells to NaN (maps show them in the RCC dry-mask grey)."""
    return field.where(~mask)
