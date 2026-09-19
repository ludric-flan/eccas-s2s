"""
Dry-season mask applied to precipitation products (step E8, decision of 2026-09-19).

A grid cell is masked ("dry") for a period when the **forecast** precipitation of
that period is too small for tercile or percentile categories to be meaningful.
Two methods, chosen in ``thresholds.yaml`` (``precip.dry_mask``) and overridable
at run time:

``relative`` (default)
    forecast < ``relative_fraction`` × climatological mean of the cell for the
    period (default 15 %).
``absolute``
    forecast < ``absolute_mm[scale]`` (mm), one threshold per time scale.

Which forecast value and which climatology to pass (see
:func:`forecast_value`):

* **raw products**: the ensemble mean, and the model's own hindcast climatology
  (raw products are expressed relative to the model climate);
* **calibrated products**: the median of the calibrated distribution, and the
  observed 1991–2020 normal (calibrated products live in the observation space).
"""
from __future__ import annotations

import numpy as np
import xarray as xr

METHODS = ("relative", "absolute")


def dry_mask_settings(thresholds: dict, method: str | None = None,
                      relative_fraction: float | None = None,
                      absolute_mm: dict | None = None) -> dict:
    """Configuration of the mask, with optional run-time overrides."""
    cfg = dict(thresholds["precip"]["dry_mask"])
    if method is not None:
        cfg["method"] = method
    if relative_fraction is not None:
        cfg["relative_fraction"] = relative_fraction
    if absolute_mm is not None:
        cfg["absolute_mm"] = {**cfg["absolute_mm"], **absolute_mm}
    if cfg["method"] not in METHODS:
        raise ValueError(f"méthode de masque sec inconnue {cfg['method']!r} ; valeurs : {METHODS}")
    if not 0 < float(cfg["relative_fraction"]) < 1:
        raise ValueError("relative_fraction doit être compris entre 0 et 1")
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


def dry_mask(forecast: xr.DataArray, settings: dict, climatology_mean: xr.DataArray | None = None,
             scale: str | None = None) -> xr.DataArray:
    """
    Boolean mask, True where the cell is dry (products are hidden there).

    Parameters
    ----------
    forecast : representative forecast value (see :func:`forecast_value`); may
        carry a ``period`` dimension with a ``scale`` coordinate.
    settings : output of :func:`dry_mask_settings`.
    climatology_mean : required for ``relative``; same grid (and periods) as
        ``forecast``.
    scale : time scale when ``forecast`` has no ``period``/``scale`` coordinate.

    Cells where the forecast is NaN (outside the domain) are not flagged dry.
    """
    method = settings["method"]
    if method == "relative":
        if climatology_mean is None:
            raise ValueError("la méthode relative demande la moyenne climatologique")
        limit = float(settings["relative_fraction"]) * climatology_mean
    else:
        limit = _threshold_by_scale(forecast, settings["absolute_mm"], scale)
    mask = (forecast < limit).where(forecast.notnull(), False).astype(bool)
    mask.name = "dry_mask"
    mask.attrs = {"long_name": "dry-season mask (True = masked)", "method": method,
                  "relative_fraction": settings.get("relative_fraction"),
                  "absolute_mm": str(settings.get("absolute_mm"))}
    return mask


def apply_mask(field: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Set masked cells to NaN (maps show them in the RCC dry-mask grey)."""
    return field.where(~mask)
