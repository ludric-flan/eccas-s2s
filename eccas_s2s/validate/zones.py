"""
Verification zones: the whole CEEAC domain and the three rainfall regimes
(``config/domains.yaml``).

Scores are reported per zone (Draft §5.3: verification over predefined domains
and homogeneous climate zones) and the eligibility of a model (§3.3) is the
fraction of the **domain** where its skill is positive.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

DOMAIN = "domain"


def zone_masks(reference: xr.DataArray, domains: dict, valid: xr.DataArray | None = None
               ) -> dict[str, xr.DataArray]:
    """
    Boolean masks of the domain and of each zone, on the grid of ``reference``.

    ``valid`` (e.g. the cells where the observation exists) restricts every
    mask, so that ocean or missing cells never enter a zone average.
    """
    lat, lon = reference["latitude"], reference["longitude"]
    lon_min, lon_max, lat_min, lat_max = domains["domain"]["extent"]
    inside = ((lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max))
    inside = inside.broadcast_like(reference.isel(
        {d: 0 for d in reference.dims if d not in ("latitude", "longitude")}, drop=True))
    if valid is not None:
        inside = inside & valid
    masks = {DOMAIN: inside}
    for name, spec in domains["zones"].items():
        lo, hi = spec["lat"]
        masks[name] = inside & (lat >= lo) & (lat <= hi)
    return {k: v.rename(k) for k, v in masks.items()}


def fraction_positive(score: xr.DataArray, mask: xr.DataArray, threshold: float = 0.0) -> float:
    """Area fraction of ``mask`` where ``score`` exceeds ``threshold`` (cos-lat weighted)."""
    w = np.cos(np.deg2rad(score["latitude"])).broadcast_like(score).where(mask & score.notnull())
    total = float(w.sum())
    if total == 0:
        return float("nan")
    return float(w.where(score > threshold).sum() / total)
