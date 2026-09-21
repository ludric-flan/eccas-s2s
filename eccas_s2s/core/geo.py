"""
CEEAC geographic mask (workflow steps E4 and E8).

Everything the chain verifies and publishes is restricted to the **land area of
the CEEAC**, defined by the shapefile of the CAPC-AC. The reference chain builds
the mask by testing whether each grid-point centre falls inside the union of the
country polygons (``create_geographic_mask`` of ``s2s_data.py``); the same
definition is kept here so that a score, a map and a bulletin always cover
exactly the same cells.

Two consequences worth remembering:

* a 1° cell whose centre is at sea is dropped even if part of the cell is over
  land — at this resolution the coastal fringe is about one cell wide, and
  ``buffer_deg`` can widen the geometry when a product needs it;
* scores are therefore computed on land points only: the ocean no longer enters
  a median, a positive-skill fraction, or a pooled reliability diagram.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import xarray as xr


@lru_cache(maxsize=4)
def ceeac_geometry(shapefile: str, buffer_deg: float = 0.0):
    """Union of the CEEAC polygons (cached; ``buffer_deg`` widens it if needed)."""
    import geopandas as gpd

    gdf = gpd.read_file(shapefile).to_crs(epsg=4326)
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
    return geom.buffer(buffer_deg) if buffer_deg else geom


def ceeac_mask(shapefile: str, latitudes, longitudes, buffer_deg: float = 0.0) -> xr.DataArray:
    """
    Boolean ``(latitude, longitude)`` mask, True for the cells inside the CEEAC.

    Uses the prepared geometry, so the point-in-polygon test stays cheap even on
    a 0.05° grid.
    """
    from shapely.geometry import Point
    from shapely.prepared import prep

    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    geom = prep(ceeac_geometry(str(shapefile), float(buffer_deg)))
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    inside = np.fromiter((geom.contains(Point(x, y))
                          for x, y in zip(lon_grid.ravel(), lat_grid.ravel())),
                         dtype=bool, count=lon_grid.size).reshape(lon_grid.shape)
    mask = xr.DataArray(inside, dims=("latitude", "longitude"),
                        coords={"latitude": lat, "longitude": lon}, name="ceeac_mask")
    mask.attrs = {"long_name": "CEEAC land mask (grid-point centre inside the shapefile)",
                  "shapefile": str(shapefile), "buffer_deg": float(buffer_deg),
                  "n_cells": int(inside.sum())}
    return mask


def mask_like(shapefile: str, template: xr.DataArray, buffer_deg: float = 0.0) -> xr.DataArray:
    """Mask on the grid of ``template`` (which must carry latitude and longitude)."""
    return ceeac_mask(shapefile, template["latitude"].values, template["longitude"].values,
                      buffer_deg)


def apply_mask(field: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Keep the cells inside the mask, NaN elsewhere (``where`` with the attrs kept)."""
    out = field.where(mask)
    out.attrs = dict(field.attrs)
    return out


def fraction_above(score: xr.DataArray, mask: xr.DataArray, threshold: float = 0.0) -> float:
    """
    Area fraction of the mask where ``score`` exceeds ``threshold`` (cos-lat weighted).

    The score is taken raw, not as a boolean: a comparison turns NaN into False,
    which would count a missing cell as a failure instead of excluding it. Cells
    where the score is missing leave both the numerator and the denominator.
    """
    valid = mask & score.notnull()
    w = np.cos(np.deg2rad(score["latitude"])).broadcast_like(score).where(valid)
    total = float(w.sum())
    if total == 0:
        return float("nan")
    return float(w.where(score > threshold).sum() / total)
