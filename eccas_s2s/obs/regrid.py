"""
Spatial harmonisation between observation and model grids (Draft Framework §3.1.7).

CHIRPS (0.05°) and the C3S grid (1°, cell centres at x.5°) are *nested*: every
1° cell contains exactly 20 × 20 CHIRPS cells. Conservative remapping then
reduces to an exact **block average**, with no interpolation. This module checks
the nesting before averaging, so a mis-aligned grid fails loudly instead of
being silently shifted.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def _resolution(coord: np.ndarray) -> float:
    """Grid spacing from the full extent (robust to float32-stored coordinates)."""
    c = coord.astype("float64")
    d = np.diff(c)
    if not np.allclose(d, d[0], rtol=1e-3):
        raise ValueError("grille non régulière")
    return float((c[-1] - c[0]) / (len(c) - 1))


def snap_coords(da: xr.DataArray, decimals: int = 4) -> xr.DataArray:
    """
    Round latitude/longitude to their nominal values (e.g. 5.024994 -> 5.025).

    CHIRPS stores coordinates in float32; rounding makes grids comparable and
    selections by value exact.
    """
    return da.assign_coords(latitude=np.round(da["latitude"].values.astype("float64"), decimals),
                            longitude=np.round(da["longitude"].values.astype("float64"), decimals))


def block_average(da: xr.DataArray, target_res: float = 1.0, min_valid: float = 0.5) -> xr.DataArray:
    """
    Average a fine regular grid onto nested ``target_res`` cells.

    Requirements checked: the target resolution is an integer multiple of the
    source resolution, and the source cell edges fall on multiples of
    ``target_res`` (e.g. CHIRPS edges at -20.0° and 5.0°). A target cell is NaN
    when less than ``min_valid`` of its source cells are valid (coasts, lakes).
    Output coordinates are the target cell centres (e.g. -19.5, -18.5 ...).
    """
    for dim in ("latitude", "longitude"):
        coord = da[dim].values
        res = abs(_resolution(coord))
        factor = target_res / res
        if abs(factor - round(factor)) > 1e-3:
            raise ValueError(f"{dim} : {target_res}° n'est pas un multiple de {res}°")
        first_edge = float(min(coord[0], coord[-1])) - res / 2
        if abs(first_edge / target_res - round(first_edge / target_res)) > 1e-3:
            raise ValueError(f"{dim} : bord {first_edge:.4f}° non aligné sur la grille {target_res}°")
        if len(coord) % round(factor):
            raise ValueError(f"{dim} : {len(coord)} mailles, non divisible par {round(factor)}")

    f_lat = round(target_res / abs(_resolution(da["latitude"].values)))
    f_lon = round(target_res / abs(_resolution(da["longitude"].values)))
    coarse = dict(latitude=f_lat, longitude=f_lon)
    valid = da.notnull().coarsen(**coarse, boundary="exact").mean()
    mean = da.coarsen(**coarse, boundary="exact").mean()
    out = mean.where(valid >= min_valid)
    out = out.assign_coords(latitude=np.round(out["latitude"].values.astype("float64"), 4),
                            longitude=np.round(out["longitude"].values.astype("float64"), 4))
    out.attrs = {**da.attrs, "regridding": f"exact block average to {target_res} deg "
                                           f"(min valid fraction {min_valid})"}
    return out


def match_model_grid(model: xr.DataArray, obs_coarse: xr.DataArray) -> xr.DataArray:
    """Select the model cells that coincide with the (block-averaged) observation cells."""
    sel = model.sel(latitude=obs_coarse["latitude"].values, longitude=obs_coarse["longitude"].values,
                    method="nearest", tolerance=1e-3)
    return sel.assign_coords(latitude=obs_coarse["latitude"].values,
                             longitude=obs_coarse["longitude"].values)
