"""
Simple CEEAC map panels for the OSF notebooks and diagnostics.

One function, :func:`map_panel`, draws one or several 2-D fields on the CEEAC
domain with country borders from the CEEAC shapefile. Product maps with the RCC
palettes (Draft Framework annex) come in phase P7.
"""
from __future__ import annotations

from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

DEFAULT_EXTENT = (4.0, 36.0, -21.0, 25.0)   # lon_min, lon_max, lat_min, lat_max


@lru_cache(maxsize=2)
def country_borders(shapefile: str):
    """CEEAC country outlines (admin-1 shapefile dissolved by country)."""
    import geopandas as gpd
    gdf = gpd.read_file(shapefile).to_crs(epsg=4326)
    key = "adm0_a3" if "adm0_a3" in gdf.columns else gdf.columns[0]
    return gdf.dissolve(by=key)


def map_panel(fields, titles=None, *, shapefile=None, extent=DEFAULT_EXTENT, cmap="viridis",
              levels=None, vmin=None, vmax=None, extend="both", ncols=None, cbar_label="",
              suptitle=None, panel_size=(3.4, 4.2)):
    """
    Draw 2-D ``(latitude, longitude)`` DataArrays side by side, one shared colour bar.

    ``levels`` gives discrete colour classes (recommended for maps read by users);
    otherwise ``vmin``/``vmax`` set a continuous scale.
    """
    import cartopy.crs as ccrs

    fields = list(fields)
    n = len(fields)
    ncols = ncols or min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    norm = None
    if levels is not None:
        norm = BoundaryNorm(levels, plt.get_cmap(cmap).N, extend=extend)
    borders = country_borders(str(shapefile)) if shapefile else None
    mesh = None
    for ax, i in zip(axes.flat, range(nrows * ncols)):
        if i >= n:
            ax.set_visible(False)
            continue
        f = fields[i]
        mesh = ax.pcolormesh(f["longitude"], f["latitude"], f.values, cmap=cmap, norm=norm,
                             vmin=None if norm else vmin, vmax=None if norm else vmax,
                             shading="nearest", transform=ccrs.PlateCarree())
        if borders is not None:
            borders.boundary.plot(ax=ax, color="#4E4E4E", linewidth=0.5, transform=ccrs.PlateCarree())
        else:
            ax.coastlines(linewidth=0.5)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        if titles:
            ax.set_title(titles[i], fontsize=9)
    fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.6,
                 pad=0.04, aspect=40, extend=extend, label=cbar_label)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    return fig
