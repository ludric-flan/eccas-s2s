"""
CEEAC maps in the CAPC-AC house style (workflow steps E4, E8, E9).

Style taken from the reference chain (``s2s_plotting_v2.py``): country
boundaries in black and first-level administrative boundaries in grey from the
CEEAC shapefile, dashed grey graticule with labels, CAPC-AC / ECCAS / UNDRR
logo in the upper-right corner, four-line title ending with the initialisation
date, horizontal colour bar, 200 dpi on a white background.

This module serves the skill maps of phase P2 and, later, the product maps of
step E8, so that every figure of the chain looks the same.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.offsetbox import AnchoredOffsetbox, OffsetImage

DEFAULT_EXTENT = (4.0, 36.0, -21.0, 25.0)
DPI = 200


@lru_cache(maxsize=4)
def load_boundaries(shapefile: str):
    """
    Country and region outlines of the CEEAC shapefile.

    The admin-1 file is used for the regions; the countries come from the
    ``_adm0`` file when it sits next to it, otherwise from dissolving admin-1.
    """
    import geopandas as gpd

    regions = gpd.read_file(shapefile).to_crs(epsg=4326)
    adm0 = Path(str(shapefile).replace("adm1", "adm0"))
    if adm0.exists() and "adm1" in str(shapefile):
        countries = gpd.read_file(adm0).to_crs(epsg=4326)
    else:
        key = "adm0_a3" if "adm0_a3" in regions.columns else regions.columns[0]
        countries = regions.dissolve(by=key)
    return countries, regions


def add_boundaries(ax, shapefile: str | None):
    """
    Draw regions (grey, thin) then countries (black), as in the reference chain.

    GeoPandas draws the outlines rather than ``ax.add_geometries``: with cartopy
    0.25 the latter raises "cannot create weak reference to 'NoneType' object"
    when the figure is rendered.
    """
    if not shapefile:
        ax.coastlines(linewidth=0.6)
        return
    import cartopy.crs as ccrs

    countries, regions = load_boundaries(str(shapefile))
    regions.boundary.plot(ax=ax, color="gray", linewidth=0.5, transform=ccrs.PlateCarree())
    countries.boundary.plot(ax=ax, color="black", linewidth=1.0, transform=ccrs.PlateCarree())


def add_logo(ax, logo_path: str | None, zoom: float = 0.18, loc: str = "upper right"):
    """Place the institutional logo on a map (ignored when the file is missing)."""
    if not logo_path or not Path(logo_path).is_file():
        return
    image = OffsetImage(plt.imread(logo_path), zoom=zoom)
    ax.add_artist(AnchoredOffsetbox(loc=loc, child=image, pad=0.1, frameon=False, borderpad=0))


def add_figure_logo(fig, logo_path: str | None, width: float = 0.13, height: float = 0.07):
    """
    Logo in the upper-right corner of a **panel figure**, outside the maps.

    On a single map the logo sits inside the axes, as in the reference chain; on
    a panel it would cover the data, so it gets its own small axes.
    """
    if not logo_path or not Path(logo_path).is_file():
        return
    ax = fig.add_axes([1.0 - width - 0.01, 1.0 - height - 0.01, width, height], zorder=5)
    ax.imshow(plt.imread(logo_path))
    ax.axis("off")
    return ax


def setup_ax(ax, extent=DEFAULT_EXTENT, shapefile=None, draw_labels=True,
             left_labels=True, right_labels=False, bottom_labels=True):
    """
    Prepare one map axis: extent, graticule, boundaries.

    In a panel, the labels are only drawn on the outer axes (``left_labels`` on
    the first column, ``bottom_labels`` on the last row), otherwise the
    latitudes of two neighbouring maps overlap.
    """
    import cartopy.crs as ccrs

    ax.set_extent(list(extent), crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    gl = ax.gridlines(draw_labels=draw_labels, linewidth=0.5, color="gray",
                      alpha=0.4, linestyle="--")
    gl.top_labels = False
    gl.left_labels = bool(draw_labels and left_labels)
    gl.right_labels = bool(draw_labels and right_labels)
    gl.bottom_labels = bool(draw_labels and bottom_labels)
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    add_boundaries(ax, shapefile)
    return gl


def build_title(model: str, product: str, period_label: str, init_date=None, system: str = "") -> str:
    """Four-line title: system and model, product, period, initialisation."""
    head = f"{system} {model}".strip()
    lines = [head, product, period_label]
    if init_date is not None:
        lines.append(f"Initialisation : {init_date}")
    return "\n".join(str(x) for x in lines if x)


def _norm(levels, cmap, extend):
    if levels is None:
        return None, plt.get_cmap(cmap)
    cm = plt.get_cmap(cmap)
    if isinstance(cm, ListedColormap) and cm.N < 32:
        return BoundaryNorm(levels, cm.N, extend=extend), cm
    return BoundaryNorm(levels, cm.N, extend=extend), cm


def map_field(field, *, shapefile=None, logo=None, extent=DEFAULT_EXTENT, cmap="viridis",
              levels=None, vmin=None, vmax=None, extend="both", title="", cbar_label="",
              hatch_where=None, output_path=None, figsize=(9, 8)):
    """
    One field on the CEEAC domain, in the house style.

    ``hatch_where`` is an optional boolean field: those cells are hatched, which
    is how the chain shows "no useful skill" on a product map.
    """
    import cartopy.crs as ccrs

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()})
    setup_ax(ax, extent, shapefile)
    norm, cm = _norm(levels, cmap, extend)
    mesh = ax.pcolormesh(field["longitude"], field["latitude"], field.values, cmap=cm, norm=norm,
                         vmin=None if norm else vmin, vmax=None if norm else vmax,
                         shading="nearest", transform=ccrs.PlateCarree())
    if hatch_where is not None:
        ax.contourf(hatch_where["longitude"], hatch_where["latitude"],
                    hatch_where.astype(float).values, levels=[0.5, 1.5], colors="none",
                    hatches=["////"], transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=12, fontweight="bold")
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.75,
                        extend=extend, label=cbar_label)
    cbar.ax.tick_params(labelsize=9)
    add_logo(ax, logo)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return Path(output_path)
    return fig


def map_panel(fields, titles, *, shapefile=None, logo=None, extent=DEFAULT_EXTENT,
              cmap="viridis", levels=None, vmin=None, vmax=None, extend="both", ncols=None,
              cbar_label="", suptitle="", output_path=None, panel_width=3.6):
    """Several fields side by side, one shared colour bar, same style."""
    import cartopy.crs as ccrs

    fields = list(fields)
    n = len(fields)
    ncols = ncols or min(n, 4)
    nrows = int(np.ceil(n / ncols))
    lon_span = extent[1] - extent[0]
    lat_span = extent[3] - extent[2]
    map_height = nrows * panel_width * lat_span / lon_span
    # fixed room for the title (and the logo beside it) above and the colour bar
    # below: letting matplotlib centre the maps in the leftover space left a wide
    # empty band under the title.
    head = 0.75 + 0.22 * suptitle.count("\n")
    foot = 0.85
    figsize = (ncols * panel_width, map_height + head + foot)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    fig.subplots_adjust(top=1 - head / figsize[1], bottom=foot / figsize[1],
                        left=0.05, right=0.98, wspace=0.08, hspace=0.14)
    norm, cm = _norm(levels, cmap, extend)
    mesh = None
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.set_visible(False)
            continue
        setup_ax(ax, extent, shapefile, left_labels=(i % ncols == 0),
                 bottom_labels=(i >= n - ncols))
        f = fields[i]
        mesh = ax.pcolormesh(f["longitude"], f["latitude"], f.values, cmap=cm, norm=norm,
                             vmin=None if norm else vmin, vmax=None if norm else vmax,
                             shading="nearest", transform=ccrs.PlateCarree())
        ax.set_title(titles[i], fontsize=9)
    # colour bar in its own axes: 0.16 inch tall, 0.40 inch above the bottom edge,
    # so it keeps the same look whatever the number of rows.
    cax = fig.add_axes([0.32, 0.40 / figsize[1], 0.36, 0.16 / figsize[1]])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend=extend, label=cbar_label)
    cbar.ax.tick_params(labelsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1 - 0.10 / figsize[1], va="top")
    add_figure_logo(fig, logo)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, facecolor="white")   # the layout is already tight
        plt.close(fig)
        return Path(output_path)
    return fig


#: colour scales of the skill maps (phase P2).
SKILL_STYLES = {
    "pearson": dict(cmap="RdYlBu_r", levels=[-1, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
                    label="corrélation"),
    "spearman": dict(cmap="RdYlBu_r", levels=[-1, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
                     label="corrélation de rang"),
    "acc": dict(cmap="RdYlBu_r", levels=[-1, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
                label="ACC"),
    "rpss": dict(cmap="RdYlGn", levels=[-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5],
                 label="RPSS"),
    "msess": dict(cmap="RdYlGn", levels=[-1, -0.6, -0.4, -0.2, 0, 0.1, 0.2, 0.3, 0.5],
                  label="MSESS"),
    "roc_area": dict(cmap="RdYlGn", levels=[0.2, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 0.85, 1.0],
                     label="aire ROC"),
    "bss": dict(cmap="RdYlGn", levels=[-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5],
                label="BSS"),
    "bias": dict(cmap="BrBG", levels=None, label="biais"),
    "rmse": dict(cmap="YlOrRd", levels=None, label="RMSE"),
}
