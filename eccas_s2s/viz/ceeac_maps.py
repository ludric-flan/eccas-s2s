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
#
# Every skill metric uses the same reading grid: **grey below the no-skill
# value, green above it**, so that a glance separates "the model brings
# something here" from "it does not", whatever the metric. The greys are
# deliberately flat (no gradient of failure), the greens graded, and the
# no-skill value always falls on a class boundary.
NO_SKILL_COLORS = ["#c7c7c7", "#adadad", "#8f8f8f"]
SKILL_COLORS = ["#ffffcc", "#d9f0a3", "#addd8e", "#78c679", "#41ab5d", "#006837"]
_SKILL_CMAP = NO_SKILL_COLORS + SKILL_COLORS

#: symmetric diverging scale for a bias (too wet / too dry, too warm / too cold).
BIAS_CMAP = {"precip": "BrBG", "t2m": "RdBu_r", "tmax": "RdBu_r", "tmin": "RdBu_r"}

SKILL_STYLES = {
    "pearson": dict(colors=_SKILL_CMAP,
                    levels=[-1, -0.4, -0.2, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 1],
                    label="corrélation de Pearson", no_skill=0.0,
                    caption="Corrélation > 0 : le modèle suit le sens des variations observées. "
                            "Au-delà de 0,40 elle est significative à 5 % sur 24 années ; "
                            "en dessous de 0 (gris) le modèle n'apporte rien."),
    "spearman": dict(colors=_SKILL_CMAP,
                     levels=[-1, -0.4, -0.2, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 1],
                     label="corrélation de rang (Spearman)", no_skill=0.0,
                     caption="Même lecture que la corrélation de Pearson, mais sur les rangs : "
                             "insensible aux valeurs extrêmes. > 0 utile, gris = sans skill."),
    "acc": dict(colors=_SKILL_CMAP,
                levels=[-1, -0.4, -0.2, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 1],
                label="ACC (corrélation d'anomalies)", no_skill=0.0,
                caption="ACC > 0 : les anomalies prévues vont dans le sens des anomalies "
                        "observées ; > 0,40 : lien net. Gris = sans skill."),
    "msess": dict(colors=_SKILL_CMAP,
                  levels=[-1, -0.3, -0.1, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 1],
                  label="MSESS", no_skill=0.0,
                  caption="MSESS > 0 : l'erreur quadratique du modèle est plus faible que celle "
                          "de la climatologie. Gris : la moyenne climatologique fait mieux que "
                          "le modèle brut — c'est fréquent tant que le biais n'est pas corrigé."),
    "rpss": dict(colors=_SKILL_CMAP,
                 levels=[-1, -0.3, -0.1, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 1],
                 label="RPSS", no_skill=0.0,
                 caption="RPSS > 0 : les probabilités des trois catégories valent mieux que la "
                         "prévision climatologique (1/3, 1/3, 1/3). Gris = sans skill "
                         "probabiliste."),
    "bss": dict(colors=_SKILL_CMAP,
                levels=[-1, -0.3, -0.1, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 1],
                label="BSS (score de Brier réduit)", no_skill=0.0,
                caption="BSS > 0 : pour cette catégorie, la probabilité prévue bat la "
                        "climatologie. Gris = sans skill."),
    "roc_area": dict(colors=_SKILL_CMAP,
                     levels=[0, 0.3, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1],
                     label="aire sous la courbe ROC", no_skill=0.5,
                     caption="AUC > 0,5 : le modèle sépare les années où la catégorie survient "
                             "de celles où elle ne survient pas ; > 0,70 : discrimination "
                             "utile pour l'alerte. Gris (≤ 0,5) : aucune discrimination."),
    "groc": dict(colors=_SKILL_CMAP,
                 levels=[0, 0.3, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1],
                 label="GROC (ROC généralisé, 3 catégories)", no_skill=0.5,
                 caption="GROC > 0,5 : sur une paire d'années de catégories différentes, le "
                         "modèle classe le plus souvent la bonne année devant l'autre ; "
                         "> 0,70 : discrimination utile. Gris (≤ 0,5) : aucune."),
    "bias": dict(cmap="BrBG", levels=None, label="biais", no_skill=0.0, symmetric=True,
                 caption="Biais du modèle brut : proche de 0 = pas de dérive systématique. "
                         "Positif = modèle trop humide (ou trop chaud), négatif = trop sec "
                         "(ou trop froid). C'est ce que la calibration (P3) corrige en premier."),
    "rmse": dict(cmap="YlOrRd", levels=None, label="RMSE", no_skill=None,
                 caption="Erreur quadratique moyenne, dans l'unité de la variable : plus elle "
                         "est faible, mieux c'est. Pour savoir si elle est bonne, la comparer "
                         "à la climatologie : c'est ce que fait le MSESS."),
    "mae": dict(cmap="YlOrRd", levels=None, label="MAE", no_skill=None,
                caption="Erreur absolue moyenne, dans l'unité de la variable : plus faible = "
                        "meilleur."),
}


#: unit of the variable, appended to the colour bar of the dimensional metrics.
UNITS = {"precip": "mm", "t2m": "°C", "tmax": "°C", "tmin": "°C"}
DIMENSIONAL = ("bias", "rmse", "mae", "rmse_clim")


def style_of(metric: str, variable: str = "precip") -> dict:
    """Colour scale and caption of a metric, adapted to the variable."""
    style = dict(SKILL_STYLES.get(metric, {"cmap": "viridis", "levels": None,
                                           "label": metric, "caption": ""}))
    if metric in DIMENSIONAL and variable in UNITS:
        style["label"] = f"{style.get('label', metric)} ({UNITS[variable]})"
    if metric == "bias":
        style["cmap"] = BIAS_CMAP.get(variable, "BrBG")
        if variable != "precip":
            style["caption"] = style["caption"].replace("trop humide (ou trop chaud)",
                                                        "trop chaud")
            style["caption"] = style["caption"].replace("trop sec\n(ou trop froid)", "trop froid")
    return style


def _discrete(style: dict, field):
    """Colour map and norm of a metric: fixed classes, or a scale fitted to the field."""
    from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm

    if style.get("colors") and style.get("levels"):
        cmap = ListedColormap(style["colors"])
        return cmap, BoundaryNorm(style["levels"], cmap.N, extend="neither"), "neither"
    values = np.asarray(field.values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return plt.get_cmap(style.get("cmap", "viridis")), None, "neither"
    if style.get("symmetric"):
        bound = float(np.nanpercentile(np.abs(finite), 98)) or 1.0
        return plt.get_cmap(style["cmap"]), TwoSlopeNorm(0.0, -bound, bound), "both"
    lo, hi = np.nanpercentile(finite, [2, 98])
    return plt.get_cmap(style["cmap"]), BoundaryNorm(np.linspace(lo, hi, 11),
                                                     256, extend="both"), "both"


def map_score(field, *, metric: str, variable: str = "precip", shapefile=None, logo=None,
              extent=DEFAULT_EXTENT, title="", subtitle="", caption=None, output_path=None,
              map_width: float = 5.4):
    """
    One metric, one period, one map in the CAPC-AC house style.

    Vertical colour bar on the right, latitudes on the left only, and under the
    map the sentence that says what "good" means for this metric — a map of ROC
    areas is unreadable for a user who does not know that 0,5 is the no-skill
    value.

    The figure is sized from the domain so the map fills its axes: a fixed
    figure size leaves a wide empty band on either side of a domain that is
    taller than it is wide, as the CEEAC is.
    """
    import cartopy.crs as ccrs

    style = style_of(metric, variable)
    cmap, norm, extend = _discrete(style, field)

    # inches: the axes box follows the aspect of the domain, the margins hold
    # the title, the caption and the colour bar.
    lon_span, lat_span = extent[1] - extent[0], extent[3] - extent[2]
    ax_w = float(map_width)
    ax_h = ax_w * lat_span / lon_span
    left, right, head, foot = 0.75, 1.70, 1.20, 1.05
    fig_w, fig_h = left + ax_w + right, head + ax_h + foot
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, foot / fig_h, ax_w / fig_w, ax_h / fig_h],
                      projection=ccrs.PlateCarree())
    setup_ax(ax, extent, shapefile, left_labels=True, right_labels=False, bottom_labels=True)
    mesh = ax.pcolormesh(field["longitude"], field["latitude"], field.values, cmap=cmap,
                         norm=norm, shading="nearest", transform=ccrs.PlateCarree())

    centre = (left + ax_w / 2) / fig_w
    if title:
        fig.text(centre, 1 - 0.30 / fig_h, title, ha="center", va="top",
                 fontsize=12.5, fontweight="bold")
    if subtitle:
        fig.text(centre, 1 - 0.68 / fig_h, _wrap(subtitle, 78), ha="center", va="top",
                 fontsize=9.5, color="#333333")

    cax = fig.add_axes([(left + ax_w + 0.22) / fig_w, (foot + 0.05 * ax_h) / fig_h,
                        0.22 / fig_w, 0.90 * ax_h / fig_h])
    cbar = fig.colorbar(mesh, cax=cax, orientation="vertical", extend=extend,
                        ticks=style.get("levels"))
    cbar.set_label(style.get("label", metric), fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    text = style.get("caption", "") if caption is None else caption
    if text:
        fig.text(centre, (foot - 0.40) / fig_h, _wrap(text, 92), ha="center", va="top",
                 fontsize=8.8, style="italic", color="#333333")
    add_figure_logo(fig, logo, width=0.9 / fig_w, height=0.5 / fig_h)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, facecolor="white")
        plt.close(fig)
        return Path(output_path)
    return fig


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width))
