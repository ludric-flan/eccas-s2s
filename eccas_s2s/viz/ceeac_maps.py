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

from eccas_s2s.viz.offscreen import offscreen_when_saved

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


#: drawing order of the region and country outlines: above every painted layer
BOUNDARY_ZORDER = 3.0



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
    # the outlines sit above everything painted on the map — product, categories
    # and the dry mask (zorder 1.4 to 2.0): a reader must still find the region
    # under a grey dry cell (the dry mask of the tercile maps hid them until
    # 20/11/2026)
    regions.boundary.plot(ax=ax, color="gray", linewidth=0.5, transform=ccrs.PlateCarree(),
                          zorder=BOUNDARY_ZORDER)
    countries.boundary.plot(ax=ax, color="black", linewidth=1.0, transform=ccrs.PlateCarree(),
                            zorder=BOUNDARY_ZORDER + 0.1)


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


@offscreen_when_saved
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


@offscreen_when_saved
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
                    title="Corrélation de Pearson", label="corrélation de Pearson", no_skill=0.0,
                    caption="Corrélation > 0 : le modèle suit le sens des variations observées ; "
                            "au-delà de 0,40 le lien est significatif à 5 % sur 24 années. "
                            "Une corrélation fortement négative (proche de −1) est tout aussi "
                            "informative : le signal existe mais son signe est inversé, et une "
                            "calibration (P3) peut le récupérer. Seules les valeurs proches de "
                            "zéro traduisent une absence de lien."),
    "spearman": dict(colors=_SKILL_CMAP,
                     levels=[-1, -0.4, -0.2, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 1],
                     title="Corrélation de rang (Spearman)", label="corrélation de rang (Spearman)", no_skill=0.0,
                     caption="Même lecture que la corrélation de Pearson, mais sur les rangs : "
                             "insensible aux valeurs extrêmes. Les valeurs fortement négatives "
                             "sont informatives (signal de signe inversé) ; c'est autour de zéro "
                             "qu'il n'y a rien à exploiter."),
    "acc": dict(colors=_SKILL_CMAP,
                levels=[-1, -0.4, -0.2, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 1],
                title="ACC — corrélation d'anomalies", label="ACC (corrélation d'anomalies)", no_skill=0.0,
                caption="ACC > 0 : les anomalies prévues vont dans le sens des anomalies "
                        "observées ; > 0,40 : lien net. Une ACC fortement négative signale un "
                        "signal systématiquement inversé, exploitable après calibration ; "
                        "l'absence de skill, c'est autour de zéro."),
    "msess": dict(colors=_SKILL_CMAP,
                  levels=[-1, -0.3, -0.1, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 1],
                  title="MSESS — erreur quadratique réduite", label="MSESS", no_skill=0.0,
                  caption="MSESS > 0 : l'erreur quadratique du modèle est plus faible que celle "
                          "de la climatologie. Gris : la moyenne climatologique fait mieux que "
                          "le modèle brut — c'est fréquent tant que le biais n'est pas corrigé."),
    "rpss": dict(colors=_SKILL_CMAP,
                 levels=[-1, -0.3, -0.1, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 1],
                 title="RPSS — score de probabilité par rangs réduit", label="RPSS", no_skill=0.0,
                 caption="RPSS > 0 : les probabilités des trois catégories valent mieux que la "
                         "prévision climatologique (1/3, 1/3, 1/3). Attention : un RPSS négatif "
                         "ne veut pas dire que le modèle est inutile — il est le plus souvent "
                         "surconfiant, défaut que la calibration (P3) corrige, et il peut garder "
                         "un pouvoir de discrimination (voir les cartes ROC et GROC). Sur 24 "
                         "années, un écart de quelques centièmes n'est pas significatif."),
    "bss": dict(colors=_SKILL_CMAP,
                levels=[-1, -0.3, -0.1, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 1],
                title="BSS — score de Brier réduit", label="BSS (score de Brier réduit)", no_skill=0.0,
                caption="BSS > 0 : pour cette catégorie, la probabilité prévue bat la "
                        "climatologie. Comme pour le RPSS, un BSS négatif traduit d'abord une "
                        "surconfiance des probabilités brutes, corrigible par calibration, et "
                        "non l'absence de tout signal : la décomposition (fiabilité, résolution) "
                        "et l'aire ROC disent lequel des deux manque."),
    "roc_area": dict(colors=_SKILL_CMAP,
                     levels=[0, 0.3, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1],
                     title="Aire sous la courbe ROC (AUC)", label="aire sous la courbe ROC", no_skill=0.5,
                     caption="AUC > 0,5 : le modèle sépare les années où la catégorie survient "
                             "de celles où elle ne survient pas ; > 0,70 : discrimination "
                             "utile pour l'alerte. Gris (≤ 0,5) : aucune discrimination."),
    "groc": dict(colors=_SKILL_CMAP,
                 levels=[0, 0.3, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1],
                 title="GROC — ROC généralisé", label="GROC (ROC généralisé, 3 catégories)", no_skill=0.5,
                 caption="GROC > 0,5 : sur une paire d'années de catégories différentes, le "
                         "modèle classe le plus souvent la bonne année devant l'autre ; "
                         "> 0,70 : discrimination utile. Gris (≤ 0,5) : aucune."),
    "bias": dict(cmap="BrBG", levels=None, title="Biais moyen", label="biais", no_skill=0.0, symmetric=True,
                 caption="Biais du modèle brut : proche de 0 = pas de dérive systématique. "
                         "Positif = modèle trop humide (ou trop chaud), négatif = trop sec "
                         "(ou trop froid). C'est ce que la calibration (P3) corrige en premier."),
    "rmse": dict(cmap="YlOrRd", levels=None, title="RMSE — erreur quadratique moyenne", label="RMSE", no_skill=None,
                 caption="Erreur quadratique moyenne, dans l'unité de la variable : plus elle "
                         "est faible, mieux c'est. Pour savoir si elle est bonne, la comparer "
                         "à la climatologie : c'est ce que fait le MSESS."),
    "mae": dict(cmap="YlOrRd", levels=None, title="MAE — erreur absolue moyenne", label="MAE", no_skill=None,
                caption="Erreur absolue moyenne, dans l'unité de la variable : plus faible = "
                        "meilleur."),
}


#: unit of the variable, appended to the colour bar of the dimensional metrics.
UNITS = {"precip": "mm", "t2m": "°C", "tmax": "°C", "tmin": "°C"}
DIMENSIONAL = ("bias", "rmse", "mae", "rmse_clim")


#: short name of each metric, for the title of a figure (the full wording stays
#: on the colour bar and in the caption, where there is room for it).
METRIC_SHORT = {"pearson": "Pearson", "spearman": "Spearman", "acc": "ACC",
                "msess": "MSESS", "rpss": "RPSS", "bss": "BSS",
                "roc_area": "AUC (ROC)", "groc": "GROC", "bias": "Biais",
                "rmse": "RMSE", "mae": "MAE"}


def metric_title(metric: str) -> str:
    """Short name of a metric, for the title of a figure."""
    return METRIC_SHORT.get(metric, metric)


#: how wide a gain is worth drawing, by metric: a hundredth of RPSS is noise on
#: 24 years, a whole point of MSESS is not
GAIN_LEVELS = {
    "rpss": [-0.20, -0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10, 0.20],
    "bss": [-0.20, -0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10, 0.20],
    "msess": [-1.0, -0.5, -0.2, -0.05, 0, 0.05, 0.2, 0.5, 1.0],
    "acc": [-0.20, -0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10, 0.20],
}


def gain_style(metric: str) -> dict:
    """
    Diverging scale of a **gain**, centred on zero.

    A gain map answers a different question from a score map — "did the
    calibration help, and where?" — so it gets its own reading grid: brown where
    the calibration degrades the score, blue-green where it improves it, white
    where it changes nothing worth seeing. The scale is symmetric, otherwise the
    eye reads an improvement into a simple difference of range.
    """
    base = metric.replace("gain_", "")
    levels = GAIN_LEVELS.get(base, GAIN_LEVELS["rpss"])
    short = METRIC_SHORT.get(base, base.upper())
    # an explicit palette rather than a named diverging map: the middle colours of
    # BrBG are so pale that a small positive gain reads as a loss
    colours = ["#8C510A", "#BF812D", "#DFC27D", "#F6E8C3",      # dégradation
               "#D9F0D3", "#A6DBA0", "#5AAE61", "#1B7837"]      # amélioration
    return {"colors": colours, "levels": levels,
            "label": f"gain de {short} (calibré − brut)",
            "caption": ("Vert : la calibration améliore le score en cette maille ; brun : elle "
                        "le dégrade ; blanc : différence négligeable. La référence est la "
                        "prévision brute interpolée sur la même grille, si bien que l'écart "
                        "ne vient que de la calibration.")}


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


def _nearest_filled(field):
    """
    Fill the missing cells with their nearest neighbour, for smooth shading only.

    Gouraud shading interpolates the colour between cell **centres**, so a
    missing cell blanks the four quads around it: on a masked domain the map
    would lose a fringe along the whole boundary. Filling first and clipping to
    the shapefile afterwards keeps the edge exactly where the mask puts it,
    while the inside is drawn smoothly. Nothing outside the clip is ever shown,
    so the filled values are never read.
    """
    import numpy as np
    from scipy import ndimage

    values = np.asarray(field.values, dtype=float)
    holes = ~np.isfinite(values)
    if not holes.any() or holes.all():
        return values
    _, index = ndimage.distance_transform_edt(holes, return_indices=True)
    return values[tuple(index)]


def clip_to_shapefile(artist, ax, shapefile):
    """
    Clip a mesh to the union of the shapefile polygons.

    This is what lets the map be smoothed inside and stay sharp at the border:
    the shading spills over the edge, the clip cuts it back to the CEEAC.
    """
    if not shapefile:
        return
    import cartopy.crs as ccrs
    import geopandas as gpd
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    shapes = gpd.read_file(str(shapefile)).to_crs(epsg=4326)
    polygons = []
    for geom in shapes.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        polygons += [list(part.exterior.coords) for part in parts]
    if not polygons:
        return
    vertices, codes = [], []
    for ring in polygons:
        vertices += ring
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY]
    patch = PathPatch(MplPath(vertices, codes), transform=ccrs.PlateCarree()._as_mpl_transform(ax),
                      facecolor="none", edgecolor="none")
    ax.add_patch(patch)
    # a mesh clips itself; a set of filled contours is a container of collections
    # in the matplotlib versions the chain runs on, and each one is clipped
    if hasattr(artist, "set_clip_path"):
        artist.set_clip_path(patch)
    else:
        for collection in getattr(artist, "collections", []):
            collection.set_clip_path(patch)


def _cell_edges(centres: np.ndarray) -> np.ndarray:
    """Cell boundaries of a regular axis, from its centres."""
    centres = np.asarray(centres, dtype=float)
    if centres.size == 1:
        return np.array([centres[0] - 0.5, centres[0] + 0.5])
    middles = 0.5 * (centres[:-1] + centres[1:])
    return np.concatenate([[centres[0] - (middles[0] - centres[0])], middles,
                           [centres[-1] + (centres[-1] - middles[-1])]])


def cells_path(lons, lats, mask: np.ndarray):
    """
    Compound path covering exactly the cells where ``mask`` is true.

    Used to clip a smoothed mesh to the cells that belong to one category: the
    shading is continuous **inside** the patch, and the patch stops exactly on
    the cell edges, so no colour ever crosses into a neighbouring category.
    """
    from matplotlib.path import Path as MplPath

    x = _cell_edges(np.asarray(lons))
    y = _cell_edges(np.asarray(lats))
    verts, codes = [], []
    for j, i in zip(*np.nonzero(mask)):
        x0, x1, y0, y1 = x[i], x[i + 1], y[j], y[j + 1]
        verts += [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        codes += [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO,
                  MplPath.CLOSEPOLY]
    return MplPath(verts, codes) if verts else None


def _mono_fontsize(text: str, width_inches: float, base: float = 8.4,
                   floor: float = 5.2) -> float:
    """
    Largest monospaced size at which the longest line still fits the figure.

    The caption of a skill-weighted map holds one column per model, and the
    columns must stay under the names they belong to. Wrapping would break that,
    so the font shrinks instead — down to a floor below which nothing is
    readable anyway.
    """
    longest = max((len(line) for line in text.splitlines()), default=0)
    if longest == 0:
        return base
    # a monospaced glyph is about 0.6 em wide; 72 points to the inch
    return max(floor, min(base, width_inches * 72.0 / (0.6 * longest)))


def _layout(extent, title: str, subtitle: str, caption: str, map_width: float,
            shapefile=None, wrap_caption: bool = True) -> dict:
    """
    Figure and axes of a single map, sized from the domain and the texts.

    Shared by the score maps and the product maps so that a skill map and the
    product it describes come out of the chain the same size, with the same
    margins and the same place for the title, the colour bar and the caption. A
    fixed figure size cannot do that: the CEEAC is taller than it is wide, and a
    three-line title pushed the subtitle over the map frame.
    """
    import cartopy.crs as ccrs

    lon_span, lat_span = extent[1] - extent[0], extent[3] - extent[2]
    ax_w = float(map_width)
    ax_h = ax_w * lat_span / lon_span
    # the header lines are long by design (product, forecast, weighting): they
    # are wrapped wide enough not to break a product name in two
    title_text = _wrap(title, 70) if title else ""
    sub_text = _wrap(subtitle, 62) if subtitle else ""
    # a caption whose columns are aligned by hand must not be re-wrapped
    caption_text = (_wrap(caption, 92) if wrap_caption else caption) if caption else ""
    n_title = title_text.count("\n") + 1 if title_text else 0
    n_sub = sub_text.count("\n") + 1 if sub_text else 0
    n_caption = caption_text.count("\n") + 1 if caption_text else 0
    left, right = 0.75, 1.70
    foot = 0.62 + 0.17 * n_caption
    head = 0.22 + 0.27 * n_title + 0.26 * n_sub + 0.16
    fig_w, fig_h = left + ax_w + right, head + ax_h + foot
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, foot / fig_h, ax_w / fig_w, ax_h / fig_h],
                      projection=ccrs.PlateCarree())
    setup_ax(ax, extent, shapefile, left_labels=True, right_labels=False,
             bottom_labels=True)
    return {"fig": fig, "ax": ax, "fig_w": fig_w, "fig_h": fig_h, "left": left,
            "ax_w": ax_w, "ax_h": ax_h, "foot": foot, "title": title_text,
            "subtitle": sub_text, "caption": caption_text, "n_title": n_title,
            "centre": (left + ax_w / 2) / fig_w}


@offscreen_when_saved
def map_score(field, *, metric: str, variable: str = "precip", shapefile=None, logo=None,
              extent=DEFAULT_EXTENT, title="", subtitle="", caption=None, output_path=None,
              map_width: float = 5.4, shading: str = "gouraud", style=None):
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

    style = style_of(metric, variable) if style is None else style
    cmap, norm, extend = _discrete(style, field)

    caption_text = style.get("caption", "") if caption is None else caption
    box = _layout(extent, title, subtitle, caption_text, map_width, shapefile)
    fig, ax = box["fig"], box["ax"]
    fig_w, fig_h, left, ax_w, ax_h, foot = (box["fig_w"], box["fig_h"], box["left"],
                                            box["ax_w"], box["ax_h"], box["foot"])
    title_text, sub_text, caption_text = box["title"], box["subtitle"], box["caption"]
    n_title = box["n_title"]
    # smoothed across the pixels (Gouraud): a score varies continuously in space,
    # and a blocky map reads as if each cell were an independent measurement
    values = _nearest_filled(field) if shading == "gouraud" else field.values
    mesh = ax.pcolormesh(field["longitude"], field["latitude"], values, cmap=cmap,
                         norm=norm, shading=shading, transform=ccrs.PlateCarree())
    if shading == "gouraud":
        clip_to_shapefile(mesh, ax, shapefile)

    centre = (left + ax_w / 2) / fig_w
    if title_text:
        fig.text(centre, 1 - 0.18 / fig_h, title_text, ha="center", va="top",
                 fontsize=12.5, fontweight="bold", linespacing=1.32)
    if sub_text:
        fig.text(centre, 1 - (0.26 + 0.27 * n_title) / fig_h, sub_text,
                 ha="center", va="top", fontsize=11.5, color="#333333", linespacing=1.32)

    cax = fig.add_axes([(left + ax_w + 0.22) / fig_w, (foot + 0.05 * ax_h) / fig_h,
                        0.22 / fig_w, 0.90 * ax_h / fig_h])
    cbar = fig.colorbar(mesh, cax=cax, orientation="vertical", extend=extend,
                        ticks=style.get("levels"))
    cbar.set_label(style.get("label", metric), fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    if caption_text:
        fig.text(centre, (foot - 0.30) / fig_h, caption_text, ha="center", va="top",
                 fontsize=8.8, style="italic", color="#333333")
    # logo inside the map frame, upper-right corner, as in the reference chain
    add_logo(ax, logo, zoom=0.27, loc="upper right")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, facecolor="white")
        plt.close(fig)
        return Path(output_path)
    return fig


def _wrap(text: str, width: int) -> str:
    """Wrap to ``width``, keeping the line breaks the caller asked for."""
    import textwrap

    return "\n".join("\n".join(textwrap.wrap(line, width)) or line
                     for line in str(text).split("\n"))


# --------------------------------------------------------------- produits RCC
#: palettes de l'annexe du Draft Framework : six classes de probabilité, bornes
#: 40-50, 50-60, 60-70, 70-80, 80-90 et **90-100 %**. Une couleur vaut un
#: intervalle, jamais une valeur isolée — une probabilité est continue, et un
#: pas de 10 points est ce qu'un utilisateur peut distinguer sur une carte.
RCC_BINS = [40, 50, 60, 70, 80, 90, 100]
RCC_TERCILE_COLORS = {
    "precip": {
        "BN": ["#FFB52E", "#FF9800", "#FF7A00", "#FF5500", "#FF3000", "#FF0000"],
        "NN": ["#E6FFFF", "#BFF9F5", "#8FEFEA", "#5DDFD8", "#2ACCC5", "#00B8B0"],
        "AN": ["#C7F0AA", "#39C00F", "#39AB06", "#157A40", "#0E5C30", "#093F21"],
    },
    "temperature": {
        "BN": ["#E6F2FF", "#B3D9FF", "#66B2FF", "#1A75CF", "#004C99", "#00336B"],
        "NN": ["#E6FFFF", "#BFF9F5", "#8FEFEA", "#5DDFD8", "#2ACCC5", "#00B8B0"],
        "AN": ["#FFE400", "#FFCC00", "#FF8800", "#FF6600", "#FF0000", "#B20000"],
    },
}
#: wording of the bulletins, and the meaning of each category for rainfall
RCC_CATEGORY_LABEL = {"BN": "Below Normal", "NN": "Near Normal", "AN": "Above Normal"}
#: no category reaches the first class: nothing to say, and the map says so
NO_DOMINANT_COLOR = "#FFFFFF"
DRY_COLOR = "#C0C0C0"


def _ramp(name: str, n: int = 6, lo: float = 0.22, hi: float = 0.95) -> list:
    """Discrete ramp of ``n`` colours taken from a matplotlib colormap."""
    from matplotlib.colors import to_hex

    return [to_hex(plt.get_cmap(name)(x)) for x in np.linspace(lo, hi, n)]


#: SPI classes, with the colours of the reference chain: dry in oranges, normal
#: in greens, wet in blues — the same dominant-class reading as the terciles
SPI_CLASS_COLORS = {"BN": _ramp("Oranges"), "NN": _ramp("Greens"), "AN": _ramp("Blues")}
SPI_CLASS_LABEL = {"BN": "SPI sec (< −1)", "NN": "SPI normal", "AN": "SPI humide (> +1)"}


@offscreen_when_saved
def map_tercile_forecast(prob, *, variable="precip", shapefile=None, logo=None,
                         extent=DEFAULT_EXTENT, title="", subtitle="", output_path=None,
                         dry_mask=None, map_width: float = 5.4, category_dim="category",
                         palette=None, category_labels=None, caption=None,
                         caption_mono: bool = False):
    """
    The tercile forecast map of the bulletins: one dominant category per cell.

    Each cell takes the category with the highest probability and its colour from
    that category's ramp, cut at the classes of the Draft annex. A cell whose
    strongest category stays under 40 % is left **white**: with three categories
    the climatological value is 33 %, so below 40 % the forecast says nothing a
    user could act on, and colouring it would suggest otherwise. The dry mask is
    drawn in sand, as in the reference chain — a probability of rainfall category
    means little where the season brings less than a millimetre.

    The three ramps are shown as three separate colour bars, because they are
    three different quantities: a reader must never compare an orange 60 % with a
    green 60 % on one scale.
    """
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch, PathPatch

    palette = palette or RCC_TERCILE_COLORS["precip" if variable == "precip" else "temperature"]
    labels = category_labels or RCC_CATEGORY_LABEL
    cats = [str(c) for c in prob[category_dim].values]
    values = prob.transpose(category_dim, "latitude", "longitude").values * 100.0
    dominant = np.nanargmax(np.where(np.isfinite(values), values, -1), axis=0)
    strongest = np.nanmax(values, axis=0)
    finite = np.isfinite(strongest)

    lon_span, lat_span = extent[1] - extent[0], extent[3] - extent[2]
    ax_w = float(map_width)
    ax_h = ax_w * lat_span / lon_span
    title_text = _wrap(title, 70) if title else ""
    sub_text = _wrap(subtitle, 62) if subtitle else ""
    caption_text = (caption if caption_mono else _wrap(caption, 92)) if caption else ""
    n_title = title_text.count("\n") + 1 if title_text else 0
    n_sub = sub_text.count("\n") + 1 if sub_text else 0
    n_caption = caption_text.count("\n") + 1 if caption_text else 0
    left, right = 0.75, 1.95
    foot = 0.62 + 0.19 * n_caption
    head = 0.22 + 0.27 * n_title + 0.26 * n_sub + 0.16
    fig_w, fig_h = left + ax_w + right, head + ax_h + foot
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, foot / fig_h, ax_w / fig_w, ax_h / fig_h],
                      projection=ccrs.PlateCarree())
    setup_ax(ax, extent, shapefile, left_labels=True, right_labels=False, bottom_labels=True)

    weak = finite & (strongest < RCC_BINS[0])
    ax.pcolormesh(prob["longitude"], prob["latitude"], np.where(weak, 1.0, np.nan),
                  cmap=ListedColormap([NO_DOMINANT_COLOR]), shading="nearest",
                  transform=ccrs.PlateCarree())

    meshes = {}
    for index, cat in enumerate(cats):
        colours = palette.get(cat, palette["NN"])
        cmap = ListedColormap(colours)
        cmap.set_over(colours[-1])
        norm = BoundaryNorm(RCC_BINS, cmap.N)
        region = finite & (dominant == index) & (strongest >= RCC_BINS[0])
        field = np.where(region, strongest, np.nan)
        # smoothed **inside** the category and nowhere else: the values are the
        # ones of the file, the shading only interpolates the colour between two
        # cells that carry the same category, and the mesh is clipped to those
        # cells. A gradient crossing a boundary would paint a colour belonging to
        # no category, which the RCC convention forbids; a plain nearest shading
        # would show the grid.
        clip = cells_path(prob["longitude"].values, prob["latitude"].values, region)
        if clip is None:
            # the category wins nowhere on this period: its bar is still drawn,
            # so that the three scales stay in the same place from one map to
            # the next and a reader never wonders which one is missing
            from matplotlib.cm import ScalarMappable

            meshes[cat] = ScalarMappable(norm=norm, cmap=cmap)
            continue
        values = _nearest_filled(prob.isel({category_dim: index}).copy(data=field))
        meshes[cat] = ax.pcolormesh(prob["longitude"], prob["latitude"], values, cmap=cmap,
                                    norm=norm, shading="gouraud",
                                    transform=ccrs.PlateCarree())
        patch = PathPatch(clip, transform=ccrs.PlateCarree()._as_mpl_transform(ax),
                          facecolor="none", edgecolor="none")
        ax.add_patch(patch)
        meshes[cat].set_clip_path(patch)

    # the dry cells are painted last, over the categories: a category drawn
    # there would be read as a forecast, and the mask exists to say there is none
    if dry_mask is not None:
        dry = np.asarray(dry_mask.values if hasattr(dry_mask, "values") else dry_mask)
        sand = ax.pcolormesh(prob["longitude"], prob["latitude"], np.where(dry, 1.0, np.nan),
                             cmap=ListedColormap([DRY_COLOR]), shading="nearest",
                             transform=ccrs.PlateCarree(), zorder=2.0)
        clip_to_shapefile(sand, ax, shapefile)

    # the three bars share the **whole** height of the map frame: a bar hanging
    # below the frame reads as if it belonged to something else
    order = [c for c in ("AN", "NN", "BN") if c in meshes]
    bar_h = 0.28 * ax_h
    gap = (ax_h - len(order) * bar_h) / max(len(order) - 1, 1)
    for row, cat in enumerate(order):
        cax = fig.add_axes([(left + ax_w + 0.30) / fig_w,
                            (foot + ax_h - (row + 1) * bar_h - row * gap) / fig_h,
                            0.20 / fig_w, bar_h / fig_h])
        cbar = fig.colorbar(meshes[cat], cax=cax, orientation="vertical", ticks=RCC_BINS)
        cbar.set_label(f"{labels.get(cat, cat)} (%)", fontsize=8.5)
        cbar.ax.tick_params(labelsize=7.5)

    handles = [Patch(facecolor=NO_DOMINANT_COLOR, edgecolor="black",
                     label="No dominant Class")]
    if dry_mask is not None:
        handles.append(Patch(facecolor=DRY_COLOR, edgecolor="black", label="dry mask"))
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, framealpha=0.9)

    centre = (left + ax_w / 2) / fig_w
    if title_text:
        fig.text(centre, 1 - 0.18 / fig_h, title_text, ha="center", va="top",
                 fontsize=12.5, fontweight="bold", linespacing=1.32)
    if sub_text:
        fig.text(centre, 1 - (0.26 + 0.27 * n_title) / fig_h, sub_text, ha="center", va="top",
                 fontsize=12.5, fontweight="bold", color="#000000", linespacing=1.32)
    if caption_text:
        size = (_mono_fontsize(caption_text, fig_w - 0.4) if caption_mono else 9.2)
        fig.text(0.5 if caption_mono else centre, (foot - 0.34) / fig_h, caption_text,
                 ha="center", va="top",
                 fontsize=size, color="#222222",
                 family="monospace" if caption_mono else None, linespacing=1.45)
    if logo:
        add_logo(ax, logo, zoom=0.27, loc="upper right")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return Path(output_path)
    return fig


# =====================================================================
# DISPLAY RULES OF THE PRODUCTS (reference chain: run_forecast_v2)
# =====================================================================
# Each product of the catalogue is drawn with the palette and the levels the
# reference chain uses for it, so that a forecaster reads the same colours in
# the operational bulletin and in this chain. The palettes are transcribed from
# ``s2s_config_v2.py`` (TP_COLORS, PROB_CMAP, SPI_DET_CMAP) and the levels from
# ``s2s_plotting_v2.py``; the tercile map alone departs from it, to follow the
# RCC convention of the Draft Framework annex (:func:`map_tercile_forecast`).
#
# One thing is added to every product: the field is **smoothed across the
# pixels** before it is drawn, as the score and gain maps are. A seasonal
# forecast is a large-scale field, and a blocky 1° map reads as if each cell
# were an independent measurement; the bands are drawn on the filled field and
# clipped back to the CEEAC boundary, so nothing spills outside the domain.

#: anomaly palette: the nine colours of the reference chain, completed with the
#: intermediate shades the twelve-level colour bar needs (decision of 29/09/2026)
TP_COLORS = ["#993300", "#CC6600", "#E6A000", "#FFD500", "#FFEE99", "#FFFFFF",
             "#CCFF66", "#8CE64D", "#2AFF00", "#00B33C", "#009933"]
TP_UNDER, TP_OVER = "#663300", "#006666"
#: anomaly levels per time scale (mm): eleven bands, the middle one neutral white
ANOMALY_LEVELS = {"decade": [-30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30],
                  "month": [-100, -75, -50, -40, -30, -20, 20, 30, 40, 50, 75, 100],
                  "season": [-150, -125, -100, -75, -50, -25, 25, 50, 75, 100, 125, 150]}
#: rainfall totals: the CAPC-AC accumulation palette, white below the first level
CUMUL_COLORS = ["#E4F5E2", "#66C168", "#CDE8F7", "#3C8FE0", "#FFE066",
                "#FFC33D", "#FF9425", "#FA6A1E", "#EE2C24", "#B01118"]
CUMUL_UNDER, CUMUL_OVER = "#FFFFFF", "#5E0A0A"
#: total levels per time scale (mm): below the first one the cell stays white,
#: because a total that small is not a rainfall pattern to read
CUMUL_LEVELS = {"decade": [1, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150],
                "month": [10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500],
                "season": [25, 50, 100, 150, 200, 300, 400, 500, 700, 900, 1100]}
#: share of the annual rainfall carried by the period (%): for a season the
#: pro-rata share is 25 %, so the upper bands mark the places where one season
#: *is* the year
SHARE_LEVELS = [1, *range(5, 95, 5)]
#: deterministic SPI: diverging brown -> white -> green, centred on zero
SPI_DET_COLORS = [(0.00, "#8c2d04"), (0.20, "#cc4c02"), (0.40, "#fec44f"),
                  (0.50, "#ffffff"), (0.60, "#a1d99b"), (0.80, "#31a354"),
                  (1.00, "#00441b")]
SPI_LEVELS = [-3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3]
#: exceedance probabilities: the continuous scale of the reference chain
PROB_CMAP_COLORS = [(0.00, "#ffffff"), (0.10, "#d9f0d3"), (0.30, "#1a9641"),
                    (0.50, "#ffff00"), (0.70, "#fdae61"), (1.00, "#d7191c")]
#: probability bins of the single-category maps: cut every 5 % from 35 % to
#: 90 %, the last band open upwards. Below 35 % nothing is drawn — with three
#: categories the climatological probability is 33 %, with five quintiles 20 %,
#: so a probability that low says only "no signal", and a colour there would
#: invite a decision the forecast does not support (decision of 30/09/2026).
PROBABILITY_LEVELS = list(range(35, 95, 5))
#: below the first level the cell stays white, never pale-coloured
PROBABILITY_UNDER = "#FFFFFF"
#: colormap of each probability product, as ``_TERCILE_VAR_CFG`` assigns them
PRODUCT_CMAP = {"categorie_BN": "YlOrBr", "categorie_NN": "YlGn", "categorie_AN": "YlGnBu",
                "quintile_bas": "OrRd", "quintile_haut": "PuBu",
                "depassement_median": "RdYlGn"}
SPI_LABEL = {"decade": "SPI (dekad)", "month": "SPI-1", "season": "SPI-3"}


def _interpolated(colors, n: int) -> list:
    """
    ``n`` shades built **inside** each colour family of a palette.

    The house palette is five families of two — a pale and a saturated green,
    then blue, yellow-amber, orange, red. A colour bar cut every 5 % needs
    eighteen bands, and interpolating across the whole palette would have to
    travel from blue to yellow: the shades in between come out khaki, and a
    reader sees a colour that means nothing in the legend. So the bands are
    shared out between the families and each family is interpolated on its own,
    from its pale anchor to its saturated one. No band ever sits between two
    hues.
    """
    from matplotlib.colors import to_hex

    families = [colors[i:i + 2] for i in range(0, len(colors), 2)]
    counts = [n // len(families)] * len(families)
    for i in range(n - sum(counts)):          # the remainder goes to the low end
        counts[i] += 1
    out = []
    for pair, count in zip(families, counts):
        if count <= 0:
            continue
        ramp = _segmented(pair, "family")
        out += [to_hex(ramp(k / max(count - 1, 1))) for k in range(count)]
    return out


def _segmented(colors, name: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, colors)


def product_style(product: str, scale: str, variable: str = "precip", label: str = "") -> dict:
    """
    Palette, levels and colour-bar label of one product.

    ``levels`` and ``colors`` describe discrete bands (the anomaly, the SPI, the
    probability bins); a product with a ``cmap`` and no ``colors`` is drawn on a
    continuous scale, which is what the reference chain does for the exceedance
    probabilities — 101 levels, a tick every 5 %.
    """
    scale = scale if scale in ANOMALY_LEVELS else "season"
    if product == "anomalie":
        return {"levels": ANOMALY_LEVELS[scale], "colors": TP_COLORS, "extend": "both",
                "under": TP_UNDER, "over": TP_OVER, "label": "Rainfall anomaly (mm)"}
    if product == "cumul":
        return {"levels": CUMUL_LEVELS[scale], "colors": CUMUL_COLORS, "extend": "both",
                "under": CUMUL_UNDER, "over": CUMUL_OVER, "label": "Rainfall total (mm)"}
    if product == "spi":
        return {"levels": SPI_LEVELS, "cmap": _segmented(SPI_DET_COLORS, "spi_det"),
                "extend": "both", "ticks": SPI_LEVELS,
                "label": f"{SPI_LABEL.get(scale, 'SPI')} forecast"}
    if product.startswith("depassement_") and product != "depassement_median":
        threshold = product.replace("depassement_", "")
        return {"levels": list(np.arange(0, 101, 1)), "cmap": _segmented(PROB_CMAP_COLORS, "prob"),
                "ticks": list(range(0, 101, 5)), "extend": "neither", "percent": True,
                "label": f"Probability of exceeding {threshold} (%)"}
    if product == "seasonal_share":
        # cut every 5 % up to 90 %, the last band open upwards: above that a
        # single period carries almost the whole year, and the arrow says so
        # better than a band nobody reaches
        return {"levels": SHARE_LEVELS,
                "colors": _interpolated(CUMUL_COLORS, len(SHARE_LEVELS) - 1),
                "extend": "both", "under": "#FFFFFF", "over": CUMUL_OVER,
                "ticks": SHARE_LEVELS, "label": "Share of the annual rainfall (%)"}
    if product in PRODUCT_CMAP:
        return {"levels": PROBABILITY_LEVELS,
                "colors": _ramp(PRODUCT_CMAP[product], len(PROBABILITY_LEVELS) - 1,
                                lo=0.18, hi=0.96),
                "under": PROBABILITY_UNDER, "extend": "max", "percent": True,
                "ticks": PROBABILITY_LEVELS, "label": f"{label or product} (%)"}
    return {"cmap": "viridis", "label": label or product, "extend": "both"}


def _product_norm(style: dict):
    """Colormap and norm of a product style, discrete bands or continuous scale."""
    levels = style.get("levels")
    if style.get("colors"):
        colors = style["colors"]
        if len(colors) != len(levels) - 1:
            raise ValueError(f"{len(levels)} seuils demandent {len(levels) - 1} couleurs, "
                             f"{len(colors)} fournies")
        cmap = ListedColormap(colors)
        # the extremes are colours of their own: below the first level a total is
        # too small to read as a pattern (white), above the last one the band is
        # open-ended and says "more than that"
        cmap.set_over(style.get("over", colors[-1]))
        cmap.set_under(style.get("under", colors[0]))
        return cmap, BoundaryNorm(levels, cmap.N, extend="neither")
    cmap = style.get("cmap", "viridis")
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    if levels is None:
        return cmap, None
    return cmap, BoundaryNorm(levels, 256, extend="neither")


@offscreen_when_saved
def map_product(field, *, product: str, scale: str, variable: str = "precip", dry=None,
                shapefile=None, logo=None, extent=DEFAULT_EXTENT, title="", subtitle="",
                caption=None, output_path=None, map_width: float = 5.4, style=None,
                percent: bool | None = None, caption_mono: bool = False):
    """
    One product, one period, one map with the display rules of the reference chain.

    The dry cells are painted first, in the RCC sand grey, and the product is
    drawn over them: a tercile or a percentile says nothing where the season
    brings almost no rain, and leaving the colour there would invite a decision
    the forecast cannot support. The field itself is smoothed across the pixels
    and clipped to the CEEAC boundary, exactly as the score maps are.
    """
    import cartopy.crs as ccrs

    style = product_style(product, scale, variable) if style is None else style
    cmap, norm = _product_norm(style)
    percent = style.get("percent", False) if percent is None else percent
    values = field.values * 100.0 if percent else field.values
    values = np.where(np.asarray(dry.values, dtype=bool), np.nan, values) if dry is not None \
        else values
    plotted = field.copy(data=values)

    caption_text = style.get("caption", "") if caption is None else caption
    box = _layout(extent, title, subtitle, caption_text, map_width, shapefile,
                  wrap_caption=not caption_mono)
    fig, ax = box["fig"], box["ax"]
    fig_w, fig_h = box["fig_w"], box["fig_h"]

    # filled contours, not a resampling: the values are never changed, matplotlib
    # only places the band boundaries between neighbouring cell centres. The
    # missing cells are filled with their nearest neighbour beforehand so the
    # bands reach the border instead of stopping one cell short, and everything
    # is clipped back to the CEEAC afterwards, so nothing invented is ever shown
    mesh = ax.contourf(field["longitude"], field["latitude"], _nearest_filled(plotted),
                       levels=style.get("levels"), cmap=cmap, norm=norm,
                       extend=style.get("extend", "both"), transform=ccrs.PlateCarree(),
                       zorder=1.4)
    clip_to_shapefile(mesh, ax, shapefile)
    # the dry cells are painted over the product, cell by cell: there the
    # forecast says nothing a user could act on, and a colour there would
    # suggest otherwise
    if dry is not None:
        painted = np.where(np.asarray(dry.fillna(False).values, dtype=bool), 1.0, np.nan)
        sand = ax.pcolormesh(field["longitude"], field["latitude"], painted,
                             cmap=ListedColormap([DRY_COLOR]), shading="nearest",
                             transform=ccrs.PlateCarree(), zorder=1.8)
        clip_to_shapefile(sand, ax, shapefile)

    # the three lines of the bulletin header, all in bold: what is shown, which
    # forecast and which weighting, and the initialisation the whole map hangs on
    centre = box["centre"]
    if box["title"]:
        fig.text(centre, 1 - 0.18 / fig_h, box["title"], ha="center", va="top",
                 fontsize=12.5, fontweight="bold", linespacing=1.32)
    if box["subtitle"]:
        fig.text(centre, 1 - (0.26 + 0.27 * box["n_title"]) / fig_h, box["subtitle"],
                 ha="center", va="top", fontsize=12.5, fontweight="bold", color="#000000",
                 linespacing=1.32)

    cax = fig.add_axes([(box["left"] + box["ax_w"] + 0.22) / fig_w,
                        (box["foot"] + 0.05 * box["ax_h"]) / fig_h,
                        0.22 / fig_w, 0.90 * box["ax_h"] / fig_h])
    cbar = fig.colorbar(mesh, cax=cax, orientation="vertical",
                        ticks=style.get("ticks", style.get("levels")))
    cbar.set_label(style.get("label", product), fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    if dry is not None and bool(np.asarray(dry.fillna(False).values).any()):
        from matplotlib.patches import Patch

        ax.legend(handles=[Patch(facecolor=DRY_COLOR, edgecolor="black", label="dry mask")],
                  loc="lower left", fontsize=8, framealpha=0.9)
    # under the map, the models the combination was built from: a multi-model map
    # without the list of its members cannot be checked by whoever reads it
    if box["caption"]:
        size = (_mono_fontsize(box["caption"], fig_w - 0.4) if caption_mono else 9.2)
        fig.text(0.5 if caption_mono else centre, (box["foot"] - 0.30) / fig_h, box["caption"],
                 ha="center", va="top",
                 fontsize=size, color="#222222",
                 family="monospace" if caption_mono else None, linespacing=1.45)
    add_logo(ax, logo, zoom=0.27, loc="upper right")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, facecolor="white")
        plt.close(fig)
        return Path(output_path)
    return fig
