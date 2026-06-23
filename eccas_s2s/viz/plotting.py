"""
All visualization functions for the C3S seasonal forecast pipeline.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnchoredOffsetbox

import cartopy.crs as ccrs

from eccas_s2s.config import MAP_EXTENT, TP_COLORS, PROB_CMAP, MODEL_NAMES, OUTPUT_DIR


# ============================================================
# SHARED HELPERS
# ============================================================
def apply_mask(data, mask_zone):
    if mask_zone is not None:
        return np.where(mask_zone, data, np.nan)
    return data


def add_shapefile_to_ax(ax, gdf_countries, gdf_regions):
    for geom in gdf_regions.geometry:
        ax.add_geometries([geom], ccrs.PlateCarree(),
                          edgecolor="gray", facecolor="none", linewidth=0.5)
    for geom in gdf_countries.geometry:
        ax.add_geometries([geom], ccrs.PlateCarree(),
                          edgecolor="black", facecolor="none", linewidth=1.0)


def add_logo(ax, logo_path=None, zoom=0.18, loc="upper right"):
    if logo_path is None:
        from eccas_s2s.config import LOGO_PATH
        logo_path = LOGO_PATH
    if os.path.isfile(logo_path):
        logo_img = plt.imread(logo_path)
        imagebox = OffsetImage(logo_img, zoom=zoom)
        ab = AnchoredOffsetbox(loc=loc, child=imagebox, pad=0.1,
                               frameon=False, borderpad=0)
        ax.add_artist(ab)


def setup_map_ax(ax, extent=None, gdf_countries=None, gdf_regions=None, draw_labels=True):
    if extent is None:
        extent = MAP_EXTENT
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    gl = ax.gridlines(draw_labels=draw_labels, linewidth=0.5,
                      color="gray", alpha=0.4, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = True
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}
    if gdf_countries is not None and gdf_regions is not None:
        add_shapefile_to_ax(ax, gdf_countries, gdf_regions)
    return gl


def _panel_figsize(extent, ncols, nrows, panel_w=3.6, extra_h=1.8):
    """
    Figure size matched to the map aspect ratio so the geographic maps fill
    their cells, minimising the blank space between sub-figures.

    panel_w is the target width (inches) of a single map column; the row
    height is derived from the domain aspect ratio (height / width).
    extra_h reserves vertical room for the suptitle and shared colorbar.
    """
    if extent is None:
        extent = MAP_EXTENT
    lon_span = float(extent[1] - extent[0])
    lat_span = float(extent[3] - extent[2])
    aspect   = lat_span / lon_span if lon_span else 1.0
    fig_w = ncols * panel_w
    fig_h = nrows * panel_w * aspect + extra_h
    return (fig_w, fig_h)


def format_period_label(start_str, end_str, period_type):
    start = pd.Timestamp(start_str)
    end   = pd.Timestamp(end_str)
    if period_type == "decade":
        decade_num = 1 if start.day <= 10 else (2 if start.day <= 20 else 3)
        return (f"Décade {decade_num} - {start.strftime('%B %Y')}\n"
                f"({start.strftime('%d %b')} - {end.strftime('%d %b %Y')})")
    elif period_type == "month":
        return (f"{start.strftime('%B %Y')}\n"
                f"({start.strftime('%d %b')} - {end.strftime('%d %b %Y')})")
    elif period_type == "season":
        return (f"Saison {start.strftime('%b')}-{end.strftime('%b %Y')}\n"
                f"({start.strftime('%d %b')} - {end.strftime('%d %b %Y')})")
    elif period_type == "custom":
        return (f"Période personnalisée\n"
                f"({start.strftime('%d %b %Y')} - {end.strftime('%d %b %Y')})")
    return f"{start.strftime('%d %b %Y')} - {end.strftime('%d %b %Y')}"


def get_levels_for_period(period_type):
    return {
        "decade": [-30, -20, -15, -10,  10,  15,  20,  30],
        "month":  [-100, -75, -50, -25,  25,  50,  75, 100],
        "season": [-200, -150, -100, -50,  50, 100, 150, 200],
        "custom": [-150, -100, -75, -50,  50,  75, 100, 150],
    }.get(period_type, [-100, -75, -50, -25, 25, 50, 75, 100])


# ============================================================
# ANOMALY MAPS — ALL MEMBERS
# ============================================================
def plot_anomaly_all_members(
    anom, period_label, period_type, start_str, end_str,
    init_date_str, gdf_countries, gdf_regions,
    mask_zone=None, model_name="ECMWF SEAS51",
    tp_levels=None, tp_colors=None, extent=None,
    output_dir=OUTPUT_DIR, save=True, show=False,
):
    if tp_levels is None:
        tp_levels = get_levels_for_period(period_type)
    if tp_colors is None:
        tp_colors = TP_COLORS
    if extent is None:
        extent = MAP_EXTENT

    if "number" in anom.dims:
        n_members = anom.sizes["number"]
    else:
        n_members = 1
        anom = anom.expand_dims("number")

    ncols = 10
    nrows = int(np.ceil(n_members / ncols))
    fig   = plt.figure(figsize=(20, 2.8 * nrows + 1.5))
    plt.subplots_adjust(hspace=0.25, wspace=0.05)
    fig.suptitle(
        f"C3S {model_name} - Anomalie de précipitation totale\n"
        f"{period_label}\nInit: {init_date_str}",
        fontsize=16, fontweight="bold", y=0.98,
    )

    im = None
    for n in range(n_members):
        ax = fig.add_subplot(nrows, ncols, n + 1, projection=ccrs.PlateCarree())
        ax.set_facecolor("white")
        data_n = apply_mask(anom.isel(number=n).values, mask_zone)
        im = ax.contourf(anom.longitude.values, anom.latitude.values, data_n,
                         levels=tp_levels, colors=tp_colors, extend="both")
        ax.set_title(f"M{n+1}", fontsize=8, fontweight="bold")
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                          color="gray", alpha=0.4, linestyle="--")
        gl.top_labels    = False
        gl.bottom_labels = (n >= n_members - ncols)
        gl.right_labels  = ((n + 1) % ncols == 0) or (n == n_members - 1)
        gl.left_labels   = (n % ncols == 0)
        gl.xlabel_style  = {"size": 7}
        gl.ylabel_style  = {"size": 7}
        add_shapefile_to_ax(ax, gdf_countries, gdf_regions)

    fig.subplots_adjust(bottom=0.06)
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
                        label="Anomalie de précipitation (mm)")
    cbar.ax.tick_params(labelsize=9)

    if save:
        fn = os.path.join(output_dir,
                          f"anomalies_{period_type}_{start_str}_to_{end_str}_members.png")
        fig.savefig(fn, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"    💾 {fn}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# ANOMALY MAPS — ENSEMBLE MEAN
# ============================================================
def plot_anomaly_ensemble_mean(
    anom, period_label, period_type, start_str, end_str,
    init_date_str, gdf_countries, gdf_regions,
    mask_zone=None, model_name="ECMWF SEAS51",
    tp_levels=None, tp_colors=None, extent=None,
    output_dir=OUTPUT_DIR, save=True, show=False,
):
    if tp_levels is None:
        tp_levels = get_levels_for_period(period_type)
    if tp_colors is None:
        tp_colors = TP_COLORS
    if extent is None:
        extent = MAP_EXTENT

    ens_mean = anom.mean("number") if "number" in anom.dims else anom
    data     = apply_mask(ens_mean.values, mask_zone)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    setup_map_ax(ax, extent, gdf_countries, gdf_regions)
    im = ax.contourf(ens_mean.longitude.values, ens_mean.latitude.values, data,
                     levels=tp_levels, colors=tp_colors, extend="both")
    ax.set_title(
        f"C3S {model_name} - Moyenne d'ensemble\n"
        f"Anomalie précipitation\n{period_label}\nInit: {init_date_str}",
        fontsize=14, fontweight="bold",
    )
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                        pad=0.05, shrink=0.7, label="Anomalie (mm)")
    cbar.ax.tick_params(labelsize=10)
    add_logo(ax)

    if save:
        fn = os.path.join(output_dir,
                          f"anomalies_{period_type}_{start_str}_to_{end_str}_ensmean.png")
        fig.savefig(fn, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"    💾 {fn}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_single_anomaly_map(
    data, lons, lats, period_label, init_date_str,
    gdf_countries, gdf_regions, mask_zone=None,
    model_name="Multi-Model Mean", title="Ensemble Mean Anomaly",
    tp_levels=None, tp_colors=None, extent=None, output_path=None,
):
    """Render a single anomaly map from a 2D numpy array (or DataArray)."""
    if tp_colors is None:
        tp_colors = TP_COLORS
    if extent is None:
        extent = MAP_EXTENT
    if tp_levels is None:
        tp_levels = get_levels_for_period("month")

    data_np = data.values if hasattr(data, "values") else data
    lons_np = lons.values if hasattr(lons, "values") else lons
    lats_np = lats.values if hasattr(lats, "values") else lats
    data_plot = apply_mask(data_np, mask_zone)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    setup_map_ax(ax, extent, gdf_countries, gdf_regions)
    im = ax.contourf(lons_np, lats_np, data_plot,
                     levels=tp_levels, colors=tp_colors, extend="both",
                     transform=ccrs.PlateCarree())
    ax.set_title(
        f"C3S {model_name}\n{title}\n{period_label}\nInit: {init_date_str}",
        fontsize=14, fontweight="bold",
    )
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                        pad=0.05, shrink=0.7, label="Anomalie (mm)")
    cbar.ax.tick_params(labelsize=10)
    add_logo(ax)

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"    💾 {output_path}")
    plt.close(fig)


def plot_all_anomalies(
    products, periods_df, period_type, init_date_str,
    gdf_countries, gdf_regions, model_name="ECMWF SEAS51",
    mask_zone=None, tp_levels=None, tp_colors=None, extent=None,
    output_dir=OUTPUT_DIR, save=True, show=False,
):
    anom      = products["anomaly"]
    n_periods = anom.sizes["period"]
    print(f"\n    {'='*60}")
    print(f"    TRACÉ ANOMALIES - {period_type.upper()} ({n_periods} périodes × 2 cartes)")
    print(f"    {'='*60}")

    for i in range(n_periods):
        anom_i    = anom.isel(period=i)
        p_start   = pd.Timestamp(anom["period_start"].isel(period=i).values)
        p_end     = pd.Timestamp(anom["period_end"].isel(period=i).values)
        start_str = p_start.strftime("%Y-%m-%d")
        end_str   = p_end.strftime("%Y-%m-%d")
        label     = format_period_label(start_str, end_str, period_type)
        print(f"\n    📊 {period_type} : {start_str} → {end_str}")

        plot_anomaly_all_members(
            anom=anom_i, period_label=label, period_type=period_type,
            start_str=start_str, end_str=end_str, init_date_str=init_date_str,
            gdf_countries=gdf_countries, gdf_regions=gdf_regions,
            mask_zone=mask_zone, model_name=model_name,
            tp_levels=tp_levels, tp_colors=tp_colors, extent=extent,
            output_dir=output_dir, save=save, show=show,
        )
        plot_anomaly_ensemble_mean(
            anom=anom_i, period_label=label, period_type=period_type,
            start_str=start_str, end_str=end_str, init_date_str=init_date_str,
            gdf_countries=gdf_countries, gdf_regions=gdf_regions,
            mask_zone=mask_zone, model_name=model_name,
            tp_levels=tp_levels, tp_colors=tp_colors, extent=extent,
            output_dir=output_dir, save=save, show=show,
        )
        plt.close("all")

    print(f"\n    ✅ {n_periods * 2} figures '{period_type}' générées dans {output_dir}")


# ============================================================
# MEMBER DISPERSION
# ============================================================
def plot_member_dispersion(
    anom, period_label, period_type, start_str, end_str,
    init_date_str, gdf_countries, gdf_regions,
    mask_zone=None, model_name="ECMWF SEAS51",
    extent=None, output_dir=OUTPUT_DIR, save=True, show=False,
):
    if extent is None:
        extent = MAP_EXTENT
    if "number" not in anom.dims:
        print("    ⚠ Pas de dimension 'number', dispersion impossible.")
        return

    ens_mean   = anom.mean("number")
    dispersion = anom - ens_mean
    n_members  = anom.sizes["number"]
    vmax       = max(float(np.abs(dispersion).max()) * 0.8, 1.0)
    levels     = np.linspace(-vmax, vmax, 21)

    ncols = 10
    nrows = int(np.ceil(n_members / ncols))
    fig   = plt.figure(figsize=(20, 2.8 * nrows + 1.5))
    plt.subplots_adjust(hspace=0.25, wspace=0.05)
    fig.suptitle(
        f"C3S {model_name} - Dispersion des membres\n"
        f"(Anomalie membre − Anomalie moyenne)\n{period_label}\nInit: {init_date_str}",
        fontsize=15, fontweight="bold", y=0.98,
    )

    im = None
    for n in range(n_members):
        ax = fig.add_subplot(nrows, ncols, n + 1, projection=ccrs.PlateCarree())
        ax.set_facecolor("white")
        data_n = apply_mask(dispersion.isel(number=n).values, mask_zone)
        im = ax.contourf(anom.longitude.values, anom.latitude.values, data_n,
                         levels=levels, cmap="RdBu_r", extend="both")
        ax.set_title(f"M{n+1}", fontsize=8, fontweight="bold")
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        add_shapefile_to_ax(ax, gdf_countries, gdf_regions)

    fig.subplots_adjust(bottom=0.06)
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
                        label="Écart à la moyenne d'ensemble (mm)")
    cbar.ax.tick_params(labelsize=9)

    if save:
        fn = os.path.join(output_dir,
                          f"dispersion_{period_type}_{start_str}_to_{end_str}.png")
        fig.savefig(fn, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"    💾 {fn}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all_dispersion(
    products, period_type, init_date_str, gdf_countries, gdf_regions,
    model_name="ECMWF SEAS51", mask_zone=None, extent=None,
    output_dir=OUTPUT_DIR, save=True, show=False,
):
    anom      = products["anomaly"]
    n_periods = anom.sizes["period"]
    for i in range(n_periods):
        anom_i    = anom.isel(period=i)
        p_start   = pd.Timestamp(anom["period_start"].isel(period=i).values)
        p_end     = pd.Timestamp(anom["period_end"].isel(period=i).values)
        start_str = p_start.strftime("%Y-%m-%d")
        end_str   = p_end.strftime("%Y-%m-%d")
        label     = format_period_label(start_str, end_str, period_type)
        plot_member_dispersion(
            anom=anom_i, period_label=label, period_type=period_type,
            start_str=start_str, end_str=end_str, init_date_str=init_date_str,
            gdf_countries=gdf_countries, gdf_regions=gdf_regions,
            mask_zone=mask_zone, model_name=model_name, extent=extent,
            output_dir=output_dir, save=save, show=show,
        )


# ============================================================
# TERCILE SUMMARY MAP
# ============================================================
def plot_tercile_summary(
    prob_BN, prob_NN, prob_AN, lons, lats, dry_mask,
    period_label, init_date_str, gdf_countries, gdf_regions,
    mask_zone=None, model_name="ECMWF SEAS51",
    extent=None, output_path=None,
):
    if extent is None:
        extent = MAP_EXTENT

    pA = prob_AN.values if hasattr(prob_AN, "values") else prob_AN
    pB = prob_BN.values if hasattr(prob_BN, "values") else prob_BN
    pN = prob_NN.values if hasattr(prob_NN, "values") else prob_NN
    dm = dry_mask.values if hasattr(dry_mask, "values") else dry_mask
    lons_np = lons.values if hasattr(lons, "values") else lons
    lats_np = lats.values if hasattr(lats, "values") else lats

    sand_color = "#C0C0C0"
    mask_A = ((pA > 0.38) & (pB < 0.33)) | ((pA > 0.38) & (pN > 0.33))
    mask_B = ((pB > 0.38) & (pA < 0.33)) | ((pB > 0.38) & (pN > 0.33))
    mask_N = (pN > 0.38) & (pA < 0.33) & (pB < 0.33)
    mask_white = ~mask_A & ~mask_B & ~mask_N

    not_dry = ~dm if dm is not None else np.ones_like(pA, dtype=bool)
    if mask_zone is not None:
        not_dry = not_dry & mask_zone
        dm_plot = dm & mask_zone if dm is not None else None
    else:
        dm_plot = dm

    fig = plt.figure(figsize=(12, 10))
    ax  = fig.add_subplot(111, projection=ccrs.PlateCarree())
    setup_map_ax(ax, extent, gdf_countries, gdf_regions)

    if dm_plot is not None:
        ax.pcolormesh(lons_np, lats_np, np.where(dm_plot, 1, np.nan),
                      cmap=ListedColormap([sand_color]), shading="auto",
                      transform=ccrs.PlateCarree())

    ax.pcolormesh(lons_np, lats_np, np.where(mask_white & not_dry, 1, np.nan),
                  cmap=ListedColormap(["white"]), shading="auto",
                  transform=ccrs.PlateCarree())

    im_A = ax.pcolormesh(lons_np, lats_np, np.where(mask_A & not_dry, pA * 100, np.nan),
                         cmap=plt.cm.Blues, vmin=33, vmax=100,
                         shading="auto", transform=ccrs.PlateCarree())
    im_B = ax.pcolormesh(lons_np, lats_np, np.where(mask_B & not_dry, pB * 100, np.nan),
                         cmap=plt.cm.Oranges, vmin=33, vmax=100,
                         shading="auto", transform=ccrs.PlateCarree())
    im_N = ax.pcolormesh(lons_np, lats_np, np.where(mask_N & not_dry, pN * 100, np.nan),
                         cmap=plt.cm.Greens, vmin=33, vmax=100,
                         shading="auto", transform=ccrs.PlateCarree())

    cbar_w, cbar_h, cbar_pad = 0.012, 0.22, 0.015
    start_y, cbar_x = 0.65, 0.80
    for pos, im_i, lbl in [(0, im_A, "Above Normal (%)"),
                           (1, im_N, "Normal (%)"),
                           (2, im_B, "Below Normal (%)")]:
        cax = fig.add_axes([cbar_x, start_y - pos * (cbar_h + cbar_pad), cbar_w, cbar_h])
        cb  = fig.colorbar(im_i, cax=cax, orientation="vertical")
        cb.set_label(lbl, fontsize=9, labelpad=2, fontweight="bold")
        cb.ax.tick_params(labelsize=7)
        cb.set_ticks([33, 50, 70, 90, 100])

    ax.legend(handles=[
        Patch(facecolor="blue",      edgecolor="darkblue",    label="RR > 66th percentile"),
        Patch(facecolor="green",     edgecolor="darkgreen",   label="33rd < RR < 66th"),
        Patch(facecolor="orange",    edgecolor="darkorange",  label="RR < 33rd percentile"),
        Patch(facecolor="white",     edgecolor="black",       label="No dominant class"),
        Patch(facecolor=sand_color,  edgecolor="black",       label="Dry area (clim < 1mm)"),
    ], loc="upper left", fontsize=8, title="Categories", title_fontsize=9)

    ax.set_title(
        f"C3S {model_name} - Tercile Summary\n{period_label}\nInit: {init_date_str}",
        fontsize=13, fontweight="bold", pad=10,
    )
    add_logo(ax)
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"    💾 {output_path}")
    plt.close(fig)


# ============================================================
# SINGLE PROBABILITY MAP
# ============================================================
def plot_single_probability_map(
    prob_data, lons, lats, dry_mask, title, period_label,
    init_date_str, gdf_countries, gdf_regions,
    mask_zone=None, model_name="ECMWF SEAS51",
    cmap=None, prob_levels=None, output_path=None, extent=None,
):
    if extent is None:
        extent = MAP_EXTENT
    if cmap is None:
        cmap = plt.cm.YlOrRd
    if prob_levels is None:
        prob_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    sand_color = "#C0C0C0"
    prob_pct   = prob_data * 100.0
    dm         = dry_mask.values if hasattr(dry_mask, "values") else dry_mask
    pct_np     = prob_pct.values if hasattr(prob_pct, "values") else prob_pct
    lons_np    = lons.values if hasattr(lons, "values") else lons
    lats_np    = lats.values if hasattr(lats, "values") else lats
    plot_mask  = mask_zone if mask_zone is not None else np.ones_like(pct_np, dtype=bool)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    setup_map_ax(ax, extent, gdf_countries, gdf_regions)

    if dm is not None:
        ax.pcolormesh(lons_np, lats_np, np.where(dm & plot_mask, 1, np.nan),
                      cmap=ListedColormap([sand_color]), shading="auto",
                      transform=ccrs.PlateCarree())

    prob_plot = (np.where((~dm) & plot_mask, pct_np, np.nan)
                 if dm is not None
                 else np.where(plot_mask, pct_np, np.nan))
    im = ax.contourf(lons_np, lats_np, prob_plot, levels=prob_levels,
                     cmap=cmap, extend="neither", transform=ccrs.PlateCarree())

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label("Probability (%)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.legend(handles=[Patch(facecolor=sand_color, edgecolor="black",
                             label="Dry area (clim < 1mm)")],
              loc="lower left", fontsize=8)
    ax.set_title(f"C3S {model_name}\n{title}\n{period_label}\nInit: {init_date_str}",
                 fontsize=13, fontweight="bold", pad=10)
    add_logo(ax)
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"    💾 {output_path}")
    plt.close(fig)


# ============================================================
# GROUPED: 7 PROBABILITY MAPS FOR ONE PERIOD
# ============================================================
_PROB_MAP_CONFIGS = [
    ("prob_BN",           "Probability for Lower Tercile (Below Normal)", plt.cm.YlOrBr),
    ("prob_NN",           "Probability for Middle Tercile (Near Normal)",  plt.cm.YlGn),
    ("prob_AN",           "Probability for Upper Tercile (Above Normal)",  plt.cm.YlGnBu),
    ("prob_low20",        "Probability for Lowest 20% of Climatology",     plt.cm.OrRd),
    ("prob_high20",       "Probability for Highest 20% of Climatology",    plt.cm.PuBu),
    ("prob_exceed_median","Probability Exceeding the Median",               plt.cm.RdYlGn),
]


def plot_all_7_maps_for_period(
    probs_i, lons, lats, period_label, period_type,
    start_str, end_str, init_date_str, gdf_countries, gdf_regions,
    model_name="ECMWF SEAS51", mask_zone=None, extent=None,
    output_dir=OUTPUT_DIR, save=True,
):
    if extent is None:
        extent = MAP_EXTENT
    prefix   = f"{period_type}_{start_str}_to_{end_str}"
    dry_mask = probs_i["dry_mask"]
    prob_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    print(f"\n      📊 7 cartes probabilités : {start_str} → {end_str}")

    plot_tercile_summary(
        prob_BN=probs_i["prob_BN"], prob_NN=probs_i["prob_NN"], prob_AN=probs_i["prob_AN"],
        lons=lons, lats=lats, dry_mask=dry_mask,
        period_label=period_label, init_date_str=init_date_str,
        gdf_countries=gdf_countries, gdf_regions=gdf_regions,
        mask_zone=mask_zone, model_name=model_name, extent=extent,
        output_path=os.path.join(output_dir, f"tercile_summary_{prefix}.png") if save else None,
    )

    for key, title, cmap in _PROB_MAP_CONFIGS:
        plot_single_probability_map(
            prob_data=probs_i[key], lons=lons, lats=lats, dry_mask=dry_mask,
            title=title, period_label=period_label, init_date_str=init_date_str,
            gdf_countries=gdf_countries, gdf_regions=gdf_regions,
            mask_zone=mask_zone, model_name=model_name, cmap=cmap,
            prob_levels=prob_levels, extent=extent,
            output_path=os.path.join(output_dir, f"{key}_{prefix}.png") if save else None,
        )
    plt.close("all")


def plot_all_probability_maps(
    probs, products, periods_df, period_type, init_date_str,
    gdf_countries, gdf_regions, model_name="ECMWF SEAS51",
    mask_zone=None, extent=None, output_dir=OUTPUT_DIR, save=True,
):
    forecast_acc = products["forecast_acc"]
    lons         = forecast_acc.longitude.values
    lats         = forecast_acc.latitude.values
    n_periods    = forecast_acc.sizes["period"]
    sub_dir      = os.path.join(output_dir, f"probabilities_{period_type}")
    os.makedirs(sub_dir, exist_ok=True)

    print(f"\n    {'#'*60}")
    print(f"    CARTES DE PROBABILITÉS - {period_type.upper()}")
    print(f"    {n_periods} périodes × 7 cartes = {n_periods * 7} figures")
    print(f"    {'#'*60}")

    for i in range(n_periods):
        p_start   = pd.Timestamp(forecast_acc["period_start"].isel(period=i).values)
        p_end     = pd.Timestamp(forecast_acc["period_end"].isel(period=i).values)
        start_str = p_start.strftime("%Y-%m-%d")
        end_str   = p_end.strftime("%Y-%m-%d")
        label     = format_period_label(start_str, end_str, period_type)
        probs_i   = {k: probs[k].isel(period=i)
                     for k in ["prob_BN", "prob_NN", "prob_AN", "prob_low20",
                                "prob_high20", "prob_exceed_median", "dry_mask"]}
        plot_all_7_maps_for_period(
            probs_i=probs_i, lons=lons, lats=lats, period_label=label,
            period_type=period_type, start_str=start_str, end_str=end_str,
            init_date_str=init_date_str, gdf_countries=gdf_countries,
            gdf_regions=gdf_regions, model_name=model_name, mask_zone=mask_zone,
            extent=extent, output_dir=sub_dir, save=save,
        )
        plt.close("all")

    print(f"\n    ✅ {n_periods * 7} figures probabilités '{period_type}' générées dans {sub_dir}")


# ============================================================
# EXCEEDANCE PROBABILITY MAP
# ============================================================
def plot_exceedance_probability_map(
    prob, threshold_mm, lons, lats, period_label,
    init_date_str, gdf_countries, gdf_regions,
    mask_zone=None, model_name="ECMWF SEAS51",
    extent=None, output_path=None,
):
    if extent is None:
        extent = MAP_EXTENT
    prob_pct    = (prob * 100.0)
    prob_pct_np = prob_pct.values if hasattr(prob_pct, "values") else prob_pct
    lons_np     = lons.values if hasattr(lons, "values") else lons
    lats_np     = lats.values if hasattr(lats, "values") else lats
    prob_plot   = apply_mask(prob_pct_np, mask_zone)
    levels      = np.arange(0, 101, 1)

    fig = plt.figure(figsize=(12, 10))
    ax  = fig.add_subplot(111, projection=ccrs.PlateCarree())
    setup_map_ax(ax, extent, gdf_countries, gdf_regions)
    im = ax.contourf(lons_np, lats_np, prob_plot, levels=levels,
                     cmap=PROB_CMAP, vmin=0, vmax=100, extend="neither",
                     transform=ccrs.PlateCarree())
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label(f"Probability of exceeding {threshold_mm} mm (%)", fontsize=11)
    cbar.set_ticks(np.arange(0, 101, 5))
    cbar.ax.tick_params(labelsize=9)
    ax.set_title(
        f"C3S {model_name}\nProbability of Rainfall ≥ {threshold_mm} mm\n"
        f"{period_label}\nInit: {init_date_str}",
        fontsize=12, fontweight="bold",
    )
    add_logo(ax)
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"    💾 {output_path}")
    plt.close(fig)


def plot_all_exceedance_maps(
    exceed_probs, products, period_type, init_date_str,
    gdf_countries, gdf_regions, model_name="ECMWF SEAS51",
    mask_zone=None, extent=None, output_dir=OUTPUT_DIR, save=True,
):
    forecast_acc = products["forecast_acc"]
    n_periods    = forecast_acc.sizes["period"]
    for i in range(n_periods):
        p_start   = pd.Timestamp(forecast_acc["period_start"].isel(period=i).values)
        p_end     = pd.Timestamp(forecast_acc["period_end"].isel(period=i).values)
        start_str = p_start.strftime("%Y-%m-%d")
        end_str   = p_end.strftime("%Y-%m-%d")
        label     = format_period_label(start_str, end_str, period_type)
        prefix    = f"{period_type}_{start_str}_to_{end_str}"
        for thr in sorted(exceed_probs.keys()):
            plot_exceedance_probability_map(
                prob=exceed_probs[thr].isel(period=i),
                threshold_mm=thr,
                lons=forecast_acc.longitude, lats=forecast_acc.latitude,
                period_label=label, init_date_str=init_date_str,
                gdf_countries=gdf_countries, gdf_regions=gdf_regions,
                mask_zone=mask_zone, model_name=model_name, extent=extent,
                output_path=(os.path.join(output_dir, f"exceed_{thr}mm_{prefix}.png")
                             if save else None),
            )


# ============================================================
# MULTI-MODEL PANELS
# ============================================================
def plot_multimodel_panel(
    data_dict, lons, lats, title_main, colorbar_label,
    period_label, init_date_str, gdf_countries, gdf_regions,
    mask_zone=None, levels=None, cmap=None, colors_list=None,
    extend="both", extent=None, output_path=None,
):
    if extent is None:
        extent = MAP_EXTENT
    n_panels = len(data_dict)
    ncols    = 3
    nrows    = int(np.ceil(n_panels / ncols))
    lons_np  = lons.values if hasattr(lons, "values") else lons
    lats_np  = lats.values if hasattr(lats, "values") else lats

    fig = plt.figure(figsize=_panel_figsize(extent, ncols, nrows))
    plt.subplots_adjust(left=0.04, right=0.96, top=0.91, bottom=0.08,
                        hspace=0.12, wspace=0.02)
    fig.suptitle(f"{title_main}\n{period_label}\nInit: {init_date_str}",
                 fontsize=16, fontweight="bold", y=0.98)

    im = None
    for idx, (key, data) in enumerate(data_dict.items()):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection=ccrs.PlateCarree())
        ax.set_facecolor("white")
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                          color="gray", alpha=0.4, linestyle="--")
        gl.top_labels   = False
        gl.right_labels = (idx % ncols == ncols - 1)
        gl.left_labels  = (idx % ncols == 0)
        gl.xlabel_style = {"size": 9}
        gl.ylabel_style = {"size": 9}
        add_shapefile_to_ax(ax, gdf_countries, gdf_regions)
        data_plot = apply_mask(data, mask_zone)

        if colors_list is not None and levels is not None:
            im = ax.contourf(lons_np, lats_np, data_plot,
                             levels=levels, colors=colors_list, extend=extend)
        elif cmap is not None and levels is not None:
            im = ax.contourf(lons_np, lats_np, data_plot,
                             levels=levels, cmap=cmap, extend=extend)
        elif cmap is not None:
            im = ax.contourf(lons_np, lats_np, data_plot, cmap=cmap, extend=extend)

        label = MODEL_NAMES.get(key, key)
        if key == "multi_model":
            ax.set_title("★ MULTI-MODEL MEAN", fontsize=11, fontweight="bold", color="red")
        else:
            ax.set_title(label, fontsize=10, fontweight="bold")
        add_logo(ax, zoom=0.12)

    fig.subplots_adjust(bottom=0.06)
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label=colorbar_label)
    cbar.ax.tick_params(labelsize=9)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  💾 {output_path}")
    plt.close(fig)


def plot_multimodel_tercile_panel(
    all_probs, lons, lats, period_label, init_date_str,
    gdf_countries, gdf_regions, mask_zone=None, extent=None, output_path=None,
):
    if extent is None:
        extent = MAP_EXTENT
    sand_color = "#C0C0C0"
    n_panels   = len(all_probs)
    ncols      = 3
    nrows      = int(np.ceil(n_panels / ncols))
    lons_np    = lons.values if hasattr(lons, "values") else lons
    lats_np    = lats.values if hasattr(lats, "values") else lats

    fig = plt.figure(figsize=_panel_figsize(extent, ncols, nrows, extra_h=2.2))
    plt.subplots_adjust(left=0.04, right=0.96, top=0.91, bottom=0.10,
                        hspace=0.12, wspace=0.02)
    fig.suptitle(f"Tercile Summary - All Models\n{period_label}\nInit: {init_date_str}",
                 fontsize=16, fontweight="bold", y=0.98)

    for idx, (key, probs_i) in enumerate(all_probs.items()):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection=ccrs.PlateCarree())
        ax.set_facecolor("white")
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                          color="gray", alpha=0.4, linestyle="--")
        gl.top_labels   = False
        gl.right_labels = (idx % ncols == ncols - 1)
        gl.left_labels  = (idx % ncols == 0)
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        add_shapefile_to_ax(ax, gdf_countries, gdf_regions)

        pA = probs_i["prob_AN"]
        pB = probs_i["prob_BN"]
        pN = probs_i["prob_NN"]
        dm = probs_i.get("dry_mask", np.zeros_like(pA, dtype=bool))

        pA = pA.values if hasattr(pA, "values") else pA
        pB = pB.values if hasattr(pB, "values") else pB
        pN = pN.values if hasattr(pN, "values") else pN
        dm = dm.values if hasattr(dm, "values") else dm

        not_dry = ~dm
        if mask_zone is not None:
            not_dry = not_dry & mask_zone
            dm_plot = dm & mask_zone
        else:
            dm_plot = dm

        mask_A = ((pA > 0.38) & (pB < 0.33)) | ((pA > 0.38) & (pN > 0.33))
        mask_B = ((pB > 0.38) & (pA < 0.33)) | ((pB > 0.38) & (pN > 0.33))
        mask_N = (pN > 0.38) & (pA < 0.33) & (pB < 0.33)
        mask_white = ~mask_A & ~mask_B & ~mask_N

        ax.pcolormesh(lons_np, lats_np, np.where(dm_plot, 1, np.nan),
                      cmap=ListedColormap([sand_color]), shading="auto",
                      transform=ccrs.PlateCarree())
        ax.pcolormesh(lons_np, lats_np, np.where(mask_white & not_dry, 1, np.nan),
                      cmap=ListedColormap(["white"]), shading="auto",
                      transform=ccrs.PlateCarree())
        ax.pcolormesh(lons_np, lats_np, np.where(mask_A & not_dry, pA * 100, np.nan),
                      cmap=plt.cm.Blues, vmin=33, vmax=100,
                      shading="auto", transform=ccrs.PlateCarree())
        ax.pcolormesh(lons_np, lats_np, np.where(mask_B & not_dry, pB * 100, np.nan),
                      cmap=plt.cm.Oranges, vmin=33, vmax=100,
                      shading="auto", transform=ccrs.PlateCarree())
        ax.pcolormesh(lons_np, lats_np, np.where(mask_N & not_dry, pN * 100, np.nan),
                      cmap=plt.cm.Greens, vmin=33, vmax=100,
                      shading="auto", transform=ccrs.PlateCarree())

        label = MODEL_NAMES.get(key, key)
        if key == "multi_model":
            ax.set_title("★ MULTI-MODEL", fontsize=11, fontweight="bold", color="red")
        else:
            ax.set_title(label, fontsize=10, fontweight="bold")

    fig.legend(handles=[
        Patch(facecolor="blue",     edgecolor="darkblue",   label="Above Normal"),
        Patch(facecolor="green",    edgecolor="darkgreen",  label="Normal"),
        Patch(facecolor="orange",   edgecolor="darkorange", label="Below Normal"),
        Patch(facecolor="white",    edgecolor="black",      label="No dominant"),
        Patch(facecolor=sand_color, edgecolor="black",      label="Dry area"),
    ], loc="lower center", ncol=5, fontsize=9, frameon=True)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  💾 {output_path}")
    plt.close(fig)
