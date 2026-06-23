"""
Forecast verification module — C3S seasonal hindcasts (1993-2016) vs CHIRPS.

Implements the WMO-1220 / IRI score recommendations:
    - RPSS  (Ranked Probability Skill Score)        — overall probabilistic skill
    - BSS   (Brier Skill Score) per tercile          — categorical probabilistic skill
    - ROC area per tercile                           — discrimination
    - Reliability diagrams (3 categories)            — calibration
    - Ensemble mean correlation and RMSE             — deterministic skill

All probabilistic scores use leave-one-out (LOO) cross-validation against the
hindcast itself for tercile thresholds, and LOO CHIRPS climatology for the
observed category.
"""
import os
import gc
import traceback
import numpy as np
import pandas as pd
import xarray as xr

# np.trapezoid is numpy 2.0+; np.trapz exists in 1.x and is still callable in 2.x
_trapz = getattr(np, "trapezoid", np.trapz)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs

from eccas_s2s.config import (
    DATADIR, OUTPUT_DIR, MODELS, MODEL_NAMES, MAP_EXTENT, INIT_MONTH_STR,
)
from eccas_s2s.core.data import open_grib_dataset, safe_close_xarray, print_memory_usage
from eccas_s2s.core.processing import (
    decumulate_precip_to_mm,
    accumulate_hindcast_over_periods,
)
from eccas_s2s.viz.plotting import (
    setup_map_ax, add_shapefile_to_ax, add_logo,
    apply_mask, format_period_label,
)


# ============================================================
# CONSTANTS
# ============================================================
CHIRPS_PATH = "/home/ludric/Desktop/Donnees_calibration_c3s/DATA/CHIRPS/TP_CHIRPS_v2.0_DAILY_1991_2025_p05_ECCAS.nc"
HINDCAST_YEARS = list(range(1993, 2017))           # 1993-2016 inclusive
N_BINS_RELIABILITY = 10
VERIFICATION_DIR = os.path.join(OUTPUT_DIR, "verification")


# ============================================================
# 1. DATA LOADING
# ============================================================
def load_chirps_aligned(chirps_path, model_template, years, periods_df):
    """
    Load CHIRPS, regrid (bilinear) to the model grid and accumulate
    precipitation over each period for each year.

    Returns
    -------
    out : DataArray (period, time, latitude, longitude) in mm
    """
    print(f"  📂 Lecture CHIRPS : {chirps_path}")
    chirps = xr.open_dataset(chirps_path, chunks={"time": 365})["precip"]  # mm/day
    chirps = chirps.sel(time=chirps["time"].dt.year.isin(years))

    target_lat = model_template["latitude"].values
    target_lon = model_template["longitude"].values
    print(f"  🔃 Régrillage CHIRPS sur grille modèle ({len(target_lat)}×{len(target_lon)})...")
    chirps_rg = chirps.interp(
        latitude=target_lat, longitude=target_lon, method="linear"
    )
    chirps_rg = chirps_rg.load()

    print(f"  📊 Cumul CHIRPS sur {len(periods_df)} périodes × {len(years)} années...")
    md = chirps_rg["time"].dt.month * 100 + chirps_rg["time"].dt.day

    accumulations = []
    for i, row in periods_df.iterrows():
        s_md = row["start_month"] * 100 + row["start_day"]
        e_md = row["end_month"]   * 100 + row["end_day"]
        if s_md <= e_md:
            mask = (md >= s_md) & (md <= e_md)
        else:
            mask = (md >= s_md) | (md <= e_md)
        masked = chirps_rg.where(mask, 0)
        annual = masked.groupby("time.year").sum(dim="time", skipna=False)
        annual = annual.expand_dims(period=[i])
        accumulations.append(annual)

    out = xr.concat(accumulations, dim="period")
    out = out.rename({"year": "time"})
    out = out.sel(time=[y for y in years if y in out["time"].values])
    out = out.assign_coords(
        period       =("period", periods_df.index.values),
        period_label =("period", periods_df["label"].values),
        period_type  =("period", periods_df["type"].values),
        period_start =("period", pd.to_datetime(periods_df["start"]).values),
        period_end   =("period", pd.to_datetime(periods_df["end"]).values),
    )
    out.attrs["units"]     = "mm"
    out.attrs["long_name"] = "CHIRPS precipitation accumulation"
    return out


def load_model_hindcast_accumulated(model, periods_df):
    """
    Open the model hindcast GRIB, decumulate to daily mm and accumulate over
    each period using a month-day mask.

    Returns
    -------
    acc : DataArray (period, number, time, latitude, longitude) in mm,
          with 'time' replaced by the calendar year.
    """
    hindcast_path = os.path.join(
        DATADIR, f"{model}_hindcast_1993_2016_{INIT_MONTH_STR}.grib"
    )
    if not os.path.isfile(hindcast_path):
        print(f"    ⚠ Hindcast manquant : {hindcast_path}")
        return None

    print(f"    📂 Lecture hindcast : {hindcast_path}")
    hindcast_raw = open_grib_dataset(hindcast_path)
    hindcast_daily = decumulate_precip_to_mm(hindcast_raw, clip_negative=True)

    acc = accumulate_hindcast_over_periods(hindcast_daily, periods_df, var="tp_daily")
    acc = acc.load()

    years = pd.to_datetime(acc["time"].values).year
    acc = acc.assign_coords(time=("time", years.values))

    safe_close_xarray(hindcast_raw)
    safe_close_xarray(hindcast_daily)
    del hindcast_raw, hindcast_daily
    gc.collect()
    return acc


# ============================================================
# 2. LEAVE-ONE-OUT TERCILES
# ============================================================
def leave_one_out_terciles(data, year_dim="time"):
    """
    Compute leave-one-out tercile thresholds (1/3 and 2/3 quantiles).

    Returns
    -------
    t_low, t_up : DataArrays with the same dims as the year axis of `data`,
                  i.e. (period, year_dim, lat, lon).
    """
    n = data.sizes[year_dim]
    t_low_list, t_up_list = [], []
    for yi in range(n):
        others_idx = [i for i in range(n) if i != yi]
        others = data.isel({year_dim: others_idx})
        if "number" in others.dims:
            sample = others.stack(sample=(year_dim, "number"))
        else:
            sample = others.rename({year_dim: "sample"})
        t_low = sample.quantile(1/3, dim="sample").drop_vars("quantile", errors="ignore")
        t_up  = sample.quantile(2/3, dim="sample").drop_vars("quantile", errors="ignore")
        t_low_list.append(t_low)
        t_up_list.append(t_up)
    t_low_all = xr.concat(t_low_list, dim=year_dim).assign_coords({year_dim: data[year_dim]})
    t_up_all  = xr.concat(t_up_list,  dim=year_dim).assign_coords({year_dim: data[year_dim]})
    return t_low_all, t_up_all


def forecast_tercile_probs(hindcast_acc, t_low_cv, t_up_cv):
    """
    Forecast tercile probabilities using LOO climatology thresholds.

    Returns
    -------
    pBN, pNN, pAN : DataArrays (period, time, lat, lon), each in [0, 1].
    """
    pBN = (hindcast_acc < t_low_cv).mean(dim="number")
    pAN = (hindcast_acc > t_up_cv).mean(dim="number")
    pNN = 1.0 - pBN - pAN
    return pBN, pNN, pAN


def observed_tercile_category(obs_acc, year_dim="time"):
    """
    Observed category 0=BN, 1=NN, 2=AN using LOO CHIRPS climatology.
    """
    t_low_cv, t_up_cv = leave_one_out_terciles(obs_acc, year_dim=year_dim)
    cat = xr.where(obs_acc < t_low_cv, 0, xr.where(obs_acc > t_up_cv, 2, 1))
    return cat.astype("int8")


# ============================================================
# 3. SCORES — PER-GRIDPOINT MAPS
# ============================================================
def compute_rpss_maps(pBN, pNN, pAN, obs_cat):
    """
    Time-mean RPS for forecast and climatology; RPSS at each (period, lat, lon).
    """
    cum_f1 = pBN
    cum_f2 = pBN + pNN
    cum_o1 = (obs_cat <= 0).astype(float)
    cum_o2 = (obs_cat <= 1).astype(float)

    rps_f = ((cum_f1 - cum_o1) ** 2 + (cum_f2 - cum_o2) ** 2) / 2.0
    rps_c = ((1/3 - cum_o1) ** 2 + (2/3 - cum_o2) ** 2) / 2.0

    rps_f_mean = rps_f.mean(dim="time")
    rps_c_mean = rps_c.mean(dim="time")
    rpss = 1.0 - rps_f_mean / rps_c_mean
    return rps_f_mean, rps_c_mean, rpss


def compute_bss_maps(prob_cat, event):
    """
    Brier Score and Brier Skill Score for one category.

    prob_cat : (period, time, lat, lon) probability of being in the category
    event    : (period, time, lat, lon) 0/1 whether obs was in the category
    """
    bs_f = ((prob_cat - event) ** 2).mean(dim="time")
    bs_c = ((1/3 - event) ** 2).mean(dim="time")
    bss = 1.0 - bs_f / bs_c
    return bs_f, bs_c, bss


def _roc_area_1d(probs, events):
    """Trapezoidal ROC area for a 1-D sample of forecast probabilities and 0/1 events."""
    if np.isnan(probs).any() or np.isnan(events).any():
        return np.nan
    n = len(events)
    n_pos = int(events.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(-probs, kind="stable")
    es = events[order].astype(float)
    tpr = np.cumsum(es) / n_pos
    fpr = np.cumsum(1.0 - es) / n_neg
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    return float(_trapz(tpr, fpr))


def compute_roc_area_maps(prob_cat, event):
    """
    Per-gridpoint ROC area for one category, using yearly samples.
    Returns DataArray (period, lat, lon).
    """
    p_np = prob_cat.values
    e_np = event.values.astype(float)
    nP, nT, nLa, nLo = p_np.shape
    out = np.full((nP, nLa, nLo), np.nan, dtype=np.float32)
    for ip in range(nP):
        for ila in range(nLa):
            for ilo in range(nLo):
                out[ip, ila, ilo] = _roc_area_1d(
                    p_np[ip, :, ila, ilo], e_np[ip, :, ila, ilo]
                )
    return xr.DataArray(
        out,
        dims=("period", "latitude", "longitude"),
        coords={
            "period":    prob_cat["period"],
            "latitude":  prob_cat["latitude"],
            "longitude": prob_cat["longitude"],
        },
        name="roc_area",
    )


def compute_ensmean_skill_maps(hindcast_acc, obs_acc):
    """
    Pearson anomaly correlation and RMSE of the ensemble mean vs CHIRPS.
    """
    ens_mean = hindcast_acc.mean(dim="number")
    em_anom  = ens_mean - ens_mean.mean(dim="time")
    o_anom   = obs_acc - obs_acc.mean(dim="time")
    num = (em_anom * o_anom).sum(dim="time")
    den = np.sqrt((em_anom ** 2).sum(dim="time") * (o_anom ** 2).sum(dim="time"))
    corr = num / den
    rmse = np.sqrt(((ens_mean - obs_acc) ** 2).mean(dim="time"))
    return corr, rmse


# ============================================================
# 4. RELIABILITY (domain-pooled)
# ============================================================
def compute_reliability_per_category(prob_cat, event, mask_zone=None,
                                      n_bins=N_BINS_RELIABILITY):
    """
    Reliability diagram inputs for ONE category, pooled over time and the
    masked spatial domain.

    Returns
    -------
    bin_centers : (n_bins,)
    mean_fcst   : (period, n_bins)
    obs_freq    : (period, n_bins)
    counts      : (period, n_bins) ints
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    nP = prob_cat.sizes["period"]
    mean_fcst = np.full((nP, n_bins), np.nan, dtype=np.float32)
    obs_freq  = np.full((nP, n_bins), np.nan, dtype=np.float32)
    counts    = np.zeros((nP, n_bins), dtype=np.int64)

    p_np = prob_cat.values
    e_np = event.values.astype(float)
    if mask_zone is not None:
        # broadcast to (time, lat, lon) then ravel within each period
        mask3 = np.broadcast_to(mask_zone, (p_np.shape[1],) + mask_zone.shape)
    else:
        mask3 = None

    for ip in range(nP):
        p = p_np[ip].ravel()
        e = e_np[ip].ravel()
        if mask3 is not None:
            m = mask3.ravel()
            p, e = p[m], e[m]
        valid = ~(np.isnan(p) | np.isnan(e))
        p, e = p[valid], e[valid]
        for ib in range(n_bins):
            lo = bins[ib]
            hi = bins[ib + 1]
            if ib == n_bins - 1:
                in_bin = (p >= lo) & (p <= hi)
            else:
                in_bin = (p >= lo) & (p <  hi)
            n_in = int(in_bin.sum())
            counts[ip, ib] = n_in
            if n_in > 0:
                mean_fcst[ip, ib] = p[in_bin].mean()
                obs_freq[ip, ib]  = e[in_bin].mean()
    return bin_centers, mean_fcst, obs_freq, counts


# ============================================================
# 5. PLOTTING
# ============================================================
_LEVELS_SKILL = np.array([-1.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
_LEVELS_ROC   = np.array([0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0])
_LEVELS_CORR  = np.array([-1.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0])


def _safe_filename(s):
    return s.replace("/", "-").replace(" ", "_")


def plot_skill_map(
    da, period_label, score_name, model_name, output_path,
    gdf_countries, gdf_regions, mask_zone=None,
    cmap="RdBu_r", levels=None, center_zero=True, extent=None,
):
    if extent is None:
        extent = MAP_EXTENT
    fig = plt.figure(figsize=(12, 10))
    ax  = fig.add_subplot(111, projection=ccrs.PlateCarree())
    setup_map_ax(ax, extent, gdf_countries, gdf_regions)

    data = apply_mask(da.values, mask_zone)
    if levels is not None:
        im = ax.contourf(da.longitude, da.latitude, data, levels=levels,
                         cmap=cmap, extend="both", transform=ccrs.PlateCarree())
    else:
        im = ax.contourf(da.longitude, da.latitude, data, cmap=cmap,
                         transform=ccrs.PlateCarree(), extend="both")

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label(score_name, fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    ax.set_title(f"{model_name} — {score_name}\n{period_label}",
                 fontsize=13, fontweight="bold", pad=10)
    add_logo(ax)
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"    💾 {output_path}")
    plt.close(fig)


def plot_reliability_diagram_three_cat(
    bin_centers, mean_fcst_all, obs_freq_all, counts_all,
    period_label, model_name, output_path,
):
    """
    3-panel reliability diagram (BN, NN, AN).
    mean_fcst_all / obs_freq_all / counts_all : dict {'BN','NN','AN'} -> (n_bins,)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cat_names = ["BN", "NN", "AN"]
    cat_labels = {"BN": "Below Normal", "NN": "Near Normal", "AN": "Above Normal"}
    colors    = {"BN": "#d95f02",       "NN": "#1b9e77",     "AN": "#3b6bd6"}

    for ax, cat in zip(axes, cat_names):
        mf  = mean_fcst_all[cat]
        of  = obs_freq_all[cat]
        cnt = counts_all[cat]
        valid = cnt > 0

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect reliability")
        ax.axhline(1/3, color="gray", linestyle=":", lw=1)
        ax.axvline(1/3, color="gray", linestyle=":", lw=1, label="Climatology (1/3)")

        if valid.any():
            ax.plot(mf[valid], of[valid], "o-",
                    color=colors[cat], lw=2.5, markersize=9, label=cat_labels[cat])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Forecast probability")
        ax.set_ylabel("Observed relative frequency")
        ax.set_title(cat_labels[cat], fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

        axin = inset_axes(ax, width="36%", height="32%", loc="lower right", borderpad=1.4)
        axin.bar(bin_centers, cnt, width=0.08,
                 color=colors[cat], alpha=0.65, edgecolor="black", linewidth=0.4)
        total = max(int(cnt.sum()), 1)
        axin.set_yscale("log")
        axin.set_title(f"Counts (N={total:,})", fontsize=8)
        axin.tick_params(labelsize=7)

    fig.suptitle(f"{model_name} — Reliability Diagram\n{period_label}",
                 fontsize=14, fontweight="bold", y=1.02)
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"    💾 {output_path}")
    plt.close(fig)


# ============================================================
# 6. SAVING
# ============================================================
def save_verification_dataset(
    pBN, pNN, pAN, obs_cat,
    rpss, bss_BN, bss_NN, bss_AN,
    roc_BN, roc_AN, corr, rmse,
    model, period_type, output_path,
):
    ds = xr.Dataset({
        "pBN":      pBN.astype("float32"),
        "pNN":      pNN.astype("float32"),
        "pAN":      pAN.astype("float32"),
        "obs_cat":  obs_cat.astype("int8"),
        "rpss":     rpss.astype("float32"),
        "bss_BN":   bss_BN.astype("float32"),
        "bss_NN":   bss_NN.astype("float32"),
        "bss_AN":   bss_AN.astype("float32"),
        "roc_BN":   roc_BN.astype("float32"),
        "roc_AN":   roc_AN.astype("float32"),
        "corr_ens": corr.astype("float32"),
        "rmse_ens": rmse.astype("float32"),
    })
    ds.attrs["model"]              = model
    ds.attrs["period_type"]        = period_type
    ds.attrs["verification_years"] = f"{HINDCAST_YEARS[0]}-{HINDCAST_YEARS[-1]}"
    ds.attrs["observation"]        = "CHIRPS v2.0"
    ds.attrs["description"]        = "LOO-CV verification scores against CHIRPS"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    try:
        ds.to_netcdf(output_path, encoding=enc)
    except Exception:
        ds.to_netcdf(output_path)
    print(f"    💾 {output_path}")


# ============================================================
# 7. PER-MODEL VERIFICATION (single period type)
# ============================================================
def _domain_stats(da, mask_zone):
    """Return (mean, fraction_positive) of a 2-D map within mask_zone."""
    vals = da.values
    if mask_zone is not None:
        vals = np.where(mask_zone, vals, np.nan)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.nan, np.nan
    return float(np.nanmean(finite)), float((finite > 0).sum() / finite.size)


def verify_model_for_period_type(
    model, period_type, periods_df, chirps_acc,
    gdf_countries, gdf_regions, mask_zone=None,
    output_root=VERIFICATION_DIR,
):
    """
    Run the full verification for one model and one period type.

    Returns
    -------
    saved_path : str  path to the saved scores NetCDF, or None on failure.
    """
    model_name = MODEL_NAMES.get(model, model)
    print(f"\n  🔬 {model_name} — {period_type} ({len(periods_df)} périodes)")

    try:
        hindcast_acc = load_model_hindcast_accumulated(model, periods_df)
        if hindcast_acc is None:
            return None
    except Exception as e:
        print(f"    ❌ Lecture hindcast {model_name} : {e}")
        traceback.print_exc()
        return None

    # Restrict CHIRPS years to those present in the hindcast
    hindcast_years = sorted(np.unique(hindcast_acc["time"].values).tolist())
    obs_for_periods = chirps_acc.sel(time=hindcast_years)

    # Align period coordinate so isel/sel match between hindcast and obs
    hindcast_acc = hindcast_acc.assign_coords(period=periods_df.index.values)
    obs_for_periods = obs_for_periods.assign_coords(period=periods_df.index.values)

    try:
        print(f"    📊 LOO terciles modèle...")
        t_low_cv, t_up_cv = leave_one_out_terciles(hindcast_acc, year_dim="time")
        pBN, pNN, pAN = forecast_tercile_probs(hindcast_acc, t_low_cv, t_up_cv)

        print(f"    📊 Catégorie observée (LOO CHIRPS)...")
        obs_cat = observed_tercile_category(obs_for_periods, year_dim="time")
        ev_BN = (obs_cat == 0).astype(float)
        ev_NN = (obs_cat == 1).astype(float)
        ev_AN = (obs_cat == 2).astype(float)

        print(f"    📊 RPS / RPSS...")
        _, _, rpss = compute_rpss_maps(pBN, pNN, pAN, obs_cat)

        print(f"    📊 BSS par catégorie...")
        _, _, bss_BN = compute_bss_maps(pBN, ev_BN)
        _, _, bss_NN = compute_bss_maps(pNN, ev_NN)
        _, _, bss_AN = compute_bss_maps(pAN, ev_AN)

        print(f"    📊 ROC area BN / AN...")
        roc_BN = compute_roc_area_maps(pBN, ev_BN)
        roc_AN = compute_roc_area_maps(pAN, ev_AN)

        print(f"    📊 Compétence ens-mean (corr, RMSE)...")
        corr, rmse = compute_ensmean_skill_maps(hindcast_acc, obs_for_periods)

        # ---- save scores ----
        model_root = os.path.join(output_root, model)
        scores_path = os.path.join(model_root, "scores", f"{period_type}_scores.nc")
        save_verification_dataset(
            pBN, pNN, pAN, obs_cat,
            rpss, bss_BN, bss_NN, bss_AN,
            roc_BN, roc_AN, corr, rmse,
            model=model, period_type=period_type, output_path=scores_path,
        )

        # ---- plot per-period maps ----
        n_periods = rpss.sizes["period"]
        maps_dir = os.path.join(model_root, "maps", period_type)
        os.makedirs(maps_dir, exist_ok=True)

        for i in range(n_periods):
            p_start = pd.Timestamp(rpss["period_start"].isel(period=i).values)
            p_end   = pd.Timestamp(rpss["period_end"].isel(period=i).values)
            sstr, estr = p_start.strftime("%Y-%m-%d"), p_end.strftime("%Y-%m-%d")
            label = format_period_label(sstr, estr, period_type)
            tag = _safe_filename(f"{period_type}_{sstr}_to_{estr}")

            plot_skill_map(rpss.isel(period=i), label, "RPSS", model_name,
                           os.path.join(maps_dir, f"RPSS_{tag}.png"),
                           gdf_countries, gdf_regions, mask_zone=mask_zone,
                           cmap="RdBu_r", levels=_LEVELS_SKILL)
            plot_skill_map(bss_BN.isel(period=i), label, "BSS (Below Normal)", model_name,
                           os.path.join(maps_dir, f"BSS_BN_{tag}.png"),
                           gdf_countries, gdf_regions, mask_zone=mask_zone,
                           cmap="RdBu_r", levels=_LEVELS_SKILL)
            plot_skill_map(bss_AN.isel(period=i), label, "BSS (Above Normal)", model_name,
                           os.path.join(maps_dir, f"BSS_AN_{tag}.png"),
                           gdf_countries, gdf_regions, mask_zone=mask_zone,
                           cmap="RdBu_r", levels=_LEVELS_SKILL)
            plot_skill_map(roc_BN.isel(period=i), label, "ROC area (Below Normal)", model_name,
                           os.path.join(maps_dir, f"ROC_BN_{tag}.png"),
                           gdf_countries, gdf_regions, mask_zone=mask_zone,
                           cmap="YlOrRd", levels=_LEVELS_ROC)
            plot_skill_map(roc_AN.isel(period=i), label, "ROC area (Above Normal)", model_name,
                           os.path.join(maps_dir, f"ROC_AN_{tag}.png"),
                           gdf_countries, gdf_regions, mask_zone=mask_zone,
                           cmap="YlOrRd", levels=_LEVELS_ROC)
            plot_skill_map(corr.isel(period=i), label, "Ens-mean Anomaly Correlation", model_name,
                           os.path.join(maps_dir, f"CORR_{tag}.png"),
                           gdf_countries, gdf_regions, mask_zone=mask_zone,
                           cmap="RdBu_r", levels=_LEVELS_CORR)
            plot_skill_map(rmse.isel(period=i), label, "Ens-mean RMSE (mm)", model_name,
                           os.path.join(maps_dir, f"RMSE_{tag}.png"),
                           gdf_countries, gdf_regions, mask_zone=mask_zone,
                           cmap="viridis", levels=None)
            plt.close("all")

        # ---- reliability diagrams (pooled over space) ----
        rel_dir = os.path.join(model_root, "reliability", period_type)
        os.makedirs(rel_dir, exist_ok=True)
        bin_centers, mf_BN, of_BN, cnt_BN = compute_reliability_per_category(pBN, ev_BN, mask_zone)
        _,            mf_NN, of_NN, cnt_NN = compute_reliability_per_category(pNN, ev_NN, mask_zone)
        _,            mf_AN, of_AN, cnt_AN = compute_reliability_per_category(pAN, ev_AN, mask_zone)
        for i in range(n_periods):
            p_start = pd.Timestamp(rpss["period_start"].isel(period=i).values)
            p_end   = pd.Timestamp(rpss["period_end"].isel(period=i).values)
            sstr, estr = p_start.strftime("%Y-%m-%d"), p_end.strftime("%Y-%m-%d")
            label = format_period_label(sstr, estr, period_type)
            tag = _safe_filename(f"{period_type}_{sstr}_to_{estr}")
            plot_reliability_diagram_three_cat(
                bin_centers,
                {"BN": mf_BN[i], "NN": mf_NN[i], "AN": mf_AN[i]},
                {"BN": of_BN[i], "NN": of_NN[i], "AN": of_AN[i]},
                {"BN": cnt_BN[i], "NN": cnt_NN[i], "AN": cnt_AN[i]},
                period_label=label, model_name=model_name,
                output_path=os.path.join(rel_dir, f"reliability_{tag}.png"),
            )

        # ---- summary text ----
        summary_path = os.path.join(model_root, "scores", f"{period_type}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"{model_name} — {period_type} verification summary\n")
            f.write(f"Years: {hindcast_years[0]}-{hindcast_years[-1]}, "
                    f"obs: CHIRPS v2.0\n\n")
            f.write(f"{'Period':40s}  {'<RPSS>':>8s}  {'%RPSS>0':>8s}  "
                    f"{'<ROC_BN>':>9s}  {'<ROC_AN>':>9s}  {'<CORR>':>8s}\n")
            for i in range(n_periods):
                p_start = pd.Timestamp(rpss["period_start"].isel(period=i).values)
                p_end   = pd.Timestamp(rpss["period_end"].isel(period=i).values)
                tag = f"{p_start:%Y-%m-%d}_to_{p_end:%Y-%m-%d}"
                m_rpss, pos = _domain_stats(rpss.isel(period=i), mask_zone)
                m_rbn, _    = _domain_stats(roc_BN.isel(period=i), mask_zone)
                m_ran, _    = _domain_stats(roc_AN.isel(period=i), mask_zone)
                m_corr, _   = _domain_stats(corr.isel(period=i), mask_zone)
                f.write(f"{tag:40s}  {m_rpss:8.3f}  {pos*100:7.1f}%  "
                        f"{m_rbn:9.3f}  {m_ran:9.3f}  {m_corr:8.3f}\n")
        print(f"    📝 {summary_path}")

        return scores_path

    except Exception as e:
        print(f"    ❌ Erreur vérification {model_name} / {period_type} : {e}")
        traceback.print_exc()
        return None
    finally:
        for n in ("hindcast_acc", "obs_for_periods", "t_low_cv", "t_up_cv",
                  "pBN", "pNN", "pAN", "obs_cat", "ev_BN", "ev_NN", "ev_AN",
                  "rpss", "bss_BN", "bss_NN", "bss_AN", "roc_BN", "roc_AN",
                  "corr", "rmse"):
            if n in dir():
                try:
                    del locals()[n]
                except Exception:
                    pass
        plt.close("all")
        gc.collect()
        print_memory_usage(f"after {model}/{period_type}")


# ============================================================
# 8. MULTI-MODEL VERIFICATION
# ============================================================
def verify_multimodel_for_period_type(
    saved_scores, period_type, periods_df,
    gdf_countries, gdf_regions, mask_zone=None,
    output_root=VERIFICATION_DIR,
):
    """
    Build a multi-model ensemble by averaging per-model tercile probabilities
    and recompute all scores against the same CHIRPS observation.

    Parameters
    ----------
    saved_scores : dict {model: path_to_scores_nc}
    """
    available = [m for m, p in saved_scores.items() if p is not None and os.path.isfile(p)]
    if not available:
        print(f"  ⚠ Aucun modèle disponible pour multi-modèle ({period_type})")
        return None

    print(f"\n  ★ MULTI-MODEL — {period_type} ({len(available)} modèles : {available})")

    datasets = {m: xr.open_dataset(saved_scores[m]).load() for m in available}
    ref = datasets[available[0]]

    # average probabilities, renormalize
    pBN = sum(datasets[m]["pBN"] for m in available) / len(available)
    pNN = sum(datasets[m]["pNN"] for m in available) / len(available)
    pAN = sum(datasets[m]["pAN"] for m in available) / len(available)
    total = pBN + pNN + pAN
    total = xr.where(total == 0, 1, total)
    pBN, pNN, pAN = pBN / total, pNN / total, pAN / total

    # observed category is identical across models (uses LOO CHIRPS)
    obs_cat = ref["obs_cat"].astype("int8")
    ev_BN = (obs_cat == 0).astype(float)
    ev_NN = (obs_cat == 1).astype(float)
    ev_AN = (obs_cat == 2).astype(float)

    _, _, rpss   = compute_rpss_maps(pBN, pNN, pAN, obs_cat)
    _, _, bss_BN = compute_bss_maps(pBN, ev_BN)
    _, _, bss_NN = compute_bss_maps(pNN, ev_NN)
    _, _, bss_AN = compute_bss_maps(pAN, ev_AN)
    roc_BN = compute_roc_area_maps(pBN, ev_BN)
    roc_AN = compute_roc_area_maps(pAN, ev_AN)

    # ens-mean skill: average per-model corr/rmse (the only available proxy)
    corr = sum(datasets[m]["corr_ens"] for m in available) / len(available)
    rmse = sum(datasets[m]["rmse_ens"] for m in available) / len(available)

    mm_root = os.path.join(output_root, "multi_model")
    scores_path = os.path.join(mm_root, "scores", f"{period_type}_scores.nc")
    save_verification_dataset(
        pBN, pNN, pAN, obs_cat,
        rpss, bss_BN, bss_NN, bss_AN,
        roc_BN, roc_AN, corr, rmse,
        model="multi_model", period_type=period_type, output_path=scores_path,
    )

    maps_dir = os.path.join(mm_root, "maps", period_type)
    os.makedirs(maps_dir, exist_ok=True)

    n_periods = rpss.sizes["period"]
    for i in range(n_periods):
        p_start = pd.Timestamp(rpss["period_start"].isel(period=i).values)
        p_end   = pd.Timestamp(rpss["period_end"].isel(period=i).values)
        sstr, estr = p_start.strftime("%Y-%m-%d"), p_end.strftime("%Y-%m-%d")
        label = format_period_label(sstr, estr, period_type)
        tag = _safe_filename(f"{period_type}_{sstr}_to_{estr}")

        plot_skill_map(rpss.isel(period=i), label, "RPSS", "★ MULTI-MODEL",
                       os.path.join(maps_dir, f"RPSS_{tag}.png"),
                       gdf_countries, gdf_regions, mask_zone=mask_zone,
                       cmap="RdBu_r", levels=_LEVELS_SKILL)
        plot_skill_map(bss_BN.isel(period=i), label, "BSS (BN)", "★ MULTI-MODEL",
                       os.path.join(maps_dir, f"BSS_BN_{tag}.png"),
                       gdf_countries, gdf_regions, mask_zone=mask_zone,
                       cmap="RdBu_r", levels=_LEVELS_SKILL)
        plot_skill_map(bss_AN.isel(period=i), label, "BSS (AN)", "★ MULTI-MODEL",
                       os.path.join(maps_dir, f"BSS_AN_{tag}.png"),
                       gdf_countries, gdf_regions, mask_zone=mask_zone,
                       cmap="RdBu_r", levels=_LEVELS_SKILL)
        plot_skill_map(roc_BN.isel(period=i), label, "ROC area (BN)", "★ MULTI-MODEL",
                       os.path.join(maps_dir, f"ROC_BN_{tag}.png"),
                       gdf_countries, gdf_regions, mask_zone=mask_zone,
                       cmap="YlOrRd", levels=_LEVELS_ROC)
        plot_skill_map(roc_AN.isel(period=i), label, "ROC area (AN)", "★ MULTI-MODEL",
                       os.path.join(maps_dir, f"ROC_AN_{tag}.png"),
                       gdf_countries, gdf_regions, mask_zone=mask_zone,
                       cmap="YlOrRd", levels=_LEVELS_ROC)
        plot_skill_map(corr.isel(period=i), label, "Ens-mean correlation", "★ MULTI-MODEL",
                       os.path.join(maps_dir, f"CORR_{tag}.png"),
                       gdf_countries, gdf_regions, mask_zone=mask_zone,
                       cmap="RdBu_r", levels=_LEVELS_CORR)
        plt.close("all")

    # reliability for multi-model
    rel_dir = os.path.join(mm_root, "reliability", period_type)
    os.makedirs(rel_dir, exist_ok=True)
    bin_centers, mf_BN, of_BN, cnt_BN = compute_reliability_per_category(pBN, ev_BN, mask_zone)
    _,            mf_NN, of_NN, cnt_NN = compute_reliability_per_category(pNN, ev_NN, mask_zone)
    _,            mf_AN, of_AN, cnt_AN = compute_reliability_per_category(pAN, ev_AN, mask_zone)
    for i in range(n_periods):
        p_start = pd.Timestamp(rpss["period_start"].isel(period=i).values)
        p_end   = pd.Timestamp(rpss["period_end"].isel(period=i).values)
        sstr, estr = p_start.strftime("%Y-%m-%d"), p_end.strftime("%Y-%m-%d")
        label = format_period_label(sstr, estr, period_type)
        tag = _safe_filename(f"{period_type}_{sstr}_to_{estr}")
        plot_reliability_diagram_three_cat(
            bin_centers,
            {"BN": mf_BN[i], "NN": mf_NN[i], "AN": mf_AN[i]},
            {"BN": of_BN[i], "NN": of_NN[i], "AN": of_AN[i]},
            {"BN": cnt_BN[i], "NN": cnt_NN[i], "AN": cnt_AN[i]},
            period_label=label, model_name="★ MULTI-MODEL",
            output_path=os.path.join(rel_dir, f"reliability_{tag}.png"),
        )

    # summary
    summary_path = os.path.join(mm_root, "scores", f"{period_type}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"MULTI-MODEL ({len(available)} models: {available}) — {period_type}\n")
        f.write(f"Years: {HINDCAST_YEARS[0]}-{HINDCAST_YEARS[-1]}, obs: CHIRPS v2.0\n\n")
        f.write(f"{'Period':40s}  {'<RPSS>':>8s}  {'%RPSS>0':>8s}  "
                f"{'<ROC_BN>':>9s}  {'<ROC_AN>':>9s}  {'<CORR>':>8s}\n")
        for i in range(n_periods):
            p_start = pd.Timestamp(rpss["period_start"].isel(period=i).values)
            p_end   = pd.Timestamp(rpss["period_end"].isel(period=i).values)
            tag = f"{p_start:%Y-%m-%d}_to_{p_end:%Y-%m-%d}"
            m_rpss, pos = _domain_stats(rpss.isel(period=i), mask_zone)
            m_rbn, _    = _domain_stats(roc_BN.isel(period=i), mask_zone)
            m_ran, _    = _domain_stats(roc_AN.isel(period=i), mask_zone)
            m_corr, _   = _domain_stats(corr.isel(period=i), mask_zone)
            f.write(f"{tag:40s}  {m_rpss:8.3f}  {pos*100:7.1f}%  "
                    f"{m_rbn:9.3f}  {m_ran:9.3f}  {m_corr:8.3f}\n")
    print(f"  📝 {summary_path}")

    for m in datasets:
        safe_close_xarray(datasets[m])
    gc.collect()
    return scores_path
