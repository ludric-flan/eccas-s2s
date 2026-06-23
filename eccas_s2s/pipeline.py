"""
Per-model and multi-model pipeline orchestration.
"""
import os
import gc
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eccas_s2s.config import (
    DATADIR, OUTPUT_DIR, MODEL_NAMES, PRECIP_THRESHOLDS, MAP_EXTENT,
    INIT_YEAR, INIT_MONTH_STR, TP_COLORS, PROB_CMAP,
)
from eccas_s2s.core.data import print_memory_usage, safe_close_xarray, open_grib_dataset
from eccas_s2s.core.processing import (
    decumulate_precip_to_mm, compute_period_products, compute_all_probabilities,
    compute_exceedance_probabilities, build_light_products_dataset,
    save_light_products_dataset, load_light_products_dataset,
)
from eccas_s2s.viz.plotting import (
    get_levels_for_period, format_period_label,
    plot_all_anomalies, plot_all_dispersion, plot_all_probability_maps,
    plot_all_exceedance_maps, plot_multimodel_panel, plot_multimodel_tercile_panel,
    plot_single_anomaly_map, plot_tercile_summary, plot_single_probability_map,
    plot_exceedance_probability_map,
)


def get_period_output_subdir(period_type):
    return {
        "decade": "decades",
        "month":  "months",
        "season": "seasons",
        "custom": "custom",
    }.get(period_type, period_type)


# ============================================================
# SINGLE-MODEL PIPELINE
# ============================================================
def process_single_model_to_disk(
    model, period_collections, init_date_str,
    gdf_countries, gdf_regions,
    mask_zone=None, extent=None, output_root=OUTPUT_DIR,
):
    """
    Process one model: read GRIB, decumulate, compute products and figures,
    save light products to disk, free memory.

    Returns a dict {period_type: path_to_light_nc} or None on total failure.
    """
    model_name = MODEL_NAMES.get(model, model)
    print(f"\n{'='*70}")
    print(f"  🔄 TRAITEMENT DU MODÈLE : {model_name} ({model})")
    print(f"{'='*70}")
    print_memory_usage(f"{model} - début")

    hindcast_path = os.path.join(DATADIR, f"{model}_hindcast_1993_2016_{INIT_MONTH_STR}.grib")
    forecast_path = os.path.join(DATADIR, f"{model}_seasonal_forecast_{INIT_YEAR}_{INIT_MONTH_STR}.grib")

    if not os.path.isfile(hindcast_path):
        print(f"    ⚠ Hindcast manquant : {hindcast_path}")
        return None
    if not os.path.isfile(forecast_path):
        print(f"    ⚠ Forecast manquant : {forecast_path}")
        return None

    model_root = os.path.join(output_root, model)
    os.makedirs(os.path.join(model_root, "_light_products"), exist_ok=True)
    os.makedirs(os.path.join(model_root, "dispersion"),      exist_ok=True)
    os.makedirs(os.path.join(model_root, "exceedance"),      exist_ok=True)

    try:
        print(f"    📂 Lecture hindcast : {hindcast_path}")
        hindcast_raw = open_grib_dataset(hindcast_path)
        print(f"    📂 Lecture forecast : {forecast_path}")
        forecast_raw = open_grib_dataset(forecast_path)
    except Exception as e:
        print(f"    ❌ Échec lecture GRIB pour {model_name} : {e}")
        traceback.print_exc()
        return None

    print_memory_usage(f"{model} - après ouverture")

    try:
        hindcast_daily = decumulate_precip_to_mm(hindcast_raw, clip_negative=True)
        forecast_daily = decumulate_precip_to_mm(forecast_raw, clip_negative=True)
    except Exception as e:
        print(f"    ❌ Échec décumulation pour {model_name} : {e}")
        traceback.print_exc()
        safe_close_xarray(hindcast_raw)
        safe_close_xarray(forecast_raw)
        return None

    print_memory_usage(f"{model} - après décumulation")

    saved_files = {}

    for period_type, periods_df in period_collections.items():
        if periods_df is None or len(periods_df) == 0:
            continue

        print(f"\n    📊 TYPE DE PÉRIODE : {period_type.upper()} ({len(periods_df)} périodes)")

        anomaly_dir = os.path.join(model_root, get_period_output_subdir(period_type))
        os.makedirs(anomaly_dir, exist_ok=True)

        try:
            products = compute_period_products(
                hindcast_daily, forecast_daily, periods_df, var="tp_daily"
            )
            print_memory_usage(f"{model} - {period_type} - après products")

            probs, thresholds = compute_all_probabilities(
                products["hindcast_acc"], products["forecast_acc"]
            )
            print_memory_usage(f"{model} - {period_type} - après probs")

            exceed_probs = compute_exceedance_probabilities(
                products["forecast_acc"], PRECIP_THRESHOLDS
            )
            print_memory_usage(f"{model} - {period_type} - après exceed_probs")

            plot_all_anomalies(
                products=products, periods_df=periods_df, period_type=period_type,
                init_date_str=init_date_str, gdf_countries=gdf_countries,
                gdf_regions=gdf_regions, model_name=model_name,
                mask_zone=mask_zone, extent=extent,
                output_dir=anomaly_dir, save=True, show=False,
            )
            plot_all_dispersion(
                products=products, period_type=period_type,
                init_date_str=init_date_str, gdf_countries=gdf_countries,
                gdf_regions=gdf_regions, model_name=model_name,
                mask_zone=mask_zone, extent=extent,
                output_dir=os.path.join(model_root, "dispersion"),
                save=True, show=False,
            )
            plot_all_probability_maps(
                probs=probs, products=products, periods_df=periods_df,
                period_type=period_type, init_date_str=init_date_str,
                gdf_countries=gdf_countries, gdf_regions=gdf_regions,
                model_name=model_name, extent=extent,
                output_dir=model_root, save=True,
            )
            plot_all_exceedance_maps(
                exceed_probs=exceed_probs, products=products,
                period_type=period_type, init_date_str=init_date_str,
                gdf_countries=gdf_countries, gdf_regions=gdf_regions,
                model_name=model_name, mask_zone=mask_zone, extent=extent,
                output_dir=os.path.join(model_root, "exceedance"),
                save=True,
            )

            ds_light = build_light_products_dataset(
                products=products, probs=probs, exceed_probs=exceed_probs,
                model=model, period_type=period_type,
            )
            light_file = os.path.join(model_root, "_light_products",
                                      f"{period_type}_light.nc")
            save_light_products_dataset(ds_light, light_file)
            saved_files[period_type] = light_file

        except Exception as e:
            print(f"    ❌ Échec période '{period_type}' pour {model_name} : {e}")
            traceback.print_exc()
        finally:
            for obj_name in ["products", "probs", "thresholds", "exceed_probs", "ds_light"]:
                if obj_name in dir():
                    try:
                        del locals()[obj_name]
                    except Exception:
                        pass
            plt.close("all")
            gc.collect()
            print_memory_usage(f"{model} - {period_type} - après nettoyage")

    safe_close_xarray(hindcast_raw)
    safe_close_xarray(forecast_raw)
    safe_close_xarray(hindcast_daily)
    safe_close_xarray(forecast_daily)
    del hindcast_raw, forecast_raw, hindcast_daily, forecast_daily
    plt.close("all")
    gc.collect()
    print_memory_usage(f"{model} - fin")
    print(f"  ✅ Modèle {model_name} traité ({len(saved_files)} types de période sauvegardés).")
    return saved_files if saved_files else None


# ============================================================
# MULTI-MODEL PIPELINE
# ============================================================
def build_multimodel_from_light_products(
    saved_registry, period_collections, init_date_str,
    gdf_countries, gdf_regions,
    mask_zone=None, extent=None, output_root=OUTPUT_DIR,
):
    """
    Reload per-model light products and produce multi-model combination maps.
    dry_mask: a pixel is considered dry if at least half of the models mark it as dry.
    """
    print(f"\n{'#'*70}")
    print("# CONSTRUCTION DES PRODUITS MULTI-MODÈLES")
    print(f"{'#'*70}")

    mm_root = os.path.join(output_root, "multi_model")
    for sub in ("anomalies", "probabilities", "exceedance"):
        os.makedirs(os.path.join(mm_root, sub), exist_ok=True)
        # Standalone single maps for the multi-model mean (alongside panels)
        os.makedirs(os.path.join(mm_root, sub, "single"), exist_ok=True)

    mm_name = "Multi-Model Mean"

    for period_type, periods_df in period_collections.items():
        available_models = [
            m for m in saved_registry
            if saved_registry[m] is not None and period_type in saved_registry[m]
        ]
        if not available_models:
            print(f"  ⚠ Aucun modèle disponible pour '{period_type}'")
            continue

        n_avail = len(available_models)
        print(f"\n  📂 {period_type} — {n_avail} modèle(s) : {available_models}")

        light_datasets = {}
        for m in available_models:
            try:
                light_datasets[m] = load_light_products_dataset(saved_registry[m][period_type])
            except Exception as e:
                print(f"  ⚠ Impossible de charger les produits légers de {m} : {e}")

        if not light_datasets:
            continue

        print_memory_usage(f"multi-model - {period_type} - après chargement")

        ref_ds    = light_datasets[available_models[0]]
        n_periods = ref_ds.dims["period"]
        lons      = ref_ds["longitude"].values
        lats      = ref_ds["latitude"].values

        for i in range(n_periods):
            p_start   = pd.Timestamp(ref_ds["period_start"].isel(period=i).values)
            p_end     = pd.Timestamp(ref_ds["period_end"].isel(period=i).values)
            start_str = p_start.strftime("%Y-%m-%d")
            end_str   = p_end.strftime("%Y-%m-%d")
            label     = format_period_label(start_str, end_str, period_type)
            prefix    = f"{period_type}_{start_str}_to_{end_str}"
            print(f"    📊 Multi-modèle {period_type}: {start_str} → {end_str}")
            mm_dry = None  # set by the tercile section; reused for single prob maps

            # ---- 1. Anomalie moyenne d'ensemble ----
            try:
                anom_stack = [light_datasets[m]["ens_mean_anomaly"].isel(period=i).values
                              for m in light_datasets]
                anom_panel = {m: light_datasets[m]["ens_mean_anomaly"].isel(period=i).values
                              for m in light_datasets}
                mm_anom = np.nanmean(np.stack(anom_stack), axis=0)
                anom_panel["multi_model"] = mm_anom

                plot_multimodel_panel(
                    data_dict=anom_panel, lons=lons, lats=lats,
                    title_main="Ensemble Mean Anomaly", colorbar_label="Anomalie (mm)",
                    period_label=label, init_date_str=init_date_str,
                    gdf_countries=gdf_countries, gdf_regions=gdf_regions,
                    mask_zone=mask_zone, levels=get_levels_for_period(period_type),
                    colors_list=TP_COLORS, extend="both", extent=extent,
                    output_path=os.path.join(mm_root, "anomalies",
                                             f"mm_anomaly_{prefix}.png"),
                )

                # Standalone single map of the multi-model mean
                plot_single_anomaly_map(
                    data=mm_anom, lons=lons, lats=lats,
                    period_label=label, init_date_str=init_date_str,
                    gdf_countries=gdf_countries, gdf_regions=gdf_regions,
                    mask_zone=mask_zone, model_name=mm_name,
                    title="Ensemble Mean Anomaly",
                    tp_levels=get_levels_for_period(period_type), extent=extent,
                    output_path=os.path.join(mm_root, "anomalies", "single",
                                             f"mm_single_anomaly_{prefix}.png"),
                )
            except Exception as e:
                print(f"    ⚠ Panel anomalies {period_type} {start_str} : {e}")

            # ---- 2. Tercile summary ----
            try:
                bn_stack, nn_stack, an_stack, dry_stack = [], [], [], []
                tercile_panel = {}
                for m in light_datasets:
                    ds_m = light_datasets[m]
                    p_bn = ds_m["prob_BN"].isel(period=i).values
                    p_nn = ds_m["prob_NN"].isel(period=i).values
                    p_an = ds_m["prob_AN"].isel(period=i).values
                    dmk  = ds_m["dry_mask"].isel(period=i).values.astype(bool)
                    tercile_panel[m] = {"prob_BN": p_bn, "prob_NN": p_nn,
                                        "prob_AN": p_an, "dry_mask": dmk}
                    bn_stack.append(p_bn)
                    nn_stack.append(p_nn)
                    an_stack.append(p_an)
                    dry_stack.append(dmk)

                mm_bn  = np.nanmean(np.stack(bn_stack),  axis=0)
                mm_nn  = np.nanmean(np.stack(nn_stack),  axis=0)
                mm_an  = np.nanmean(np.stack(an_stack),  axis=0)

                # Majority-vote dry mask: dry if >= half of models say so
                dry_arr = np.stack(dry_stack, axis=0)
                mm_dry  = dry_arr.sum(axis=0) >= (n_avail / 2)

                total  = mm_bn + mm_nn + mm_an
                total  = np.where(total == 0, 1, total)
                mm_bn /= total
                mm_nn /= total
                mm_an /= total

                tercile_panel["multi_model"] = {"prob_BN": mm_bn, "prob_NN": mm_nn,
                                                "prob_AN": mm_an, "dry_mask": mm_dry}

                plot_multimodel_tercile_panel(
                    all_probs=tercile_panel, lons=lons, lats=lats,
                    period_label=label, init_date_str=init_date_str,
                    gdf_countries=gdf_countries, gdf_regions=gdf_regions,
                    mask_zone=mask_zone, extent=extent,
                    output_path=os.path.join(mm_root, "probabilities",
                                             f"mm_tercile_{prefix}.png"),
                )

                # Standalone single tercile summary of the multi-model mean
                plot_tercile_summary(
                    prob_BN=mm_bn, prob_NN=mm_nn, prob_AN=mm_an,
                    lons=lons, lats=lats, dry_mask=mm_dry,
                    period_label=label, init_date_str=init_date_str,
                    gdf_countries=gdf_countries, gdf_regions=gdf_regions,
                    mask_zone=mask_zone, model_name=mm_name, extent=extent,
                    output_path=os.path.join(mm_root, "probabilities", "single",
                                             f"mm_single_tercile_summary_{prefix}.png"),
                )
            except Exception as e:
                print(f"    ⚠ Panel tercile {period_type} {start_str} : {e}")

            # ---- 3. Individual probability panels ----
            prob_vars = {
                "prob_BN":           ("Prob. Below Normal",      plt.cm.YlOrBr),
                "prob_NN":           ("Prob. Near Normal",        plt.cm.YlGn),
                "prob_AN":           ("Prob. Above Normal",       plt.cm.YlGnBu),
                "prob_low20":        ("Prob. Lowest 20%",         plt.cm.OrRd),
                "prob_high20":       ("Prob. Highest 20%",        plt.cm.PuBu),
                "prob_exceed_median":("Prob. Exceeding Median",   plt.cm.RdYlGn),
            }
            for pvar, (title_main, cmap) in prob_vars.items():
                try:
                    stack_list  = [light_datasets[m][pvar].isel(period=i).values * 100.0
                                   for m in light_datasets]
                    panel_dict  = {m: light_datasets[m][pvar].isel(period=i).values * 100.0
                                   for m in light_datasets}
                    mm_prob_pct = np.nanmean(np.stack(stack_list), axis=0)
                    panel_dict["multi_model"] = mm_prob_pct
                    plot_multimodel_panel(
                        data_dict=panel_dict, lons=lons, lats=lats,
                        title_main=title_main, colorbar_label="Probability (%)",
                        period_label=label, init_date_str=init_date_str,
                        gdf_countries=gdf_countries, gdf_regions=gdf_regions,
                        mask_zone=mask_zone,
                        levels=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        cmap=cmap, extend="neither", extent=extent,
                        output_path=os.path.join(mm_root, "probabilities",
                                                 f"mm_{pvar}_{prefix}.png"),
                    )

                    # Standalone single map of the multi-model mean
                    plot_single_probability_map(
                        prob_data=mm_prob_pct / 100.0, lons=lons, lats=lats,
                        dry_mask=mm_dry, title=title_main, period_label=label,
                        init_date_str=init_date_str,
                        gdf_countries=gdf_countries, gdf_regions=gdf_regions,
                        mask_zone=mask_zone, model_name=mm_name, cmap=cmap,
                        extent=extent,
                        output_path=os.path.join(mm_root, "probabilities", "single",
                                                 f"mm_single_{pvar}_{prefix}.png"),
                    )
                except Exception as e:
                    print(f"    ⚠ Panel {pvar} {period_type} {start_str} : {e}")

            # ---- 4. Exceedance panels ----
            for thr in PRECIP_THRESHOLDS:
                try:
                    exc_stack = [
                        light_datasets[m]["prob_exceed_threshold"].sel(threshold=thr).isel(period=i).values * 100.0
                        for m in light_datasets
                    ]
                    exc_panel = {
                        m: light_datasets[m]["prob_exceed_threshold"].sel(threshold=thr).isel(period=i).values * 100.0
                        for m in light_datasets
                    }
                    mm_exc_pct = np.nanmean(np.stack(exc_stack), axis=0)
                    exc_panel["multi_model"] = mm_exc_pct
                    plot_multimodel_panel(
                        data_dict=exc_panel, lons=lons, lats=lats,
                        title_main=f"Probability of Rainfall ≥ {thr} mm",
                        colorbar_label="Probability (%)",
                        period_label=label, init_date_str=init_date_str,
                        gdf_countries=gdf_countries, gdf_regions=gdf_regions,
                        mask_zone=mask_zone, levels=np.arange(0, 101, 5),
                        cmap=PROB_CMAP, extend="neither", extent=extent,
                        output_path=os.path.join(mm_root, "exceedance",
                                                 f"mm_exceed_{thr}mm_{prefix}.png"),
                    )

                    # Standalone single map of the multi-model mean
                    plot_exceedance_probability_map(
                        prob=mm_exc_pct / 100.0, threshold_mm=thr,
                        lons=lons, lats=lats, period_label=label,
                        init_date_str=init_date_str,
                        gdf_countries=gdf_countries, gdf_regions=gdf_regions,
                        mask_zone=mask_zone, model_name=mm_name, extent=extent,
                        output_path=os.path.join(mm_root, "exceedance", "single",
                                                 f"mm_single_exceed_{thr}mm_{prefix}.png"),
                    )
                except Exception as e:
                    print(f"    ⚠ Panel exceedance {thr}mm {period_type} {start_str} : {e}")

            plt.close("all")

        for m in light_datasets:
            safe_close_xarray(light_datasets[m])
        del light_datasets
        gc.collect()
        print_memory_usage(f"multi-model - {period_type} - après nettoyage")

    print(f"\n✅ Produits multi-modèles générés dans : {mm_root}")
