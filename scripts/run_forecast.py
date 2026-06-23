"""
Entry point for the C3S multi-model seasonal forecast pipeline.

Usage:
    python run_forecast.py
"""
import gc
import traceback
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eccas_s2s.config import MODELS, MODEL_NAMES, DATADIR, OUTPUT_DIR, MAP_EXTENT, MAX_END_DATE, INIT_YEAR, INIT_MONTH_STR
from eccas_s2s.core.data import load_shapefile, create_geographic_mask, open_grib_dataset, print_memory_usage, safe_close_xarray
from eccas_s2s.core.processing import decumulate_precip_to_mm, generate_calendar_periods_from_forecast, build_custom_period
from eccas_s2s.pipeline import process_single_model_to_disk, build_multimodel_from_light_products

import os


def count_figures(base_dir):
    total = 0
    for root, dirs, files in os.walk(base_dir):
        total += sum(1 for f in files if f.endswith(".png"))
    return total


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  PIPELINE DE PRÉVISIONS SAISONNIÈRES C3S")
    print("  Multi-modèle - Région CEEAC")
    print("=" * 70)

    print_memory_usage("début script")

    # ---- Shapefile ----
    print("\n📍 Chargement shapefile CEEAC...")
    gdf_countries, gdf_regions, country_geometry = load_shapefile()

    # ---- Fichier sample pour la grille ----
    print("\n📂 Recherche d'un fichier forecast sample...")
    sample_file = None
    for m in MODELS:
        fp = os.path.join(DATADIR, f"{m}_seasonal_forecast_{INIT_YEAR}_{INIT_MONTH_STR}.grib")
        if os.path.isfile(fp):
            sample_file = fp
            break

    if sample_file is None:
        raise FileNotFoundError(
            f"Aucun fichier forecast trouvé dans {DATADIR} pour l'initialisation {INIT_YEAR}_{INIT_MONTH_STR}."
        )

    print(f"  Sample : {sample_file}")
    sample = open_grib_dataset(sample_file)
    lons   = sample["longitude"].values
    lats   = sample["latitude"].values
    print(f"  Grille : {len(lats)} × {len(lons)}  "
          f"({lats.min():.1f}°–{lats.max():.1f}°N, {lons.min():.1f}°–{lons.max():.1f}°E)")

    # ---- Masque géographique ----
    mask_zone = create_geographic_mask(lons, lats, country_geometry)

    # ---- Périodes calendaires ----
    print("\n📅 Génération des périodes calendaires...")
    sample_daily = decumulate_precip_to_mm(sample, clip_negative=True)
    periods = generate_calendar_periods_from_forecast(
        sample_daily,
        max_end_date=MAX_END_DATE,
        season_window_months=3,
        include_partial_first=True,
        include_partial_last=False,
    )
    print(f"  Décades : {len(periods['decades'])}")
    print(f"  Mois    : {len(periods['months'])}")
    print(f"  Saisons : {len(periods['seasons'])}")

    # ---- Période personnalisée (adapter si besoin) ----
    custom_period = build_custom_period("2026-07-01", "2026-08-31",
                                        label="01st_july_31st_august")

    period_collections = {
        "decade": periods["decades"],
        "month":  periods["months"],
        "season": periods["seasons"],
        "custom": custom_period,
    }

    # ---- Date d'initialisation ----
    init_date_str = pd.Timestamp(sample["time"].values).strftime("%B %Y")
    print(f"  Init    : {init_date_str}")

    safe_close_xarray(sample)
    safe_close_xarray(sample_daily)
    del sample, sample_daily
    gc.collect()

    # ---- Traitement modèle par modèle ----
    saved_registry = {}
    failed_models  = []

    for model in MODELS:
        try:
            result = process_single_model_to_disk(
                model=model,
                period_collections=period_collections,
                init_date_str=init_date_str,
                gdf_countries=gdf_countries,
                gdf_regions=gdf_regions,
                mask_zone=mask_zone,
                extent=MAP_EXTENT,
                output_root=OUTPUT_DIR,
            )
            saved_registry[model] = result
            if result is None:
                failed_models.append(model)
        except Exception as e:
            print(f"\n❌ Modèle {model} — erreur inattendue : {e}")
            traceback.print_exc()
            saved_registry[model] = None
            failed_models.append(model)
        finally:
            gc.collect()
            plt.close("all")
            print_memory_usage(f"après modèle {model}")

    # ---- Multi-modèle ----
    successful = [m for m in MODELS if saved_registry.get(m) is not None]
    print(f"\n📊 Modèles réussis : {successful}")
    if failed_models:
        print(f"⚠ Modèles échoués  : {failed_models}")

    if successful:
        build_multimodel_from_light_products(
            saved_registry=saved_registry,
            period_collections=period_collections,
            init_date_str=init_date_str,
            gdf_countries=gdf_countries,
            gdf_regions=gdf_regions,
            mask_zone=mask_zone,
            extent=MAP_EXTENT,
            output_root=OUTPUT_DIR,
        )
    else:
        print("⚠ Aucun modèle traité avec succès — pipeline multi-modèle ignoré.")

    gc.collect()
    plt.close("all")
    print_memory_usage("fin script")

    # ---- Résumé ----
    print("\n" + "=" * 70)
    print("  RÉSUMÉ FINAL")
    print("=" * 70)
    total_figs = count_figures(OUTPUT_DIR)
    print(f"\n  📊 Total figures générées : {total_figs}")
    print(f"  📁 Répertoire de sortie   : {OUTPUT_DIR}")
    for m in MODELS + ["multi_model"]:
        m_dir = os.path.join(OUTPUT_DIR, m)
        if os.path.isdir(m_dir):
            n     = count_figures(m_dir)
            label = "★ MULTI-MODEL" if m == "multi_model" else MODEL_NAMES.get(m, m)
            print(f"    {label:30s} : {n:5d} figures")

    if failed_models:
        print(f"\n  ⚠ Modèles avec erreurs : {failed_models}")
    print("\n✅ Pipeline terminé.")
