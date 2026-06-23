"""
Entry point for the C3S hindcast verification pipeline (CHIRPS reference).

Usage:
    python run_verification.py

For each model and each period type (decade / month / season):
  - Loads the hindcast GRIB and CHIRPS daily NetCDF
  - Computes RPSS, BSS (per tercile), ROC area, reliability diagram,
    ensemble-mean correlation and RMSE — all under LOO cross-validation
  - Saves NetCDF scores, PNG skill maps, reliability diagrams, summary TXT

Then builds the multi-model average and verifies it the same way.
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
    MODELS, MODEL_NAMES, DATADIR, MAP_EXTENT, MAX_END_DATE,
    INIT_YEAR, INIT_MONTH_STR,
)
from eccas_s2s.core.data import (
    load_shapefile, create_geographic_mask, open_grib_dataset,
    safe_close_xarray, print_memory_usage,
)
from eccas_s2s.core.processing import (
    decumulate_precip_to_mm, generate_calendar_periods_from_forecast,
)
from eccas_s2s.validate.verification import (
    CHIRPS_PATH, HINDCAST_YEARS, VERIFICATION_DIR,
    load_chirps_aligned,
    verify_model_for_period_type, verify_multimodel_for_period_type,
)


def _count_pngs(base_dir):
    total = 0
    for root, _, files in os.walk(base_dir):
        total += sum(1 for f in files if f.endswith(".png"))
    return total


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  VÉRIFICATION DES PRÉVISIONS HINDCASTS C3S vs CHIRPS")
    print("  Région CEEAC — multi-modèle")
    print("=" * 72)
    print_memory_usage("début vérification")

    # ---- shapefile ----
    print("\n📍 Chargement shapefile CEEAC...")
    gdf_countries, gdf_regions, country_geometry = load_shapefile()

    # ---- find a sample hindcast file to anchor the grid and derive periods ----
    sample_path = None
    for m in MODELS:
        fp = os.path.join(DATADIR, f"{m}_hindcast_1993_2016_{INIT_MONTH_STR}.grib")
        if os.path.isfile(fp):
            sample_path = fp
            sample_model = m
            break
    if sample_path is None:
        raise FileNotFoundError(
            f"Aucun hindcast trouvé dans {DATADIR}/*_hindcast_1993_2016_{INIT_MONTH_STR}.grib"
        )

    print(f"\n📂 Hindcast d'amorçage : {sample_path}")
    sample = open_grib_dataset(sample_path)
    sample_daily = decumulate_precip_to_mm(sample, clip_negative=True)

    lons = sample["longitude"].values
    lats = sample["latitude"].values
    print(f"  Grille modèle : {len(lats)} × {len(lons)}  "
          f"({lats.min():.1f}°–{lats.max():.1f}°N, {lons.min():.1f}°–{lons.max():.1f}°E)")

    # ---- mask ----
    mask_zone = create_geographic_mask(lons, lats, country_geometry)

    # ---- periods (use single-year first slice of hindcast valid_time) ----
    sample_one_year = sample_daily.isel(time=0)
    print("\n📅 Génération des périodes calendaires...")
    periods = generate_calendar_periods_from_forecast(
        sample_one_year,
        max_end_date=MAX_END_DATE,
        season_window_months=3,
        include_partial_first=True,
        include_partial_last=False,
    )
    period_collections = {
        "decade": periods["decades"],
        "month":  periods["months"],
        "season": periods["seasons"],
    }
    for k, v in period_collections.items():
        print(f"  {k:>6s} : {len(v)} périodes")

    safe_close_xarray(sample)
    safe_close_xarray(sample_daily)
    del sample, sample_daily, sample_one_year
    gc.collect()

    # ---- CHIRPS aligned + accumulated, once per period type ----
    print("\n🌧  Pré-traitement CHIRPS sur la grille modèle...")
    grid_template = open_grib_dataset(sample_path)
    chirps_by_ptype = {}
    for ptype, periods_df in period_collections.items():
        if periods_df is None or len(periods_df) == 0:
            continue
        print(f"\n  ▶ CHIRPS / {ptype}")
        chirps_by_ptype[ptype] = load_chirps_aligned(
            CHIRPS_PATH, grid_template, HINDCAST_YEARS, periods_df
        )
    safe_close_xarray(grid_template)
    del grid_template
    gc.collect()
    print_memory_usage("après chargement CHIRPS")

    # ---- per-model verification ----
    os.makedirs(VERIFICATION_DIR, exist_ok=True)
    saved_paths = {ptype: {} for ptype in period_collections}
    failed = []

    for ptype, periods_df in period_collections.items():
        if periods_df is None or len(periods_df) == 0:
            continue
        if ptype not in chirps_by_ptype:
            continue
        chirps_acc = chirps_by_ptype[ptype]

        print(f"\n{'='*72}")
        print(f"  PÉRIODE : {ptype.upper()}  ({len(periods_df)} périodes)")
        print(f"{'='*72}")

        for model in MODELS:
            try:
                path = verify_model_for_period_type(
                    model=model, period_type=ptype, periods_df=periods_df,
                    chirps_acc=chirps_acc,
                    gdf_countries=gdf_countries, gdf_regions=gdf_regions,
                    mask_zone=mask_zone, output_root=VERIFICATION_DIR,
                )
                saved_paths[ptype][model] = path
                if path is None:
                    failed.append((model, ptype))
            except Exception as e:
                print(f"❌ {model} / {ptype} : erreur inattendue {e}")
                traceback.print_exc()
                saved_paths[ptype][model] = None
                failed.append((model, ptype))
            finally:
                plt.close("all")
                gc.collect()

        # ---- multi-model for this period type ----
        try:
            verify_multimodel_for_period_type(
                saved_scores=saved_paths[ptype],
                period_type=ptype, periods_df=periods_df,
                gdf_countries=gdf_countries, gdf_regions=gdf_regions,
                mask_zone=mask_zone, output_root=VERIFICATION_DIR,
            )
        except Exception as e:
            print(f"❌ multi-modèle / {ptype} : {e}")
            traceback.print_exc()
        finally:
            plt.close("all")
            gc.collect()

    # ---- summary ----
    print("\n" + "=" * 72)
    print("  RÉSUMÉ DE LA VÉRIFICATION")
    print("=" * 72)
    n_png = _count_pngs(VERIFICATION_DIR)
    print(f"\n  📁 Sortie : {VERIFICATION_DIR}")
    print(f"  📊 Figures générées : {n_png}")
    for ptype, paths in saved_paths.items():
        ok = [m for m, p in paths.items() if p is not None]
        ko = [m for m, p in paths.items() if p is None]
        print(f"  {ptype:>6s} : {len(ok)} OK  {('— échec: ' + ', '.join(ko)) if ko else ''}")
    if failed:
        print(f"\n  ⚠ Échecs : {failed}")
    print("\n✅ Vérification terminée.")
