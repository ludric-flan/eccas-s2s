"""
Rebuild ONLY the multi-model products from the per-model light products
already saved on disk (no GRIB re-download, no per-model reprocessing).

Usage:
    python rebuild_multimodel.py
"""
import os
import gc
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eccas_s2s.config import (
    MODELS, DATADIR, OUTPUT_DIR, MAP_EXTENT, MAX_END_DATE,
    INIT_YEAR, INIT_MONTH_STR,
)
from eccas_s2s.core.data import (
    load_shapefile, create_geographic_mask, open_grib_dataset, safe_close_xarray,
)
from eccas_s2s.core.processing import (
    decumulate_precip_to_mm, generate_calendar_periods_from_forecast,
    build_custom_period,
)
from eccas_s2s.pipeline import build_multimodel_from_light_products, get_period_output_subdir


if __name__ == "__main__":
    print("Rebuild multi-model products from existing light products")

    gdf_countries, gdf_regions, country_geometry = load_shapefile()

    # Grid from any available forecast sample
    sample_file = None
    for m in MODELS:
        fp = os.path.join(DATADIR, f"{m}_seasonal_forecast_{INIT_YEAR}_{INIT_MONTH_STR}.grib")
        if os.path.isfile(fp):
            sample_file = fp
            break
    if sample_file is None:
        raise FileNotFoundError("No forecast sample GRIB found.")

    sample = open_grib_dataset(sample_file)
    lons = sample["longitude"].values
    lats = sample["latitude"].values
    mask_zone = create_geographic_mask(lons, lats, country_geometry)

    sample_daily = decumulate_precip_to_mm(sample, clip_negative=True)
    periods = generate_calendar_periods_from_forecast(
        sample_daily, max_end_date=MAX_END_DATE, season_window_months=3,
        include_partial_first=True, include_partial_last=False,
    )
    custom_period = build_custom_period("2026-07-01", "2026-08-31",
                                        label="01st_july_31st_august")
    period_collections = {
        "decade": periods["decades"],
        "month":  periods["months"],
        "season": periods["seasons"],
        "custom": custom_period,
    }
    init_date_str = pd.Timestamp(sample["time"].values).strftime("%B %Y")
    safe_close_xarray(sample)
    safe_close_xarray(sample_daily)
    gc.collect()

    # Rebuild registry pointing at the existing light .nc files
    saved_registry = {}
    for m in MODELS:
        light_dir = os.path.join(OUTPUT_DIR, m, "_light_products")
        files = {}
        for ptype in period_collections:
            fp = os.path.join(light_dir, f"{ptype}_light.nc")
            if os.path.isfile(fp):
                files[ptype] = fp
        saved_registry[m] = files if files else None

    build_multimodel_from_light_products(
        saved_registry=saved_registry,
        period_collections=period_collections,
        init_date_str=init_date_str,
        gdf_countries=gdf_countries, gdf_regions=gdf_regions,
        mask_zone=mask_zone, extent=MAP_EXTENT, output_root=OUTPUT_DIR,
    )
    plt.close("all")
    print("Done.")
