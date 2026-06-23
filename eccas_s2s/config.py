"""
Global configuration for the C3S multi-model seasonal forecast pipeline.
Edit this file to adapt the pipeline to a new forecast cycle.
"""
import os
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# PATHS
# ============================================================
DATADIR = "grib_files"
OUTPUT_DIR = os.path.join(DATADIR, "PREVISIONS_C3S_JUIN2026")

SHAPEFILE_PATH = "/home/ludric/Desktop/Donnees_calibration_c3s/pays_cceac_shapefile/ceeac_pays_adm1.shp"
LOGO_PATH = "/home/ludric/Desktop/Donnees_calibration_c3s/pays_cceac_shapefile/logo_creuw_eccas_undrr.png"

# ============================================================
# FORECAST CYCLE
# ============================================================
INIT_YEAR = 2026
INIT_MONTH_STR = "06_01"   # used in GRIB filenames: {model}_seasonal_forecast_{YEAR}_{INIT_MONTH_STR}.grib
MAX_END_DATE = "2026-11-30"

# ============================================================
# DOMAIN
# ============================================================
MAP_EXTENT = [4, 36, -21, 25]   # [lon_min, lon_max, lat_min, lat_max]

# ============================================================
# MODELS
# ============================================================
MODELS = ["ecmwf", "bom", "cmcc", "dwd", "eccc", "meteo_france", "ukmo"]  # "ncep" excluded

MODEL_NAMES = {
    "bom":          "BoM ACCESS-S2",
    "cmcc":         "CMCC SPS3.5",
    "dwd":          "DWD GCFS2.1",
    "eccc":         "ECCC GEM5-NEMO",
    "ecmwf":        "ECMWF SEAS5.1",
    "meteo_france": "Météo-France 8",
    "ncep":         "NCEP CFSv2",
    "ukmo":         "UKMO GloSea6",
}

# ============================================================
# PRECIPITATION THRESHOLDS (mm) FOR EXCEEDANCE PROBABILITIES
# ============================================================
PRECIP_THRESHOLDS = [5, 10, 20, 50, 100, 200, 300, 400, 500, 600]

# ============================================================
# COLORMAPS
# ============================================================
TP_COLORS = [
    (153/255., 51/255.,  0),
    (204/255., 136/255., 0),
    (1,        213/255., 0),
    (1,        238/255., 153/255.),
    (1,        1,        1),
    (204/255., 1,        102/255.),
    (42/255.,  1,        0),
    (0,        153/255., 51/255.),
    (0,        102/255., 102/255.),
]

_PROB_CMAP_COLORS = [
    (0.00, "#ffffff"),
    (0.10, "#d9f0d3"),
    (0.30, "#1a9641"),
    (0.50, "#ffff00"),
    (0.70, "#fdae61"),
    (1.00, "#d7191c"),
]
PROB_CMAP = LinearSegmentedColormap.from_list("prob_map", _PROB_CMAP_COLORS)
