"""
Data I/O utilities: GRIB reading, shapefile loading, geographic masking.
"""
import os
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

try:
    import psutil
except ImportError:
    psutil = None

from eccas_s2s.config import SHAPEFILE_PATH


# ============================================================
# MEMORY UTILITIES
# ============================================================
def print_memory_usage(tag=""):
    if psutil is None:
        return
    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / (1024 ** 3)
    print(f"🧠 MEM [{tag}] RSS = {rss_gb:.2f} Go")


def safe_close_xarray(ds):
    try:
        ds.close()
    except Exception:
        pass


# ============================================================
# SHAPEFILE
# ============================================================
def load_shapefile(shapefile_path=SHAPEFILE_PATH):
    """Load CEEAC shapefile and return country/region GeoDataFrames and unified geometry."""
    gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    country_col = "adm0_a3" if "adm0_a3" in gdf.columns else gdf.columns[0]
    region_col  = "name"    if "name"    in gdf.columns else gdf.columns[1]

    try:
        gdf_countries = gdf.dissolve(by=country_col)
        gdf_regions   = gdf.dissolve(by=region_col)
    except Exception:
        gdf_countries = gdf.copy()
        gdf_regions   = gdf.copy()

    country_geometry = gdf.geometry.unary_union
    print(f"  ✓ Shapefile chargé : {len(gdf_countries)} pays, {len(gdf_regions)} régions")
    return gdf_countries, gdf_regions, country_geometry


# ============================================================
# GEOGRAPHIC MASK
# ============================================================
def create_geographic_mask(lons, lats, geometry):
    """Return a boolean (lat, lon) mask — True inside the CEEAC boundary."""
    print("  🗺  Création du masque géographique...")
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = [Point(x, y) for x, y in zip(lon_grid.flatten(), lat_grid.flatten())]
    mask = np.array([geometry.contains(pt) for pt in points]).reshape(lon_grid.shape)
    print(f"  ✓ Masque : {mask.sum()} points dans la zone sur {mask.size} total")
    return mask


# ============================================================
# GRIB I/O
# ============================================================
def standardize_spatial_coords(ds):
    """Rename lon/lat → longitude/latitude if needed."""
    rename = {}
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename["lon"] = "longitude"
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename["lat"] = "latitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def open_grib_dataset(path):
    """Open a GRIB file with cfgrib and standardize coordinate names."""
    ds = xr.open_dataset(path, engine="cfgrib")
    ds = standardize_spatial_coords(ds)
    return ds
