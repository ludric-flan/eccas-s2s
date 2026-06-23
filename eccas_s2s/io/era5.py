"""
ERA5 / ERA5 monthly-means reanalysis acquisition from the Copernicus CDS.

Two roles in eccas-s2s:
  * driver monitoring  -- SST, mean sea-level pressure, OLR, winds, ...
  * predictors         -- SST indices and gridded fields for statistical models.

Requires CDS credentials in ``~/.cdsapirc`` (same as ``io.c3s``). Single-level and
pressure-level monthly means are supported.
"""
import os

from eccas_s2s.io.c3s import _area_to_cds  # shared (lon_min,lon_max,lat_min,lat_max)->[N,W,S,E]

#: friendly variable name -> CDS variable name (single-level fields).
ERA5_VARIABLES = {
    "SST":   "sea_surface_temperature",
    "SLP":   "mean_sea_level_pressure",
    "OLR":   "top_net_thermal_radiation",
    "TEMP":  "2m_temperature",
    "PRCP":  "total_precipitation",
    "U10":   "10m_u_component_of_wind",
    "V10":   "10m_v_component_of_wind",
    # pressure-level fields (use level_type="pressure", pressure_level=...)
    "U":     "u_component_of_wind",
    "V":     "v_component_of_wind",
    "Q":     "specific_humidity",
}

_SINGLE = "reanalysis-era5-single-levels-monthly-means"
_PRESSURE = "reanalysis-era5-pressure-levels-monthly-means"


def build_era5_request(variables, years, months, area,
                       level_type="single", pressure_level=None):
    """
    Build the ``(dataset, request)`` pair for an ERA5 monthly-means retrieve (pure).

    ``variables`` is a list of friendly keys in ERA5_VARIABLES; ``months`` a list of
    1-12 ints; ``years`` a list of year ints/strings. For ``level_type="pressure"``
    pass ``pressure_level`` (e.g. "850").
    """
    cds_vars = [ERA5_VARIABLES.get(v, v) for v in variables]
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": cds_vars,
        "year": [str(y) for y in years],
        "month": [f"{int(m):02d}" for m in months],
        "time": ["00:00"],
        "data_format": "netcdf",
        "area": _area_to_cds(area),
    }
    if level_type == "pressure":
        if pressure_level is None:
            raise ValueError("pressure_level is required for level_type='pressure'")
        request["pressure_level"] = [str(pressure_level)]
        return _PRESSURE, request
    return _SINGLE, request


def download_era5(variables, years, months, area, dir_to_save, name,
                  level_type="single", pressure_level=None, force_download=False):
    """Download an ERA5 monthly-means subset to NetCDF and return its path."""
    import cdsapi  # lazy import: package imports without CDS installed

    dataset, request = build_era5_request(
        variables, years, months, area,
        level_type=level_type, pressure_level=pressure_level,
    )
    y0, y1 = request["year"][0], request["year"][-1]
    dest = os.path.join(dir_to_save, f"era5_{name}_{y0}_{y1}.nc")
    if os.path.isfile(dest) and os.path.getsize(dest) > 0 and not force_download:
        print(f"  ✔ cached: {dest}")
        return dest

    os.makedirs(dir_to_save, exist_ok=True)
    print(f"  ⬇ CDS retrieve: ERA5 {variables} {y0}-{y1} -> {dest}")
    cdsapi.Client().retrieve(dataset, request, dest)
    print(f"  ✔ saved: {dest}")
    return dest
