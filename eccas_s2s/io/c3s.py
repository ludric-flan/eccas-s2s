"""
C3S seasonal forecast acquisition from the Copernicus Climate Data Store (CDS).

Requires CDS credentials in ``~/.cdsapirc`` (see the CDS "API how-to"). Unlike the
public IRIDL/NMME server, CDS needs an account + access token.

We use the ``seasonal-original-single-levels`` dataset (the daily-step product), which
is the same source the current pipeline downloads. Daily steps let us build decade /
month / season accumulations downstream.

Area convention
---------------
Throughout eccas-s2s an area is ``(lon_min, lon_max, lat_min, lat_max)`` (matching
``config.MAP_EXTENT``). The CDS API instead wants ``[North, West, South, East]``;
``_area_to_cds`` does the conversion so callers never deal with two conventions.
"""
import os

#: originating centre -> default system id (C3S seasonal systems).
C3S_CENTRE_SYSTEMS = {
    "ecmwf": "51",
    "cmcc": "4",
    "dwd": "22",
    "eccc": "5",
    "jma": "4",
    "meteo_france": "9",
    "ncep": "2",
    "ukmo": "610",
    "bom": "2",
}

#: friendly variable name -> CDS variable name.
C3S_VARIABLES = {
    "PRCP": "total_precipitation",
    "TEMP": "2m_temperature",
}

#: daily lead-time steps in hours (24h .. 216 days), as the current pipeline uses.
DEFAULT_LEADTIME_HOURS = [str(h) for h in range(24, 5184, 24)]


def _area_to_cds(area):
    """(lon_min, lon_max, lat_min, lat_max) -> CDS [North, West, South, East]."""
    lon_min, lon_max, lat_min, lat_max = area
    return [lat_max, lon_min, lat_min, lon_max]


def build_c3s_request(centre, variable, years, init_month, area,
                      system=None, init_day="01", leadtime_hours=None):
    """
    Build the ``(dataset, request)`` pair for a CDS retrieve call (pure, no network).

    ``years`` is a list of year strings: one entry for a forecast, many for a hindcast.
    """
    if variable not in C3S_VARIABLES:
        raise KeyError(f"variable must be one of {list(C3S_VARIABLES)}")
    if system is None:
        system = C3S_CENTRE_SYSTEMS.get(centre)
        if system is None:
            raise KeyError(f"unknown centre '{centre}'; pass system= explicitly")

    request = {
        "originating_centre": centre,
        "system": str(system),
        "variable": [C3S_VARIABLES[variable]],
        "year": [str(y) for y in years],
        "month": [f"{int(init_month):02d}"],
        "day": [str(init_day)],
        "leadtime_hour": list(leadtime_hours) if leadtime_hours else DEFAULT_LEADTIME_HOURS,
        "data_format": "grib",
        "area": _area_to_cds(area),
    }
    return "seasonal-original-single-levels", request


def download_c3s(centre, variable, years, init_month, area, dir_to_save,
                 system=None, init_day="01", leadtime_hours=None,
                 force_download=False, kind="forecast"):
    """
    Download a C3S seasonal subset to a GRIB file and return its path.

    ``kind`` ("forecast" / "hindcast") only affects the output filename.
    """
    import cdsapi  # imported lazily so the package imports without CDS installed

    dataset, request = build_c3s_request(
        centre, variable, years, init_month, area,
        system=system, init_day=init_day, leadtime_hours=leadtime_hours,
    )
    syst = request["system"]
    if kind == "hindcast":
        tag = f"{years[0]}_{years[-1]}"
    else:
        tag = str(years[0])
    fname = f"c3s_{centre}_{syst}_{variable}_{kind}_{tag}_{int(init_month):02d}.grib"
    dest = os.path.join(dir_to_save, fname)

    if os.path.isfile(dest) and os.path.getsize(dest) > 0 and not force_download:
        print(f"  ✔ cached: {dest}")
        return dest

    os.makedirs(dir_to_save, exist_ok=True)
    print(f"  ⬇ CDS retrieve: {centre} sys {syst} {variable} {kind} {tag} -> {dest}")
    cdsapi.Client().retrieve(dataset, request, dest)
    print(f"  ✔ saved: {dest}")
    return dest


def download_c3s_forecast(centre, variable, year, init_month, area, dir_to_save, **kw):
    """Download a single-year C3S forecast."""
    return download_c3s(centre, variable, [year], init_month, area, dir_to_save,
                        kind="forecast", **kw)


def download_c3s_hindcast(centre, variable, year_start, year_end, init_month,
                          area, dir_to_save, **kw):
    """Download a C3S hindcast over a year range (inclusive)."""
    years = [str(y) for y in range(int(year_start), int(year_end) + 1)]
    return download_c3s(centre, variable, years, init_month, area, dir_to_save,
                        kind="hindcast", **kw)
