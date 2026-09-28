"""
Ocean drivers of Central African rainfall, as SST indices (workflow E3).

The indices are the cheapest predictor that exists: five numbers per season,
computed from a 2° SST analysis (ERSSTv5, 79 MB for 46 years) instead of the
gigabytes a field of predictors costs. They are also the ones the CAPC-AC
drivers framework names, and only those — the thirteen indices of the West
African chains include several basins whose link to Central Africa is not
established, and adding them would mostly add ways to find a correlation by
chance on 24 to 46 years.

============  =========================================  ===========================
Indice        Zone                                        Référence
============  =========================================  ===========================
``NINO34``    170°W–120°W, 5°S–5°N                        ENSO, définition standard
``DMI``       WTIO (50–70°E, 10°S–10°N) − SETIO           Saji et al. 1999 ; Jiang et
              (90–110°E, 10°S–0°)                         al. 2021 pour l'Afrique
                                                          centrale
``ATL3``      20°W–0°, 3°S–3°N                            Zebiak 1993 ; « Atlantic
                                                          Niño » de la littérature
``GG``        10°W–10°E, 5°S–5°N                          Golfe de Guinée, façade
                                                          océanique de la CEEAC
``SAOD``      NEP (20°W–10°E, 15°S–0°) − SWP              Nnamchi et al. 2011, repris
              (40°W–10°W, 40°S–25°S)                      par l'étude Afrique centrale
============  =========================================  ===========================

Two conventions matter as much as the boxes:

* an index is a **standardised anomaly**: the seasonal mean of the box, minus the
  climatological mean of that season, divided by its standard deviation, both
  computed on the reference normal (1991–2020). A raw SST in °C would make
  ENSO — whose amplitude is a degree — look negligible beside the Gulf of
  Guinea, whose seasonal cycle is ten times larger;
* a **dipole** (DMI, SAOD) is the difference of the two standardised poles, not
  the standardised difference: that is what the papers define, and the two are
  not the same when the poles have different variances.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

#: (longitude min, longitude max, latitude min, latitude max) in −180..180 degrees
BOXES = {
    "NINO34": (-170.0, -120.0, -5.0, 5.0),
    "WTIO": (50.0, 70.0, -10.0, 10.0),
    "SETIO": (90.0, 110.0, -10.0, 0.0),
    "ATL3": (-20.0, 0.0, -3.0, 3.0),
    "GG": (-10.0, 10.0, -5.0, 5.0),
    "SAOD_NEP": (-20.0, 10.0, -15.0, 0.0),
    "SAOD_SWP": (-40.0, -10.0, -40.0, -25.0),
}
#: indices published, and how they are built
DIPOLES = {"DMI": ("WTIO", "SETIO"), "SAOD": ("SAOD_NEP", "SAOD_SWP")}
INDICES = ("NINO34", "DMI", "ATL3", "GG", "SAOD")
LABELS = {"NINO34": "Niño 3.4 (Pacifique équatorial)",
          "DMI": "Dipôle de l'océan Indien",
          "ATL3": "Atlantic Niño (ATL3)",
          "GG": "Golfe de Guinée",
          "SAOD": "Dipôle de l'Atlantique Sud"}


def load_ersst(directory, year_start: int = 1981, year_end: int = 2026) -> xr.DataArray:
    """
    Monthly ERSSTv5 read from the NCEI files, longitudes in −180..180.

    One file per month (~170 kB); they are opened together and the singleton
    level dimension is dropped. Longitudes are rewrapped because every index box
    of the literature is written in −180..180 and a box crossing the date line
    would otherwise be silently empty.
    """
    files = sorted(Path(directory).glob("ersst.v5.*.nc"))
    files = [f for f in files if year_start <= int(f.stem.split(".")[-1][:4]) <= year_end]
    if not files:
        raise FileNotFoundError(f"aucun fichier ERSST dans {directory}")
    # the files do not share one calendar (some carry a 360-day one), so the
    # date is read from the file name: one file is one month, unambiguously
    parts = []
    for f in files:
        stamp = f.stem.split(".")[-1]
        one = xr.open_dataset(f, decode_times=False)["sst"]
        for drop in ("lev", "zlev", "time"):
            if drop in one.dims:
                one = one.isel({drop: 0}, drop=True)
        parts.append(one.expand_dims(time=[pd.Timestamp(int(stamp[:4]), int(stamp[4:]), 1)]))
    sst = xr.concat(parts, dim="time")
    sst = sst.assign_coords(lon=(((sst["lon"] + 180) % 360) - 180)).sortby("lon")
    sst.name = "sst"
    return sst.load()


def box_mean(sst: xr.DataArray, box) -> xr.DataArray:
    """Area-weighted mean of a box (cosine of latitude, as everywhere in the chain)."""
    lon0, lon1, lat0, lat1 = box
    sel = sst.sel(lon=slice(lon0, lon1), lat=slice(lat0, lat1))
    if sel.sizes.get("lat", 0) == 0:                    # some files store lat descending
        sel = sst.sel(lon=slice(lon0, lon1), lat=slice(lat1, lat0))
    weights = np.cos(np.deg2rad(sel["lat"]))
    return sel.weighted(weights).mean(("lat", "lon"))


def seasonal_mean(monthly: xr.DataArray, months, years) -> xr.DataArray:
    """
    Mean of a series over the calendar months of a season, one value per year.

    A season crossing the new year (NDJ, DJF) is attached to the year of its
    **first** month, which is the convention of the period keys of the chain.
    """
    months = list(months)
    out = []
    for year in years:
        stamps = []
        y = year
        for k, month in enumerate(months):
            if k and month < months[k - 1]:
                y += 1
            stamps.append(pd.Timestamp(y, month, 1))
        got = monthly.sel(time=[s for s in stamps if s in pd.DatetimeIndex(monthly["time"].values)])
        out.append(got.mean("time") if got.sizes.get("time") else xr.full_like(monthly.isel(time=0, drop=True), np.nan))
    return xr.concat(out, dim=pd.Index(list(years), name="year"))


def standardise(series: xr.DataArray, clim=(1991, 2020), year_dim: str = "year") -> xr.DataArray:
    """Anomaly of the reference normal, divided by its standard deviation."""
    ref = series.sel({year_dim: slice(clim[0], clim[1])})
    return (series - ref.mean(year_dim)) / ref.std(year_dim, ddof=1)


def compute_indices(sst: xr.DataArray, months, years, clim=(1991, 2020),
                    indices=INDICES) -> xr.Dataset:
    """
    The five drivers of Central Africa, standardised, one value per year.

    ``months`` is the season the predictor is read on — for a 1 September
    initialisation, the June-July-August season that precedes it, which is what
    an operational forecast knows at the time it is issued.
    """
    poles = {name: standardise(seasonal_mean(box_mean(sst, box), months, years), clim)
             for name, box in BOXES.items()}
    out = {}
    for name in indices:
        if name in DIPOLES:
            a, b = DIPOLES[name]
            out[name] = poles[a] - poles[b]
        else:
            out[name] = poles[name]
        out[name].attrs = {"long_name": LABELS.get(name, name),
                           "definition": (f"{DIPOLES[name][0]} − {DIPOLES[name][1]}"
                                          if name in DIPOLES else str(BOXES[name])),
                           "standardisation": f"anomalie normalisée {clim[0]}-{clim[1]}",
                           "season_months": ",".join(str(m) for m in months)}
    ds = xr.Dataset(out)
    ds.attrs = {"source": "NOAA ERSSTv5 (2°)", "indices": ", ".join(indices)}
    return ds
