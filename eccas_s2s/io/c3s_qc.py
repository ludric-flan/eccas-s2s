"""
Quality control of raw C3S files (workflow steps E0/E2).

For each downloaded GRIB file: number of members and initialisation years,
number and range of daily steps, **effective horizon** (last lead day available
for every year), missing (year, step) pairs, and warnings.

Decision of 2026-09-19: raw files are kept as downloaded and processed with the
members available (as the legacy ``run_forecast_v2.py`` does). UKMO and BoM,
whose ensembles are built from several start dates, therefore have few members
on the 1st of the month; this is reported here, not corrected.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from eccas_s2s.io.c3s_read import open_c3s_grib

#: below this number of members, probabilities are flagged as coarse.
MIN_MEMBERS_WARNING = 10

_NAME = re.compile(r"c3s_(?P<centre>.+)_(?P<system>[^_]+)_(?P<var>[A-Z]+)_(?P<kind>forecast|hindcast)_")


def parse_c3s_filename(path: str | Path) -> dict:
    """Centre, system, variable and kind encoded in an eccas-s2s C3S filename."""
    m = _NAME.match(Path(path).name)
    if not m:
        raise ValueError(f"Nom de fichier C3S non reconnu : {Path(path).name}")
    return m.groupdict()


def inspect_c3s_file(path: str | Path) -> dict:
    """
    QC summary of one C3S GRIB file.

    Missing steps are detected at the central grid point over all members: a
    missing daily field is missing everywhere, so one point suffices and keeps
    the check fast.
    """
    info = parse_c3s_filename(path)
    ds = open_c3s_grib(path)
    tp = ds["tp"]
    point = tp.isel(latitude=ds.sizes["latitude"] // 2, longitude=ds.sizes["longitude"] // 2)
    if "time" not in point.dims:
        point = point.expand_dims(time=[ds["time"].values])
    steps = (ds["step"].values / np.timedelta64(1, "D")).astype(int)
    missing = point.isnull().any("number").values          # (time, step)
    years = pd.DatetimeIndex(point["time"].values).year

    missing_pairs = [(int(years[i]), int(steps[j])) for i, j in np.argwhere(missing)]
    # last step present for every year -> effective horizon in days
    complete = ~missing.any(axis=0)
    last_complete = int(steps[complete].max()) if complete.any() else 0
    last_any = int(steps[(~missing).any(axis=0)].max()) if (~missing).any() else 0

    n_members = int(ds.sizes["number"])
    warnings = []
    if n_members < MIN_MEMBERS_WARNING:
        warnings.append(f"ensemble réduit ({n_members} membres au 1er du mois)")
    if missing_pairs:
        warnings.append(f"{len(missing_pairs)} couple(s) (année, échéance) manquant(s)")

    return {
        **info,
        "file": Path(path).name,
        "size_mb": round(Path(path).stat().st_size / 1e6, 1),
        "members": n_members,
        "years": len(years),
        "first_year": int(years.min()),
        "last_year": int(years.max()),
        "n_steps": len(steps),
        "first_step_day": int(steps.min()),
        "last_step_day": int(steps.max()),
        "horizon_all_years_days": last_complete,
        "horizon_any_year_days": last_any,
        "grid": f"{ds.sizes['latitude']}x{ds.sizes['longitude']}",
        "missing_year_step": missing_pairs,
        "warnings": "; ".join(warnings),
    }


def qc_table(paths) -> pd.DataFrame:
    """QC summary of several files, one row per file."""
    rows = [inspect_c3s_file(p) for p in sorted(paths)]
    return pd.DataFrame(rows)


def effective_horizons(qc: pd.DataFrame) -> pd.DataFrame:
    """
    Usable daily horizon per model = the forecast horizon, capped by the hindcast
    horizon common to all years (a period must be complete in both).
    """
    out = []
    for centre, g in qc.groupby("centre"):
        fc = g.loc[g["kind"] == "forecast", "horizon_all_years_days"]
        hc = g.loc[g["kind"] == "hindcast", "horizon_any_year_days"]
        if fc.empty or hc.empty:
            continue
        out.append({"centre": centre, "forecast_days": int(fc.iloc[0]),
                    "hindcast_days": int(hc.iloc[0]),
                    "usable_days": int(min(fc.iloc[0], hc.iloc[0])),
                    "members_forecast": int(g.loc[g["kind"] == "forecast", "members"].iloc[0]),
                    "members_hindcast": int(g.loc[g["kind"] == "hindcast", "members"].iloc[0])})
    return pd.DataFrame(out)
