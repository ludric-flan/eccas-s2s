"""
Quality control of raw C3S files (workflow steps E0/E2).

For each downloaded GRIB file: number of members and initialisation years,
number and range of steps, **effective horizon**, missing (year, step) pairs,
and warnings. Both streams are covered, recognised from the file name:

``daily`` (``…_MM.grib``)
    24-hourly steps (precipitation, Tmax, Tmin); the horizon is the last lead
    day available for every year.
``monthly`` (``…_MM_monthly.grib``, decision of 23/09/2026)
    monthly means read with :func:`~eccas_s2s.io.c3s_read.load_c3s_monthly`,
    which attaches the lagged start dates of UKMO, BoM and NCEP to their
    nominal 1st-of-month initialisation. The horizon is the number of target
    months complete for every year, counted from the initialisation month; the
    members are counted **per year**, because a lagged ensemble does not always
    carry the same number of members from one year to the next.

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

from eccas_s2s.io.c3s_read import load_c3s_monthly, open_c3s_grib

#: below this number of members, probabilities are flagged as coarse.
MIN_MEMBERS_WARNING = 10

_NAME = re.compile(r"c3s_(?P<centre>.+)_(?P<system>[^_]+)_(?P<var>[A-Z]+)_(?P<kind>forecast|hindcast)_")
_INIT_MONTH = re.compile(r"_(?P<month>\d{2})(?P<monthly>_monthly)?\.grib$")


def stream_of(path: str | Path) -> str:
    """``monthly`` for a ``…_monthly.grib`` file, ``daily`` otherwise."""
    return "monthly" if Path(path).name.endswith("_monthly.grib") else "daily"


def _data_variable(ds) -> str:
    """The field of a C3S file: ``tp`` for the daily rainfall, the only variable otherwise."""
    return "tp" if "tp" in ds.data_vars else list(ds.data_vars)[0]


def parse_c3s_filename(path: str | Path) -> dict:
    """Centre, system, variable and kind encoded in an eccas-s2s C3S filename."""
    m = _NAME.match(Path(path).name)
    if not m:
        raise ValueError(f"Nom de fichier C3S non reconnu : {Path(path).name}")
    return m.groupdict()


def inspect_c3s_file(path: str | Path) -> dict:
    """QC summary of one C3S GRIB file, daily or monthly (see :func:`stream_of`)."""
    if stream_of(path) == "monthly":
        return inspect_c3s_monthly_file(path)
    return inspect_c3s_daily_file(path)


def inspect_c3s_daily_file(path: str | Path) -> dict:
    """
    QC summary of one daily C3S GRIB file (rainfall, Tmax or Tmin).

    Missing steps are detected at the central grid point over all members: a
    missing daily field is missing everywhere, so one point suffices and keeps
    the check fast.
    """
    info = parse_c3s_filename(path)
    ds = open_c3s_grib(path)
    field = ds[_data_variable(ds)]
    point = field.isel(latitude=ds.sizes["latitude"] // 2, longitude=ds.sizes["longitude"] // 2)
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
        "stream": "daily",
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


def inspect_c3s_monthly_file(path: str | Path) -> dict:
    """
    QC summary of one monthly C3S GRIB file (``seasonal-monthly-single-levels``).

    Read through :func:`load_c3s_monthly`, exactly as the chain reads it: the
    lagged start dates are merged into their nominal initialisation, so the
    members counted here are those the products will use. A target month is
    missing for a year when no member has a value at the central grid point.
    """
    info = parse_c3s_filename(path)
    match = _INIT_MONTH.search(Path(path).name)
    init_month = int(match.group("month")) if match else None
    ds = open_c3s_grib(path)
    var = _data_variable(ds)
    da = load_c3s_monthly(path, var=var, init_month=init_month)
    point = da.isel(latitude=da.sizes["latitude"] // 2,
                    longitude=da.sizes["longitude"] // 2).load()
    offsets = point["month_offset"].values.astype(int)
    years = point["year"].values.astype(int)
    missing = point.isnull().all("number").values               # (year, month_offset)
    members_per_year = point.notnull().any("month_offset").sum("number").values

    missing_pairs = [(int(years[i]), int(offsets[j])) for i, j in np.argwhere(missing)]
    # months complete for every year, counted from the initialisation month
    complete = ~missing.any(axis=0)
    expected = np.arange(offsets.max() + 1) if len(offsets) else np.array([], int)
    present = set(offsets[complete].tolist())
    all_years = 0
    for k in expected:
        if k not in present:
            break
        all_years += 1
    any_year = int(offsets[(~missing).any(axis=0)].max()) + 1 if (~missing).any() else 0

    n_members = int(members_per_year.max()) if len(members_per_year) else 0
    n_min = int(members_per_year.min()) if len(members_per_year) else 0
    warnings = []
    if n_min < MIN_MEMBERS_WARNING:
        warnings.append(f"ensemble réduit ({n_min} membres au minimum sur une année)")
    if n_min != n_members:
        warnings.append(f"membres variables d'une année à l'autre ({n_min} à {n_members})")
    if missing_pairs:
        warnings.append(f"{len(missing_pairs)} couple(s) (année, mois cible) manquant(s)")

    return {
        **info,
        "stream": "monthly",
        "file": Path(path).name,
        "size_mb": round(Path(path).stat().st_size / 1e6, 1),
        "members": n_members,
        "members_min": n_min,
        "start_dates": int(da.attrs.get("n_start_dates", 0)),
        "lagged_ensemble": bool(da.attrs.get("lagged_ensemble", False)),
        "years": len(years),
        "first_year": int(years.min()),
        "last_year": int(years.max()),
        "n_steps": len(offsets),
        "first_step_month": int(offsets.min()),
        "last_step_month": int(offsets.max()),
        "horizon_all_years_months": all_years,
        "horizon_any_year_months": any_year,
        "grid": f"{da.sizes['latitude']}x{da.sizes['longitude']}",
        "missing_year_month": missing_pairs,
        "warnings": "; ".join(warnings),
    }


def qc_table(paths) -> pd.DataFrame:
    """QC summary of several files, one row per file."""
    rows = [inspect_c3s_file(p) for p in sorted(paths)]
    # entiers « à trous » (colonnes propres à un flux) plutôt que des flottants
    return pd.DataFrame(rows).convert_dtypes()


def effective_horizons(qc: pd.DataFrame) -> pd.DataFrame:
    """
    Usable daily horizon per model = the forecast horizon, capped by the hindcast
    horizon common to all years (a period must be complete in both).

    Daily files only; see :func:`monthly_horizons` for the monthly stream.
    """
    if "stream" in qc.columns:
        qc = qc[qc["stream"] == "daily"]
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


def monthly_horizons(qc: pd.DataFrame) -> pd.DataFrame:
    """
    Usable monthly horizon per model: the number of target months complete in
    the forecast **and** for every hindcast year, with the members of each side.
    """
    if "stream" not in qc.columns:
        return pd.DataFrame()
    qc = qc[qc["stream"] == "monthly"]
    out = []
    for centre, g in qc.groupby("centre"):
        fc, hc = g[g["kind"] == "forecast"], g[g["kind"] == "hindcast"]
        if fc.empty or hc.empty:
            continue
        out.append({"centre": centre,
                    "forecast_months": int(fc["horizon_all_years_months"].iloc[0]),
                    "hindcast_months": int(hc["horizon_all_years_months"].iloc[0]),
                    "usable_months": int(min(fc["horizon_all_years_months"].iloc[0],
                                             hc["horizon_all_years_months"].iloc[0])),
                    "members_forecast": int(fc["members"].iloc[0]),
                    "members_hindcast_min": int(hc["members_min"].iloc[0]),
                    "members_hindcast": int(hc["members"].iloc[0])})
    return pd.DataFrame(out)
