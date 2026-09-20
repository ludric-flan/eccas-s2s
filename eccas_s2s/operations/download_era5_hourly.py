"""
Observed temperature reference: ERA5 hourly 2 m temperature → daily mean,
maximum and minimum, computed locally (workflow step E1).

Method of the CAPC-AC script ``download_era5_land.py`` (decision of
2026-09-19): **one request per year**, several years requested at the same time
(decision of 2026-09-20: concurrent requests get a better position in the
CDS-MARS queue), for the hourly
``2m_temperature`` of ``reanalysis-era5-single-levels`` (0.25°) over the ECCAS
box, in GRIB. The CDS queue of the post-processed daily statistics was
saturated (thousands of queued requests), whereas hourly ERA5 is served from
the archive.

For each year:

1. download ``era5_t2m_hourly_<YYYY>.grib`` (24 h × every day of the year);
2. read it (cfgrib), convert K → °C;
3. compute, for every UTC day, the mean, maximum and minimum of the 24 hourly
   values (``skipna=False``: a day with a missing hour is NaN);
4. write ``era5_t2m_daily_<YYYY>.nc`` (variables ``tmean``, ``tmax``, ``tmin``);
5. delete the hourly GRIB (≈ 400 MB per year) unless ``keep_hourly``; its
   fingerprint stays in the run manifest.

Note: daily extremes taken from the 24 hourly *instantaneous* values can be
slightly less extreme than the true maximum/minimum within the hour; the same
definition is used for every year, so percentiles and anomalies stay
consistent.

Years already processed are skipped; a failed year does not stop the others
(re-run to complete).

Example::

    python scripts/run_download_era5_hourly.py --config config/cycle_202609.yaml --years 1981 2026
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.provenance import RunContext, file_fingerprint
from eccas_s2s.settings import load_cycle

DATASET = "reanalysis-era5-single-levels"
#: ECCAS box as CDS [North, West, South, East] — same box as CHIRPS.
AREA = [25, 5, -20, 35]
MAX_ATTEMPTS = 3


def build_request(year: int, months) -> dict:
    """Hourly 2 m temperature for one year (pure, no network)."""
    return {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": AREA,
    }


def months_for_year(year: int, today=None) -> list[int]:
    """Complete months available for ``year`` (ERA5 is published with ~5 days of delay)."""
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    last = (today - pd.Timedelta(days=6)).to_period("M") - 1
    if year < last.year:
        return list(range(1, 13))
    if year == last.year:
        return list(range(1, last.month + 1))
    return []


def hourly_to_daily(hourly: xr.DataArray) -> xr.Dataset:
    """
    Daily mean, max and min (°C) of hourly 2 m temperature in K (UTC days).

    A day that does not have its 24 hourly values is NaN (``skipna=False`` plus a
    count check): a daily statistic is never computed from an incomplete day.
    """
    t2m = (hourly - 273.15).sortby("time")
    t = pd.DatetimeIndex(t2m["time"].values)
    if t.has_duplicates:
        t2m = t2m.isel(time=~t.duplicated())
    ones = xr.DataArray(np.ones(t2m.sizes["time"]), dims="time", coords={"time": t2m["time"]})
    counts = ones.resample(time="1D").sum()
    daily = xr.Dataset({
        "tmean": t2m.resample(time="1D").mean(skipna=False),
        "tmax": t2m.resample(time="1D").max(skipna=False),
        "tmin": t2m.resample(time="1D").min(skipna=False),
    })
    daily = daily.where(counts == 24)
    for v, name in (("tmean", "daily mean"), ("tmax", "daily maximum"), ("tmin", "daily minimum")):
        daily[v].attrs = {"units": "degC", "long_name": f"{name} of hourly 2 m temperature (UTC day)"}
    daily.attrs["days_incomplete"] = int((counts != 24).sum())
    return daily


def read_hourly_grib(grib: Path) -> xr.DataArray:
    """Hourly 2 m temperature (K) of an ERA5 GRIB file, as ``(time, latitude, longitude)``."""
    ds = xr.open_dataset(grib, engine="cfgrib", backend_kwargs={"indexpath": ""})
    t2m = ds["t2m"]
    if "step" in t2m.dims:          # forecast-style layout: (time, step) -> valid time
        t2m = t2m.stack(vt=("time", "step"))
        t2m = t2m.drop_vars(["vt", "time", "step"]).assign_coords(vt=ds["valid_time"].values.ravel()).rename(vt="time")
    elif "valid_time" in t2m.coords:
        t2m = t2m.assign_coords(time=t2m["valid_time"].values)
    for c in ("number", "surface", "step", "valid_time", "heightAboveGround"):
        if c in t2m.coords:
            t2m = t2m.drop_vars(c)
    return t2m.sortby("latitude").load()


def _process_year(year: int, months, out_dir: Path, keep_hourly: bool, ctx) -> Path:
    """Download one year of hourly T2m, compute the daily statistics, drop the GRIB."""
    import cdsapi

    daily_path = out_dir / f"era5_t2m_daily_{year}.nc"
    grib = out_dir / f"era5_t2m_hourly_{year}.grib"
    if not grib.exists():
        ctx.log.info("%d : requête CDS (T2m horaire, mois %d–%d)", year, months[0], months[-1])
        part = grib.with_suffix(f".{os.getpid()}.part")
        for k in range(1, MAX_ATTEMPTS + 1):
            try:
                cdsapi.Client(quiet=True, progress=False).retrieve(DATASET, build_request(year, months), str(part))
                break
            except Exception as exc:
                if k == MAX_ATTEMPTS:
                    raise
                ctx.log.warning("%d : tentative %d échouée (%s)", year, k, exc)
                time.sleep(120 * k)
        part.replace(grib)
    daily = hourly_to_daily(read_hourly_grib(grib))
    n_days = daily.sizes["time"]
    expected = sum(pd.Timestamp(year, m, 1).days_in_month for m in months)
    if n_days != expected:
        ctx.warn(f"{year} : {n_days} jours calculés pour {expected} attendus")
    if daily.attrs.get("days_incomplete"):
        ctx.warn(f"{year} : {daily.attrs['days_incomplete']} jour(s) sans leurs 24 heures")
    daily.attrs.update({**ctx.netcdf_attrs(), "source": f"{DATASET} hourly 2m_temperature",
                        "source_grib_sha256": file_fingerprint(grib)["sha256"]})
    tmp = daily_path.with_suffix(".tmp.nc")
    daily.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4, "dtype": "float32"}
                                   for v in ("tmean", "tmax", "tmin")})
    tmp.replace(daily_path)
    if not keep_hourly:
        grib.unlink()
    ctx.log.info("%d : %d jours, fichier %s", year, n_days, daily_path.name)
    return daily_path


def run(config: str, first_year: int, last_year: int, keep_hourly: bool = False,
        workers: int = 4) -> RunContext:
    cfg = load_cycle(config)
    out_dir = cfg.data_root / "raw" / "era5" / "hourly_0p25"
    out_dir.mkdir(parents=True, exist_ok=True)
    with RunContext(cfg, step="download_era5_hourly") as ctx:
        ctx.record_parameter("dataset", DATASET)
        ctx.record_parameter("area_NWSE", AREA)
        ctx.record_parameter("method", f"1 request per year, {workers} at a time; "
                                       "daily mean/max/min computed locally")
        todo = []
        for year in range(first_year, last_year + 1):
            months = months_for_year(year)
            if not months:
                continue
            if (out_dir / f"era5_t2m_daily_{year}.nc").exists():
                ctx.record_output(out_dir / f"era5_t2m_daily_{year}.nc", role="era5_t2m_daily", year=year)
                continue
            todo.append((year, months))
        ctx.log.info("%d année(s) à traiter, %d requête(s) simultanée(s)", len(todo), workers)

        failures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_year, y, m, out_dir, keep_hourly, ctx): y for y, m in todo}
            for fut in as_completed(futures):
                year = futures[fut]
                try:
                    ctx.record_output(fut.result(), role="era5_t2m_daily", year=year)
                except Exception as exc:
                    failures[str(year)] = f"{type(exc).__name__}: {exc}"
                    ctx.warn(f"échec {year} : {exc}")
        ctx.record_parameter("failures", failures)
        ctx.log.info("terminé : %d année(s) traitée(s), %d échec(s) — relancer pour compléter",
                     len(todo) - len(failures), len(failures))
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", nargs=2, type=int, required=True, metavar=("PREMIERE", "DERNIERE"))
    ap.add_argument("--keep-hourly", action="store_true", help="conserver les GRIB horaires")
    ap.add_argument("--workers", type=int, default=4, help="années demandées simultanément au CDS")
    args = ap.parse_args(argv)
    run(args.config, args.years[0], args.years[1], args.keep_hourly, args.workers)


if __name__ == "__main__":
    main()
