"""
Observed temperature reference: ERA5 daily statistics at 0.25° (workflow step E1).

Dataset ``derived-era5-single-levels-daily-statistics`` of the CDS: daily
**mean, maximum and minimum** of the 2 m temperature, computed by the CDS from
the 24 hourly values of each UTC day. UTC days match the C3S
``maximum/minimum_2m_temperature_in_the_last_24_hours`` fields (24 h ending at
00 UTC), so model and observation cover the same hours.

Decision of 2026-09-19: the temperature reference is ERA5 at 0.25° (the local
ERA5 ``mx2t``/``mn2t`` file holds one-hour extremes at 00 UTC, not daily ones).

One file per statistic and year in ``<data_root>/raw/era5/daily_0p25/``:
``era5_t2m_<statistic>_<YYYY>.nc``. Existing files are skipped. If the CDS
refuses a whole-year request, the year is fetched month by month and merged.

Example::

    python scripts/run_download_era5_daily.py --config config/cycle_202609.yaml --years 1981 2026
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import xarray as xr

from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle

DATASET = "derived-era5-single-levels-daily-statistics"
STATISTICS = {"mean": "daily_mean", "max": "daily_maximum", "min": "daily_minimum"}
#: ECCAS box as CDS [North, West, South, East] — same box as the CHIRPS file.
AREA = [25, 5, -20, 35]


def build_request(statistic: str, year: int, months) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": str(year),
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "daily_statistic": STATISTICS[statistic],
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": AREA,
    }


def _retrieve(request: dict, target: Path, log, attempts: int = 4) -> None:
    import cdsapi
    for k in range(1, attempts + 1):
        try:
            cdsapi.Client(quiet=True).retrieve(DATASET, request, str(target))
            return
        except Exception as exc:
            if k == attempts:
                raise
            wait = 60 * 2 ** (k - 1)
            log.warning("tentative %d/%d échouée (%s) — nouvel essai dans %d s", k, attempts, exc, wait)
            time.sleep(wait)


def fetch_year(statistic: str, year: int, months, dest: Path, log) -> Path:
    """Fetch one statistic for one year (whole year first, month by month if refused)."""
    try:
        tmp = dest.with_suffix(".part.nc")
        _retrieve(build_request(statistic, year, months), tmp, log)
        tmp.replace(dest)
        return dest
    except Exception as exc:
        log.warning("%d %s : requête annuelle refusée (%s) → mois par mois", year, statistic, exc)
    parts = []
    for m in months:
        part = dest.with_name(f"{dest.stem}_{m:02d}.part.nc")
        if not part.exists():
            _retrieve(build_request(statistic, year, [m]), part, log)
        parts.append(part)
    merged = xr.concat([xr.open_dataset(p) for p in parts], dim="valid_time").sortby("valid_time")
    merged.to_netcdf(dest)
    for p in parts:
        p.unlink()
    return dest


def run(config: str, first_year: int, last_year: int, statistics=("mean", "max", "min")) -> RunContext:
    cfg = load_cycle(config)
    out_dir = cfg.data_root / "raw" / "era5" / "daily_0p25"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    with RunContext(cfg, step="download_era5_daily") as ctx:
        ctx.record_parameter("dataset", DATASET)
        ctx.record_parameter("area_NWSE", AREA)
        ctx.record_parameter("years", [first_year, last_year])
        failures = {}
        for year in range(first_year, last_year + 1):
            # ERA5 is released with about 5 days of delay: last complete month only
            last_month = 12 if year < today.year else (today - pd.Timedelta(days=6)).month - 1
            if last_month < 1:
                continue
            months = list(range(1, last_month + 1))
            for stat in statistics:
                dest = out_dir / f"era5_t2m_{stat}_{year}.nc"
                if dest.exists():
                    ctx.record_output(dest, role=f"era5_t2m_{stat}", year=year)
                    continue
                try:
                    ctx.log.info("ERA5 t2m %s %d (mois 1–%d)", stat, year, last_month)
                    fetch_year(stat, year, months, dest, ctx.log)
                    ctx.record_output(dest, role=f"era5_t2m_{stat}", year=year)
                except Exception as exc:
                    failures[f"{stat}.{year}"] = str(exc)
                    ctx.warn(f"échec ERA5 {stat} {year} : {exc}")
        ctx.record_parameter("failures", failures)
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", nargs=2, type=int, required=True, metavar=("PREMIERE", "DERNIERE"))
    ap.add_argument("--statistics", nargs="+", default=["mean", "max", "min"], choices=sorted(STATISTICS))
    args = ap.parse_args(argv)
    run(args.config, args.years[0], args.years[1], args.statistics)


if __name__ == "__main__":
    main()
