"""
Observed temperature reference: ERA5 post-processed daily statistics at 0.25°
(workflow step E1).

Dataset ``derived-era5-single-levels-daily-statistics`` ("ERA5 post-processed
daily statistics on single levels from 1940 to present"): daily **mean,
maximum and minimum** of the 2 m temperature, computed by the CDS from the 24
hourly values of each UTC day. UTC days match the C3S
``maximum/minimum_2m_temperature_in_the_last_24_hours`` (24 h ending 00 UTC).

Light requests (decision of 2026-09-19, inspired by the CAPC-AC script
``download_era5_land.py``): the CDS computes these statistics on demand from
hourly data, and large requests wait for hours. Each request therefore covers
**one month × one statistic** over the ECCAS box only; a few requests are kept
in the queue at the same time. As in the CAPC-AC script, a block already on disk
is skipped and a failed block does not stop the others (re-run to complete).
When the 12 months of a year are present they are merged into one yearly file
and the monthly blocks are removed.

Files in ``<data_root>/raw/era5/daily_0p25/``::

    era5_t2m_<mean|max|min>_<YYYY>.nc            complete years
    blocks/era5_t2m_<stat>_<YYYY>_<MM>.nc         months waiting for their year

Example::

    python scripts/run_download_era5_daily.py --config config/cycle_202609.yaml --years 1981 2026
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import xarray as xr

from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle

DATASET = "derived-era5-single-levels-daily-statistics"
STATISTICS = {"mean": "daily_mean", "max": "daily_maximum", "min": "daily_minimum"}
#: ECCAS box as CDS [North, West, South, East] — same box as the CHIRPS file.
AREA = [25, 5, -20, 35]
MAX_ATTEMPTS = 3


def build_request(statistic: str, year: int, month: int) -> dict:
    """One month, one statistic, ECCAS box (pure, no network)."""
    ndays = pd.Timestamp(year, month, 1).days_in_month
    return {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": str(year),
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, ndays + 1)],
        "daily_statistic": STATISTICS[statistic],
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": AREA,
    }


def available_months(first_year: int, last_year: int, today=None) -> list[tuple[int, int]]:
    """(year, month) blocks to request: complete months only (ERA5 has ~5 days of delay)."""
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    last = (today - pd.Timedelta(days=6)).to_period("M") - 1
    months = pd.period_range(f"{first_year}-01", f"{last_year}-12", freq="M")
    return [(p.year, p.month) for p in months if p <= last]


def _fetch_block(statistic: str, year: int, month: int, dest: Path, log) -> Path:
    import cdsapi
    tmp = dest.with_suffix(".part")
    for k in range(1, MAX_ATTEMPTS + 1):
        try:
            cdsapi.Client(quiet=True, progress=False).retrieve(DATASET, build_request(statistic, year, month), str(tmp))
            tmp.replace(dest)
            return dest
        except Exception as exc:
            if k == MAX_ATTEMPTS:
                raise
            log.warning("%s %d-%02d : tentative %d/%d échouée (%s)", statistic, year, month, k, MAX_ATTEMPTS, exc)
            time.sleep(60 * k)


def merge_year(stat: str, year: int, blocks_dir: Path, out_dir: Path) -> Path | None:
    """Merge the 12 monthly blocks of a year into one file (then delete them)."""
    parts = [blocks_dir / f"era5_t2m_{stat}_{year}_{m:02d}.nc" for m in range(1, 13)]
    if not all(p.exists() for p in parts):
        return None
    dest = out_dir / f"era5_t2m_{stat}_{year}.nc"
    datasets = [xr.open_dataset(p) for p in parts]
    time_dim = "valid_time" if "valid_time" in datasets[0].dims else "time"
    merged = xr.concat(datasets, dim=time_dim).sortby(time_dim)
    var = list(merged.data_vars)[0]
    tmp = dest.with_suffix(".tmp.nc")
    merged.to_netcdf(tmp, encoding={var: {"zlib": True, "complevel": 4}})
    for d in datasets:
        d.close()
    tmp.replace(dest)
    for p in parts:
        p.unlink()
    return dest


def run(config: str, first_year: int, last_year: int, statistics=("mean", "max", "min"),
        workers: int = 4) -> RunContext:
    cfg = load_cycle(config)
    out_dir = cfg.data_root / "raw" / "era5" / "daily_0p25"
    blocks_dir = out_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    with RunContext(cfg, step="download_era5_daily") as ctx:
        ctx.record_parameter("dataset", DATASET)
        ctx.record_parameter("area_NWSE", AREA)
        ctx.record_parameter("block", "1 month x 1 statistic")
        ctx.record_parameter("years", [first_year, last_year])
        todo = []
        for year, month in available_months(first_year, last_year):
            for stat in statistics:
                if (out_dir / f"era5_t2m_{stat}_{year}.nc").exists():
                    continue
                block = blocks_dir / f"era5_t2m_{stat}_{year}_{month:02d}.nc"
                if not block.exists():
                    todo.append((stat, year, month, block))
        ctx.log.info("%d bloc(s) mensuel(s) à télécharger, %d requête(s) simultanée(s)", len(todo), workers)

        failures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_block, s, y, m, b, ctx.log): (s, y, m) for s, y, m, b in todo}
            for n, fut in enumerate(as_completed(futures), start=1):
                s, y, m = futures[fut]
                try:
                    fut.result()
                    ctx.log.info("[%d/%d] %s %d-%02d reçu", n, len(todo), s, y, m)
                    if m == 12 or (y, m) == available_months(first_year, last_year)[-1]:
                        merged = merge_year(s, y, blocks_dir, out_dir)
                        if merged:
                            ctx.log.info("année %d (%s) regroupée : %s", y, s, merged.name)
                except Exception as exc:
                    failures[f"{s}.{y}-{m:02d}"] = str(exc)
                    ctx.warn(f"échec {s} {y}-{m:02d} : {exc}")

        # merge any year whose 12 blocks are now all present (blocks may finish out of order)
        for stat in statistics:
            for year in range(first_year, last_year + 1):
                if not (out_dir / f"era5_t2m_{stat}_{year}.nc").exists():
                    merge_year(stat, year, blocks_dir, out_dir)
        for f in sorted(out_dir.glob("era5_t2m_*.nc")):
            ctx.record_output(f, role="era5_t2m_daily_year")
        ctx.record_parameter("failures", failures)
        ctx.log.info("terminé : %d échec(s) — relancer pour compléter", len(failures))
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", nargs=2, type=int, required=True, metavar=("PREMIERE", "DERNIERE"))
    ap.add_argument("--statistics", nargs="+", default=["mean", "max", "min"], choices=sorted(STATISTICS))
    ap.add_argument("--workers", type=int, default=4, help="requêtes CDS simultanées")
    args = ap.parse_args(argv)
    run(args.config, args.years[0], args.years[1], args.statistics, args.workers)


if __name__ == "__main__":
    main()
