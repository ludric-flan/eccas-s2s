"""
Download CHIRPS v2.0 daily years missing from the local archive (workflow step E1).

For each requested year, the global daily file (``chirps-v2.0.<YYYY>.days_p05.nc``,
about 1.1 GB) is downloaded from UCSB, cut to the ECCAS box used by the existing
local file (5–35°E, 20°S–25°N, same 0.05° cells), written compressed to
``<data_root>/raw/chirps/daily_p05/chirps-v2.0.<YYYY>.days_p05_ECCAS.nc`` and the
global file is deleted (disk space).

Example (complete 1981–1990 for the NMME hindcast period)::

    python scripts/run_download_chirps.py --config config/cycle_202609.yaml --years 1981 1990
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from eccas_s2s.io._base import http_download
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05"
#: ECCAS box of the local CHIRPS file: cell edges (lon 5..35, lat -20..25).
BOX = {"lon": (5.0, 35.0), "lat": (-20.0, 25.0)}


def subset_to_box(src: Path, dest: Path) -> Path:
    """Cut a global CHIRPS daily file to the ECCAS box (cell centres strictly inside)."""
    with xr.open_dataset(src, chunks={"time": 31}) as ds:
        lat, lon = ds["latitude"], ds["longitude"]
        sub = ds.sel(latitude=lat[(lat > BOX["lat"][0]) & (lat < BOX["lat"][1])],
                     longitude=lon[(lon > BOX["lon"][0]) & (lon < BOX["lon"][1])])
        sub = sub.sortby("latitude")
        enc = {"precip": {"zlib": True, "complevel": 4, "dtype": "float32",
                          "_FillValue": np.float32(-9999.0),
                          "chunksizes": (1, sub.sizes["latitude"], sub.sizes["longitude"])}}
        tmp = dest.with_suffix(".tmp.nc")
        sub.to_netcdf(tmp, encoding=enc)
    tmp.replace(dest)
    return dest


def run(config: str, first_year: int, last_year: int, keep_global: bool = False) -> RunContext:
    cfg = load_cycle(config)
    out_dir = cfg.data_root / "raw" / "chirps" / "daily_p05"
    out_dir.mkdir(parents=True, exist_ok=True)
    with RunContext(cfg, step="download_chirps") as ctx:
        ctx.record_parameter("years", [first_year, last_year])
        ctx.record_parameter("box", BOX)
        for year in range(first_year, last_year + 1):
            dest = out_dir / f"chirps-v2.0.{year}.days_p05_ECCAS.nc"
            if dest.exists():
                ctx.log.info("%d déjà présent : %s", year, dest.name)
                ctx.record_output(dest, role="chirps_daily_eccas", year=year)
                continue
            glob_path = out_dir / f"chirps-v2.0.{year}.days_p05.nc"
            http_download(f"{BASE_URL}/chirps-v2.0.{year}.days_p05.nc", str(glob_path))
            ctx.record_input(glob_path, role="chirps_daily_global", year=year)
            subset_to_box(glob_path, dest)
            ctx.record_output(dest, role="chirps_daily_eccas", year=year)
            if not keep_global:
                glob_path.unlink()
            ctx.log.info("%d : découpé sur la CEEAC (%s)", year, dest.name)
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", nargs=2, type=int, required=True, metavar=("PREMIERE", "DERNIERE"))
    ap.add_argument("--keep-global", action="store_true")
    args = ap.parse_args(argv)
    run(args.config, args.years[0], args.years[1], args.keep_global)


if __name__ == "__main__":
    main()
