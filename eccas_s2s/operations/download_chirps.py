"""
Download CHIRPS v2.0 daily years missing from the local archive (workflow step E1).

The chain works at **0.25°** (``p25``), the resolution the calibration of phase
P3 runs on: CHIRPS publishes that product itself, so it is taken as delivered
rather than degraded from 0.05° — one source, no intermediate aggregation to
justify. ``--resolution p05`` still fetches the fine product when a study needs
it.

For each requested year the global daily file is either taken from
``--import-dir`` (files already on disk) or downloaded from UCSB, cut to the
ECCAS box (5–35°E, 20°S–25°N), written compressed to
``<data_root>/raw/chirps/daily_<res>/chirps-v2.0.<YYYY>.days_<res>_ECCAS.nc``,
and the global copy is deleted unless it was imported or ``--keep-global`` is set.

Examples::

    # importer les fichiers 0.25° déjà téléchargés, puis compléter 1981-1992
    python scripts/run_download_chirps.py --config config/cycle_202609.yaml \
        --years 1993 2026 --import-dir /chemin/vers/chirps/global
    python scripts/run_download_chirps.py --config config/cycle_202609.yaml --years 1981 1992
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from eccas_s2s.io._base import http_download
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf"
#: ECCAS box of the local CHIRPS file: cell edges (lon 5..35, lat -20..25).
BOX = {"lon": (5.0, 35.0), "lat": (-20.0, 25.0)}


def subset_to_box(src: Path, dest: Path) -> Path:
    """
    Cut a global CHIRPS daily file to the ECCAS box (cell centres strictly inside).

    The subset is read fully into memory (~0.8 GB per year) before writing:
    reading with dask while writing NetCDF in the same process can deadlock the
    HDF5 library (observed on 2026-09-19).
    """
    with xr.open_dataset(src) as ds:
        lat, lon = ds["latitude"], ds["longitude"]
        sub = ds.sel(latitude=lat[(lat > BOX["lat"][0]) & (lat < BOX["lat"][1])],
                     longitude=lon[(lon > BOX["lon"][0]) & (lon < BOX["lon"][1])])
        sub = sub.sortby("latitude").load()
    enc = {"precip": {"zlib": True, "complevel": 4, "dtype": "float32",
                      "_FillValue": np.float32(-9999.0),
                      "chunksizes": (1, sub.sizes["latitude"], sub.sizes["longitude"])}}
    tmp = dest.with_suffix(".tmp.nc")
    sub.to_netcdf(tmp, encoding=enc)
    tmp.replace(dest)
    return dest


def run(config: str, first_year: int, last_year: int, keep_global: bool = False,
        resolution: str = "p25", import_dir: str | None = None) -> RunContext:
    cfg = load_cycle(config)
    out_dir = cfg.data_root / "raw" / "chirps" / f"daily_{resolution}"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(import_dir) if import_dir else None
    with RunContext(cfg, step="download_chirps") as ctx:
        ctx.record_parameter("years", [first_year, last_year])
        ctx.record_parameter("resolution", resolution)
        ctx.record_parameter("box", BOX)
        if source_dir:
            ctx.record_parameter("import_dir", str(source_dir))
        for year in range(first_year, last_year + 1):
            dest = out_dir / f"chirps-v2.0.{year}.days_{resolution}_ECCAS.nc"
            if dest.exists():
                ctx.log.info("%d déjà présent : %s", year, dest.name)
                ctx.record_output(dest, role="chirps_daily_eccas", year=year)
                continue
            name = f"chirps-v2.0.{year}.days_{resolution}.nc"
            imported = source_dir / name if source_dir else None
            if imported and imported.is_file():
                glob_path, downloaded = imported, False
                ctx.log.info("%d : fichier global déjà sur disque (%s)", year, imported)
            else:
                glob_path, downloaded = out_dir / name, True
                http_download(f"{BASE_URL}/{resolution}/{name}", str(glob_path))
            ctx.record_input(glob_path, role="chirps_daily_global", year=year)
            subset_to_box(glob_path, dest)
            ctx.record_output(dest, role="chirps_daily_eccas", year=year)
            if downloaded and not keep_global:
                glob_path.unlink()
            ctx.log.info("%d : découpé sur la CEEAC (%s)", year, dest.name)
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", nargs=2, type=int, required=True, metavar=("PREMIERE", "DERNIERE"))
    ap.add_argument("--keep-global", action="store_true")
    ap.add_argument("--resolution", default="p25", choices=["p25", "p05"])
    ap.add_argument("--import-dir", help="dossier contenant déjà les fichiers globaux CHIRPS")
    args = ap.parse_args(argv)
    run(args.config, args.years[0], args.years[1], args.keep_global, args.resolution,
        args.import_dir)


if __name__ == "__main__":
    main()
