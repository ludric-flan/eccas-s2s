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

**The year in progress is refreshed.** A year whose local file stops before
31 December is incomplete: each new cycle must add the month just elapsed. For
such a year the date of the UCSB file (``Last-Modified``, read by a light HEAD
request) is compared with the one stored in the local file; when UCSB has
published a newer file, the year is downloaded again, cut, and replaces the
local one, and the log says up to which day it now goes. A complete year is
never downloaded again. ``--refresh`` forces the download of the requested
years (incident of 13/11/2026: the 2026 file had to be deleted by hand for
October to come in).

Without ``--years``, the year of the last elapsed month before the
initialisation is the one refreshed (2026 for the November 2026 cycle, 2026
still for the January 2027 cycle).

Examples::

    # début de cycle : compléter l'année en cours avec le mois écoulé
    python scripts/run_download_chirps.py --config config/cycle_202611.yaml

    # importer les fichiers 0.25° déjà téléchargés, puis compléter 1981-1992
    python scripts/run_download_chirps.py --config config/cycle_202609.yaml \
        --years 1993 2026 --import-dir /chemin/vers/chirps/global
    python scripts/run_download_chirps.py --config config/cycle_202609.yaml --years 1981 1992
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.io._base import http_download
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle

#: UCSB moved the archive from ``CHIRPS-2.0/`` to ``CHIRPS/v2.0/`` (seen 13/11/2026)
BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v2.0/global_daily/netcdf"
#: ECCAS box of the local CHIRPS file: cell edges (lon 5..35, lat -20..25).
BOX = {"lon": (5.0, 35.0), "lat": (-20.0, 25.0)}


def year_url(year: int, resolution: str = "p25") -> str:
    return f"{BASE_URL}/{resolution}/chirps-v2.0.{year}.days_{resolution}.nc"


def last_day(path: Path) -> pd.Timestamp | None:
    """Last day held by a local CHIRPS file (``None`` if unreadable)."""
    try:
        with xr.open_dataset(path) as ds:
            return pd.Timestamp(ds["time"].values[-1]).normalize()
    except Exception:  # noqa: BLE001 — a damaged file is simply re-downloaded
        return None


def remote_stamp(url: str, timeout: int = 60) -> str | None:
    """``Last-Modified`` (or, failing that, size) of the UCSB file, by a HEAD request."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "eccas-s2s/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        stamp = resp.headers.get("Last-Modified") or resp.headers.get("Content-Length")
    return str(stamp) if stamp else None


def local_stamp(path: Path) -> str | None:
    with xr.open_dataset(path) as ds:
        return ds.attrs.get("source_last_modified")


def default_years(cfg) -> tuple[int, int]:
    """The year of the last month elapsed before the initialisation."""
    year = (pd.Timestamp(cfg.init_date) - pd.Timedelta(days=1)).year
    return year, year


def subset_to_box(src: Path, dest: Path, stamp: str | None = None) -> Path:
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
    if stamp:
        sub.attrs["source_last_modified"] = stamp
    enc = {"precip": {"zlib": True, "complevel": 4, "dtype": "float32",
                      "_FillValue": np.float32(-9999.0),
                      "chunksizes": (1, sub.sizes["latitude"], sub.sizes["longitude"])}}
    tmp = dest.with_suffix(".tmp.nc")
    sub.to_netcdf(tmp, encoding=enc)
    tmp.replace(dest)
    return dest


def _needs_download(ctx, year: int, dest: Path, url: str, refresh: bool) -> tuple[bool, str | None]:
    """
    Whether the local file of ``year`` must be (re)downloaded, and the UCSB stamp.

    A missing file is downloaded; a complete year (up to 31 December) never
    again; an incomplete year when UCSB holds a newer file than the local one.
    """
    if not dest.exists():
        return True, None
    end = last_day(dest)
    if end is None:
        ctx.warn(f"{dest.name} illisible : retéléchargé")
        return True, None
    if refresh:
        ctx.log.info("%d : retéléchargement demandé (--refresh), local jusqu'au %s",
                     year, end.date())
        return True, None
    if end >= pd.Timestamp(year=year, month=12, day=31):
        ctx.log.info("%d complet (jusqu'au %s) : %s", year, end.date(), dest.name)
        return False, None
    try:
        stamp = remote_stamp(url)
    except Exception as exc:  # noqa: BLE001 — no network: keep what is there
        ctx.warn(f"{year} incomplet (jusqu'au {end.date()}) mais UCSB injoignable ({exc}) : "
                 "fichier local conservé")
        return False, None
    if stamp and stamp == local_stamp(dest):
        ctx.log.info("%d incomplet (jusqu'au %s) et UCSB n'a rien publié de plus "
                     "(fichier du %s) : conservé", year, end.date(), stamp)
        return False, stamp
    ctx.log.info("%d incomplet (jusqu'au %s) ; UCSB publie un fichier plus récent (%s) : "
                 "mise à jour", year, end.date(), stamp or "date inconnue")
    return True, stamp


def run(config: str, first_year: int | None = None, last_year: int | None = None,
        keep_global: bool = False, resolution: str = "p25", import_dir: str | None = None,
        refresh: bool = False) -> RunContext:
    cfg = load_cycle(config)
    if first_year is None:
        first_year, last_year = default_years(cfg)
    last_year = first_year if last_year is None else last_year
    out_dir = cfg.data_root / "raw" / "chirps" / f"daily_{resolution}"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(import_dir) if import_dir else None
    with RunContext(cfg, step="download_chirps") as ctx:
        ctx.record_parameter("years", [first_year, last_year])
        ctx.record_parameter("resolution", resolution)
        ctx.record_parameter("box", BOX)
        ctx.record_parameter("refresh", refresh)
        if source_dir:
            ctx.record_parameter("import_dir", str(source_dir))
        updated = {}
        for year in range(first_year, last_year + 1):
            dest = out_dir / f"chirps-v2.0.{year}.days_{resolution}_ECCAS.nc"
            name = f"chirps-v2.0.{year}.days_{resolution}.nc"
            url = year_url(year, resolution)
            imported = source_dir / name if source_dir else None
            from_disk = bool(imported and imported.is_file())
            before = last_day(dest) if dest.exists() else None
            if from_disk and not dest.exists():
                needed, stamp = True, None
            else:
                needed, stamp = _needs_download(ctx, year, dest, url, refresh)
            if not needed:
                ctx.record_output(dest, role="chirps_daily_eccas", year=year)
                continue
            if from_disk and not refresh:
                glob_path, downloaded = imported, False
                ctx.log.info("%d : fichier global déjà sur disque (%s)", year, imported)
            else:
                glob_path, downloaded = out_dir / name, True
                if stamp is None:
                    try:
                        stamp = remote_stamp(url)
                    except Exception:  # noqa: BLE001 — the stamp is only a convenience
                        stamp = None
                http_download(url, str(glob_path), force_download=True)
            ctx.record_input(glob_path, role="chirps_daily_global", year=year)
            subset_to_box(glob_path, dest, stamp=stamp)
            ctx.record_output(dest, role="chirps_daily_eccas", year=year)
            if downloaded and not keep_global:
                glob_path.unlink()
            after = last_day(dest)
            updated[year] = [str(before.date()) if before is not None else None,
                             str(after.date()) if after is not None else None]
            if before is not None and after is not None and after <= before:
                ctx.log.info("%d : retéléchargé, mais toujours jusqu'au %s (rien de neuf)",
                             year, after.date())
            else:
                ctx.log.info("%d : découpé sur la CEEAC (%s), jusqu'au %s%s", year, dest.name,
                             after.date() if after is not None else "?",
                             f" (avant : {before.date()})" if before is not None else "")
        ctx.record_parameter("updated_years", updated)
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", nargs=2, type=int, metavar=("PREMIERE", "DERNIERE"),
                    help="défaut : l'année du dernier mois écoulé avant l'initialisation")
    ap.add_argument("--refresh", action="store_true",
                    help="retélécharger les années demandées même si elles semblent à jour")
    ap.add_argument("--keep-global", action="store_true")
    ap.add_argument("--resolution", default="p25", choices=["p25", "p05"])
    ap.add_argument("--import-dir", help="dossier contenant déjà les fichiers globaux CHIRPS")
    args = ap.parse_args(argv)
    first, last = args.years if args.years else (None, None)
    run(args.config, first, last, args.keep_global, args.resolution, args.import_dir,
        refresh=args.refresh)


if __name__ == "__main__":
    main()
