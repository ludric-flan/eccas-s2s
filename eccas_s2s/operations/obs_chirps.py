"""
Observed precipitation reference: CHIRPS archive and normals (workflow step E1).

Builds, once, the derived archives shared by all cycles, in
``<data_root>/derived/obs/chirps/``:

* ``chirps_p05_dekads_<Y0>_<Y1>.nc`` / ``chirps_p05_months_<Y0>_<Y1>.nc`` —
  dekadal and monthly totals at 0.05°;
* ``chirps_1p0_dekads_...nc`` / ``chirps_1p0_months_...nc`` — the same, block
  averaged onto the 1° C3S grid (exact nesting);
* ``chirps_p05_normals_1991_2020.nc`` and ``chirps_1p0_normals_1991_2020.nc`` —
  mean, std and percentiles per grid point and calendar period (D12);
* ``chirps_daily_qc_<Y0>_<Y1>.csv`` — monthly quality control of the daily file.

The step is idempotent: when the daily source file is unchanged (same size and
modification time, stored in the archive attributes) nothing is recomputed,
unless ``rebuild=True``.

Example::

    python scripts/run_obs_chirps.py --config config/cycle_202609.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from eccas_s2s.obs.chirps import open_chirps_daily, process_daily, write_totals
from eccas_s2s.obs.climatology import normals
from eccas_s2s.obs.regrid import block_average, snap_coords
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle


def _source_signature(path: Path) -> str:
    st = path.stat()
    return f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"


def derived_paths(cfg) -> dict[str, Path]:
    """Locations of the CHIRPS derived archives for this configuration."""
    d = cfg.data_root / "derived" / "obs" / "chirps"
    src = open_chirps_daily(cfg.raw["observations"]["precip"]["daily_path"])
    t = pd.DatetimeIndex(src["time"].values)
    y0, y1 = t[0].year, t[-1].year
    n0, n1 = cfg.reference_period("obs_normal")
    return {
        "dir": d,
        "dekads_p05": d / f"chirps_p05_dekads_{y0}_{y1}.nc",
        "months_p05": d / f"chirps_p05_months_{y0}_{y1}.nc",
        "dekads_1p0": d / f"chirps_1p0_dekads_{y0}_{y1}.nc",
        "months_1p0": d / f"chirps_1p0_months_{y0}_{y1}.nc",
        "normals_p05": d / f"chirps_p05_normals_{n0}_{n1}.nc",
        "normals_1p0": d / f"chirps_1p0_normals_{n0}_{n1}.nc",
        "qc": d / f"chirps_daily_qc_{y0}_{y1}.csv",
    }


def _is_current(path: Path, signature: str) -> bool:
    if not path.exists():
        return False
    with xr.open_dataset(path) as ds:
        return ds.attrs.get("source_signature") == signature


def load_archives(cfg, resolution: str = "p05") -> tuple[xr.DataArray, xr.DataArray]:
    """Open the dekadal and monthly totals (``resolution`` = "p05" or "1p0")."""
    p = derived_paths(cfg)
    return (snap_coords(xr.open_dataset(p[f"dekads_{resolution}"])["precip"]),
            snap_coords(xr.open_dataset(p[f"months_{resolution}"])["precip"]))


def load_normals(cfg, resolution: str = "p05") -> xr.Dataset:
    return snap_coords(xr.open_dataset(derived_paths(cfg)[f"normals_{resolution}"]))


def run(config: str, rebuild: bool = False) -> RunContext:
    cfg = load_cycle(config)
    source = Path(cfg.raw["observations"]["precip"]["daily_path"])
    paths = derived_paths(cfg)
    signature = _source_signature(source)
    thr = cfg.thresholds
    n0, n1 = cfg.reference_period("obs_normal")

    with RunContext(cfg, step="obs_chirps") as ctx:
        ctx.record_input(source, role="chirps_daily")
        ctx.record_parameter("normal_period", [n0, n1])
        ctx.record_parameter("percentiles", thr["percentiles"])
        ctx.record_parameter("percentile_method", thr["percentile_method"])
        common = {"source_signature": signature, "source_file": str(source.resolve()),
                  **ctx.netcdf_attrs()}

        # 1) QC + calendar totals at 0.05° (one streaming pass over the daily file)
        if rebuild or not (_is_current(paths["dekads_p05"], signature)
                           and _is_current(paths["months_p05"], signature) and paths["qc"].exists()):
            ctx.log.info("lecture de %s mois par mois (QC + cumuls décadaires et mensuels)", source.name)
            da = open_chirps_daily(source)
            qc, dekads, months = process_daily(
                da, progress=lambda y, m: ctx.log.info("  %d-%02d", y, m) if m == 1 else None)
            paths["dir"].mkdir(parents=True, exist_ok=True)
            qc.to_csv(paths["qc"], index=False)
            write_totals(dekads, paths["dekads_p05"], common)
            write_totals(months, paths["months_p05"], common)
            rebuilt = True
        else:
            ctx.log.info("archives 0.05° à jour (source inchangée) : pas de recalcul")
            rebuilt = False
        dekads, months = load_archives(cfg, "p05")

        # 2) 1° versions by exact block averaging
        if rebuilt or not (_is_current(paths["dekads_1p0"], signature) and _is_current(paths["months_1p0"], signature)):
            ctx.log.info("moyenne par blocs 0.05° → 1°")
            write_totals(block_average(dekads.load(), 1.0).astype("float32"), paths["dekads_1p0"], common)
            write_totals(block_average(months.load(), 1.0).astype("float32"), paths["months_1p0"], common)
            rebuilt = True

        # 3) normals 1991-2020 at both resolutions
        for res in ("p05", "1p0"):
            target = paths[f"normals_{res}"]
            if rebuilt or not _is_current(target, signature):
                ctx.log.info("normales %d-%d à %s", n0, n1, res)
                dk, mo = load_archives(cfg, res)
                norm = normals(dk, mo, (n0, n1), thr["percentiles"], thr["percentile_method"],
                               thr["min_years_normal"])
                norm.attrs.update(common)
                enc = {v: {"zlib": True, "complevel": 4} for v in norm.data_vars}
                tmp = target.with_suffix(".tmp.nc")
                norm.to_netcdf(tmp, encoding=enc)
                tmp.replace(target)

        qc = pd.read_csv(paths["qc"])
        bad = qc[(qc.days_present != qc.days_expected) | (qc.missing_land_values > 0)
                 | (qc.negative_values > 0)]
        if len(bad):
            ctx.warn(f"{len(bad)} mois avec jours manquants, valeurs manquantes sur terre ou négatives")
        n_susp = int(qc.suspicious_values.sum())
        if n_susp:
            ctx.warn(f"{n_susp} valeur(s) journalière(s) > 300 mm à examiner")
        ctx.record_parameter("qc_months_flagged", int(len(bad)))
        ctx.record_parameter("qc_suspicious_daily_values", n_susp)
        for key in ("qc", "dekads_p05", "months_p05", "dekads_1p0", "months_1p0", "normals_p05", "normals_1p0"):
            ctx.record_output(paths[key], role=f"chirps_{key}")
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rebuild", action="store_true", help="tout recalculer même si la source est inchangée")
    args = ap.parse_args(argv)
    run(args.config, args.rebuild)


if __name__ == "__main__":
    main()
