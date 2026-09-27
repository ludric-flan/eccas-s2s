"""
Observed precipitation reference: CHIRPS archive and normals (workflow step E1).

Builds, once, the derived archives shared by all cycles, in
``<data_root>/derived/obs/chirps/``:

* ``chirps_<res>_dekads_<Y0>_<Y1>.nc`` / ``chirps_<res>_months_<Y0>_<Y1>.nc`` —
  dekadal and monthly totals at 0.05°;
The base resolution is the one of the daily files given in the configuration:
0.25° (CHIRPS ``p25``, the grid the calibration of phase P3 works on) since the
decision of 22/09/2026, 0.05° before that. Coarser archives are derived from it
by exact block averaging, never the other way round.
* ``chirps_1p0_dekads_...nc`` / ``chirps_1p0_months_...nc`` — the same, block
  averaged onto the 1° C3S grid (exact nesting);
* ``chirps_<res>_normals_1991_2020.nc`` (one per resolution) —
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

from eccas_s2s.obs.chirps import expand_paths, open_chirps_daily, process_daily, write_totals
from eccas_s2s.obs.climatology import normals
from eccas_s2s.obs.regrid import block_average, snap_coords
from eccas_s2s.io.netcdf import save as save_cf
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle


def _source_signature(files) -> str:
    """Identity of the daily source files (path, size, modification time of each)."""
    parts = []
    for f in files:
        st = Path(f).stat()
        parts.append(f"{Path(f).resolve()}|{st.st_size}|{int(st.st_mtime)}")
    return ";".join(parts)


def source_files(cfg) -> list[Path]:
    """Daily CHIRPS files of the configuration (``daily_paths``: files or glob patterns)."""
    obs = cfg.raw["observations"]["precip"]
    return expand_paths(obs.get("daily_paths") or obs["daily_path"])


#: CHIRPS resolutions the chain knows, from the daily product to the model grid.
RES_NAME = {0.25: "0p25", 0.05: "p05"}
#: coarser archives derived from each base by exact block averaging.
DERIVED = {"0p25": (("1p0", 1.0),), "p05": (("0p25", 0.25), ("1p0", 1.0))}


def base_resolution(cfg) -> str:
    """Name of the resolution of the daily files (``0p25`` or ``p05``)."""
    deg = float(cfg.raw["observations"]["precip"].get("resolution_deg", 0.25))
    if deg not in RES_NAME:
        raise ValueError(f"résolution CHIRPS non gérée : {deg}°")
    return RES_NAME[deg]


def resolutions(cfg) -> list[str]:
    """Base resolution first, then the archives derived from it."""
    base = base_resolution(cfg)
    return [base] + [name for name, _ in DERIVED[base]]


def derived_paths(cfg) -> dict[str, Path]:
    d = cfg.data_root / "derived" / "obs" / "chirps"
    y0, y1 = cfg.raw["observations"]["precip"].get("years", [1981, 2026])
    n0, n1 = cfg.reference_period("obs_normal")
    paths = {"dir": d, "qc": d / "chirps_qc_monthly.csv"}
    for res in resolutions(cfg):
        paths[f"dekads_{res}"] = d / f"chirps_{res}_dekads_{y0}_{y1}.nc"
        paths[f"months_{res}"] = d / f"chirps_{res}_months_{y0}_{y1}.nc"
        paths[f"normals_{res}"] = d / f"chirps_{res}_normals_{n0}_{n1}.nc"
    return paths


def _is_current(path: Path, signature: str) -> bool:
    if not path.exists():
        return False
    with xr.open_dataset(path) as ds:
        return ds.attrs.get("source_signature") == signature


def load_archives(cfg, resolution: str | None = None) -> tuple[xr.DataArray, xr.DataArray]:
    """Open the dekadal and monthly totals (default: the base resolution of the cycle)."""
    res = resolution or base_resolution(cfg)
    p = derived_paths(cfg)
    return (snap_coords(xr.open_dataset(p[f"dekads_{res}"])["precip"]),
            snap_coords(xr.open_dataset(p[f"months_{res}"])["precip"]))


def load_normals(cfg, resolution: str | None = None) -> xr.Dataset:
    res = resolution or base_resolution(cfg)
    return snap_coords(xr.open_dataset(derived_paths(cfg)[f"normals_{res}"]))


def run(config: str, rebuild: bool = False) -> RunContext:
    cfg = load_cycle(config)
    sources = source_files(cfg)
    paths = derived_paths(cfg)
    signature = _source_signature(sources)
    thr = cfg.thresholds
    n0, n1 = cfg.reference_period("obs_normal")

    with RunContext(cfg, step="obs_chirps") as ctx:
        for f in sources:
            ctx.record_input(f, role="chirps_daily")
        ctx.record_parameter("normal_period", [n0, n1])
        ctx.record_parameter("percentiles", thr["percentiles"])
        ctx.record_parameter("percentile_method", thr["percentile_method"])
        common = {"source_signature": signature,
                  "source_files": ";".join(str(f.resolve()) for f in sources),
                  **ctx.netcdf_attrs()}

        # 1) QC + calendar totals at the base resolution (one streaming pass)
        base = base_resolution(cfg)
        ctx.record_parameter("base_resolution", base)
        if rebuild or not (_is_current(paths[f"dekads_{base}"], signature)
                           and _is_current(paths[f"months_{base}"], signature)
                           and paths["qc"].exists()):
            ctx.log.info("lecture de %d fichier(s) CHIRPS %s mois par mois (QC + cumuls)",
                         len(sources), base)
            da = open_chirps_daily(sources)
            qc, dekads, months = process_daily(
                da, progress=lambda y, m: ctx.log.info("  %d-%02d", y, m) if m == 1 else None)
            paths["dir"].mkdir(parents=True, exist_ok=True)
            qc.to_csv(paths["qc"], index=False)
            write_totals(dekads, paths[f"dekads_{base}"], common)
            write_totals(months, paths[f"months_{base}"], common)
            rebuilt = True
        else:
            ctx.log.info("archives %s à jour (source inchangée) : pas de recalcul", base)
            rebuilt = False
        dekads, months = load_archives(cfg, base)

        # 2) coarser versions by exact block averaging: 0.25° (grid of the
        #    calibration, P3) and 1° (grid of the models, P2)
        for res, target_res in DERIVED[base]:
            if rebuilt or not (_is_current(paths[f"dekads_{res}"], signature)
                               and _is_current(paths[f"months_{res}"], signature)):
                ctx.log.info("moyenne par blocs 0.05° → %.2f°", target_res)
                write_totals(block_average(dekads.load(), target_res).astype("float32"),
                             paths[f"dekads_{res}"], common)
                write_totals(block_average(months.load(), target_res).astype("float32"),
                             paths[f"months_{res}"], common)
                rebuilt = True

        # 3) normals 1991-2020 at every resolution
        for res in resolutions(cfg):
            target = paths[f"normals_{res}"]
            if rebuilt or not _is_current(target, signature):
                ctx.log.info("normales %d-%d à %s", n0, n1, res)
                dk, mo = load_archives(cfg, res)
                norm = normals(dk, mo, (n0, n1), thr["percentiles"], thr["percentile_method"],
                               thr["min_years_normal"])
                norm.attrs.update(common)
                save_cf(norm, target)

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
        keys = ["qc"] + [f"{kind}_{res}" for res in resolutions(cfg)
                         for kind in ("dekads", "months", "normals")]
        for key in keys:
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
