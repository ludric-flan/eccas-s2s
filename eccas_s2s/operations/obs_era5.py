"""
Observed temperature reference: ERA5 archive and normals (workflow step E1).

Builds, once, the derived archives shared by all cycles, in
``<data_root>/derived/obs/era5/``:

* ``era5_0p25_dekads_<Y0>_<Y1>.nc`` / ``era5_0p25_months_<Y0>_<Y1>.nc`` —
  dekadal and monthly means of ``tmean``, ``tmax`` and ``tmin`` (°C);
* ``era5_1p0_...`` — the same on the 1° C3S grid (conservative remapping: the
  0.25° cells are not nested in the 1° cells);
* ``era5_0p25_normals_1991_2020.nc`` and ``era5_1p0_normals_1991_2020.nc`` —
  mean, standard deviation and percentiles per grid point and calendar period,
  which give the **extreme percentiles per grid point** of decision D13
  (Tmax > P90/P95, Tmin > P90/P95, Tmin < P10/P5, T2m outside P20–P80);
* ``era5_daily_qc_<Y0>_<Y1>.csv`` — monthly quality control.

The step is idempotent: unchanged source files are not processed again unless
``rebuild=True``.

Example::

    python scripts/run_obs_era5.py --config config/cycle_202609.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from eccas_s2s.obs.era5 import VARIABLES, expand_paths, open_era5_daily, process_daily
from eccas_s2s.obs.regrid import conservative_to_degree, snap_coords
from eccas_s2s.obs.climatology import normals
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle


def source_files(cfg) -> list[Path]:
    """Daily ERA5 files of the configuration (``daily_paths``: files or glob patterns)."""
    obs = cfg.raw["observations"]["temperature"]
    return expand_paths(obs.get("daily_paths") or obs["daily_path"])


def _signature(files) -> str:
    parts = []
    for f in files:
        st = Path(f).stat()
        parts.append(f"{Path(f).resolve()}|{st.st_size}|{int(st.st_mtime)}")
    return ";".join(parts)


def derived_paths(cfg) -> dict[str, Path]:
    d = cfg.data_root / "derived" / "obs" / "era5"
    files = source_files(cfg)
    y0 = int(Path(files[0]).stem.split("_")[-1])
    y1 = int(Path(files[-1]).stem.split("_")[-1])
    n0, n1 = cfg.reference_period("obs_normal")
    return {
        "dir": d,
        "dekads_0p25": d / f"era5_0p25_dekads_{y0}_{y1}.nc",
        "months_0p25": d / f"era5_0p25_months_{y0}_{y1}.nc",
        "dekads_1p0": d / f"era5_1p0_dekads_{y0}_{y1}.nc",
        "months_1p0": d / f"era5_1p0_months_{y0}_{y1}.nc",
        "normals_0p25": d / f"era5_0p25_normals_{n0}_{n1}.nc",
        "normals_1p0": d / f"era5_1p0_normals_{n0}_{n1}.nc",
        "qc": d / f"era5_daily_qc_{y0}_{y1}.csv",
    }


def _is_current(path: Path, signature: str) -> bool:
    if not path.exists():
        return False
    with xr.open_dataset(path) as ds:
        return ds.attrs.get("source_signature") == signature


def load_archives(cfg, resolution: str = "0p25") -> tuple[xr.Dataset, xr.Dataset]:
    """Open the dekadal and monthly means (``resolution`` = "0p25" or "1p0")."""
    p = derived_paths(cfg)
    return (snap_coords(xr.open_dataset(p[f"dekads_{resolution}"])),
            snap_coords(xr.open_dataset(p[f"months_{resolution}"])))


def load_normals(cfg, variable: str, resolution: str = "0p25") -> xr.Dataset:
    """Normals of one temperature variable (``tmean``, ``tmax`` or ``tmin``)."""
    ds = snap_coords(xr.open_dataset(derived_paths(cfg)[f"normals_{resolution}"]))
    return ds.sel(variable=variable)


def _write(ds: xr.Dataset, path: Path, attrs: dict) -> Path:
    ds = ds.copy()
    ds.attrs.update(attrs)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    tmp = path.with_suffix(".tmp.nc")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(tmp, encoding=enc)
    tmp.replace(path)
    return path


def run(config: str, rebuild: bool = False) -> RunContext:
    cfg = load_cycle(config)
    files = source_files(cfg)
    paths = derived_paths(cfg)
    signature = _signature(files)
    thr = cfg.thresholds
    n0, n1 = cfg.reference_period("obs_normal")

    with RunContext(cfg, step="obs_era5") as ctx:
        for f in files:
            ctx.record_input(f, role="era5_daily")
        ctx.record_parameter("normal_period", [n0, n1])
        ctx.record_parameter("percentiles", thr["percentiles"])
        common = {"source_signature": signature,
                  "source_files": ";".join(str(f.resolve()) for f in files), **ctx.netcdf_attrs()}

        if rebuild or not (_is_current(paths["dekads_0p25"], signature)
                           and _is_current(paths["months_0p25"], signature) and paths["qc"].exists()):
            ctx.log.info("lecture de %d fichier(s) ERA5 année par année (QC + moyennes)", len(files))
            qc, dekads, months = process_daily(open_era5_daily(files),
                                               progress=lambda y: ctx.log.info("  %d", y))
            paths["dir"].mkdir(parents=True, exist_ok=True)
            qc.to_csv(paths["qc"], index=False)
            _write(dekads, paths["dekads_0p25"], common)
            _write(months, paths["months_0p25"], common)
            rebuilt = True
        else:
            ctx.log.info("archives 0.25° à jour : pas de recalcul")
            rebuilt = False
        dekads, months = load_archives(cfg, "0p25")

        if rebuilt or not (_is_current(paths["dekads_1p0"], signature)
                           and _is_current(paths["months_1p0"], signature)):
            ctx.log.info("remaillage conservatif 0.25° → 1°")
            for key, src in (("dekads_1p0", dekads), ("months_1p0", months)):
                coarse = xr.Dataset({v: conservative_to_degree(src[v].load(), 1.0) for v in VARIABLES})
                if "dekad" in src.coords:
                    coarse = coarse.assign_coords(dekad=src["dekad"])
                _write(coarse, paths[key], common)
            rebuilt = True

        for res in ("0p25", "1p0"):
            target = paths[f"normals_{res}"]
            if rebuilt or not _is_current(target, signature):
                ctx.log.info("normales %d-%d à %s", n0, n1, res)
                dk, mo = load_archives(cfg, res)
                per_var = {v: normals(dk[v], mo[v], (n0, n1), thr["percentiles"],
                                      thr["percentile_method"], thr["min_years_normal"], how="mean")
                           for v in VARIABLES}
                norm = xr.concat(list(per_var.values()), dim=pd.Index(list(per_var), name="variable"))
                _write(norm, target, common)

        qc = pd.read_csv(paths["qc"])
        bad = qc[(qc.days_present != qc.days_expected) | (qc.missing_values > 0)]
        if len(bad):
            ctx.warn(f"{len(bad)} mois avec jours ou valeurs manquants")
        out_of_range = int(qc.outside_range.sum())
        if out_of_range:
            ctx.warn(f"{out_of_range} valeur(s) journalière(s) hors de la plage plausible")
        ctx.record_parameter("qc_months_flagged", int(len(bad)))
        ctx.record_parameter("qc_values_outside_range", out_of_range)
        for key in ("qc", "dekads_0p25", "months_0p25", "dekads_1p0", "months_1p0",
                    "normals_0p25", "normals_1p0"):
            ctx.record_output(paths[key], role=f"era5_{key}")
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    run(args.config, args.rebuild)


if __name__ == "__main__":
    main()
