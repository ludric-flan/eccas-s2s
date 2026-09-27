"""
Housekeeping: make the summaries and the figures match the scores (E4, E5).

The netCDF tree is the **authority**: it holds exactly what the last run
computed. The summary tables and the figures are derived from it, and both can
outlive their source — a table because rows are merged rather than replaced, a
figure because nothing deletes a PNG whose period no longer exists.

That happened on NMME: its horizon was capped at the longest C3S horizon
(decision of 23/09/2026), the scores were recomputed on 12 periods, but the
summary kept 22 and the figure tree kept maps for periods that are no longer
produced. This step removes whatever the scores no longer justify, and it is
meant to be run after any change of scope.

Example::

    python scripts/run_housekeeping.py --config config/cycle_202609.yaml --kinds raw calibrated
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from eccas_s2s.io.netcdf import from_cf, open_cf, save as save_cf
from eccas_s2s.operations.skill_raw import skill_dir
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle

SUMMARY = {"raw": "skill_raw_summary.csv", "calibrated": "skill_calibrated_summary.csv"}


def scored_periods(root: Path) -> set[tuple[str, str]]:
    """``(branch, period)`` pairs the netCDF tree actually contains."""
    kept = set()
    for folder in sorted((root / "netcdf").glob("**/")):
        files = sorted(folder.glob("*.nc"))
        if not files:
            continue
        with open_cf(files[0]) as ds:
            periods = [str(p) for p in ds["period"].values]
        branch = str(folder.relative_to(root / "netcdf")).rstrip("/")
        kept |= {(branch, p) for p in periods}
    return kept


def convert_to_cf(config: str, roots=None, dry_run: bool = False) -> RunContext:
    """
    Rewrite existing netCDF files through the CF writer.

    Files written before the CF convention was adopted carry string dimensions,
    which CDO and ncview cannot read ("Unsupported file structure"). Rewriting
    them costs a read and a write per file and changes nothing for the chain,
    which reads both forms.

    A file whose data has five dimensions (year, member, period, lat, lon) stays
    beyond CDO's data model whatever we do — it is made readable by ncview and
    xarray, and the message says so.
    """
    cfg = load_cycle(config)
    targets = [Path(r) for r in roots] if roots else [
        cfg.output_root / "skill" / cfg.cycle_id,
        cfg.output_root / "calibrated" / cfg.cycle_id,
        cfg.data_root / "derived" / "c3s" / cfg.cycle_id,
        cfg.data_root / "derived" / "obs",
    ]
    with RunContext(cfg, step="netcdf_to_cf") as ctx:
        ctx.record_parameter("roots", [str(t) for t in targets])
        ctx.record_parameter("dry_run", bool(dry_run))
        converted = skipped = 0
        for root in targets:
            for path in sorted(Path(root).rglob("*.nc")):
                if path.name.endswith(".tmp.nc"):
                    continue
                with xr.open_dataset(path) as ds:
                    # a textual axis, or any text variable left over: a character
                    # array carried by the time axis is read as a "character
                    # coordinate" by CDO, which then fails on the whole file
                    needs = any(ds[d].dtype.kind in ("U", "O")
                                for d in ds.dims if d in ds.coords)
                    needs = needs or any(v.dtype.kind in ("U", "O", "S")
                                         for v in ds.variables.values())
                    # a `coordinates` attribute naming a variable the file no
                    # longer holds makes CDO warn on every read
                    needs = needs or any(
                        c not in ds.variables
                        for v in ds.variables.values()
                        for c in str(v.attrs.get("coordinates", "")
                                     or v.encoding.get("coordinates", "")).split())
                    data = from_cf(ds.load()) if needs else None
                if not needs:
                    skipped += 1
                    continue
                converted += 1
                if dry_run:
                    continue
                try:
                    save_cf(data, path, init_year=cfg.init_date.year,
                            init_month=cfg.init_date.month)
                except Exception as exc:          # one unusual file must not stop the pass
                    converted -= 1
                    ctx.warn(f"{path.name} : conversion impossible ({exc})")
        ctx.log.info("conversion CF : %d fichier(s) réécrit(s), %d déjà conformes",
                     converted, skipped)
        ctx.record_parameter("converted", converted)
        ctx.record_parameter("already_cf", skipped)
    return ctx


def prune(config: str, kinds=("raw",), dry_run: bool = False) -> RunContext:
    cfg = load_cycle(config)
    with RunContext(cfg, step="housekeeping") as ctx:
        ctx.record_parameter("kinds", list(kinds))
        ctx.record_parameter("dry_run", bool(dry_run))
        for kind in kinds:
            root = skill_dir(cfg, kind)
            if not (root / "netcdf").exists():
                ctx.warn(f"{kind} : aucun score, rien à nettoyer")
                continue
            kept = scored_periods(root)
            kept_periods = {p for _, p in kept}
            ctx.log.info("%s : %d période(s) distincte(s) dans les scores", kind,
                         len(kept_periods))

            # 1) figures whose period is no longer produced
            removed = 0
            for png in sorted((root / "figures").rglob("*.png")):
                period = png.stem.rsplit("_", 1)[0] if png.stem.rsplit("_", 1)[-1] in (
                    "BN", "NN", "AN") else png.stem
                if period not in kept_periods:
                    removed += 1
                    if not dry_run:
                        png.unlink()
            ctx.log.info("%s : %d carte(s) supprimée(s)", kind, removed)
            ctx.record_parameter(f"{kind}_figures_removed", removed)

            # 2) diagrams, same rule
            removed_d = 0
            for png in sorted((root / "diagrams").rglob("*.png")):
                if png.stem not in kept_periods:
                    removed_d += 1
                    if not dry_run:
                        png.unlink()
            if removed_d:
                ctx.log.info("%s : %d diagramme(s) supprimé(s)", kind, removed_d)
            ctx.record_parameter(f"{kind}_diagrams_removed", removed_d)

            # 3) summary rows that no longer have a score behind them
            path = root / SUMMARY[kind]
            if path.exists():
                table = pd.read_csv(path)
                before = len(table)
                table = table[[p in kept_periods for p in table["period"]]]
                ctx.log.info("%s : %d ligne(s) de synthèse retirée(s) sur %d",
                             kind, before - len(table), before)
                ctx.record_parameter(f"{kind}_rows_removed", before - len(table))
                if not dry_run:
                    table.to_csv(path, index=False)
                    ctx.record_output(path, role="summary")
                    if kind == "raw":
                        elig = (table.groupby(["system", "model", "variable", "scale"])["eligible"]
                                .agg(["sum", "count"]).reset_index()
                                .rename(columns={"sum": "periods_eligible", "count": "periods"}))
                        elig["eligible"] = elig["periods_eligible"] > 0
                        reg = cfg.path_of("output_root") / "registry" / "models_eligibility.csv"
                        reg.parent.mkdir(parents=True, exist_ok=True)
                        elig.to_csv(reg, index=False)
                        ctx.record_output(reg, role="eligibility")
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--kinds", nargs="+", default=["raw"], choices=["raw", "calibrated"])
    ap.add_argument("--dry-run", action="store_true", help="lister sans supprimer")
    ap.add_argument("--to-cf", action="store_true",
                    help="réécrire les netCDF existants au format CF (axes lisibles par CDO)")
    ap.add_argument("--roots", nargs="+", help="dossiers à convertir (défaut : skill, derived)")
    args = ap.parse_args(argv)
    if args.to_cf:
        convert_to_cf(args.config, args.roots, args.dry_run)
        return
    prune(args.config, args.kinds, args.dry_run)


if __name__ == "__main__":
    main()
