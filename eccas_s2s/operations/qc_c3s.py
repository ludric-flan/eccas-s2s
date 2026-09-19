"""
Quality control of the raw C3S files of a cycle (workflow step E2).

Writes ``<output_root>/qc/<YYYYMM>/c3s_<variable>_files.csv`` (one row per file)
and ``c3s_<variable>_horizons.csv`` (usable horizon and members per model), with
a run manifest.

Example::

    python scripts/run_qc_c3s.py --config config/cycle_202609.yaml --variable precip
"""
from __future__ import annotations

import argparse

from eccas_s2s.io.c3s_qc import effective_horizons, qc_table
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle

VARIABLE_KEYS = {"precip": "PRCP", "t2m": "TEMP"}


def run(config: str, variable: str = "precip"):
    """Run the QC and return ``(files_table, horizons_table, run_context)``."""
    cfg = load_cycle(config)
    raw_dir = cfg.raw_dir("c3s")
    pattern = f"c3s_*_{VARIABLE_KEYS[variable]}_*_{cfg.init_date:%m}.grib"
    out_dir = cfg.output_root / "qc" / cfg.cycle_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with RunContext(cfg, step=f"qc_c3s_{variable}") as ctx:
        files = sorted(raw_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"Aucun fichier {pattern} dans {raw_dir}")
        ctx.log.info("%d fichier(s) à contrôler dans %s", len(files), raw_dir)
        for f in files:
            ctx.record_input(f, role="c3s_raw")

        qc = qc_table(files)
        horizons = effective_horizons(qc)
        expected = set(cfg.c3s_models)
        present = set(qc["centre"])
        for centre in sorted(expected - present):
            ctx.warn(f"{centre} : aucun fichier trouvé")
        for _, row in qc[qc["warnings"] != ""].iterrows():
            ctx.warn(f"{row['file']} : {row['warnings']}")
        for _, h in horizons.iterrows():
            declared = cfg.c3s_models[h["centre"]].max_lead_days
            if h["usable_days"] != declared:
                ctx.warn(f"{h['centre']} : horizon utilisable {h['usable_days']} j "
                         f"≠ max_lead_days {declared} j dans la configuration")

        f_files = out_dir / f"c3s_{variable}_files.csv"
        f_hor = out_dir / f"c3s_{variable}_horizons.csv"
        qc.to_csv(f_files, index=False)
        horizons.to_csv(f_hor, index=False)
        ctx.record_output(f_files, role="qc_files")
        ctx.record_output(f_hor, role="qc_horizons")
        ctx.record_parameter("usable_days", dict(zip(horizons["centre"], horizons["usable_days"])))
    return qc, horizons, ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variable", default="precip", choices=sorted(VARIABLE_KEYS))
    args = ap.parse_args(argv)
    run(args.config, args.variable)


if __name__ == "__main__":
    main()
