"""
Quality control of the raw C3S files of a cycle (workflow step E2).

Writes ``<output_root>/qc/<YYYYMM>/c3s_<variable>_files.csv`` (one row per
file, daily and monthly, column ``stream``) and ``c3s_<variable>_horizons.csv``
(usable horizon and members per model and stream: ``usable_days`` for the daily
stream, ``usable_months`` for the monthly one), with a run manifest.

Each model is expected to have the streams its configuration asks for
(``precip_from``, or the monthly statistics for the T2m): a missing stream is a
warning, and so is a monthly horizon shorter than ``horizon.max_lead_months``.

Example::

    python scripts/run_qc_c3s.py --config config/cycle_202611.yaml --variables precip tmoy tmax tmin
"""
from __future__ import annotations

import argparse

import pandas as pd

from eccas_s2s.io.c3s_qc import effective_horizons, monthly_horizons, qc_table
from eccas_s2s.operations.download_c3s import (VARIABLE_KEYS, canonical_variables,
                                               streams_needed)
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle


def run(config: str, variable: str = "precip"):
    """Run the QC and return ``(files_table, horizons_table, run_context)``."""
    cfg = load_cycle(config)
    variable = canonical_variables(variable)[0]
    raw_dir = cfg.raw_dir("c3s")
    key, mm = VARIABLE_KEYS[variable], f"{cfg.init_date:%m}"
    patterns = {"daily": f"c3s_*_{key}_*_{mm}.grib",
                "monthly": f"c3s_*_{key}_*_{mm}_monthly.grib"}
    out_dir = cfg.output_root / "qc" / cfg.cycle_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with RunContext(cfg, step=f"qc_c3s_{variable}") as ctx:
        files = sorted({f for pat in patterns.values() for f in raw_dir.glob(pat)})
        if not files:
            raise FileNotFoundError(f"Aucun fichier {' ni '.join(patterns.values())} dans {raw_dir}")
        ctx.log.info("%d fichier(s) à contrôler dans %s", len(files), raw_dir)
        for f in files:
            ctx.record_input(f, role="c3s_raw")

        qc = qc_table(files)
        ctx.log.info("flux : %s", qc["stream"].value_counts().to_dict())
        # chaque modèle doit avoir les flux que sa configuration demande
        for centre, model in cfg.c3s_models.items():
            for stream in streams_needed(model, variable):
                for kind in ("forecast", "hindcast"):
                    got = qc[(qc["centre"] == centre) & (qc["stream"] == stream)
                             & (qc["kind"] == kind)]
                    if got.empty:
                        ctx.warn(f"{centre} : fichier {kind} du flux {stream} absent")
        for _, row in qc[qc["warnings"] != ""].iterrows():
            ctx.warn(f"{row['file']} : {row['warnings']}")

        daily = effective_horizons(qc)
        for _, h in daily.iterrows():
            declared = cfg.c3s_models[h["centre"]].max_lead_days
            if h["usable_days"] != declared:
                ctx.warn(f"{h['centre']} : horizon utilisable {h['usable_days']} j "
                         f"≠ max_lead_days {declared} j dans la configuration")
        monthly = monthly_horizons(qc)
        for _, h in monthly.iterrows():
            if cfg.max_lead_months and h["usable_months"] < cfg.max_lead_months:
                ctx.warn(f"{h['centre']} : {h['usable_months']} mois complets dans le flux "
                         f"mensuel < horizon commun de {cfg.max_lead_months} mois")
        horizons = pd.concat([daily.assign(stream="daily"), monthly.assign(stream="monthly")],
                             ignore_index=True).convert_dtypes()
        if not horizons.empty:
            first = ["centre", "stream"]
            horizons = horizons[first + [c for c in horizons.columns if c not in first]]

        f_files = out_dir / f"c3s_{variable}_files.csv"
        f_hor = out_dir / f"c3s_{variable}_horizons.csv"
        qc.to_csv(f_files, index=False)
        horizons.to_csv(f_hor, index=False)
        ctx.record_output(f_files, role="qc_files")
        ctx.record_output(f_hor, role="qc_horizons")
        ctx.record_parameter("usable_days", dict(zip(daily.get("centre", []),
                                                     daily.get("usable_days", []))))
        ctx.record_parameter("usable_months", dict(zip(monthly.get("centre", []),
                                                       monthly.get("usable_months", []))))
    return qc, horizons, ctx


def run_variables(config: str, variables=("precip",)) -> dict:
    """QC of several variables, daily and monthly files: ``{variable: (files, horizons, ctx)}``."""
    return {variable: run(config, variable) for variable in canonical_variables(variables)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variables", "--variable", nargs="+", default=["precip"],
                    help="une ou plusieurs variables : precip, t2m (ou tmoy), tmax, tmin")
    args = ap.parse_args(argv)
    run_variables(args.config, args.variables)


if __name__ == "__main__":
    main()
