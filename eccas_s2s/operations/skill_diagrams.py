"""
Reliability and ROC diagrams of the raw hindcasts (workflow step E4, figures).

Complements the skill **maps** (:mod:`eccas_s2s.operations.plot_skill_raw`) with
the two diagrams the WMO guidance asks for on a probabilistic forecast, drawn in
R with the ``verification`` package as in the reference chain of the CAPC-AC:

* the **attributes (reliability) diagram**: does a "60 % chance of above normal"
  actually verify six times out of ten? With the no-skill line, the skill region
  and a sharpness inset;
* the **ROC curve**: can the system separate the years when the category occurs
  from the years when it does not, whatever the calibration?

Both are drawn from the **pooled grid-point pairs of the CEEAC mask**
(:mod:`eccas_s2s.validate.pooled`): 24 hindcast years alone cannot fill ten
probability bins. The confidence intervals resample whole years, so the
dependence between neighbouring grid points is not mistaken for extra
information.

Output, on the same branches as the scores and the maps::

    diagrams/<system>_<model>/<scale>/<variable>/reliability/<period>.png
    diagrams/<system>_<model>/<scale>/<variable>/roc/<period>.png

A system distributed as an ensemble mean (NMME) has no probability: it gets no
diagram here, only the deterministic scores, until phase P3 calibrates it.

Example::

    python scripts/run_skill_diagrams.py --config config/cycle_202609.yaml \\
        --variables precip --scales season
"""
from __future__ import annotations

import argparse
import gc

from eccas_s2s.operations.skill_raw import (SYSTEM_SCALES, prepare_pairs, score_paths,
                                            skill_dir)
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle
from eccas_s2s.validate.pooled import MAX_PIXELS, pooled_frame
from eccas_s2s.validate.r_bridge import RNotAvailable, check_packages, run_zone_diagrams

N_BOOT = 300


def run(config: str, systems=("c3s",), variables=("precip",), models=None, scales=None,
        n_boot: int = N_BOOT, max_pixels: int = MAX_PIXELS) -> RunContext:
    cfg = load_cycle(config)
    out = skill_dir(cfg)
    scales = list(scales) if scales else cfg.scales

    with RunContext(cfg, step="skill_diagrams") as ctx:
        info = check_packages()
        ctx.record_parameter("r_version", info["r_version"])
        ctx.record_parameter("n_boot", n_boot)
        ctx.record_parameter("max_pixels", max_pixels)
        ctx.record_parameter("scales", scales)

        for system in systems:
            candidates = (list(cfg.c3s_models) if system == "c3s"
                          else list(cfg.raw["systems"]["nmme"]["models"]))
            sys_scales = [s for s in scales if s in SYSTEM_SCALES[system]]
            for model in [m for m in candidates if models is None or m in models]:
                for variable in variables:
                    if system == "nmme":
                        ctx.warn(f"nmme {model} : moyenne d'ensemble, pas de diagramme "
                                 "probabiliste (voir la calibration, P3)")
                        continue
                    if not sys_scales:
                        continue
                    try:
                        pairs, periods = prepare_pairs(cfg, system, model, variable,
                                                       sys_scales, ctx)
                    except (FileNotFoundError, KeyError) as exc:
                        ctx.warn(f"{system} {model} {variable} : données absentes ({exc})")
                        continue
                    if "prob" not in pairs:
                        ctx.warn(f"{system} {model} {variable} : pas de membres, aucun diagramme")
                        continue

                    # the observation is already restricted to the CEEAC mask, so
                    # the pooled sample covers the verified cells and no other
                    mask = pairs["obs"].notnull().any(["year", "period"])
                    labels = {p.key: p.label_fr(cfg.init_date.year, with_dates=True)
                              for p in periods}
                    model_label = (cfg.c3s_models[model].label if system == "c3s"
                                   else model)
                    for scale in sorted({p.scale for p in periods}):
                        keys = [p.key for p in periods if p.scale == scale]
                        frame = pooled_frame(pairs.sel(period=keys), mask,
                                             max_pixels=max_pixels)
                        frame["period_label"] = frame["period"].map(labels)
                        zdir = score_paths(out / "diagrams", system, model, variable, scale)
                        label = (f"{model_label} — {variable} — masque CEEAC "
                                 f"(init. {cfg.init_date.date()})")
                        try:
                            figures = run_zone_diagrams(frame, zdir, label, n_boot=n_boot)
                        except RNotAvailable as exc:
                            ctx.warn(f"diagrammes {system} {model} {variable} {scale} : {exc}")
                            continue
                        ctx.log.info("%s %s %s %s : %d figure(s), %d paires",
                                     system, model, variable, scale, len(figures), len(frame))
                        for f in figures:
                            ctx.record_output(f, role="skill_diagram", system=system,
                                              model=model, variable=variable, scale=scale)
                        for name in ("diagram_scores.csv", "diagram_bins.csv"):
                            if (zdir / name).exists():
                                ctx.record_output(zdir / name, role="diagram_table",
                                                  system=system, model=model,
                                                  variable=variable, scale=scale)
                    del pairs
                    gc.collect()
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--systems", nargs="+", default=["c3s"], choices=["c3s", "nmme"])
    ap.add_argument("--variables", nargs="+", default=["precip"],
                    choices=["precip", "t2m", "tmax", "tmin"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--scales", nargs="+", choices=["decade", "month", "season"])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    args = ap.parse_args(argv)
    run(args.config, args.systems, args.variables, args.models, args.scales,
        args.n_boot, args.max_pixels)


if __name__ == "__main__":
    main()
