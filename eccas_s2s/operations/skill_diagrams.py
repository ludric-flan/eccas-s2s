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

Both are drawn from the **pooled grid-point pairs** of each zone
(:mod:`eccas_s2s.validate.pooled`): 24 hindcast years alone cannot fill ten
probability bins. The confidence intervals resample whole years, so the
dependence between neighbouring grid points is not mistaken for extra
information.

A system distributed as an ensemble mean (NMME) has no probability: it gets no
diagram here, only the deterministic scores, until phase P3 calibrates it.

Example::

    python scripts/run_skill_diagrams.py --config config/cycle_202609.yaml \\
        --variables precip --zones domain
"""
from __future__ import annotations

import argparse
import gc

from eccas_s2s.operations.skill_raw import SYSTEM_SCALES, prepare_pairs, skill_dir
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle
from eccas_s2s.validate.pooled import MAX_PIXELS, pooled_frame
from eccas_s2s.validate.r_bridge import RNotAvailable, check_packages, run_zone_diagrams
from eccas_s2s.validate.zones import zone_masks

N_BOOT = 300


def run(config: str, systems=("c3s",), variables=("precip",), models=None, scales=None,
        zones=None, n_boot: int = N_BOOT, max_pixels: int = MAX_PIXELS) -> RunContext:
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
            for model in [m for m in candidates if models is None or m in models]:
                for variable in variables:
                    if system == "nmme":
                        ctx.warn(f"nmme {model} : moyenne d'ensemble, pas de diagramme "
                                 "probabiliste (voir la calibration, P3)")
                        continue
                    if not [s for s in scales if s in SYSTEM_SCALES[system]]:
                        continue
                    try:
                        pairs, periods = prepare_pairs(cfg, system, model, variable, scales, ctx)
                    except (FileNotFoundError, KeyError) as exc:
                        ctx.warn(f"{system} {model} {variable} : données absentes ({exc})")
                        continue
                    if "prob" not in pairs:
                        ctx.warn(f"{system} {model} {variable} : pas de membres, aucun diagramme")
                        continue

                    masks = zone_masks(pairs["obs"], cfg.domains,
                                       valid=pairs["obs"].notnull().all("year").any("period"))
                    labels = {p.key: p.label(cfg.init_date.year) for p in periods}
                    for zone, mask in masks.items():
                        if zones and zone not in zones:
                            continue
                        frame = pooled_frame(pairs, mask, max_pixels=max_pixels)
                        frame["period"] = frame["period"].map(lambda k: labels.get(k, k))
                        zdir = out / "diagrams" / f"{system}_{model}_{variable}" / zone
                        label = (f"{cfg.c3s_models[model].label} — {variable} — {zone} "
                                 f"(init. {cfg.init_date.date()})")
                        try:
                            figures = run_zone_diagrams(frame, zdir, label, n_boot=n_boot)
                        except RNotAvailable as exc:
                            ctx.warn(f"diagrammes {system} {model} {variable} {zone} : {exc}")
                            continue
                        ctx.log.info("%s %s %s %s : %d figure(s), %d paires",
                                     system, model, variable, zone, len(figures), len(frame))
                        for f in figures:
                            ctx.record_output(f, role="skill_diagram", system=system,
                                              model=model, variable=variable, zone=zone)
                        for name in ("diagram_scores.csv", "diagram_bins.csv"):
                            if (zdir / name).exists():
                                ctx.record_output(zdir / name, role="diagram_table",
                                                  system=system, model=model,
                                                  variable=variable, zone=zone)
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
    ap.add_argument("--zones", nargs="+", help="par défaut toutes les zones de domains.yaml")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    args = ap.parse_args(argv)
    run(args.config, args.systems, args.variables, args.models, args.scales, args.zones,
        args.n_boot, args.max_pixels)


if __name__ == "__main__":
    main()
