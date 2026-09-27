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

from eccas_s2s.calibrate.base import RawEnsemble
from eccas_s2s.core.periods import (add_period_arguments, announce_subset,
                                    selected_periods)
from eccas_s2s.products.catalogue import FAMILY_LABEL, families
from eccas_s2s.validate.pooled import family_frame, family_series
from eccas_s2s.validate.product_scores import ObservationContext
from eccas_s2s.operations.skill_raw import (SYSTEM_SCALES, hindcast_streams,
                                            prepare_pairs, score_paths, skill_dir)
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle
from eccas_s2s.validate.pooled import MAX_PIXELS
from eccas_s2s.validate.r_bridge import RNotAvailable, check_packages, run_zone_diagrams

N_BOOT = 300
#: these diagrams describe the *raw* hindcast; the word opens their titles.
KIND = "Raw"
KIND_LABEL = {"raw": "Raw", "calibrated": "Calibrated"}
VARIABLE_LABEL = {"precip": "Rainfall", "t2m": "Température moyenne",
                  "tmax": "Température maximale", "tmin": "Température minimale"}


def family_method(register, system: str, model: str, variable: str, scale: str,
                  products) -> str:
    """
    Method whose diagrams represent a family.

    Every class of a family must come from the **same** predictive distribution —
    three curves fitted by three different methods could not be compared on one
    set of axes. The family therefore takes the method recommended for the
    majority of its products (ties going to the first product of the family,
    which is its reference: the terciles for the tercile family, the SPI classes
    for the SPI one).
    """
    if register is None or register.empty:
        return ""
    names = [p.name for p in products]
    sub = register[(register["system"] == system) & (register["model"] == model)
                   & (register["variable"] == variable) & (register["scale"] == scale)
                   & (register["product"].isin(names))]
    if sub.empty:
        return ""
    counts = sub["method"].value_counts()
    top = counts[counts == counts.max()].index
    if len(top) == 1:
        return str(top[0])
    first = sub[sub["product"] == names[0]]["method"]
    return str(first.iloc[0]) if len(first) else str(top[0])


def run(config: str, systems=("c3s",), variables=("precip",), models=None, scales=None,
        n_boot: int = N_BOOT, max_pixels: int = MAX_PIXELS,
        selection=None, kind: str = "raw") -> RunContext:
    """
    Reliability and ROC diagrams of one tree, family by family.

    ``kind="calibrated"`` draws the diagrams of phase P3: for each family, the
    **recommended** method is refitted on the observation grid — one method
    instead of ten — and its probabilities feed the same figures as the raw ones,
    so the two can be laid side by side.
    """
    cfg = load_cycle(config)
    out = skill_dir(cfg, kind)
    word = KIND_LABEL.get(kind, KIND)
    register = None
    if kind == "calibrated":
        import pandas as pd

        reg_path = cfg.path_of("output_root") / "registry" / "calibration_methods.csv"
        if not reg_path.exists():
            raise FileNotFoundError(
                f"{reg_path} absent : lancer d'abord run_calibrate_hindcast.py")
        register = pd.read_csv(reg_path)
    scales = list(scales) if scales else cfg.scales

    # a system switched off in the configuration is skipped, not attempted
    systems = [s for s in systems if s in cfg.enabled_systems()]
    if not systems:
        raise ValueError("aucun système actif : voir `enabled` dans la configuration du cycle")

    with RunContext(cfg, step=f"skill_diagrams_{kind}") as ctx:
        ctx.record_parameter("kind", kind)
        announce_subset(cfg, ctx, selection)
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
                    groups = hindcast_streams(cfg, system, model, variable, sys_scales)
                    for stream, stream_scales in groups.items():
                        if not stream_scales:
                            continue
                        try:
                            pairs, periods = prepare_pairs(cfg, system, model, variable,
                                                           stream_scales, ctx, selection,
                                                           stream)
                        except (FileNotFoundError, KeyError) as exc:
                            ctx.warn(f"{system} {model} {variable} ({stream}) : "
                                     f"données absentes ({exc})")
                            continue
                        if "members" not in pairs:
                            ctx.warn(f"{system} {model} {variable} ({stream}) : "
                                     "pas de membres, aucun diagramme")
                            continue

                        # the observation is already restricted to the CEEAC mask, so
                        # the pooled sample covers the verified cells and no other
                        mask = pairs["obs"].notnull().any(["year", "period"])
                        model_label = (cfg.c3s_models[model].label if system == "c3s" else model)
                        var_thresholds = cfg.thresholds.get(variable, {})
                        n_figures = 0
                        for scale in sorted({p.scale for p in periods}):
                            groups_of_family = families(variable, scale, var_thresholds)
                            for period in [p for p in periods if p.scale == scale]:
                                key = period.key
                                ref = pairs["obs"].sel(period=key, drop=True)
                                octx = ObservationContext(ref, variable, scale, var_thresholds,
                                                          mask, "year")
                                raw_dist = RawEnsemble(
                                    pairs["members"].sel(period=key, drop=True))
                                # the calibrated distribution lives on the observation
                                # grid (0.25°): its context and its mask must too, or
                                # the pooled sample would look for 1° cells in a 0.25°
                                # field and come back empty
                                fine = {}
                                for family, products in groups_of_family.items():
                                    dist, method = raw_dist, ""
                                    fam_ctx, fam_mask = octx, mask
                                    if kind == "calibrated":
                                        method = family_method(register, system, model,
                                                               variable, scale, products)
                                        if not method or method == "raw":
                                            ctx.log.info("  %s %s : méthode recommandée « %s »"
                                                         " — diagrammes bruts déjà tracés",
                                                         family, key, method or "aucune")
                                            continue
                                        got = _calibrated_distribution(
                                            cfg, system, model, variable, scale, key,
                                            method, pairs, ctx, cache=fine)
                                        if got is None:
                                            continue
                                        dist, obs_fine = got
                                        fam_mask = obs_fine.notnull().any("year")
                                        fam_ctx = ObservationContext(obs_fine, variable, scale,
                                                                     var_thresholds, fam_mask,
                                                                     "year")
                                    series = family_series(dist, fam_ctx, family, products)
                                    frame = family_frame(
                                        series, fam_mask, key,
                                        period.label_fr(cfg.init_date.year),
                                        max_pixels=max_pixels,
                                        drop_degenerate=family == "seuils_cumul")
                                    if frame.empty:
                                        ctx.warn(f"{system} {model} {variable} {scale} "
                                                 f"{family} {key} : aucune paire exploitable")
                                        continue
                                    branch = family if not method else f"{family}/{method}"
                                    zdir = score_paths(out / "diagrams", system, model,
                                                       variable, scale, branch)
                                    label = (f"{model_label} — "
                                             f"{VARIABLE_LABEL.get(variable, variable)}"
                                             f" — Init : {cfg.init_date.date()} | "
                                             f"Hindcast Period : {pairs.attrs.get('years', '')}")
                                    try:
                                        figures = run_zone_diagrams(
                                            frame, zdir, label, n_boot=n_boot, kind=word,
                                            family=FAMILY_LABEL.get(family, family))
                                    except RNotAvailable as exc:
                                        ctx.warn(f"diagrammes {system} {model} {variable} "
                                                 f"{scale} {family} : {exc}")
                                        continue
                                    n_figures += len(figures)
                                    for f in figures:
                                        ctx.record_output(f, role="skill_diagram", system=system,
                                                          model=model, variable=variable,
                                                          scale=scale, family=family, period=key)
                                    for name in ("diagram_scores.csv", "diagram_bins.csv"):
                                        if (zdir / name).exists():
                                            ctx.record_output(zdir / name, role="diagram_table",
                                                              system=system, model=model,
                                                              variable=variable, scale=scale,
                                                              family=family)
                            ctx.log.info("%s %s %s %s [%s] : %d figure(s) pour %d famille(s)",
                                         system, model, variable, scale, stream, n_figures,
                                         len(groups_of_family))
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
    ap.add_argument("--kind", default="raw", choices=["raw", "calibrated"],
                    help="arbre à tracer : brut (P2) ou calibré (P3, méthode recommandée)")
    add_period_arguments(ap)
    args = ap.parse_args(argv)
    run(args.config, args.systems, args.variables, args.models, args.scales,
        args.n_boot, args.max_pixels, selection=selected_periods(args), kind=args.kind)


if __name__ == "__main__":
    main()


def _calibrated_distribution(cfg, system: str, model: str, variable: str, scale: str,
                             period_key: str, method: str, pairs, ctx=None, cache=None):
    """
    Refit one calibration method for one period, on the observation grid.

    The diagrams need the predictive **distribution**, which a file of scores
    cannot carry; refitting is therefore the honest way to draw them — and it
    costs one method instead of the ten the register compared. The fit follows
    exactly the path of :mod:`eccas_s2s.operations.calibrate_hindcast`: members
    interpolated onto the observation grid, leave-one-year-out, same observation.
    """
    from eccas_s2s.operations.calibrate_hindcast import (calibrators_for, model_on_grid,
                                                         observation_on_grid)

    years = [int(y) for y in pairs["year"].values]
    from eccas_s2s.core.periods import build_periods

    periods = [p for p in build_periods(cfg.init_date, 400, (scale,)) if p.key == period_key]
    if not periods:
        return None
    cache = {} if cache is None else cache
    obs = cache.get("obs")
    if obs is None:
        obs = cache["obs"] = observation_on_grid(cfg, variable, periods,
                                                 years).sel(period=period_key, drop=True)
    members = pairs["members"].sel(period=period_key, drop=True)
    ens = cache.get("ens")
    if ens is None:
        ens = cache["ens"] = model_on_grid(members, obs)
    cals = {c.name: c for c in calibrators_for(variable, has_members=True)}
    cal = cals.get(method)
    if cal is None:
        if ctx:
            ctx.warn(f"méthode « {method} » inconnue pour {model} {variable} {scale}")
        return None
    if ctx:
        ctx.log.info("  ré-ajustement de « %s » pour %s %s %s", method, model, scale, period_key)
    return cal.fit_predict_loyo(ens, obs), obs
