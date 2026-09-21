"""
Skill maps of the raw hindcasts (workflow step E4, figures).

Reads the score files written by :mod:`eccas_s2s.operations.skill_raw` and draws
them in the CAPC-AC house style (CEEAC shapefile, logo, four-line title), with
:mod:`eccas_s2s.viz.ceeac_maps`.

Two kinds of figure per model and variable:

* one **panel per score**, with all the periods of a scale side by side
  (``<system>_<model>_<variable>/<scale>_<score>.png``);
* one **multi-model panel per score and period**, to compare the models at a
  glance (``comparison/<variable>_<period>_<score>.png``).

Example::

    python scripts/run_plot_skill_raw.py --config config/cycle_202609.yaml --scores pearson rpss
"""
from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from eccas_s2s.operations.skill_raw import skill_dir, split_skill_name
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle
from eccas_s2s.viz.ceeac_maps import SKILL_STYLES, map_panel

DEFAULT_SCORES = ("pearson", "rpss", "roc_area", "msess")
VARIABLE_LABEL = {"precip": "Pluie", "t2m": "Température moyenne",
                  "tmax": "Température maximale", "tmin": "Température minimale"}


def _model_label(cfg, system: str, model: str) -> str:
    if system == "c3s":
        return cfg.c3s_models[model].label
    return cfg.raw["systems"]["nmme"]["models"].get(model, {}).get("label", model)


def run(config: str, scores=DEFAULT_SCORES, scales=None, category: str = "AN",
        comparison: bool = True) -> RunContext:
    cfg = load_cycle(config)
    root = skill_dir(cfg)
    figures = root / "figures"
    shapefile = cfg.raw["paths"]["shapefile"]
    logo = cfg.raw["paths"].get("logo")
    extent = tuple(cfg.domains["domain"]["map_extent"])
    scales = list(scales) if scales else cfg.scales
    init = cfg.init_date.date()

    with RunContext(cfg, step="plot_skill_raw") as ctx:
        ctx.record_parameter("scores", list(scores))
        ctx.record_parameter("scales", scales)
        ctx.record_parameter("category", category)
        files = sorted((root / "maps").glob("*_skill.nc"))
        if not files:
            raise FileNotFoundError(f"aucune carte de skill dans {root / 'maps'} "
                                    "(lancer d'abord run_skill_raw.py)")
        by_key: dict[tuple[str, str], list] = {}

        for f in files:
            system, model, variable = split_skill_name(f.stem)
            ds = xr.open_dataset(f)
            ctx.record_input(f, role="skill_maps")
            label = _model_label(cfg, system, model)
            for scale in scales:
                sel = ds.sel(period=[p for p in ds["period"].values
                                     if str(ds["scale"].sel(period=p).values) == scale])
                if not sel.sizes.get("period"):
                    continue
                titles = [str(x) for x in sel["label"].values]
                for score in scores:
                    if score not in sel:
                        continue
                    field = sel[score]
                    if "category" in field.dims:
                        field = field.sel(category=category)
                    style = SKILL_STYLES.get(score, {"cmap": "viridis", "levels": None,
                                                     "label": score})
                    name = f"{score} ({category})" if "category" in sel[score].dims else score
                    out = figures / f"{system}_{model}_{variable}" / f"{scale}_{score}.png"
                    map_panel([field.sel(period=p) for p in sel["period"].values], titles,
                              shapefile=shapefile, logo=logo, extent=extent,
                              cmap=style["cmap"], levels=style["levels"],
                              cbar_label=style["label"], ncols=min(6, sel.sizes["period"]),
                              suptitle=(f"{system.upper()} {label} — {VARIABLE_LABEL.get(variable, variable)}"
                                        f"\nSkill brut ({name}), hindcast {ds.attrs.get('n_years', '')} ans"
                                        f" — Initialisation : {init}"),
                              output_path=out)
                    ctx.record_output(out, role="skill_figure", system=system, model=model,
                                      variable=variable, score=score, scale=scale)
                    for p, title in zip(sel["period"].values, titles):
                        by_key.setdefault((variable, str(p), score), []).append(
                            (f"{label}", field.sel(period=p), title))
            ds.close()

        if comparison:
            for (variable, period, score), items in by_key.items():
                if len(items) < 2:
                    continue
                style = SKILL_STYLES.get(score, {"cmap": "viridis", "levels": None, "label": score})
                out = figures / "comparison" / f"{variable}_{period}_{score}.png"
                map_panel([f for _, f, _ in items], [m for m, _, _ in items],
                          shapefile=shapefile, logo=logo, extent=extent, cmap=style["cmap"],
                          levels=style["levels"], cbar_label=style["label"], ncols=4,
                          suptitle=(f"{VARIABLE_LABEL.get(variable, variable)} — {items[0][2]}"
                                    f"\n{score} brut, tous modèles — Initialisation : {init}"),
                          output_path=out)
                ctx.record_output(out, role="skill_comparison", variable=variable,
                                  period=period, score=score)
        ctx.log.info("%d figure(s) écrite(s) dans %s", len(ctx.outputs), figures)
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scores", nargs="+", default=list(DEFAULT_SCORES))
    ap.add_argument("--scales", nargs="+", choices=["decade", "month", "season"])
    ap.add_argument("--category", default="AN", choices=["BN", "NN", "AN"])
    ap.add_argument("--no-comparison", action="store_true")
    args = ap.parse_args(argv)
    run(args.config, args.scores, args.scales, args.category, not args.no_comparison)


if __name__ == "__main__":
    main()
