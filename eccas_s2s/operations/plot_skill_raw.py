"""
Skill maps of the raw hindcasts (workflow step E4, figures).

Reads the score files written by :mod:`eccas_s2s.operations.skill_raw`
(``netcdf/<system>_<model>/<scale>/<variable>/<metric>.nc``) and draws them in
the CAPC-AC house style with :mod:`eccas_s2s.viz.ceeac_maps`.

**One map per period**: a panel of six seasons is convenient for a developer but
useless in a bulletin, where one period is discussed at a time. Each figure
carries one metric and one period, an explicit French date (``Novembre 2026``,
``OND 2026``, ``1ʳᵉ décade de Novembre 2026``), a vertical colour bar on the
right, and under the map the sentence that tells the reader what counts as good
skill for that metric.

Only the cells of the CEEAC mask are drawn — exactly the cells on which the
score was computed.

Output::

    figures/<system>_<model>/<scale>/<variable>/<metric>/<period>.png
    figures/<system>_<model>/<scale>/<variable>/<metric>/<period>_<category>.png

Example::

    python scripts/run_plot_skill_raw.py --config config/cycle_202609.yaml \\
        --metrics pearson rpss roc_area groc
"""
from __future__ import annotations

import argparse

import xarray as xr

from eccas_s2s.operations.skill_raw import METRICS, score_paths, skill_dir
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle
from eccas_s2s.viz.ceeac_maps import map_score, metric_title

VARIABLE_LABEL = {"precip": "Rainfall", "t2m": "Température moyenne",
                  "tmax": "Température maximale", "tmin": "Température minimale"}
#: category names in the wording of the regional bulletins (RCC).
CATEGORY_LABEL = {"BN": "Below Normal", "NN": "Near Normal", "AN": "Above Normal"}


def _model_label(cfg, system: str, model: str) -> str:
    if system == "c3s" and model in cfg.c3s_models:
        return cfg.c3s_models[model].label
    return cfg.raw["systems"]["nmme"]["models"].get(model, {}).get("label", model)


def run(config: str, metrics=METRICS, scales=None, variables=None, models=None,
        systems=("c3s", "nmme")) -> RunContext:
    cfg = load_cycle(config)
    root = skill_dir(cfg)
    shapefile = cfg.raw["paths"]["shapefile"]
    logo = cfg.raw["paths"].get("logo")
    extent = tuple(cfg.domains["domain"]["map_extent"])
    init = cfg.init_date.date()

    with RunContext(cfg, step="plot_skill_raw") as ctx:
        ctx.record_parameter("metrics", list(metrics))
        folders = sorted((root / "netcdf").glob("*/*/*"))
        if not folders:
            raise FileNotFoundError(f"aucun score dans {root / 'netcdf'} "
                                    "(lancer d'abord run_skill_raw.py)")
        for folder in folders:
            system, model = folder.parent.parent.name.split("_", 1)
            scale, variable = folder.parent.name, folder.name
            if system not in systems or (models and model not in models):
                continue
            if (scales and scale not in scales) or (variables and variable not in variables):
                continue
            label = _model_label(cfg, system, model)

            for metric in metrics:
                f = folder / f"{metric}.nc"
                if not f.exists():
                    continue
                with xr.open_dataset(f) as ds:
                    da = ds[metric].load()
                    attrs = dict(ds.attrs)
                ctx.record_input(f, role="skill_netcdf")
                out_dir = score_paths(root / "figures", system, model, variable, scale) / metric
                cats = list(da["category"].values) if "category" in da.dims else [None]
                hind = attrs.get("hindcast_period", "")
                for key in [str(k) for k in da["period"].values]:
                    # the title carries the period alone; its exact window is in
                    # the netCDF and would only crowd the figure
                    fr = str(da["label_fr"].sel(period=key).values).split(" (")[0]
                    for cat in cats:
                        field, name, extra = da.sel(period=key), key, ""
                        if cat is not None:
                            field = field.sel(category=cat)
                            name = f"{key}_{cat}"
                            extra = f" — {CATEGORY_LABEL.get(str(cat), cat)}"
                        out = out_dir / f"{name}.png"
                        map_score(
                            field, metric=metric, variable=variable, shapefile=shapefile,
                            logo=logo, extent=extent,
                            title=(f"{label} — {VARIABLE_LABEL.get(variable, variable)}{extra}"
                                   f" — {metric_title(metric)}\n{fr}"),
                            subtitle=f"Init : {init}   |   Hindcast Period : {hind}",
                            output_path=out)
                        ctx.record_output(out, role="skill_map", system=system, model=model,
                                          variable=variable, scale=scale, metric=metric,
                                          period=key)
                ctx.log.info("%s %s %s %s %s : %d carte(s)", system, model, variable, scale,
                             metric, da.sizes["period"] * len(cats))
        ctx.log.info("%d figure(s) écrite(s) dans %s", len(ctx.outputs), root / "figures")
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", nargs="+", default=list(METRICS))
    ap.add_argument("--scales", nargs="+", choices=["decade", "month", "season"])
    ap.add_argument("--variables", nargs="+", choices=["precip", "t2m", "tmax", "tmin"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--systems", nargs="+", default=["c3s", "nmme"], choices=["c3s", "nmme"])
    args = ap.parse_args(argv)
    run(args.config, args.metrics, args.scales, args.variables, args.models, args.systems)


if __name__ == "__main__":
    main()
