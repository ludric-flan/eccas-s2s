"""
Skill maps of the raw hindcasts (workflow step E4, figures).

Reads the score files written by :mod:`eccas_s2s.operations.skill_raw`
(``netcdf/<system>_<model>/<scale>/<variable>/<product>/<metric>.nc``) and draws
them in the CAPC-AC house style with :mod:`eccas_s2s.viz.ceeac_maps`.

**One product, one metric, one period per figure.** By default only the metric
that **decides** for each product is drawn — the map a forecaster reads to know
whether that product can be trusted. The companions kept in the netCDF (the
debiased score, the discrimination) are drawn on request, so nothing is lost and
the default run stays readable.

**One map per period**: a panel of six seasons is convenient for a developer but
useless in a bulletin, where one period is discussed at a time. Each figure
carries one metric and one period, an explicit French date (``Novembre 2026``,
``OND 2026``, ``1ʳᵉ décade de Novembre 2026``), a vertical colour bar on the
right, and under the map the sentence that tells the reader what counts as good
skill for that metric.

Only the cells of the CEEAC mask are drawn — exactly the cells on which the
score was computed.

Output::

    figures/<system>_<model>/<scale>/<variable>/<product>/<metric>/<period>.png
    figures/<system>_<model>/<scale>/<variable>/<product>/<metric>/<period>_<category>.png

Example::

    python scripts/run_plot_skill_raw.py --config config/cycle_202609.yaml
    python scripts/run_plot_skill_raw.py --config config/cycle_202609.yaml --metrics all
"""
from __future__ import annotations

import argparse

import xarray as xr

from eccas_s2s.io.netcdf import open_cf
from eccas_s2s.operations.skill_raw import score_paths, skill_dir
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle
from eccas_s2s.viz.ceeac_maps import gain_style, map_score, metric_title

VARIABLE_LABEL = {"precip": "Rainfall", "t2m": "Température moyenne",
                  "tmax": "Température maximale", "tmin": "Température minimale"}
#: category names in the wording of the regional bulletins (RCC).
CATEGORY_LABEL = {"BN": "Below Normal", "NN": "Near Normal", "AN": "Above Normal"}
#: the scores of this step are those of the *raw* hindcast; the prefix says so on
#: every figure, so a map cannot be confused with its calibrated counterpart (P3).
KIND = "Raw"
#: word printed before the metric name, per tree
KIND_LABEL = {"raw": "Raw", "calibrated": "Calibrated", "gain": "Gain"}


def _product_label(cfg, variable: str, scale: str, product: str) -> str:
    """Human label of a product, read from the catalogue of the cycle."""
    from eccas_s2s.products.catalogue import catalogue

    for item in catalogue(variable, scale, cfg.thresholds.get(variable, {})):
        if item.name == product:
            return item.label or product
    return product


def _recommended_methods(cfg) -> dict:
    """``(system, model, variable, scale, product) -> recommended method``, from the register."""
    import pandas as pd

    path = cfg.path_of("output_root") / "registry" / "calibration_methods.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    keys = ["system", "model", "variable", "scale", "product"]
    return {tuple(row[k] for k in keys): row["method"] for _, row in table.iterrows()}


def _model_label(cfg, system: str, model: str) -> str:
    if system == "c3s" and model in cfg.c3s_models:
        return cfg.c3s_models[model].label
    return cfg.raw["systems"]["nmme"]["models"].get(model, {}).get("label", model)


#: diagnostics kept beside the scores in the netCDF, never drawn as a skill map
NOT_A_SCORE = ("n_years", "base_rate")


def _metrics_of(folder, deciding: str, wanted, kind: str = "raw") -> list[str]:
    """Which metric files of a product folder to draw."""
    available = [f.stem for f in sorted(folder.glob("*.nc")) if f.stem not in NOT_A_SCORE]
    if kind == "gain":
        gains = [m for m in available if m.startswith("gain_")]
        if wanted in (None, "deciding"):
            return [m for m in gains if m == f"gain_{deciding}"]
        return gains if wanted == "all" else [m for m in gains
                                              if m.replace("gain_", "") in set(wanted)]
    available = [m for m in available if not m.startswith("gain_")]
    if wanted in (None, "deciding"):
        return [m for m in available if m == deciding]
    if wanted == "all":
        return available
    return [m for m in available if m in set(wanted)]


def run(config: str, metrics="deciding", scales=None, variables=None, models=None,
        systems=("c3s", "nmme"), products=None, kind: str = "raw",
        methods=None) -> RunContext:
    """
    Draw the skill maps of one tree.

    ``kind`` selects the tree: ``raw`` (phase P2) or ``calibrated`` (phase P3).
    The calibrated tree carries one more level — the method — and, by default,
    only the **recommended** method of each product is drawn, read from
    ``registry/calibration_methods.csv``: that is the map a forecaster needs, the
    others being available on request through ``methods``.
    """
    cfg = load_cycle(config)
    # the gains live in the calibrated tree, beside the scores they compare
    tree = "calibrated" if kind == "gain" else kind
    root = skill_dir(cfg, tree)
    word = KIND_LABEL.get(kind, KIND)
    recommended = _recommended_methods(cfg) if tree == "calibrated" else {}
    shapefile = cfg.raw["paths"]["shapefile"]
    logo = cfg.raw["paths"].get("logo")
    extent = tuple(cfg.domains["domain"]["map_extent"])
    init = cfg.init_date.date()

    with RunContext(cfg, step=f"plot_skill_{kind}") as ctx:
        ctx.record_parameter("kind", kind)
        ctx.record_parameter("metrics", metrics if isinstance(metrics, str) else list(metrics))
        pattern = "*/*/*/*/*" if tree == "calibrated" else "*/*/*/*"
        folders = sorted((root / "netcdf").glob(pattern))
        if not folders:
            raise FileNotFoundError(f"aucun score dans {root / 'netcdf'} "
                                    "(lancer d'abord run_skill_raw.py)")
        for folder in folders:
            method = ""
            if tree == "calibrated":
                method, folder_product = folder.name, folder.parent
            else:
                folder_product = folder
            system, model = folder_product.parent.parent.parent.name.split("_", 1)
            scale, variable = folder_product.parent.parent.name, folder_product.parent.name
            product = folder_product.name
            if method:
                wanted = (methods if methods else
                          [recommended.get((system, model, variable, scale, product), "")])
                if method not in set(wanted):
                    continue
            if system not in systems or (models and model not in models):
                continue
            if (scales and scale not in scales) or (variables and variable not in variables):
                continue
            if products and product not in products:
                continue
            label = _model_label(cfg, system, model)
            with open_cf(next(iter(sorted(folder.glob("*.nc"))))) as probe:
                deciding = probe.attrs.get("deciding", "")
            # the label comes from the catalogue: in a CF file the global
            # attribute `label` holds the period labels, not the product's
            product_label = _product_label(cfg, variable, scale, product)

            for metric in _metrics_of(folder, deciding, metrics, kind):
                f = folder / f"{metric}.nc"
                if not f.exists():
                    continue
                with open_cf(f) as ds:
                    da = ds[metric].load()
                    attrs = dict(ds.attrs)
                style = gain_style(metric) if kind == "gain" else None
                ctx.record_input(f, role="skill_netcdf")
                branch = product if not method else f"{product}/{method}"
                out_dir = (score_paths(root / "figures", system, model, variable, scale, branch)
                           / metric)
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
                            style=style,
                            logo=logo, extent=extent,
                            title=(f"{label} — {VARIABLE_LABEL.get(variable, variable)} — "
                                   f"{product_label}{extra}"
                                   f"\n{word} {metric_title(metric.replace('gain_', ''))}"
                                   f"{' · ' + method if method else ''} — {fr}"),
                            subtitle=f"Init : {init}   |   Hindcast Period : {hind}",
                            output_path=out)
                        ctx.record_output(out, role="skill_map", system=system, model=model,
                                          variable=variable, scale=scale, product=product,
                                          metric=metric, period=key)
                ctx.log.info("%s %s %s %s %s/%s : %d carte(s)", system, model, variable, scale,
                             product, metric, da.sizes["period"] * len(cats))
        ctx.log.info("%d figure(s) écrite(s) dans %s", len(ctx.outputs), root / "figures")
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", nargs="+", default=["deciding"],
                    help="deciding (défaut : la métrique qui décide pour chaque produit), "
                         "all, ou une liste explicite (rpss groc roc_area)")
    ap.add_argument("--products", nargs="+", help="ne tracer que ces produits du catalogue")
    ap.add_argument("--kind", default="raw", choices=["raw", "calibrated", "gain"],
                    help="arbre à tracer : brut (P2), calibré (P3) ou gain (calibré − brut)")
    ap.add_argument("--methods", nargs="+",
                    help="arbre calibré : méthodes à tracer (défaut : la méthode recommandée)")
    ap.add_argument("--scales", nargs="+", choices=["decade", "month", "season"])
    ap.add_argument("--variables", nargs="+", choices=["precip", "t2m", "tmax", "tmin"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--systems", nargs="+", default=["c3s", "nmme"], choices=["c3s", "nmme"])
    args = ap.parse_args(argv)
    metrics = args.metrics[0] if len(args.metrics) == 1 else tuple(args.metrics)
    run(args.config, metrics, args.scales, args.variables, args.models, args.systems,
        args.products, kind=args.kind, methods=args.methods)


if __name__ == "__main__":
    main()
