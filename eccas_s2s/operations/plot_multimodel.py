"""
Maps of the multi-model products (workflow E8).

The skill maps of phases P2 and P3 say what a product is worth; these are the
products themselves — what goes into the bulletin. Every product is drawn with
**its own** rule, taken from the reference chain (``run_forecast_v2`` →
``s2s_plotting_v2``), because a forecaster must read the same colours here and
there:

* **terciles**, categorical: the dominant category of each cell, in the
  probability classes of the Draft annex (40-50 … 90-100 %), one colour bar per
  category;
* **SPI classes**, the same dominant-class reading, with the colours of the
  reference chain — dry in oranges, normal in greens, wet in blues;
* **anomaly**: the nine brown-to-green bands of the reference, with its levels
  per time scale (±200 mm for a season, ±100 mm for a month);
* **SPI**: the diverging brown-white-green scale, cut at ±0,5 … ±3;
* **exceedance of a millimetre threshold**: the continuous white → green →
  yellow → orange → red scale of the reference, 0 to 100 %, a tick every 5 %;
* **tercile, quintile and median probabilities**: one ramp per product
  (YlOrBr for below normal, YlGnBu for above normal, OrRd and PuBu for the
  quintiles, RdYlGn for the median), in 10 % bins;
* **total**: the published cumul, on a rainfall scale by time scale.

Two rules apply to all of them. The dry cells are painted in the RCC sand grey
and the product is not drawn there; and the field is **smoothed across the
pixels**, as the score and gain maps are, then clipped to the CEEAC boundary —
a seasonal forecast is a large-scale field, and a blocky map reads as if each
cell were an independent measurement. The categorical maps are the exception:
interpolating colours between "below normal" and "above normal" would invent an
intermediate category that does not exist.

Both trees (raw and calibrated) and every weighting scheme are drawn, so that a
forecaster can see what the choice of weights actually changes.
"""
from __future__ import annotations

import argparse

from eccas_s2s.core.periods import period_from_key
from eccas_s2s.io.netcdf import open_cf
from eccas_s2s.operations.multimodel import multimodel_dir
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle
from eccas_s2s.viz.ceeac_maps import (SPI_CLASS_COLORS, SPI_CLASS_LABEL, map_product,
                                      map_tercile_forecast, product_style)

VARIABLE_LABEL = {"precip": "Rainfall", "t2m": "Mean temperature",
                  "tmax": "Maximum temperature", "tmin": "Minimum temperature"}
KIND_LABEL = {"raw": "Raw forecast", "calibrated": "Calibrated forecast"}
SCHEME_LABEL = {"equal": "equal weighting", "sqrt_members": "√(members) weighting",
                "skill": "skill weighting"}
#: how each weighting scheme is written in a file name
SCHEME_SLUG = {"equal": "equal_weight", "sqrt_members": "sqrt_members_weight",
               "skill": "skill_weight"}
#: what the first line of the header names, product by product
PRODUCT_TITLE = {"cumul": "Cumul(mm)", "anomalie": "Anomaly(mm)", "spi": "SPI",
                 "terciles": "Tercile probabilities(%)", "spi_classes": "SPI classes(%)",
                 "categorie_BN": "Below Normal probability(%)",
                 "categorie_AN": "Above Normal probability(%)",
                 "quintile_bas": "Lowest 20% probability(%)",
                 "quintile_haut": "Highest 20% probability(%)",
                 "depassement_median": "Probability of exceeding the median(%)"}
#: short names of the models, for the line under the map
MODEL_SHORT = {"ecmwf": "ECMWF", "ukmo": "UKMO", "meteo_france": "Météo-France",
               "dwd": "DWD", "cmcc": "CMCC", "eccc": "ECCC", "bom": "BoM", "ncep": "NCEP"}
#: how each calibration method is named under a calibrated map
METHOD_SHORT = {"raw": "RAW", "elr": "ELR", "logistic": "LR", "eqm": "EQM",
                "qmap": "QMAP", "qmap_ratio": "QMAP-R", "ngr": "NGR", "ngr_const": "NGR-C",
                "bias_mean": "BIAS-M", "bias_ratio": "BIAS-R", "bias_scaling": "BIAS-S",
                "pcr": "PCR"}


def product_title(product: str) -> str:
    """First line of the header: what the map shows, in the bulletin's wording."""
    if product.startswith("depassement_") and product != "depassement_median":
        return f"Probability of exceeding {product.replace('depassement_', '')}(%)"
    return PRODUCT_TITLE.get(product, product)


def caption_of(attrs: dict, kind: str, scheme: str = "equal") -> tuple[str, bool]:
    """
    The lines under the map, and whether they must be drawn in a monospaced font.

    All the models stay on **one** line, as on the raw maps, and what belongs to
    a model is written under its name: the calibration method it entered with,
    and, for a skill-weighted combination, its weight. A calibrated multi-model
    is not one method applied to eight models — each enters with the method the
    register of phase P3 recommends for it — and a weight of 0,034 beside one of
    0,169 is the real content of a skill map; neither can be guessed from the
    colours.

    One column per model, aligned by a monospaced font, so the caller is told to
    use one as soon as there is more than the single line of names.
    """
    used = [m for m in str(attrs.get("models", "")).split(", ") if m]
    methods, weights = {}, {}
    for item in str(attrs.get("methods", "")).split(", "):
        if "=" in item:
            model, method = item.split("=", 1)
            methods[model] = METHOD_SHORT.get(method, method.upper())
    for item in str(attrs.get("weights", "")).split(", "):
        if "=" in item:
            model, weight = item.split("=", 1)
            weights[model] = weight
    names = [MODEL_SHORT.get(m, m.upper()) for m in used]

    rows = [names]
    if kind == "calibrated" and methods:
        rows.append([methods.get(m, "") for m in used])
    if scheme == "skill" and weights:
        rows.append([weights.get(m, "") for m in used])
    if len(rows) == 1:                       # raw, equal weights: one plain line
        return " | ".join(names), False

    width = max(len(cell) for row in rows for cell in row) + 2
    lines = ["".join(cell.center(width) for cell in row) for row in rows]
    source = str(attrs.get("weight_source", ""))
    if scheme == "skill" and source:
        lines.append(f"weights from {source}".center(width * len(names)))
    return "\n".join(lines), True


def run(config: str, variables=("precip",), scales=None, kinds=("raw", "calibrated"),
        schemes=None, products=None, periods=None) -> RunContext:
    cfg = load_cycle(config)
    shapefile = cfg.raw["paths"]["shapefile"]
    logo = cfg.raw["paths"].get("logo")
    extent = tuple(cfg.domains["domain"]["map_extent"])
    init = cfg.init_date.date()

    with RunContext(cfg, step="plot_multimodel") as ctx:
        ctx.record_parameter("kinds", list(kinds))
        for kind in kinds:
            root = multimodel_dir(cfg, kind)
            if not root.exists():
                ctx.warn(f"{kind} : aucun produit multi-modèle")
                continue
            for path in sorted(root.glob("*/*/*/*/*.nc")):
                variable, scale, scheme, product = [p.name for p in
                                                    (path.parents[3], path.parents[2],
                                                     path.parents[1], path.parents[0])]
                if variable not in variables or (scales and scale not in scales):
                    continue
                if (schemes and scheme not in schemes) or (products and product not in products):
                    continue
                if periods and path.stem not in periods:
                    continue
                with open_cf(path) as ds:
                    field = ds[product].load()
                    dry = ds["dry_mask"].load().astype(bool) if "dry_mask" in ds else None
                    attrs = dict(ds.attrs)
                # the title carries the period as a bulletin writes it
                # (September 2026, OND 2026, 1st Decad of September 2026), never
                # the compact key the files are sorted on
                period = period_from_key(attrs.get("period", path.stem), cfg.init_date.month)
                year = cfg.init_date.year
                label = period.label_en(year)
                head = (f"C3S Multi-Model Mean - {VARIABLE_LABEL.get(variable, variable)}"
                        f" - {product_title(product)}")
                sub_head = (f"{KIND_LABEL.get(kind, kind)} - "
                            f"{SCHEME_LABEL.get(scheme, scheme)} {label}")
                subtitle = f"Init: {init}"
                # the models that actually entered the combination, in short form
                caption, mono = caption_of(attrs, kind, scheme)
                # named as the reference chain names its maps: a file taken out
                # of its folder still says what it shows
                name = (f"{product}_{kind}_{SCHEME_SLUG.get(scheme, scheme)}"
                        f"_forecast_{period.slug(year)}.png")
                out = root / "figures" / variable / scale / scheme / product / name

                if "category" in field.dims:
                    spi = product == "spi_classes"
                    written = map_tercile_forecast(
                        field, variable=variable, shapefile=shapefile, logo=logo, extent=extent,
                        title=f"{head}\n{sub_head}", subtitle=subtitle, dry_mask=dry,
                        palette=SPI_CLASS_COLORS if spi else None,
                        category_labels=SPI_CLASS_LABEL if spi else None,
                        caption=caption, caption_mono=mono, output_path=out)
                else:
                    # the colour bar is labelled like the header, in the wording
                    # of the bulletin, not with the French catalogue name
                    style = product_style(product, scale, variable,
                                          label=product_title(product).replace("(%)", "")
                                          .replace("(mm)", "").strip())
                    written = map_product(field, product=product, scale=scale, variable=variable,
                                          dry=dry, style=style, shapefile=shapefile, logo=logo,
                                          extent=extent, title=f"{head}\n{sub_head}",
                                          subtitle=subtitle, caption=caption,
                                          caption_mono=mono, output_path=out)
                ctx.record_output(written, role="mme_map", kind=kind, scheme=scheme,
                                  product=product, scale=scale)
        ctx.log.info("%d carte(s) de produit écrite(s)", len(ctx.outputs))
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variables", nargs="+", default=["precip"])
    ap.add_argument("--scales", nargs="+", choices=["decade", "month", "season"])
    ap.add_argument("--kinds", nargs="+", default=["raw", "calibrated"],
                    choices=["raw", "calibrated"])
    ap.add_argument("--schemes", nargs="+", help="ne tracer que ces schémas de poids")
    ap.add_argument("--products", nargs="+", help="ne tracer que ces produits")
    ap.add_argument("--periods", nargs="+", help="ne tracer que ces périodes (clé du fichier)")
    args = ap.parse_args(argv)
    run(args.config, args.variables, args.scales, args.kinds, args.schemes, args.products,
        args.periods)


if __name__ == "__main__":
    main()
