"""
Seasonality of the rainfall: what share of the year each period carries (E8, informative).

This is not a forecast, and that is the point. For each period of the cycle, the
map shows — cell by cell, on the observed 1991-2020 normal — the share of the
annual rainfall that falls during that period. It answers a question the forecast
maps cannot: *where does this season matter?* A 60 % share means the year of that
cell is decided in three months, and a below-normal forecast there is a very
different statement from the same forecast where the season carries 10 %.

It is also the quantity the **IRI dry mask** is built on
(:mod:`eccas_s2s.products.masks`): for a 3-month season a cell is masked when its
share falls below 15 % of the year (or its normal below 30 mm), both thresholds
scaled to the length of the period for a month or a dekad. The map therefore
lets a forecaster see exactly what the mask is doing, instead of taking a grey
area on trust.

Everything is computed grid point by grid point; no zone average enters anywhere.

Output::

    <output_root>/climatology/<cycle>/seasonal_share/<variable>/<scale>/<period>.nc
    <output_root>/climatology/<cycle>/seasonal_share/figures/<variable>/<scale>/<period>.png

Example::

    python scripts/run_season_share.py --config config/cycle_202609.yaml --scales season
"""
from __future__ import annotations

import argparse

from eccas_s2s.core.geo import mask_like
from eccas_s2s.core.periods import (add_period_arguments, announce_subset,
                                    selected_periods)
from eccas_s2s.io.netcdf import save as save_cf
from eccas_s2s.obs.climatology import obs_period_totals
from eccas_s2s.operations import obs_chirps, obs_era5
from eccas_s2s.operations.skill_raw import AGGREGATION, OBS_VARIABLE
from eccas_s2s.products.masks import DAYS_IN_YEAR, seasonal_share
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle
from eccas_s2s.viz.ceeac_maps import map_product, product_style

SOURCE_LABEL = {"precip": "CHIRPS v2.0", "t2m": "ERA5", "tmax": "ERA5", "tmin": "ERA5"}


def share_dir(cfg):
    return cfg.output_root / "climatology" / cfg.cycle_id / "seasonal_share"


def _archives(cfg, variable: str, resolution: str):
    if variable == "precip":
        return obs_chirps.load_archives(cfg, resolution)
    dek_ds, mon_ds = obs_era5.load_archives(cfg, resolution)
    name = OBS_VARIABLE[variable]
    return dek_ds[name], mon_ds[name]


def annual_normal(cfg, variable: str, months, resolution: str = "0p25"):
    """Mean annual total (or mean) of each cell over the reference normal."""
    y0, y1 = [int(v) for v in cfg.reference_period("obs_normal")]
    sub = months.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    grouped = sub.groupby("time.year")
    yearly = grouped.sum("time") if AGGREGATION[variable] == "sum" else grouped.mean("time")
    return yearly.mean("year").load()


def run(config: str, variables=("precip",), scales=None, resolution: str = "0p25",
        selection=None, figures: bool = True) -> RunContext:
    cfg = load_cycle(config)
    out = share_dir(cfg)
    scales = list(scales) if scales else list(cfg.scales)
    shapefile = cfg.raw["paths"]["shapefile"]
    logo = cfg.raw["paths"].get("logo")
    extent = tuple(cfg.domains["domain"]["map_extent"])
    y0, y1 = [int(v) for v in cfg.reference_period("obs_normal")]

    with RunContext(cfg, step="season_share") as ctx:
        announce_subset(cfg, ctx, selection)
        ctx.record_parameter("variables", list(variables))
        ctx.record_parameter("scales", scales)
        ctx.record_parameter("resolution", resolution)
        ctx.record_parameter("normal", f"{y0}-{y1}")

        for variable in variables:
            dekads, months = _archives(cfg, variable, resolution)
            annual = annual_normal(cfg, variable, months, resolution)
            years = list(range(y0, y1 + 1))
            for scale in scales:
                periods = cfg.periods_for(cfg.max_lead_months * 31, (scale,),
                                          selection=selection)
                if not periods:
                    continue
                normal = obs_period_totals(dekads, months, periods, years,
                                           how=AGGREGATION[variable]).load()
                mask = mask_like(shapefile, normal.isel(year=0, period=0, drop=True))
                for period in periods:
                    clim = normal.sel(period=period.key, drop=True).mean("year").where(mask)
                    share = seasonal_share(clim, annual.where(mask))
                    prorata = 100.0 * period.n_days(cfg.init_date.year) / DAYS_IN_YEAR
                    ds = share.to_dataset(name="seasonal_share")
                    ds["climatology"] = clim
                    ds["annual"] = annual.where(mask)
                    ds.attrs = {**ctx.netcdf_attrs(), "variable": variable, "scale": scale,
                                "period": period.key,
                                "label": period.label(cfg.init_date.year),
                                "label_fr": period.label_fr(cfg.init_date.year, with_dates=True),
                                "normal": f"{y0}-{y1}", "source": SOURCE_LABEL.get(variable, ""),
                                "period_days": period.n_days(cfg.init_date.year),
                                "prorata_share_percent": round(prorata, 2),
                                "grid": f"observation {resolution}"}
                    path = save_cf(ds, out / variable / scale / f"{period.key}.nc",
                                   init_year=cfg.init_date.year,
                                   init_month=cfg.init_date.month)
                    ctx.record_output(path, role="seasonal_share", variable=variable,
                                      scale=scale, period=period.key)
                    if figures:
                        title = (f"{SOURCE_LABEL.get(variable, '')} {y0}-{y1} - Rainfall - "
                                 f"Share of annual total(%)\nClimatological seasonality - "
                                 f"{period.label(cfg.init_date.year)}")
                        drawn = map_product(
                            share, product="seasonal_share", scale=scale, variable=variable,
                            style=product_style("seasonal_share", scale, variable),
                            shapefile=shapefile, logo=logo, extent=extent, title=title,
                            subtitle=f"Normal: {y0}-{y1}",
                            # no caption: the map already shows real percentages,
                            # and there is nothing to warn the reader about
                            caption="",
                            output_path=out / "figures" / variable / scale / f"{period.key}.png")
                        ctx.record_output(drawn, role="seasonal_share_map", variable=variable,
                                          scale=scale, period=period.key)
                ctx.log.info("%s %s : %d période(s)", variable, scale, len(periods))
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variables", nargs="+", default=["precip"],
                    choices=["precip", "t2m", "tmax", "tmin"])
    ap.add_argument("--scales", nargs="+", choices=["decade", "month", "season"])
    ap.add_argument("--resolution", default="0p25", choices=["0p25", "1p0"])
    ap.add_argument("--no-figures", action="store_true")
    add_period_arguments(ap)
    args = ap.parse_args(argv)
    run(args.config, args.variables, args.scales, args.resolution,
        selection=selected_periods(args), figures=not args.no_figures)


if __name__ == "__main__":
    main()
