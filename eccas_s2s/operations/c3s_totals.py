"""
C3S period totals (workflow step E2 → input of E4/E5).

For each model of the cycle, reads the raw forecast and hindcast GRIB files and
sums the rainfall over every complete decade, month and season of the model's
horizon. Writes, in ``<data_root>/derived/c3s/<YYYYMM>/``::

    c3s_<centre>_precip_forecast_periods.nc            daily stream
    c3s_<centre>_precip_hindcast_periods.nc            dims (year, number, period, lat, lon)
    c3s_<centre>_precip_forecast_periods_monthly.nc    monthly stream (months and seasons)
    c3s_<centre>_precip_hindcast_periods_monthly.nc

Two streams, because they do not carry the same ensemble
--------------------------------------------------------
The daily archive (``seasonal-original-single-levels``) stores, for the lagged
systems, only the members of the nominal start date; the monthly archive keeps
every lagged start, which is where their real ensemble is. September 2026::

    membres          quotidien           mensuel
    UKMO      prévision  2 | hindcast 7    62 | 28
    BoM       prévision 11 | hindcast 3   121 | 27
    NCEP           aucun hindcast          124 | 24

Seven members cannot support a tercile probability, twenty-eight can, which is
the whole reason for the change.

So, for UKMO, BoM and NCEP, the months and the seasons are built from the monthly
stream, while the dekads — impossible to obtain from monthly means — stay on the
daily stream (UKMO and BoM only). Which scale comes from which
stream is declared per model in the configuration (``precip_from``), never
hard-coded here.

The two streams are written to **two files** rather than merged, precisely
because their ensembles differ: padding a 3-member dekad up to 27 members with
missing values would hide that difference behind a NaN.

Units line up by construction: the monthly archive gives ``tprate``, a mean rate
in m/s over the target month, which :func:`~eccas_s2s.io.c3s_read.load_c3s_monthly`
turns into millimetres with the **true length of that month**, so a monthly total
is the same physical quantity as the sum of the daily totals.

Members are those present in the raw file (decision D19: UKMO/BoM kept as
downloaded). Every period of every year must be complete: missing values are
reported in the manifest.

Example::

    python scripts/run_c3s_totals.py --config config/cycle_202609.yaml
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.core.daily import aggregate_periods
from eccas_s2s.core.monthly import aggregate_monthly
from eccas_s2s.core.periods import (add_period_arguments, announce_subset,
                                    selected_periods)
from eccas_s2s.io.c3s_read import load_c3s_monthly, load_c3s_precip_daily
from eccas_s2s.io.netcdf import open_cf, save as save_cf
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle


def derived_dir(cfg) -> Path:
    return cfg.data_root / "derived" / "c3s" / cfg.cycle_id


def totals_path(cfg, centre: str, kind: str, variable: str = "precip",
                stream: str = "daily") -> Path:
    """
    File holding the period totals of one model and one stream.

    The daily name is left as it was, so the totals already computed for the five
    daily models stay readable; the monthly stream adds a suffix.
    """
    suffix = "" if stream == "daily" else f"_{stream}"
    return derived_dir(cfg) / f"c3s_{centre}_{variable}_{kind}_periods{suffix}.nc"


def load_totals(cfg, centre: str, kind: str, variable: str = "precip",
                stream: str = "daily") -> xr.DataArray:
    """Open the period totals of one model (``kind`` = "forecast" or "hindcast")."""
    return open_cf(totals_path(cfg, centre, kind, variable, stream))[variable]


def streams_for(model, scales) -> dict[str, list[str]]:
    """
    Which scales each stream has to produce for a model, e.g. for UKMO
    ``{"daily": ["decade"], "monthly": ["month", "season"]}``.

    A scale the model does not cover (NCEP has no daily hindcast, hence no
    dekads) is simply absent from ``precip_from`` and is skipped.
    """
    out: dict[str, list[str]] = {}
    for scale in scales:
        stream = model.stream_for(scale, "precip")
        if stream:
            out.setdefault(stream, []).append(scale)
    return out


def raw_file(cfg, centre: str, kind: str, var_key: str = "PRCP", stream: str = "daily") -> Path:
    m = cfg.c3s_models[centre]
    suffix = "" if stream == "daily" else f"_{stream}"
    pattern = f"c3s_{centre}_{m.system}_{var_key}_{kind}_*_{cfg.init_date:%m}{suffix}.grib"
    hits = sorted(cfg.raw_dir("c3s").glob(pattern))
    if not hits:
        raise FileNotFoundError(f"{centre} {kind} ({stream}) : aucun fichier brut dans "
                                f"{cfg.raw_dir('c3s')}")
    return hits[-1]


def run(config: str, models=None, streams=None, selection=None) -> RunContext:
    """
    Compute the period totals.

    ``streams`` restricts the work to one archive ("daily" or "monthly"), which
    is what makes it possible to add the monthly totals of a model without
    rewriting — and so without shortening — the daily file another step already
    uses. ``selection`` chooses the periods (see :func:`cfg.periods_for`).
    """
    cfg = load_cycle(config)
    selected = list(models) if models else list(cfg.c3s_models)
    wanted_streams = set(streams) if streams else None
    out_dir = derived_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    with RunContext(cfg, step="c3s_totals_precip") as ctx:
        announce_subset(cfg, ctx, selection)
        summary = []
        for centre in selected:
            m = cfg.c3s_models[centre]
            all_periods = cfg.periods_for(m.max_lead_days, cfg.scales, selection=selection)
            for stream, scales in streams_for(m, cfg.scales).items():
                if wanted_streams and stream not in wanted_streams:
                    continue
                wanted = [p for p in all_periods if p.scale in scales]
                if not wanted:
                    continue
                for kind in ("forecast", "hindcast"):
                    periods = list(wanted)
                    labels = [p.label(cfg.init_date.year) for p in periods]
                    try:
                        src = raw_file(cfg, centre, kind, stream=stream)
                    except FileNotFoundError as exc:
                        ctx.warn(str(exc))
                        continue
                    ctx.record_input(src, role=f"c3s_raw_{kind}", centre=centre, stream=stream)
                    ctx.log.info("%s %s (%s) : lecture et cumuls sur %d périodes (%s)",
                                 centre, kind, stream, len(periods), ", ".join(scales))
                    if stream == "daily":
                        raw = load_c3s_precip_daily(src).load()
                        tot = aggregate_periods(raw, periods, how="sum")
                    else:
                        raw = load_c3s_monthly(src, "tprate", init_month=cfg.init_date.month).load()
                        available = set(int(o) for o in raw["month_offset"].values)
                        short = [p for p in periods
                                 if not set(range(p.month_offset,
                                                  p.month_offset + p.n_months)) <= available]
                        if short:
                            ctx.warn(f"{centre} {kind} (mensuel) : {len(short)} période(s) hors "
                                     f"des mois disponibles ({sorted(available)}) : "
                                     f"{[p.key for p in short]}")
                            periods = [p for p in periods if p not in short]
                            labels = [p.label(cfg.init_date.year) for p in periods]
                        tot = aggregate_monthly(raw, periods, how="sum")
                    tot = tot.astype("float32").transpose("year", "number", "period",
                                                          "latitude", "longitude")
                    tot = tot.assign_coords(label=("period", labels),
                                            calendar_key=("period", [p.calendar_key for p in periods]))
                    tot.name = "precip"
                    tot.attrs.update({"units": "mm", "long_name": "period total precipitation",
                                      "centre": centre, "system": m.system, "model_label": m.label,
                                      "kind": kind, "horizon_days": m.max_lead_days, "stream": stream,
                                      "n_negative_clipped": int(raw.attrs.get("n_negative_clipped", 0))})
                    # netCDF has no boolean attribute type (load_c3s_monthly flags
                    # the lagged ensembles with one), so booleans are stored as 0/1
                    tot.attrs = {k: int(v) if isinstance(v, bool) else v
                                 for k, v in tot.attrs.items()}
                    n_nan = tot.isnull().sum(["number", "latitude", "longitude"]).transpose("year", "period")
                    yi, pi = np.nonzero(n_nan.values)
                    bad = [(int(tot["year"].values[a]), str(tot["period"].values[b])) for a, b in zip(yi, pi)]
                    if bad:
                        ctx.warn(f"{centre} {kind} : valeurs manquantes pour (année, période) {bad[:6]}...")

                    ds = tot.to_dataset()
                    ds.attrs.update({**ctx.netcdf_attrs(), "source_file": str(src.resolve()),
                                     "periods_selected": ", ".join(labels)})
                    # a run on a selection rewrites the whole file: say it plainly when
                    # that would drop periods a previous run had computed
                    existing = totals_path(cfg, centre, kind, stream=stream)
                    if existing.exists():
                        with xr.open_dataset(existing) as prev:
                            lost = sorted(set(str(k) for k in prev["period"].values)
                                          - {p.key for p in periods})
                        if lost:
                            ctx.warn(f"{centre} {kind} ({stream}) : le fichier existant "
                                     f"contenait {len(lost)} période(s) de plus "
                                     f"({', '.join(lost)}) — elles sont remplacées")
                    path = save_cf(ds, totals_path(cfg, centre, kind, stream=stream),
                                   init_year=cfg.init_date.year,
                                   init_month=cfg.init_date.month)
                    ctx.record_output(path, role=f"c3s_totals_{kind}", centre=centre, stream=stream)
                    summary.append({"centre": centre, "stream": stream, "kind": kind,
                                    "members": int(tot.sizes["number"]),
                                    "years": int(tot.sizes["year"]),
                                    "periods": int(tot.sizes["period"]),
                                    "scales": "+".join(scales),
                                    "last_period": labels[-1] if labels else "",
                                    "missing_year_period": len(bad)})
                    del raw, tot, ds
                    gc.collect()

        table = pd.DataFrame(summary)
        table_path = out_dir / "c3s_precip_totals_summary.csv"
        if table_path.exists() and (models or streams):
            # a partial run completes the table instead of replacing it
            keys = ["centre", "stream", "kind"]
            old = pd.read_csv(table_path)
            if "stream" not in old.columns:
                old["stream"] = "daily"
            table = (pd.concat([old, table]).drop_duplicates(keys, keep="last")
                     .sort_values(keys).reset_index(drop=True))
        table.to_csv(table_path, index=False)
        ctx.record_output(table_path, role="summary")
        ctx.record_parameter("summary", summary)
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--streams", nargs="+", choices=["daily", "monthly"],
                    help="ne traiter que ces flux (par défaut : ceux déclarés par le modèle)")
    add_period_arguments(ap)
    args = ap.parse_args(argv)
    run(args.config, args.models, args.streams, selection=selected_periods(args))


if __name__ == "__main__":
    main()
