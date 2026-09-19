"""
C3S temperature period values (workflow step E2 → input of E4/E5).

* ``t2m``  — 2 m mean temperature from the C3S **monthly statistics**
  (decision of 2026-09-19): months and seasons only. Lagged ensembles (UKMO,
  BoM) are merged over all their start dates, so these two models have many more
  members here than in the daily fields.
* ``tmax`` / ``tmin`` — daily maximum / minimum 2 m temperature (24 h ending at
  00 UTC), averaged over decades, months and seasons.

Output in ``<data_root>/derived/c3s/<YYYYMM>/``::

    c3s_<centre>_<t2m|tmax|tmin>_<forecast|hindcast>_periods.nc
    dims (year, number, period, latitude, longitude), °C

Variables whose raw files are not yet downloaded are skipped with a warning.

Example::

    python scripts/run_c3s_temperature.py --config config/cycle_202609.yaml
"""
from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd

from eccas_s2s.core.daily import aggregate_periods
from eccas_s2s.core.monthly import aggregate_monthly
from eccas_s2s.core.periods import build_periods
from eccas_s2s.io.c3s_read import load_c3s_daily_last24h, load_c3s_monthly
from eccas_s2s.operations.c3s_totals import derived_dir, totals_path
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle

#: variable -> (raw file key, monthly?)
VARIABLES = {"t2m": ("TEMP", True), "tmax": ("TMAX", False), "tmin": ("TMIN", False)}


def _raw(cfg, centre, key, kind, monthly):
    m = cfg.c3s_models[centre]
    suffix = "_monthly" if monthly else ""
    hits = sorted(cfg.raw_dir("c3s").glob(f"c3s_{centre}_{m.system}_{key}_{kind}_*_{cfg.init_date:%m}{suffix}.grib"))
    return hits[-1] if hits else None


def run(config: str, variables=None, models=None):
    cfg = load_cycle(config)
    variables = list(variables) if variables else list(VARIABLES)
    selected = list(models) if models else list(cfg.c3s_models)
    init = cfg.init_date
    out_dir = derived_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    with RunContext(cfg, step="c3s_temperature") as ctx:
        summary = []
        for var in variables:
            key, monthly = VARIABLES[var]
            for centre in selected:
                m = cfg.c3s_models[centre]
                for kind in ("forecast", "hindcast"):
                    src = _raw(cfg, centre, key, kind, monthly)
                    if src is None:
                        ctx.warn(f"{var} {centre} {kind} : fichier brut absent (téléchargement non terminé ?)")
                        continue
                    ctx.record_input(src, role=f"c3s_raw_{var}_{kind}", centre=centre)
                    if monthly:
                        data = load_c3s_monthly(src, "t2m", init_month=init.month).load()
                        n_months = int(data["month_offset"].max()) + 1
                        horizon = ((init + pd.DateOffset(months=n_months)) - init).days
                        periods = build_periods(init, horizon, scales=("month", "season"))
                        vals = aggregate_monthly(data, periods, how="mean")
                    else:
                        data = load_c3s_daily_last24h(src).load()
                        periods = build_periods(init, m.max_lead_days, cfg.scales)
                        vals = aggregate_periods(data, periods, how="mean")
                    vals = vals.transpose("year", "number", "period", "latitude", "longitude").astype("float32")
                    vals = vals.assign_coords(label=("period", [p.label(init.year) for p in periods]))
                    vals.name = var
                    vals.attrs = {"units": "degC", "centre": centre, "system": m.system, "kind": kind,
                                  "model_label": m.label, "source": "monthly statistics" if monthly else "daily",
                                  "n_start_dates": int(data.attrs.get("n_start_dates", 1))}
                    # members absent for some years (lagged ensembles) are NaN: count fully empty periods only
                    empty = vals.isnull().all(["number", "latitude", "longitude"]).transpose("year", "period")
                    holes = int(empty.sum())
                    if holes:
                        ctx.warn(f"{var} {centre} {kind} : {holes} couple(s) (année, période) sans valeur")
                    ds = vals.to_dataset()
                    ds.attrs.update({**ctx.netcdf_attrs(), "source_file": str(src.resolve())})
                    path = totals_path(cfg, centre, kind, var)
                    tmp = path.with_suffix(".tmp.nc")
                    ds.to_netcdf(tmp, encoding={var: {"zlib": True, "complevel": 4}})
                    tmp.replace(path)
                    ctx.record_output(path, role=f"c3s_{var}_{kind}", centre=centre)
                    n_mem = vals.notnull().any(["period", "latitude", "longitude"]).sum("number")
                    summary.append({"variable": var, "centre": centre, "kind": kind,
                                    "members_min": int(n_mem.min()), "members_max": int(n_mem.max()),
                                    "years": int(vals.sizes["year"]), "periods": int(vals.sizes["period"]),
                                    "last_period": str(vals["label"].values[-1]), "empty_year_periods": holes})
                    del data, vals, ds
                    gc.collect()
        table = pd.DataFrame(summary)
        path = out_dir / "c3s_temperature_summary.csv"
        if path.exists() and not table.empty:            # keep rows of variables not re-run
            old = pd.read_csv(path)
            old = old[~old.set_index(["variable", "centre", "kind"]).index.isin(
                table.set_index(["variable", "centre", "kind"]).index)]
            table = pd.concat([old, table], ignore_index=True)
        table.sort_values(["variable", "centre", "kind"]).to_csv(path, index=False)
        ctx.record_output(path, role="summary")
        ctx.record_parameter("summary", summary)
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variables", nargs="+", choices=sorted(VARIABLES))
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args(argv)
    run(args.config, args.variables, args.models)


if __name__ == "__main__":
    main()
