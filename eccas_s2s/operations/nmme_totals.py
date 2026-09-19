"""
NMME period values: monthly and seasonal precipitation totals and mean 2 m
temperature, per model (workflow step E2 → input of E4/E5).

NMME files are monthly ensemble means (no members). For each model, all
initialisation years of the cycle's month are read and converted:

* ``prate`` (mm/s) → monthly total (mm) = rate × 86 400 × days in the month;
  season = sum of its three months;
* ``tmp2m`` (K) → monthly mean (°C); season = mean of its three months weighted
  by their number of days.

Only complete months and seasons within the model's forecast months are kept.
Values are cut to the C3S download area. Output in
``<data_root>/derived/nmme/<YYYYMM>/``::

    nmme_<MODEL>_<precip|t2m>_hindcast_periods.nc   dims (year, period, latitude, longitude)
    nmme_<MODEL>_<precip|t2m>_forecast_periods.nc   dims (year=1, period, latitude, longitude)

``period`` uses the same keys as C3S (``month_m0``, ``season_m1`` …).

Example::

    python scripts/run_nmme_totals.py --config config/cycle_202609.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.core.periods import build_periods
from eccas_s2s.io.nmme_cpc import open_nmme_file
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle

VARIABLES = {"prate": "precip", "tmp2m": "t2m"}


def derived_dir(cfg) -> Path:
    return cfg.data_root / "derived" / "nmme" / cfg.cycle_id


def totals_path(cfg, model: str, variable: str, kind: str) -> Path:
    return derived_dir(cfg) / f"nmme_{model}_{variable}_{kind}_periods.nc"


def load_totals(cfg, model: str, variable: str, kind: str) -> xr.DataArray:
    return xr.open_dataset(totals_path(cfg, model, variable, kind))[variable]


def monthly_values(da: xr.DataArray, nmme_var: str) -> xr.DataArray:
    """Convert one file's monthly field to totals (mm) or means (°C)."""
    days = xr.DataArray(pd.DatetimeIndex(da["target"].values).days_in_month, dims="target",
                        coords={"target": da["target"]})
    if nmme_var == "prate":
        out = da * 86400.0 * days
        out.attrs = {"units": "mm", "long_name": "monthly total precipitation (ensemble mean)"}
    else:
        out = da - 273.15
        out.attrs = {"units": "degC", "long_name": "monthly mean 2 m temperature (ensemble mean)"}
    return out.assign_coords(days=days)


def period_values(monthly: xr.DataArray, periods, init_year: int, nmme_var: str) -> xr.DataArray:
    """Values of the cycle's periods for one initialisation year (NaN if months are missing)."""
    out = []
    targets = pd.DatetimeIndex(monthly["target"].values)
    for p in periods:
        start, end = p.dates(init_year)
        wanted = pd.date_range(start, end, freq="MS")
        if not all(t in targets for t in wanted):
            out.append(xr.full_like(monthly.isel(target=0, drop=True), np.nan))
            continue
        sel = monthly.sel(target=wanted)
        if nmme_var == "prate":
            out.append(sel.sum("target", skipna=False))
        else:
            w = sel["days"]
            out.append((sel * w).sum("target", skipna=False) / w.sum())
    res = xr.concat(out, dim=pd.Index([p.key for p in periods], name="period"))
    return res.drop_vars(["days"], errors="ignore")


def run(config: str, models=None) -> RunContext:
    cfg = load_cycle(config)
    nmme = cfg.raw["systems"]["nmme"]
    selected = list(models) if models else list(nmme["models"])
    lon0, lon1, lat0, lat1 = cfg.c3s_download_area
    raw = cfg.raw_dir("nmme")
    out_dir = derived_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    init = cfg.init_date

    with RunContext(cfg, step="nmme_totals") as ctx:
        summary = []
        for model in selected:
            for nmme_var, name in VARIABLES.items():
                files = sorted((raw / model).glob(f"{model}.{nmme_var}.*{init:%m}.ENSMEAN.fcst.nc"))
                if not files:
                    ctx.warn(f"{model} {nmme_var} : aucun fichier")
                    continue
                by_year, n_target = {}, {}
                for f in files:
                    da = open_nmme_file(f).sel(longitude=slice(lon0, lon1), latitude=slice(lat0, lat1))
                    year = pd.Timestamp(da.attrs["init"]).year
                    by_year[year] = monthly_values(da.load(), nmme_var)
                    n_target[year] = da.sizes["target"]
                # periods reachable by the shortest year of this model (complete months only)
                n_months = min(n_target.values())
                last_day = (init + pd.DateOffset(months=n_months)) - pd.Timedelta(days=1)
                periods = build_periods(init, (last_day - init).days + 1, scales=("month", "season"))
                if min(n_target.values()) != max(n_target.values()):
                    ctx.warn(f"{model} {nmme_var} : nombre de mois cibles variable {sorted(set(n_target.values()))}"
                             f" → {n_months} mois retenus")
                years = sorted(by_year)
                stack = xr.concat([period_values(by_year[y], periods, y, nmme_var) for y in years],
                                  dim=pd.Index(years, name="year")).astype("float32")
                stack = stack.assign_coords(
                    scale=("period", [p.scale for p in periods]),
                    label=("period", [p.label(init.year) for p in periods]),
                    calendar_key=("period", [p.calendar_key for p in periods]))
                stack.name = name
                stack.attrs = {**by_year[years[0]].attrs, "model": model, "label_model": nmme["models"][model]["label"],
                               "ensemble": "ensemble mean only (NMME CPC files)", "nmme_variable": nmme_var}
                n_nan = int(stack.isnull().sum())
                empty = stack.isnull().all(["latitude", "longitude"]).transpose("year", "period")
                holes = [(int(stack["year"][i]), str(stack["label"][j].values)) for i, j in np.argwhere(empty.values)]
                if holes:
                    ctx.warn(f"{model} {name} : périodes absentes (mois cibles manquants dans le fichier) {holes}")
                for kind, ys in (("hindcast", [y for y in years if y < init.year]), ("forecast", [init.year])):
                    if not set(ys) <= set(years) or not ys:
                        ctx.warn(f"{model} {name} {kind} : années absentes")
                        continue
                    ds = stack.sel(year=ys).to_dataset()
                    ds.attrs.update(ctx.netcdf_attrs())
                    path = totals_path(cfg, model, name, kind)
                    ds.to_netcdf(path, encoding={name: {"zlib": True, "complevel": 4}})
                    ctx.record_output(path, role=f"nmme_{name}_{kind}", model=model)
                summary.append({"model": model, "variable": name, "hindcast_years": f"{years[0]}–{years[-2]}",
                                "n_hindcast": len(years) - 1, "months": n_months, "periods": len(periods),
                                "last_period": periods[-1].label(init.year), "missing_values": n_nan,
                                "empty_year_periods": len(holes)})
        table = pd.DataFrame(summary)
        table.to_csv(out_dir / "nmme_totals_summary.csv", index=False)
        ctx.record_output(out_dir / "nmme_totals_summary.csv", role="summary")
        ctx.record_parameter("summary", summary)
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args(argv)
    run(args.config, args.models)


if __name__ == "__main__":
    main()
