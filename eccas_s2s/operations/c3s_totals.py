"""
C3S period totals (workflow step E2 → input of E4/E5).

For each model of the cycle, reads the raw forecast and hindcast GRIB files,
converts accumulated precipitation to daily totals (lead day 0 = initialisation
day) and sums them over every complete decade, month and season of the model's
horizon. Writes, in ``<data_root>/derived/c3s/<YYYYMM>/``::

    c3s_<centre>_precip_forecast_periods.nc    dims (year=1, number, period, lat, lon)
    c3s_<centre>_precip_hindcast_periods.nc    dims (year=24, number, period, lat, lon)

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
from eccas_s2s.core.periods import build_periods
from eccas_s2s.io.c3s_read import load_c3s_precip_daily
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle


def derived_dir(cfg) -> Path:
    return cfg.data_root / "derived" / "c3s" / cfg.cycle_id


def totals_path(cfg, centre: str, kind: str, variable: str = "precip") -> Path:
    return derived_dir(cfg) / f"c3s_{centre}_{variable}_{kind}_periods.nc"


def load_totals(cfg, centre: str, kind: str, variable: str = "precip") -> xr.DataArray:
    """Open the period totals of one model (``kind`` = "forecast" or "hindcast")."""
    return xr.open_dataset(totals_path(cfg, centre, kind, variable))[variable]


def raw_file(cfg, centre: str, kind: str, var_key: str = "PRCP") -> Path:
    m = cfg.c3s_models[centre]
    hits = sorted(cfg.raw_dir("c3s").glob(f"c3s_{centre}_{m.system}_{var_key}_{kind}_*_{cfg.init_date:%m}.grib"))
    if not hits:
        raise FileNotFoundError(f"{centre} {kind} : aucun fichier brut dans {cfg.raw_dir('c3s')}")
    return hits[-1]


def run(config: str, models=None) -> RunContext:
    cfg = load_cycle(config)
    selected = list(models) if models else list(cfg.c3s_models)
    out_dir = derived_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    with RunContext(cfg, step="c3s_totals_precip") as ctx:
        summary = []
        for centre in selected:
            m = cfg.c3s_models[centre]
            periods = build_periods(cfg.init_date, m.max_lead_days, cfg.scales)
            labels = [p.label(cfg.init_date.year) for p in periods]
            for kind in ("forecast", "hindcast"):
                try:
                    src = raw_file(cfg, centre, kind)
                except FileNotFoundError as exc:
                    ctx.warn(str(exc))
                    continue
                ctx.record_input(src, role=f"c3s_raw_{kind}", centre=centre)
                ctx.log.info("%s %s : lecture et cumuls sur %d périodes", centre, kind, len(periods))
                daily = load_c3s_precip_daily(src).load()
                tot = aggregate_periods(daily, periods, how="sum").astype("float32")
                tot = tot.transpose("year", "number", "period", "latitude", "longitude")
                tot = tot.assign_coords(label=("period", labels),
                                        calendar_key=("period", [p.calendar_key for p in periods]))
                tot.name = "precip"
                tot.attrs.update({"units": "mm", "long_name": "period total precipitation",
                                  "centre": centre, "system": m.system, "model_label": m.label,
                                  "kind": kind, "horizon_days": m.max_lead_days,
                                  "n_negative_clipped": int(daily.attrs.get("n_negative_clipped", 0))})
                n_nan = tot.isnull().sum(["number", "latitude", "longitude"]).transpose("year", "period")
                yi, pi = np.nonzero(n_nan.values)
                bad = [(int(tot["year"].values[a]), str(tot["period"].values[b])) for a, b in zip(yi, pi)]
                if bad:
                    ctx.warn(f"{centre} {kind} : valeurs manquantes pour (année, période) {bad[:6]}...")

                ds = tot.to_dataset()
                ds.attrs.update({**ctx.netcdf_attrs(), "source_file": str(src.resolve())})
                enc = {"precip": {"zlib": True, "complevel": 4}}
                path = totals_path(cfg, centre, kind)
                tmp = path.with_suffix(".tmp.nc")
                ds.to_netcdf(tmp, encoding=enc)
                tmp.replace(path)
                ctx.record_output(path, role=f"c3s_totals_{kind}", centre=centre)
                summary.append({"centre": centre, "kind": kind, "members": int(tot.sizes["number"]),
                                "years": int(tot.sizes["year"]), "periods": int(tot.sizes["period"]),
                                "last_period": labels[-1], "missing_year_period": len(bad)})
                del daily, tot, ds
                gc.collect()

        table = pd.DataFrame(summary)
        table_path = out_dir / "c3s_precip_totals_summary.csv"
        table.to_csv(table_path, index=False)
        ctx.record_output(table_path, role="summary")
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
