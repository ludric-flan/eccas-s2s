"""
Data processing: decumulation, period generation, accumulation,
climatology/anomaly computation, probability products.
"""
import os
import calendar
import numpy as np
import xarray as xr
import pandas as pd

from eccas_s2s.config import PRECIP_THRESHOLDS


# ============================================================
# DECUMULATION
# ============================================================
def decumulate_precip_to_mm(ds, var="tp", step_dim="step",
                             outvar="tp_daily", clip_negative=True):
    """Decumulate accumulated precipitation and convert m → mm."""
    if var not in ds:
        raise KeyError(f"Variable '{var}' absente du dataset.")
    da = ds[var]
    if step_dim not in da.dims:
        raise ValueError(f"Dimension '{step_dim}' absente.")

    ds_out = ds.copy()
    original_attrs = da.attrs.copy()

    with xr.set_options(keep_attrs=True):
        da_mm = da * 1000.0
        first_step_value = da[step_dim].values[0]
        first_is_zero = (first_step_value == np.timedelta64(0, "ns"))

        first = (xr.zeros_like(da_mm.isel({step_dim: slice(0, 1)}))
                 if first_is_zero
                 else da_mm.isel({step_dim: slice(0, 1)}))

        rest  = da_mm.diff(step_dim, label="upper")
        daily = xr.concat([first, rest], dim=step_dim)
        daily = daily.assign_coords({step_dim: da[step_dim]})
        daily = daily.transpose(*da.dims)

        if clip_negative:
            daily = daily.clip(min=0)

    new_attrs = original_attrs.copy()
    new_attrs["units"]     = "mm"
    new_attrs["long_name"] = "Daily total precipitation"
    daily.attrs = new_attrs
    ds_out[outvar] = daily
    return ds_out


# ============================================================
# PERIOD GENERATION
# ============================================================
def _make_period_row(ptype, label, start, end):
    start = pd.Timestamp(start)
    end   = pd.Timestamp(end)
    return {
        "type": ptype, "label": label,
        "start": start, "end": end,
        "start_month": start.month, "start_day": start.day,
        "end_month":   end.month,   "end_day":   end.day,
        "n_days": (end - start).days + 1,
    }


def generate_calendar_periods_from_forecast(
    forecast_ds, max_end_date=None,
    season_window_months=3,
    include_partial_first=True,
    include_partial_last=False,
):
    """Generate decade, month and season DataFrames from the forecast valid_time."""
    vt          = pd.to_datetime(forecast_ds["valid_time"].values)
    first_valid = pd.Timestamp(vt.min())
    last_valid  = pd.Timestamp(vt.max())

    if max_end_date is not None:
        last_valid = min(last_valid, pd.Timestamp(max_end_date))

    decades_rows, months_rows, seasons_rows = [], [], []

    cur = pd.Timestamp(first_valid.year, first_valid.month, 1)
    while cur <= last_valid:
        y, m = cur.year, cur.month
        ndays      = calendar.monthrange(y, m)[1]
        month_end  = pd.Timestamp(y, m, ndays)

        p_start = (first_valid
                   if include_partial_first and y == first_valid.year and m == first_valid.month
                   else pd.Timestamp(y, m, 1))
        if p_start < first_valid:
            p_start = first_valid

        p_end = month_end
        if p_end > last_valid:
            p_end = last_valid if include_partial_last else None

        if p_end is not None and p_start <= p_end:
            months_rows.append(_make_period_row("month", f"{y}-{m:02d}", p_start, p_end))

        for d0, d1, tag in [(1, 10, "D1"), (11, 20, "D2"), (21, ndays, "D3")]:
            ps = pd.Timestamp(y, m, d0)
            pe = pd.Timestamp(y, m, d1)

            if (include_partial_first and y == first_valid.year
                    and m == first_valid.month
                    and first_valid.day > d0 and first_valid.day <= d1):
                ps = first_valid
            if ps < first_valid:
                ps = first_valid
            if pe > last_valid:
                pe = last_valid if include_partial_last else None
            if pe is not None and ps <= pe:
                decades_rows.append(_make_period_row("decade", f"{y}-{m:02d}-{tag}", ps, pe))

        cur = cur + pd.offsets.MonthBegin(1)

    season_starts = [first_valid]
    cur = pd.Timestamp(first_valid.year, first_valid.month, 1) + pd.offsets.MonthBegin(1)
    while cur <= last_valid:
        season_starts.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)

    for s in season_starts:
        p_end = (pd.Timestamp(s.year, s.month, 1)
                 + pd.DateOffset(months=season_window_months - 1)
                 + pd.offsets.MonthEnd(0))
        if p_end > last_valid:
            if not include_partial_last:
                continue
            p_end = last_valid
        seasons_rows.append(
            _make_period_row("season", f"{s:%Y-%m-%d}_to_{p_end:%Y-%m-%d}", s, p_end)
        )

    return {
        "decades":     pd.DataFrame(decades_rows),
        "months":      pd.DataFrame(months_rows),
        "seasons":     pd.DataFrame(seasons_rows),
        "first_valid": first_valid,
        "last_valid":  last_valid,
    }


def build_custom_period(start_date, end_date, label=None):
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("end_date doit être >= start_date")
    if label is None:
        label = f"{start:%Y-%m-%d}_to_{end:%Y-%m-%d}"
    return pd.DataFrame([_make_period_row("custom", label, start, end)])


# ============================================================
# PERIOD ACCUMULATION
# ============================================================
def _monthday_mask(valid_time, start_month, start_day, end_month, end_day):
    md       = valid_time.dt.month * 100 + valid_time.dt.day
    start_md = start_month * 100 + start_day
    end_md   = end_month   * 100 + end_day
    if start_md <= end_md:
        return (md >= start_md) & (md <= end_md)
    else:
        return (md >= start_md) | (md <= end_md)


def _attach_period_coords(da_out, periods_df):
    return da_out.assign_coords(
        period       =("period", periods_df.index.values),
        period_label =("period", periods_df["label"].values),
        period_type  =("period", periods_df["type"].values),
        period_start =("period", pd.to_datetime(periods_df["start"]).values),
        period_end   =("period", pd.to_datetime(periods_df["end"]).values),
    )


def accumulate_forecast_over_periods(ds, periods_df, var="tp_daily"):
    da  = ds[var]
    vt  = ds["valid_time"]
    out = []
    for i, row in periods_df.iterrows():
        start = np.datetime64(pd.Timestamp(row["start"]))
        end   = np.datetime64(pd.Timestamp(row["end"]))
        mask  = (vt >= start) & (vt <= end)
        acc   = da.where(mask, 0).sum(dim="step")
        out.append(acc.expand_dims(period=[i]))
    out = xr.concat(out, dim="period")
    out = _attach_period_coords(out, periods_df)
    out.attrs = {**da.attrs, "long_name": "Forecast precipitation accumulation", "units": "mm"}
    return out


def accumulate_hindcast_over_periods(ds, periods_df, var="tp_daily"):
    da  = ds[var]
    vt  = ds["valid_time"]
    out = []
    for i, row in periods_df.iterrows():
        s    = pd.Timestamp(row["start"])
        e    = pd.Timestamp(row["end"])
        mask = _monthday_mask(vt, s.month, s.day, e.month, e.day)
        acc  = da.where(mask, 0).sum(dim="step")
        out.append(acc.expand_dims(period=[i]))
    out = xr.concat(out, dim="period")
    out = _attach_period_coords(out, periods_df)
    out.attrs = {**da.attrs, "long_name": "Hindcast precipitation accumulation", "units": "mm"}
    return out


# ============================================================
# CLIMATOLOGY & ANOMALY
# ============================================================
def compute_climatology_and_anomaly(hindcast_acc, forecast_acc):
    clim = hindcast_acc.mean(dim=("number", "time"), keep_attrs=True)
    anom = forecast_acc - clim
    anom.attrs = {**forecast_acc.attrs, "long_name": "Precipitation anomaly", "units": "mm"}
    return clim, anom


def compute_period_products(hindcast_ds, forecast_ds, periods_df, var="tp_daily"):
    forecast_acc = accumulate_forecast_over_periods(forecast_ds, periods_df, var=var)
    hindcast_acc = accumulate_hindcast_over_periods(hindcast_ds, periods_df, var=var)
    clim, anom   = compute_climatology_and_anomaly(hindcast_acc, forecast_acc)
    return {
        "forecast_acc": forecast_acc,
        "hindcast_acc": hindcast_acc,
        "climatology":  clim,
        "anomaly":      anom,
    }


# ============================================================
# PROBABILITY PRODUCTS
# ============================================================
def compute_climatological_thresholds(hindcast_acc):
    da      = hindcast_acc.stack(sample=("number", "time"))
    t_low   = da.quantile(1/3,  dim="sample").drop_vars("quantile")
    t_up    = da.quantile(2/3,  dim="sample").drop_vars("quantile")
    q_low   = da.quantile(0.20, dim="sample").drop_vars("quantile")
    q_up    = da.quantile(0.80, dim="sample").drop_vars("quantile")
    med     = da.quantile(0.50, dim="sample").drop_vars("quantile")
    clim_mean = hindcast_acc.mean(dim=("number", "time"))
    dry_mask  = clim_mean < 1.0
    return {
        "tercile_lower": t_low, "tercile_upper": t_up,
        "quintile_lower": q_low, "quintile_upper": q_up,
        "median": med, "clim_mean": clim_mean, "dry_mask": dry_mask,
    }


def compute_forecast_probabilities(forecast_acc, thresholds):
    n = forecast_acc.sizes["number"]
    t_low = thresholds["tercile_lower"]
    t_up  = thresholds["tercile_upper"]
    q_low = thresholds["quintile_lower"]
    q_up  = thresholds["quintile_upper"]
    med   = thresholds["median"]

    return {
        "prob_BN":           (forecast_acc < t_low).sum(dim="number") / n,
        "prob_AN":           (forecast_acc > t_up).sum(dim="number")  / n,
        "prob_NN":           ((forecast_acc >= t_low) & (forecast_acc <= t_up)).sum(dim="number") / n,
        "prob_low20":        (forecast_acc < q_low).sum(dim="number") / n,
        "prob_high20":       (forecast_acc > q_up).sum(dim="number")  / n,
        "prob_exceed_median":(forecast_acc > med).sum(dim="number")   / n,
        "dry_mask":          thresholds["dry_mask"],
        "clim_mean":         thresholds["clim_mean"],
    }


def compute_exceedance_probabilities(forecast_acc, thresholds_mm=PRECIP_THRESHOLDS):
    n = forecast_acc.sizes["number"]
    exceed_probs = {}
    for thr in thresholds_mm:
        prob = (forecast_acc >= thr).sum(dim="number") / n
        prob.attrs = {
            "long_name":    f"P(precip >= {thr} mm)",
            "units":        "fraction (0-1)",
            "threshold_mm": thr,
        }
        exceed_probs[thr] = prob
    return exceed_probs


def compute_all_probabilities(hindcast_acc, forecast_acc):
    print("    📊 Seuils climatologiques...")
    thresholds = compute_climatological_thresholds(hindcast_acc)
    print("    📊 Probabilités terciles/quintiles/médiane...")
    probs = compute_forecast_probabilities(forecast_acc, thresholds)

    total   = probs["prob_BN"] + probs["prob_NN"] + probs["prob_AN"]
    max_err = float(np.abs(total - 1.0).max())
    print(f"    ✅ Cohérence BN+NN+AN=1 : erreur max = {max_err:.2e}")
    return probs, thresholds


# ============================================================
# LIGHT PRODUCTS (for multi-model combination)
# ============================================================
def build_light_products_dataset(products, probs, exceed_probs, model=None, period_type=None):
    ds = xr.Dataset()

    ens_mean_anomaly = products["anomaly"].mean("number", keep_attrs=True).astype("float32")
    ens_mean_anomaly.attrs["long_name"] = "Ensemble mean precipitation anomaly"
    ds["ens_mean_anomaly"] = ens_mean_anomaly

    for v in ["prob_BN", "prob_NN", "prob_AN", "prob_low20", "prob_high20", "prob_exceed_median"]:
        ds[v] = probs[v].astype("float32")
    ds["clim_mean"] = probs["clim_mean"].astype("float32")
    ds["dry_mask"]  = probs["dry_mask"].astype("int8")

    threshold_values = sorted(exceed_probs.keys())
    exc_list = [exceed_probs[thr].astype("float32").expand_dims(threshold=[thr])
                for thr in threshold_values]
    exceed_stack = xr.concat(exc_list, dim="threshold")
    exceed_stack = exceed_stack.transpose("period", "threshold", "latitude", "longitude")
    exceed_stack.attrs = {
        "long_name": "Probability of exceeding fixed precipitation thresholds",
        "units":     "fraction (0-1)",
    }
    ds["prob_exceed_threshold"] = exceed_stack

    ref = products["forecast_acc"]
    for cname in ["period", "period_label", "period_type", "period_start", "period_end",
                  "latitude", "longitude"]:
        if cname in ref.coords:
            ds = ds.assign_coords({cname: ref[cname]})

    ds = ds.assign_coords(threshold=("threshold", np.array(threshold_values, dtype="int32")))

    ds.attrs["description"] = "Light products for multi-model post-processing"
    if model is not None:
        ds.attrs["model"] = model
    if period_type is not None:
        ds.attrs["period_type_saved"] = period_type
    return ds


def save_light_products_dataset(ds_light, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    encoding = {}
    for v in ds_light.data_vars:
        if np.issubdtype(ds_light[v].dtype, np.floating):
            encoding[v] = {"zlib": True, "complevel": 4}
        elif np.issubdtype(ds_light[v].dtype, np.integer):
            encoding[v] = {"zlib": True, "complevel": 4}
    try:
        ds_light.to_netcdf(output_path, encoding=encoding)
    except Exception:
        ds_light.to_netcdf(output_path)
    print(f"    💾 Produits légers sauvegardés : {output_path}")


def load_light_products_dataset(path):
    ds = xr.open_dataset(path)
    ds.load()
    ds.close()
    return ds
