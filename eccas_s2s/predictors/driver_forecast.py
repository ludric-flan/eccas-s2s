"""
Forecast of the ocean drivers, month by month (workflow E3, drivers monitoring).

The diagnostic of the phase established two things: the model forecasts the
drivers well (Niño 3.4 at 0.98, the Indian dipole at 0.92, the Atlantic modes
between 0.68 and 0.77 from a 1 September start), and a regression from those
drivers onto rainfall adds nothing that the model's own rainfall does not
already carry. The drivers are therefore worth **publishing as such**, not as
predictors: a forecaster who knows that the Atlantic Niño will be warm in
November reads every rainfall map differently, and no other centre issues these
series for Central Africa.

What this module produces, for each driver and each lead month:

* the **ensemble forecast** of the index — median, quartiles and deciles, so the
  spread is visible rather than hidden behind a mean;
* the **skill of that forecast**, measured on the hindcast against ERSSTv5: the
  correlation and the mean absolute error of the ensemble mean, lead by lead.
  An index forecast without its skill is an opinion.

Two conventions carry the method:

* the index is standardised **against the model's own climatology at that lead**,
  never against the observed one. A seasonal model drifts, and the drift grows
  with the lead: standardising on the observation would turn drift into a
  forecast anomaly. The hindcast is the model's own reference;
* the **linear trend is removed**. The hindcast stops in 2016 and the ocean has
  warmed since: without detrending, the September 2026 forecast reads +3.1 σ on
  Niño 3.4 at every lead — the warming, not El Niño. The trend is fitted on the
  hindcast years and extrapolated to the forecast year, so what remains is the
  interannual mode the drivers are about. The raw anomaly stays available for
  anyone who wants the absolute warmth;
* the anomaly is computed **member by member**, then the distribution is read.
  Standardising the ensemble mean would shrink the spread by the very factor the
  ensemble is there to show.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eccas_s2s.predictors.sst_indices import BOXES, DIPOLES, INDICES, LABELS

MEMBER = "number"
LEAD = "month_offset"
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def _prepare(sst: xr.DataArray) -> xr.DataArray:
    """Model SST with the chain's coordinate names and longitudes in −180..180."""
    out = sst
    rename = {k: v for k, v in (("latitude", "lat"), ("longitude", "lon")) if k in out.dims}
    if rename:
        out = out.rename(rename)
    return out.assign_coords(lon=(((out["lon"] + 180) % 360) - 180)).sortby("lon").sortby("lat")


def box_mean(sst: xr.DataArray, box) -> xr.DataArray:
    lon0, lon1, lat0, lat1 = box
    sel = sst.sel(lon=slice(lon0, lon1), lat=slice(lat0, lat1))
    weights = np.cos(np.deg2rad(sel["lat"]))
    return sel.weighted(weights).mean(("lat", "lon"))


def _linear_trend(series: xr.DataArray, years: np.ndarray):
    """Least-squares slope and intercept of a series against the year, per lead."""
    x = years - years.mean()
    y = series.transpose("year", ...)
    slope = (y * xr.DataArray(x, dims="year", coords={"year": y["year"]})).sum("year") / (x ** 2).sum()
    intercept = y.mean("year") - slope * years.mean()
    return slope, intercept


def model_indices(sst: xr.DataArray, reference: xr.DataArray | None = None,
                  indices=INDICES, detrend: bool = True) -> xr.Dataset:
    """
    Standardised drivers of a model forecast, per member and per lead month.

    ``reference`` is the hindcast the climatology is taken from — the model's
    own, at the same lead. When it is omitted the series standardises itself,
    which is only correct for a hindcast.
    """
    sst = _prepare(sst)
    ref = _prepare(reference) if reference is not None else sst
    poles = {}
    for name, box in BOXES.items():
        series = box_mean(sst, box)
        base = box_mean(ref, box)
        dims = [d for d in ("year", MEMBER) if d in base.dims]
        if detrend and "year" in base.dims:
            # a linear trend fitted on the hindcast, extrapolated to the forecast
            # year: the warming since 2016 is real, but it is not a driver anomaly
            ref_mean = base.mean(MEMBER) if MEMBER in base.dims else base
            years_ref = ref_mean["year"].values.astype(float)
            slope, intercept = _linear_trend(ref_mean, years_ref)
            base = base - (slope * xr.DataArray(years_ref, dims="year",
                                                coords={"year": base["year"]}) + intercept)
            target_years = sst["year"].values.astype(float)
            series = series - (slope * xr.DataArray(target_years, dims="year",
                                                    coords={"year": sst["year"]}) + intercept)
        mean = base.mean(dims)
        sd = base.std(dims, ddof=1)
        poles[name] = (series - mean) / sd.where(sd > 0)
    out = {}
    for name in indices:
        if name in DIPOLES:
            a, b = DIPOLES[name]
            out[name] = poles[a] - poles[b]
        else:
            out[name] = poles[name]
        out[name].attrs = {"long_name": LABELS.get(name, name),
                           "standardisation": "climatologie du modèle, par échéance",
                           "detrended": int(bool(detrend))}
    return xr.Dataset(out)


def forecast_table(indices: xr.Dataset, init_date, quantiles=QUANTILES) -> pd.DataFrame:
    """Tidy table: one row per driver and lead month, with the ensemble quantiles."""
    init = pd.Timestamp(init_date)
    rows = []
    for name, da in indices.data_vars.items():
        values = da.squeeze(drop=True)
        for lead in [int(v) for v in values[LEAD].values]:
            sample = values.sel({LEAD: lead})
            members = np.asarray(sample.values, dtype=float).ravel()
            members = members[np.isfinite(members)]
            target = init + pd.DateOffset(months=lead)
            row = {"driver": name, "label": LABELS.get(name, name), "lead_month": lead,
                   "target": target.strftime("%Y-%m"), "n_members": len(members),
                   "mean": float(np.mean(members)),
                   "sign_agreement": float(np.mean(np.sign(members) == np.sign(np.mean(members))))}
            for q in quantiles:
                row[f"q{int(100 * q):02d}"] = float(np.quantile(members, q))
            rows.append(row)
    return pd.DataFrame(rows)


def hindcast_skill(model: xr.Dataset, observed: xr.Dataset) -> pd.DataFrame:
    """
    Skill of the forecast drivers, lead by lead, against the observed indices.

    The correlation says whether the model gets the year right; the mean absolute
    error says by how much it misses, in standard deviations — an index forecast
    correlated at 0.9 but systematically half as strong is still misleading.
    """
    rows = []
    for name in model.data_vars:
        for lead in [int(v) for v in model[name][LEAD].values]:
            m = model[name].sel({LEAD: lead})
            if MEMBER in m.dims:
                m = m.mean(MEMBER)
            o = observed[name].sel(lead_month=lead) if "lead_month" in observed[name].dims \
                else observed[name]
            a, b = np.asarray(m.values, float).ravel(), np.asarray(o.values, float).ravel()
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 5:
                continue
            rows.append({"driver": name, "label": LABELS.get(name, name), "lead_month": lead,
                         "n_years": int(ok.sum()),
                         "correlation": float(np.corrcoef(a[ok], b[ok])[0, 1]),
                         "mae_sigma": float(np.mean(np.abs(a[ok] - b[ok]))),
                         "amplitude_ratio": float(np.std(a[ok], ddof=1) / np.std(b[ok], ddof=1))})
    return pd.DataFrame(rows)


def plot_forecast(table: pd.DataFrame, skill: pd.DataFrame, driver: str, output_path,
                  init_date=None, logo=None):
    """
    One driver, one figure: the ensemble forecast by lead, and its skill below.

    The envelope is the ensemble, not a confidence interval — it says what the
    members disagree on, and the hindcast correlation printed under each lead
    says how much that disagreement is worth.
    """
    import matplotlib.pyplot as plt

    sub = table[table["driver"] == driver].sort_values("lead_month")
    sk = skill[skill["driver"] == driver].set_index("lead_month")
    x = sub["lead_month"].to_numpy()
    fig, (ax, axs) = plt.subplots(2, 1, figsize=(7.2, 5.6), height_ratios=(3, 1), sharex=True)

    ax.axhline(0, color="#555555", lw=1)
    ax.axhspan(-0.5, 0.5, color="#EEEEEE", zorder=0)
    ax.fill_between(x, sub["q10"], sub["q90"], color="#9ECAE1", alpha=0.45, label="déciles 10–90 %")
    ax.fill_between(x, sub["q25"], sub["q75"], color="#4292C6", alpha=0.55, label="quartiles")
    ax.plot(x, sub["q50"], color="#08306B", lw=2.2, marker="o", label="médiane de l'ensemble")
    ax.set_ylabel("indice standardisé (σ)")
    ax.set_title(f"{LABELS.get(driver, driver)} — prévision par échéance"
                 + (f"\nInit : {pd.Timestamp(init_date).date()}" if init_date is not None else ""),
                 fontweight="bold", fontsize=11.5)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(alpha=0.25)

    corr = [sk["correlation"].get(int(v), np.nan) for v in x]
    axs.bar(x, corr, color=["#1B7837" if c >= 0.6 else "#F6E8C3" if c >= 0.4 else "#BF812D"
                            for c in corr], width=0.6)
    axs.axhline(0.6, color="#1B7837", ls=":", lw=1)
    axs.set_ylim(0, 1)
    axs.set_ylabel("corrélation\nhindcast")
    axs.set_xlabel("échéance (mois après l'initialisation)")
    axs.set_xticks(x)
    axs.set_xticklabels([f"{v}\n{t}" for v, t in zip(x, sub["target"])], fontsize=8)
    axs.grid(alpha=0.25, axis="y")

    fig.text(0.5, 0.015, "Vert : le modèle retrouve l'année observée (r ≥ 0,6) ; brun : r < 0,4, "
             "la prévision de l'indice est peu fiable à cette échéance.",
             ha="center", fontsize=8, style="italic", color="#333333")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return Path(output_path)
