"""
Do the ocean drivers carry a usable signal for CEEAC rainfall? (workflow E3, phase E7)

This is the cheapest step of the downscaling phase, and it is deliberately the
first: five SST indices from an 80 MB analysis answer, before any field is
downloaded, whether a pattern-based downscaling has anything to work with.

Two questions, and they must not be confused:

**Is there a link at all?** measured with the **concurrent** season — the SST of
the very season being forecast. This is the ceiling: no forecast of an ocean
driver can beat the relationship the driver has with rainfall. If the concurrent
link is at the chance level, no predictor built on that driver will work, whatever
the method.

**Is that link usable in advance?** measured with the season **preceding the
initialisation** (June-July-August for a 1 September start), which is what an
operational forecast knows when it is issued. The gap between the two is the
part that depends on the ocean being predictable, and that only the model's own
forecast SST can recover (third step of the phase).

The statistic is the share of the CEEAC mask, weighted by the cosine of latitude,
where the correlation is significant at 5 %. On 44 years the threshold is
|r| > 0.297, and **5 % of the mask is significant by chance** — that is the line
below which a driver says nothing.

Outputs in ``<output_root>/drivers/<cycle>/``::

    driver_correlations.csv      one row per target season, driver and timing
    maps/<season>/<driver>_<timing>.png
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from eccas_s2s.core.geo import mask_like
from eccas_s2s.operations import obs_chirps
from eccas_s2s.predictors.sst_indices import INDICES, LABELS, compute_indices, load_ersst
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle

#: months of the seasons the diagnostic is run on, by key
SEASONS = {"SON": (9, 10, 11), "OND": (10, 11, 12), "NDJ": (11, 12, 1), "DJF": (12, 1, 2),
           "MAM": (3, 4, 5), "JJA": (6, 7, 8), "JJAS": (6, 7, 8, 9)}
#: share of the mask a driver must beat to say anything: at 5 % significance,
#: 5 % of the cells are significant by chance alone
CHANCE = 0.05


def season_totals(monthly: xr.DataArray, months, years) -> xr.DataArray:
    """Seasonal totals of an observed monthly archive, one field per year."""
    months, out = list(months), []
    stamps_available = pd.DatetimeIndex(monthly["time"].values)
    for year in years:
        stamps, y = [], year
        for k, month in enumerate(months):
            if k and month < months[k - 1]:
                y += 1
            stamps.append(pd.Timestamp(y, month, 1))
        got = monthly.sel(time=[s for s in stamps if s in stamps_available])
        out.append(got.sum("time") if got.sizes.get("time") == len(months)
                   else xr.full_like(monthly.isel(time=0, drop=True), np.nan))
    return xr.concat(out, dim=pd.Index(list(years), name="year"))


def correlation_map(field: xr.DataArray, series: xr.DataArray,
                    year_dim: str = "year") -> xr.DataArray:
    """Pearson correlation between a field and a time series, per grid point."""
    a = field - field.mean(year_dim)
    b = series - series.mean(year_dim)
    denom = np.sqrt((a ** 2).sum(year_dim) * (b ** 2).sum(year_dim))
    out = (a * b).sum(year_dim) / denom.where(denom > 0)
    out.name = "correlation"
    return out


def significant_fraction(correlation: xr.DataArray, mask: xr.DataArray, n_years: int,
                         alpha: float = 0.05) -> float:
    """Share of the mask where the correlation is significant, cosine-weighted."""
    t = stats.t.ppf(1 - alpha / 2, n_years - 2)
    r_crit = t / np.sqrt(n_years - 2 + t ** 2)
    weights = np.cos(np.deg2rad(correlation["latitude"])).broadcast_like(correlation)
    hit = (abs(correlation) > r_crit).where(mask)
    return float(hit.weighted(weights.where(mask).fillna(0)).mean())


def run(config: str, seasons=("SON", "OND"), years=(1981, 2024), indices=INDICES,
        ersst_dir=None, draw_maps: bool = True) -> RunContext:
    cfg = load_cycle(config)
    years = list(range(int(years[0]), int(years[1]) + 1))
    ersst_dir = ersst_dir or (cfg.data_root / "raw" / "ersst" / "monthly")
    out_dir = cfg.output_root / "drivers" / cfg.cycle_id
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)

    with RunContext(cfg, step="driver_diagnostic") as ctx:
        ctx.record_parameter("seasons", list(seasons))
        ctx.record_parameter("years", [years[0], years[-1]])
        ctx.record_parameter("indices", list(indices))
        sst = load_ersst(ersst_dir, years[0], years[-1] + 1)
        _, monthly = obs_chirps.load_archives(cfg)
        ctx.log.info("ERSST %d mois | CHIRPS %d mois | %d années",
                     sst.sizes["time"], monthly.sizes["time"], len(years))

        rows = []
        for season in seasons:
            target_months = SEASONS[season]
            rain = season_totals(monthly, target_months, years)
            mask = mask_like(cfg.raw["paths"]["shapefile"], rain.isel(year=0, drop=True))
            rain = rain.where(mask)
            # the season preceding the initialisation, and the season itself
            timings = {"prévisionnel": SEASONS["JJA"] if season in ("SON", "OND", "NDJ")
                       else target_months, "simultané": target_months}
            for timing, sst_months in timings.items():
                idx = compute_indices(sst, sst_months, years, indices=indices)
                for name in indices:
                    corr = correlation_map(rain, idx[name])
                    share = significant_fraction(corr, mask, len(years))
                    rows.append({"season": season, "driver": name, "label": LABELS.get(name, name),
                                 "timing": timing,
                                 "sst_months": "-".join(str(m) for m in sst_months),
                                 "n_years": len(years),
                                 "median_r": float(corr.where(mask).median()),
                                 "max_abs_r": float(abs(corr).where(mask).max()),
                                 "significant_fraction": share,
                                 "above_chance": share > 2 * CHANCE})
                    if draw_maps:
                        _draw(cfg, corr, mask, season, name, timing, out_dir, ctx, len(years))
                ctx.log.info("%s ← SST %s : %s", season, timing,
                             ", ".join(f"{r['driver']} {100*r['significant_fraction']:.1f}%"
                                       for r in rows[-len(indices):]))

        table = pd.DataFrame(rows)
        path = out_dir / "driver_correlations.csv"
        table.to_csv(path, index=False)
        ctx.record_output(path, role="driver_diagnostic")
        for timing in sorted(set(table["timing"])):
            useful = table[(table["timing"] == timing) & table["above_chance"]]
            names = sorted(set(useful["driver"]))
            ctx.record_parameter(f"drivers_above_chance_{timing}", names)
            ctx.log.info("pilotes au-dessus du hasard (%s) : %s", timing,
                         ", ".join(names) or "aucun")
    return ctx


def _draw(cfg, corr, mask, season, driver, timing, out_dir, ctx, n_years: int):
    """One correlation map per driver and timing, in the house style."""
    from eccas_s2s.viz.ceeac_maps import map_score

    levels = [-0.6, -0.5, -0.4, -0.3, -0.2, 0, 0.2, 0.3, 0.4, 0.5, 0.6]
    colours = ["#8C510A", "#BF812D", "#DFC27D", "#F6E8C3", "#FFFFFF",
               "#D9F0D3", "#A6DBA0", "#5AAE61", "#1B7837", "#00441B"]
    style = {"colors": colours, "levels": levels,
             "label": f"corrélation avec {driver}",
             "caption": (f"Corrélation entre l'indice océanique et le cumul saisonnier observé, "
                         f"{n_years} années. |r| > 0,30 est significatif à 5 % ; 5 % du domaine "
                         "l'est par hasard, donc seule une part nettement supérieure signale un "
                         "pilote utile.")}
    path = out_dir / "maps" / season / f"{driver}_{timing}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    map_score(corr.where(mask), metric="correlation", variable="precip", style=style,
              shapefile=cfg.raw["paths"]["shapefile"], logo=cfg.raw["paths"].get("logo"),
              extent=tuple(cfg.domains["domain"]["map_extent"]),
              title=f"{LABELS.get(driver, driver)} — Rainfall {season}"
                    f"\nCorrélation observée · SST {timing}",
              subtitle="ERSSTv5 · CHIRPS 0,25° · 1981-2024", output_path=path)
    ctx.record_output(path, role="driver_map", season=season, driver=driver, timing=timing)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seasons", nargs="+", default=["SON", "OND"], choices=sorted(SEASONS))
    ap.add_argument("--years", nargs=2, type=int, default=[1981, 2024])
    ap.add_argument("--indices", nargs="+", default=list(INDICES))
    ap.add_argument("--no-maps", action="store_true")
    args = ap.parse_args(argv)
    run(args.config, args.seasons, args.years, args.indices, draw_maps=not args.no_maps)


if __name__ == "__main__":
    main()
