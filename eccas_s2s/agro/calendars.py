"""
Season-indicator calendars (onset, cessation, dry/wet spells) and their
feasibility for a given initialisation.

Parameters come from ``config/agro_calendars.yaml`` (decision D14, taken from
``capc_sitroom/seasonal_engine`` v10).

Feasibility rule (decision of 2026-09-18): an indicator is produced for an
initialisation only if its **whole reference window** lies inside the forecast,
i.e. starts on or after the initialisation day and ends within the horizon. A
window that has already started is *not* completed with observations.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FEASIBLE = "réalisable"
ALREADY_STARTED = "fenêtre déjà commencée"
BEYOND_HORIZON = "au-delà de l'horizon"


@dataclass(frozen=True)
class AgroProduct:
    name: str
    label: str
    zone: str
    metric: str                       # onset | cessation | both | seasonal
    reference: tuple[str, str]        # MM-DD, MM-DD
    onset_search: tuple[str, str] | None = None
    cessation_search: tuple[str, str] | None = None
    persistence_days: int | None = None
    min_wet_days: int | None = None
    max_dry_spell: int | None = None
    min_dry_days: int | None = None
    recommended_init: int | None = None


def load_agro_products(agro_cfg: dict) -> dict[str, AgroProduct]:
    """Build :class:`AgroProduct` objects from the ``agro_calendars`` configuration."""
    out = {}
    for name, spec in agro_cfg["products"].items():
        def _pair(key):
            v = spec.get(key)
            return tuple(v) if v else None
        out[name] = AgroProduct(
            name=name, label=spec["label"], zone=spec["zone"], metric=spec["metric"],
            reference=tuple(spec["reference"]), onset_search=_pair("onset_search"),
            cessation_search=_pair("cessation_search"),
            persistence_days=spec.get("persistence_days"), min_wet_days=spec.get("min_wet_days"),
            max_dry_spell=spec.get("max_dry_spell"), min_dry_days=spec.get("min_dry_days"),
            recommended_init=spec.get("recommended_init"),
        )
    return out


def _mmdd(year: int, mmdd: str) -> pd.Timestamp:
    month, day = (int(x) for x in mmdd.split("-"))
    if month == 2 and day == 29:
        day = 29 if pd.Timestamp(year, 1, 1).is_leap_year else 28
    return pd.Timestamp(year, month, day)


def window_dates(window: tuple[str, str], year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve a MM-DD window starting in ``year`` (end may fall in ``year + 1``)."""
    start = _mmdd(year, window[0])
    end = _mmdd(year, window[1])
    if end < start:
        end = _mmdd(year + 1, window[1])
    return start, end


def target_window(product: AgroProduct, init_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The next occurrence of the product's reference window that ends on/after init."""
    init_date = pd.Timestamp(init_date)
    for year in (init_date.year - 1, init_date.year, init_date.year + 1):
        start, end = window_dates(product.reference, year)
        if end >= init_date:
            return start, end
    raise RuntimeError("fenêtre introuvable")  # unreachable for valid MM-DD windows


def feasibility(product: AgroProduct, init_date: pd.Timestamp, horizon_days: int) -> dict:
    """Whether the product can be computed from a forecast initialised on ``init_date``."""
    init_date = pd.Timestamp(init_date)
    start, end = target_window(product, init_date)
    last_day = init_date + pd.Timedelta(days=horizon_days - 1)
    if start < init_date:
        status = ALREADY_STARTED
    elif end > last_day:
        status = BEYOND_HORIZON
    else:
        status = FEASIBLE
    return {"product": product.name, "label": product.label, "zone": product.zone,
            "metric": product.metric, "window_start": start.date(), "window_end": end.date(),
            "last_forecast_day": last_day.date(), "status": status,
            "recommended_init": product.recommended_init}


def feasibility_table(products: dict[str, AgroProduct], init_date, horizons: dict[str, int]) -> pd.DataFrame:
    """
    Feasibility of every product for each horizon.

    ``horizons`` maps a label to a number of forecast days, e.g.
    ``{"tous modèles (184 j)": 184, "modèles 215 j": 215}``.
    """
    rows = []
    for p in products.values():
        row = None
        for hlabel, h in horizons.items():
            f = feasibility(p, init_date, h)
            if row is None:
                row = {k: f[k] for k in ("product", "zone", "metric", "window_start",
                                         "window_end", "recommended_init")}
            row[hlabel] = f["status"]
        rows.append(row)
    return pd.DataFrame(rows)
