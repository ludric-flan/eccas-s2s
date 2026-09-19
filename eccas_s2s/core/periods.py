"""
Target periods of a forecast cycle: decades, months and seasons (decision D8).

A period is defined *relative to the initialisation month* (``month_offset`` = 0 for
the initialisation month, 1 for the next one, ...). This lets the same period be
resolved to calendar dates for the forecast year and for every hindcast year, which
all start on the same day of the same month::

    periods = build_periods(pd.Timestamp("2026-09-01"), horizon_days=184)
    p = periods[0]
    p.dates(2026)   # -> (2026-09-01, 2026-09-10)
    p.dates(1993)   # -> (1993-09-01, 1993-09-10)
    p.day_slice(1993)  # -> (0, 9): lead-day indices of the period

Rules
-----
* Decades: D1 = days 1-10, D2 = 11-20, D3 = 21 to month end (8 to 11 days).
* Months: calendar months.
* Seasons: three consecutive calendar months, labelled by their initials
  (SON, OND, NDJ, DJF ...).
* Only **complete** periods are kept: a period whose last day is beyond the
  forecast horizon is dropped (no partial periods).
* Lead-day index 0 is the initialisation day itself (see ``core.daily``).
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass

import pandas as pd

_MONTH_INITIALS = "JFMAMJJASOND"
_MONTH_ABBR_FR = ["janv.", "févr.", "mars", "avr.", "mai", "juin",
                  "juil.", "août", "sept.", "oct.", "nov.", "déc."]


def _shift_month(year: int, month: int, offset: int) -> tuple[int, int]:
    idx = (month - 1) + offset
    return year + idx // 12, idx % 12 + 1


@dataclass(frozen=True)
class Period:
    """A target period, defined relative to the initialisation month."""

    scale: str            # "decade" | "month" | "season"
    init_month: int       # 1..12, month of the initialisation
    month_offset: int     # months between the init month and the first month of the period
    decade: int | None = None   # 1, 2, 3 for decades
    n_months: int = 1           # 3 for seasons

    # -------------------------------------------------------------- dates
    def dates(self, init_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        """First and last calendar day of the period for a given initialisation year."""
        y, m = _shift_month(init_year, self.init_month, self.month_offset)
        if self.scale == "decade":
            last = calendar.monthrange(y, m)[1]
            d0, d1 = {1: (1, 10), 2: (11, 20), 3: (21, last)}[self.decade]
            return pd.Timestamp(y, m, d0), pd.Timestamp(y, m, d1)
        y1, m1 = _shift_month(y, m, self.n_months - 1)
        return pd.Timestamp(y, m, 1), pd.Timestamp(y1, m1, calendar.monthrange(y1, m1)[1])

    def day_slice(self, init_year: int) -> tuple[int, int]:
        """Inclusive lead-day indices (0 = initialisation day) covered by the period."""
        init = pd.Timestamp(init_year, self.init_month, 1)
        start, end = self.dates(init_year)
        return (start - init).days, (end - init).days

    def n_days(self, init_year: int) -> int:
        i0, i1 = self.day_slice(init_year)
        return i1 - i0 + 1

    # -------------------------------------------------------------- labels
    @property
    def key(self) -> str:
        """Stable identifier independent of the year, e.g. ``decade_m1_d2``."""
        if self.scale == "decade":
            return f"decade_m{self.month_offset}_d{self.decade}"
        return f"{self.scale}_m{self.month_offset}"

    @property
    def months(self) -> list[int]:
        return [_shift_month(2000, self.init_month, self.month_offset + k)[1]
                for k in range(self.n_months)]

    def label(self, init_year: int) -> str:
        """Human-readable label, e.g. ``2026-09-D1``, ``2026-10``, ``NDJ 2026-27``."""
        start, end = self.dates(init_year)
        if self.scale == "decade":
            return f"{start:%Y-%m}-D{self.decade}"
        if self.scale == "month":
            return f"{start:%Y-%m}"
        acronym = "".join(_MONTH_INITIALS[m - 1] for m in self.months)
        years = f"{start:%Y}" if start.year == end.year else f"{start:%Y}-{end:%y}"
        return f"{acronym} {years}"

    def label_fr(self, init_year: int) -> str:
        """French label for maps and bulletins."""
        start, end = self.dates(init_year)
        if self.scale == "decade":
            ordinal = "1ʳᵉ" if self.decade == 1 else f"{self.decade}ᵉ"
            return f"{ordinal} décade {_MONTH_ABBR_FR[start.month - 1]} {start:%Y}"
        if self.scale == "month":
            return f"{_MONTH_ABBR_FR[start.month - 1]} {start:%Y}"
        return self.label(init_year)


def build_periods(init_date: pd.Timestamp, horizon_days: int,
                  scales=("decade", "month", "season")) -> list[Period]:
    """
    All complete target periods reachable by a forecast.

    Parameters
    ----------
    init_date : initialisation date (must be the 1st of a month).
    horizon_days : number of complete forecast days available (lead days
        0 .. horizon_days-1), e.g. the ``max_lead_days`` of the shortest model.
    scales : subset of ("decade", "month", "season").
    """
    init_date = pd.Timestamp(init_date)
    if init_date.day != 1:
        raise ValueError("init_date doit être le 1er du mois.")
    last_index = horizon_days - 1
    init_m, init_y = init_date.month, init_date.year

    candidates: list[Period] = []
    for off in range(0, horizon_days // 28 + 2):
        if "decade" in scales:
            candidates += [Period("decade", init_m, off, decade=d) for d in (1, 2, 3)]
        if "month" in scales:
            candidates.append(Period("month", init_m, off))
        if "season" in scales:
            candidates.append(Period("season", init_m, off, n_months=3))

    order = {"decade": 0, "month": 1, "season": 2}
    kept = [p for p in candidates if p.day_slice(init_y)[1] <= last_index]
    return sorted(kept, key=lambda p: (order[p.scale], p.month_offset, p.decade or 0))


def periods_table(periods: list[Period], init_year: int) -> pd.DataFrame:
    """Tabular view of the periods for one initialisation year."""
    rows = []
    for p in periods:
        start, end = p.dates(init_year)
        i0, i1 = p.day_slice(init_year)
        rows.append({"scale": p.scale, "key": p.key, "label": p.label(init_year),
                     "start": start.date(), "end": end.date(), "n_days": i1 - i0 + 1,
                     "lead_month": p.month_offset, "first_lead_day": i0, "last_lead_day": i1})
    return pd.DataFrame(rows)
