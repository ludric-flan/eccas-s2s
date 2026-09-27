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
import unicodedata
from dataclasses import dataclass

import pandas as pd

_MONTH_INITIALS = "JFMAMJJASOND"
_MONTH_ABBR_FR = ["janv.", "févr.", "mars", "avr.", "mai", "juin",
                  "juil.", "août", "sept.", "oct.", "nov.", "déc."]
_MONTH_NAME_FR = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet",
                  "Août", "Septembre", "Octobre", "Novembre", "Décembre"]


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
    def calendar_key(self) -> str:
        """
        Cycle-independent calendar identifier, used to share observed normals
        across cycles: ``dekad_MM_D``, ``month_MM`` or ``season_MM`` (MM = first
        calendar month of the period).
        """
        mm = self.months[0]
        if self.scale == "decade":
            return f"dekad_{mm:02d}_{self.decade}"
        return f"{self.scale}_{mm:02d}"

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

    def label_fr(self, init_year: int, with_dates: bool = False) -> str:
        """
        Explicit French label for maps and bulletins.

        ``Novembre 2026``, ``OND 2026``, ``1ʳᵉ décade de Novembre 2026``; with
        ``with_dates`` the exact window is appended, as the reference chain does
        on its product maps (``format_period_label`` of ``s2s_plotting_v2.py``).
        """
        start, end = self.dates(init_year)
        if self.scale == "decade":
            ordinal = "1ʳᵉ" if self.decade == 1 else f"{self.decade}ᵉ"
            head = f"{ordinal} décade de {_MONTH_NAME_FR[start.month - 1]} {start:%Y}"
        elif self.scale == "month":
            head = f"{_MONTH_NAME_FR[start.month - 1]} {start:%Y}"
        else:
            head = self.label(init_year)
        if not with_dates:
            return head
        return (f"{head} ({start:%d} {_MONTH_ABBR_FR[start.month - 1]} – "
                f"{end:%d} {_MONTH_ABBR_FR[end.month - 1]} {end:%Y})")


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


#: selection tokens that are not a period key or a label
ALL = "all"
SCALE_TOKENS = {"all-decades": "decade", "all-dekads": "decade",
                "all-months": "month", "all-seasons": "season"}


class SelectionError(ValueError):
    """Raised when a period selector matches nothing."""


def _tokens(selection) -> list[str]:
    """Normalise a selector into a list of tokens (a string may hold several, comma-separated)."""
    if selection is None:
        return []
    if isinstance(selection, str):
        selection = [selection]
    out = []
    for item in selection:
        out += [t.strip() for t in str(item).split(",") if t.strip()]
    return out


#: superscripts of the French ordinals used in the map labels ("1ʳᵉ décade…")
_SUPERSCRIPTS = str.maketrans({"ʳ": "r", "ᵉ": "e"})


def normalise(text: str) -> str:
    """
    Comparison form of a selector or a label: no case, no accent, no superscript.

    A forecaster types what is written on the map, and that label carries
    accents and an ordinal superscript ("1ʳᵉ décade de Novembre 2026"). Matching
    on the exact characters would make the selector unusable from a keyboard, so
    both sides are reduced to the same plain form: "1re decade de novembre 2026".
    """
    text = unicodedata.normalize("NFKD", str(text).translate(_SUPERSCRIPTS))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.lower().split())


def matches(period: Period, token: str, init_year: int | None = None) -> bool:
    """
    Whether one selector token designates this period.

    Everything here is relative to the **initialisation month** of the cycle:
    ``all-seasons`` means the seasons this initialisation reaches, ``season_m1``
    the second one, whatever the month. The labels are generated from the cycle's
    own initialisation date, so the same selector serves every monthly cycle —
    with a September init ``season_m1`` is OND, with a March init it is AMJ.
    """
    low = normalise(token)
    if low == ALL:
        return True
    if low in SCALE_TOKENS:
        return period.scale == SCALE_TOKENS[low]
    if low in ("decade", "dekad", "month", "season"):
        return period.scale == ("decade" if low in ("decade", "dekad") else low)
    if normalise(period.key) == low:
        return True
    if init_year is not None:
        # the labels a forecaster reads on the maps and in the bulletin
        for label in (period.label(init_year), period.label_fr(init_year)):
            if normalise(label) == low:
                return True
    return False


def select_periods(periods: list[Period], max_lead_months: int | None = None,
                   per_scale: dict | None = None, selection=None,
                   init_year: int | None = None) -> list[Period]:
    """
    Restrict a list of periods to the horizon, then to what was asked for.

    ``max_lead_months`` counts the lead months a period needs: a period is kept
    when ``month_offset + n_months <= max_lead_months``. Six lead months keep
    February and DJF for a September initialisation, and drop March and JFM —
    which is exactly what the C3S monthly archive offers for UKMO, BoM and NCEP
    (``leadtime_month`` 7 is refused by the CDS), so this is the horizon every
    model can honour.

    ``selection`` is what an operational run asks for, and it decides alone when
    it is given — the development subset is then ignored:

    =========================  ===================================================
    ``"all"``                  every period of the horizon
    ``"all-months"``           every month (likewise ``all-seasons``, ``all-decades``)
    ``"season_m1"``            one period by its key
    ``"OND 2026"``             one period by the label printed on the maps
    ``["all-seasons", "2026-10"]``  several selectors at once (also ``"a,b"``)
    =========================  ===================================================

    ``per_scale`` (e.g. ``{"decade": 2, "month": 2, "season": 2}``) keeps only the
    first *n* periods of each scale. It is the **development** convenience — the
    first dekads, months and seasons are enough to exercise the whole chain at a
    fraction of the cost — and it applies only when no selection is given, which
    is why an operational command never depends on what that key contains.

    A selector that matches nothing raises :class:`SelectionError` rather than
    producing an empty run: a typo in a period name must not look like a model
    with no data.
    """
    kept = list(periods)
    if max_lead_months is not None:
        kept = [p for p in kept if p.month_offset + p.n_months <= max_lead_months]

    tokens = _tokens(selection)
    if tokens:
        chosen = [p for p in kept if any(matches(p, t, init_year) for t in tokens)]
        unmatched = [t for t in tokens
                     if not any(matches(p, t, init_year) for p in kept)]
        if unmatched:
            known = ", ".join(p.key for p in kept)
            raise SelectionError(
                f"période(s) inconnue(s) dans l'horizon : {', '.join(unmatched)}. "
                f"Valeurs admises : all, all-decades, all-months, all-seasons, "
                f"une clé ({known}) ou un libellé de carte.")
        return chosen

    if per_scale:
        counted: dict[str, int] = {}
        subset = []
        for p in kept:                      # build_periods orders by scale, then lead
            limit = per_scale.get(p.scale)
            counted[p.scale] = counted.get(p.scale, 0) + 1
            if limit is None or counted[p.scale] <= int(limit):
                subset.append(p)
        kept = subset
    return kept


def announce_subset(cfg, ctx=None, selection=None) -> dict:
    """
    State, in the log and in the manifest, which periods a run covers.

    Three cases, and each leaves a trace: an **operational** run says which
    selector it was given, a **development** run is logged as a warning (a
    partial score table must never be mistaken for a complete one), and a run
    over the whole horizon says so plainly.
    """
    tokens = _tokens(selection)
    subset = {} if tokens else cfg.periods_per_scale
    info = {"max_lead_months": cfg.max_lead_months,
            "selection": tokens or None,
            "periods_per_scale": subset or None}
    if ctx is not None:
        ctx.record_parameter("horizon", info)
        if cfg.max_lead_months:
            ctx.log.info("horizon : %d mois d'échéance", cfg.max_lead_months)
        if tokens:
            ctx.log.info("périodes demandées : %s", ", ".join(tokens))
        elif subset:
            ctx.warn("SOUS-ENSEMBLE DE DÉVELOPPEMENT : "
                     + ", ".join(f"{n} première(s) période(s) {s}" for s, n in subset.items())
                     + " — en exploitation, passer --periods (all, all-months, "
                       "all-seasons, une clé ou un libellé)")
        else:
            ctx.log.info("toutes les périodes de l'horizon")
    return info


def add_period_arguments(parser) -> None:
    """
    Add the period selector to a command-line parser.

    Two options, one meaning: ``--periods`` is the operational selector,
    ``--all-periods`` its shorthand for "everything in the horizon". With
    neither, the run uses the development subset of the configuration — and says
    so in its log and its manifest.
    """
    parser.add_argument("--periods", nargs="+", metavar="SÉLECTEUR",
                        help="périodes à traiter : all, all-decades, all-months, all-seasons, "
                             "une clé (season_m1) ou un libellé de carte (\"OND 2026\") ; "
                             "par défaut, le sous-ensemble de développement")
    parser.add_argument("--all-periods", action="store_true",
                        help="raccourci de --periods all")


def selected_periods(args):
    """The selector a parser built by :func:`add_period_arguments` carries."""
    if getattr(args, "periods", None):
        return list(args.periods)
    return [ALL] if getattr(args, "all_periods", False) else None


def period_from_key(key: str, init_month: int) -> Period:
    """
    Rebuild a :class:`Period` from its key and the initialisation month.

    The key (``season_m1``, ``month_m0``, ``decade_m2_d3``) is what the netCDF
    files carry; this is what lets a reader turn it back into calendar dates
    without the object that produced it.
    """
    parts = key.split("_")
    scale = parts[0]
    offset = int(parts[1].lstrip("m"))
    if scale == "decade":
        return Period("decade", init_month, offset, decade=int(parts[2].lstrip("d")))
    if scale == "season":
        return Period("season", init_month, offset, n_months=3)
    if scale == "month":
        return Period("month", init_month, offset)
    raise ValueError(f"clé de période inconnue : {key}")
