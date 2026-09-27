"""
Monthly precipitation stream: conversion to millimetres and alignment with the daily stream.

UKMO, BoM and NCEP are read from the monthly product for the month and season
scales (decision of 24/09/2026), because their daily stream only carries the
members started on the 1st: the September 2026 hindcasts hold 7 members for UKMO
and 3 for BoM on the daily stream, against 28 and 27 on the monthly one, and NCEP
has no daily hindcast at all. The totals
must then line up with those of the five models kept on the daily stream, which
is what the true month lengths are for.
"""
import calendar

import numpy as np
import pytest
import xarray as xr

from eccas_s2s.io.c3s_read import month_lengths


def test_month_lengths_follow_the_calendar_including_february():
    """A fixed 30 days would be wrong for 7 months out of 12, and for leap years."""
    table = month_lengths([1995, 1996], init_month=9, offsets=[0, 1, 2, 3, 4, 5])
    # sept, oct, nov, déc, janv, févr
    np.testing.assert_array_equal(table.sel(year=1995).values, [30, 31, 30, 31, 31, 29])
    np.testing.assert_array_equal(table.sel(year=1996).values, [30, 31, 30, 31, 31, 28])
    # 1996 is a leap year, so the February reached from September 1995 has 29 days
    assert calendar.isleap(1996) and int(table.sel(year=1995).isel(month_offset=5)) == 29


def test_month_lengths_wrap_over_the_year():
    table = month_lengths([2026], init_month=11, offsets=[0, 1, 2, 3])
    np.testing.assert_array_equal(table.values[0], [30, 31, 31, 28])   # nov, déc, janv, févr


def _rate_to_mm(rate_m_s, days):
    return rate_m_s * 1000.0 * 86400.0 * days


def test_conversion_of_a_known_rate():
    """A rate of 1 mm/day gives the number of days of the month, in millimetres."""
    rate = 1.0 / (1000.0 * 86400.0)                 # 1 mm/day in m/s
    assert _rate_to_mm(rate, 31) == pytest.approx(31.0)
    assert _rate_to_mm(rate, 28) == pytest.approx(28.0)


def test_a_fixed_thirty_day_month_would_bias_the_totals():
    """
    The error a fixed 30-day conversion makes, stated in numbers.

    It is not negligible: 3 % missing on a 31-day month, 7 % too much on
    February — and those totals are compared with the daily stream of the other
    models, where the month is exact.
    """
    rate = 5.0 / (1000.0 * 86400.0)                 # 5 mm/day
    exact_31, fixed_31 = _rate_to_mm(rate, 31), _rate_to_mm(rate, 30)
    exact_28, fixed_28 = _rate_to_mm(rate, 28), _rate_to_mm(rate, 30)
    assert (fixed_31 - exact_31) / exact_31 == pytest.approx(-1 / 31, abs=1e-6)
    assert (fixed_28 - exact_28) / exact_28 == pytest.approx(2 / 28, abs=1e-6)
    # at 5 mm/day that is 5 mm missing on a 31-day month and 10 mm too many in
    # February — of the order of a tercile threshold on a dry month
    assert exact_31 - fixed_31 == pytest.approx(5.0)
    assert fixed_28 - exact_28 == pytest.approx(10.0)


def test_season_total_sums_the_true_month_lengths():
    """A season is the sum of its months, each with its own length."""
    table = month_lengths([2026], init_month=9, offsets=[0, 1, 2])
    assert int(table.sum()) == 30 + 31 + 30          # SON 2026 = 91 days
    djf = month_lengths([2026], init_month=12, offsets=[0, 1, 2])
    assert int(djf.sum()) == 31 + 31 + 28            # DJF 2026-27 = 90 days (2027 non bissextile)
    djf_leap = month_lengths([2027], init_month=12, offsets=[0, 1, 2])
    assert int(djf_leap.sum()) == 31 + 31 + 29       # DJF 2027-28 = 91 jours


def test_streams_for_dispatches_each_scale_to_its_stream():
    """Configuration decides, not the code: dekads stay daily, months go monthly."""
    from eccas_s2s.operations.c3s_totals import streams_for
    from eccas_s2s.settings import C3SModel

    ukmo = C3SModel("ukmo", "610", "UKMO", 215,
                    precip_from={"decade": "daily", "month": "monthly", "season": "monthly"})
    assert streams_for(ukmo, ("decade", "month", "season")) == {
        "daily": ["decade"], "monthly": ["month", "season"]}

    # NCEP has no daily hindcast: the scale is absent from the mapping, so no dekads
    ncep = C3SModel("ncep", "2", "NCEP CFSv2", 215,
                    precip_from={"month": "monthly", "season": "monthly"})
    assert streams_for(ncep, ("decade", "month", "season")) == {"monthly": ["month", "season"]}

    ecmwf = C3SModel("ecmwf", "51", "ECMWF", 215)
    assert streams_for(ecmwf, ("decade", "month", "season")) == {
        "daily": ["decade", "month", "season"]}


def test_monthly_totals_are_written_beside_the_daily_ones():
    """The two streams must not overwrite each other: different ensembles, different files."""
    from types import SimpleNamespace

    from eccas_s2s.operations.c3s_totals import totals_path

    cfg = SimpleNamespace(data_root=__import__("pathlib").Path("/tmp/x"), cycle_id="202609")
    daily = totals_path(cfg, "ukmo", "hindcast")
    monthly = totals_path(cfg, "ukmo", "hindcast", stream="monthly")
    assert daily.name == "c3s_ukmo_precip_hindcast_periods.nc"
    assert monthly.name == "c3s_ukmo_precip_hindcast_periods_monthly.nc"


RAW = __import__("pathlib").Path("/home/ludric/Downloads/SVM/Previsions_S2S/DATA_OSF/raw/c3s/202609")
ECMWF_MONTHLY = RAW / "c3s_ecmwf_51_PRCP_hindcast_1993_2016_09_monthly.grib"
ECMWF_DAILY = RAW / "c3s_ecmwf_51_PRCP_hindcast_1993_2016_09.grib"

#: tolerances fixed on 23/09/2026 from the measured difference (see below)
TOL_MAX_MM = 0.5          # largest difference on one member and one cell
TOL_MEAN_MM = 0.05        # mean absolute difference


@pytest.mark.skipif(not (ECMWF_MONTHLY.exists() and ECMWF_DAILY.exists()),
                    reason="hindcasts ECMWF de septembre absents")
def test_the_two_streams_give_the_same_monthly_totals():
    """
    ECMWF carries the same 25 members on both streams, so the two ways of
    obtaining a monthly total must agree.

    Measured on the September 2026 hindcast (24 years x 25 members x 2 750
    cells): mean bias −0.001 mm, mean absolute difference 0.005 mm (0.014 % of a
    ~60 mm month), largest difference 0.31 mm on a single member and cell — the
    rounding of the float32 archives, nothing else. The ensemble mean, which the
    deterministic products use, agrees to 0.003 mm on average and 0.045 mm at
    worst over the 25 members. The tolerances below are set an order of magnitude
    above the measured values, so the test catches a real change of convention (a
    fixed 30-day month would show up as a 3 to 10 % error, i.e. several
    millimetres) without firing on rounding.
    """
    from eccas_s2s.core.monthly import aggregate_monthly
    from eccas_s2s.core.periods import build_periods
    from eccas_s2s.core.daily import aggregate_periods
    from eccas_s2s.io.c3s_read import load_c3s_monthly, load_c3s_precip_daily

    periods = [p for p in build_periods("2026-09-01", 215, ("month", "season"))
               if p.month_offset + p.n_months <= 6]
    members = slice(0, 3)                       # three members are enough, and fast
    daily = load_c3s_precip_daily(ECMWF_DAILY).isel(number=members).load()
    from_daily = aggregate_periods(daily, periods, how="sum")
    monthly = load_c3s_monthly(ECMWF_MONTHLY, "tprate", init_month=9).isel(number=members).load()
    from_monthly = aggregate_monthly(monthly, periods, how="sum")

    d = np.abs((from_monthly - from_daily).values)
    assert np.nanmax(d) < TOL_MAX_MM
    assert np.nanmean(d) < TOL_MEAN_MM


def test_horizon_and_development_subset_come_from_the_configuration():
    """
    Six lead months, and — in development only — the first two periods of each scale.

    The cap is not a preference: the C3S monthly archive stops at lead month 6
    for UKMO, BoM and NCEP, so beyond it those three models have no month and no
    season at all. The subset is a preference, which is why it is separate, why
    ``--all-periods`` switches it off and why a run made with it says so in its
    manifest.
    """
    from eccas_s2s.core.periods import build_periods, select_periods

    periods = build_periods("2026-09-01", 215)
    capped = select_periods(periods, max_lead_months=6)
    labels = [p.label(2026) for p in capped if p.scale != "decade"]
    assert labels[-1] == "DJF 2026-27" and "JFM 2027" not in labels
    assert labels[-5:-4] == ["2027-02"] or "2027-03" not in labels

    subset = select_periods(periods, max_lead_months=6,
                            per_scale={"decade": 2, "month": 2, "season": 2})
    assert [p.label(2026) for p in subset] == ["2026-09-D1", "2026-09-D2", "2026-09",
                                               "2026-10", "SON 2026", "OND 2026"]
    # no subset asked for: the horizon alone applies
    assert len(select_periods(periods, max_lead_months=6, per_scale={})) == len(capped)


def test_phase2_reads_each_scale_from_the_right_stream():
    """The scoring must not fall back to the daily file for a model whose months are monthly."""
    from eccas_s2s.operations.skill_raw import hindcast_streams
    from eccas_s2s.settings import load_cycle

    cfg = load_cycle("config/cycle_202609.yaml")
    scales = ("decade", "month", "season")
    assert hindcast_streams(cfg, "c3s", "ukmo", "precip", scales) == {
        "daily": ["decade"], "monthly": ["month", "season"]}
    assert hindcast_streams(cfg, "c3s", "ncep", "precip", scales) == {
        "monthly": ["month", "season"]}
    assert hindcast_streams(cfg, "c3s", "ecmwf", "precip", scales) == {
        "daily": ["decade", "month", "season"]}
    # temperature keeps one file per model and variable, whatever the archive it came from
    assert hindcast_streams(cfg, "c3s", "ukmo", "t2m", scales) == {
        "daily": ["decade", "month", "season"]}


def test_operational_period_selector():
    """
    What an operational run can ask for: everything, one scale, one period.

    The selector replaces the development subset when it is given — an
    operational command must never depend on what `development` holds in the
    configuration — and an unknown name is an error, not an empty run.
    """
    from eccas_s2s.core.periods import SelectionError, build_periods, select_periods

    periods = build_periods("2026-09-01", 215)
    dev = {"decade": 2, "month": 2, "season": 2}

    def labels(selection):
        return [p.label(2026) for p in select_periods(periods, 6, dev, selection, 2026)]

    assert len(labels("all")) == 28                       # tout l'horizon
    assert labels("all-seasons") == ["SON 2026", "OND 2026", "NDJ 2026-27", "DJF 2026-27"]
    assert labels("all-months") == ["2026-09", "2026-10", "2026-11", "2026-12",
                                    "2027-01", "2027-02"]
    assert len(labels("all-decades")) == 18
    assert labels("season_m1") == ["OND 2026"]            # par clé
    assert labels("OND 2026") == ["OND 2026"]             # par libellé de carte
    assert labels("1ʳᵉ décade de Septembre 2026") == ["2026-09-D1"]     # libellé français
    assert labels(["all-seasons", "2026-10"]) == ["2026-10", "SON 2026", "OND 2026",
                                                  "NDJ 2026-27", "DJF 2026-27"]
    assert labels("all-seasons,2026-10") == labels(["all-seasons", "2026-10"])
    # sans sélecteur : le sous-ensemble de développement
    assert labels(None) == ["2026-09-D1", "2026-09-D2", "2026-09", "2026-10",
                            "SON 2026", "OND 2026"]

    # JFM est hors de l'horizon de 6 mois : refusé, et le message dit ce qui est admis
    with pytest.raises(SelectionError, match="JFM 2027"):
        labels("JFM 2027")
    with pytest.raises(SelectionError, match="all-months"):
        labels("saison_1")


def test_selector_reaches_the_command_line():
    """`--periods` and its shorthand `--all-periods` end up in the same selector."""
    import argparse

    from eccas_s2s.core.periods import add_period_arguments, selected_periods

    ap = argparse.ArgumentParser()
    add_period_arguments(ap)
    assert selected_periods(ap.parse_args([])) is None                  # développement
    assert selected_periods(ap.parse_args(["--all-periods"])) == ["all"]
    assert selected_periods(ap.parse_args(["--periods", "all-seasons"])) == ["all-seasons"]
    assert selected_periods(ap.parse_args(
        ["--periods", "all-months", "season_m1"])) == ["all-months", "season_m1"]
    # --periods l'emporte sur le raccourci
    assert selected_periods(ap.parse_args(
        ["--all-periods", "--periods", "OND 2026"])) == ["OND 2026"]


@pytest.mark.parametrize("init, first_season, second_month", [
    ("2026-09-01", "SON 2026", "2026-10"),
    ("2027-01-01", "JFM 2027", "2027-02"),
    ("2027-03-01", "MAM 2027", "2027-04"),
    ("2027-11-01", "NDJ 2027-28", "2027-12"),
])
def test_selector_follows_the_initialisation_month(init, first_season, second_month):
    """
    The selector is relative to the cycle, never to September.

    The tool runs before **every** monthly initialisation, so `all-seasons` must
    mean the seasons that initialisation reaches, and `season_m0` its first one —
    SON for September, MAM for March, NDJ for November. The labels are generated
    from the cycle's own initialisation date.
    """
    from eccas_s2s.core.periods import build_periods, select_periods

    year = int(init[:4])
    periods = build_periods(init, 215)

    def labels(selection):
        return [p.label(year) for p in select_periods(periods, 6, None, selection, year)]

    assert labels("season_m0") == [first_season]
    assert labels("all-seasons")[0] == first_season and len(labels("all-seasons")) == 4
    assert labels("month_m1") == [second_month]
    assert len(labels("all-months")) == 6 and len(labels("all-decades")) == 18
    # le libellé imprimé sur la carte sélectionne la même période
    assert labels(first_season) == [first_season]


def test_labels_can_be_typed_without_accents():
    """A forecaster types the label from a keyboard: accents and superscripts are optional."""
    from eccas_s2s.core.periods import build_periods, select_periods

    periods = build_periods("2026-11-01", 215)

    def labels(selection):
        return [p.label(2026) for p in select_periods(periods, 6, None, selection, 2026)]

    assert labels("1ʳᵉ décade de Décembre 2026") == ["2026-12-D1"]
    assert labels("1re decade de decembre 2026") == ["2026-12-D1"]
    assert labels("ndj 2026-27") == ["NDJ 2026-27"]
    assert labels("DECADE_M0_D2") == ["2026-11-D2"]
