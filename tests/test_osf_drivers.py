"""
Indices de pilotes océaniques et diagnostic (E3, marche 1 de la phase E7).

Les boîtes viennent de la littérature reprise par le cadre « drivers » du
CAPC-AC ; ces tests verrouillent les définitions et les conventions, parce
qu'une boîte mal orientée ou une standardisation oubliée produit un indice
plausible et faux.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eccas_s2s.predictors.sst_indices import (BOXES, DIPOLES, INDICES, box_mean,
                                              compute_indices, seasonal_mean, standardise)


def _sst(years=range(1991, 2021)):
    """SST synthétique : un signal dans Niño3.4, un dipôle dans l'Indien."""
    lat = np.arange(-40, 41, 2.0)
    lon = np.arange(-180, 180, 2.0)
    times = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-01", freq="MS")
    rng = np.random.default_rng(0)
    data = 20 + rng.normal(0, 0.1, (len(times), len(lat), len(lon)))
    da = xr.DataArray(data, dims=("time", "lat", "lon"),
                      coords={"time": times, "lat": lat, "lon": lon})
    # +2 °C dans la boîte Niño3.4 une année sur deux
    nino = (da["lon"] >= -170) & (da["lon"] <= -120) & (da["lat"] >= -5) & (da["lat"] <= 5)
    pair = xr.DataArray([(t.year % 2 == 0) * 2.0 for t in times], dims="time",
                        coords={"time": times})
    return da + nino * pair


def test_les_boites_suivent_la_litterature():
    assert BOXES["NINO34"] == (-170.0, -120.0, -5.0, 5.0)
    assert BOXES["ATL3"] == (-20.0, 0.0, -3.0, 3.0)          # Zebiak 1993
    assert BOXES["WTIO"] == (50.0, 70.0, -10.0, 10.0)        # Saji 1999
    assert BOXES["SETIO"] == (90.0, 110.0, -10.0, 0.0)
    assert BOXES["SAOD_NEP"] == (-20.0, 10.0, -15.0, 0.0)    # Nnamchi 2011
    assert BOXES["SAOD_SWP"] == (-40.0, -10.0, -40.0, -25.0)
    assert DIPOLES == {"DMI": ("WTIO", "SETIO"), "SAOD": ("SAOD_NEP", "SAOD_SWP")}
    # les cinq pilotes de l'Afrique centrale, et eux seuls
    assert set(INDICES) == {"NINO34", "DMI", "ATL3", "GG", "SAOD"}


def test_un_indice_est_une_anomalie_standardisee():
    """Sans standardisation, le Golfe de Guinée écraserait ENSO d'un facteur dix."""
    sst = _sst()
    years = list(range(1991, 2021))
    idx = compute_indices(sst, months=(6, 7, 8), years=years, clim=(1991, 2020))
    nino = idx["NINO34"].to_series()
    assert abs(nino.mean()) < 1e-9 and abs(nino.std(ddof=1) - 1) < 1e-9
    # le signal injecté une année sur deux doit ressortir avec le bon signe
    pairs = nino[[y % 2 == 0 for y in years]]
    impairs = nino[[y % 2 == 1 for y in years]]
    assert pairs.mean() > 0.9 and impairs.mean() < -0.9


def test_un_dipole_est_la_difference_des_poles_standardises():
    """Et non la standardisation de la différence : les deux diffèrent si les variances diffèrent."""
    sst = _sst()
    years = list(range(1991, 2021))
    idx = compute_indices(sst, months=(9, 10, 11), years=years)
    wtio = standardise(seasonal_mean(box_mean(sst, BOXES["WTIO"]), (9, 10, 11), years))
    setio = standardise(seasonal_mean(box_mean(sst, BOXES["SETIO"]), (9, 10, 11), years))
    np.testing.assert_allclose(idx["DMI"].values, (wtio - setio).values, atol=1e-10)


def test_une_saison_a_cheval_sur_l_annee_est_rattachee_a_son_premier_mois():
    """NDJ 2015 = novembre et décembre 2015, janvier 2016."""
    sst = _sst(range(2014, 2018))
    serie = box_mean(sst, BOXES["NINO34"])
    ndj = seasonal_mean(serie, (11, 12, 1), [2015])
    attendu = serie.sel(time=[pd.Timestamp(2015, 11, 1), pd.Timestamp(2015, 12, 1),
                              pd.Timestamp(2016, 1, 1)]).mean("time")
    np.testing.assert_allclose(float(ndj.isel(year=0)), float(attendu), atol=1e-10)


def test_la_part_significative_se_compare_au_hasard():
    """5 % du masque est significatif par hasard : c'est la ligne de flottaison."""
    from eccas_s2s.operations.driver_diagnostic import CHANCE, significant_fraction

    lat, lon = np.linspace(-10, 10, 21), np.linspace(10, 30, 21)
    coords = {"latitude": lat, "longitude": lon}
    corr = xr.DataArray(np.full((21, 21), 0.10), dims=("latitude", "longitude"), coords=coords)
    mask = xr.full_like(corr, True, dtype=bool)
    assert significant_fraction(corr, mask, 44) == 0.0          # |r| = 0,10 : jamais significatif
    fort = xr.full_like(corr, 0.45)
    assert significant_fraction(fort, mask, 44) == pytest.approx(1.0)
    assert CHANCE == 0.05
