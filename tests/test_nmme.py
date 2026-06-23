"""
Offline tests for the NMME / IRIDL acquisition module: URL construction and the
'months since 1960-01-01' start-time decoding. No network required.
"""
import pandas as pd

from eccas_s2s.io import nmme


def test_build_nmme_url_matches_iridl_syntax():
    area = (4, 36, -21, 25)  # ECCAS box: lon_min, lon_max, lat_min, lat_max
    url = nmme.build_nmme_url(
        "ncep_cfsv2", "PRCP", "2000-06-01", area,
        dataset="HINDCAST", leads=(0.5, 2.5), fmt="data.nc",
    )
    expected = (
        "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME"
        "/.NCEP-CFSv2/.HINDCAST/.MONTHLY/.prec"
        "/X/(4E)/(36E)/RANGEEDGES"
        "/Y/(21S)/(25N)/RANGEEDGES"
        "/L/(0.5)/(2.5)/RANGEEDGES"
        "/S/(0000%201%20Jun%202000)/VALUES"
        "/data.nc"
    )
    assert url == expected


def test_build_nmme_url_dods_and_no_leads():
    url = nmme.build_nmme_url("ncep_cfsv2", "PRCP", "1998-01-01",
                              (4, 36, -21, 25), fmt="dods")
    assert url.endswith("/dods")
    assert "/L/" not in url  # leads=None -> no L subset


def test_build_nmme_url_negative_lon_uses_W():
    url = nmme.build_nmme_url("ncep_cfsv2", "PRCP", "2000-06-01",
                              (-10, 36, -21, 25))
    assert "/X/(10W)/(36E)/RANGEEDGES" in url


def test_decode_S():
    # 485 months after 1960-01-01 == June 2000
    idx = nmme.decode_S([485])
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx[0] == pd.Timestamp("2000-06-01")
