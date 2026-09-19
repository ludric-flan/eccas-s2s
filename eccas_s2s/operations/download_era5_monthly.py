"""
Observed 2 m mean temperature: ERA5 monthly means at 0.25° (workflow step E1).

Decision of 2026-09-19: the model T2m comes from the C3S monthly statistics
(months and seasons only), so the observed T2m reference is the ERA5 monthly
mean (``reanalysis-era5-single-levels-monthly-means``), which the CDS serves
quickly. Daily statistics (much slower to obtain) are kept for Tmax and Tmin,
which need decades.

Files in ``<data_root>/raw/era5/monthly_0p25/``: one for the complete years,
one for the current year up to the last published month.

Example::

    python scripts/run_download_era5_monthly.py --config config/cycle_202609.yaml --years 1981 2026
"""
from __future__ import annotations

import argparse

import pandas as pd

from eccas_s2s.io.era5 import download_era5
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle

#: ECCAS box (lon_min, lon_max, lat_min, lat_max) — same box as CHIRPS.
AREA = (5.0, 35.0, -20.0, 25.0)


def run(config: str, first_year: int, last_year: int) -> RunContext:
    cfg = load_cycle(config)
    out_dir = cfg.data_root / "raw" / "era5" / "monthly_0p25"
    today = pd.Timestamp.today()
    # ERA5 monthly means are published a few days after the end of the month
    last_complete = (today - pd.Timedelta(days=6)).to_period("M") - 1
    with RunContext(cfg, step="download_era5_monthly") as ctx:
        ctx.record_parameter("area", list(AREA))
        full_last = min(last_year, last_complete.year - (0 if last_complete.month == 12 else 1))
        chunks = [(list(range(first_year, full_last + 1)), list(range(1, 13)), "t2m_monthly")]
        if last_year > full_last and last_complete.year == full_last + 1:
            chunks.append(([last_complete.year], list(range(1, last_complete.month + 1)),
                           f"t2m_monthly_to{last_complete.month:02d}"))
        for years, months, name in chunks:
            ctx.log.info("ERA5 moyennes mensuelles T2m %d–%d, mois %d–%d", years[0], years[-1], months[0], months[-1])
            path = download_era5(["TEMP"], years, months, AREA, str(out_dir), name)
            ctx.record_output(path, role="era5_t2m_monthly", years=f"{years[0]}-{years[-1]}")
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", nargs=2, type=int, required=True, metavar=("PREMIERE", "DERNIERE"))
    args = ap.parse_args(argv)
    run(args.config, args.years[0], args.years[1])


if __name__ == "__main__":
    main()
