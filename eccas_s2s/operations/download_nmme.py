"""
Download the NMME ensemble-mean files of a cycle from the NOAA/CPC server (step E2).

For each model of ``systems.nmme.models`` and each variable of
``systems.nmme.variables``, every year listed on the server for the cycle's
initialisation month is downloaded (the full hindcast period, decision D4, plus
the current forecast) into ``<data_root>/raw/nmme/<YYYYMM>/<MODEL>/``. Files are
kept exactly as delivered (global, 1°). Existing files are not downloaded again.
Files are fetched ``workers`` at a time (the server answers file by file slowly).

Example::

    python scripts/run_download_nmme.py --config config/cycle_202609.yaml
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from eccas_s2s.io.nmme_cpc import DEFAULT_BASE_URL, download_file, list_available
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle


def disk_summary(cfg) -> pd.DataFrame:
    """Files present on disk for every configured model and variable (whatever was re-run)."""
    nmme = cfg.raw["systems"]["nmme"]
    dest = cfg.raw_dir("nmme")
    rows = []
    for model, spec in nmme["models"].items():
        for var in nmme["variables"]:
            years = sorted(int(f.name.split(".")[-4][:4]) for f in (dest / model).glob(f"{model}.{var}.*.nc"))
            hind = [y for y in years if y < cfg.init_date.year]
            rows.append({"model": model, "label": spec["label"], "variable": var,
                         "hindcast_first": min(hind) if hind else None,
                         "hindcast_last": max(hind) if hind else None,
                         "n_hindcast_years": len(hind),
                         "forecast_present": cfg.init_date.year in years, "files": len(years)})
    return pd.DataFrame(rows)


def run(config: str, models=None, variables=None, workers: int = 6) -> RunContext:
    cfg = load_cycle(config)
    nmme = cfg.raw["systems"]["nmme"]
    base = nmme.get("base_url", DEFAULT_BASE_URL)
    selected = list(models) if models else list(nmme["models"])
    variables = list(variables) if variables else list(nmme["variables"])
    dest = cfg.raw_dir("nmme")

    with RunContext(cfg, step="download_nmme") as ctx:
        ctx.record_parameter("base_url", base)
        ctx.record_parameter("models", selected)
        ctx.record_parameter("variables", variables)
        summary, failures = [], {}
        for model in selected:
            try:
                avail = list_available(base, cfg.init_date, model)
            except Exception as exc:
                failures[model] = f"liste indisponible : {exc}"
                ctx.warn(f"{model} : liste des fichiers indisponible ({exc})")
                continue
            for var in variables:
                years = sorted(avail.loc[avail.variable == var, "year"])
                if not years:
                    ctx.warn(f"{model} {var} : aucun fichier sur le serveur")
                    continue
                if cfg.init_date.year not in years:
                    ctx.warn(f"{model} {var} : prévision {cfg.init_date.year} absente du serveur")
                n_ok = 0
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(download_file, base, cfg.init_date, model, var, y, dest): y
                               for y in years}
                    for fut in as_completed(futures):
                        y = futures[fut]
                        try:
                            ctx.record_output(fut.result(), role=f"nmme_{var}", model=model, year=y)
                            n_ok += 1
                        except Exception as exc:
                            failures[f"{model}.{var}.{y}"] = str(exc)
                            ctx.warn(f"échec {model} {var} {y} : {exc}")
                hindcast = [y for y in years if y < cfg.init_date.year]
                summary.append({"model": model, "label": nmme["models"][model]["label"], "variable": var,
                                "hindcast_first": min(hindcast) if hindcast else None,
                                "hindcast_last": max(hindcast) if hindcast else None,
                                "n_hindcast_years": len(hindcast),
                                "forecast_present": cfg.init_date.year in years,
                                "files_ok": n_ok})
        table = disk_summary(cfg)
        out = dest / "nmme_download_summary.csv"
        dest.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        ctx.record_output(out, role="summary")
        ctx.record_parameter("summary", summary)
        ctx.record_parameter("failures", failures)
        ctx.log.info("terminé : %d fichier(s), %d échec(s)", len(ctx.outputs) - 1, len(failures))
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--variables", nargs="+")
    args = ap.parse_args(argv)
    run(args.config, args.models, args.variables)


if __name__ == "__main__":
    main()
