"""
Download the C3S forecasts and hindcasts of a cycle (workflow step E2).

Everything (models, systems, horizon, area, hindcast years, destination) comes
from the cycle configuration. Each file is written once to the immutable raw
directory ``<data_root>/raw/c3s/<YYYYMM>/`` and fingerprinted in the run manifest.

Examples
--------
Show the CDS requests without downloading::

    python scripts/run_download_c3s.py --config config/cycle_202609.yaml --dry-run

Download precipitation forecasts and hindcasts for all models::

    python scripts/run_download_c3s.py --config config/cycle_202609.yaml --variable precip

A model that fails does not stop the others; failures are listed in the manifest.
"""
from __future__ import annotations

import argparse
import json
import time

from eccas_s2s.io.c3s import build_c3s_monthly_request, build_c3s_request, download_c3s
from eccas_s2s.provenance import RunContext
from eccas_s2s.settings import load_cycle

#: OSF variable name -> key of eccas_s2s.io.c3s.C3S_VARIABLES
VARIABLE_KEYS = {"precip": "PRCP", "t2m": "TEMP", "tmax": "TMAX", "tmin": "TMIN"}

#: variables taken from the C3S *monthly* statistics (months and seasons only):
#: the 2 m mean temperature (decision of 2026-09-19, lighter than 6-hourly fields).
MONTHLY_VARIABLES = {"t2m"}

MAX_ATTEMPTS = 5


def _with_retry(func, log, what):
    """Call ``func`` with exponential back-off (1, 2, 4, 8 min) on failure."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return func()
        except Exception as exc:  # CDS errors are not typed consistently
            if attempt == MAX_ATTEMPTS:
                raise
            wait = 60 * 2 ** (attempt - 1)
            log.warning("%s : tentative %d/%d échouée (%s) — nouvel essai dans %d s",
                        what, attempt, MAX_ATTEMPTS, exc, wait)
            time.sleep(wait)


def run(config: str, variable: str = "precip", kinds=("forecast", "hindcast"),
        models=None, dry_run: bool = False) -> RunContext:
    """
    Download (or, with ``dry_run``, only log) the C3S files of a cycle.

    Files already present are not downloaded again. Returns the finished
    :class:`RunContext` (its ``outputs`` list the files, ``parameters`` the failures).
    """
    cfg = load_cycle(config)
    all_models = cfg.c3s_models
    selected = list(models) if models else list(all_models)
    unknown = sorted(set(selected) - set(all_models))
    if unknown:
        raise ValueError(f"modèles absents de la configuration : {unknown}")
    if variable not in VARIABLE_KEYS:
        raise ValueError(f"variable inconnue {variable!r} ; valeurs : {sorted(VARIABLE_KEYS)}")

    init = cfg.init_date
    area = cfg.c3s_download_area
    dest_dir = cfg.raw_dir("c3s")
    var_key = VARIABLE_KEYS[variable]

    with RunContext(cfg, step=f"download_c3s_{variable}") as ctx:
        ctx.record_parameter("variable", variable)
        ctx.record_parameter("kinds", list(kinds))
        ctx.record_parameter("models", selected)
        ctx.record_parameter("dry_run", dry_run)
        ctx.record_parameter("destination", str(dest_dir))
        failures = {}

        for centre in selected:
            m = all_models[centre]
            monthly = variable in MONTHLY_VARIABLES
            leadtime_hours = [str(h) for h in range(24, 24 * m.max_lead_days + 1, 24)]
            for kind in kinds:
                years = [str(init.year)] if kind == "forecast" else [str(y) for y in cfg.c3s_hindcast_years]
                what = f"{centre} sys {m.system} {kind}"
                if monthly:
                    dataset, request = build_c3s_monthly_request(centre, var_key, years, init.month,
                                                                  area, system=m.system)
                    shown = dict(request)
                else:
                    dataset, request = build_c3s_request(
                        centre, var_key, years, init.month, area,
                        system=m.system, leadtime_hours=leadtime_hours)
                    shown = {**request, "leadtime_hour": f"24..{24 * m.max_lead_days} (pas 24 h)"}
                ctx.record_parameter(f"request.{centre}.{kind}", {"dataset": dataset, **shown})
                if dry_run:
                    ctx.log.info("[dry-run] %s : %s", what, json.dumps(
                        {k: v for k, v in request.items() if k != "leadtime_hour"}))
                    continue
                try:
                    path = _with_retry(
                        lambda: download_c3s(centre, var_key, years, init.month, area, str(dest_dir),
                                             system=m.system, leadtime_hours=leadtime_hours, kind=kind,
                                             monthly=monthly),
                        ctx.log, what)
                    ctx.record_output(path, role=f"c3s_{variable}_{kind}", centre=centre,
                                      system=m.system, years=f"{years[0]}-{years[-1]}")
                except Exception as exc:
                    failures[what] = f"{type(exc).__name__}: {exc}"
                    ctx.warn(f"échec {what} : {exc}")

        ctx.record_parameter("failures", failures)
        ctx.log.info("terminé : %d fichier(s), %d échec(s)", len(ctx.outputs), len(failures))
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="fichier config/cycle_YYYYMM.yaml")
    ap.add_argument("--variable", default="precip", choices=sorted(VARIABLE_KEYS))
    ap.add_argument("--kind", nargs="+", default=["forecast", "hindcast"],
                    choices=["forecast", "hindcast"])
    ap.add_argument("--models", nargs="+", help="sous-ensemble de modèles (défaut : tous)")
    ap.add_argument("--dry-run", action="store_true", help="afficher les requêtes sans télécharger")
    args = ap.parse_args(argv)
    run(args.config, args.variable, args.kind, args.models, args.dry_run)


if __name__ == "__main__":
    main()
