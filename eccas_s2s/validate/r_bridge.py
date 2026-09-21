"""
Bridge to the R ``verification`` package (workflow step E4/E9, Draft §5.2).

The Draft Framework asks for standardised metric *implementations*; the
reference chain of the CAPC-AC computes them with the R ``verification``
package, and this chain keeps that dependency for the zone-average scores:
RPS/RPSS, Brier and its reliability/resolution/uncertainty decomposition,
ROC area with its p-value, CRPS, Heidke/Peirce/Gerrity, plus the reliability
bins. The grid-point maps are computed in Python
(:mod:`eccas_s2s.validate.scores`); the two are cross-checked in the tests.

Install R and the package once::

    conda env update -f environment.yml     # r-base, r-verification, r-boot
    # or, with an R already installed:
    Rscript -e 'install.packages(c("verification","boot"), repos="https://cloud.r-project.org")'
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import xarray as xr

R_SCRIPT = Path(__file__).with_name("zone_scores.R")
R_DIAGRAMS = Path(__file__).with_name("zone_diagrams.R")
REQUIRED_PACKAGES = ("verification",)
OUTPUT_FILES = ("deterministic_scores.csv", "tercile_scores.csv",
                "category_scores.csv", "reliability_bins.csv")


class RNotAvailable(RuntimeError):
    """Raised when Rscript or the verification package cannot be used."""


def find_rscript(rscript: str | None = None) -> str:
    """Path of the ``Rscript`` executable (raises if missing)."""
    path = rscript or shutil.which("Rscript")
    if not path:
        raise RNotAvailable(
            "Rscript est introuvable. Installer R et le paquet verification :\n"
            "  conda env update -f environment.yml\n"
            "  ou Rscript -e 'install.packages(c(\"verification\",\"boot\"))'")
    return path


def check_packages(rscript: str | None = None) -> dict:
    """Version of R and availability of the required packages."""
    path = find_rscript(rscript)
    probe = ('cat(R.version.string, "|");'
             'for (p in c("verification","boot")) cat(p, requireNamespace(p, quietly=TRUE), "|")')
    out = subprocess.run([path, "-e", probe], capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise RNotAvailable(f"Rscript a échoué : {out.stderr[:300]}")
    parts = [p.strip() for p in out.stdout.split("|") if p.strip()]
    info = {"rscript": path, "r_version": parts[0]}
    for item in parts[1:]:
        name, ok = item.split()
        info[name] = ok == "TRUE"
    missing = [p for p in REQUIRED_PACKAGES if not info.get(p)]
    if missing:
        raise RNotAvailable(
            f"paquet(s) R manquant(s) : {missing}. Installer avec :\n"
            "  conda env update -f environment.yml\n"
            "  ou Rscript -e 'install.packages(c(\"verification\",\"boot\"))'")
    return info


def pairs_to_frame(index: xr.Dataset) -> pd.DataFrame:
    """Turn a zone index (from :func:`eccas_s2s.validate.pairs.zone_index`) into a table."""
    periods = (list(index["period"].values) if "period" in index.dims
               else [index.attrs.get("period", "single")])
    rows = []
    for p in periods:
        sel = index.sel(period=p) if "period" in index.dims else index
        row = {"period": p, "year": sel["year"].values, "ensmean": sel["ensmean"].values,
               "ens_sd": sel["ens_sd"].values, "obs": sel["obs"].values,
               "obs_cat": sel["obs_cat"].values}
        if "prob" in sel:        # a system without members has no probabilities
            row["pBN"] = sel["prob"].sel(category="BN").values
            row["pNN"] = sel["prob"].sel(category="NN").values
            row["pAN"] = sel["prob"].sel(category="AN").values
        elif "fcst_cat" in sel:  # ensemble mean only: a categorical forecast instead
            row["fcst_cat"] = sel["fcst_cat"].values
        rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True)


def run_zone_diagrams(frame: pd.DataFrame, out_dir: str | Path, label: str,
                      n_boot: int = 200, rscript: str | None = None) -> list[Path]:
    """
    Draw the reliability and ROC diagrams of a zone with R.

    Two figures per period: the attributes/reliability diagram and the ROC
    diagram, each carrying the three tercile categories on the same axes.

    ``frame`` holds the **pooled grid-point pairs** of the zone
    (:func:`eccas_s2s.validate.pooled.pooled_frame`); a frame without
    probabilities (ensemble-mean system) produces no figure, which the script
    reports without failing. Returns the figures written.
    """
    path = find_rscript(rscript)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled_csv = out_dir / "pooled_pairs.csv"
    frame.to_csv(pooled_csv, index=False)
    res = subprocess.run([path, str(R_DIAGRAMS), str(pooled_csv), str(out_dir), label,
                          str(int(n_boot))], capture_output=True, text=True, timeout=7200)
    if res.returncode != 0:
        raise RNotAvailable(f"zone_diagrams.R a échoué ({label}) :\n{res.stderr[-800:]}")
    return sorted(list(out_dir.glob("reliability_*.png")) + list(out_dir.glob("roc_*.png")))


def run_zone_scores(frame: pd.DataFrame, out_dir: str | Path, label: str,
                    rscript: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Compute the zone scores of ``frame`` with R and read the CSVs back.

    ``label`` identifies the run (model, variable, zone) and is copied into
    every output row.
    """
    path = find_rscript(rscript)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = out_dir / "pairs.csv"
    frame.to_csv(pairs_csv, index=False)
    res = subprocess.run([path, str(R_SCRIPT), str(pairs_csv), str(out_dir), label],
                         capture_output=True, text=True, timeout=3600)
    if res.returncode != 0:
        raise RNotAvailable(f"zone_scores.R a échoué ({label}) :\n{res.stderr[-800:]}")
    tables = {}
    for name in OUTPUT_FILES:
        f = out_dir / name
        if f.exists():
            tables[name.replace(".csv", "")] = pd.read_csv(f)
    return tables
