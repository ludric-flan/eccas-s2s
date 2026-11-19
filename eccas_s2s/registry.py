"""
Registers of the chain, one folder per cycle.

A register answers an operational question for **one initialisation**: which
models are admitted (``models_eligibility.csv``), how each product behaves raw
(``products_eligibility.csv``), which calibration each model receives for each
product (``calibration_methods.csv``). The answer depends on the init month —
the target periods and the leads change from one cycle to the next — so each
cycle keeps its own registers::

    <output_root>/registry/<YYYYMM>/calibration_methods.csv
    <output_root>/registry/<YYYYMM>/products_eligibility.csv
    <output_root>/registry/<YYYYMM>/models_eligibility.csv

A register is never the only copy of what it says: it is **derived** from the
summary tables of the cycle (``skill_raw_summary.csv``,
``comparison_raw_calibrated.csv``), themselves merged run after run and
rebuildable from the score netCDFs. Re-running one model therefore refreshes
that model's lines and keeps the others (incident of 17/11/2026: a run on NCEP
alone had left a register of thirteen lines).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

#: file name of each register
CALIBRATION = "calibration_methods.csv"
PRODUCTS = "products_eligibility.csv"
MODELS = "models_eligibility.csv"


def registry_dir(cfg) -> Path:
    """``<output_root>/registry/<cycle>/``."""
    return cfg.path_of("output_root") / "registry" / cfg.cycle_id


def registry_path(cfg, name: str) -> Path:
    return registry_dir(cfg) / name


def read_register(cfg, name: str) -> pd.DataFrame | None:
    """The register of the cycle, or ``None`` when it has not been produced yet."""
    path = registry_path(cfg, name)
    return pd.read_csv(path) if path.exists() else None


def write_register(cfg, table: pd.DataFrame, name: str) -> Path:
    path = registry_path(cfg, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def recommended_methods(cfg, keys=("model", "variable", "scale", "product")) -> dict:
    """``keys -> method`` from the calibration register of the cycle (empty if absent)."""
    table = read_register(cfg, CALIBRATION)
    if table is None:
        return {}
    keys = list(keys)
    return {tuple(row[k] for k in keys): row["method"] for _, row in table.iterrows()}


def merge_rows(new: pd.DataFrame, path: Path, keys: list[str]) -> pd.DataFrame:
    """
    ``new`` merged into the table already at ``path``: new rows replace the old
    rows with the same key, the other old rows are kept.
    """
    if not path.exists():
        return new
    old = pd.read_csv(path)
    if not set(keys) <= set(old.columns):
        return new
    for k in keys:              # same dtype on both sides, or isin() misses
        old[k] = old[k].astype(str)
        new = new.assign(**{k: new[k].astype(str)})
    old = old[~old.set_index(keys).index.isin(new.set_index(keys).index)]
    return pd.concat([old, new], ignore_index=True)
