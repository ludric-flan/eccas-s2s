"""
Cycle configuration for the OSF chain (workflow step E0, rule R2).

A forecast cycle is fully described by one YAML file, ``config/cycle_YYYYMM.yaml``,
plus the shared files it includes (domains, thresholds, agro calendars). Nothing
cycle-specific is hard-coded in the package: every script receives a
:class:`CycleConfig` built by :func:`load_cycle`.

This module replaces, for the OSF chain, the constants of the legacy
``eccas_s2s.config`` module (which is kept unchanged for the existing pipeline).
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

VALID_SCALES = ("decade", "month", "season")
VALID_VARIABLES = ("precip", "t2m", "tmax", "tmin")


class ConfigError(ValueError):
    """Raised when a cycle configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class C3SModel:
    """One C3S forecasting system, as requested from the CDS."""

    centre: str
    system: str
    label: str
    max_lead_days: int


@dataclass(frozen=True)
class CycleConfig:
    """Validated configuration of one forecast cycle."""

    path: Path
    raw: dict = field(repr=False)
    sha256: dict = field(repr=False)

    # ---- cycle ------------------------------------------------------------
    @property
    def init_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.raw["cycle"]["init_date"])

    @property
    def cycle_id(self) -> str:
        """Short identifier used in paths, e.g. ``202609``."""
        return self.init_date.strftime("%Y%m")

    @property
    def name(self) -> str:
        return self.raw["cycle"].get("name", self.cycle_id)

    # ---- paths ------------------------------------------------------------
    def path_of(self, key: str) -> Path:
        return Path(self.raw["paths"][key])

    @property
    def data_root(self) -> Path:
        return self.path_of("data_root")

    @property
    def output_root(self) -> Path:
        return self.path_of("output_root")

    @property
    def archive_root(self) -> Path:
        return self.path_of("archive_root")

    # ---- scientific choices ------------------------------------------------
    @property
    def scales(self) -> list[str]:
        return list(self.raw["scales"])

    @property
    def variables(self) -> list[str]:
        return list(self.raw["variables"])

    def reference_period(self, key: str) -> tuple[int, int]:
        start, end = self.raw["reference_periods"][key]
        return int(start), int(end)

    @property
    def cv_scheme(self) -> str:
        return self.raw["cross_validation"]["scheme"]

    # ---- systems ----------------------------------------------------------
    @property
    def c3s_models(self) -> dict[str, C3SModel]:
        models = self.raw["systems"]["c3s"]["models"]
        return {
            centre: C3SModel(centre=centre, system=str(spec["system"]),
                             label=spec["label"], max_lead_days=int(spec["max_lead_days"]))
            for centre, spec in models.items()
        }

    @property
    def c3s_hindcast_years(self) -> list[int]:
        y0, y1 = self.raw["systems"]["c3s"]["hindcast_years"]
        return list(range(int(y0), int(y1) + 1))

    @property
    def c3s_download_area(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.raw["systems"]["c3s"]["download_area"])

    # ---- included files ---------------------------------------------------
    @property
    def domains(self) -> dict:
        return self.raw["domains"]

    @property
    def thresholds(self) -> dict:
        return self.raw["thresholds"]

    @property
    def agro_calendars(self) -> dict:
        return self.raw["agro_calendars"]

    def raw_dir(self, system: str) -> Path:
        """Immutable raw-data directory of a system for this cycle."""
        return self.data_root / "raw" / system / self.cycle_id


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"Fichier de configuration introuvable : {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} ne contient pas un dictionnaire YAML.")
    return data


def _require(data: dict, dotted: str, where: Path):
    node = data
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"Clé obligatoire « {dotted} » absente de {where}.")
        node = node[key]
    return node


def _validate(raw: dict, path: Path) -> None:
    for key in ("cycle.init_date", "paths.data_root", "paths.output_root",
                "paths.archive_root", "scales", "variables",
                "reference_periods.c3s_fit", "reference_periods.obs_normal",
                "cross_validation.scheme", "systems.c3s.models",
                "systems.c3s.hindcast_years", "systems.c3s.download_area"):
        _require(raw, key, path)

    init = pd.Timestamp(raw["cycle"]["init_date"])
    if init.day != 1:
        raise ConfigError(f"init_date doit être le 1er du mois (reçu {init.date()}).")

    bad = sorted(set(raw["scales"]) - set(VALID_SCALES))
    if bad:
        raise ConfigError(f"Échelles inconnues {bad} ; valeurs admises : {VALID_SCALES}.")
    bad = sorted(set(raw["variables"]) - set(VALID_VARIABLES))
    if bad:
        raise ConfigError(f"Variables inconnues {bad} ; valeurs admises : {VALID_VARIABLES}.")

    for key in ("c3s_fit", "obs_normal"):
        y0, y1 = raw["reference_periods"][key]
        if int(y0) > int(y1):
            raise ConfigError(f"reference_periods.{key} : début {y0} > fin {y1}.")

    if raw["cross_validation"]["scheme"] != "loyo":
        raise ConfigError("Seul le schéma de validation croisée « loyo » est implémenté.")

    for centre, spec in raw["systems"]["c3s"]["models"].items():
        for k in ("system", "label", "max_lead_days"):
            if k not in spec:
                raise ConfigError(f"systems.c3s.models.{centre} : clé « {k} » manquante.")

    lon0, lon1, lat0, lat1 = raw["systems"]["c3s"]["download_area"]
    if not (lon0 < lon1 and lat0 < lat1):
        raise ConfigError("download_area doit être [lon_min, lon_max, lat_min, lat_max].")


def load_cycle(path: str | Path) -> CycleConfig:
    """
    Read, merge and validate a cycle configuration.

    The ``includes`` section names shared YAML files (relative to the cycle file);
    each is merged under its key (``domains``, ``thresholds``, ``agro_calendars``).
    The SHA-256 of every file read is kept for the provenance manifest.
    """
    path = Path(path).resolve()
    raw = _read_yaml(path)
    hashes = {str(path): _sha256(path)}

    for key, rel in (raw.get("includes") or {}).items():
        inc_path = (path.parent / rel).resolve()
        raw[key] = _read_yaml(inc_path)
        hashes[str(inc_path)] = _sha256(inc_path)

    _validate(raw, path)
    return CycleConfig(path=path, raw=raw, sha256=hashes)


def config_snapshot(cfg: CycleConfig) -> dict:
    """A JSON-serialisable deep copy of the merged configuration."""
    return copy.deepcopy(cfg.raw)
