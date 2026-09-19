"""
Provenance, logging and archiving of every run (workflow step E0).

The Draft Framework (§3.1.10) requires each issued forecast to carry enough
metadata to identify its data, versions, periods, processing and code. Every
script of the OSF chain therefore runs inside a :class:`RunContext`::

    cfg = load_cycle("config/cycle_202609.yaml")
    with RunContext(cfg, step="download") as ctx:
        ctx.log.info("...")
        ctx.record_input(path, role="c3s_forecast")
        ...
        ctx.record_output(out_path, role="decade_totals")

On exit the context writes ``manifest.json`` next to ``run.log`` in
``<output_root>/runs/<run_id>/``. It records the status (``success`` or
``failed`` with the error) even when the run crashes.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from eccas_s2s import __version__
from eccas_s2s.settings import CycleConfig, config_snapshot

#: libraries whose versions are recorded in every manifest.
TRACKED_PACKAGES = ("numpy", "scipy", "pandas", "xarray", "dask", "cfgrib", "eccodes",
                    "scikit-learn", "xesmf", "xskillscore", "netCDF4", "cdsapi", "pyyaml")

#: files larger than this are fingerprinted from their first/last MiB only.
FULL_HASH_LIMIT = 64 * 1024 * 1024


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def git_state(repo_dir: Path) -> dict:
    """Commit, branch and dirty flag of the code repository (empty if not a repo)."""
    def _git(*args):
        return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def package_versions(names=TRACKED_PACKAGES) -> dict:
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def file_fingerprint(path: Path) -> dict:
    """Size, modification time and SHA-256 of a file (partial for very large files)."""
    path = Path(path)
    stat = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as fh:
        if stat.st_size <= FULL_HASH_LIMIT:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
            mode = "full"
        else:
            h.update(fh.read(1 << 20))
            fh.seek(-(1 << 20), 2)
            h.update(fh.read(1 << 20))
            mode = "head+tail 1MiB"
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_utc": _dt.datetime.fromtimestamp(stat.st_mtime, _dt.timezone.utc).isoformat(),
        "sha256": h.hexdigest(),
        "sha256_mode": mode,
    }


class RunContext:
    """Logging + manifest for one execution of one step of the chain."""

    def __init__(self, cfg: CycleConfig, step: str, repo_dir: Path | None = None):
        self.cfg = cfg
        self.step = step
        self.started = _utcnow()
        self.run_id = f"{cfg.cycle_id}_{step}_{self.started:%Y%m%dT%H%M%SZ}"
        self.run_dir = cfg.output_root / "runs" / self.run_id
        self.repo_dir = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[1]
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []
        self.parameters: dict = {}
        self.warnings: list[str] = []
        self.status = "running"
        self.error: str | None = None
        self._t0 = time.monotonic()
        self.log = self._make_logger()

    # ------------------------------------------------------------------ setup
    def _make_logger(self) -> logging.Logger:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"eccas_s2s.run.{self.run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        for handler in (logging.FileHandler(self.run_dir / "run.log", encoding="utf-8"),
                        logging.StreamHandler(sys.stdout)):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
        return logger

    # --------------------------------------------------------------- records
    def record_input(self, path, role: str, **extra) -> None:
        entry = {"role": role, **file_fingerprint(Path(path)), **extra}
        self.inputs.append(entry)
        self.log.info("entrée [%s] %s", role, path)

    def record_output(self, path, role: str, **extra) -> None:
        entry = {"role": role, **file_fingerprint(Path(path)), **extra}
        self.outputs.append(entry)
        self.log.info("sortie [%s] %s", role, path)

    def record_parameter(self, key: str, value) -> None:
        self.parameters[key] = value

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self.log.warning(message)

    # -------------------------------------------------------------- manifest
    def manifest(self) -> dict:
        return {
            "run_id": self.run_id,
            "step": self.step,
            "status": self.status,
            "error": self.error,
            "started_utc": self.started.isoformat(),
            "duration_s": round(time.monotonic() - self._t0, 2),
            "cycle": {
                "id": self.cfg.cycle_id,
                "init_date": str(self.cfg.init_date.date()),
                "config_file": str(self.cfg.path),
                "config_sha256": self.cfg.sha256,
                "config": config_snapshot(self.cfg),
            },
            "code": {"eccas_s2s_version": __version__, "git": git_state(self.repo_dir)},
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "packages": package_versions(),
            },
            "parameters": self.parameters,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "warnings": self.warnings,
        }

    def write_manifest(self) -> Path:
        path = self.run_dir / "manifest.json"
        path.write_text(json.dumps(self.manifest(), indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
        return path

    def netcdf_attrs(self) -> dict:
        """Global attributes linking a NetCDF output back to this run."""
        git = git_state(self.repo_dir)
        return {
            "institution": "CAPC-AC — Centre d'Application et de Prévisions Climatologiques d'Afrique Centrale",
            "source": f"eccas-s2s {__version__} (OSF chain), step {self.step}",
            "run_id": self.run_id,
            "cycle_init_date": str(self.cfg.init_date.date()),
            "config_file": self.cfg.path.name,
            "code_commit": git.get("commit", "unknown") + ("-dirty" if git.get("dirty") else ""),
            "history": f"{_utcnow():%Y-%m-%dT%H:%M:%SZ} created by eccas-s2s run {self.run_id}",
            "Conventions": "CF-1.8",
        }

    # ------------------------------------------------------ context manager
    def __enter__(self) -> "RunContext":
        self.log.info("=== début %s | cycle %s | run %s ===", self.step, self.cfg.cycle_id, self.run_id)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            self.status = "success"
        else:
            self.status = "failed"
            self.error = f"{exc_type.__name__}: {exc}"
            self.log.error("échec : %s", self.error)
        path = self.write_manifest()
        self.log.info("=== fin %s | statut %s | manifeste %s ===", self.step, self.status, path)
        for handler in list(self.log.handlers):
            handler.close()
            self.log.removeHandler(handler)
        return False  # never swallow the exception


def archive_run(ctx: RunContext, files=None) -> Path:
    """
    Copy a finished run (manifest, log and selected output files) to the archive:
    ``<archive_root>/<YYYY>/<MM>/<run_id>/``.

    The archive is what real-time verification (step E9) reads, so it is written
    once and never modified.
    """
    init = ctx.cfg.init_date
    dest = ctx.cfg.archive_root / f"{init:%Y}" / f"{init:%m}" / ctx.run_id
    if dest.exists():
        raise FileExistsError(f"Archive déjà existante, non écrasée : {dest}")
    dest.mkdir(parents=True)
    for name in ("manifest.json", "run.log"):
        src = ctx.run_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
    for f in files or []:
        shutil.copy2(f, dest / Path(f).name)
    return dest
