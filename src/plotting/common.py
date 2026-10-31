"""Shared TOML checks and the reproducible PVPlots/<name>/ output folder.

Every plot is produced from one TOML. The folder receives an exact copy of
that TOML, the PNG(s) and metadata.json (resolved settings, input files with
size/mtime, plotted events, package versions and code hashes), so the plot
can be reconstructed from the folder alone.
"""

import hashlib
import json
import math
import os
import platform
import re
from pathlib import Path
import tomllib

import h5py
import matplotlib
import numpy as np
import pandas as pd
import pyarrow


PLOTS_ROOT = "PVPlots"
_FOLDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def load_toml(path) -> tuple[Path, bytes, dict]:
    """Return (absolute path, exact bytes, parsed contents)."""
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    return path, data, tomllib.loads(data.decode("utf-8"))


def check_keys(table, section, *, required=(), optional=()):
    """Reject a non-table, unknown keys and missing required keys."""
    if not isinstance(table, dict):
        raise ValueError(f"[{section}] must be a TOML table")
    unknown = set(table) - set(required) - set(optional)
    if unknown:
        raise ValueError(f"Unknown [{section}] settings {sorted(unknown)}; allowed: {sorted({*required, *optional})}")
    missing = set(required) - set(table)
    if missing:
        raise ValueError(f"Missing [{section}] settings: {sorted(missing)}")
    return table


def text(value, name):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def folder_name(value, name):
    if not isinstance(value, str) or not _FOLDER_NAME.fullmatch(value):
        raise ValueError(f"{name} = {value!r} must be one folder name of letters, digits, '.', '_' or '-'")
    return value


def utc_ns(value, name) -> int:
    """ISO timestamp with explicit timezone -> UTC Unix nanoseconds."""
    timestamp = pd.Timestamp(text(value, name))
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must include a timezone (Z or an offset), got {value!r}")
    return int(timestamp.tz_convert("UTC").value)


def utc(ns: int) -> pd.Timestamp:
    return pd.Timestamp(int(ns), unit="ns", tz="UTC")


def resolve_root(config_path: Path, project_root) -> Path:
    root = Path(text(project_root, "project_root")).expanduser()
    return (root if root.is_absolute() else config_path.parent / root).resolve()


def resolve_path(root: Path, value, name, *, must_exist=True) -> Path:
    path = Path(text(value, name)).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{name}: path does not exist: {path}")
    return path


def file_fingerprint(path) -> dict:
    stat = Path(path).stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def code_sha256() -> dict:
    """Hashes of the plotting package and the src/dataset files it relies on."""
    here = Path(__file__).resolve().parent
    hashes = {f"plotting/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob("*.py"))}
    import dataset
    for p in sorted(Path(dataset.__file__).resolve().parent.glob("*.py")):
        hashes[f"dataset/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__, "matplotlib": matplotlib.__version__, "h5py": h5py.__version__}


class PlotFolder:
    """Project/PVPlots/<name>/: created once, never overwritten."""

    def __init__(self, project_root: Path, name: str, config_bytes: bytes) -> None:
        self.path = Path(project_root) / PLOTS_ROOT / folder_name(name, "name")
        if self.path.exists():
            raise FileExistsError(f"Plot folder already exists: {self.path}; choose another name")
        self.path.mkdir(parents=True)
        (self.path / "config.toml").write_bytes(config_bytes)

    def write_json(self, filename: str, payload: dict) -> Path:
        path = self.path / filename
        temporary = path.with_name(f".{filename}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=_json_default) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
        return path


def _json_default(item):
    if isinstance(item, np.generic):
        return item.item()
    if isinstance(item, (Path, pd.Timestamp)):
        return str(item)
    raise TypeError(f"Unsupported metadata type: {type(item).__name__}")


def finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None
