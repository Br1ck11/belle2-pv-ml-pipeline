import tomllib
import argparse
from pathlib import Path
from copy import deepcopy
from decimal import Decimal
import math

import pandas as pd

def load_toml(path):
    """Read a TOML file and return its contents as a dictionary."""
     
    # with block closes the file after opening and loading into config
    with open(path, "rb") as stream: # read file in binary
        config = tomllib.load(stream)
     
    return config
    
    
def parse_cli(argv=None):
    """Select a TOML file; all dataset settings come from that file."""
    # Create object that stores argument defintions
    parser = argparse.ArgumentParser(
        description="Configure a multivariate PV dataset."
    )
    
    parser.add_argument("--config", type=Path, required=True)
    
    return parser.parse_args(argv)


def seconds_to_ns(value, name, *, positive=False):
    """Resolve decimal seconds exactly; reject sub-nanosecond durations."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number of seconds")
    ns = Decimal(str(value)) * 1_000_000_000
    if ns != ns.to_integral_value() or not (0 <= ns <= 2**63 - 1) or (positive and ns == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier} and exactly representable as int64 nanoseconds")
    return int(ns)


def resolve_pv_filepaths(entries, root, *, pv_id) -> list[str]:
    """Expand one PV's file/directory entries into absolute file paths.

    Relative entries are resolved against root. Explicit files retain their
    configured order. Each directory expands in place to its immediate
    *.parquet files, sorted by filename; subdirectories are not searched.
    Discovery does not inspect measurements or select dates. The loader
    subsequently verifies PV identity and sorts actual timestamps.

    Returns a new flat list of resolved path strings. The input list is not
    modified. Missing paths, directories without matching files, invalid
    entries and duplicate resolved paths (including symlink aliases) raise
    an error naming the PV. A directory and an explicitly listed member
    therefore cannot accidentally load the same file twice.
    """
    name = f"PV {pv_id!r} filepaths"
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{name} must be a nonempty list")
    root = Path(root).expanduser().resolve()
    resolved_files = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip() or "\x00" in entry:
            raise ValueError(f"{name}: entries must be nonempty path strings without null characters")
        candidate = Path(entry).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()

        if candidate.is_dir():
            files = sorted(path for path in candidate.glob("*.parquet") if path.is_file())
            if not files:
                raise ValueError(f"{name}: directory contains no immediate .parquet files: {candidate}")
        elif candidate.is_file():
            files = [candidate]
        elif not candidate.exists():
            raise FileNotFoundError(f"{name}: path does not exist: {candidate}")
        else:
            raise ValueError(f"{name}: expected a file or directory: {candidate}")

        for file in files:
            resolved = str(file.resolve())
            if resolved in seen:
                raise ValueError(f"{name} must not repeat the same file: {resolved}")
            seen.add(resolved)
            resolved_files.append(resolved)
    return resolved_files


def resolve_config(config_path):
    """Validate TOML and return a detached configuration with derived fields.

    Paths are relative to dataset.project_root, itself relative to the TOML
    directory. When omitted, the TOML directory is the root. This works for
    an installed package without depending on where its Python files live.
    The original TOML is never rewritten. Required PVs retain model order,
    followed by filtering currents not already selected as model channels.
    Selected PV directories are expanded to explicit file lists so the
    builder and saved resolved configuration use the same discovered files.
    """
    config_path = Path(config_path).expanduser().resolve()
    config = deepcopy(load_toml(config_path))
    if set(config) != {"dataset", "pvs"}:
        raise ValueError("Configuration must contain [dataset] and [[pvs]] tables only")
    settings = config["dataset"]
    if not isinstance(settings, dict):
        raise ValueError("dataset must be a TOML table")
    common = {
        "mode", "start", "end", "pvs", "sample_period_seconds", "window_size_samples",
        "stride_samples", "ler_pv", "her_pv", "ler_threshold", "her_threshold",
        "aborts_filepath", "exclude_before_seconds", "exclude_after_seconds",
        "norm_mode", "batch_size", "output_dir",
    }
    optional = {"project_root", "train_fraction", "pre_abort_seconds", "post_abort_seconds", "statistics_filepath",
                "log_transform"}
    unknown = set(settings) - common - optional
    if unknown:
        raise ValueError(f"Unknown [dataset] settings: {sorted(unknown)}")
    required = common.copy()
    if settings.get("mode") == "train_val":
        required.add("train_fraction")
    elif settings.get("mode") == "test":
        required.update(("pre_abort_seconds", "post_abort_seconds"))
        if settings.get("norm_mode") != "none":
            required.add("statistics_filepath")
    missing = required - settings.keys()
    if missing:
        raise ValueError(f"Missing [dataset] settings: {sorted(missing)}")
    if settings["mode"] not in ("train_val", "test"):
        raise ValueError("dataset.mode must be 'train_val' or 'test'")
    if settings["norm_mode"] not in ("zscore", "minmax", "none"):
        raise ValueError("dataset.norm_mode must be 'zscore', 'minmax', or 'none'")

    def text(value, name):
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"{name} must be a nonempty string without null characters")
        return value

    root = Path(text(settings.get("project_root", "."), "dataset.project_root")).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve()
    settings["project_root"] = str(root)

    def path(value, name, *, must_exist=True):
        resolved = Path(text(value, name)).expanduser()
        if not resolved.is_absolute():
            resolved = root / resolved
        resolved = resolved.resolve()
        if must_exist and not resolved.is_file():
            raise FileNotFoundError(f"{name}: file does not exist: {resolved}")
        return str(resolved)

    model_ids = settings["pvs"]
    if not isinstance(model_ids, list) or not model_ids:
        raise ValueError("dataset.pvs must be a nonempty list of PV ids")
    for index, pv_id in enumerate(model_ids):
        text(pv_id, f"dataset.pvs[{index}]")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("dataset.pvs must not contain duplicate PV ids")
    for key in ("ler_pv", "her_pv"):
        text(settings[key], f"dataset.{key}")
    if settings["ler_pv"] == settings["her_pv"]:
        raise ValueError("dataset.ler_pv and dataset.her_pv must refer to different PV ids")
    required_ids = model_ids.copy()
    for current in (settings["ler_pv"], settings["her_pv"]):
        if current not in required_ids:
            required_ids.append(current)

    definitions = config["pvs"]
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("Provide at least one [[pvs]] definition")
    by_id = {}
    for pv in definitions:
        if not isinstance(pv, dict):
            raise ValueError("Each [[pvs]] entry must be a table")
        pv_id = text(pv.get("id"), "PV id")
        if pv_id in by_id:
            raise ValueError(f"Duplicate PV id: {pv_id!r}")
        by_id[pv_id] = pv
    for pv_id in required_ids:
        if pv_id not in by_id:
            raise ValueError(f"Required PV id {pv_id!r} has no [[pvs]] definition")
        pv = by_id[pv_id]
        if set(pv) != {"id", "expected_pv", "filepaths", "max_age_seconds"}:
            raise ValueError(f"PV {pv_id!r} needs exactly id, expected_pv, filepaths, max_age_seconds")
        text(pv["expected_pv"], f"PV {pv_id!r} expected_pv")
        pv["filepaths"] = resolve_pv_filepaths(pv["filepaths"], root, pv_id=pv_id)
        pv["max_age_ns"] = seconds_to_ns(pv["max_age_seconds"], f"PV {pv_id!r} max_age_seconds")
    if by_id[settings["ler_pv"]]["expected_pv"] == by_id[settings["her_pv"]]["expected_pv"]:
        raise ValueError("LER and HER must have distinct expected_pv identities")

    for key in ("start", "end"):
        value = text(settings[key], f"dataset.{key}")
        try:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp) or timestamp.tzinfo is None:
                raise ValueError("An explicit timezone is required")
            settings[f"{key}_ns"] = int(timestamp.tz_convert("UTC").value)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"dataset.{key} must be an ISO timestamp with an explicit timezone (Z or UTC offset), fitting int64 ns") from error
    span = settings["end_ns"] - settings["start_ns"]
    if not 0 < span <= 2**63 - 1:
        raise ValueError("Requested [start, end) interval must be positive and fit int64 nanoseconds")
    settings["step_ns"] = seconds_to_ns(settings["sample_period_seconds"], "sample_period_seconds", positive=True)
    if settings["start_ns"] + settings["step_ns"] > 2**63 - 1:
        raise ValueError("Grid arithmetic would overflow int64 nanoseconds")
    for key in ("window_size_samples", "stride_samples", "batch_size"):
        if type(settings[key]) is not int or not 0 < settings[key] <= 2**63 - 1:
            raise ValueError(f"dataset.{key} must be a positive int64 integer")
    if (settings["window_size_samples"] - 1) * settings["step_ns"] > 2**63 - 1:
        raise ValueError("Window endpoint span must fit int64 nanoseconds")
    for key in ("exclude_before", "exclude_after", "pre_abort", "post_abort"):
        settings[f"{key}_ns"] = seconds_to_ns(settings.get(f"{key}_seconds", 0), f"dataset.{key}_seconds")
    for key in ("ler_threshold", "her_threshold"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"dataset.{key} must be finite and numeric")
    if settings["mode"] == "train_val":
        value = settings["train_fraction"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
            raise ValueError("dataset.train_fraction must be strictly between zero and one")
    # Optional log10 of the MODEL channels before sampling and statistics.
    # Filter-only currents are never transformed (thresholds stay in mA).
    log_transform = settings.get("log_transform", False)
    if type(log_transform) is not bool:
        raise ValueError("dataset.log_transform must be true or false")
    settings["log_transform"] = log_transform
    settings["aborts_filepath"] = path(settings["aborts_filepath"], "dataset.aborts_filepath")
    if "statistics_filepath" in settings:
        if settings["mode"] != "test" or settings["norm_mode"] == "none":
            raise ValueError("statistics_filepath is only used by normalized test builds")
        settings["statistics_filepath"] = path(settings["statistics_filepath"], "dataset.statistics_filepath")
    settings["output_dir"] = path(settings["output_dir"], "dataset.output_dir", must_exist=False)
    if Path(settings["output_dir"]).exists():
        raise FileExistsError(f"Output directory already exists: {settings['output_dir']}")
    settings["required_pvs"] = required_ids
    return config

    



