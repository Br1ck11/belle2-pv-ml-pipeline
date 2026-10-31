"""Independently verify one dataset.h5 window against the raw Parquet archive.

    python scripts/crosscheck_dataset_windows.py --config <toml>

This check deliberately does NOT use dataset.read_pv or dataset.time_grid:
files are read with pyarrow directly and previous-value sampling uses
pandas.merge_asof. Only the build's own record is shared: source file list
(path, size, mtime), PV identities, max ages, split interval, thresholds and
normalization statistics from build_metadata_json / normalization_json.

TOML tables (see examples/crosscheck.toml):
    [check]    name (-> Project/PVPlots/<name>/), optional project_root=".", dpi=200
    [dataset]  path, split, index (row in that split)
    [sources]  optional path_prefix_map = [["<recorded prefix>", "<current prefix>"], ...]
               for archives that moved since the build; sizes must still match

Checks: (1) the recorded source files exist with the recorded byte size;
(2) every model-channel value equals the independent reference (raw builds)
or (reference - offset) / scale computed in float64 (normalized builds), with
log10 applied to the reference first when the build has value_transform log10,
compared exactly; (3) all model channels and both currents are valid
throughout; (4) windows labelled only non_abort have both currents strictly
above their thresholds. The folder gets comparison.json and crosscheck.png.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import (PlotFolder, check_keys, code_sha256, file_fingerprint, folder_name, load_toml,
                     positive_int, resolve_path, resolve_root, utc, versions)
from .style import science_style, PV_COLORS, tidy, utc_time_axis


def resolve_crosscheck_config(config_path):
    path, data, raw = load_toml(config_path)
    check_keys(raw, "top level", required=("check", "dataset"), optional=("sources",))
    check = check_keys(raw["check"], "check", required=("name",), optional=("project_root", "dpi"))
    dataset = check_keys(raw["dataset"], "dataset", required=("path", "split", "index"))
    sources = check_keys(raw.get("sources", {}), "sources", optional=("path_prefix_map",))
    prefix_map = sources.get("path_prefix_map", [])
    if not isinstance(prefix_map, list) or any(
            not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(p, str) and p for p in pair)
            for pair in prefix_map):
        raise ValueError("sources.path_prefix_map must be a list of [recorded_prefix, current_prefix] pairs")
    if dataset["split"] not in ("train", "val", "test"):
        raise ValueError("dataset.split must be 'train', 'val' or 'test'")
    if type(dataset["index"]) is not int or dataset["index"] < 0:
        raise ValueError("dataset.index must be a nonnegative integer")
    root = resolve_root(path, check.get("project_root", "."))
    return path, data, {
        "project_root": str(root), "name": folder_name(check["name"], "check.name"),
        "dpi": positive_int(check.get("dpi", 200), "check.dpi"),
        "dataset": {"path": str(resolve_path(root, dataset["path"], "dataset.path")),
                    "split": dataset["split"], "index": dataset["index"]},
        "path_prefix_map": prefix_map,
    }


def _mapped(path, prefix_map):
    for old, new in prefix_map:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def independent_reference(expected_pv, files, interval, grid_ns, max_age_ns):
    """pyarrow read + merge_asof previous-value sampling on ``grid_ns``.

    Sources are restricted to the split's half-open interval. At equal
    timestamps the last measurement in recorded file order wins (stable
    sort). A latest non-finite value or an age above max_age is invalid.
    """
    times, values = [], []
    for path in files:
        table = pq.read_table(path, columns=["pv", "t0", "times", "values"])
        if table.num_rows != 1 or table["pv"][0].as_py() != expected_pv:
            raise ValueError(f"{path}: expected one row of {expected_pv!r}")
        t0_ns = int(table["t0"][0].as_py()) * 1_000_000
        relative = np.asarray(table["times"][0].as_py(), dtype=np.float64)
        times.append(t0_ns + np.round(relative * 1e9).astype(np.int64))
        values.append(np.asarray(table["values"][0].as_py(), dtype=np.float64))
    source = pd.DataFrame({"t": np.concatenate(times), "value": np.concatenate(values)})
    source = source[(source["t"] >= interval[0]) & (source["t"] < interval[1])]
    source = source.sort_values("t", kind="mergesort").reset_index(drop=True)
    # Carry a row number, not the source time: merge_asof turns unmatched
    # rows into float64 NaN, and float64 cannot hold ~1.7e18 ns exactly.
    source["row"] = np.arange(len(source))
    grid_ns = np.asarray(grid_ns, dtype=np.int64)
    merged = pd.merge_asof(pd.DataFrame({"t": grid_ns}), source, on="t",
                           direction="backward", allow_exact_matches=True)
    matched = merged["row"].notna().to_numpy()
    rows = merged["row"].fillna(0).to_numpy().astype(np.int64)
    source_times = source["t"].to_numpy(dtype=np.int64)
    source_values = source["value"].to_numpy(dtype=np.float64)
    valid = np.zeros(grid_ns.size, dtype=bool)
    if len(source):
        age = grid_ns - source_times[rows]
        valid = matched & (age <= max_age_ns) & np.isfinite(source_values[rows])
    return np.where(valid, source_values[rows] if len(source) else np.nan, np.nan), valid


@science_style()
def crosscheck_window(config_path, *, progress=print):
    """Write comparison.json and crosscheck.png; return (folder, passed)."""
    progress = progress or (lambda message: None)
    path, data, settings = resolve_crosscheck_config(config_path)
    split, index, dataset_path = (settings["dataset"][key] for key in ("split", "index", "path"))
    with h5py.File(dataset_path, "r") as h5:
        if not bool(h5.attrs.get("complete", False)):
            raise ValueError(f"Dataset is incomplete: {dataset_path}")
        group = h5[split]
        if index >= group["values"].shape[0]:
            raise IndexError(f"dataset.index {index} outside the {group['values'].shape[0]} {split} windows")
        stored = group["values"][index].astype(np.float64)
        start_ns, end_ns = int(group["window_start_ns"][index]), int(group["window_end_ns"][index])
        labels = json.loads(group["labels_json"].asstr()[index])
        step_ns, window_size = int(h5.attrs["step_ns"]), int(h5.attrs["window_size"])
        values_state = str(h5.attrs["values_state"])
        value_transform = str(h5.attrs.get("value_transform", "none"))   # absent in older builds
        normalization = json.loads(h5["normalization_json"].asstr()[()])
        build = json.loads(h5["build_metadata_json"].asstr()[()])
        dataset_id = str(h5.attrs["dataset_id"])

    resolved = build["resolved_config"]
    model_ids = resolved["dataset"]["pvs"]
    required = resolved["dataset"]["required_pvs"]
    definitions = {pv["id"]: pv for pv in resolved["pvs"]}
    interval = build["provenance"]["split_intervals_ns"][split]
    grid = start_ns + np.arange(window_size, dtype=np.int64) * step_ns
    checks = {"window_span_matches_grid": end_ns == int(grid[-1])}

    file_report, references = {}, {}
    for pv_id in required:
        recorded = build["provenance"]["sources"][pv_id]["files"]
        files = []
        for entry in recorded:
            current = _mapped(entry["path"], settings["path_prefix_map"])
            if not Path(current).is_file():
                raise FileNotFoundError(f"Recorded source {entry['path']} not found as {current}; "
                                        "add [sources].path_prefix_map if the archive moved")
            fingerprint = file_fingerprint(current)
            if fingerprint["size_bytes"] != entry["size_bytes"]:
                raise ValueError(f"{current}: size {fingerprint['size_bytes']} != recorded {entry['size_bytes']}")
            files.append(current)
        file_report[pv_id] = {"files": len(files),
                              "mtime_changed": sum(Path(f).stat().st_mtime_ns != e["mtime_ns"]
                                                   for f, e in zip(files, recorded))}
        definition = definitions[pv_id]
        references[pv_id] = independent_reference(definition["expected_pv"], files, interval, grid,
                                                  int(definition["max_age_ns"]))
        progress(f"REFERENCE {definition['expected_pv']}: {int(references[pv_id][1].sum())}/{window_size} valid")

    channel_results = []
    for c, pv_id in enumerate(model_ids):
        reference, valid = references[pv_id]
        expected = reference.copy()
        if value_transform == "log10":
            with np.errstate(divide="ignore", invalid="ignore"):
                expected = np.log10(expected)      # same float64 log10 the builder applies
        if values_state == "normalized":
            expected -= normalization["offset"][c]
            expected /= normalization["scale"][c]
        difference = np.abs(stored[c] - expected)
        channel_results.append({
            "id": pv_id, "expected_pv": definitions[pv_id]["expected_pv"], "all_valid": bool(valid.all()),
            "exact_equal": bool(valid.all() and np.array_equal(stored[c], expected)),
            "max_abs_difference": float(np.nanmax(difference)) if valid.any() else None,
        })
    checks["model_channels_valid"] = all(r["all_valid"] for r in channel_results)
    checks["model_channels_exact"] = all(r["exact_equal"] for r in channel_results)
    current_ids = {"LER": resolved["dataset"]["ler_pv"], "HER": resolved["dataset"]["her_pv"]}
    thresholds = {"LER": resolved["dataset"]["ler_threshold"], "HER": resolved["dataset"]["her_threshold"]}
    checks["currents_valid"] = all(bool(references[i][1].all()) for i in current_ids.values())
    if labels == ["non_abort"]:
        checks["currents_above_thresholds"] = all(
            bool(np.all(references[current_ids[ring]][0] > thresholds[ring])) for ring in ("LER", "HER"))
    passed = all(checks.values())

    folder = PlotFolder(settings["project_root"], settings["name"], data)
    rows = [*model_ids, *[i for i in current_ids.values() if i not in model_ids]]
    figure = Figure(figsize=(12, 2.2 + 2.0 * len(rows)), layout="constrained")
    axes = figure.subplots(len(rows), 1, sharex=True, squeeze=False)[:, 0]
    t = grid.astype("datetime64[ns]")
    for k, (ax, pv_id) in enumerate(zip(axes, rows)):
        reference = references[pv_id][0]
        name = definitions[pv_id]["expected_pv"]
        if pv_id in model_ids:
            c = model_ids.index(pv_id)
            shown = stored[c] * normalization["scale"][c] + normalization["offset"][c] \
                if values_state == "normalized" else stored[c].copy()
            if value_transform == "log10":
                shown = 10.0 ** shown
            ax.plot(t, shown, color=PV_COLORS[k % len(PV_COLORS)], linewidth=2.2, alpha=0.6, label="dataset.h5 (raw units)")
        ax.plot(t, reference, color="#111827", linestyle="None", marker=".", markersize=4, label="independent archive read")
        for ring, current in current_ids.items():
            if pv_id == current:
                ax.axhline(thresholds[ring], color="#C05A22", linestyle="--", linewidth=1, label=f"{ring} threshold")
        ax.set_ylabel(name, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="best")
        tidy(ax)
    utc_time_axis(axes[-1], utc(start_ns), utc(end_ns))
    axes[0].set_title(f"{'PASS' if passed else 'FAIL'} · {split} window {index} · labels {', '.join(labels)}", fontsize=11)
    figure.savefig(folder.path / "crosscheck.png", dpi=settings["dpi"], facecolor="white")
    comparison = {
        "passed": passed, "checks": checks, "channels": channel_results, "labels": labels,
        "window_start_utc": utc(start_ns).isoformat(), "window_end_utc": utc(end_ns).isoformat(),
        "split_interval_ns": interval, "source_files": file_report, "values_state": values_state, "value_transform": value_transform,
    }
    folder.write_json("comparison.json", comparison)
    folder.write_json("metadata.json", {
        "plot": "crosscheck", "config": {"original_path": str(path), "copy": "config.toml"}, "resolved": settings,
        "dataset": {**file_fingerprint(dataset_path), "dataset_id": dataset_id}, "versions": versions(),
        "code_sha256": code_sha256(), "outputs": ["crosscheck.png", "comparison.json"],
    })
    progress(f"{'PASS' if passed else 'FAIL'}: {checks} -> {folder.path}")
    return folder.path, passed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify a dataset.h5 window against an independent archive read.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        _, passed = crosscheck_window(args.config)
    except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
        print(f"Crosscheck failed: {error}", file=sys.stderr)
        return 1
    return 0 if passed else 2
