"""Assemble the independent PV pipeline into a reproducible dataset build."""

from collections import Counter
import hashlib
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pandas as pd
import pyarrow

from . import __version__
from .parse_config import parse_cli, resolve_config
from .read_pv import read_pv_files
from .time_grid import make_time_grid, sample_previous
from .split import split_time_grid
from .window_data import window_segment
from .filter_pv import complete_window_mask, WindowFilter, training_coverage_mask
from .normalize import NormalizeWindows, log10_values
from .save_dataset import DatasetWriter


def _sample_to_disk(storage, config, grids, intervals, progress):
    """Load one PV at a time; sample strictly inside each split's bounds.

    Raw files for ONE PV are concatenated in memory by read_pv_files.
    Sampled channels are held on disk, so RAM does not grow with C times
    the whole requested interval. No windows are materialized here.
    """
    settings = config["dataset"]
    required_ids = settings["required_pvs"]
    by_id = {pv["id"]: pv for pv in config["pvs"]}
    for split, grid in grids.items():
        group = storage.create_group(split)
        for name, dtype in (("values", np.float64), ("valid", bool)):
            group.create_dataset(name, shape=(len(required_ids), grid.size), dtype=dtype, chunks=True)
    sources = {}
    for channel, pv_id in enumerate(required_ids):
        pv = by_id[pv_id]
        progress(f"Reading and sampling {pv_id} ({channel + 1}/{len(required_ids)})")
        file_info = []
        for filename in pv["filepaths"]:
            stat = Path(filename).stat()
            file_info.append({"path": filename, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        times, values = read_pv_files(pv["filepaths"], pv["expected_pv"])
        sources[pv_id] = {"expected_pv": pv["expected_pv"], "files": file_info,
                          "source_sample_count": int(times.size), "splits": {}}
        if settings["log_transform"] and pv_id in settings["pvs"]:
            # Transform raw measurements BEFORE sampling: statistics and
            # normalization then describe log10 values. Values <= 0 become
            # NaN and are handled like any other invalid measurement.
            values, nonpositive = log10_values(values)
            sources[pv_id]["value_transform"] = "log10"
            sources[pv_id]["nonpositive_values_invalidated"] = nonpositive
            if nonpositive:
                progress(f"  {pv_id}: {nonpositive} non-positive measurements invalid under log10")
        for split, grid in grids.items():
            start, end = intervals[split]
            first, stop = np.searchsorted(times, [start, end], side="left")
            sampled, valid = sample_previous(times[first:stop], values[first:stop], grid, pv["max_age_ns"])
            storage[split]["values"][channel] = sampled
            storage[split]["valid"][channel] = valid
            sources[pv_id]["splits"][split] = {
                "source_sample_count": int(stop - first), "valid_grid_points": int(valid.sum()),
            }
            del sampled, valid
        del times, values
    return sources


def _write_split(writer, sampled, split, grid, settings, selector, normalizer, progress):
    """Window, filter, label and append batches; then fit unique train points."""
    size, stride, batch_size = (settings[key] for key in ("window_size_samples", "stride_samples", "batch_size"))
    n_model = len(settings["pvs"])
    required_ids = settings["required_pvs"]
    ler_index = required_ids.index(settings["ler_pv"])
    her_index = required_ids.index(settings["her_pv"])
    n_candidates = 1 + (grid.size - size) // stride
    # Persist across batches: overlapping windows never count a point twice.
    coverage = np.zeros(grid.size, dtype=bool) if split == "train" else None
    counts = {"grid_points": int(grid.size), "candidate_windows": int(n_candidates),
              "rejected_incomplete": 0, "rejected_selection": 0, "retained_windows": 0,
              "unused_trailing_grid_points": int(grid.size - ((n_candidates - 1) * stride + size))}
    label_counts = Counter()
    for first_window in range(0, n_candidates, batch_size):
        stop_window = min(first_window + batch_size, n_candidates)
        start_indices = np.arange(first_window, stop_window, dtype=np.int64) * stride
        span_start, span_end = int(start_indices[0]), int(start_indices[-1]) + size
        # Read only the span supporting this batch, then use the existing 1D
        # window function per channel. Stack in the fixed required-PV order.
        values = sampled["values"][:, span_start:span_end]
        valid = sampled["valid"][:, span_start:span_end]
        windows = np.stack([window_segment(row, size, stride) for row in values], axis=1)
        window_valid = np.stack([window_segment(row, size, stride) for row in valid], axis=1)
        starts = grid[start_indices]
        ends = grid[start_indices + size - 1]
        complete = complete_window_mask(window_valid)
        keep = selector.filter_pv_windows(complete, windows[:, ler_index], windows[:, her_index], starts, ends)
        counts["rejected_incomplete"] += int((~complete).sum())
        counts["rejected_selection"] += int((complete & ~keep).sum())
        counts["retained_windows"] += int(keep.sum())
        labels, associations = selector.label_windows(starts[keep], ends[keep])
        for row in labels:
            label_counts.update(row)
        # Model channels are the prefix; filter-only currents stay out of
        # both the output tensor and the normalization statistics.
        writer.append(split, windows[keep, :n_model], starts[keep], ends[keep],
                      labels=labels, event_associations=associations)
        if coverage is not None:
            local = training_coverage_mask(start_indices[keep] - span_start, size, span_end - span_start)
            coverage[span_start:span_end] |= local
        if first_window == 0 or stop_window == n_candidates or (first_window // batch_size) % 100 == 0:
            progress(f"{split}: checked {stop_window}/{n_candidates}, retained {counts['retained_windows']}")

    if counts["retained_windows"] == 0:
        raise ValueError(f"{split} has no accepted windows: {counts}. Check coverage, currents and abort buffers.")
    counts["label_counts"] = dict(label_counts)
    if coverage is not None:
        counts["unique_covered_grid_points"] = int(coverage.sum())
        if normalizer.norm_mode != "none":
            # Disjoint grid slices, not flattened overlapping windows.
            points_per_batch = batch_size * size
            for first in range(0, grid.size, points_per_batch):
                stop = min(first + points_per_batch, grid.size)
                selected = coverage[first:stop]
                if selected.any():
                    normalizer.calculate_statistics(sampled["values"][:n_model, first:stop][:, selected])
        normalizer.finalize()
    return counts


def build_dataset(config_path, *, progress=print):
    """Build one new output directory from TOML; return dataset.h5's Path.

    Train/val fits only the union of retained training timestamps. Normalized
    test builds reuse a completed training dataset's statistics, checked by
    DatasetWriter for channel identity and interval separation. Existing
    datasets are never overwritten. Errors leave complete=False on partial
    output. Pass progress=None for a quiet Python call.
    """
    progress = progress or (lambda message: None)
    config_path = Path(config_path).expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config = resolve_config(config_path)
    if config_path.read_bytes() != config_bytes:
        raise ValueError("Configuration changed while being read; rerun with a fixed TOML")
    settings = config["dataset"]
    grid = make_time_grid(settings["start_ns"], settings["end_ns"], settings["step_ns"])
    grids = split_time_grid(grid, settings["mode"], settings.get("train_fraction"))
    if settings["mode"] == "train_val":
        boundary = int(grids["val"][0])
        intervals = {"train": [settings["start_ns"], boundary], "val": [boundary, settings["end_ns"]]}
    else:
        intervals = {"test": [settings["start_ns"], settings["end_ns"]]}
    for split, split_grid in grids.items():
        if split_grid.size < settings["window_size_samples"]:
            raise ValueError(f"{split} has {split_grid.size} grid points, fewer than window_size_samples={settings['window_size_samples']}")
    selector = WindowFilter(
        settings["mode"], settings["ler_threshold"], settings["her_threshold"], settings["aborts_filepath"],
        **{key: settings[key] for key in ("exclude_before_ns", "exclude_after_ns", "pre_abort_ns", "post_abort_ns")},
    )
    events = selector.load_events()
    normalizer = None
    if settings["mode"] == "train_val" or settings["norm_mode"] == "none":
        normalizer = NormalizeWindows(settings["norm_mode"], n_channels=len(settings["pvs"]))

    # TODO: Supervised containment/forecasting modes need distinct target
    # definitions; current test labels only describe each input window.
    # TODO: For supervised train/val, exclude complete event episodes whose
    # associated input spans cross the chronological boundary.
    provenance = {
        "split_intervals_ns": intervals,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "versions": {"belle2_dataset": __version__, "python": platform.python_version(),
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "pyarrow": pyarrow.__version__, "h5py": h5py.__version__},
        "implementation_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "abort_catalog": {"path": settings["aborts_filepath"], "sha256": selector.catalog_sha256,
                          "naive_timezone": "Asia/Tokyo", "missing_category": "Unspecified",
                          "events": list(events)},
        "policies": {
            "sampling": "latest source <= grid time within max_age; latest nonfinite value stays invalid",
            "duplicates": "stable file-list order; last measurement at equal timestamp wins",
            "splitting": "floor(train_fraction * grid_size) before sampling; sources restricted to each split",
            "windowing": "complete windows only; trailing partial windows discarded; endpoints inclusive",
            "completeness": "all model channels AND both currents must be valid throughout",
            "normal": "both currents strictly above thresholds and no inclusive abort-exclusion overlap",
            "test": "normal OR event-related; event-related bypasses thresholds but never completeness",
            "labels": "non_abort; category x if containing; pre_abort_x or post_abort_x by window end; all events retained",
            "statistics": "unique grid timestamps covered by retained TRAIN windows; ddof=0",
            "value_transform": ("log10 of model-channel measurements before sampling, statistics and "
                                "normalization; values <= 0 invalid; currents untransformed"
                                if settings["log_transform"] else "none"),
            "source_fingerprints": "PV path, byte size and mtime_ns; raw PV contents are not hashed",
        },
    }
    with DatasetWriter(settings["output_dir"], config_path, mode=settings["mode"],
                       channel_ids=settings["pvs"], window_size=settings["window_size_samples"],
                       step_ns=settings["step_ns"],
                       value_transform="log10" if settings["log_transform"] else "none") as writer:
        if (Path(settings["output_dir"]) / "config.toml").read_bytes() != config_bytes:
            raise ValueError("Configuration changed before it could be copied")
        # Temporary sampled arrays disappear on success or failure.
        with TemporaryDirectory(prefix=".sampling-", dir=settings["output_dir"]) as temporary:
            with h5py.File(Path(temporary) / "sampled.h5", "w") as sampled:
                provenance["sources"] = _sample_to_disk(sampled, config, grids, intervals, progress)
                provenance["selection"] = {}
                for split, split_grid in grids.items():
                    provenance["selection"][split] = _write_split(
                        writer, sampled[split], split, split_grid, settings, selector, normalizer, progress,
                    )
        progress("Applying frozen normalization and saving metadata")
        writer.finalize(normalizer, resolved_config=config, provenance=provenance,
                        statistics_filepath=settings.get("statistics_filepath"), batch_size=settings["batch_size"])
        result = writer.dataset_path
    progress(f"DATASET: {result}")
    return result


def main(argv=None):
    """CLI entry point; TOML is the sole source of dataset settings."""
    args = parse_cli(argv)
    try:
        build_dataset(args.config)
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        print(f"Dataset build failed: {error}", file=sys.stderr)
        return 1
    return 0
