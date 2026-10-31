"""Single-file HDF5 storage for the independent dataset implementation."""

from pathlib import Path
import json
import shutil
import uuid

import h5py
import numpy as np

from .normalize import NormalizeWindows


def _json_text(value):
    """Encode metadata without pickle, NaN, or loss of integer timestamps."""
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Unsupported metadata type: {type(item).__name__}")

    return json.dumps(value, default=convert, ensure_ascii=False, allow_nan=False)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer scalar, not a Boolean")
    value = int(value)
    if not np.iinfo(np.int64).min <= value <= np.iinfo(np.int64).max:
        raise ValueError(f"{name} must fit in int64")
    return value


def _restore_normalizer(metadata, n_channels):
    """Validate saved statistics and reconstruct a frozen normalizer.

    Recomputing derived parameters checks the metadata's internal consistency;
    the saved offset and scale are then retained exactly for reproducibility.
    No dataset samples are used to fit statistics here.
    """
    if not isinstance(metadata, dict) or metadata.get("n_channels") != n_channels:
        raise ValueError("Normalization channel count does not match")
    normalizer = NormalizeWindows(metadata.get("norm_mode"), n_channels=n_channels)

    def vector(name):
        array = np.asarray(metadata.get(name))
        if array.shape != (n_channels,) or array.dtype.kind not in "iuf":
            raise ValueError(f"Normalization {name} must be a numeric vector of length C")
        array = array.astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"Normalization {name} must be finite")
        return array

    if normalizer.norm_mode != "none":
        normalizer.count = _integer(metadata.get("count"), "normalization count")
        if normalizer.count <= 0:
            raise ValueError("Normalization needs positive training-point count")
        if normalizer.norm_mode == "minmax":
            normalizer.minimum = vector("minimum")
            normalizer.maximum = vector("maximum")
        else:
            normalizer.mean = vector("mean")
            normalizer.m2 = vector("m2")
            if np.any(normalizer.m2 < 0):
                raise ValueError("Normalization m2 must be nonnegative")
        normalizer.finalize()

    expected = normalizer.to_metadata()
    if set(metadata) != set(expected):
        raise ValueError("Normalization metadata has missing or unknown fields")
    for name, value in expected.items():
        if isinstance(value, list) and name != "scale_fallback_mask":
            if not np.allclose(vector(name), value, rtol=1e-12, atol=0):
                raise ValueError(f"Inconsistent normalization {name}")
        elif metadata[name] != value:
            raise ValueError(f"Inconsistent normalization {name}")
    fallback = metadata["scale_fallback_mask"]
    if not isinstance(fallback, list) or any(type(item) is not bool for item in fallback):
        raise ValueError("scale_fallback_mask must contain Booleans")
    for name in ("offset", "scale"):
        setattr(normalizer, name, vector(name))
    if np.any(normalizer.scale <= 0):
        raise ValueError("Normalization scale must be positive")
    if normalizer.norm_mode != "none":
        normalizer.raw_scale = vector("raw_scale")
    return normalizer


class DatasetWriter:
    """Own one open HDF5 file and the arrays for its dataset splits.

    Append filtered RAW windows, with their timestamps and labels. Once all
    batches are stored and training statistics are frozen, finalize applies
    normalization in bounded batches and saves metadata. Consumers must check
    the root ``complete`` attribute before reading a dataset for training.

    Only dataset.h5 and an unchanged config.toml copy are produced. Closing
    without successful finalization leaves an explicitly incomplete build.
    """

    def __init__(
        self, output_dir, config_path, *, mode, channel_ids, window_size, step_ns, value_transform="none",
    ):
        """Create empty split arrays and copy the input TOML without rewriting it.

        Args:
            output_dir: New directory for dataset.h5 and config.toml. An
                existing directory is rejected to prevent overwriting a build.
            config_path: Original TOML file, copied byte-for-byte. The caller
                is responsible for parsing/validating it and passing matching
                resolved settings here.
            mode: "train_val" creates train and val groups; "test" creates
                only a test group.
            channel_ids: Nonempty, unique ordered list/tuple of model PV ids.
                Filter-only currents are not included unless selected here.
            window_size: Positive integer T, the number of samples per window.
            step_ns: Positive integer spacing of the regular grid in nanoseconds.

        Each split has float64 values of shape (0, C, T) and int64
        window_start_ns/window_end_ns of shape (0,). Only the first axis may
        grow. End timestamps denote the LAST INCLUDED sample, so a window's
        sample k occurs at start_ns + k * step_ns, for 0 <= k < T.

        datasets retains the HDF5 array handles; counts starts at zero for
        each split. Labels and event associations are JSON lists stored in
        UTF-8 HDF5 arrays, with one entry per window (including empty event
        lists for normal windows). No pickled objects are stored.
        Use a with block or call close() to release the file. If construction
        fails after creating the directory, any partial output is left there
        for inspection and an opened HDF5 file is closed.
        """
        if not isinstance(mode, str) or mode not in ("train_val", "test"):
            raise ValueError("mode must be 'train_val' or 'test'")
        if not isinstance(channel_ids, (list, tuple)):
            raise TypeError("channel_ids must be an ordered list or tuple")
        if not channel_ids or any(
            not isinstance(channel, str) or not channel.strip() or "\x00" in channel
            for channel in channel_ids
        ):
            raise ValueError("channel_ids must contain nonempty strings without null characters")
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError("channel_ids must be unique")
        for name, value in (("window_size", window_size), ("step_ns", step_ns)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer scalar, not a Boolean")
            if not 0 < int(value) <= np.iinfo(np.int64).max:
                raise ValueError(f"{name} must be positive and fit in int64")
        if (int(window_size) - 1) * int(step_ns) > np.iinfo(np.int64).max:
            raise ValueError("The window's endpoint span must fit in int64")

        # "log10": stored model-channel values (and the saved statistics) are
        # in log10 units, i.e. normalized log10 values unless norm_mode="none".
        if value_transform not in ("none", "log10"):
            raise ValueError("value_transform must be 'none' or 'log10'")
        self.value_transform = value_transform

        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"TOML file does not exist: {config_path}")

        self.output_dir = Path(output_dir)
        self.dataset_path = self.output_dir / "dataset.h5"
        self.mode = mode
        self.channel_ids = tuple(channel_ids)
        self.n_channels = len(self.channel_ids)
        self.window_size = int(window_size)
        self.step_ns = int(step_ns)
        self.splits = ("train", "val") if mode == "train_val" else ("test",)
        self.counts = {split: 0 for split in self.splits}
        self.datasets = {}
        self.file = None
        self.finalized = False
        self.failed = False
        self.grid_phase = None
        self.events = {}

        self.output_dir.mkdir(parents=True, exist_ok=False)
        try:
            shutil.copyfile(config_path, self.output_dir / "config.toml")
            # Exclusive creation: never truncate an existing HDF5 file.
            self.file = h5py.File(self.dataset_path, "x")
            self.file.attrs["schema_version"] = 1
            self.file.attrs["dataset_id"] = uuid.uuid4().hex
            self.file.attrs["mode"] = self.mode
            self.file.attrs["window_size"] = self.window_size
            self.file.attrs["step_ns"] = self.step_ns
            self.file.attrs["time_reference"] = "Unix UTC nanoseconds"
            self.file.attrs["window_end_convention"] = "last_included_sample"
            self.file.attrs["complete"] = False
            self.file.attrs["values_state"] = "raw"
            self.file.attrs["value_transform"] = self.value_transform
            self.file.create_dataset(
                "channel_ids", data=self.channel_ids, dtype=h5py.string_dtype("utf-8"),
            )

            for split in self.splits:
                group = self.file.create_group(split)
                # HDF5 chunks are storage blocks inside this same file.
                # They allow resizing without holding all windows in RAM.
                arrays = {
                    "values": group.create_dataset(
                        "values", shape=(0, self.n_channels, self.window_size),
                        maxshape=(None, self.n_channels, self.window_size),
                        chunks=True, dtype=np.float64,
                    ),
                }
                for name in ("window_start_ns", "window_end_ns"):
                    arrays[name] = group.create_dataset(
                        name, shape=(0,), maxshape=(None,), chunks=True, dtype=np.int64,
                    )
                for name in ("labels_json", "event_associations_json"):
                    arrays[name] = group.create_dataset(
                        name, shape=(0,), maxshape=(None,), chunks=True,
                        dtype=h5py.string_dtype("utf-8"),
                    )
                self.datasets[split] = arrays
        except BaseException:
            self.close()
            raise

    def _require_writable(self):
        if self.file is None or not self.file.id.valid:
            raise RuntimeError("DatasetWriter is closed")
        if self.failed:
            raise RuntimeError("A write failed; create a new build instead of resuming this file")
        if self.finalized:
            raise RuntimeError("DatasetWriter is already finalized")

    def _mark_failed(self):
        """Prevent retries that could normalize a partly transformed file twice."""
        self.failed = True
        try:
            self.file.attrs["complete"] = False
            self.file.attrs["values_state"] = "failed"
            self.file.flush()
        except Exception:
            # Preserve the original I/O error if the file cannot be updated.
            pass

    def _encode_labels(self, labels, event_associations, count):
        for name, rows in (("labels", labels), ("event_associations", event_associations)):
            if not isinstance(rows, (list, tuple)) or len(rows) != count:
                raise ValueError(f"{name} must contain one list per window")
        label_rows, event_rows, new_events = [], [], {}
        for window_labels, associations in zip(labels, event_associations):
            if not isinstance(window_labels, (list, tuple)) or not window_labels or any(
                not isinstance(label, str) or not label.strip() for label in window_labels
            ):
                raise ValueError("Each window needs a nonempty list of string labels")
            if len(set(window_labels)) != len(window_labels):
                raise ValueError("A window's labels must be unique")
            if not isinstance(associations, (list, tuple)):
                raise ValueError("Each window's event associations must be a list")
            if self.mode == "train_val" and associations:
                raise ValueError("Abort-free train/val windows cannot have event associations")
            seen = set()
            for association in associations:
                if not isinstance(association, dict):
                    raise ValueError("An event association must be a dictionary")
                for name in ("event_id", "category", "label"):
                    value = association.get(name)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"Event association {name} must be a nonempty string")
                event_id = association["event_id"]
                if association["label"] not in window_labels:
                    raise ValueError("Each event association's label must occur in window labels")
                if event_id in seen:
                    raise ValueError("An event may appear only once in each window's associations")
                seen.add(event_id)
                identity = {
                    "event_id": event_id, "category": association["category"],
                    "timestamp_ns": _integer(association.get("timestamp_ns"), "event timestamp_ns"),
                }
                previous = new_events.get(event_id, self.events.get(event_id))
                if previous is not None and previous != identity:
                    raise ValueError("An event_id cannot refer to different events")
                new_events[event_id] = identity
            label_rows.append(_json_text(window_labels))
            event_rows.append(_json_text(associations))
        return label_rows, event_rows, new_events

    def append(self, split, values, window_start_ns, window_end_ns, *, labels, event_associations):
        """Append one filtered, unnormalized batch and its aligned metadata.

        values is finite real NumPy data (N, C, T), stored as float64. Times
        are int64 arrays (N,) giving first/last INCLUDED samples. Starts must
        increase strictly, including across calls; overlapping windows within
        a split are allowed. All windows share the grid spacing/phase. Train
        and val input spans must not overlap. Empty batches are harmless.

        labels is a list of N nonempty lists, e.g. [["non_abort"], ["SBL"]].
        event_associations is a list of N lists of dictionaries. Each event
        dictionary requires event_id, category, timestamp_ns, and label;
        its label must occur in that window's labels. For example:
        {"event_id": "event-1", "category": "SBL", "timestamp_ns": 123,
         "label": "pre_abort_SBL"}. Multiple events can be retained without
        choosing a primary event. Normal windows use an empty event list.

        The caller assigns scientific labels and fits normalization on unique
        accepted TRAIN grid points separately. This method does not filter,
        label, fit statistics, or alter caller arrays. Invalid inputs are
        rejected before writing. An I/O failure makes this build unusable.
        """
        self._require_writable()
        if split not in self.splits:
            raise ValueError(f"split must be one of {self.splits}")
        if not isinstance(values, np.ndarray):
            raise TypeError("values must be a NumPy array")
        if values.ndim != 3 or values.shape[1:] != (self.n_channels, self.window_size):
            raise ValueError("values must have shape (N, C, T) matching the writer")
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("values must contain finite real numbers")
        with np.errstate(over="raise", invalid="raise"):
            values = values.astype(np.float64, copy=False)
        count = len(values)
        for name, times in (("window_start_ns", window_start_ns), ("window_end_ns", window_end_ns)):
            if not isinstance(times, np.ndarray) or times.dtype != np.dtype("int64"):
                raise TypeError(f"{name} must be an int64 NumPy array")
            if times.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
        span = (self.window_size - 1) * self.step_ns
        if np.any(window_start_ns > np.iinfo(np.int64).max - span) or np.any(
            window_end_ns != window_start_ns + span
        ):
            raise ValueError("Window endpoints must match T and step_ns without overflow")
        arrays = self.datasets[split]
        if count:
            if np.any(window_start_ns[1:] <= window_start_ns[:-1]) or (
                self.counts[split] and window_start_ns[0] <= arrays["window_start_ns"][-1]
            ):
                raise ValueError("Window starts must increase strictly within each split")
            phase = self.grid_phase if self.grid_phase is not None else int(window_start_ns[0]) % self.step_ns
            if np.any(window_start_ns % self.step_ns != phase):
                raise ValueError("All windows must use the same regular grid phase")
            if self.mode == "train_val":
                if split == "train" and self.counts["val"] and (
                    window_end_ns[-1] >= self.datasets["val"]["window_start_ns"][0]
                ):
                    raise ValueError("Train windows must end before validation begins")
                if split == "val" and self.counts["train"] and (
                    window_start_ns[0] <= self.datasets["train"]["window_end_ns"][-1]
                ):
                    raise ValueError("Validation windows must start after training ends")
        label_rows, event_rows, new_events = self._encode_labels(labels, event_associations, count)
        if count == 0:
            return
        payload = dict(values=values, window_start_ns=window_start_ns, window_end_ns=window_end_ns,
                       labels_json=label_rows, event_associations_json=event_rows)
        start = self.counts[split]
        stop = start + count
        try:
            for name, array in arrays.items():
                array.resize(stop, axis=0)
                array[start:stop] = payload[name]
        except BaseException:
            self._mark_failed()
            raise
        self.counts[split] = stop
        self.grid_phase = phase
        self.events.update(new_events)

    def _build_metadata(self, resolved_config, provenance, norm_mode):
        """Check the settings needed to interpret data and prevent split leakage."""
        # Round-trip to detach caller containers and convert supported NumPy/Path values.
        config = json.loads(_json_text(resolved_config))
        provenance = json.loads(_json_text(provenance))
        if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
            raise ValueError("resolved_config needs a dataset table")
        settings = config["dataset"]
        for key, expected in (("mode", self.mode), ("pvs", list(self.channel_ids)),
                              ("window_size_samples", self.window_size), ("norm_mode", norm_mode)):
            if settings.get(key) != expected:
                raise ValueError(f"Resolved configuration {key} does not match the writer")
        if settings.get("step_ns") != self.step_ns:
            raise ValueError("resolved_config.dataset.step_ns must match the resolved integer spacing")
        definitions = config.get("pvs")
        if not isinstance(definitions, list) or any(not isinstance(pv, dict) for pv in definitions):
            raise ValueError("resolved_config.pvs must contain PV definitions")
        identities = []
        for channel in self.channel_ids:
            matches = [pv for pv in definitions if pv.get("id") == channel]
            if len(matches) != 1 or not isinstance(matches[0].get("expected_pv"), str) or not matches[0]["expected_pv"].strip():
                raise ValueError(f"Exactly one PV identity is required for channel {channel!r}")
            identities.append({"id": channel, "expected_pv": matches[0]["expected_pv"]})
        if not isinstance(provenance, dict) or not isinstance(provenance.get("split_intervals_ns"), dict):
            raise ValueError("provenance needs split_intervals_ns with half-open UTC intervals")
        intervals = provenance["split_intervals_ns"]
        if set(intervals) != set(self.splits):
            raise ValueError("Provenance must describe exactly the stored splits")
        for split, interval in intervals.items():
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("Each split interval must be [start_ns, end_ns)")
            start, end = (_integer(value, "split endpoint") for value in interval)
            if start >= end:
                raise ValueError("Split intervals must have positive duration")
            arrays = self.datasets[split]
            if int(arrays["window_start_ns"][0]) < start or int(arrays["window_end_ns"][-1]) >= end:
                raise ValueError("Saved windows must lie inside their declared split interval")
        if self.mode == "train_val" and intervals["train"][1] > intervals["val"][0]:
            raise ValueError("Train and validation split intervals must not overlap")
        return {"resolved_config": config, "provenance": provenance, "channel_identity": identities}

    def finalize(self, normalizer=None, *, resolved_config, provenance,
                 statistics_filepath=None, batch_size=256):
        """Normalize stored windows in bounded batches, save metadata, and seal.

        Train/val: supply a finalized NormalizeWindows fitted only on unique
        accepted training-grid timestamps. Test with normalization: supply
        statistics_filepath pointing to a completed TRAIN/VAL dataset.h5;
        do not pass a newly fitted normalizer. For explicit none mode in
        either build, pass NormalizeWindows("none", n_channels=C).

        resolved_config is the parsed configuration plus resolved values,
        including dataset.step_ns (integer nanoseconds). provenance must
        include split_intervals_ns, e.g. {"train": [a, b], "val": [b, c]},
        using half-open UTC ns intervals; other provenance fields are saved.
        Selected ids and exact expected_pv identities must match the source
        training file, and test intervals must not overlap its train/val.

        The original TOML stays unchanged. JSON metadata inside HDF5 records
        resolved settings, provenance, labels/events, training statistics,
        ddof, constant-channel fallback, and any source dataset identity.
        Every configured split must have at least one accepted window.

        Validation occurs before transformation. An I/O or transform failure
        leaves complete=False and prevents retries on partly normalized data.
        Successful finalization forbids further appends or finalizations;
        close() remains safe. Readers must require complete=True.
        """
        self._require_writable()
        batch_size = _integer(batch_size, "batch_size")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if any(count == 0 for count in self.counts.values()):
            raise ValueError("Cannot finalize an empty split")
        source = None
        source_build = None
        if statistics_filepath is not None:
            if self.mode != "test" or normalizer is not None:
                raise ValueError("statistics_filepath is only for test mode, without a normalizer")
            with h5py.File(statistics_filepath, "r") as training:
                if not training.attrs.get("complete", False) or training.attrs.get("mode") != "train_val":
                    raise ValueError("Statistics source must be a completed train_val dataset")
                if training.attrs.get("schema_version") != 1:
                    raise ValueError("Unsupported statistics source schema")
                if training["channel_ids"].asstr()[:].tolist() != list(self.channel_ids):
                    raise ValueError("Statistics source channel order does not match")
                # Datasets written before the option existed carry no attribute: "none".
                source_transform = str(training.attrs.get("value_transform", "none"))
                if source_transform != self.value_transform:
                    raise ValueError(f"Statistics source value_transform {source_transform!r} does not match "
                                     f"this build's {self.value_transform!r} (dataset.log_transform)")
                normalizer = _restore_normalizer(
                    json.loads(training["normalization_json"].asstr()[()]), self.n_channels,
                )
                source_build = json.loads(training["build_metadata_json"].asstr()[()])
                source = {"dataset_id": training.attrs["dataset_id"], "path": str(statistics_filepath)}
        else:
            if not isinstance(normalizer, NormalizeWindows):
                raise TypeError("Provide a finalized NormalizeWindows or a training statistics_filepath")
            if self.mode == "test" and normalizer.norm_mode != "none":
                raise ValueError("Normalized test data must reuse statistics_filepath from training")
            # A detached snapshot keeps the saved parameters and applied transform identical.
            normalizer = _restore_normalizer(normalizer.to_metadata(), self.n_channels)
        build = self._build_metadata(resolved_config, provenance, normalizer.norm_mode)
        if source_build is not None:
            if build["channel_identity"] != source_build["channel_identity"]:
                raise ValueError("Statistics source PV identities do not match")
            test_start, test_end = build["provenance"]["split_intervals_ns"]["test"]
            for start, end in source_build["provenance"]["split_intervals_ns"].values():
                if test_start < end and test_end > start:
                    raise ValueError("Test interval overlaps the statistics source train/val intervals")
        build["normalization_source"] = source
        build["window_counts"] = dict(self.counts)
        metadata = {
            "normalization_json": _json_text(normalizer.to_metadata()),
            "build_metadata_json": _json_text(build),
            "events_json": _json_text(list(self.events.values())),
        }
        try:
            for name, contents in metadata.items():
                self.file.create_dataset(name, data=contents, dtype=h5py.string_dtype("utf-8"))
            self.file.attrs["values_state"] = "transforming"
            if normalizer.norm_mode != "none":
                for split in self.splits:
                    values = self.datasets[split]["values"]
                    for start in range(0, self.counts[split], batch_size):
                        stop = min(start + batch_size, self.counts[split])
                        values[start:stop] = normalizer.apply_normalization(values[start:stop])
            for split, count in self.counts.items():
                self.file[split].attrs["window_count"] = count
            self.file.attrs["values_state"] = "raw" if normalizer.norm_mode == "none" else "normalized"
            self.file.flush()
            # Publish completion only after all values and metadata have been flushed.
            self.file.attrs["complete"] = True
            self.file.flush()
        except BaseException:
            self._mark_failed()
            raise
        self.finalized = True

    def close(self):
        """Flush and close the HDF5 file; repeated calls are harmless."""
        if self.file is not None:
            self.file.close()

    def __enter__(self):
        """Return this writer when entering a with block."""
        if self.file is None or not self.file.id.valid:
            raise RuntimeError("Cannot enter a closed DatasetWriter")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close on normal exit or an exception; let exceptions propagate."""
        self.close()
