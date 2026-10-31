"""Known-answer HDF5 round trips; all output lives in temporary directories."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import numpy as np

from dataset.filter_pv import training_coverage_mask
from dataset.normalize import NormalizeWindows
from dataset.save_dataset import DatasetWriter


SECOND = 1_000_000_000
BASE = 1_700_000_000_000_000_000


def configuration(mode, norm_mode):
    return {
        "dataset": {"mode": mode, "pvs": ["b", "a"], "window_size_samples": 4,
                    "step_ns": SECOND, "norm_mode": norm_mode},
        "pvs": [{"id": "b", "expected_pv": "physical:b"},
                {"id": "a", "expected_pv": "physical:a"}],
    }


def make_writer(root, name, mode):
    config = root / f"{name}.toml"
    config.write_text(
        f'# Synthetic fixture\n[dataset]\nmode = "{mode}"\noutput_dir = "{name}"\n',
        encoding="utf-8",
    )
    return DatasetWriter(root / name, config, mode=mode, channel_ids=["b", "a"],
                         window_size=4, step_ns=SECOND)


def append_windows(writer, split, values, seconds, *, labels=None, associations=None):
    starts = BASE + np.array(seconds, dtype=np.int64) * SECOND
    writer.append(
        split, values, starts, starts + 3 * SECOND,
        labels=[["non_abort"] for _ in starts] if labels is None else labels,
        event_associations=[[] for _ in starts] if associations is None else associations,
    )


def provenance(mode):
    intervals = {"train": [BASE, BASE + 15 * SECOND], "val": [BASE + 15 * SECOND, BASE + 30 * SECOND]}
    if mode == "test":
        intervals = {"test": [BASE + 40 * SECOND, BASE + 60 * SECOND]}
    return {"split_intervals_ns": intervals, "source_files": [Path("synthetic.parquet")]}


def test_round_trip_all_normalization_modes_and_event_associations():
    grid = np.array([[0., 1., 3., 7., 11., 17., 1e9, 1e9, 41., 47., 59., 71.], [5.] * 12])
    starts = np.array([0, 2, 8], dtype=np.int64)
    coverage = training_coverage_mask(starts, window_size=4, grid_size=12)
    raw = np.stack([grid[:, start:start + 4] for start in starts])
    original = raw.copy()
    for mode in ("zscore", "minmax", "none"):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            normalizer = NormalizeWindows(mode, n_channels=2)
            for start, stop in ((0, 3), (3, 7), (7, 12)):
                normalizer.calculate_statistics(grid[:, start:stop][:, coverage[start:stop]])
            normalizer.finalize()
            if mode == "zscore":
                offset, scale = grid[:, coverage].mean(axis=1), grid[:, coverage].std(axis=1)
            elif mode == "minmax":
                offset, scale = grid[:, coverage].min(axis=1), np.ptp(grid[:, coverage], axis=1)
            else:
                offset, scale = np.zeros(2), np.ones(2)
            scale[scale == 0] = 1
            expected = (raw - offset[None, :, None]) / scale[None, :, None]

            training = make_writer(root, "training", "train_val")
            with training:
                append_windows(training, "train", raw[:1], [0])
                append_windows(training, "train", raw[1:], [2, 8])
                append_windows(training, "val", raw[:2], [20, 22])
                np.testing.assert_array_equal(training.datasets["train"]["values"][:], raw)
                training.finalize(normalizer, resolved_config=configuration("train_val", mode),
                                  provenance=provenance("train_val"), batch_size=1)
                with np.testing.assert_raises(RuntimeError):
                    append_windows(training, "train", raw[:1], [12])
                with np.testing.assert_raises(RuntimeError):
                    training.finalize(normalizer, resolved_config={}, provenance={})

            with h5py.File(training.dataset_path, "r") as saved:
                assert saved.attrs["complete"]
                assert saved.attrs["values_state"] == ("raw" if mode == "none" else "normalized")
                np.testing.assert_allclose(saved["train"]["values"][:], expected)
                np.testing.assert_allclose(saved["val"]["values"][:], expected[:2])
                for split, seconds in (("train", [0, 2, 8]), ("val", [20, 22])):
                    times = saved[split]["window_start_ns"][:]
                    np.testing.assert_array_equal(times, BASE + np.array(seconds) * SECOND)
                    reconstructed = times[:, None] + np.arange(4) * saved.attrs["step_ns"]
                    np.testing.assert_array_equal(reconstructed[:, -1], saved[split]["window_end_ns"][:])
                    assert saved[split].attrs["window_count"] == len(seconds)
                metadata = json.loads(saved["normalization_json"].asstr()[()])
                assert metadata == normalizer.to_metadata()
                assert metadata["count"] == (0 if mode == "none" else 10)
                if mode != "none":
                    assert metadata["raw_scale"][1] == 0 and metadata["scale"][1] == 1
                    assert metadata["scale_fallback_mask"] == [False, True]
                build = json.loads(saved["build_metadata_json"].asstr()[()])
                assert build["provenance"]["source_files"] == ["synthetic.parquet"]

            event1 = {"event_id": "e1", "category": "SBL", "timestamp_ns": np.int64(BASE + 45 * SECOND)}
            event2 = {"event_id": "e2", "category": "RF", "timestamp_ns": BASE + 46 * SECOND}
            labels = [["pre_abort_SBL"], ["SBL", "pre_abort_RF"], ["non_abort"]]
            associations = [[event1 | {"label": "pre_abort_SBL"}],
                            [event1 | {"label": "SBL"}, event2 | {"label": "pre_abort_RF"}], []]
            test = make_writer(root, "test", "test")
            with test:
                append_windows(test, "test", raw, [40, 42, 48], labels=labels, associations=associations)
                kwargs = {"statistics_filepath": training.dataset_path}
                if mode == "none":
                    kwargs = {"normalizer": NormalizeWindows("none", n_channels=2)}
                test.finalize(**kwargs, resolved_config=configuration("test", mode),
                              provenance=provenance("test"), batch_size=2)
            with h5py.File(test.dataset_path, "r") as saved:
                assert saved.attrs["complete"]
                np.testing.assert_allclose(saved["test"]["values"][:], expected)
                assert [json.loads(text) for text in saved["test"]["labels_json"].asstr()[:]] == labels
                assert [json.loads(text) for text in saved["test"]["event_associations_json"].asstr()[:]] == associations
                assert json.loads(saved["events_json"].asstr()[()]) == [event1, event2]
                build = json.loads(saved["build_metadata_json"].asstr()[()])
                if mode != "none":
                    assert build["normalization_source"]["path"] == str(training.dataset_path)
                    assert json.loads(saved["normalization_json"].asstr()[()]) == normalizer.to_metadata()
    np.testing.assert_array_equal(raw, original)


def test_invalid_batches_and_metadata_leave_data_unchanged():
    with TemporaryDirectory() as temporary:
        with make_writer(Path(temporary), "build", "train_val") as writer:
            values = np.ones((1, 2, 4))
            starts = np.array([BASE], dtype=np.int64)
            arguments = dict(split="train", values=values, window_start_ns=starts,
                             window_end_ns=starts + 3 * SECOND, labels=[["non_abort"]], event_associations=[[]])
            for replacement in (
                {"split": "test"}, {"values": np.ones((1, 4, 2))},
                {"values": np.full((1, 2, 4), np.nan)},
                {"window_start_ns": starts.astype(float)}, {"window_end_ns": starts + 2 * SECOND},
                {"labels": []}, {"labels": [[]]}, {"event_associations": []},
                {"window_start_ns": np.array([np.iinfo(np.int64).max], dtype=np.int64)},
            ):
                with np.testing.assert_raises((TypeError, ValueError)):
                    writer.append(**(arguments | replacement))
                assert writer.counts["train"] == 0
                assert all(len(array) == 0 for array in writer.datasets["train"].values())
            writer.append(**arguments)
            for replacement in ({}, {"split": "val"},
                                {"window_start_ns": starts + SECOND + 1, "window_end_ns": starts + 4 * SECOND + 1}):
                with np.testing.assert_raises(ValueError):
                    writer.append(**(arguments | replacement))
            append_windows(writer, "train", np.empty((0, 2, 4)), [])
            normalizer = NormalizeWindows("none", n_channels=2)
            with np.testing.assert_raises_regex(ValueError, "empty split"):
                writer.finalize(normalizer, resolved_config=configuration("train_val", "none"), provenance=provenance("train_val"))
            append_windows(writer, "val", values, [20])
            with np.testing.assert_raises_regex(ValueError, "dataset table"):
                writer.finalize(normalizer, resolved_config={}, provenance={})
            assert not writer.failed and not writer.file.attrs["complete"]
            assert "normalization_json" not in writer.file
            np.testing.assert_array_equal(writer.datasets["train"]["values"][:], values)
            writer.finalize(normalizer, resolved_config=configuration("train_val", "none"), provenance=provenance("train_val"))


def test_test_statistics_guards_and_failed_transform():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        normalizer = NormalizeWindows("zscore", n_channels=2)
        normalizer.calculate_statistics(np.array([[1., 3.], [5., 5.]]))
        normalizer.finalize()
        training = make_writer(root, "training", "train_val")
        with training:
            append_windows(training, "train", np.ones((1, 2, 4)), [0])
            append_windows(training, "val", np.ones((1, 2, 4)), [20])
            training.finalize(normalizer, resolved_config=configuration("train_val", "zscore"), provenance=provenance("train_val"))
        with make_writer(root, "test", "test") as writer:
            event = {"event_id": "e", "category": "SBL", "timestamp_ns": BASE + 45 * SECOND, "label": "pre_abort_SBL"}
            append_windows(writer, "test", np.ones((1, 2, 4)), [40], labels=[["pre_abort_SBL"]], associations=[[event]])
            for invalid_event in (event | {"category": "changed"}, event | {"label": "missing"}, event | {"timestamp_ns": 1.5}):
                with np.testing.assert_raises((TypeError, ValueError)):
                    append_windows(writer, "test", np.ones((1, 2, 4)), [41], labels=[["pre_abort_SBL"]], associations=[[invalid_event]])
            assert writer.counts["test"] == 1
            with np.testing.assert_raises_regex(ValueError, "reuse statistics_filepath"):
                writer.finalize(normalizer, resolved_config=configuration("test", "zscore"), provenance=provenance("test"))
            invalid = deepcopy(configuration("test", "zscore"))
            invalid["pvs"][0]["expected_pv"] = "different:PV"
            with np.testing.assert_raises_regex(ValueError, "PV identities"):
                writer.finalize(statistics_filepath=training.dataset_path, resolved_config=invalid, provenance=provenance("test"))
            overlapping = {"split_intervals_ns": {"test": [BASE, BASE + 60 * SECOND]}}
            with np.testing.assert_raises_regex(ValueError, "overlaps"):
                writer.finalize(statistics_filepath=training.dataset_path, resolved_config=configuration("test", "zscore"), provenance=overlapping)
            with h5py.File(training.dataset_path, "r+") as saved:
                saved.attrs["complete"] = False
            with np.testing.assert_raises_regex(ValueError, "completed"):
                writer.finalize(statistics_filepath=training.dataset_path, resolved_config=configuration("test", "zscore"), provenance=provenance("test"))
            with h5py.File(training.dataset_path, "r+") as saved:
                saved.attrs["complete"] = True
                saved["channel_ids"][:] = ["a", "b"]
            with np.testing.assert_raises_regex(ValueError, "channel order"):
                writer.finalize(statistics_filepath=training.dataset_path, resolved_config=configuration("test", "zscore"), provenance=provenance("test"))

        with make_writer(root, "failed", "train_val") as writer:
            append_windows(writer, "train", np.ones((2, 2, 4)), [0, 2])
            append_windows(writer, "val", np.ones((1, 2, 4)), [20])
            original = NormalizeWindows.apply_normalization
            sizes = []

            def fail_second_batch(instance, values):
                sizes.append(len(values))
                if len(sizes) == 2:
                    raise OSError("synthetic write failure")
                return original(instance, values)

            with patch.object(NormalizeWindows, "apply_normalization", fail_second_batch):
                with np.testing.assert_raises_regex(OSError, "synthetic write failure"):
                    writer.finalize(normalizer, resolved_config=configuration("train_val", "zscore"),
                                    provenance=provenance("train_val"), batch_size=1)
            assert sizes == [1, 1]
            assert writer.failed and not writer.file.attrs["complete"]
            assert writer.file.attrs["values_state"] == "failed"
            with np.testing.assert_raises(RuntimeError):
                append_windows(writer, "train", np.ones((1, 2, 4)), [8])
            with np.testing.assert_raises(RuntimeError):
                writer.finalize(normalizer, resolved_config=configuration("train_val", "zscore"), provenance=provenance("train_val"))


if __name__ == "__main__":
    test_round_trip_all_normalization_modes_and_event_associations()
    test_invalid_batches_and_metadata_leave_data_unchanged()
    test_test_statistics_guards_and_failed_transform()
    print("PASS: HDF5 round trips, labels/events, UTC times, normalization reuse, and failure handling")
