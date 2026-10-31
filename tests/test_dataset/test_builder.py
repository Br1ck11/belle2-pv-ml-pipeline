"""End-to-end tests with explicit expected values, labels and provenance."""

import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from dataset import build_dataset
from dataset.demo import ORIGIN_NS, write_demo_inputs
from dataset.filter_pv import WindowFilter
from dataset.parse_config import resolve_config, resolve_pv_filepaths, seconds_to_ns


SECOND = 1_000_000_000


def contents(path):
    with h5py.File(path, "r") as data:
        assert data.attrs["complete"]
        result = {"channels": data["channel_ids"].asstr()[:].tolist(),
                  "norm": json.loads(data["normalization_json"].asstr()[()]),
                  "metadata": json.loads(data["build_metadata_json"].asstr()[()]),
                  "events": json.loads(data["events_json"].asstr()[()])}
        for split in ("train", "val", "test"):
            if split in data:
                group = data[split]
                result[split] = {name: group[name][:] for name in ("values", "window_start_ns", "window_end_ns")}
                for name in ("labels_json", "event_associations_json"):
                    result[split][name] = [json.loads(text) for text in group[name].asstr()[:]]
    return result


@pytest.mark.parametrize("mode", ["zscore", "minmax", "none"])
def test_full_build_known_values_and_batch_invariance(tmp_path, mode):
    train_config, test_config = write_demo_inputs(tmp_path / "archive")
    train_template = train_config.read_text().replace('norm_mode = "zscore"', f'norm_mode = "{mode}"')
    test_template = test_config.read_text().replace('norm_mode = "zscore"', f'norm_mode = "{mode}"')
    if mode == "none":
        test_template = test_template.replace('statistics_filepath = "train_val/dataset.h5"\n', '')
    unique = np.array([[7., 7., 7., 7., 7.], [0., 1., 4., 9., 16.]])
    offset = unique.mean(axis=1) if mode == "zscore" else unique.min(axis=1)
    scale = unique.std(axis=1) if mode == "zscore" else np.ptp(unique, axis=1)
    scale[scale == 0] = 1.
    if mode == "none":
        offset, scale = np.zeros(2), np.ones(2)
    previous = None
    for batch in (1, 2, 7):
        train_config.write_text(train_template.replace('output_dir = "train_val"', f'output_dir = "train_{batch}"')
                                .replace('batch_size = 2', f'batch_size = {batch}'))
        train = contents(build_dataset(train_config, progress=None))
        assert train["channels"] == ["constant", "signal"]
        # Batch Welford and direct NumPy may differ by floating-point roundoff.
        np.testing.assert_allclose(train["norm"]["offset"], offset, rtol=1e-14, atol=0)
        np.testing.assert_allclose(train["norm"]["scale"], scale, rtol=1e-14, atol=0)
        assert train["norm"]["count"] == (0 if mode == "none" else 5)
        assert train["norm"]["scale_fallback_mask"] == ([False, False] if mode == "none" else [True, False])
        assert train["norm"]["variance_ddof"] == (0 if mode == "zscore" else None)
        # Training windows overlap at t=2; their union contains five points.
        for split, starts, signal in (
            ("train", [0, 2], [[0, 1, 4], [4, 9, 16]]),
            # Missing t=12 may NOT borrow t=11 from train; t=17 can use t=16.
            ("val", [14, 16, 18], [[196, 225, 256], [256, 256, 324], [324, 361, 400]]),
        ):
            actual = train[split]
            raw = np.stack([np.full((len(starts), 3), 7.), np.array(signal, dtype=float)], axis=1)
            expected = (raw - offset[None, :, None]) / scale[None, :, None]
            np.testing.assert_allclose(actual["values"], expected, rtol=1e-14, atol=1e-14)
            np.testing.assert_array_equal(actual["window_start_ns"], ORIGIN_NS + np.array(starts) * SECOND)
            np.testing.assert_array_equal(actual["window_end_ns"], actual["window_start_ns"] + 2 * SECOND)
            assert actual["labels_json"] == [["non_abort"]] * len(starts)
            assert actual["event_associations_json"] == [[]] * len(starts)
        provenance = train["metadata"]["provenance"]
        assert provenance["selection"]["train"]["unique_covered_grid_points"] == 5
        assert provenance["selection"]["train"]["unused_trailing_grid_points"] == 1
        assert provenance["selection"]["val"]["rejected_incomplete"] == 2
        output = train_config.parent / f"train_{batch}"
        assert {item.name for item in output.iterdir()} == {"dataset.h5", "config.toml"}
        assert (output / "config.toml").read_bytes() == train_config.read_bytes()

        test_config.write_text(test_template.replace('output_dir = "test"', f'output_dir = "test_{batch}"')
                               .replace('train_val/dataset.h5', f'train_{batch}/dataset.h5')
                               .replace('batch_size = 2', f'batch_size = {batch}'))
        test = contents(build_dataset(test_config, progress=None))
        starts = list(range(101, 111)) + list(range(114, 122))
        actual = test["test"]
        np.testing.assert_array_equal(actual["window_start_ns"], ORIGIN_NS + np.array(starts) * SECOND)
        raw_signal = np.array([[s**2, (s+1)**2, (s+2)**2] for s in starts], dtype=float)
        raw = np.stack([np.full_like(raw_signal, 7.), raw_signal], axis=1)
        np.testing.assert_allclose(actual["values"], (raw - offset[None, :, None]) / scale[None, :, None], rtol=1e-14)
        assert test["norm"] == train["norm"]
        # Independent scalar relation checks; no use of the filter helper.
        for row, start in enumerate(starts):
            expected = []
            for event_time, category in ((105.5, "SBL"), (108, "RF")):
                end = start + 2
                if start <= event_time <= end:
                    label = category
                elif event_time - 4 <= end < event_time:
                    label = f"pre_abort_{category}"
                elif start > event_time and end <= event_time + 4:
                    label = f"post_abort_{category}"
                else:
                    continue
                expected.append((category, label, ORIGIN_NS + int(event_time * SECOND)))
            assert actual["labels_json"][row] == ([item[1] for item in expected] or ["non_abort"])
            associations = actual["event_associations_json"][row]
            assert [(e["category"], e["label"], e["timestamp_ns"]) for e in associations] == expected
            for association in associations:
                assert association["event_id"] in {event["event_id"] for event in test["events"]}
        if previous is not None:
            for split in ("train", "val"):
                np.testing.assert_allclose(train[split]["values"], previous[0][split]["values"], rtol=1e-14)
            assert actual["event_associations_json"] == previous[1]["test"]["event_associations_json"]
        previous = train, test


def test_event_ids_match_reference_and_preserve_multiple_same_category_events(tmp_path):
    # Expected identities were computed once with the original abort_windows
    # catalogue (load_abort_catalog) before it was removed on 2026-10-27.
    reference = [
        ("abort-d1e93e65d6938b7c", 1767225610123456789, "SBL"),
        ("abort-eeba12dd13350f6a", 1767225611123456789, "SBL"),
        ("abort-1bccc29cc4f50296", 1767225620000000000, "Unspecified"),
    ]
    path = tmp_path / "events.json"
    records = [
        {"timestamp": "2026-01-01 09:00:10.123456789", "category": " SBL ", "ring": "ler"},
        {"timestamp": "2026-01-01 09:00:11.123456789", "category": "SBL", "ring": "HER"},
        {"timestamp": "2026-01-01 09:00:20.000000000", "category": None, "ring": "Both"},
    ]
    path.write_text(json.dumps(records))
    selector = WindowFilter("test", 400, 400, path, exclude_before_ns=0, exclude_after_ns=0,
                            pre_abort_ns=5*SECOND, post_abort_ns=5*SECOND)
    assert [(e["event_id"], e["timestamp_ns"], e["category"]) for e in selector.load_events()] == reference
    # End at 8 s is pre-abort for BOTH SBL events. One label, two event IDs.
    labels, events = selector.label_windows(np.array([ORIGIN_NS + 6*SECOND]), np.array([ORIGIN_NS + 8*SECOND]))
    assert labels == [["pre_abort_SBL"]]
    assert len(events[0]) == 2 and events[0][0]["event_id"] != events[0][1]["event_id"]
    # Reading again must use the original cached snapshot, even if file changes.
    path.write_text('[]')
    assert selector.label_windows(np.array([ORIGIN_NS + 6*SECOND]), np.array([ORIGIN_NS + 8*SECOND])) == (labels, events)


def test_empty_selection_leaves_incomplete_output_and_cleans_scratch(tmp_path):
    config, _ = write_demo_inputs(tmp_path / "inputs")
    config.write_text(config.read_text().replace('ler_threshold = 400.0', 'ler_threshold = 9999.0'))
    with pytest.raises(ValueError, match="train has no accepted windows"):
        build_dataset(config, progress=None)
    output = config.parent / "train_val"
    assert {path.name for path in output.iterdir()} == {"dataset.h5", "config.toml"}
    with h5py.File(output / "dataset.h5") as data:
        assert not data.attrs["complete"]
    with pytest.raises(FileExistsError):
        build_dataset(config, progress=None)


def test_normalized_test_rejects_overlap_with_training(tmp_path):
    train, test = write_demo_inputs(tmp_path / "inputs")
    build_dataset(train, progress=None)
    test.write_text(test.read_text().replace('00:01:40Z', '00:00:00Z').replace('00:02:04Z', '00:00:24Z'))
    with pytest.raises(ValueError, match="overlaps"):
        build_dataset(test, progress=None)
    with h5py.File(test.parent / "test/dataset.h5") as data:
        assert not data.attrs["complete"]


def test_relative_paths_are_config_based_and_cli_builds(tmp_path):
    config, _ = write_demo_inputs(tmp_path / "inputs")
    resolved = resolve_config(config)
    assert resolved["dataset"]["required_pvs"] == ["constant", "signal", "ler", "her"]
    assert resolved["dataset"]["pvs"] == ["constant", "signal"]
    # Run from outside the project; uses the installed package, not cwd imports.
    result = subprocess.run([sys.executable, "-m", "dataset", "--config", str(config)],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "DATASET:" in result.stdout
    contents(config.parent / "train_val/dataset.h5")


@pytest.mark.parametrize("before,after,message", [
    ('start = "2026-01-01T00:00:00Z"', '', 'Missing'),
    ('start = "2026-01-01T00:00:00Z"', 'start = "2026-01-01T00:00:00"', 'timezone'),
    ('window_size_samples = 3', 'window_size_samples = 99', 'fewer than'),
    ('stride_samples = 2', 'stride_samples = 0', 'stride_samples'),
    ('batch_size = 2', 'batch_size = true', 'batch_size'),
    ('pvs = ["constant", "signal"]', 'pvs = ["signal", "signal"]', 'duplicate'),
    ('pvs = ["constant", "signal"]', 'pvs = ["missing"]', 'no.*definition'),
    ('her_pv = "her"', 'her_pv = "ler"', 'different PV ids'),
    ('sample_period_seconds = 1.0', 'sample_period_seconds = 0.0000000001', 'nanoseconds'),
    ('train_fraction = 0.5', 'train_fraction = 0.01', 'at least one grid point'),
    ('ler_threshold = 400.0', 'ler_threshold = nan', 'finite'),
])
def test_invalid_configuration_fails_before_output(tmp_path, before, after, message):
    config, _ = write_demo_inputs(tmp_path / "inputs")
    config.write_text(config.read_text().replace(before, after))
    with pytest.raises(ValueError, match=message):
        build_dataset(config, progress=None)
    assert not (config.parent / "train_val").exists()


def test_seconds_are_exact_not_truncated():
    assert seconds_to_ns(0.123456789, "duration") == 123456789
    for bad in (True, float("nan"), -1, 1e-10):
        with pytest.raises(ValueError):
            seconds_to_ns(bad, "duration")


def test_discovery_preserves_entry_order_and_sorts_each_directory(tmp_path):
    for name in ("first.parquet", "last.parquet", "z_dir/b.parquet", "z_dir/a.parquet",
                 "z_dir/readme.txt", "z_dir/nested/skip.parquet", "a_dir/c.parquet"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "z_dir/directory.parquet").mkdir()
    entries = ["first.parquet", "z_dir", str(tmp_path / "a_dir"), "last.parquet"]
    original = entries.copy()
    actual = resolve_pv_filepaths(entries, tmp_path, pv_id="pressure")
    expected = ["first.parquet", "z_dir/a.parquet", "z_dir/b.parquet", "a_dir/c.parquet", "last.parquet"]
    assert actual == [str(tmp_path / name) for name in expected]
    assert entries == original
    # Explicit-file order is preserved even when it is not alphabetical.
    assert resolve_pv_filepaths(["last.parquet", "first.parquet"], tmp_path, pv_id="pressure") == [
        str(tmp_path / "last.parquet"), str(tmp_path / "first.parquet"),
    ]


@pytest.mark.parametrize("entries", [
    ["pv", "pv/a.parquet"], ["pv", "pv"], ["pv/a.parquet", "pv/../pv/a.parquet"],
    ["pv/a.parquet", "alias.parquet"],
])
def test_discovery_rejects_duplicate_resolved_files(tmp_path, entries):
    folder = tmp_path / "pv"
    folder.mkdir()
    (folder / "a.parquet").touch()
    (tmp_path / "alias.parquet").symlink_to(folder / "a.parquet")
    with pytest.raises(ValueError, match="PV 'pressure'.*repeat the same file"):
        resolve_pv_filepaths(entries, tmp_path, pv_id="pressure")


def test_discovery_missing_and_empty_directories(tmp_path):
    with pytest.raises(FileNotFoundError, match="PV 'pressure'.*path does not exist"):
        resolve_pv_filepaths(["missing"], tmp_path, pv_id="pressure")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no immediate .parquet files"):
        resolve_pv_filepaths(["empty"], tmp_path, pv_id="pressure")
    (empty / "notes.txt").touch()
    nested = empty / "nested"
    nested.mkdir()
    (nested / "only_nested.parquet").touch()
    with pytest.raises(ValueError, match="no immediate .parquet files"):
        resolve_pv_filepaths(["empty"], tmp_path, pv_id="pressure")


@pytest.mark.parametrize("entries", [[], "pv", [None], [" "], ["bad\x00path"]])
def test_discovery_rejects_invalid_entries(tmp_path, entries):
    with pytest.raises(ValueError, match="PV 'pressure' filepaths"):
        resolve_pv_filepaths(entries, tmp_path, pv_id="pressure")


def test_directory_and_explicit_files_produce_identical_builds(tmp_path):
    train_config, test_config = write_demo_inputs(tmp_path / "archive")
    root = train_config.parent
    # Give each PV its own directory, as in the real daily archive.
    templates = {path: path.read_text() for path in (train_config, test_config)}
    for channel in ("signal", "constant", "ler", "her"):
        folder = root / "raw" / channel
        folder.mkdir()
        for index in (0, 1):
            old = f"raw/{channel}_{index}.parquet"
            new = f"raw/{channel}/{channel}_{index}.parquet"
            (root / old).rename(root / new)
            for path in templates:
                templates[path] = templates[path].replace(old, new)
        (folder / "notes.txt").write_text("Not a measurement file")
    # Explicit files provide the comparison result and the saved train stats.
    expected = {}
    for path, text in templates.items():
        path.write_text(text)
        expected[path] = contents(build_dataset(path, progress=None))

    for original, text in templates.items():
        mode = original.stem
        text = text.replace(f'output_dir = "{mode}"', f'output_dir = "{mode}_directories"')
        for channel in ("signal", "constant", "ler", "her"):
            file_list = f'["raw/{channel}/{channel}_0.parquet", "raw/{channel}/{channel}_1.parquet"]'
            text = text.replace(f'filepaths = {file_list}', f'filepaths = ["raw/{channel}"]')
        directory_config = root / f"{mode}_directories.toml"
        directory_config.write_text(text)
        path = build_dataset(directory_config, progress=None)
        actual, reference = contents(path), expected[original]
        for key in ("channels", "norm", "events"):
            assert actual[key] == reference[key]
        for split in (("train", "val") if mode == "train_val" else ("test",)):
            for name in ("values", "window_start_ns", "window_end_ns"):
                np.testing.assert_array_equal(actual[split][name], reference[split][name], strict=True)
            for name in ("labels_json", "event_associations_json"):
                assert actual[split][name] == reference[split][name]
        for key in ("sources", "selection", "split_intervals_ns"):
            assert actual["metadata"]["provenance"][key] == reference["metadata"]["provenance"][key]
        assert actual["metadata"]["resolved_config"]["pvs"] == reference["metadata"]["resolved_config"]["pvs"]
        # The copied TOML keeps directory entries; HDF5 stores expanded files.
        assert (path.parent / "config.toml").read_bytes() == directory_config.read_bytes()
