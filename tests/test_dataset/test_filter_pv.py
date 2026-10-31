"""Small synthetic checks for completeness, configuration, and selection."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from dataset.filter_pv import WindowFilter, complete_window_mask


def test_complete_window_mask():
    # Four windows, two channels, three time points per channel.
    window_valid = np.ones((4, 2, 3), dtype=bool)
    window_valid[1, 0, 2] = False
    window_valid[2, 1, 0] = False
    original = window_valid.copy()

    keep = complete_window_mask(window_valid)
    expected = np.array([True, False, False, True])
    np.testing.assert_array_equal(keep, expected, strict=True)
    np.testing.assert_array_equal(window_valid, original, strict=True)

    # Reference creator.py uses this rule with min_valid_fraction = 1.0.
    reference_keep = (window_valid.mean(axis=2) >= 1.0).all(axis=1)
    np.testing.assert_array_equal(keep, reference_keep, strict=True)

    # The same decision retains all channels and the matching timestamps.
    values = np.arange(24, dtype=np.float64).reshape(4, 2, 3)
    starts_ns = np.array([100, 200, 300, 400], dtype=np.int64)
    np.testing.assert_array_equal(values[keep], values[[0, 3]], strict=True)
    np.testing.assert_array_equal(
        starts_ns[keep], np.array([100, 400], dtype=np.int64), strict=True
    )


def test_empty_batch():
    keep = complete_window_mask(np.empty((0, 2, 3), dtype=bool))
    np.testing.assert_array_equal(keep, np.array([], dtype=bool), strict=True)


def test_invalid_inputs():
    with np.testing.assert_raises(TypeError):
        complete_window_mask([[[True]]])
    with np.testing.assert_raises(ValueError):
        complete_window_mask(np.ones((2, 3), dtype=bool))
    # Values are not validity flags: even nonzero floats must be rejected.
    with np.testing.assert_raises(ValueError):
        complete_window_mask(np.ones((4, 2, 3), dtype=np.float64))
    for shape in [(4, 0, 3), (4, 2, 0)]:
        with np.testing.assert_raises(ValueError):
            complete_window_mask(np.empty(shape, dtype=bool))


def make_window_filter(**overrides):
    """Use valid settings by default and override individual test inputs."""
    settings = {
        "mode": "train_val",
        "ler_threshold": 400.0,
        "her_threshold": 450.0,
        "aborts_filepath": "synthetic_aborts.json",
        "exclude_before_ns": 5_000_000_000,
        "exclude_after_ns": 3_000_000_000,
        "pre_abort_ns": 2_000_000_000,
        "post_abort_ns": 1_000_000_000,
    }
    settings.update(overrides)
    return WindowFilter(**settings)


def test_window_filter_valid_settings():
    for mode in ("train_val", "test"):
        selector = make_window_filter(
            mode=mode,
            ler_threshold=np.float32(400.5),
            her_threshold=np.int64(450),
            exclude_before_ns=0,
            exclude_after_ns=np.int32(3),
            pre_abort_ns=np.int64(2_000_000_000),
            post_abort_ns=np.uint64(1_000_000_000),
        )
        assert selector.mode == mode
        assert selector.ler_threshold == 400.5
        assert selector.her_threshold == 450
        assert selector.aborts_filepath == Path("synthetic_aborts.json")
        assert selector.exclude_before_ns == 0
        assert selector.exclude_after_ns == 3
        assert selector.pre_abort_ns == 2_000_000_000
        assert selector.post_abort_ns == 1_000_000_000

    # The agreed threshold restriction is finiteness, not positivity.
    selector = make_window_filter(ler_threshold=-1.0, her_threshold=0)
    assert selector.ler_threshold == -1.0
    assert selector.her_threshold == 0


def test_window_filter_invalid_mode():
    for mode in ("train", "", None, True, np.array(["test"])):
        with np.testing.assert_raises_regex(ValueError, "mode"):
            make_window_filter(mode=mode)


def test_window_filter_invalid_thresholds():
    for name in ("ler_threshold", "her_threshold"):
        for value in (True, np.bool_(False), None, "400", 400 + 0j,
                      np.array(400.0), [400.0]):
            with np.testing.assert_raises_regex(TypeError, name):
                make_window_filter(**{name: value})
        for value in (np.nan, np.inf, -np.inf, np.float32(np.nan)):
            with np.testing.assert_raises_regex(ValueError, name):
                make_window_filter(**{name: value})


def test_window_filter_invalid_durations():
    for name in ("exclude_before_ns", "exclude_after_ns", "pre_abort_ns", "post_abort_ns"):
        for value in (True, np.bool_(False), 1.0, 1.5, np.float64(1),
                      np.nan, np.inf, "1", None, 1 + 0j, np.array(1)):
            with np.testing.assert_raises_regex(TypeError, name):
                make_window_filter(**{name: value})
        for value in (-1, np.int64(-1)):
            with np.testing.assert_raises_regex(ValueError, name):
                make_window_filter(**{name: value})


def write_abort_catalog(path, timestamps):
    # Uncategorized aborts still participate in exclusion and association.
    records = [{"timestamp": timestamp, "category": None} for timestamp in timestamps]
    path.write_text(json.dumps(records), encoding="utf-8")


def test_filter_modes_on_synthetic_windows():
    """Each window spans 20 seconds and contains 21 samples at 1 Hz."""
    second = 1_000_000_000
    origin_ns = pd.Timestamp("2026-01-01", tz="UTC").value
    # Event at 100 s: normal exclusion [70,150], pre ends [80,100),
    # post starts >100 and ends <=130. One bad current sample is sufficient.
    # name, start(s), one LER sample, one HER sample, complete, test, train
    cases = [
        ("normal before", 40, 410., 460., True, True, True),
        ("LER at threshold", 41, 400., 460., True, False, False),
        ("HER at threshold", 42, 410., 450., True, False, False),
        ("excluded but too early for pre", 59, 410., 460., True, False, False),
        ("pre lower boundary", 60, 100., 100., True, True, False),
        ("pre before event", 79, 100., 100., True, True, False),
        ("ends on event", 80, 100., 100., True, True, False),
        ("contains event", 90, 100., 100., True, True, False),
        ("starts on event", 100, 100., 100., True, True, False),
        ("post", 101, 100., 100., True, True, False),
        ("post upper boundary", 110, 100., 100., True, True, False),
        ("overlaps post but ends too late", 111, 410., 460., True, False, False),
        ("incomplete pre", 61, np.nan, 460., False, False, False),
        ("incomplete abort", 81, 410., np.nan, False, False, False),
        ("incomplete post", 102, np.nan, 460., False, False, False),
        ("incomplete normal", 170, 410., 460., False, False, False),
        ("normal after", 180, 410., 460., True, True, True),
    ]
    starts = origin_ns + np.array([case[1] for case in cases], dtype=np.int64) * second
    ends = starts + 20 * second
    complete = np.array([case[4] for case in cases], dtype=bool)
    ler = np.full((len(cases), 21), 410.)
    her = np.full((len(cases), 21), 460.)
    ler[:, 10] = [case[2] for case in cases]
    her[:, 10] = [case[3] for case in cases]
    inputs = (complete, ler, her, starts, ends)
    originals = [array.copy() for array in inputs]

    with TemporaryDirectory() as directory:
        path = Path(directory) / "aborts.json"
        # Tokyo local time corresponds to 00:01:40 UTC, i.e. origin + 100 s.
        write_abort_catalog(path, ["2026-01-01 09:01:40"])
        for mode, expected_column in (("test", 5), ("train_val", 6)):
            selector = make_window_filter(
                mode=mode, aborts_filepath=path,
                exclude_before_ns=30 * second, exclude_after_ns=50 * second,
                pre_abort_ns=20 * second, post_abort_ns=30 * second,
            )
            keep = selector.filter_pv_windows(*inputs)
            expected = np.array([case[expected_column] for case in cases], dtype=bool)
            np.testing.assert_array_equal(keep, expected, strict=True)
            for actual, original in zip(inputs, originals, strict=True):
                np.testing.assert_array_equal(actual, original, strict=True)


def test_test_filter_multiple_events_and_catalog_order():
    second = 1_000_000_000
    origin_ns = pd.Timestamp("2026-01-01", tz="UTC").value
    starts = origin_ns + np.array([60, 90, 110, 160, 190, 210, 250], dtype=np.int64) * second
    ends = starts + 20 * second
    complete = np.ones(len(starts), dtype=bool)
    low_currents = np.full((len(starts), 21), 100.)
    expected = np.array([True, True, True, True, True, True, False])

    with TemporaryDirectory() as directory:
        path = Path(directory) / "aborts.json"
        timestamps = ["2026-01-01 09:01:40", "2026-01-01 09:03:20"]
        for order in (timestamps, timestamps[::-1]):
            write_abort_catalog(path, order)
            selector = make_window_filter(
                mode="test", aborts_filepath=path,
                pre_abort_ns=20 * second, post_abort_ns=30 * second,
            )
            keep = selector.filter_pv_windows(
                complete, low_currents, low_currents, starts, ends,
            )
            np.testing.assert_array_equal(keep, expected, strict=True)


def test_test_filter_zero_association_durations_and_exact_times():
    second = 1_000_000_000
    event_ns = pd.Timestamp("2026-01-01 00:01:40.123456789", tz="UTC").value
    # Contains-abort windows remain eligible even when their end is after
    # the post interval. Windows missing the event by 1 ns are not associated.
    starts = event_ns + np.array([-20 * second - 1, -20 * second, 0, 1], dtype=np.int64)
    ends = starts + 20 * second
    complete = np.ones(4, dtype=bool)
    low_currents = np.full((4, 21), 100.)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "aborts.json"
        write_abort_catalog(path, ["2026-01-01 09:01:40.123456789"])
        selector = make_window_filter(
            mode="test", aborts_filepath=path, pre_abort_ns=0, post_abort_ns=0,
        )
        keep = selector.filter_pv_windows(
            complete, low_currents, low_currents, starts, ends,
        )
        np.testing.assert_array_equal(keep, np.array([False, True, True, False]), strict=True)


def test_test_filter_empty_catalog_and_batch():
    origin_ns = pd.Timestamp("2026-01-01", tz="UTC").value
    starts = origin_ns + np.arange(4, dtype=np.int64) * 1_000_000_000
    ends = starts + 2_000_000_000
    complete = np.array([True, True, True, False])
    ler = np.full((4, 3), 410.)
    her = np.full((4, 3), 460.)
    ler[1, 1] = 400.
    her[2, 1] = 440.

    with TemporaryDirectory() as directory:
        path = Path(directory) / "aborts.json"
        write_abort_catalog(path, [])
        selector = make_window_filter(mode="test", aborts_filepath=path)
        keep = selector.filter_pv_windows(complete, ler, her, starts, ends)
        np.testing.assert_array_equal(keep, np.array([True, False, False, False]), strict=True)
        for timestamps in ([], ["2026-01-01 09:01:40"]):
            write_abort_catalog(path, timestamps)
            keep = selector.filter_pv_windows(
                np.empty(0, dtype=bool), np.empty((0, 3)), np.empty((0, 3)),
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            )
            np.testing.assert_array_equal(keep, np.empty(0, dtype=bool), strict=True)


if __name__ == "__main__":
    test_complete_window_mask()
    test_empty_batch()
    test_invalid_inputs()
    print("PASS: joint completeness, metadata alignment, reference rule, and input checks")
    test_window_filter_valid_settings()
    test_window_filter_invalid_mode()
    test_window_filter_invalid_thresholds()
    test_window_filter_invalid_durations()
    print("PASS: filter settings, numeric types, zero buffers, and invalid inputs")
    test_filter_modes_on_synthetic_windows()
    test_test_filter_multiple_events_and_catalog_order()
    test_test_filter_zero_association_durations_and_exact_times()
    test_test_filter_empty_catalog_and_batch()
    print("PASS: test selection, train regression, association boundaries, completeness, and empty inputs")
