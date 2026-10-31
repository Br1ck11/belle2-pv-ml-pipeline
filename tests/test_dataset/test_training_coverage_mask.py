"""Small checks for unique accepted training-grid coverage."""

import numpy as np

from dataset.filter_pv import training_coverage_mask


def test_overlapping_windows_select_unique_points():
    starts = np.array([0, 2, 8], dtype=np.int64)
    original = starts.copy()
    starts.flags.writeable = False
    expected = np.array([True] * 6 + [False] * 2 + [True] * 4)
    mask = training_coverage_mask(starts, window_size=4, grid_size=12)
    np.testing.assert_array_equal(mask, expected, strict=True)
    np.testing.assert_array_equal(starts, original, strict=True)
    assert mask.sum() == 10

    # The mask selects each covered grid point once: the 12 window samples
    # (overlap at indices 2-3) reduce to the 10 unique points 0-5 and 8-11.
    values = np.array([0., 1., 3., 7., 11., 17., 23., 31., 41., 47., 59., 71.])
    np.testing.assert_array_equal(np.flatnonzero(mask), [0, 1, 2, 3, 4, 5, 8, 9, 10, 11])
    assert values[mask].mean() == (0 + 1 + 3 + 7 + 11 + 17 + 41 + 47 + 59 + 71) / 10

    # Duplicate windows and file/window ordering must not alter coverage.
    repeated = np.array([8, 2, 0, 2], dtype=np.uint64)
    actual = training_coverage_mask(repeated, np.uint64(4), np.int32(12))
    np.testing.assert_array_equal(actual, expected, strict=True)


def test_empty_inputs_and_exact_boundary():
    empty = np.array([], dtype=np.int64)
    for grid_size in (0, 2, 12):
        actual = training_coverage_mask(empty, window_size=4, grid_size=grid_size)
        np.testing.assert_array_equal(actual, np.zeros(grid_size, dtype=bool), strict=True)
    actual = training_coverage_mask(np.array([8]), window_size=4, grid_size=12)
    np.testing.assert_array_equal(actual, np.array([False] * 8 + [True] * 4), strict=True)


def test_invalid_input_types_and_shapes():
    for starts in ([0, 2], None, 0):
        with np.testing.assert_raises(TypeError):
            training_coverage_mask(starts, 4, 12)
    for starts in (np.array(0), np.array([[0, 2]])):
        with np.testing.assert_raises_regex(ValueError, "one-dimensional"):
            training_coverage_mask(starts, 4, 12)
    for starts in (
        np.array([0.]), np.array([True]), np.array([0j]),
        np.array([0], dtype=object), np.array(["0"]),
    ):
        with np.testing.assert_raises_regex(ValueError, "integer dtype"):
            training_coverage_mask(starts, 4, 12)
    for name in ("window_size", "grid_size"):
        for invalid in (True, np.bool_(False), 4., np.nan, None, "4", np.array(4)):
            sizes = {"window_size": 4, "grid_size": 12, name: invalid}
            with np.testing.assert_raises_regex(TypeError, name):
                training_coverage_mask(np.array([0]), **sizes)


def test_invalid_sizes_and_out_of_bounds_windows():
    for window_size, grid_size in ((0, 12), (-1, 12), (4, -1)):
        with np.testing.assert_raises(ValueError):
            training_coverage_mask(np.array([], dtype=np.int64), window_size, grid_size)
    for starts, window_size, grid_size in (
        (np.array([-1]), 4, 12),
        (np.array([0, 9]), 4, 12),  # Check later entries too.
        (np.array([12]), 4, 12),
        (np.array([0]), 13, 12),
        (np.array([0]), 1, 0),
        (np.array([np.iinfo(np.int64).max]), np.int64(4), 12),
        (np.array([np.iinfo(np.uint64).max], dtype=np.uint64), np.uint64(4), 12),
    ):
        with np.testing.assert_raises_regex(ValueError, "outside grid"):
            training_coverage_mask(starts, window_size, grid_size)


if __name__ == "__main__":
    test_overlapping_windows_and_reference()
    test_empty_inputs_and_exact_boundary()
    test_invalid_input_types_and_shapes()
    test_invalid_sizes_and_out_of_bounds_windows()
    print("PASS: unique coverage, reference comparison, empty inputs, types, shapes, and bounds")
