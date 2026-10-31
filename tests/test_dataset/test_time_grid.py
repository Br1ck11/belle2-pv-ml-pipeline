"""Small, known-answer checks for conversion, grid creation, and sampling."""

import numpy as np

from dataset.time_grid import convert_to_unix_ns, make_time_grid, sample_previous


def test_convert_to_unix_ns():
    # Edit the inputs and expected timestamps together for your own example.
    t0_ms = 1_700_000_000_000
    relative_times = np.array([0.0, 0.5, 2.0], dtype=np.float64)
    expected = np.array(
        [1_700_000_000_000_000_000,
         1_700_000_000_500_000_000,
         1_700_000_002_000_000_000],
        dtype=np.int64,
    )

    result = convert_to_unix_ns(t0_ms, relative_times)
    # strict=True also checks shape and dtype.
    np.testing.assert_array_equal(result, expected, strict=True)


def test_make_time_grid():
    result = make_time_grid(start_ns=10, end_ns=16, step_ns=2)
    expected = np.array([10, 12, 14], dtype=np.int64)  # End is excluded.
    np.testing.assert_array_equal(result, expected, strict=True)

    empty = make_time_grid(start_ns=10, end_ns=10, step_ns=2)
    np.testing.assert_array_equal(empty, np.array([], dtype=np.int64), strict=True)


def test_sample_previous():
    # The same example as in our discussion, now using real Unix timestamps.
    start_ns = 1_700_000_000_000_000_000
    second_ns = 1_000_000_000
    source_times = start_ns + np.array([1, 3, 6], dtype=np.int64) * second_ns
    source_values = np.array([10.0, 30.0, 60.0], dtype=np.float64)
    grid = make_time_grid(start_ns, start_ns + 7 * second_ns, second_ns)
    original_times, original_values, original_grid = (
        source_times.copy(), source_values.copy(), grid.copy()
    )

    values, mask = sample_previous(source_times, source_values, grid, second_ns)
    expected_values = np.array([np.nan, 10, 10, 30, 30, np.nan, 60], dtype=np.float64)
    expected_mask = np.array([False, True, True, True, True, False, True])
    # Matching NaNs are accepted by assert_array_equal.
    np.testing.assert_array_equal(values, expected_values, strict=True)
    np.testing.assert_array_equal(mask, expected_mask, strict=True)

    # Inputs and timestamp positions must remain unchanged.
    np.testing.assert_array_equal(source_times, original_times, strict=True)
    np.testing.assert_array_equal(source_values, original_values, strict=True)
    np.testing.assert_array_equal(grid, original_grid, strict=True)
    assert grid[5] == start_ns + 5 * second_ns  # Its value is NaN, not removed.

    # With zero maximum age, only exact timestamp matches are valid.
    values, mask = sample_previous(source_times, source_values, grid, 0)
    expected_values = np.array([np.nan, 10, np.nan, 30, np.nan, np.nan, 60])
    expected_mask = np.array([False, True, False, True, False, False, True])
    np.testing.assert_array_equal(values, expected_values, strict=True)
    np.testing.assert_array_equal(mask, expected_mask, strict=True)


def test_duplicate_and_nonfinite_sources():
    # Tiny integer nanoseconds are sufficient to test these selection rules.
    times = np.array([0, 1, 1, 2, 3], dtype=np.int64)
    source_values = np.array([5.0, 10.0, 20.0, np.nan, np.inf])
    grid = np.array([1, 2, 3], dtype=np.int64)

    values, mask = sample_previous(times, source_values, grid, max_age_ns=10)
    # Last duplicate wins. Invalid latest values do not fall back to 20.0.
    np.testing.assert_array_equal(
        values, np.array([20.0, np.nan, np.nan]), strict=True
    )
    np.testing.assert_array_equal(mask, np.array([True, False, False]), strict=True)


def test_empty_inputs():
    empty_times = np.array([], dtype=np.int64)
    empty_values = np.array([], dtype=np.float64)
    grid = np.array([0, 1, 2], dtype=np.int64)

    values, mask = sample_previous(empty_times, empty_values, grid, 1)
    np.testing.assert_array_equal(values, np.full(3, np.nan), strict=True)
    np.testing.assert_array_equal(mask, np.zeros(3, dtype=bool), strict=True)

    values, mask = sample_previous(
        np.array([0], dtype=np.int64), np.array([10.0]), empty_times, 1
    )
    np.testing.assert_array_equal(values, empty_values, strict=True)
    np.testing.assert_array_equal(mask, np.array([], dtype=bool), strict=True)


def test_invalid_inputs():
    times = np.array([1, 3, 6], dtype=np.int64)
    values = np.array([10.0, 30.0, 60.0])
    grid = np.array([0, 1, 2], dtype=np.int64)

    # Each block fails if the function does NOT raise the expected error.
    with np.testing.assert_raises(ValueError):
        convert_to_unix_ns(0, np.array([np.nan]))
    with np.testing.assert_raises(ValueError):
        make_time_grid(0, 10, 0)
    with np.testing.assert_raises(ValueError):
        make_time_grid(10, 0, 1)
    with np.testing.assert_raises(ValueError):
        sample_previous(times, values, grid, -1)
    with np.testing.assert_raises(ValueError):
        sample_previous(times[::-1], values, grid, 1)
    with np.testing.assert_raises(ValueError):
        sample_previous(times, values[:2], grid, 1)
    with np.testing.assert_raises_regex(ValueError, "one-dimensional"):
        sample_previous(times, values.reshape(1, 3), grid, 1)
    with np.testing.assert_raises_regex(ValueError, "one-dimensional"):
        sample_previous(times, values, grid.reshape(1, 3), 1)


if __name__ == "__main__":
    test_convert_to_unix_ns()
    test_make_time_grid()
    test_sample_previous()
    test_duplicate_and_nonfinite_sources()
    test_empty_inputs()
    test_invalid_inputs()
    print("PASS: conversion, grid, sampling, edge cases, and invalid inputs")
