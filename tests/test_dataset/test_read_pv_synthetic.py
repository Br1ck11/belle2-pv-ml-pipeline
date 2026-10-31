"""Check both PV readers using small, temporary synthetic Parquet files."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dataset.read_pv import read_pv, read_pv_files


def write_synthetic_pv(path, pv, t0_ms, relative_times, values):
    """Save explicitly supplied inputs in the archive's one-row format."""
    # Each outer list represents one table row. The times/values cells
    # themselves contain lists of measurements.
    table = pa.table(
        {
            "pv": [pv],
            "t0": pa.array([t0_ms], type=pa.int64()),
            "times": pa.array([relative_times], type=pa.list_(pa.float64())),
            "values": pa.array([values], type=pa.list_(pa.float64())),
        }
    )
    pq.write_table(table, path)


def test_read_pv_round_trip():
    # Edit these inputs to choose exactly what goes into the Parquet file.
    expected_pv = "SYNTHETIC:CURRENT"
    t0_ms = 1_700_000_000_000
    relative_times = [0.0, 0.5, 2.0, 3.0]  # Seconds after t0_ms.
    values = [10.0, np.nan, 12.5, 11.0]

    # The synthetic file is removed automatically when this block finishes.
    with TemporaryDirectory(prefix="read_pv_experiment_") as directory:
        path = Path(directory) / "synthetic.parquet"
        write_synthetic_pv(path, expected_pv, t0_ms, relative_times, values)

        result = read_pv(path, expected_pv)

        assert result["pv"] == expected_pv, "PV name changed during the round trip"
        assert result["t0_ms"] == t0_ms, "Reference timestamp changed"
        # strict=True also checks shape and dtype. Matching NaNs are accepted.
        np.testing.assert_array_equal(
            result["relative_times"],
            np.asarray(relative_times, dtype=np.float64),
            strict=True,
        )
        np.testing.assert_array_equal(
            result["values"],
            np.asarray(values, dtype=np.float64),
            strict=True,
        )

        # These checks fail if the reader does not raise the expected error.
        with np.testing.assert_raises(ValueError):
            read_pv(path, expected_pv + ":WRONG")

        with np.testing.assert_raises(FileNotFoundError):
            read_pv(path.with_name("missing.parquet"), expected_pv)


def test_read_pv_files_round_trip():
    expected_pv = "SYNTHETIC:CURRENT"
    # Two UTC midnights: 2024-01-01 and 2024-01-02.
    first_t0_ms = 1_704_067_200_000
    second_t0_ms = 1_704_153_600_000
    first_relative_times = [0.0, 0.5, 0.5]  # Duplicate timestamp on purpose.
    first_values = [10.0, 11.0, -5.0]  # Values must not be sorted independently.
    second_relative_times = [0.0, 0.5, 1.0]
    second_values = [30.0, np.nan, np.inf]

    # Write expected outputs explicitly, independently of the converter.
    expected_times = np.array(
        [1_704_067_200_000_000_000,
         1_704_067_200_500_000_000,
         1_704_067_200_500_000_000,
         1_704_153_600_000_000_000,
         1_704_153_600_500_000_000,
         1_704_153_601_000_000_000],
        dtype=np.int64,
    )
    expected_values = np.array([10.0, 11.0, -5.0, 30.0, np.nan, np.inf])

    with TemporaryDirectory(prefix="read_pv_files_experiment_") as directory:
        root = Path(directory)
        first = root / "CURRENT-2024_01_01.parquet"
        second = root / "CURRENT-2024_01_02.parquet"
        write_synthetic_pv(first, expected_pv, first_t0_ms,
                           first_relative_times, first_values)
        write_synthetic_pv(second, expected_pv, second_t0_ms,
                           second_relative_times, second_values)

        # Both file orders must produce the same chronological measurements.
        for paths in ([first, second], [second, first]):
            times, values = read_pv_files(paths, expected_pv)
            np.testing.assert_array_equal(times, expected_times, strict=True)
            np.testing.assert_array_equal(values, expected_values, strict=True)

        # Validation must also apply to files after the first one.
        wrong_pv = root / "wrong_pv.parquet"
        write_synthetic_pv(wrong_pv, "OTHER:CURRENT", second_t0_ms, [0.0], [1.0])
        with np.testing.assert_raises(ValueError):
            read_pv_files([first, wrong_pv], expected_pv)
        with np.testing.assert_raises(FileNotFoundError):
            read_pv_files([first, root / "missing.parquet"], expected_pv)


def test_empty_inputs():
    expected_pv = "SYNTHETIC:CURRENT"
    empty_times = np.array([], dtype=np.int64)
    empty_values = np.array([], dtype=np.float64)

    with TemporaryDirectory(prefix="read_pv_empty_experiment_") as directory:
        path = Path(directory) / "empty.parquet"
        write_synthetic_pv(path, expected_pv, 0, [], [])

        result = read_pv(path, expected_pv)
        np.testing.assert_array_equal(result["relative_times"], empty_values, strict=True)
        np.testing.assert_array_equal(result["values"], empty_values, strict=True)

        # No files and a file with no measurements are distinct cases.
        for paths in ([], [path]):
            times, values = read_pv_files(paths, expected_pv)
            np.testing.assert_array_equal(times, empty_times, strict=True)
            np.testing.assert_array_equal(values, empty_values, strict=True)


if __name__ == "__main__":
    test_read_pv_round_trip()
    print("PASS: single-file round trip, PV mismatch, and missing path")
    test_read_pv_files_round_trip()
    print("PASS: multi-file ordering, pairing, duplicates, and non-finite values")
    test_empty_inputs()
    print("PASS: empty file list and empty measurements, including shapes and dtypes")
