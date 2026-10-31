from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from dataset.time_grid import convert_to_unix_ns

import numpy as np


def read_pv(filepath, expected_pv):
    """Read and validate one PV's raw measurements from a Parquet file.

    The file must contain exactly one row with columns pv, t0, times, and
    values. The times and values cells must be non-null Arrow lists of
    integer or floating-point elements with equal lengths. Time offsets
    must be finite and nondecreasing; equal consecutive offsets are allowed.

    Args:
        filepath: String or pathlib.Path identifying the Parquet file.
        expected_pv: Expected PV identity string, compared exactly with
            the value stored in the pv column.

    Returns:
        A dictionary containing:
            pv: The stored PV identity, matching expected_pv.
            t0_ms: The non-null reference timestamp from the t0 column,
                expressed in milliseconds since the Unix epoch.
            relative_times: One-dimensional float64 NumPy array of
                measurement offsets in seconds from t0_ms.
            values: One-dimensional float64 NumPy array of measurements,
                with each value paired with the offset at the same index.

        Measurement NaNs and infinities are retained for later validity
        masking; measurement positions are not removed.
        Empty numeric lists produce empty arrays. No resampling or
        conversion to absolute timestamps is performed.

    Raises:
        FileNotFoundError: If filepath does not identify an existing file.
        ValueError: If the row count, PV identity, reference presence,
            list types, array shapes or lengths, or time offsets violate
            the requirements above.

    Notes:
        The file is not modified. Array conversion may reuse Arrow buffers,
        so returned arrays are not guaranteed to be writable. Make an
        explicit copy if independent writable storage is needed. The
        integer type of t0_ms is checked by convert_to_unix_ns, not here.
        Arrow read and schema errors propagate unchanged.
    """
    
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Missing PV data file under path: {path}")
        
    parquet_file = pq.ParquetFile(path) # load parquet file in "path"
    
    
    table = parquet_file.read(["pv", "t0", "times", "values"]) # read 4 columns
    if table.num_rows != 1:
        raise ValueError(f"Expected one row in {path}, found {table.num_rows}")
        
    
    
    # Assign all read columns of table to names -> choose first element -> convert to python object
    pv = table.column("pv")[0].as_py()
    if pv != expected_pv:
        raise ValueError(f"PV mismatch in {path}: expected {expected_pv!r}, found {pv!r}")
    
    # Extract reference file instance t0 in milliseconds after UTC epoch
    t0_ms = table.column("t0")[0].as_py()
    if t0_ms is None:
        raise ValueError(f"Missing t0 in {path}")
    
    # a bunch of error and testing statements for debugging
    
    # measurement offset relative to t0 in seconds
    times_entry = table.column("times")[0]
    # Check of times_entry is an Arrow scalar
    if not isinstance(times_entry, (pa.ListScalar, pa.LargeListScalar)):
        raise ValueError(
        f"Expected a list in 'times' in {path}, found {times_entry.type}"
                        )
    # Check if list "times_entry" is not None
    if not times_entry.is_valid:
        raise ValueError(f"Missing 'times' list in {path}")

    # Check if loaded values in values_entry are integer or floating point type
    element_type = times_entry.type.value_type
    if not (
        pa.types.is_integer(element_type)
        or pa.types.is_floating(element_type)
           ):
        raise ValueError(
            f"Expected numeric elements in 'times' in {path}, "
            f"found {element_type}"
                        )
    
    # corresponding measurement values
    values_entry = table.column("values")[0]
    # Check of values_entry is an Arrow scalar
    if not isinstance(values_entry, (pa.ListScalar, pa.LargeListScalar)):
        raise ValueError(
        f"Expected a list in 'values' in {path}, found {values_entry.type}"
                        )
    # Check if list "values_entry" is not None
    if not values_entry.is_valid:
        raise ValueError(f"Missing 'values' list in {path}")

    # Check if loaded values in values_entry are integer or floating point type
    element_type = values_entry.type.value_type
    if not (
        pa.types.is_integer(element_type)
        or pa.types.is_floating(element_type)
           ):
        raise ValueError(
            f"Expected numeric elements in 'values' in {path}, "
            f"found {element_type}"
                        )
    
    # Convert to numpy array objects
    relative_times = np.asarray(times_entry.values.to_numpy(zero_copy_only=False),                                  dtype=np.float64,
                                )
    values = np.asarray(values_entry.values.to_numpy(zero_copy_only=False),
                        dtype=np.float64,
                        )
    # relative_times and values should be a 1D arrays of equal length
    if relative_times.ndim != 1 or values.ndim != 1 or len(relative_times) != len(values):
        raise ValueError(f"Expected matching one-dimensional 'times' and 'values' arrays "
                         f"in {path}; got {relative_times.shape} and {values.shape}"
                        )
    # array entries should be finite; do not impose on values as later masking and windowing works with them
    if not np.isfinite(relative_times).all():
        raise ValueError(f"relative_times contains non-finite values in {path}")
        
    # Check for correct ordering of relative_times
    if (np.diff(relative_times) < 0).any():
        raise ValueError(f"found reversed ordered values in relative_times in {path}")
    
    
    
    return {
    "pv": pv,
    "t0_ms": t0_ms,
    "relative_times": relative_times,
    "values": values,
    }
    
    
    
    
def read_pv_files(filepaths, expected_pv):
    """Read multiple files for one PV into one chronological series."""
     
    timestamps = []
    pv_values = []
     
    for filepath in filepaths:
        pv_dict = read_pv(filepath, expected_pv)
        
        # Extract parameters
        t0_ms = pv_dict["t0_ms"]
        relative_times = pv_dict["relative_times"]
        values = pv_dict["values"]
        
        # Convert to unix time
        unix_times = convert_to_unix_ns(t0_ms, relative_times)
        
        # Append to lists
        pv_values.append(values)
        timestamps.append(unix_times)
        
    # Return empty arrays when timestamps list is empty
    if len(timestamps) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        
    concat_timestamps = np.concatenate(timestamps)
    concat_pv_values = np.concatenate(pv_values)
    
    # Order measurements by timestamp
    order = np.argsort(concat_timestamps, kind="stable")
    ordered_timestamps = concat_timestamps[order]
    ordered_pv_values = concat_pv_values[order]
    
    return ordered_timestamps, ordered_pv_values
    
    
    
    
    
    
