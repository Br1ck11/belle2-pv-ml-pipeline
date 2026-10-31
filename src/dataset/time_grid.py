import numpy as np


def convert_to_unix_ns(t0_ms, relative_times) -> np.ndarray:
    """Convert a Unix reference time and relative offsets to nanoseconds.

    Convert the reference from milliseconds to integer nanoseconds. Scale
    the relative seconds to nanoseconds, round with np.rint (halfway cases
    go to the nearest even integer), and add them to the integer reference.
    Keeping the reference in integer arithmetic avoids representing the
    full absolute timestamp as a floating-point number.

    Args:
        t0_ms: Python or NumPy integer giving milliseconds since the Unix
            epoch. None and Boolean values are not accepted.
        relative_times: One-dimensional float64 NumPy array of finite
            offsets in seconds from t0_ms. An empty array is allowed.

    Returns:
        A new one-dimensional int64 NumPy array of absolute UTC Unix
        timestamps in nanoseconds, with the same shape and ordering as
        relative_times. Empty input produces an empty array. The input
        array is not modified, and no sorting or resampling is performed.

    Raises:
        TypeError: If t0_ms is not an integer or is a Boolean, or if
            relative_times is not a NumPy array.
        ValueError: If t0_ms is None, or relative_times has the wrong
            dimensionality or dtype, or contains non-finite offsets.

    Notes:
        The reference, converted offsets, and resulting timestamps must
        fit in int64. Overflow is not explicitly checked by this function.
    """
    
    # Check validity of inputs (redundant if you only use the outputs of "read_pv.py" as this function's inputs)
    if t0_ms is None:
        raise ValueError("t0_ms must not be None")
    if isinstance(t0_ms, (bool, np.bool_)) or not isinstance(t0_ms, (int, np.integer)):
        raise TypeError("t0_ms must be an integer number of milliseconds")
    
    if not isinstance(relative_times, np.ndarray):
        raise TypeError("relative_times must be a NumPy array")
    if relative_times.ndim != 1:
        raise ValueError( f"relative_times must be one-dimensional; got shape {relative_times.shape}"
                        )
    # Check if relative_times elements are float64
    if relative_times.dtype != np.dtype("float64"):
        raise ValueError("relative_times must have dtype float64 to preserve temporal precision")
    if not np.isfinite(relative_times).all():
        raise ValueError(f"relative_times contains non-finite values")
        
        
    
    # Convert milliseconds to nanoseconds
    t0_ns = int(t0_ms) * 1_000_000
    # Convert seconds to nanoseconds
    relative_times_ns = np.rint(relative_times * 1_000_000_000).astype(np.int64)
    
    # Convert to absolute unix integers
    unix_times_ns = relative_times_ns + t0_ns

    return unix_times_ns




def make_time_grid(start_ns, end_ns, step_ns) -> np.ndarray:
    """Build a regular timestamp grid over the interval [start_ns, end_ns).

    Grid points are start_ns + k * step_ns for nonnegative integers k,
    stopping before end_ns. The requested start defines the grid origin;
    the function does not infer an origin from individual PV measurements.

    Args:
        start_ns: Inclusive start as an integer UTC Unix timestamp in
            nanoseconds.
        end_ns: Exclusive end in the same units. Must be greater than or
            equal to start_ns.
        step_ns: Positive integer spacing between consecutive grid points,
            in nanoseconds. All three arguments accept Python or NumPy
            integers; Booleans are not accepted.

    Returns:
        A new one-dimensional int64 NumPy array of regularly spaced UTC
        Unix timestamps. Equal bounds produce an empty array. For a
        nonempty interval, start_ns is included even if step_ns exceeds
        the interval length; end_ns is always excluded.

    Raises:
        ValueError: If an argument is not an integer or is a Boolean,
            step_ns is nonpositive, or end_ns precedes start_ns.

    Notes:
        Bounds, spacing, and grid arithmetic must fit in int64. Overflow
        is not explicitly checked. This function only creates timestamps;
        it does not split data, sample PV values, or apply validity masks.
    """
    
    # Check if inputs are not integers
    if isinstance(step_ns, (bool, np.bool_)) or not isinstance(step_ns, (int, np.integer)):
        raise ValueError("step_ns must be an integer")
    if isinstance(start_ns, (bool, np.bool_)) or not isinstance(start_ns, (int, np.integer)):
        raise ValueError("start_ns must be an integer")
    if isinstance(end_ns, (bool, np.bool_)) or not isinstance(end_ns, (int, np.integer)):
        raise ValueError("end_ns must be an integer")
        
    if step_ns <= 0:
        raise ValueError("step_ns must not be smaller or equal than 0")
    if start_ns > end_ns:
        raise ValueError("end_ns must be greater than or equal to start_ns")
        
    
    return np.arange(start_ns, end_ns, step_ns, dtype=np.int64)
    
    
    
def sample_previous(
    source_times_ns,
    source_values,
    grid_ns,
    max_age_ns,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Sample one PV on a target grid using bounded previous-value lookup.

    For each grid timestamp g, select the last source entry whose timestamp
    s satisfies s <= g. Accept its value only when it is finite and
    g - s <= max_age_ns. The age limit is inclusive and measures elapsed
    time, not how often a source measurement may be reused. Future source
    measurements are never used.

    Args:
        source_times_ns: One-dimensional int64 NumPy array of UTC Unix
            timestamps in nanoseconds, sorted in nondecreasing order.
            If timestamps are duplicated, the last entry at that time wins.
        source_values: One-dimensional float64 NumPy array with the same
            length as source_times_ns. Each value belongs to the timestamp
            at the same index. Non-finite measurement values are allowed.
        grid_ns: One-dimensional int64 NumPy array of target UTC Unix
            timestamps in nanoseconds. Its positions define output order.
        max_age_ns: Nonnegative Python or NumPy integer specifying the
            maximum source age in nanoseconds; Booleans are not accepted.
            Zero permits only measurements exactly at a grid timestamp.

    Returns:
        A tuple (sampled_values, valid_mask), both with shape grid_ns.shape.
        sampled_values has dtype float64; valid_mask has dtype bool.
        A valid position contains the selected source value and True.
        If there is no preceding source entry, the entry is too old, or
        its value is non-finite, the position contains NaN and False.
        A non-finite latest value does not trigger a search for an older
        finite value. An empty source produces all-invalid outputs; an
        empty grid produces empty outputs.

    Notes:
        The caller must restrict source measurements and target timestamps
        to the same train, validation, or test interval before this call.
        The sampler does not infer split boundaries or borrow measurements
        from another split. Input arrays and target timestamps are not
        modified, and invalid positions are retained rather than removed.
    """
    
    if grid_ns.dtype != np.dtype("int64"):
        raise ValueError("grid_ns must have dtype int64")
    if max_age_ns < 0:
        raise ValueError("max_age_ns must not be negative")
    if isinstance(max_age_ns, (bool, np.bool_)) or not isinstance(max_age_ns, (int, np.integer)):
        raise ValueError("max_age_ns must be an integer")
        
    # Dimensionality check
    if source_times_ns.ndim != 1:
        raise ValueError("source_times_ns must be one-dimensional")
    if source_values.ndim != 1:
        raise ValueError("source_times_ns must be one-dimensional")
    if grid_ns.ndim != 1:
        raise ValueError("grid_ns must be one-dimensional")
        
    if len(source_times_ns) != len(source_values):
        raise ValueError("source_times_ns and source_values must have the same length")
    if (np.diff(source_times_ns) < 0).any():
        raise ValueError("source_times_ns must be non-decreasing")
        
    if source_times_ns.dtype != np.dtype("int64"):
        raise ValueError("source_times_ns must have dtype int64")
    if grid_ns.dtype != np.dtype("int64"):
        raise ValueError("grid_ns must have dtype int64")
    if source_values.dtype != np.dtype("float64"):
        raise ValueError("source_values must have dtype float64")
    
    
    # Fill array of shape grid_ns with nan values
    sampled_values = np.full(grid_ns.shape, np.nan, dtype=np.float64)
    valid_mask = np.zeros(grid_ns.shape, dtype=bool)
    
    if source_times_ns.size == 0 or grid_ns.size == 0:
        return sampled_values, valid_mask
    
    indices = np.searchsorted(source_times_ns, grid_ns, side="right") - 1
    has_previous = indices >= 0
    source_indices = indices[has_previous]
    
    ages_ns = (grid_ns[has_previous] - source_times_ns[source_indices])
    candidate_values = source_values[source_indices]
    
    # Create boolean that satisfies joint conditions
    accepted = (ages_ns <= max_age_ns) & np.isfinite(candidate_values)
    
    # At the position selected by has_previous write values from accepted
    valid_mask[has_previous] = accepted
    sampled_values[valid_mask] = candidate_values[accepted]
    
    return sampled_values, valid_mask # Find values and mask for which (ages_ns <= max_age_ns) & np.isfinite(candidate_values) hold
