import numpy as np
import pandas as pd

from pathlib import Path
import hashlib
import json



# Validity check over split data s.t. sets contain only valid data points
def complete_window_mask(window_valid) -> np.ndarray:
    """Return one completeness decision per multivariate window.

    Args:
        window_valid: Boolean NumPy array with shape (N, C, T), where True
            marks a valid sampled point. C and T must be positive; N may
            be zero when no candidate windows were produced.

    Returns:
        A new Boolean array with shape (N,). A window is True only when
        every channel and time point is valid. An empty batch returns an
        empty array. The input is not modified.

    Raises:
        TypeError: If window_valid is not a NumPy array.
        ValueError: If the array is not three-dimensional or Boolean,
            or has no channels or time points.

    Notes:
        This rule applies to both train/validation and test windows.
        The caller applies the returned mask to window values and their
        matching metadata. Current thresholds and abort rules are separate.
    """
    if not isinstance(window_valid, np.ndarray):
        raise TypeError("window_valid must be a NumPy array")
    if window_valid.ndim != 3:
        raise ValueError("window_valid must have shape (N, C, T)")
    if window_valid.dtype != np.dtype(bool):
        raise ValueError("window_valid must have dtype bool")
    if window_valid.shape[1] == 0 or window_valid.shape[2] == 0:
        raise ValueError("Windows must contain at least one channel and one time point")

    # Reduce channels and time points, leaving one decision per window.
    return np.all(window_valid, axis=(1, 2))



class WindowFilter:
    """Store validated settings for multivariate window selection."""

    def __init__(
        self, mode, ler_threshold, her_threshold, aborts_filepath,
        *, exclude_before_ns, exclude_after_ns, pre_abort_ns, post_abort_ns,
    ):
        """Initialize the filtering mode, current thresholds, and time buffers.

        Args:
            mode: Either "train_val" or "test".
            ler_threshold: Finite Python or NumPy integer or floating-point
                scalar specifying the LER current threshold in mA.
            her_threshold: HER current threshold with the same requirements.
            aborts_filepath: Path to the abort catalog, stored as a Path.
                The file is opened when read_abort_records is called.
            exclude_before_ns: Normal-window exclusion duration before aborts.
            exclude_after_ns: Normal-window exclusion duration after aborts.
            pre_abort_ns: Test-window association duration before aborts.
            post_abort_ns: Test-window association duration after aborts.
                All durations must be nonnegative Python or NumPy integer
                scalars in nanoseconds. Zero is allowed; Booleans are not.

        Raises:
            TypeError: If a threshold or duration has an unsupported type,
                including Boolean, or aborts_filepath cannot form a Path.
            ValueError: If mode is unknown, a threshold is non-finite, or
                a duration is negative.
        """
        if not isinstance(mode, str) or mode not in ("train_val", "test"):
            raise ValueError("mode must be either 'train_val' or 'test'")

        for name, threshold in (
            ("ler_threshold", ler_threshold),
            ("her_threshold", her_threshold),
        ):
            if isinstance(threshold, (bool, np.bool_)) or not isinstance(
                threshold, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{name} must be a real numeric scalar, not a Boolean")
            # Integers are always finite; floating-point scalars can be NaN/inf.
            if isinstance(threshold, (float, np.floating)) and not np.isfinite(threshold):
                raise ValueError(f"{name} must be finite")

        for name, duration in (
            ("exclude_before_ns", exclude_before_ns),
            ("exclude_after_ns", exclude_after_ns),
            ("pre_abort_ns", pre_abort_ns),
            ("post_abort_ns", post_abort_ns),
        ):
            if isinstance(duration, (bool, np.bool_)) or not isinstance(
                duration, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer number of nanoseconds, not a Boolean")
            if duration < 0:
                raise ValueError(f"{name} must be nonnegative")

        self.mode = mode
        self.ler_threshold = ler_threshold
        self.her_threshold = her_threshold
        self.aborts_filepath = Path(aborts_filepath)
        self.exclude_before_ns = exclude_before_ns
        self.exclude_after_ns = exclude_after_ns
        self.pre_abort_ns = pre_abort_ns
        self.post_abort_ns = post_abort_ns
        self.events = None
        self.catalog_sha256 = None


    def read_abort_records(self):
        """Read the catalog and fingerprint the exact bytes being parsed."""
        contents = self.aborts_filepath.read_bytes()
        records = json.loads(contents)
        self.catalog_sha256 = hashlib.sha256(contents).hexdigest()
            
        if not isinstance(records, list):
            raise ValueError("abort json must contain a list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("abort json list must contain dictionaries")
        
        return records

    def load_events(self):
        """Cache chronological event identities and exact UTC nanoseconds.

        Naive catalog times mean Asia/Tokyo; explicit offsets are respected.
        Missing/blank categories become Unspecified, matching the reference.
        IDs use the reference identity fields (time, ring, category, trigger)
        and therefore remain stable across batches and catalog reordering.
        Duplicate identities are rejected rather than silently merged.
        """
        if self.events is not None:
            return self.events
        events, seen = [], set()
        for index, record in enumerate(self.read_abort_records()):
            text_timestamp = record.get("timestamp")
            if not isinstance(text_timestamp, str) or not text_timestamp.strip():
                raise ValueError(f"Abort record {index}: timestamp must be a nonempty string")
            try:
                timestamp = pd.Timestamp(text_timestamp)
                if pd.isna(timestamp):
                    raise ValueError("missing timestamp")
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize("Asia/Tokyo")
                timestamp_ns = int(timestamp.tz_convert("UTC").value)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"Abort record {index}: invalid timestamp {text_timestamp!r}") from error
            category = record.get("category")
            if category is not None and not isinstance(category, str):
                raise ValueError(f"Abort record {index}: category must be a string or null")
            category = (category or "").strip() or "Unspecified"
            ring = record.get("ring")
            if ring is not None and not isinstance(ring, str):
                raise ValueError(f"Abort record {index}: ring must be a string or null")
            ring = (ring or "").strip()
            ring = {"ler": "LER", "her": "HER", "both": "Both"}.get(ring.casefold(), ring)
            trigger = record.get("critical_trigger")
            if trigger is not None and not isinstance(trigger, str):
                raise ValueError(f"Abort record {index}: critical_trigger must be a string or null")
            trigger = (trigger or "").strip()
            identity = f"{timestamp_ns}|{ring}|{category}|{trigger}"
            event_id = "abort-" + hashlib.blake2s(identity.encode("utf-8"), digest_size=8).hexdigest()
            if event_id in seen:
                raise ValueError(f"Duplicate abort identity at catalog record {index}: {event_id}")
            seen.add(event_id)
            events.append({
                "event_id": event_id, "timestamp_ns": timestamp_ns, "category": category,
                "source_index": index, "source_timestamp": text_timestamp,
                "ring": ring, "critical_trigger": trigger,
            })
        self.events = tuple(sorted(events, key=lambda event: (event["timestamp_ns"], event["event_id"])))
        return self.events

    @staticmethod
    def _validate_window_times(window_start_ns, window_end_ns):
        for times in (window_start_ns, window_end_ns):
            if not isinstance(times, np.ndarray) or times.ndim != 1 or times.dtype != np.dtype("int64"):
                raise ValueError("Window timestamps must be one-dimensional int64 arrays")
        if window_start_ns.shape != window_end_ns.shape or np.any(window_end_ns < window_start_ns):
            raise ValueError("Window endpoints must have matching shapes and end >= start")

    def _event_masks(self, timestamp, window_start_ns, window_end_ns):
        """One shared definition of exclusion, containment, pre, and post."""
        timestamp = int(timestamp)
        # Clamp interval endpoints to the timestamp representation, avoiding
        # integer overflow when adding a configured duration near its limits.
        limits = np.iinfo(np.int64)
        before = max(limits.min, timestamp - int(self.exclude_before_ns))
        after = min(limits.max, timestamp + int(self.exclude_after_ns))
        pre_start = max(limits.min, timestamp - int(self.pre_abort_ns))
        post_end = min(limits.max, timestamp + int(self.post_abort_ns))
        overlap = (window_start_ns <= after) & (window_end_ns >= before)
        contains = (window_start_ns <= timestamp) & (window_end_ns >= timestamp)
        pre = (window_end_ns >= pre_start) & (window_end_ns < timestamp)
        post = (window_start_ns > timestamp) & (window_end_ns <= post_end)
        return overlap, contains, pre, post

    def label_windows(self, window_start_ns, window_end_ns):
        """Label RETAINED windows and return aligned event-association lists.

        A normal window has ["non_abort"] and no associations. Test windows
        may have multiple categories/events: category x for containment,
        pre_abort_x before, post_abort_x after. A window gets at most one
        relation to each event, but every matching event is preserved.
        Label strings are unique per window; event IDs disambiguate equal
        categories. These are descriptive labels, not future prediction
        targets. Apply filtering before calling this method.
        """
        self._validate_window_times(window_start_ns, window_end_ns)
        labels = [[] for _ in window_start_ns]
        associations = [[] for _ in window_start_ns]
        if self.mode == "test":
            for event in self.load_events():
                _, contains, pre, post = self._event_masks(
                    event["timestamp_ns"], window_start_ns, window_end_ns,
                )
                category = event["category"]
                for mask, label in ((contains, category), (pre, f"pre_abort_{category}"),
                                    (post, f"post_abort_{category}")):
                    for index in np.flatnonzero(mask):
                        if label not in labels[index]:
                            labels[index].append(label)
                        associations[index].append({**event, "label": label})
        for row in labels:
            if not row:
                row.append("non_abort")
        return labels, associations

    def filter_pv_windows(self, complete_mask, ler_windows, her_windows, window_start_ns, window_end_ns):
        """Return one keep decision per aligned window, according to mode.

        Args:
            complete_mask: Boolean array (N,) covering all required model
                channels and both currents. False always rejects a window.
            ler_windows: LER currents in mA, shaped (N, T).
            her_windows: HER currents in mA, shaped (N, T).
            window_start_ns: First sample timestamps, int64 UTC nanoseconds
                with shape (N,).
            window_end_ns: Last sample timestamps, with the same units and
                shape as window_start_ns. Each end must be at or after its corresponding start.

        Returns:
            A Boolean array (N,), in input order. Train/validation windows
            must be complete, exceed both current thresholds throughout,
            and avoid the inclusive abort-exclusion intervals.

            Test also keeps complete event-related windows regardless of
            current thresholds. A window contains an abort when the event
            is between its inclusive sample endpoints. Pre-abort windows
            end in [event - pre_abort_ns, event); post-abort windows start
            strictly after the event and end no later than
            event + post_abort_ns. Containing an abort is sufficient even
            when the window ends beyond the post-abort interval.

        Notes:
            Inputs are assumed to have matching window order and shapes,
            with T positive and N possibly zero. They are not modified.
            The caller applies the mask to data and matching metadata.
            Labels and event associations are not returned by this method.
            The catalog is parsed once per instance and cached.
        """
        self._validate_window_times(window_start_ns, window_end_ns)
        if not isinstance(complete_mask, np.ndarray) or complete_mask.dtype != np.dtype(bool) or complete_mask.shape != window_start_ns.shape:
            raise ValueError("complete_mask must be Boolean with one entry per window")
        for currents in (ler_windows, her_windows):
            if not isinstance(currents, np.ndarray) or currents.ndim != 2 or currents.shape[0] != len(complete_mask) or currents.shape[1] == 0:
                raise ValueError("Current windows must have shape (N, T) with T positive")
        if ler_windows.shape != her_windows.shape:
            raise ValueError("LER and HER windows must have identical shapes")

        # One decision per window, initially no overlaps or associations.
        abort_overlap = np.zeros(complete_mask.shape, dtype=bool)
        event_related = np.zeros(complete_mask.shape, dtype=bool)
        
        for event in self.load_events():
            overlaps, contains, pre, post = self._event_masks(
                event["timestamp_ns"], window_start_ns, window_end_ns,
            )
            abort_overlap |= overlaps
            if self.mode == "test":
                event_related |= contains | pre | post
            
        abort_free = ~abort_overlap
        
        # Require every time point in a window to pass
        ler_ok = np.all(ler_windows > self.ler_threshold, axis=1)
        her_ok = np.all(her_windows > self.her_threshold, axis=1)
        current_ok = ler_ok & her_ok
        
        if self.mode == "train_val":
            keep = complete_mask & abort_free & current_ok
            return keep
        elif self.mode == "test":
            normal_ok = abort_free & current_ok
            # Completeness is required for both normal and event-related windows.
            keep = complete_mask & (normal_ok | event_related)
            return keep


def training_coverage_mask(kept_start_indices, window_size, grid_size) -> np.ndarray:
    """Mark training-grid points covered by at least one accepted window.

    Args:
        kept_start_indices: One-dimensional NumPy array with an integer
            dtype. Entries are positions in the original training grid,
            not timestamps. Repeated or unordered starts are allowed.
        window_size: Positive Python or NumPy integer giving the number
            of samples in each window. Booleans are not accepted.
        grid_size: Nonnegative Python or NumPy integer giving the number
            of grid points. Booleans are not accepted.

    Returns:
        A new Boolean array of shape (grid_size,). A start s marks the
        half-open slice [s, s + window_size). Overlaps remain True, so a
        point contributes only once when this mask selects training data
        for normalization. The input array is not modified.

        Empty starts return an all-False mask, including an empty mask
        when grid_size is zero. No window has to fit if none were kept.

    Raises:
        TypeError: If starts are not a NumPy array, or either size is not
            an integer scalar or is a Boolean.
        ValueError: If starts are not one-dimensional with integer dtype,
            window_size is nonpositive, grid_size is negative, or any
            accepted window extends outside the grid.
    """
    if not isinstance(kept_start_indices, np.ndarray):
        raise TypeError("kept_start_indices must be a NumPy array")
    if kept_start_indices.ndim != 1:
        raise ValueError("kept_start_indices must be one-dimensional")
    if not np.issubdtype(kept_start_indices.dtype, np.integer):
        raise ValueError("kept_start_indices must have an integer dtype")

    for name, size in (("window_size", window_size), ("grid_size", grid_size)):
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)):
            raise TypeError(f"{name} must be an integer scalar, not a Boolean")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if grid_size < 0:
        raise ValueError("grid_size must be nonnegative")

    window_size = int(window_size)
    grid_size = int(grid_size)
    # Initialise all False boolean array with length of grid_size.
    coverage_mask = np.zeros(grid_size, dtype=bool)

    for start in kept_start_indices:
        # Python integer arithmetic prevents NumPy integer overflow before
        # checking the endpoint. Slicing alone would silently clip it.
        start = int(start)
        end = start + window_size
        if start < 0 or end > grid_size:
            raise ValueError(
                f"Window [{start}, {end}) falls outside grid [0, {grid_size})"
            )
        coverage_mask[start:end] = True

    return coverage_mask
