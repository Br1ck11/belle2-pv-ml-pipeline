"""State for fitting and applying per-channel window normalization."""

import numpy as np


def log10_values(values):
    """Return (log10(values), number of finite non-positive values).

    Optional value transform applied to raw measurements BEFORE sampling,
    statistics and normalization (dataset.log_transform = true), so that
    statistics describe log10 values and windows spanning several orders of
    magnitude remain distinguishable.

    The logarithm is undefined for values <= 0: these become NaN and are
    counted. NaN stays NaN and +inf stays +inf. Every non-finite result is
    later treated like any invalid measurement (a latest non-finite value
    invalidates the grid point; incomplete windows are rejected). The input
    is not modified; a new float64 array is returned.
    """
    values = np.asarray(values, dtype=np.float64)
    nonpositive = np.isfinite(values) & (values <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log10(values)
    result[nonpositive] = np.nan
    return result, int(nonpositive.sum())


class NormalizeWindows:
    """Hold normalization settings and statistics across training batches."""

    def __init__(
        self, norm_mode, *, n_channels,
    ):
        """Initialize mode-specific statistics and normalization state.

        Args:
            norm_mode: Explicitly selected "minmax", "zscore", or "none".
                Python None and unknown strings are not supported modes.
            n_channels: Positive Python or NumPy integer giving the channel
                count C. Booleans are not accepted. Future statistics
                batches have shape (C, M); windows have shape (N, C, T).

        State:
            count starts at zero and will count unique accepted training
            grid points per channel. A single count is sufficient because
            each statistics batch must contain complete data for every
            channel. The caller must prevent duplication across batches.

            Min-max allocates float64 minimum/maximum arrays of shape (C,)
            initialized to positive/negative infinity. Z-score instead
            allocates zero mean and m2 arrays; m2 is the accumulated sum of
            squared deviations from the mean. Unused statistics are None.

            offset, raw_scale, scale, and scale_fallback_mask are None
            until finalization. The raw scale must retain a zero standard
            deviation or range for a constant channel; its effective scale
            becomes one. Metadata must save constant_channel_policy,
            raw_scale, scale, and the per-channel scale_fallback_mask so
            this adjustment is explicit.

            Mode "none" needs no fitting: offset is zero, scale is one,
            and finalized is True immediately. raw_scale remains None
            because no scale was estimated, and no fallback is applied.
            apply_normalization returns a copy with unchanged values and
            dtype in this mode. Other modes start with finalized False. 
            finalized is a boolean specifying whether the statistic accumulation
            of the train set is finished (-> finalized = True) or is unfinished
            (-> finalized = False).

        Raises:
            ValueError: If norm_mode is unsupported or n_channels is not
                positive.
            TypeError: If n_channels is not an integer scalar or is a
                Boolean, or a required argument is omitted.

        Notes:
            This constructor allocates state only. Batch updates,
            finalization, transformation, and metadata export are separate
            steps and are not performed here.
        """
        if not isinstance(norm_mode, str) or norm_mode not in ("minmax", "zscore", "none"):
            raise ValueError("norm_mode must be 'minmax', 'zscore', or 'none'")
        if isinstance(n_channels, (bool, np.bool_)) or not isinstance(n_channels, (int, np.integer)):
            raise TypeError("n_channels must be an integer scalar, not a Boolean")
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")

        self.norm_mode = norm_mode
        self.n_channels = int(n_channels)
        self.count = 0
        self.finalized = False

        # None means that this statistic is not computed for the chosen mode.
        self.minimum = None
        self.maximum = None
        self.mean = None
        self.m2 = None

        # Finalization will populate these parameters and metadata fields.
        self.offset = None
        self.raw_scale = None
        self.scale = None
        self.scale_fallback_mask = None
        self.constant_channel_policy = "scale_one"

        if self.norm_mode == "minmax":
            self.minimum = np.full(self.n_channels, np.inf, dtype=np.float64)
            self.maximum = np.full(self.n_channels, -np.inf, dtype=np.float64)
        elif self.norm_mode == "zscore":
            self.mean = np.zeros(self.n_channels, dtype=np.float64)
            self.m2 = np.zeros(self.n_channels, dtype=np.float64)
        else:  # Explicit "none" mode has no statistics to fit.
            self.offset = np.zeros(self.n_channels, dtype=np.float64)
            self.scale = np.ones(self.n_channels, dtype=np.float64)
            self.scale_fallback_mask = np.zeros(self.n_channels, dtype=bool)
            self.finalized = True

    def _validate_values(self, values, *, ndim, name):
        """Check a real array whose channel axis is second from last."""
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if values.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions")
        if values.shape[-2] != self.n_channels:
            raise ValueError(f"{name} must contain {self.n_channels} channels")
        if values.dtype.kind not in "iuf":
            raise ValueError(f"{name} must have a real integer or floating-point dtype")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")

    def streaming_Welford(self, batch):
        """Merge a (C, M) training batch into count, mean, and m2.

        This is the batch form of Welford accumulation. The batch mean and
        squared deviations are calculated with NumPy, then merged with the
        previous state. No earlier samples are retained. m2 is the sum of
        squared deviations, not the variance or standard deviation.

        batch must be a finite, real NumPy array in the agreed channel
        order. Statistics are computed in float64. M may be zero, which
        leaves the state unchanged. Each selected training-grid point must
        appear only once across all batches; timestamps are not available
        here to detect duplicates or validation/test data.

        calculate_statistics calls this method for z-score mode. Do not
        also call it directly for the same batch, which would count twice.
        Returns None; the updated statistics live on this instance.

        Raises:
            TypeError: If batch is not a NumPy array.
            ValueError: If this is not z-score mode, the batch is invalid,
                or the statistics exceed the float64 range.
            RuntimeError: If statistics have already been finalized.
        """
        if self.norm_mode != "zscore":
            raise ValueError("streaming_Welford requires zscore mode")
        if self.finalized:
            raise RuntimeError("Cannot update finalized normalization statistics")
        self._validate_values(batch, ndim=2, name="batch")
        batch_count = batch.shape[1]
        if batch_count == 0:
            return

        # Compute locally before assigning state, so a failed calculation
        # does not leave partially updated statistics on the instance.
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                batch = batch.astype(np.float64, copy=False)
                # Center before summation: this reduces cancellation for
                # large offsets and keeps exactly constant channels at m2=0,
                # even for values such as 0.1 that are inexact in binary.
                origin = batch[:, 0]
                centered = batch - origin[:, None]
                mean_shift = np.mean(centered, axis=1)
                batch_mean = origin + mean_shift
                centered -= mean_shift[:, None]
                batch_m2 = np.sum(centered ** 2, axis=1)
                total_count = self.count + batch_count
                if self.count == 0:
                    mean = batch_mean
                    m2 = batch_m2
                else:
                    delta = batch_mean - self.mean
                    mean = self.mean + delta * (batch_count / total_count)
                    # The correction accounts for the two groups having
                    # different means; adding their m2 values alone is wrong.
                    m2 = (
                        self.m2 + batch_m2
                        + delta ** 2 * (self.count / total_count) * batch_count
                    )
        except FloatingPointError as error:
            raise ValueError("Welford statistics exceed the float64 range") from error

        self.count = total_count
        self.mean = mean
        self.m2 = m2

    def calculate_statistics(self, batch):
        """Accumulate one batch of unique accepted training points (C, M).

        Supply disjoint batches selected using the training coverage mask,
        keeping channel order fixed. Do not flatten overlapping windows:
        that would count shared timestamps more than once. This method
        cannot identify duplicate timestamps or the split a value came from.

        Min-max updates only minimum, maximum, and count. Z-score delegates
        to streaming_Welford. Explicit "none" mode validates the batch but
        computes nothing. Empty batches are allowed. Inputs are unchanged;
        accumulation uses float64 and retains no batch data.

        Returns None. Call repeatedly, then finalize once all training
        batches have been processed. Further updates to fitted modes raise
        RuntimeError after finalization. Invalid arrays raise TypeError or
        ValueError; arrays must be real, finite, and have C matching channels.
        """
        if self.norm_mode == "zscore":
            self.streaming_Welford(batch)
            return

        if self.norm_mode == "minmax" and self.finalized:
            raise RuntimeError("Cannot update finalized normalization statistics")
            
        self._validate_values(batch, ndim=2, name="batch")
        if self.norm_mode == "none" or batch.shape[1] == 0:
            return

        # The constructor has already validated the mode: this is min-max.
        try:
            with np.errstate(over="raise", invalid="raise"):
                batch = batch.astype(np.float64, copy=False)
        except FloatingPointError as error:
            raise ValueError("Batch values exceed the float64 range") from error
        # Save the minmum / maximum between the i-th batch and the i+1-th batch to find the global min / max
        self.minimum = np.minimum(self.minimum, np.min(batch, axis=1))
        self.maximum = np.maximum(self.maximum, np.max(batch, axis=1))
        self.count += batch.shape[1]

    def finalize(self):
        """Freeze offset and scale after the last training batch.

        Min-max uses minimum and maximum - minimum. Z-score uses mean and
        the population standard deviation sqrt(m2 / count), i.e. ddof=0.
        Exactly zero raw scales are recorded and replaced with effective
        scale 1. Positive scales are unchanged; no epsilon is added.

        Returns None. Repeated calls, including in "none" mode, do nothing.
        Raises ValueError if a fitted mode has no points or its parameters
        cannot be represented as finite float64 values.
        """
        if self.finalized:
            return
        if self.count == 0:
            raise ValueError("Cannot finalize without accepted training points")

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if self.norm_mode == "minmax":
                    offset = self.minimum.copy()
                    raw_scale = self.maximum - self.minimum
                else:  # Z-score; none is finalized in the constructor.
                    offset = self.mean.copy()
                    raw_scale = np.sqrt(self.m2 / self.count)
        except FloatingPointError as error:
            raise ValueError("Normalization parameters exceed the float64 range") from error
        if not np.isfinite(offset).all() or not np.isfinite(raw_scale).all() or np.any(raw_scale < 0):
            raise ValueError("Normalization parameters must be finite with nonnegative raw scales")

        self.offset = offset
        self.raw_scale = raw_scale
        self.scale_fallback_mask = raw_scale == 0
        self.scale = np.where(self.scale_fallback_mask, 1., raw_scale)
        self.finalized = True

    def apply_normalization(self, windows):
        """Apply frozen per-channel parameters to finite windows (N, C, T).

        Channel order must match the statistics batches. Both fitted modes
        return a new float64 array using (windows - offset) / scale, with
        no clipping. "none" returns a copy preserving the input dtype.
        N may be zero; T must be positive. Inputs and statistics are never
        modified, so the same instance can transform train, val, and test.

        Raises:
            RuntimeError: If the parameters have not been finalized.
            TypeError: If windows is not a NumPy array.
            ValueError: If shape, channels, dtype, or finiteness is invalid,
                or the result exceeds the float64 range.
        """
        if not self.finalized:
            raise RuntimeError("Finalize statistics before applying normalization")
        self._validate_values(windows, ndim=3, name="windows")
        if windows.shape[2] == 0:
            raise ValueError("windows must have a positive time dimension")
        if self.norm_mode == "none":
            return windows.copy()

        # (C,) -> (1, C, 1): one parameter per channel, shared by all
        # windows and all time points through NumPy broadcasting.
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = windows.astype(np.float64, copy=True)
                result -= self.offset[None, :, None]
                result /= self.scale[None, :, None]
        except FloatingPointError as error:
            raise ValueError("Normalized values exceed the float64 range") from error
        return result

    def to_metadata(self):
        """Return a JSON-serializable copy of finalized normalization state.

        The dataset writer must save this dictionary alongside the channel
        identities/order and training-split provenance. This method does
        not write a file. The caller is responsible for fitting only on
        unique accepted training-grid points, as recorded by the policy.

        Uncomputed statistics are None. raw_scale preserves zero ranges or
        standard deviations; scale and scale_fallback_mask explicitly show
        the scale-one fallback. No statistics are fitted in "none" mode.
        Raises RuntimeError if parameters have not been finalized.
        """
        if not self.finalized:
            raise RuntimeError("Finalize statistics before exporting metadata")
        return {
            "norm_mode": self.norm_mode,
            "n_channels": self.n_channels,
            "fitted_on": None if self.norm_mode == "none" else "train",
            "weighting": (
                None if self.norm_mode == "none"
                else "unique accepted regular-grid timestamps per channel"
            ),
            "count": self.count,
            "minimum": None if self.minimum is None else self.minimum.tolist(),
            "maximum": None if self.maximum is None else self.maximum.tolist(),
            "mean": None if self.mean is None else self.mean.tolist(),
            "m2": None if self.m2 is None else self.m2.tolist(),
            "variance_ddof": 0 if self.norm_mode == "zscore" else None,
            "offset": self.offset.tolist(),
            "raw_scale": None if self.raw_scale is None else self.raw_scale.tolist(),
            "scale": self.scale.tolist(),
            "scale_fallback_mask": self.scale_fallback_mask.tolist(),
            "constant_channel_policy": self.constant_channel_policy,
        }
