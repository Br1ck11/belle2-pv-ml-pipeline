"""Read one PV over [start, end) with the src/dataset machinery.

Paths: ``dataset.parse_config.resolve_pv_filepaths`` (files or directories,
sorted immediate *.parquet, duplicates rejected). Measurements:
``dataset.read_pv.read_pv_files`` (PV identity check, t0 ms + relative s ->
int64 UTC ns, stable chronological sort).

To avoid reading a whole run period for a one-day plot, files are first
preselected by their Parquet column statistics: the stored ``t0`` plus the
min/max of ``times`` give each file's exact time coverage from the file's
contents (not its name). Files without statistics are always read. The
interval is then applied exactly to the measured timestamps.
"""

from dataclasses import dataclass, field

import numpy as np
import pyarrow.parquet as pq

from dataset.parse_config import resolve_pv_filepaths
from dataset.read_pv import read_pv_files

from .common import file_fingerprint


@dataclass(frozen=True)
class PVData:
    pv: str
    times_ns: np.ndarray               # int64 UTC ns, nondecreasing
    values: np.ndarray                 # float64, non-finite values retained
    files: list = field(default_factory=list)   # fingerprints of the files read


def file_time_range_ns(path):
    """(first_ns, last_ns) from Parquet statistics, or None if unavailable."""
    metadata = pq.ParquetFile(path).metadata
    paths = [metadata.schema.column(i).path for i in range(metadata.num_columns)]
    try:
        t0_index, times_index = paths.index("t0"), paths.index("times.list.element")
    except ValueError:
        return None
    lows, highs = [], []
    for group in range(metadata.num_row_groups):
        t0_stats = metadata.row_group(group).column(t0_index).statistics
        time_stats = metadata.row_group(group).column(times_index).statistics
        if t0_stats is None or time_stats is None or not (t0_stats.has_min_max and time_stats.has_min_max):
            return None
        if t0_stats.min != t0_stats.max:
            return None
        t0_ns = int(t0_stats.min) * 1_000_000
        lows.append(t0_ns + int(np.floor(time_stats.min * 1e9)))
        highs.append(t0_ns + int(np.ceil(time_stats.max * 1e9)))
    return (min(lows), max(highs)) if lows else None


def overlapping_files(filepaths, start_ns, end_ns, *, lookback_ns=0):
    """Files whose statistics overlap [start_ns - lookback_ns, end_ns)."""
    selected = []
    for path in filepaths:
        span = file_time_range_ns(path)
        if span is None or (span[1] >= start_ns - lookback_ns and span[0] < end_ns):
            selected.append(path)
    return selected


def load_pv(pv, filepaths, root, start_ns, end_ns, *, lookback_ns=0) -> PVData:
    """Measurements of ``pv`` with start_ns - lookback_ns <= t < end_ns."""
    if end_ns <= start_ns:
        raise ValueError("end must be later than start")
    resolved = resolve_pv_filepaths(filepaths, root, pv_id=pv)
    files = overlapping_files(resolved, start_ns, end_ns, lookback_ns=lookback_ns)
    times, values = read_pv_files(files, pv)
    left, right = np.searchsorted(times, [start_ns - lookback_ns, end_ns], side="left")
    return PVData(pv=pv, times_ns=times[left:right], values=values[left:right],
                  files=[file_fingerprint(path) for path in files])
