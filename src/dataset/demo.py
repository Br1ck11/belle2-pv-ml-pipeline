"""Create a small synthetic archive and build both dataset modes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .builder import build_dataset


ORIGIN_NS = 1_767_225_600_000_000_000  # 2026-01-01T00:00:00Z


def write_demo_inputs(output_dir):
    """Create new synthetic inputs and two complete TOMLs; do not build yet.

    Includes overlapping windows, two files per channel, a missing boundary
    measurement, a short fillable gap, invalid data and low currents. Values
    are intentionally simple enough to calculate expected statistics by hand.
    Returns (training_config, test_config). Existing directories are refused.
    """
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    raw = root / "raw"
    raw.mkdir()
    for channel in ("signal", "constant", "ler", "her"):
        for file_index, (lower, upper) in enumerate(((0, 12), (12, 124))):
            times = np.array([i for i in range(lower, upper) if i < 24 or i >= 99], dtype=np.float64)
            if channel == "signal":
                times = times[~np.isin(times, [12, 17, 100])]
                values = times ** 2
                values[np.isin(times, [6, 113])] = np.nan
            elif channel == "constant":
                values = np.full(times.size, 7.)
            else:
                values = np.full(times.size, 500.)
                if channel == "her":
                    values[(times == 8) | ((times >= 103) & (times <= 110))] = 100.
                else:
                    values[times == 22] = np.nan
            table = pa.table({
                "pv": [f"DEMO:{channel}"],
                "t0": pa.array([ORIGIN_NS // 1_000_000 + lower * 1000], type=pa.int64()),
                "times": pa.array([(times - lower).tolist()], type=pa.list_(pa.float64())),
                "values": pa.array([values.tolist()], type=pa.list_(pa.float64())),
            })
            pq.write_table(table, raw / f"{channel}_{file_index}.parquet")
    events = [
        {"timestamp": "2026-01-01 09:00:10", "category": "Manual", "ring": "Both"},
        {"timestamp": "2026-01-01 09:01:45.500000000", "category": "SBL", "ring": "LER"},
        {"timestamp": "2026-01-01T00:01:48Z", "category": "RF", "ring": "HER"},
    ]
    (root / "aborts.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    paths = []
    for mode in ("train_val", "test"):
        text = f'''# Entirely synthetic inputs; suitable for checking the installation.
[dataset]
mode = "{mode}"
start = "2026-01-01T00:{'00:00' if mode == 'train_val' else '01:40'}Z"
end = "2026-01-01T00:{'00:24' if mode == 'train_val' else '02:04'}Z"
pvs = ["constant", "signal"]
sample_period_seconds = 1.0
window_size_samples = 3
stride_samples = {2 if mode == 'train_val' else 1}
train_fraction = 0.5
ler_pv = "ler"
her_pv = "her"
ler_threshold = 400.0
her_threshold = 400.0
aborts_filepath = "aborts.json"
exclude_before_seconds = 0.0
exclude_after_seconds = 0.0
pre_abort_seconds = 4.0
post_abort_seconds = 4.0
norm_mode = "zscore"
batch_size = 2
output_dir = "{mode}"
'''
        if mode == "test":
            text += 'statistics_filepath = "train_val/dataset.h5"\n'
        for channel in ("signal", "constant", "ler", "her"):
            text += f'''
[[pvs]]
id = "{channel}"
expected_pv = "DEMO:{channel}"
filepaths = ["raw/{channel}_0.parquet", "raw/{channel}_1.parquet"]
max_age_seconds = 1.0
'''
        config_path = root / f"{mode}.toml"
        config_path.write_text(text, encoding="utf-8")
        paths.append(config_path)
    return tuple(paths)


def main(argv=None):
    """Generate synthetic inputs and both completed datasets in a new folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    args = parser.parse_args(argv)
    try:
        for config_path in write_demo_inputs(args.output):
            build_dataset(config_path)
    except (ValueError, TypeError, OSError) as error:
        parser.exit(1, f"Demo failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
