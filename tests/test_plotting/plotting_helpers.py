"""Overview TOMLs are written into a per-test directory with absolute archive
paths, so each test gets its own PVPlots/ while sharing one read-only archive.

Synthetic two-day archive (daily files like the real converter) and TOML writers.

Coverage: 2026-01-01T23:00Z - 2026-01-02T01:00Z at 1 Hz, split into one file
per UTC day. Currents are 500 mA except 0 mA for 60 s after the BeamLoss
abort at 00:20Z. Two pressure-like PVs sit at locations D01_L01 and D02_H05.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dataset import build_dataset


T0_MS = int(pd.Timestamp("2024-01-01T00:00:00+09:00").value // 1_000_000)   # common epoch, as in the archive
START = pd.Timestamp("2026-01-01T23:00:00Z")
END = pd.Timestamp("2026-01-02T01:00:00Z")
MIDNIGHT = pd.Timestamp("2026-01-02T00:00:00Z")
PVS = {"ler": "SYN:LER:CURRENT", "her": "SYN:HER:CURRENT", "a": "SYN:D01_L01:PRES", "b": "SYN:D02_H05:PRES"}
EVENTS = [
    {"timestamp": "2026-01-01T23:30:00Z", "category": "Manual", "ring": "HER", "pressure_burst": ""},
    {"timestamp": "2026-01-02 09:10:00", "category": "SBL", "ring": "LER",          # naive -> Tokyo = 00:10Z
     "pressure_burst": "D01_L01,taper/GV(D02_H05)"},
    {"timestamp": "2026-01-02T00:20:00Z", "category": "BeamLoss", "ring": "Both", "pressure_burst": ",D02_H05,"},
    {"timestamp": "2026-01-03T00:00:00Z", "category": "SBL", "ring": "LER", "pressure_burst": "D01_L01"},
]


def write_pv_file(path, pv, times_ns, values, *, statistics=True):
    relative = (np.asarray(times_ns, dtype=np.int64) - T0_MS * 1_000_000) / 1e9
    pq.write_table(pa.table({
        "pv": [pv], "t0": pa.array([T0_MS], type=pa.int64()),
        "times": pa.array([relative.tolist()], type=pa.list_(pa.float64())),
        "values": pa.array([np.asarray(values, dtype=np.float64).tolist()], type=pa.list_(pa.float64())),
    }), path, write_statistics=statistics)


def signal(key, t_seconds):
    if key in ("ler", "her"):
        current = np.full(t_seconds.size, 500.0)
        abort = (MIDNIGHT + pd.Timedelta(minutes=20)).value / 1e9
        current[(t_seconds >= abort) & (t_seconds < abort + 60)] = 0.0
        return current
    phase = 0.0 if key == "a" else 1.0
    return 1e-7 * (2 + np.sin(2 * np.pi * t_seconds / 300 + phase))


def make_archive(root: Path) -> Path:
    """Write root/archive/<key>/<KEY>-YYYY_MM_DD.parquet and root/aborts.json."""
    grid = np.arange(START.value, END.value, 1_000_000_000, dtype=np.int64)
    for key, pv in PVS.items():
        directory = root / "archive" / key
        directory.mkdir(parents=True)
        for day_start, day_end in ((START.value, MIDNIGHT.value), (MIDNIGHT.value, END.value)):
            times = grid[(grid >= day_start) & (grid < day_end)]
            day = pd.Timestamp(day_start, tz="UTC").strftime("%Y_%m_%d")
            write_pv_file(directory / f"{key.upper()}-{day}.parquet", pv, times, signal(key, times / 1e9))
    (root / "aborts.json").write_text(json.dumps(EVENTS), encoding="utf-8")
    return root


def overview_toml(directory: Path, archive_root: Path, *, name="ov", start="2026-01-01T23:00:00Z", end="2026-01-02T01:00:00Z",
                  categories="[]", pvs=("a", "b"), currents=True, extra_plot="", units=None) -> Path:
    units = units or {key: "unknown" for key in pvs}
    text = f"""
[plot]
name = "{name}"
start = "{start}"
end = "{end}"
summary_period = "10s"
dpi = 60
{extra_plot}
[aborts]
filepath = "{archive_root / 'aborts.json'}"
categories = {categories}
"""
    if currents:
        text += f"""
[currents]
ler = {{ pv = "{PVS['ler']}", filepaths = ["{archive_root / 'archive/ler'}"] }}
her = {{ pv = "{PVS['her']}", filepaths = ["{archive_root / 'archive/her'}"] }}
"""
    for key in pvs:
        text += (f'\n[[pvs]]\npv = "{PVS[key]}"\nfilepaths = ["{archive_root / "archive" / key}"]\n'
                 f'unit = "{units[key]}"\n')
    path = directory / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def build(root: Path, *, mode, norm_mode, output, statistics=None, log_transform=False) -> Path:
    """dataset.h5 over the archive with T=30 s windows, stride 10 s, channels a, b."""
    tables = "".join(f'\n[[pvs]]\nid = "{key}"\nexpected_pv = "{pv}"\nfilepaths = ["archive/{key}"]\n'
                     "max_age_seconds = 1.0\n" for key, pv in PVS.items())
    extra = 'train_fraction = 0.5\n' if mode == "train_val" else 'pre_abort_seconds = 60.0\npost_abort_seconds = 60.0\n'
    if statistics:
        extra += f'statistics_filepath = "{statistics}"\n'
    extra += f"log_transform = {str(log_transform).lower()}\n"
    config = root / f"{output}.toml"
    config.write_text(f"""
[dataset]
mode = "{mode}"
start = "{START.isoformat()}"
end = "{END.isoformat()}"
pvs = ["a", "b"]
sample_period_seconds = 1.0
window_size_samples = 30
stride_samples = 10
ler_pv = "ler"
her_pv = "her"
ler_threshold = 400.0
her_threshold = 400.0
aborts_filepath = "aborts.json"
exclude_before_seconds = 60.0
exclude_after_seconds = 60.0
norm_mode = "{norm_mode}"
batch_size = 64
output_dir = "{output}"
{extra}{tables}""", encoding="utf-8")
    return build_dataset(config, progress=None)
