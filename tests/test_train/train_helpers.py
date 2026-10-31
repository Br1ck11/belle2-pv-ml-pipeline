"""Helpers: small synthetic dataset.h5 files built with the real belle2-dataset builder.

Using build_dataset (rather than hand-writing HDF5) checks the training
code against the dataset package's actual output contract.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dataset import build_dataset


ORIGIN_MS = 1_767_225_600_000  # 2026-01-01T00:00:00Z
DURATION_S = 1800


def _write_pv(path, pv, values):
    times = np.arange(values.size, dtype=np.float64)
    pq.write_table(pa.table({
        "pv": [pv], "t0": pa.array([ORIGIN_MS], type=pa.int64()),
        "times": pa.array([times.tolist()], type=pa.list_(pa.float64())),
        "values": pa.array([values.tolist()], type=pa.list_(pa.float64())),
    }), path)


def make_dataset(root: Path, *, n_channels, window=32, stride=8) -> Path:
    """Channels s0, s1, ... with stored PV identities DEMO:S0, DEMO:S1, ..."""
    raw = root / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    t = np.arange(DURATION_S, dtype=np.float64)
    ids = []
    for channel in range(n_channels):
        signal = np.sin(2 * np.pi * t / (40 + 13 * channel)) + 0.05 * rng.standard_normal(t.size)
        _write_pv(raw / f"s{channel}.parquet", f"DEMO:S{channel}", signal)
        ids.append(f"s{channel}")
    for ring in ("ler", "her"):
        _write_pv(raw / f"{ring}.parquet", f"DEMO:{ring.upper()}", np.full(t.size, 500.0))
    (root / "aborts.json").write_text(json.dumps([]), encoding="utf-8")
    pv_tables = "".join(
        f'\n[[pvs]]\nid = "{pv_id}"\nexpected_pv = "DEMO:{pv_id.upper()}"\n'
        f'filepaths = ["raw/{pv_id}.parquet"]\nmax_age_seconds = 1.0\n'
        for pv_id in (*ids, "ler", "her")
    )
    config = root / "dataset.toml"
    config.write_text(f"""
[dataset]
mode = "train_val"
start = "2026-01-01T00:00:00Z"
end = "2026-01-01T00:30:00Z"
pvs = {json.dumps(ids)}
sample_period_seconds = 1.0
window_size_samples = {window}
stride_samples = {stride}
train_fraction = 0.7
ler_pv = "ler"
her_pv = "her"
ler_threshold = 400.0
her_threshold = 400.0
aborts_filepath = "aborts.json"
exclude_before_seconds = 0.0
exclude_after_seconds = 0.0
norm_mode = "zscore"
batch_size = 64
output_dir = "built"
{pv_tables}""", encoding="utf-8")
    return build_dataset(config, progress=None)


UNIVARIATE_PVS = '[[pvs]]\npv = "DEMO:S0"\ntype = "Synthetic"\nname = "S0"\n'
MULTIVARIATE_PVS = '[[pvs]]\npv = "DEMO:S0"\n\n[[pvs]]\npv = "DEMO:S1"\n'
VAE_MODEL = ('name = "cnn_vae"\nhidden_channels = [8, 16]\nkernel_size = 5\n'
             'stride = 2\nlatent_dim = 4\nbeta = 1.0\n')


def write_train_config(directory: Path, dataset_path: Path, *, run_name="run", epochs=2,
                       pvs=UNIVARIATE_PVS, extra_training="", optimizer=None, model=VAE_MODEL) -> Path:
    """TOML inside ``directory`` with project_root = "." so outputs stay in the test folder."""
    optimizer = optimizer or 'name = "adam"\nlearning_rate = 1e-3\nweight_decay = 0.0\n'
    path = directory / "train.toml"
    path.write_text(f"""
[run]
name = "{run_name}"
seed = 7
device = "cpu"

[data]
dataset_path = "{dataset_path}"

{pvs}
[optimizer]
{optimizer}
[training]
epochs = {epochs}
batch_size = 16
{extra_training}
[model]
{model}""", encoding="utf-8")
    return path
