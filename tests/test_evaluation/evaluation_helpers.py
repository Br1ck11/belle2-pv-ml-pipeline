"""Synthetic archive, train/test datasets and a tiny trained CNN VAE for the evaluation tests.

Coverage 2026-01-01T00:00Z - 02:00Z at 1 Hz. Training uses 00:00-01:00
(no aborts), testing 01:00-02:00 with four aborts. Currents drop to 0 mA
for 60 s after each abort.
"""

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dataset import build_dataset
from train import train_model


T0 = pd.Timestamp("2026-01-01T00:00:00Z")
PV = "SYN:D01_L01:PRES"
ABORTS = [("01:10:00", "SBL", "LER"), ("01:25:00", "BeamLoss", "Both"),
          ("01:40:00", "SBL", "HER"), ("01:50:00", "Injection", "LER")]


def _write(path, pv, values):
    pq.write_table(pa.table({
        "pv": [pv], "t0": pa.array([T0.value // 1_000_000], type=pa.int64()),
        "times": pa.array([np.arange(values.size, dtype=np.float64).tolist()], type=pa.list_(pa.float64())),
        "values": pa.array([values.tolist()], type=pa.list_(pa.float64())),
    }), path)


def make_archive(root: Path) -> Path:
    t = np.arange(7200, dtype=np.float64)
    rng = np.random.default_rng(3)
    pressure = 1e-7 * (2 + np.sin(2 * np.pi * t / 240)) + 2e-9 * rng.standard_normal(t.size)
    current = np.full(t.size, 500.0)
    for clock, _, _ in ABORTS:
        start = int((pd.Timestamp(f"2026-01-01T{clock}Z") - T0).total_seconds())
        current[start:start + 60] = 0.0
        pressure[start - 30:start] *= 3            # a pressure rise before each abort
    raw = root / "raw"
    raw.mkdir(parents=True)
    _write(raw / "p.parquet", PV, pressure)
    _write(raw / "ler.parquet", "SYN:LER", current)
    _write(raw / "her.parquet", "SYN:HER", current)
    (root / "aborts.json").write_text(json.dumps([
        {"timestamp": f"2026-01-01T{clock}Z", "category": category, "ring": ring} for clock, category, ring in ABORTS
    ]), encoding="utf-8")
    return root


def build(root: Path, name, *, mode, start, end, window=32, stride=4, statistics=None):
    extra = ("train_fraction = 0.7\n" if mode == "train_val"
             else f'pre_abort_seconds = 60.0\npost_abort_seconds = 90.0\nstatistics_filepath = "{statistics}"\n')
    config = root / f"{name}.toml"
    config.write_text(f"""
[dataset]
mode = "{mode}"
start = "2026-01-01T{start}Z"
end = "2026-01-01T{end}Z"
pvs = ["p"]
sample_period_seconds = 1.0
window_size_samples = {window}
stride_samples = {stride}
ler_pv = "ler"
her_pv = "her"
ler_threshold = 400.0
her_threshold = 400.0
aborts_filepath = "aborts.json"
exclude_before_seconds = 120.0
exclude_after_seconds = 120.0
norm_mode = "zscore"
batch_size = 128
output_dir = "{name}"
{extra}
[[pvs]]
id = "p"
expected_pv = "{PV}"
filepaths = ["raw/p.parquet"]
max_age_seconds = 1.0

[[pvs]]
id = "ler"
expected_pv = "SYN:LER"
filepaths = ["raw/ler.parquet"]
max_age_seconds = 1.0

[[pvs]]
id = "her"
expected_pv = "SYN:HER"
filepaths = ["raw/her.parquet"]
max_age_seconds = 1.0
""", encoding="utf-8")
    return build_dataset(config, progress=None)


def train_vae(root: Path, train_dataset: Path) -> Path:
    config = root / "train.toml"
    config.write_text(f"""
[run]
name = "vae"
seed = 1
device = "cpu"

[data]
dataset_path = "{train_dataset}"

[[pvs]]
pv = "{PV}"
type = "Synthetic"
name = "D01_L01"

[optimizer]
name = "adam"
learning_rate = 3e-3
weight_decay = 0.0

[training]
epochs = 3
batch_size = 32

[model]
name = "cnn_vae"
hidden_channels = [8, 16]
kernel_size = 5
stride = 2
latent_dim = 4
beta = 1.0
""", encoding="utf-8")
    return train_model(config, progress=None) / "best.pt"


def write_test_toml(directory: Path, dataset: Path, checkpoint: Path, *, name="eval", model_table=None,
                    extra="") -> Path:
    model_table = model_table or 'VAE = true\nlatent_mode = "z_mean"\n'
    # Test results are written next to the checkpoint, so every test gets its
    # own copy of the shared trained model: <directory>/TrainedModel/best.pt.
    model_folder = directory / "TrainedModel"
    model_folder.mkdir(exist_ok=True)
    if not (model_folder / Path(checkpoint).name).exists():
        shutil.copy2(checkpoint, model_folder / Path(checkpoint).name)
    checkpoint = model_folder / Path(checkpoint).name
    path = directory / f"{name}.toml"
    path.write_text(f"""
[run]
name = "{name}"
seed = 5
device = "cpu"

[data]
dataset_path = "{dataset}"
batch_size = 100

[model]
checkpoint = "{checkpoint}"
{model_table}
{extra}""", encoding="utf-8")
    return path


# ---------------------------------------------------------------- a minimal forecasting model

from dataclasses import dataclass as _dataclass

from torch import nn as _nn

import models as _models
from models.base import ForecastingModel, ModelSpec

FORECASTER = "test_linear_forecaster"


@_dataclass(frozen=True)
class LinearForecasterHyperparameters:
    horizon: int


class LinearForecaster(ForecastingModel):
    """Predicts the last `horizon` samples linearly from the context before them."""

    def __init__(self, hparams, *, n_channels, window_size):
        super().__init__(horizon=hparams.horizon, n_channels=n_channels, window_size=window_size)
        self.linear = _nn.Linear(n_channels * self.context_size, n_channels * self.horizon)

    def forecast(self, context):
        return self.linear(context.flatten(start_dim=1)).view(-1, self.n_channels, self.horizon)


FORECASTER_SPEC = ModelSpec(
    name=FORECASTER, hyperparameters=LinearForecasterHyperparameters,
    build=lambda hparams, *, n_channels, window_size: LinearForecaster(hparams, n_channels=n_channels,
                                                                        window_size=window_size))


def register_forecaster():
    _models.MODELS[FORECASTER] = FORECASTER_SPEC


def unregister_forecaster():
    _models.MODELS.pop(FORECASTER, None)


def train_forecaster(root: Path, train_dataset: Path, *, horizon=8) -> Path:
    config = root / "train_forecaster.toml"
    config.write_text(f"""
[run]
name = "forecaster"
seed = 1
device = "cpu"

[data]
dataset_path = "{train_dataset}"

[[pvs]]
pv = "{PV}"
type = "Synthetic"
name = "D01_L01"

[optimizer]
name = "adam"
learning_rate = 3e-3
weight_decay = 0.0

[training]
epochs = 3
batch_size = 32

[model]
name = "{FORECASTER}"
horizon = {horizon}
""", encoding="utf-8")
    return train_model(config, progress=None) / "best.pt"
