# belle2-train

Trains a custom defined model (that satisfies the model blueprint defined in `models/base.py`)from `src/models` on a completed belle2-dataset `dataset.h5`.
All settings come from one TOML file.

## Install

`belle2-train` needs `belle2-models` (`src/models`), which is not on PyPI, so
always install both. Python ≥ 3.11 is required. Other dependencies (numpy, h5py,
torch ≥ 2.2, matplotlib, SciencePlots) are installed automatically.

**From GitHub or GitLab.** The git repository root is `Project/`, so the
packages live in `src/models` and `src/train`.

```bash
REPO=https://github.com/<user>/<repo>.git        # GitLab: https://gitlab.<host>/<group>/<repo>.git
REF=v0.1.0                                       # tag, branch or commit; pin it for reproducible runs
pip install "belle2-models @ git+$REPO@$REF#subdirectory=src/models" \
            "belle2-train  @ git+$REPO@$REF#subdirectory=src/train"
```

**Development (editable) install from a clone:**

```bash
git clone $REPO Belle2Project && cd Belle2Project       # contents of Project/
python -m pip install -e ./src/models -e ./src/train
```

`.gitignore` already excludes `data/`, `datasets/` and the virtual
environments. Trained-model folders (`*TrainedModels/`, which hold `.pt`
checkpoints) and `PVPlots/` are not excluded yet: add them before the first
commit unless you want them versioned.

**Check the installation:** `belle2-train --help`, and
`python -c "import models, train; print(sorted(models.MODELS))"`.

## Package folder (development version)

In a clone of the repository the package is the folder `src/train/`; with an
editable install (`pip install -e ./src/train`) changes there take effect
immediately. The models themselves live in `src/models/` (see "How several
models are supported" below); this package only knows how to train them. A
complete example TOML is `src/models/cnn_vae/exampleTrain.toml`.

| File | Task |
| --- | --- |
| `train.py` | Runs one training from a TOML (`train_model`, CLI `main`): validates the settings, loads the data, builds the model, trains, and writes the output folder. |
| `config.py` | Parses and validates the training TOML into typed settings; the `[model]` table is checked against the selected model's own hyperparameter class. |
| `data.py` | Checks that `dataset.h5` is a complete train/val build with the expected PVs and serves its windows to PyTorch (in memory or lazily from disk). |
| `engine.py` | The model-independent training loop: batches to the device, `compute_losses`, optimizer step, epoch means, early stopping. |
| `progress.py` | The live progress line within an epoch, shown with `[run] verbose = 1`. |
| `style.py` | The figure style of `loss.png` (SciencePlots `science` with LaTeX text, `science_style()`). |
| `outputs.py` | Writes the trained-model folder (config copy, metadata, `history.csv`, `loss.png`, `best.pt`, `last.pt`) and reloads a model from a checkpoint (`load_checkpoint`). |
| `__init__.py`, `__main__.py` | Package exports and `python -m train --config <toml>`. |
| `pyproject.toml` | Package metadata, dependencies and the `belle2-train` command. |

## Run

```bash
belle2-train --config <run>.toml          # or: python -m train --config <run>.toml
```

From Python: `from train import train_model; folder = train_model("run.toml")`.
Relative paths in the TOML resolve against its `project_root` (see below),
not against the current directory. A complete example is
`src/models/cnn_vae/exampleTrain.toml`.

## How several models are supported

The training code never imports a specific model. It looks up the model
named in `[model] name` in a registry in `src/models`, and every model
follows the same small interface. Adding a model therefore only touches
`src/models`; `src/train` (and `src/evaluation`) stay unchanged.

**The three pieces** (all in `src/models`):
| Piece | Where | Role |
| --- | --- | --- |
| Registry `MODELS` | `__init__.py` | Maps each name to its `ModelSpec`. `get_model_spec(name)` looks it up; an unknown name is an error listing the registered models. |
| `ModelSpec(name, hyperparameters, build)` | `base.py`, one `SPEC` per model | `name`: the string used in `[model] name`. `hyperparameters`: a frozen dataclass whose fields are the allowed `[model]` keys, with range checks in `__post_init__`. `build(hparams, *, n_channels, window_size)`: returns the network. |
| `TrainableModel` | `base.py` | Base class of every model (an `nn.Module`). The model defines its own loss in `compute_losses(batch)`. |

**What happens in a run:**
1. **Look up the model:** `name` is taken out of `[model]` and its `ModelSpec` is looked up.
2. **Check the keys:** all other `[model]` keys are parsed into that model's dataclass. Unknown keys, missing required keys and wrong types are errors, so a typo never falls back to a default.
3. **Build:** `build` is called with C and T read from `dataset.h5` (never from the TOML).
4. **Train:** the shared engine does the same for every model:
   - batches of `{"values": (B, C, T) float32, "index": (B,)}`;
   - the optimizer, gradient clipping and epochs;
   - early stopping on the validation `loss`;
   - `history.csv`, `loss.png`, `best.pt` and `last.pt`.

   It minimises `compute_losses(batch)["loss"]` and logs every other entry
   as a `history.csv` column and a `loss.png` panel.
5. **Save:** the checkpoint stores the lowest loss and the last-epoch model as well as the model name and hyperparameters, so
   `load_checkpoint` (and `testModel.py`) rebuild the right model through the
   same registry.

**Rules for a model:**

- **Loss dictionary:** `compute_losses` returns a dict that contains `"loss"`.
  Every entry is a finite scalar averaged over the batch's windows, with the
  same keys in training and validation.
- **Inputs:** the model receives only `values` and `index`; labels are not
  used in training.
- **For testing (`src/evaluation`):** the default test output suits models
  whose `forward` returns a reconstruction with the input's shape. Models with
  a random latent set `is_variational = True` and accept a `latent_mode`.

**Adding a model, e.g. an LSTM autoencoder `lstm_ae`:**

1. Create `src/models/lstm_ae/` with `__init__.py` (`from .model import SPEC`),
   `model.py` and `example.toml` (a complete training TOML with
   `[model] name = "lstm_ae"`).
2. In `model.py`, define the dataclass, the model and `SPEC`:

   ```python
   from dataclasses import dataclass
   from ..base import ModelSpec, TrainableModel

   @dataclass(frozen=True)
   class LSTMAEHyperparameters:          # fields = allowed [model] keys
       hidden_size: int
       num_layers: int = 1

       def __post_init__(self):
           if self.hidden_size <= 0:
               raise ValueError("model.hidden_size must be positive")

   class LSTMAE(TrainableModel):
       def __init__(self, hparams, *, n_channels, window_size): ...
       def forward(self, values): ...    # (B, C, T) -> (B, C, T)

       def compute_losses(self, batch):
           values = batch["values"]
           mse = (self(values) - values).square().mean()
           return {"loss": mse, "mse": mse}

   def build(hparams, *, n_channels, window_size):
       return LSTMAE(hparams, n_channels=n_channels, window_size=window_size)

   SPEC = ModelSpec(name="lstm_ae", hyperparameters=LSTMAEHyperparameters, build=build)
   ```

3. Register it in `src/models/__init__.py`:

   ```python
   from .lstm_ae import SPEC as LSTM_AE
   MODELS = {spec.name: spec for spec in (CNN_VAE, LSTM_AE)}
   ```

The training TOML then selects it with `[model] name = "lstm_ae"` and
`hidden_size = 64`. Check what is registered with
`python -c "import models; print(sorted(models.MODELS))"`.

## TOML format
The file has exactly six tables: `[run]`, `[data]`, `[[pvs]]`, `[optimizer]`,
`[training]`, `[model]`.

- Unknown or missing keys are errors.
- `int` must be a TOML integer (`5`, not `5.0`). `float` also accepts integers.
- `string` must be non-empty.
- Optional keys show their default value. Optional keys marked "omit" are off
  unless set.
- Paths are absolute, or relative to `project_root`.

Window length, stride, channel order and normalization are fixed by the
dataset build. They are read from `dataset.h5` and are not set here.

### `[run]`

| Key | Format | Meaning |
| --- | --- | --- |
| `name` | string: letters, digits, `.`, `_`, `-` | Name the folder the trained model should be saved to. Training refuses to start if that folder already exists. |
| `seed` | int in [0, 2³²) | Seeds Python, NumPy and torch, and fixes the shuffle order. |
| `project_root` | string path, default `"."` | Base for all paths in the file. A relative value is resolved from the TOML's own folder. |
| `device` | `"auto"`, `"cpu"`, `"cuda"` or `"mps"`, default `"auto"` | `"auto"` picks cuda, then mps, then cpu. Naming an unavailable device is an error. |
| `verbose` | `0` or `1`, default `1` | `1`: live progress within each training and validation epoch (progress bar, batch k / n, running mean loss, windows per second, elapsed and remaining time). `0`: only the one-line summary per epoch. |

### `[data]`

| Key | Format | Meaning |
| --- | --- | --- |
| `dataset_path` | string path | A complete `dataset.h5` built with `mode = "train_val"`. The `train` split is used for training, `val` for validation. |
| `in_memory` | bool, default `true` | `true`: load each split once as float32 (C·T·4 bytes per window). `false`: read windows lazily from disk. |
| `num_workers` | int ≥ 0, default `0` | Number of DataLoader worker processes (can speed up training if in\_memory = False). |

### `[[pvs]]` (one table per channel, in the dataset's channel order)

| Key | Format | Meaning |
| --- | --- | --- |
| `pv` | string, e.g. `"VALCCG:D01_L01A:PRES"` | Must equal the `expected_pv` stored in the dataset for that channel. The number and order of entries must match the dataset. |
| `type` | folder-name string, e.g. `"Pressure"` | Required when there is exactly one `[[pvs]]` entry; optional otherwise. First folder level of saving directory (see below). |
| `name` | folder-name string, e.g. `"D01_L01A"` | Required when there is exactly one `[[pvs]]` entry; optional otherwise. Second folder level of saving directory (see below). |

The output folder depends on the number of entries / channels used:
- one entry: `<project_root>/UnivariateTrainedModels/<type>/<name>/<run.name>/`
- several entries: `<project_root>/MultivariateTrainedModels/<run.name>/`

### `[optimizer]`

| Key | Format | Meaning |
| --- | --- | --- |
| `name` | `"adam"` or `"adamw"` | `torch.optim.Adam` or `torch.optim.AdamW`. |
| `learning_rate` | float > 0 | Step size. |
| `weight_decay` | float ≥ 0 | L2 penalty for Adam; decoupled weight decay for AdamW. |
| `betas` | array of 2 floats in [0, 1), default `[0.9, 0.999]` | Decay rates of the first- and second-moment estimates. |
| `eps` | float > 0, default `1e-8` | Added to the denominator for numerical stability. |

### `[training]`

| Key | Format | Meaning |
| --- | --- | --- |
| `epochs` | int > 0 | Maximum number of passes over the training split. |
| `batch_size` | int > 0 | Windows per optimizer step. The training split is reshuffled every epoch. |
| `early_stopping_patience` | int > 0, omit | Stop after this many epochs without a strictly lower validation `loss`. |
| `gradient_clip_norm` | float > 0, omit | Clip the global gradient L2 norm to this value. |
**Training losses should be specified in the model defining script.**

## Minimal example

```toml
[run]
project_root = "../../.."            # this file lives in src/models/cnn_vae/
name = "cnn_vae_baseline"
seed = 42

[data]
dataset_path = "datasets/D01_L01A_dataset/dataset.h5"

[[pvs]]
pv = "VALCCG:D01_L01A:PRES"
type = "Pressure"
name = "D01_L01A"

[optimizer]
name = "adam"
learning_rate = 1e-3
weight_decay = 0.0

[training]
epochs = 30
batch_size = 256
early_stopping_patience = 5

[model]
name = "cnn_vae"
hidden_channels = [16, 32, 64]
kernel_size = 7
stride = 2
latent_dim = 8
beta = 1.0
```

## Output (one folder per run)

| File | Contents |
| --- | --- |
| `config.toml` | Exact copy of the input TOML. |
| `run_metadata.json` | Status, the resolved config (defaults filled in), PVs, and the dataset id, path, `value_transform` and normalization statistics. Also the model source path and code hashes, versions, and the best and last epoch metrics. |
| `history.csv` | One row per epoch: `train_*` and `val_*` for every loss term, learning rate, seconds. |
| `loss.png` | Train and validation curves for every loss term, with the best epoch marked. |
| `best.pt` / `last.pt` | Weights at the lowest validation loss / weights plus optimizer state after the last epoch. |

To reload a trained model:
`from train.outputs import load_checkpoint; model, ckpt = load_checkpoint(".../best.pt")`.

Tests: `PYTHONPATH=src venv/bin/python -m pytest -q tests/test_train`.
