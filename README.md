# Belle II PV ML Pipeline

Machine-learning pipeline developed for SuperKEKB/Belle II process variables (PVs,
e.g. vacuum gauge pressures) aimed at analysing beam aborts at SuperKEKB. It builds windowed datasets
from the PV archive, trains models on normal operation, and tests how
anomalous the PVs look before, during and after aborts. Every step is driven
by one TOML file and writes a reproducible output folder.

## Layout

| Folder | Content |
| --- | --- |
| `src/dataset/` | `belle2-dataset`: PV Parquet archive + abort catalogue → `dataset.h5` with train/val or labelled test windows. |
| `src/models/` | `belle2-models`: the model definitions (currently `cnn_vae`, a 1D convolutional VAE), each with an example TOML. |
| `src/train/` | `belle2-train`: model-independent training (checkpoints, history, loss plot). |
| `src/evaluation/` | `belle2-evaluation`: tests a trained model on a test dataset (scores, reconstructions, abort plots, statistics). |
| `src/plotting/` | `belle2-plotting`: plots of the raw PVs and of stored dataset windows. |
| `scripts/` | Thin command-line wrappers around the packages (see below). |
| `tests/` | `pytest` tests for dataset, train, evaluation and plotting. |

Each package has its own README with the full TOML reference.
The PV archive, built datasets and trained models are not part of the
repository.

## Setup

Python ≥ 3.11. From the repository root:

```bash
python -m venv venv && source venv/bin/activate
python -m pip install -e ./src/dataset -e ./src/models -e ./src/train -e ./src/evaluation
python -m pip install -e ./src/plotting          # used for plotting of PVs
```

Figures use the SciencePlots style with LaTeX text; without a TeX
installation (e.g. MacTeX) they fall back to the same style without LaTeX.

## Workflow

| Step | Command | Example TOML |
| --- | --- | --- |
| 1. Build a train/val dataset, then a test dataset (`mode = "test"`, normalized with the train/val statistics) | `python scripts/build_abort_dataset.py --config <toml>` | `src/dataset/dataset.toml` |
| 2. Train a model | `python scripts/trainModel.py --config <toml>` | `src/models/cnn_vae/exampleTrain.toml` |
| 3. Test the trained model | `python scripts/testModel.py --config <toml>` | `src/evaluation/example.toml` |
| Plot PVs around aborts | `python scripts/plot_pv_overview.py --config <toml>` | `src/plotting/examples/pv_overview.toml` |
| Plot stored dataset windows | `python scripts/plot_dataset_windows.py --config <toml>` | `src/plotting/examples/dataset_windows.toml` |
| Check a stored window against the archive | `python scripts/crosscheck_dataset_windows.py --config <toml>` | `src/plotting/examples/crosscheck.toml` |

Outputs: a dataset folder with `dataset.h5` and a copy of its TOML; a
trained-model folder `UnivariateTrainedModels/<type>/<PV>/<run>/` (one PV) or
`MultivariateTrainedModels/<run>/` (several PVs); and each model test in a
subfolder of the trained-model folder it tests.

## Tests

```bash
python -m pytest -q tests
```
