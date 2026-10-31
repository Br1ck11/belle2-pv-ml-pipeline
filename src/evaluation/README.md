# belle2-evaluation

Tests a trained model (a checkpoint written by `belle2-train`) on a
belle2-dataset built with `mode = "test"`. It scores every window, plots
reconstructions per label region, plots the score against the distance to an
abort, and computes region statistics. All settings come from one TOML file.
The import name is `evaluation`.

## Install

This package needs `belle2-models` and `belle2-train`, so install all three.
Python ≥ 3.11 is required. Other dependencies (numpy, pandas, h5py, torch,
matplotlib, SciencePlots) are installed automatically.

```bash
REPO=https://github.com/<user>/<repo>.git        # GitLab: https://gitlab.<host>/<group>/<repo>.git
REF=v0.1.0                                       # tag, branch or commit
pip install "belle2-models     @ git+$REPO@$REF#subdirectory=src/models" \
            "belle2-train      @ git+$REPO@$REF#subdirectory=src/train" \
            "belle2-evaluation @ git+$REPO@$REF#subdirectory=src/evaluation"
```

For development, from`Project/`, run:
`python -m pip install -e ./src/models -e ./src/train -e ./src/evaluation`.

## Package folder (development version)

In a clone of the repository the package is the folder `src/evaluation/`; with
an editable install (`pip install -e ./src/evaluation`) changes there take
effect immediately. A test run goes as: settings, test set and checks, scores,
then the plots and tables.

| File | Task |
| --- | --- |
| `evaluate.py` | Runs one model test from a TOML (`evaluate_model`, CLI `main`): loads the checkpoint, checks compatibility, scores all windows, and writes all outputs and `metadata.json`. |
| `config.py` | Parses and validates the test TOML into typed settings, and places the output folder next to the tested checkpoint. |
| `data.py` | Reads a `mode = "test"` `dataset.h5` with its labels and abort associations, assigns the label regions, and checks that model and test set are compatible. |
| `scoring.py` | Runs the model over all test windows in batches and computes the per-window, per-channel MSE from `evaluation_output` (reconstruction or forecast). |
| `outputs.py` | Draws the figures and writes the CSVs behind them: `ModelReconstructions/`, `ScoreVsAbortDistance/` and `RegionStatistics/` (including the threshold exceedance - a given percentile of normal windows - per abort). |
| `style.py` | The figure style (SciencePlots `science` with LaTeX text, `science_style()`); a copy of the one in `src/plotting`, so the plotting package is not needed. |
| `example.toml` | A complete, runnable test TOML with every key explained. |
| `__init__.py`, `__main__.py` | Package exports and `python -m evaluation --config <toml>`. |
| `pyproject.toml` | Package metadata, dependencies and the `belle2-test-model` command. |

## Run

```bash
python scripts/testModel.py --config <test>.toml      # or: python -m evaluation --config <test>.toml
```

From Python: `from evaluation import evaluate_model; folder = evaluate_model("test.toml")`.
A complete example with every key is `src/evaluation/example.toml`. It tests the T = 60 quick-test CNN VAE on the May 2026 D01_L01A test set.

## TOML format
- `[run]`, `[data]` and `[model]` are required. `[selection]`,
  `[reconstructions]` and `[abort_distance]` are optional; their keys default
  as shown.
- Unknown or missing keys are errors.
- `int` must be a TOML integer.
- Optional settings a marked with "omit" below.
- Paths are absolute, or relative to `project_root`.

### `[run]`

| Key | Format | Meaning |
| --- | --- | --- |
| `name` | string: letters, digits, `.`, `_`, `-` | Results folder `<name>/` inside the trained-model folder of `model.checkpoint` (next to `best.pt`). Refused if it exists. |
| `seed` | int in [0, 2³²) | Seeds `sampling_z` and the random choice of windows and aborts. |
| `project_root` | string path, default `"."` | Base for all paths. A relative value is resolved from the TOML's own folder. |
| `device` | `"auto"`, `"cpu"`, `"cuda"` or `"mps"`, default `"auto"` | `"auto"` picks cuda, then mps, then cpu. |
| `verbose` | `0` or `1`, default `1` | `1`: live progress within each scoring pass over the test windows (progress bar, batch k / n, running mean loss, windows per second, elapsed and remaining time). `0`: only the start and end messages. |

### `[data]`

| Key | Format | Meaning |
| --- | --- | --- |
| `dataset_path` | string path | A complete `dataset.h5` with `mode = "test"`. If normalized, it must use the statistics of the model's training dataset. |
| `batch_size` | int > 0, default `512` | Windows per forward pass. This limits memory only. |

### `[model]`

| Key | Format | Meaning |
| --- | --- | --- |
| `checkpoint` | string path | `best.pt` or `last.pt` from a trained-model folder. |
| `VAE` | bool | `true` if the model has a stochastic latent. It must agree with the checkpoint's model. |
| `latent_mode` | `"z_mean"` or `"sampling_z"`, only when `VAE = true` (then required) | `z_mean`: decode the posterior mean (deterministic). `sampling_z`: decode one sample z = μ + σ·ε (seeded). |

### `[selection]`

| Key | Format | Meaning |
| --- | --- | --- |
| `categories` | array of strings, e.g. `["SBL"]` or `["SBL", "RF"]`; omit = all | Only aborts of these categories are evaluated: they define the abort, pre-abort and post-abort regions, the aborts in `ScoreVsAbortDistance/` and the explicit `event_ids` allowed. Windows related only to other categories are left out (they are not counted as no-abort). Case-insensitive. Every listed category must have at least one abort in the test set, otherwise the run stops with an error listing the categories present. |
| `pvs` | array of PV strings, omit = all channels | Channels shown in the score-vs-distance plots and region statistics. Reconstructions always show all channels. |

### `[reconstructions]`

| Key | Format | Meaning |
| --- | --- | --- |
| `windows_per_region` | int > 0, default `3` | Windows plotted per region. Windows that belong to only one region are preferred. |
| `sample` | `"random"` or `"first"`, default `"random"` | Seeded random choice, or the earliest windows. |

### `[abort_distance]`

| Key | Format | Meaning |
| --- | --- | --- |
| `n_aborts` | int > 0, default `5` | Number of aborts that get a score-vs-distance plot. |
| `selection` | `"first"` or `"random"`, default `"first"` | Chronological or seeded random choice among aborts with windows in the test set. |
| `event_ids` | array of event-ID strings, omit | Choose explicit aborts. Replaces `n_aborts`/`selection`. |
| `non_abort_seconds` | float ≥ 0, default `0.0` | Extra time shown before the pre-abort region in the score-vs-distance plots, so the figure starts with normal operation: windows whose last sample lies in [−pre − non_abort_seconds, −pre). They are no-abort windows unless they belong to another abort. If the dataset's `exclude_before_seconds` is larger than `pre_abort_seconds`, the build removed the windows in between, so this stretch has a gap right before the pre-abort region. |

### `[exceedance]`

| Key | Format | Meaning |
| --- | --- | --- |
| `percentiles` | array of floats in (0, 100), no repeats, default `[95.0, 99.0, 99.9]` | One threshold per value: that percentile of the test set's no-abort window scores, one plot per PV. See "Threshold exceedance per abort" below. |

## Checks before scoring
- **Errors (the scores would be meaningless):**
  - `VAE` disagrees with the model;
  - the channel PVs or their order differ from training;
  - `value_transform` or the values state (raw vs normalized) differs;
  - the test set was normalized with statistics of a different dataset.
- **Warnings (stored in `metadata.json`):**
  - the window length T differs;
  - the sample spacing differs.

  In both cases the model is still applied. A model with a fixed input length then stops with a clear error.

## Score and regions
- **Score:** per window and channel,
  mse = mean over the model's output samples of (output − target)², using
  `model.evaluation_output`, in the dataset's stored units. For a
  reconstruction model these are all T samples. For a forecasting model they
  are the predicted horizon, so the score is its prediction error (see
  "Reconstruction and forecasting models" below).
- **Regions** come from the test labels: `abort` (the window contains the
  abort), `pre_abort`, `post_abort` and `non_abort`. A window related to two
  aborts or containing pre-abort and abort data points can belong to two labelled regions.
- **Distance to an abort** = time of the window's last sample − abort time:
  - pre-abort windows lie in [−pre, 0);
  - windows containing the abort lie in [0, (T − 1)·step];
  - post-abort windows lie after that, up to +post.

## Output: `<trained-model folder>/<name>/`

For example `UnivariateTrainedModels/Pressure/D01_L01A/<training run>/<name>/`,
next to the tested `best.pt`, so a model and all its tests stay together.
All figures use the SciencePlots `science` style with LaTeX text (`style.py`; needs a TeX installation, otherwise a no-LaTeX fallback with a warning).


| Item | Contents |
| --- | --- |
| `config.toml`, `metadata.json` | Exact TOML copy; resolved settings, warnings, checkpoint and test-set identity, the model output kind (`score.output`: `reconstruction`, or `forecast` with its window positions), region counts, output list, versions, code hashes. |
| `window_scores.csv` | Every test window: row, start/end time, labels, regions, event IDs, `mse <PV>` per channel, mean over the selected PVs. |
| `ModelReconstructions/` | One subfolder per label region: `abort/`, `pre_abort/`, `post_abort/`, `non_abort/`. In each, `example_<k>_row<row>.png` shows the whole input window and the model output in one plot, with the abort marked. For a forecaster, the prediction is drawn only there. `.csv` holds every window sample: `time_ns`, `time_utc`, `predicted` (true where the model outputs a value), `<PV> input`, `<PV> target` and `<PV> output` (the last two are empty where `predicted` is false). |
| `ScoreVsAbortDistance/` | One folder per abort, `<time>_<category>_<event id>/`. In it, one two-panel figure per selected PV, `<PV>.png` (`:` replaced by `_`). Both panels share the x-axis "time − abort time". Top: the PV data (gaps where the test set has no windows), and the model output of every window as its own segment, coloured by the window's region (no averaging over overlapping windows). Bottom: the score of each window at its last sample, with the regions shaded, vertical lines at −pre − non_abort, −pre, 0, (T − 1)·step and +post, and a dashed line at the median no-abort score. `<PV>_scores.csv` holds the bottom panel, and `<PV>_signal.csv` the top panel (long format: `series` = `input` or `output`, `row`, `time_s`, `time_utc`, `value`). |
| `RegionStatistics/` | `exceedance_<PV>.png`, `exceedance.csv` and `exceedance_by_abort.csv`: the fraction of aborts with at least one window above each threshold (see "Threshold exceedance per abort"). `region_statistics.csv` gives the counts n, mean, standard error of the mean, median and std per PV and region. `region_statistics_by_category.csv` splits this by abort category. `region_statistics.png` shows mean ± standard error next to the median. |

### Threshold exceedance per abort
This counts **aborts, not windows**. For every PV and every percentile p in
`[exceedance] percentiles`:

1. **Threshold:** s_p = the p-th percentile of the scores of all no-abort
   windows in the test set. By construction, a fraction 1 − p/100 of the
   no-abort windows lies above it, e.g. 1 % for p = 99.
2. **Fraction per region:** for each region R (pre-abort, abort, post-abort),

   f_R = (number of aborts whose region R contains at least one window with a score above s_p) / N,

   where N is the total number of aborts with at least one window in the test
   set (only the selected categories). Every region uses the same denominator
   N, so f_R ≤ 1 and the regions can be compared directly. An abort without
   windows in a region (e.g. no pre-abort windows because the beam current
   was below the threshold) counts as "not exceeded" there. The figure shows,
   as a function of each region, how many aborts have windows in it.

`exceedance_<PV>.png` shows f_R on a log y-axis, with pre-abort, abort and
post-abort on the x-axis and one bar per percentile, labelled k / N (k = number of windows exceeding a threshold in a given region). No-abort windows set the threshold, so they have no bar of their own.
`exceedance.csv` has one row per PV, percentile and region: the threshold,
the number of no-abort windows, k, N, f_R, the aborts with windows in the
region, and the number of region windows and of those above the threshold.
`exceedance_by_abort.csv` has one row per PV, percentile, abort and region:
the number of windows, how many are above the threshold, whether the region
counts as exceeded, and the region's highest score.

Overlapping windows are correlated. The standard error of the mean,
std(ddof = 1)/√n, assumes independent windows, so it is a lower bound on the
real uncertainty.

## Reconstruction and forecasting models

The evaluation never looks inside a model. It calls
`model.evaluation_output(batch, latent_mode)`, which returns:

| Key | Shape | Meaning |
| --- | --- | --- |
| `output` | (B, C, T′) | what the model produced |
| `target` | (B, C, T′) | the true values it is compared with |
| `time_index` | (T′,) integers | window positions (0 … T − 1) of the T′ output samples; |

`time_index` must be strictly increasing, inside [0, T), have T′ entries and
be the same for every batch. Otherwise the run stops with an error, because
the scores would not be comparable.

- **Reconstruction models** (e.g. `cnn_vae`): T′ = T, and target = input.
  Nothing to implement if `forward(values)` returns an input-shaped tensor.
  VAE-like models set `is_variational = True` and honour `latent_mode`.
### Forecasting was not explicitely tested yet, test to verify the functionality!
- **Forecasting models** predict samples and are scored by the distance of
  each prediction to the measured value. Subclass `models.ForecastingModel`
  and implement only `forecast(context)`:
  - each window of T samples is split into a context (the first T − H
    samples, which the model sees) and a horizon (the last H, which it
    predicts);
  - the training loss is the MSE between prediction and horizon (override
    `compute_losses` for another objective);
  - the test score is the same MSE, on the horizon only;
  - `time_index` = T − H … T − 1.

  The horizon must lie inside the window, because a test window stores
  only its T samples. To predict H samples from a context of L samples, build
  the dataset with T = L + H (e.g. T = 60 for 30 s of context and a 30 s
  horizon at 1 s spacing). The distance to an abort is still measured from
  the window's last sample, which for a forecaster is the end of the horizon.

Everything else (regions, score-vs-distance plots, statistics) is
identical for both kinds.

Tests (from `Project/`): `python -m pytest -q tests/test_evaluation`.
