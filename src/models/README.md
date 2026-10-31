# belle2-models

PyTorch models trained by [`src/train`](../train/README.md), one folder per
model (import name **models**). The training TOML selects a model with
`[model].name`; every other `[model]` key is parsed into that model's
hyperparameter dataclass.

```text
models/
├── __init__.py      registry: MODELS, get_model_spec(name)
├── base.py          contract: TrainableModel, ForecastingModel, ModelSpec, Batch
└── cnn_vae/
    ├── model.py     CNNVAEHyperparameters, CNNVAE, build, SPEC
    └── example.toml complete training configuration
```

## Contract (`base.py`)
**Defines what a custom model class must specify to work with the train package.**
- `TrainableModel(nn.Module)` must implement
  `compute_losses(batch) -> {"loss": ..., ... }`where ` batch["values"]` is float32 and 
  `(B, C, T)` shaped and `batch["index"]` gives the HDF5 rows. `compute_losse`s returns a finite
  scalar tensor averaged over the batch's windows and it must not be NaN or infinite. `"loss"` is minimised and
  all entries are logged and plotted.
- `ForecastingModel(TrainableModel)` is for models that predict samples
  instead of reconstructing them:
  - a subclass calls `super().__init__(horizon=H, n_channels=C,
    window_size=T)` (1 ≤ H < T) and implements `forecast(context)`, which
    takes the first T − H samples and returns the last H, shape (B, C, H);
  - the base class provides `compute_losses` (MSE on the horizon),
    `evaluation_output` (prediction, true horizon, `time_index` =
    T − H … T − 1), `split(values)` and `context_size`.
- `ModelSpec(name, hyperparameters, build, checkpoint_defaults={})`: `checkpoint_defaults` gives the values for hyperparameters that did not exist when older checkpoints were written, so they keep loading.
- `ModelSpec(name, hyperparameters, build)`: `hyperparameters` is a frozen
  dataclass with range checks in `__post_init__`.
  `build(hparams, *, n_channels, window_size)` receives C and T from
  dataset.h5.

## Adding a model

The full explanation, with a worked example, is in
[`src/train/README.md`](../train/README.md), section "How several models are
supported". In short:

1. Create `models/<name>/model.py` with the dataclass, the
   `TrainableModel` subclass, `build` and `SPEC`. Then create
   `models/<name>/__init__.py` re-exporting `SPEC`, and
   `models/<name>/example.toml`.
2. Import its `SPEC` in `models/__init__.py` and add it to the `MODELS`
   tuple.
3. For a non-editable wheel only, list the folder in `pyproject.toml`.

## `cnn_vae`

| Key | Meaning |
| --- | --- |
| `hidden_channels` | Encoder widths, one Conv1d per entry |
| `kernel_size` | One value or one per layer |
| `stride` | One value or one per layer; 1 <= stride <= kernel_size |
| `padding` | `"same"` (= kernel_size // 2, default) or int, one value or one per layer; L -> floor((L + 2p - k) / s) + 1 |
| `latent_dim` | Dimension of z |
| `beta` | KL weight (1.0 = standard ELBO) |
| `activation` | `linear`, `relu`, `silu` (default), `gelu`; one value or one per layer, mirrored in the decoder |
| `output_kernel_size` | Final stride-1 Conv1d (odd kernel, default 7) after the last transposed conv, against the stride-2 checkerboard pattern; `0` = none (pre-option checkpoints load as `0` via `SPEC.checkpoint_defaults`) |

The decoder picks each ConvTranspose1d `output_padding` so the reconstruction
has exactly T samples for any T. Per window:
`reconstruction = 0.5 * sum (x_hat - x)^2`, `kl = KL(q(z|x) || N(0, I))`,
`loss = reconstruction + beta * kl`, plus per-value `mse`. Train mode samples
z. Eval mode uses z = mu, so validation and scoring are deterministic.
