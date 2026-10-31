"""Contract tests for the TOML-driven training pipeline and the CNN VAE."""

import csv
import hashlib
import json
import shutil

import h5py
import pytest
import torch

from models.cnn_vae.model import CNNVAE, CNNVAEHyperparameters, encoder_lengths
from train import train_model
from train.config import resolve_config
from train.data import H5WindowDataset, inspect_dataset
from train.engine import run_epoch
from train.outputs import load_checkpoint
from train_helpers import MULTIVARIATE_PVS, write_train_config


RUN_FILES = {"config.toml", "run_metadata.json", "history.csv", "loss.png", "best.pt", "last.pt"}


def hparams(**changes):
    base = dict(hidden_channels=(8, 16), kernel_size=5, stride=2, latent_dim=4, beta=1.0)
    return CNNVAEHyperparameters(**{**base, **changes})


# ---------------------------------------------------------------- CNN VAE (src/models)

@pytest.mark.parametrize("window_size", [1, 2, 7, 32, 61, 120, 128])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("n_channels", [1, 3])
def test_reconstruction_has_input_shape(window_size, stride, n_channels):
    model = CNNVAE(hparams(stride=stride), n_channels=n_channels, window_size=window_size).eval()
    values = torch.randn(4, n_channels, window_size)
    reconstruction, mu, logvar = model(values)
    assert reconstruction.shape == values.shape
    assert mu.shape == logvar.shape == (4, 4)


def test_encoder_lengths_match_conv_arithmetic():
    # floor((L + 2p - k) / s) + 1 with odd k and p = k // 2 -> (L - 1) // s + 1
    assert encoder_lengths(120, (7, 7, 7), (2, 2, 2), (3, 3, 3)) == [120, 60, 30, 15]
    assert encoder_lengths(61, (5, 5), (2, 2), (2, 2)) == [61, 31, 16]
    assert encoder_lengths(60, (7, 5, 5), (1, 2, 2), (3, 2, 2)) == [60, 60, 30, 15]
    assert encoder_lengths(10, (5,), (1,), (0,)) == [10, 6]


def test_loss_terms_match_hand_calculation():
    torch.manual_seed(0)
    model = CNNVAE(hparams(beta=0.5), n_channels=2, window_size=16).eval()
    values = torch.randn(3, 2, 16)
    losses = model.compute_losses({"values": values})
    reconstruction, mu, logvar = model(values)
    expected_reconstruction = 0.5 * ((reconstruction - values) ** 2).sum(dim=(1, 2)).mean()
    expected_kl = (0.5 * (mu ** 2 + logvar.exp() - 1 - logvar)).sum(dim=1).mean()
    torch.testing.assert_close(losses["reconstruction"], expected_reconstruction)
    torch.testing.assert_close(losses["kl"], expected_kl)
    torch.testing.assert_close(losses["loss"], expected_reconstruction + 0.5 * expected_kl)


def test_kl_is_zero_for_standard_normal_posterior():
    model = CNNVAE(hparams(), n_channels=1, window_size=16).eval()
    for layer in (model.to_mu, model.to_logvar):
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
    assert model.compute_losses({"values": torch.randn(5, 1, 16)})["kl"].item() == 0.0


def test_eval_is_deterministic_train_samples():
    model = CNNVAE(hparams(), n_channels=1, window_size=16)
    values = torch.randn(2, 1, 16)
    model.eval()
    assert torch.equal(model(values)[0], model(values)[0])
    model.train()
    assert not torch.equal(model(values)[0], model(values)[0])


def test_every_parameter_receives_gradient():
    model = CNNVAE(hparams(), n_channels=2, window_size=20).train()
    model.compute_losses({"values": torch.randn(4, 2, 20)})["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("changes", [dict(kernel_size=4), dict(stride=6), dict(latent_dim=0),
                                     dict(beta=-1.0), dict(hidden_channels=(8, 0))])
def test_invalid_hyperparameters_rejected(changes):
    with pytest.raises(ValueError):
        hparams(**changes)


PER_LAYER = dict(hidden_channels=(8, 12, 16), kernel_size=(7, 5, 5), padding=(3, "same", "same"),
                 stride=(1, 2, 2), activation=("linear", "silu", "linear"), latent_dim=8)


@pytest.mark.parametrize("window_size", [7, 60, 61, 120])
def test_per_layer_settings_build_mirrored_model(window_size):
    model = CNNVAE(hparams(**PER_LAYER), n_channels=1, window_size=window_size).eval()
    convs = [layer for layer in model.encoder if isinstance(layer, torch.nn.Conv1d)]
    assert [(c.kernel_size[0], c.stride[0], c.padding[0]) for c in convs] == [(7, 1, 3), (5, 2, 2), (5, 2, 2)]
    assert [type(layer).__name__ for layer in model.encoder[1::2]] == ["Identity", "SiLU", "Identity"]
    # decoder mirrors: latent -> activation of encoder layer 3, then layers 3, 2, 1 with activations 2, 1, none
    assert type(model.latent_activation).__name__ == "Identity"
    # default output layer: last transposed conv keeps width 8 (+ activation 0), then a stride-1 Conv1d -> C
    assert [type(layer).__name__ for layer in model.decoder] == [
        "ConvTranspose1d", "SiLU", "ConvTranspose1d", "Identity", "ConvTranspose1d", "Identity", "Conv1d"]
    assert model.decoder[4].out_channels == 8 and model.decoder[-1].stride == (1,) and model.decoder[-1].in_channels == 8
    # output_kernel_size = 0: the earlier decoder, last transposed conv straight to C channels
    plain = CNNVAE(hparams(**PER_LAYER, output_kernel_size=0), n_channels=1, window_size=window_size).eval()
    assert [type(layer).__name__ for layer in plain.decoder] == [
        "ConvTranspose1d", "SiLU", "ConvTranspose1d", "Identity", "ConvTranspose1d"]
    assert plain(torch.randn(2, 1, window_size))[0].shape == (2, 1, window_size)
    values = torch.randn(3, 1, window_size)
    assert model(values)[0].shape == values.shape
    if window_size == 60:
        assert model.lengths == [60, 60, 30, 15]


@pytest.mark.parametrize("changes", [dict(kernel_size=(5, 5, 5)), dict(stride=(1, 2, 2)), dict(activation=("silu",)),
                                     dict(padding=4), dict(padding="same", kernel_size=4, stride=2),
                                     dict(kernel_size=(5, 4), padding="same")])
def test_invalid_per_layer_settings_rejected(changes):
    with pytest.raises(ValueError):
        hparams(**changes)   # base has two layers


def test_per_layer_toml_forms_parse(tmp_path, dataset_1ch):
    model = ('name = "cnn_vae"\nhidden_channels = [8, 12, 16]\nkernel_size = [7, 5, 5]\n'
             'padding = [3, "same", "same"]\nstride = [1, 2, 2]\nactivation = ["linear", "silu", "linear"]\n'
             'latent_dim = 8\nbeta = 1.0\n')
    config = resolve_config(write_train_config(tmp_path, dataset_1ch, model=model))
    assert config.model.padding == (3, "same", "same") and config.model.paddings == (3, 2, 2)
    with pytest.raises(ValueError, match="none of the allowed forms"):
        resolve_config(write_train_config(tmp_path, dataset_1ch, run_name="bad",
                                          model=model.replace('padding = [3, "same", "same"]', 'padding = "valid"')))


# ---------------------------------------------------------------- config

def test_univariate_and_multivariate_folders(tmp_path, dataset_1ch, dataset_2ch):
    uni = resolve_config(write_train_config(tmp_path, dataset_1ch, run_name="a"))
    assert uni.layout == "univariate"
    assert uni.output_dir == tmp_path.resolve() / "UnivariateTrainedModels" / "Synthetic" / "S0" / "a"
    assert uni.model.hidden_channels == (8, 16) and uni.optimizer.betas == (0.9, 0.999)
    multi = resolve_config(write_train_config(tmp_path, dataset_2ch, run_name="b", pvs=MULTIVARIATE_PVS))
    assert multi.layout == "multivariate"
    assert multi.output_dir == tmp_path.resolve() / "MultivariateTrainedModels" / "b"


@pytest.mark.parametrize("pvs, message", [
    ('[[pvs]]\npv = "DEMO:S0"\n', "univariate run needs"),
    ('[[pvs]]\npv = "DEMO:S0"\ntype = "../x"\nname = "S0"\n', "single folder name"),
    ('[[pvs]]\npv = "DEMO:S0"\n\n[[pvs]]\npv = "DEMO:S0"\n', "repeat"),
    ('[[pvs]]\npv = "DEMO:S0"\ntype = "P"\nname = "S0"\nunit = "Pa"\n', "Unknown"),
])
def test_bad_pv_tables_rejected(tmp_path, dataset_1ch, pvs, message):
    with pytest.raises(ValueError, match=message):
        resolve_config(write_train_config(tmp_path, dataset_1ch, pvs=pvs))


@pytest.mark.parametrize("model, message", [
    ('name = "nope"\n', "Unknown model"),
    ('name = "cnn_vae"\nhidden_channels = [8]\nkernel_size = 5\nstride = 2\nlatent_dim = 4\n', "Missing"),
    ('name = "cnn_vae"\nhidden_channels = [8]\nkernel_size = 5\nstride = 2\nlatent_dim = 4\nbeta = 1.0\nlatnt = 3\n', "Unknown"),
    ('name = "cnn_vae"\nhidden_channels = 8\nkernel_size = 5\nstride = 2\nlatent_dim = 4\nbeta = 1.0\n', "array"),
    ('name = "cnn_vae"\nhidden_channels = [8]\nkernel_size = 5.0\nstride = 2\nlatent_dim = 4\nbeta = 1.0\n', "integer"),
])
def test_bad_model_tables_rejected(tmp_path, dataset_1ch, model, message):
    with pytest.raises(ValueError, match=message):
        resolve_config(write_train_config(tmp_path, dataset_1ch, model=model))


def test_optimizer_training_keys_and_existing_folder(tmp_path, dataset_1ch):
    with pytest.raises(ValueError, match="Missing \\[optimizer\\] setting: weight_decay"):
        resolve_config(write_train_config(tmp_path, dataset_1ch, optimizer='name = "adam"\nlearning_rate = 1e-3\n'))
    with pytest.raises(ValueError, match="betas"):
        resolve_config(write_train_config(tmp_path, dataset_1ch, optimizer=(
            'name = "adam"\nlearning_rate = 1e-3\nweight_decay = 0.0\nbetas = [0.9]\n')))
    with pytest.raises(ValueError, match="Unknown \\[training\\]"):
        resolve_config(write_train_config(tmp_path, dataset_1ch, extra_training="learning_rate = 1.0\n"))
    (tmp_path / "UnivariateTrainedModels" / "Synthetic" / "S0" / "run").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        resolve_config(write_train_config(tmp_path, dataset_1ch))


def test_pvs_must_match_dataset_channels(tmp_path, dataset_1ch, dataset_2ch):
    wrong = '[[pvs]]\npv = "DEMO:S1"\ntype = "Synthetic"\nname = "S1"\n'
    with pytest.raises(ValueError, match="do not match the dataset channels"):
        train_model(write_train_config(tmp_path, dataset_1ch, pvs=wrong), progress=None)
    with pytest.raises(ValueError, match="do not match the dataset channels"):
        train_model(write_train_config(tmp_path, dataset_2ch), progress=None)   # 1 PV vs 2 channels
    assert not (tmp_path / "UnivariateTrainedModels").exists()                  # nothing created


# ---------------------------------------------------------------- data

def test_dataset_items_and_modes_agree(dataset_2ch):
    info = inspect_dataset(dataset_2ch)
    assert (info.n_channels, info.window_size) == (2, 32)
    assert info.split_counts == {"train": 154, "val": 64}
    lazy = H5WindowDataset(dataset_2ch, "val", in_memory=False)
    eager = H5WindowDataset(dataset_2ch, "val", in_memory=True)
    for index in (0, 31, 63):
        a, b = lazy[index], eager[index]
        assert a["values"].dtype == torch.float32 and a["values"].shape == (2, 32)
        assert torch.equal(a["values"], b["values"]) and a["index"] == index
    lazy.close()


def test_incomplete_dataset_rejected(tmp_path, dataset_1ch):
    copy = tmp_path / "copy.h5"
    shutil.copy(dataset_1ch, copy)
    with h5py.File(copy, "r+") as data:
        data.attrs["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        inspect_dataset(copy)


# ---------------------------------------------------------------- end to end

def test_univariate_run_folder_is_complete_and_reproducible(tmp_path, dataset_1ch):
    config_path = write_train_config(tmp_path, dataset_1ch, run_name="run_a")
    folder = train_model(config_path, progress=None)
    assert folder == tmp_path.resolve() / "UnivariateTrainedModels" / "Synthetic" / "S0" / "run_a"
    assert {path.name for path in folder.iterdir()} == RUN_FILES
    assert (folder / "config.toml").read_bytes() == config_path.read_bytes()

    metadata = json.loads((folder / "run_metadata.json").read_text())
    assert metadata["status"] == "completed" and metadata["layout"] == "univariate"
    assert metadata["pvs"] == [{"pv": "DEMO:S0", "type": "Synthetic", "name": "S0", "dataset_channel_id": "s0"}]
    assert metadata["resolved_config"]["optimizer"]["betas"] == [0.9, 0.999]
    assert metadata["dataset"]["value_transform"] == "none"   # dataset built without log_transform
    source = metadata["model"]["source_path"]
    assert source.endswith("models/cnn_vae/model.py")
    with open(source, "rb") as stream:
        assert metadata["model"]["source_sha256"]["model.py"] == hashlib.sha256(stream.read()).hexdigest()
    rows = list(csv.DictReader((folder / "history.csv").open()))
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    assert {"train_loss", "train_kl", "val_reconstruction", "val_mse"} <= set(rows[0])
    assert metadata["summary"]["best_epoch_metrics"]["epoch"] == metadata["summary"]["best_epoch"]

    # Reloaded best weights reproduce the recorded validation loss (eval uses z = mu).
    model, checkpoint = load_checkpoint(folder / "best.pt")
    best = min(rows, key=lambda r: float(r["val_loss"]))
    assert checkpoint["epoch"] == int(best["epoch"]) and checkpoint["pvs"] == ["DEMO:S0"]
    assert checkpoint["value_transform"] == "none"
    loader = torch.utils.data.DataLoader(H5WindowDataset(dataset_1ch, "val", in_memory=True), batch_size=16)
    assert run_epoch(model, loader, torch.device("cpu"))["loss"] == pytest.approx(float(best["val_loss"]), rel=1e-6)

    # The same settings (only run.name changed) reproduce the losses exactly on CPU.
    rows_b = list(csv.DictReader((train_model(write_train_config(tmp_path, dataset_1ch, run_name="run_b"),
                                              progress=None) / "history.csv").open()))
    assert [(r["train_loss"], r["val_loss"]) for r in rows] == [(r["train_loss"], r["val_loss"]) for r in rows_b]


def test_multivariate_run_folder(tmp_path, dataset_2ch):
    folder = train_model(write_train_config(tmp_path, dataset_2ch, run_name="multi", pvs=MULTIVARIATE_PVS),
                         progress=None)
    assert folder == tmp_path.resolve() / "MultivariateTrainedModels" / "multi"
    assert {path.name for path in folder.iterdir()} == RUN_FILES
    assert json.loads((folder / "run_metadata.json").read_text())["layout"] == "multivariate"


def test_training_reduces_loss(tmp_path, dataset_1ch):
    folder = train_model(write_train_config(tmp_path, dataset_1ch, epochs=8), progress=None)
    rows = list(csv.DictReader((folder / "history.csv").open()))
    assert float(rows[-1]["val_loss"]) < float(rows[0]["val_loss"])


# ---------------------------------------------------------------- verbose progress

import io

from train.progress import BatchProgress


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def test_progress_in_a_log_prints_every_ten_percent():
    stream = io.StringIO()
    live = BatchProgress("Epoch 1/2 train", 50, stream=stream)
    for k in range(1, 51):
        live.update(k, 10 * k, 1.0 / k)
    live.close()
    lines = stream.getvalue().splitlines()
    assert len(lines) == 10 and "100.0%" in lines[-1] and "batch 50/50" in lines[-1] and "windows/s" in lines[0]


def test_progress_in_a_terminal_rewrites_one_line():
    stream = _Terminal()
    live = BatchProgress("Scoring", 3, stream=stream)
    for k in range(1, 4):
        live.update(k, 4 * k)
    live.close()
    text = stream.getvalue()
    assert "\n" not in text and text.startswith("\r") and text.endswith("\r") and "batch 3/3" in text


@pytest.mark.parametrize("value", ["2", "true", "-1"])
def test_verbose_must_be_zero_or_one(tmp_path, dataset_1ch, value):
    path = write_train_config(tmp_path, dataset_1ch)
    path.write_text(path.read_text().replace("[run]\n", f"[run]\nverbose = {value}\n", 1))
    with pytest.raises(ValueError, match="verbose|integer"):
        resolve_config(path)


def test_verbose_training_shows_batch_progress(tmp_path, dataset_1ch, capsys):
    path = write_train_config(tmp_path, dataset_1ch, run_name="verbose")
    path.write_text(path.read_text().replace("[run]\n", "[run]\nverbose = 1\n", 1))
    train_model(path, progress=print)
    out = capsys.readouterr().out
    assert "train [" in out and "val   [" in out and "windows/s" in out and "train_loss=" in out
    quiet = write_train_config(tmp_path, dataset_1ch, run_name="quiet")
    quiet.write_text(quiet.read_text().replace("[run]\n", "[run]\nverbose = 0\n", 1))
    train_model(quiet, progress=print)
    assert "windows/s" not in capsys.readouterr().out


# ---------------------------------------------------------------- stride-1 output layer

@pytest.mark.parametrize("kernel", [-1, 4])
def test_output_kernel_size_must_be_zero_or_odd(kernel):
    with pytest.raises(ValueError, match="output_kernel_size"):
        hparams(output_kernel_size=kernel)


@pytest.mark.parametrize("window_size", [7, 32, 61])
@pytest.mark.parametrize("kernel", [1, 3, 7])
def test_output_layer_keeps_shape(window_size, kernel):
    model = CNNVAE(hparams(output_kernel_size=kernel), n_channels=3, window_size=window_size).eval()
    values = torch.randn(2, 3, window_size)
    assert model(values)[0].shape == values.shape
    assert model.decoder[-1].kernel_size == (kernel,) and model.decoder[-1].out_channels == 3


def test_checkpoint_without_output_kernel_size_loads_as_before(tmp_path, dataset_1ch):
    """A checkpoint written before output_kernel_size existed has no output layer."""
    folder = train_model(write_train_config(tmp_path, dataset_1ch, run_name="legacy"), progress=None)
    checkpoint = torch.load(folder / "best.pt", weights_only=True)
    assert checkpoint["hyperparameters"]["output_kernel_size"] == 7           # new runs store the key
    # rebuild the file as an old one: plain decoder weights, key absent
    plain = CNNVAE(CNNVAEHyperparameters(**{**checkpoint["hyperparameters"], "output_kernel_size": 0}),
                   n_channels=checkpoint["n_channels"], window_size=checkpoint["window_size"])
    hyper = dict(checkpoint["hyperparameters"])
    del hyper["output_kernel_size"]
    torch.save({**checkpoint, "hyperparameters": hyper, "model_state": plain.state_dict()}, tmp_path / "old.pt")
    model, _ = load_checkpoint(tmp_path / "old.pt")
    assert model.hparams.output_kernel_size == 0 and not isinstance(model.decoder[-1], torch.nn.Conv1d)
