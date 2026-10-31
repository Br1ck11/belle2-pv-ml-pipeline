"""Contract tests for model testing (evaluation package)."""

import dataclasses
import json

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from evaluation.config import resolve_test_config
from evaluation.data import REGIONS, check_compatibility, load_test_set, window_regions
from evaluation.evaluate import evaluate_model
from evaluation.outputs import describe
from models.base import TrainableModel
from train.outputs import load_checkpoint
from evaluation_helpers import PV, write_test_toml


def run(tmp_path, setup, **kwargs):
    return evaluate_model(write_test_toml(tmp_path, setup["test"], setup["checkpoint"], **kwargs), progress=None)


# ---------------------------------------------------------------- configuration

@pytest.mark.parametrize("model_table, message", [
    ("VAE = true\n", "requires model.latent_mode"),
    ('VAE = false\nlatent_mode = "z_mean"\n', "only used when model.VAE = true"),
    ('VAE = true\nlatent_mode = "mean"\n', "must be one of"),
    ('VAE = "yes"\nlatent_mode = "z_mean"\n', "true or false"),
])
def test_vae_flag_and_latent_mode(tmp_path, setup, model_table, message):
    with pytest.raises(ValueError, match=message):
        resolve_test_config(write_test_toml(tmp_path, setup["test"], setup["checkpoint"], model_table=model_table))


def test_vae_flag_must_match_model(tmp_path, setup):
    with pytest.raises(ValueError, match="is variational"):
        run(tmp_path, setup, model_table="VAE = false\n")
    assert not (tmp_path / "TrainedModel" / "eval").exists()


def test_unknown_key_and_existing_folder(tmp_path, setup):
    with pytest.raises(ValueError, match="Unknown \\[abort_distance\\]"):
        resolve_test_config(write_test_toml(tmp_path, setup["test"], setup["checkpoint"],
                                            extra="[abort_distance]\nn_abort = 3\n"))
    (tmp_path / "TrainedModel" / "eval").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Test folder already exists"):
        resolve_test_config(write_test_toml(tmp_path, setup["test"], setup["checkpoint"]))


# ---------------------------------------------------------------- compatibility

def test_compatibility_errors_and_warnings(setup):
    model, checkpoint = load_checkpoint(setup["checkpoint"])
    test = load_test_set(setup["test"])
    assert check_compatibility(checkpoint, test, vae=True, model=model) == []
    for changes, message in ((dict(pvs=["OTHER:PV"]), "differ from the model's training PVs"),
                             (dict(value_transform="log10"), "value_transform differs"),
                             (dict(normalization_source={"dataset_id": "elsewhere"}), "normalized with statistics")):
        with pytest.raises(ValueError, match=message):
            check_compatibility(checkpoint, dataclasses.replace(test, **changes), vae=True, model=model)
    warnings = check_compatibility(checkpoint, dataclasses.replace(test, window_size=24, step_ns=2 * 10**9),
                                   vae=True, model=model)
    assert len(warnings) == 2 and "T = 24" in warnings[0] and "spacing" in warnings[1]


def test_window_length_mismatch_warns_then_reports_model_limit(tmp_path, setup):
    messages = []
    config = write_test_toml(tmp_path, setup["test_t24"], setup["checkpoint"])
    with pytest.raises(RuntimeError, match="could not process test windows"):
        evaluate_model(config, progress=messages.append)
    assert any(m.startswith("WARNING: Window length differs: test T = 24") for m in messages)


# ---------------------------------------------------------------- end to end

def test_results_folder_contents_and_scores(tmp_path, setup):
    folder = run(tmp_path, setup)
    assert folder == tmp_path / "TrainedModel" / "eval"     # next to the tested checkpoint
    assert {p.name for p in folder.iterdir()} == {"config.toml", "metadata.json", "window_scores.csv",
                                                  "ModelReconstructions", "ScoreVsAbortDistance", "RegionStatistics"}
    test = load_test_set(setup["test"])
    table = pd.read_csv(folder / "window_scores.csv")
    assert len(table) == test.n_windows
    # Scores equal a direct z_mean reconstruction MSE.
    model, _ = load_checkpoint(setup["checkpoint"])
    with h5py.File(setup["test"]) as h5:
        values = torch.from_numpy(h5["test/values"][:].astype(np.float32))
    with torch.no_grad():
        mu, _ = model.encode(values)
        direct = ((model.decode(mu) - values) ** 2).mean(dim=2).double().numpy()[:, 0]
    np.testing.assert_allclose(table[f"mse {PV}"], direct, rtol=1e-6)
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["warnings"] == [] and metadata["score"]["latent_mode"] == "z_mean"
    assert set(metadata["region_window_counts"]) == set(REGIONS)


def test_reconstructions_three_per_region(tmp_path, setup):
    folder = run(tmp_path, setup)
    metadata = json.loads((folder / "metadata.json").read_text())
    records = metadata["outputs"]["reconstructions"]
    counts = {region: sum(r["region"] == region for r in records) for region in REGIONS}
    assert counts == {region: 3 for region in REGIONS}
    record = next(r for r in records if r["region"] == "pre_abort")
    plotted = pd.read_csv(folder / record["csv"])
    assert len(plotted) == 32 and {f"{PV} input", f"{PV} output"} <= set(plotted.columns)
    scores = pd.read_csv(folder / "window_scores.csv").set_index("row")
    mse = ((plotted[f"{PV} output"] - plotted[f"{PV} input"]) ** 2).mean()
    assert mse == pytest.approx(scores.loc[record["row"], f"mse {PV}"], rel=1e-6)
    assert all((folder / r["png"]).is_file() for r in records)
    # one subfolder per label region, named like the region
    reconstructions = folder / "ModelReconstructions"
    assert {p.name for p in reconstructions.iterdir()} == set(REGIONS)
    for r in records:
        assert r["png"] == f"ModelReconstructions/{r['region']}/example_{r['example']}_row{r['row']}.png"


def test_score_vs_abort_distance(tmp_path, setup):
    folder = run(tmp_path, setup, extra="[abort_distance]\nn_aborts = 5\n")
    records = json.loads((folder / "metadata.json").read_text())["outputs"]["abort_distance"]
    assert [r["category"] for r in records] == ["SBL", "BeamLoss", "SBL", "Injection"]   # only 4 aborts exist
    # one folder per abort, one two-panel figure (+ two CSVs) per PV
    assert [p["pv"] for p in records[0]["pvs"]] == [PV]
    files = {p.name for p in (folder / records[0]["folder"]).iterdir()}
    assert files == {"SYN_D01_L01_PRES.png", "SYN_D01_L01_PRES_scores.csv", "SYN_D01_L01_PRES_signal.csv"}
    table = pd.read_csv(folder / records[0]["pvs"][0]["scores_csv"])
    pre = table[table["relation_to_this_abort"] == "pre_abort"]["distance_s"]
    contains = table[table["relation_to_this_abort"] == "abort"]["distance_s"]
    post = table[table["relation_to_this_abort"] == "post_abort"]["distance_s"]
    assert pre.between(-60, -1e-9).all() and contains.between(0, 31).all() and post.between(31, 90).all()
    assert len(pre) and len(contains) and len(post)
    assert (table["distance_s"].between(-60, 90)).all()        # non_abort_seconds = 0 by default


def test_abort_figure_shows_no_abort_time_and_window_outputs(tmp_path, setup):
    folder = run(tmp_path, setup, extra="[abort_distance]\nn_aborts = 1\nnon_abort_seconds = 120.0\n")
    record = json.loads((folder / "metadata.json").read_text())["outputs"]["abort_distance"][0]
    table = pd.read_csv(folder / record["pvs"][0]["scores_csv"])
    assert table["distance_s"].between(-180, 90).all()
    no_abort = table[table["relation_to_this_abort"] == "non_abort"]["distance_s"]
    # exclude_before_seconds = 120 > pre_abort_seconds = 60 in the synthetic build: the no-abort
    # windows shown lie before -120 s, the windows ending in (-120, -60) were excluded by the build.
    assert len(no_abort) and (no_abort < -120).all()
    signal = pd.read_csv(folder / record["pvs"][0]["signal_csv"])
    outputs = signal[signal["series"] == "output"]
    assert set(outputs["row"]) == set(table["row"]) and len(outputs) == 32 * len(table)   # T = 32 per window
    data = signal[signal["series"] == "input"]
    assert data["time_s"].is_monotonic_increasing and data["time_s"].min() == pytest.approx(table["distance_s"].min() - 31)
    # every plotted output equals the model output used for that window's score
    first = outputs[outputs["row"] == table["row"].iloc[0]]
    scored = data.set_index("time_s").loc[first["time_s"], "value"].to_numpy()
    mse = np.mean((first["value"].to_numpy() - scored) ** 2)
    assert mse == pytest.approx(table[f"mse {PV}"].iloc[0], rel=1e-6)


def test_region_statistics_match_window_scores(tmp_path, setup):
    folder = run(tmp_path, setup)
    scores = pd.read_csv(folder / "window_scores.csv")
    stats = pd.read_csv(folder / "RegionStatistics" / "region_statistics.csv").set_index("region")
    for region in REGIONS:
        members = scores[scores["regions"].fillna("").str.split(";").apply(lambda r: region in r)][f"mse {PV}"]
        expected = describe(members.to_numpy())
        assert stats.loc[region, "n"] == expected["n"] > 0
        for key in ("mean", "sem", "median"):
            assert stats.loc[region, key] == pytest.approx(expected[key], rel=1e-12)
    assert expected["sem"] == pytest.approx(members.std(ddof=1) / np.sqrt(len(members)))
    by_category = pd.read_csv(folder / "RegionStatistics" / "region_statistics_by_category.csv")
    assert set(by_category["category"]) == {"SBL", "BeamLoss", "Injection"}


def test_category_selection(tmp_path, setup):
    folder = run(tmp_path, setup, extra='[selection]\ncategories = ["sbl"]\n')
    records = json.loads((folder / "metadata.json").read_text())["outputs"]["abort_distance"]
    assert [r["category"] for r in records] == ["SBL", "SBL"]
    by_category = pd.read_csv(folder / "RegionStatistics" / "region_statistics_by_category.csv")
    assert set(by_category["category"]) == {"SBL"}


@pytest.mark.parametrize("categories", ['["RF"]', '["SBL", "Tuning"]'])
def test_absent_category_is_an_error(tmp_path, setup, categories):
    with pytest.raises(ValueError, match="no such aborts in the test set.*SBL \\(2\\)"):
        run(tmp_path, setup, extra=f"[selection]\ncategories = {categories}\n")
    assert not (tmp_path / "TrainedModel" / "eval").exists()


def test_event_ids_must_match_selected_categories(tmp_path, setup):
    test = load_test_set(setup["test"])
    beamloss = next(e["event_id"] for e in test.events if e["category"] == "BeamLoss")
    with pytest.raises(ValueError, match="are not of selection.categories"):
        run(tmp_path, setup, extra=f'[selection]\ncategories = ["SBL"]\n\n[abort_distance]\nevent_ids = ["{beamloss}"]\n')


def test_sampling_z_is_seeded_and_differs_from_z_mean(tmp_path, setup):
    sampled = 'VAE = true\nlatent_mode = "sampling_z"\n'
    a = pd.read_csv(run(tmp_path, setup, name="s1", model_table=sampled) / "window_scores.csv")
    b = pd.read_csv(run(tmp_path, setup, name="s2", model_table=sampled) / "window_scores.csv")
    mean = pd.read_csv(run(tmp_path, setup, name="m") / "window_scores.csv")
    np.testing.assert_array_equal(a[f"mse {PV}"], b[f"mse {PV}"])
    assert not np.array_equal(a[f"mse {PV}"], mean[f"mse {PV}"])


# ---------------------------------------------------------------- model independence

class _Smoother(TrainableModel):
    """Non-variational stand-in: forward returns a reconstruction-shaped output."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, values):
        return self.scale * values

    def compute_losses(self, batch):
        return {"loss": ((self(batch["values"]) - batch["values"]) ** 2).mean()}


class _Forecaster(_Smoother):
    def forward(self, values):
        return values[..., :-1]               # not reconstruction-shaped


def test_default_evaluation_output_contract():
    values = torch.randn(2, 1, 8)
    result = _Smoother().evaluation_output({"values": values})
    assert torch.equal(result["target"], values) and result["output"].shape == values.shape
    with pytest.raises(ValueError, match="not variational"):
        _Smoother().evaluation_output({"values": values}, latent_mode="z_mean")
    with pytest.raises(NotImplementedError, match="override evaluation_output"):
        _Forecaster().evaluation_output({"values": values})


def test_window_regions():
    pre = {"label": "pre_abort_SBL", "category": "SBL"}
    post = {"label": "post_abort_RF", "category": "RF"}
    assert window_regions(["non_abort"], []) == {"non_abort"}
    assert window_regions(["pre_abort_SBL", "post_abort_RF"], [pre, post]) == {"pre_abort", "post_abort"}
    assert window_regions(["pre_abort_SBL", "post_abort_RF"], [pre, post], ["rf"]) == {"post_abort"}
    assert window_regions(["SBL"], [{"label": "SBL", "category": "SBL"}], ["RF"]) == set()


# ---------------------------------------------------------------- forecasting models

from evaluation.scoring import check_time_index, score_windows
from evaluation_helpers import LinearForecaster, LinearForecasterHyperparameters


def linear_forecaster(horizon=8, window_size=32):
    return LinearForecaster(LinearForecasterHyperparameters(horizon=horizon), n_channels=1, window_size=window_size)


@pytest.mark.parametrize("horizon", [0, 32, 40])
def test_forecasting_horizon_must_leave_context(horizon):
    with pytest.raises(ValueError, match="horizon must satisfy"):
        linear_forecaster(horizon=horizon)


def test_forecasting_model_splits_window_into_context_and_horizon():
    torch.manual_seed(0)
    model = linear_forecaster().eval()
    values = torch.randn(4, 1, 32)
    losses = model.compute_losses({"values": values})
    with torch.no_grad():
        prediction = model.forecast(values[:, :, :24])
    assert losses["loss"].item() == pytest.approx((prediction - values[:, :, 24:]).square().mean().item())
    result = model.evaluation_output({"values": values})
    assert torch.equal(result["target"], values[:, :, 24:]) and result["output"].shape == (4, 1, 8)
    assert result["time_index"].tolist() == list(range(24, 32))
    with pytest.raises(ValueError, match="not variational"):
        model.evaluation_output({"values": values}, latent_mode="z_mean")
    with pytest.raises(ValueError, match="was built for"):
        model.evaluation_output({"values": torch.randn(4, 1, 24)})


def test_forecast_with_wrong_shape_is_rejected():
    model = linear_forecaster()
    model.forecast = lambda context: context[:, :, :5]          # 5 instead of horizon 8
    with pytest.raises(ValueError, match=r"forecast returned \(2, 1, 5\), expected \(2, 1, 8\)"):
        model.compute_losses({"values": torch.randn(2, 1, 32)})


@pytest.mark.parametrize("time_index, steps, message", [
    (None, 8, "without a time_index"),
    (np.arange(24, 31), 8, "has 7 entries"),
    (np.arange(25, 33), 8, "strictly increasing window positions"),
    (np.array([24, 26, 25, 27, 28, 29, 30, 31]), 8, "strictly increasing"),
    (np.arange(24, 32, dtype=float), 8, "integer"),
])
def test_time_index_validation(time_index, steps, message):
    with pytest.raises(ValueError, match=message):
        check_time_index(time_index, steps, 32)


def test_time_index_must_not_change_between_batches(setup):
    class Drifting(TrainableModel):
        calls = 0

        def compute_losses(self, batch):
            raise NotImplementedError

        def evaluation_output(self, batch, *, latent_mode=None):
            Drifting.calls += 1
            start = 24 if Drifting.calls == 1 else 23
            values = batch["values"]
            return {"output": values[:, :, start:start + 8], "target": values[:, :, start:start + 8],
                    "time_index": torch.arange(start, start + 8)}

    with pytest.raises(ValueError, match="time_index changed between batches"):
        score_windows(Drifting(), load_test_set(setup["test"]), batch_size=50, device="cpu", latent_mode=None)


def test_forecaster_trains_and_is_tested_end_to_end(tmp_path, setup, forecaster):
    folder = evaluate_model(write_test_toml(tmp_path, setup["test"], forecaster, model_table="VAE = false\n"),
                            progress=None)
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["score"]["output"] == {"kind": "forecast", "samples": 8, "first_window_position": 24,
                                           "last_window_position": 31, "time_index": list(range(24, 32))}
    # Score = MSE of the prediction on the last 8 samples only.
    model, checkpoint = load_checkpoint(forecaster)
    assert checkpoint["model_name"] == "test_linear_forecaster"
    with h5py.File(setup["test"]) as h5:
        values = torch.from_numpy(h5["test/values"][:].astype(np.float32))
    with torch.no_grad():
        direct = ((model.forecast(values[:, :, :24]) - values[:, :, 24:]) ** 2).mean(dim=2).double().numpy()[:, 0]
    table = pd.read_csv(folder / "window_scores.csv")
    np.testing.assert_allclose(table[f"mse {PV}"], direct, rtol=1e-6)
    # Reconstruction CSVs hold the whole window; target/output only at the predicted samples.
    record = metadata["outputs"]["reconstructions"][0]
    plotted = pd.read_csv(folder / record["csv"])
    assert len(plotted) == 32 and plotted["predicted"].sum() == 8 and plotted["predicted"].iloc[24:].all()
    assert plotted[f"{PV} output"].iloc[:24].isna().all() and plotted[f"{PV} input"].notna().all()
    np.testing.assert_allclose(plotted[f"{PV} target"].iloc[24:], plotted[f"{PV} input"].iloc[24:])
    mse = ((plotted[f"{PV} output"] - plotted[f"{PV} target"]) ** 2).mean()   # NaN rows are skipped
    assert mse == pytest.approx(table.set_index("row").loc[record["row"], f"mse {PV}"], rel=1e-6)
    assert (folder / record["png"]).is_file()


def test_reconstruction_metadata_and_full_window_csv(tmp_path, setup):
    folder = run(tmp_path, setup)
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["score"]["output"] == {"kind": "reconstruction", "samples": 32}
    plotted = pd.read_csv(folder / metadata["outputs"]["reconstructions"][0]["csv"])
    assert plotted["predicted"].all()
    np.testing.assert_allclose(plotted[f"{PV} target"], plotted[f"{PV} input"])


# ---------------------------------------------------------------- threshold exceedance per abort

from evaluation.data import relation


def test_exceedance_fractions_match_definition(tmp_path, setup):
    folder = run(tmp_path, setup, extra="[exceedance]\npercentiles = [90.0, 99.0]\n")
    table = pd.read_csv(folder / "window_scores.csv")
    score = table[f"mse {PV}"].to_numpy()
    no_abort = score[(table["regions"] == "non_abort").to_numpy()]
    test = load_test_set(setup["test"])
    per_abort = {}
    for row, associations in enumerate(test.associations):
        for a in associations:
            per_abort.setdefault(a["event_id"], {}).setdefault(relation(a["label"]), []).append(row)
    summary = pd.read_csv(folder / "RegionStatistics" / "exceedance.csv")
    assert len(summary) == 2 * 3 and set(summary["aborts_total"]) == {len(per_abort)} == {4}
    for p in (90.0, 99.0):
        threshold = np.percentile(no_abort, p)
        for region in ("pre_abort", "abort", "post_abort"):
            hits = sum(bool((score[rows.get(region, [])] > threshold).any()) for rows in per_abort.values())
            got = summary[(summary["percentile"] == p) & (summary["region"] == region)].iloc[0]
            assert got["threshold"] == pytest.approx(threshold)
            assert got["aborts_exceeded"] == hits and got["fraction"] == pytest.approx(hits / 4)
    by_abort = pd.read_csv(folder / "RegionStatistics" / "exceedance_by_abort.csv")
    assert len(by_abort) == 2 * 4 * 3          # percentiles x aborts x regions
    assert (folder / "RegionStatistics" / "exceedance_SYN_D01_L01_PRES.png").is_file()
    record = json.loads((folder / "metadata.json").read_text())["outputs"]["exceedance"]
    assert record["aborts_total"] == 4 and record["percentiles"] == [90.0, 99.0]


def test_exceedance_uses_selected_categories(tmp_path, setup):
    folder = run(tmp_path, setup, extra='[selection]\ncategories = ["SBL"]\n')
    summary = pd.read_csv(folder / "RegionStatistics" / "exceedance.csv")
    assert set(summary["aborts_total"]) == {2} and set(summary["percentile"]) == {95.0, 99.0, 99.9}   # defaults
    assert (summary["aborts_exceeded"] <= summary["aborts_total"]).all()


@pytest.mark.parametrize("percentiles, message", [
    ("[0.0]", "strictly between 0 and 100"), ("[99.0, 100.0]", "strictly between 0 and 100"),
    ("[99.0, 99.0]", "must not repeat"), ("99.0", "array"),
])
def test_exceedance_percentiles_validated(tmp_path, setup, percentiles, message):
    with pytest.raises(ValueError, match=message):
        resolve_test_config(write_test_toml(tmp_path, setup["test"], setup["checkpoint"],
                                            extra=f"[exceedance]\npercentiles = {percentiles}\n"))
