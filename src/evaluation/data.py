"""Read a test-mode dataset.h5, assign label regions, check model compatibility."""

from dataclasses import dataclass, field
import json

import h5py
import numpy as np


REGIONS = ("abort", "pre_abort", "post_abort", "non_abort")


@dataclass
class EvalDataset:
    path: str
    dataset_id: str
    values_state: str
    value_transform: str
    channel_ids: list
    pvs: list                       # expected_pv per channel, in channel order
    window_size: int
    step_ns: int
    pre_abort_ns: int
    post_abort_ns: int
    normalization_source: dict | None
    starts: np.ndarray              # int64 (N,)
    ends: np.ndarray                # int64 (N,), last included sample
    labels: list                    # N lists of label strings
    associations: list              # N lists of event-association dicts
    events: list = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return int(self.starts.size)


def load_test_set(path) -> EvalDataset:
    """Metadata, times and labels of a complete mode="test" dataset (values are read later in batches)."""
    with h5py.File(path, "r") as data:
        if not bool(data.attrs.get("complete", False)):
            raise ValueError(f"Dataset is incomplete: {path}")
        if data.attrs.get("mode") != "test":
            raise ValueError(f"Testing needs a mode='test' dataset (labels); found mode={data.attrs.get('mode')!r}")
        build = json.loads(data["build_metadata_json"].asstr()[()])
        settings = build["resolved_config"]["dataset"]
        group = data["test"]
        return EvalDataset(
            path=str(path), dataset_id=str(data.attrs["dataset_id"]), values_state=str(data.attrs["values_state"]),
            value_transform=str(data.attrs.get("value_transform", "none")),
            channel_ids=data["channel_ids"].asstr()[:].tolist(),
            pvs=[entry["expected_pv"] for entry in build["channel_identity"]],
            window_size=int(data.attrs["window_size"]), step_ns=int(data.attrs["step_ns"]),
            pre_abort_ns=int(settings["pre_abort_ns"]), post_abort_ns=int(settings["post_abort_ns"]),
            normalization_source=build.get("normalization_source"),
            starts=group["window_start_ns"][:], ends=group["window_end_ns"][:],
            labels=[json.loads(text) for text in group["labels_json"].asstr()[:]],
            associations=[json.loads(text) for text in group["event_associations_json"].asstr()[:]],
            events=json.loads(data["events_json"].asstr()[()]),
        )


def iter_values(path, batch_size):
    """Yield (first_row, float32 (B, C, T)) in row order; bounded memory."""
    with h5py.File(path, "r") as data:
        values = data["test/values"]
        for start in range(0, values.shape[0], batch_size):
            yield start, values[start:start + batch_size].astype(np.float32)


def relation(label: str) -> str:
    """Region of one association label: category x -> abort, pre_abort_x, post_abort_x."""
    if label == "non_abort":
        return "non_abort"
    if label.startswith("pre_abort_"):
        return "pre_abort"
    if label.startswith("post_abort_"):
        return "post_abort"
    return "abort"


def window_regions(labels, associations, categories=None) -> set:
    """Regions of one window. A window can belong to several (different events).

    categories (case-insensitive) restricts which abort categories count for
    the abort/pre_abort/post_abort regions; windows related only to other
    categories get an empty set and are left out of region results.
    """
    if labels == ["non_abort"]:
        return {"non_abort"}
    wanted = None if not categories else {c.casefold() for c in categories}
    return {relation(a["label"]) for a in associations if wanted is None or a["category"].casefold() in wanted}


def check_compatibility(checkpoint: dict, test: EvalDataset, *, vae: bool, model) -> list:
    """Raise on errors that make scores meaningless; return warnings for shape differences."""
    warnings = []
    if bool(vae) != bool(model.is_variational):
        raise ValueError(f"model.VAE = {str(vae).lower()} but the checkpoint's model {checkpoint['model_name']!r} "
                         f"is {'variational' if model.is_variational else 'not variational'}")
    if test.pvs != list(checkpoint["pvs"]):
        raise ValueError(f"Test channels {test.pvs} differ from the model's training PVs {checkpoint['pvs']} "
                         "(identity and order must match)")
    if checkpoint.get("value_transform", "none") != test.value_transform:
        raise ValueError(f"value_transform differs: model trained on {checkpoint.get('value_transform', 'none')!r}, "
                         f"test set is {test.value_transform!r}")
    if checkpoint.get("values_state") != test.values_state:
        raise ValueError(f"Model trained on {checkpoint.get('values_state')!r} values, test set is {test.values_state!r}")
    if test.values_state == "normalized":
        source = (test.normalization_source or {}).get("dataset_id")
        if source != checkpoint.get("dataset_id"):
            raise ValueError(f"Test set was normalized with statistics of dataset {source!r}, but the model was "
                             f"trained on dataset {checkpoint.get('dataset_id')!r}; rebuild the test set with "
                             "statistics_filepath pointing at the model's training dataset.h5")
    if test.window_size != checkpoint["window_size"]:
        warnings.append(f"Window length differs: test T = {test.window_size}, model expects T = "
                        f"{checkpoint['window_size']}; the model is still applied")
    if "step_ns" in checkpoint and test.step_ns != checkpoint["step_ns"]:
        warnings.append(f"Sample spacing differs: test {test.step_ns / 1e9:g} s, model trained on "
                        f"{checkpoint['step_ns'] / 1e9:g} s")
    return warnings
