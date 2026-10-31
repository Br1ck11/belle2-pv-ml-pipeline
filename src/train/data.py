"""Read completed belle2-dataset train/val splits as PyTorch datasets.

All scientific choices (window size, stride/overlap, channel order, current
and abort selection, normalization) were fixed when the dataset was built.
This module only checks and reads them; it never re-windows or re-normalizes.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SPLITS = ("train", "val")


@dataclass(frozen=True)
class DatasetInfo:
    """What a model and a run's metadata need to know about dataset.h5."""

    path: str
    dataset_id: str
    values_state: str                 # "normalized" or "raw"
    value_transform: str              # "none" or "log10" (dataset.log_transform)
    channel_ids: tuple[str, ...]      # C axis order
    channel_identity: list            # [{"id", "expected_pv"}, ...]
    window_size: int                  # T
    step_ns: int                      # grid spacing
    split_counts: dict[str, int]
    split_intervals_ns: dict[str, list[int]]
    normalization: dict               # saved training statistics (offset/scale per channel)

    @property
    def n_channels(self) -> int:
        return len(self.channel_ids)

    def to_dict(self) -> dict:
        return {**self.__dict__, "channel_ids": list(self.channel_ids)}


def inspect_dataset(path) -> DatasetInfo:
    """Validate a completed train_val dataset.h5 and return its description."""
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as data:
        if not bool(data.attrs.get("complete", False)):
            raise ValueError(f"Dataset is incomplete (complete != True): {path}")
        if int(data.attrs.get("schema_version", -1)) != 1:
            raise ValueError("Only belle2-dataset schema_version 1 is supported")
        if data.attrs.get("mode") != "train_val":
            raise ValueError(f"Training needs a train_val dataset, found mode={data.attrs.get('mode')!r}")
        values_state = str(data.attrs.get("values_state"))
        if values_state not in ("normalized", "raw"):
            raise ValueError(f"Unexpected values_state {values_state!r}; the build may have failed")
        channel_ids = tuple(data["channel_ids"].asstr()[:].tolist())
        window_size = int(data.attrs["window_size"])
        counts = {}
        for split in SPLITS:
            if split not in data:
                raise ValueError(f"Dataset has no {split!r} split")
            shape = data[f"{split}/values"].shape
            if len(shape) != 3 or shape[1:] != (len(channel_ids), window_size):
                raise ValueError(f"{split}/values has shape {shape}, expected (N, {len(channel_ids)}, {window_size})")
            if shape[0] == 0:
                raise ValueError(f"Split {split!r} is empty")
            counts[split] = int(shape[0])
        build = json.loads(data["build_metadata_json"].asstr()[()])
        return DatasetInfo(
            path=str(path), dataset_id=str(data.attrs["dataset_id"]), values_state=values_state,
            value_transform=str(data.attrs.get("value_transform", "none")),
            channel_ids=channel_ids, channel_identity=build["channel_identity"], window_size=window_size,
            step_ns=int(data.attrs["step_ns"]), split_counts=counts,
            split_intervals_ns=build["provenance"]["split_intervals_ns"],
            normalization=json.loads(data["normalization_json"].asstr()[()]),
        )


def check_pvs(pvs, info: DatasetInfo) -> None:
    """Require the TOML [[pvs]] to name exactly the dataset's channels, in order.

    The comparison uses the exact stored PV identity (expected_pv), not the
    dataset's channel alias, so a TOML cannot silently train on another PV.
    """
    stored = [channel["expected_pv"] for channel in info.channel_identity]
    requested = [pv.pv for pv in pvs]
    if requested != stored:
        raise ValueError(f"[[pvs]] {requested} do not match the dataset channels {stored} (same order required)")


class H5WindowDataset(Dataset):
    """Map-style access to one split; items are {"values": (C, T) float32, "index": int}.

    in_memory=True reads the whole split once and converts it to float32
    (N * C * T * 4 bytes). in_memory=False opens the HDF5 file lazily per
    process and reads one window per item; a DataLoader worker never reuses
    a handle opened by its parent.
    """

    def __init__(self, path, split: str, *, in_memory: bool) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.path = Path(path).expanduser().resolve()
        self.split = split
        self.in_memory = in_memory
        self._file: h5py.File | None = None
        self._process_id: int | None = None
        self._tensor: torch.Tensor | None = None
        with h5py.File(self.path, "r") as data:
            values = data[f"{split}/values"]
            self.length = int(values.shape[0])
            if in_memory:
                # float64 on disk -> float32 model input; one deliberate copy.
                self._tensor = torch.from_numpy(values[:].astype(np.float32))

    def __len__(self) -> int:
        return self.length

    def _values(self) -> h5py.Dataset:
        process_id = os.getpid()
        if self._file is None or self._process_id != process_id:
            self.close()
            self._file = h5py.File(self.path, "r")
            self._process_id = process_id
        return self._file[f"{self.split}/values"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.length:
            raise IndexError(f"Window index {index} is outside [0, {self.length})")
        if self._tensor is not None:
            values = self._tensor[index]
        else:
            values = torch.from_numpy(self._values()[index].astype(np.float32))
        return {"values": values, "index": index}

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._process_id = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_process_id"] = None
        return state

    def __del__(self) -> None:
        self.close()


def make_loaders(train_dataset, val_dataset, *, batch_size, num_workers, seed, device):
    """Shuffled training loader (seeded order) and fixed-order validation loader."""
    common = {
        "batch_size": batch_size, "num_workers": num_workers,
        "pin_memory": device.type == "cuda", "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader
