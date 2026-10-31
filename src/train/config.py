"""Validate a training TOML into typed, immutable settings.

Tables:
    [run]        trained-model folder name, seed, device, project_root
    [data]       which completed dataset.h5 to read and how to load it
    [[pvs]]      one entry per model channel: exact PV identity (+ folder names)
    [optimizer]  optimizer and its settings
    [training]   epochs, batch size, early stopping, gradient clipping
    [model]      ``name`` selects a registered model in src/models; every other
                 key is parsed into THAT model's hyperparameter dataclass

Window size, stride/overlap, selection and normalization are not configured
here: they were fixed by the dataset build and are read from dataset.h5.

Output location (inside project_root):
    one PV     UnivariateTrainedModels/<pv type>/<pv name>/<run name>/
    several    MultivariateTrainedModels/<run name>/
"""

from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
import math
from pathlib import Path
import re
import tomllib
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from models import get_model_spec


UNIVARIATE_ROOT = "UnivariateTrainedModels"
MULTIVARIATE_ROOT = "MultivariateTrainedModels"
_FOLDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


# --------------------------------------------------------------------------
# Generic "TOML table -> frozen dataclass" conversion, shared by the settings
# below and by every model's hyperparameter dataclass.
# --------------------------------------------------------------------------

def _coerce(value, annotation, name):
    """Check one TOML value against a dataclass field annotation.

    Supported: bool, int, float, str, Literal[...], tuple[X, ...] (a TOML
    array), ``X | None`` (TOML has no null: omit the key to keep None) and
    unions such as ``int | tuple[int, ...]`` (one shared value or one per
    layer; the first matching option wins). Integers are accepted where
    floats are expected.
    """
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        options = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(options) == 1:
            return _coerce(value, options[0], name)
        errors = []
        for option in options:
            try:
                return _coerce(value, option, name)
            except ValueError as error:
                errors.append(str(error))
        raise ValueError(f"{name} = {value!r} matches none of the allowed forms: " + "; ".join(errors))
    if origin is Literal:
        choices = get_args(annotation)
        if value not in choices:
            raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")
        return value
    if origin is tuple:
        element_type, ellipsis = get_args(annotation)
        if ellipsis is not Ellipsis:
            raise TypeError(f"{name}: only tuple[X, ...] is supported")
        if not isinstance(value, list) or not value:
            raise ValueError(f"{name} must be a nonempty TOML array")
        return tuple(_coerce(item, element_type, f"{name}[{index}]") for index, item in enumerate(value))
    if annotation is bool:
        if type(value) is not bool:
            raise ValueError(f"{name} must be true or false")
        return value
    if annotation is int:
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"{name} must be a nonempty string")
        return value
    raise TypeError(f"{name}: unsupported annotation {annotation}")


def parse_table(cls, table, section):
    """Build the frozen dataclass ``cls`` from one TOML table.

    Unknown keys and missing required keys (fields without defaults) are
    errors, so a typo cannot silently fall back to a default. Range checks
    live in each dataclass's ``__post_init__``.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    if not isinstance(table, dict):
        raise ValueError(f"[{section}] must be a TOML table")
    hints = get_type_hints(cls)
    init_fields = [field for field in fields(cls) if field.init]
    allowed = {field.name for field in init_fields}
    unknown = set(table) - allowed
    if unknown:
        raise ValueError(f"Unknown [{section}] settings {sorted(unknown)}; allowed: {sorted(allowed)}")
    values = {}
    for field in init_fields:
        if field.name in table:
            values[field.name] = _coerce(table[field.name], hints[field.name], f"{section}.{field.name}")
        elif field.default is MISSING and field.default_factory is MISSING:
            raise ValueError(f"Missing [{section}] setting: {field.name}")
    return cls(**values)


def _folder_name(value, name):
    if value is not None and not _FOLDER_NAME.fullmatch(value):
        raise ValueError(f"{name} = {value!r} must be a single folder name of letters, digits, '.', '_' or '-'")


# --------------------------------------------------------------------------
# Model-independent settings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSettings:
    name: str                          # trained-model folder name
    seed: int                          # seeds Python, NumPy, torch and the shuffle order
    project_root: str = "."            # relative to the TOML's folder
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    verbose: int = 1                   # 1: live progress within each epoch / scoring pass; 0: off

    def __post_init__(self):
        _folder_name(self.name, "run.name")
        if not 0 <= self.seed < 2**32:
            raise ValueError("run.seed must be in [0, 2**32)")
        if self.verbose not in (0, 1):
            raise ValueError("run.verbose must be 0 or 1")


@dataclass(frozen=True)
class DataSettings:
    dataset_path: str                  # completed train_val dataset.h5
    in_memory: bool = True             # load each split once as float32 (C*T*4 bytes per window)
    num_workers: int = 0               # DataLoader worker processes

    def __post_init__(self):
        if self.num_workers < 0:
            raise ValueError("data.num_workers must be nonnegative")


@dataclass(frozen=True)
class PVSettings:
    pv: str                            # exact PV identity, e.g. "VALCCG:D01_L01A:PRES"
    type: str | None = None            # e.g. "Pressure"; folder level 1 (univariate)
    name: str | None = None            # e.g. "D01_L01A"; folder level 2 (univariate)

    def __post_init__(self):
        _folder_name(self.type, "pvs.type")
        _folder_name(self.name, "pvs.name")


@dataclass(frozen=True)
class OptimizerSettings:
    name: Literal["adam", "adamw"]
    learning_rate: float
    weight_decay: float
    betas: tuple[float, ...] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("optimizer.learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("optimizer.weight_decay must be nonnegative")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError("optimizer.betas must be two numbers in [0, 1)")
        if self.eps <= 0:
            raise ValueError("optimizer.eps must be positive")


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int
    batch_size: int                    # ML minibatch size (windows per optimizer step)
    gradient_clip_norm: float | None = None       # omit to disable clipping
    early_stopping_patience: int | None = None    # omit to always run all epochs

    def __post_init__(self):
        for name in ("epochs", "batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("training.gradient_clip_norm must be positive (omit it to disable)")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("training.early_stopping_patience must be positive (omit it to disable)")


@dataclass(frozen=True)
class TrainConfig:
    """Resolved configuration: absolute paths plus validated settings."""

    config_path: Path
    project_root: Path
    dataset_path: Path
    output_dir: Path
    layout: str                        # "univariate" or "multivariate"
    run: RunSettings
    data: DataSettings
    pvs: tuple[PVSettings, ...]
    optimizer: OptimizerSettings
    training: TrainingSettings
    model_name: str
    model: Any                         # the selected model's hyperparameter dataclass

    def to_dict(self) -> dict:
        """JSON-ready snapshot with every default filled in."""
        return {
            "config_path": str(self.config_path), "project_root": str(self.project_root),
            "dataset_path": str(self.dataset_path), "output_dir": str(self.output_dir),
            "layout": self.layout, "run": asdict(self.run), "data": asdict(self.data),
            "pvs": [asdict(pv) for pv in self.pvs], "optimizer": _listify(asdict(self.optimizer)),
            "training": asdict(self.training),
            "model": {"name": self.model_name, **_listify(asdict(self.model))},
        }


def _listify(mapping):
    return {key: list(value) if isinstance(value, tuple) else value for key, value in mapping.items()}


def load_toml(path):
    with open(path, "rb") as stream:
        return tomllib.load(stream)


def resolve_config(config_path) -> TrainConfig:
    """Validate the TOML, resolve paths against [run].project_root and choose the output folder."""
    config_path = Path(config_path).expanduser().resolve()
    raw = load_toml(config_path)
    expected = {"run", "data", "pvs", "optimizer", "training", "model"}
    if set(raw) != expected:
        raise ValueError(f"Configuration must contain exactly the tables {sorted(expected)}; found {sorted(raw)}")

    run = parse_table(RunSettings, raw["run"], "run")
    data = parse_table(DataSettings, raw["data"], "data")
    optimizer = parse_table(OptimizerSettings, raw["optimizer"], "optimizer")
    training = parse_table(TrainingSettings, raw["training"], "training")

    if not isinstance(raw["pvs"], list) or not raw["pvs"]:
        raise ValueError("Provide at least one [[pvs]] entry")
    pvs = tuple(parse_table(PVSettings, table, f"pvs[{index}]") for index, table in enumerate(raw["pvs"]))
    if len({pv.pv for pv in pvs}) != len(pvs):
        raise ValueError("[[pvs]] must not repeat a PV")

    model_table = dict(raw["model"]) if isinstance(raw["model"], dict) else None
    if model_table is None or "name" not in model_table:
        raise ValueError("[model] must be a table with a 'name' key")
    model_name = _coerce(model_table.pop("name"), str, "model.name")
    model = parse_table(get_model_spec(model_name).hyperparameters, model_table, "model")

    root = Path(run.project_root).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve()
    dataset_path = Path(data.dataset_path).expanduser()
    dataset_path = (dataset_path if dataset_path.is_absolute() else root / dataset_path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"data.dataset_path does not exist: {dataset_path}")

    if len(pvs) == 1:
        if pvs[0].type is None or pvs[0].name is None:
            raise ValueError("A univariate run needs pvs.type and pvs.name (they define its folder)")
        layout = "univariate"
        output_dir = root / UNIVARIATE_ROOT / pvs[0].type / pvs[0].name / run.name
    else:
        layout = "multivariate"
        output_dir = root / MULTIVARIATE_ROOT / run.name
    if output_dir.exists():
        raise FileExistsError(f"Trained-model folder already exists: {output_dir}; choose another run.name")

    return TrainConfig(
        config_path=config_path, project_root=root, dataset_path=dataset_path, output_dir=output_dir,
        layout=layout, run=run, data=data, pvs=pvs, optimizer=optimizer, training=training,
        model_name=model_name, model=model,
    )
