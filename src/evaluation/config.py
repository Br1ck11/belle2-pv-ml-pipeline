"""Validate a model-test TOML into typed, immutable settings.

Tables: [run], [data], [model] (required); [selection], [reconstructions],
[abort_distance], [exceedance] (optional, defaults below). Parsing reuses the strict
TOML -> dataclass conversion of the training package: unknown keys and
missing required keys are errors.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Literal

from train.config import load_toml, parse_table


_FOLDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class RunSettings:
    name: str                                    # -> <trained-model folder of the checkpoint>/<name>/
    seed: int                                    # seeds sampling_z and the random window choices
    project_root: str = "."
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    verbose: int = 1                   # 1: live progress within each epoch / scoring pass; 0: off

    def __post_init__(self):
        if not _FOLDER_NAME.fullmatch(self.name):
            raise ValueError(f"run.name = {self.name!r} must be one folder name of letters, digits, '.', '_' or '-'")
        if not 0 <= self.seed < 2**32:
            raise ValueError("run.seed must be in [0, 2**32)")
        if self.verbose not in (0, 1):
            raise ValueError("run.verbose must be 0 or 1")


@dataclass(frozen=True)
class DataSettings:
    dataset_path: str                            # completed dataset.h5 with mode = "test"
    batch_size: int = 512                        # windows per forward pass

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("data.batch_size must be positive")


@dataclass(frozen=True)
class ModelSettings:
    checkpoint: str                              # best.pt / last.pt of a trained-model folder
    VAE: bool                                    # true: the model has a stochastic latent
    latent_mode: Literal["z_mean", "sampling_z"] | None = None   # required iff VAE = true

    def __post_init__(self):
        if self.VAE and self.latent_mode is None:
            raise ValueError("model.VAE = true requires model.latent_mode = 'z_mean' or 'sampling_z'")
        if not self.VAE and self.latent_mode is not None:
            raise ValueError("model.latent_mode is only used when model.VAE = true")


@dataclass(frozen=True)
class SelectionSettings:
    categories: tuple[str, ...] | None = None    # abort categories to evaluate, e.g. ["SBL"]; omit = all;
                                                 # each must occur in the test set (else error)
    pvs: tuple[str, ...] | None = None           # PVs (expected_pv) for score plots/statistics; omit = all channels


@dataclass(frozen=True)
class ReconstructionSettings:
    windows_per_region: int = 3
    sample: Literal["random", "first"] = "random"

    def __post_init__(self):
        if self.windows_per_region <= 0:
            raise ValueError("reconstructions.windows_per_region must be positive")


@dataclass(frozen=True)
class AbortDistanceSettings:
    n_aborts: int = 5
    selection: Literal["first", "random"] = "first"   # chronological or seeded random choice of aborts
    event_ids: tuple[str, ...] | None = None          # explicit aborts instead; omit to use n_aborts/selection
    non_abort_seconds: float = 0.0                    # no-abort time shown before the pre-abort region

    def __post_init__(self):
        if self.n_aborts <= 0:
            raise ValueError("abort_distance.n_aborts must be positive")
        if self.non_abort_seconds < 0:
            raise ValueError("abort_distance.non_abort_seconds must be nonnegative")


@dataclass(frozen=True)
class ExceedanceSettings:
    # Thresholds = these percentiles of the test set's no-abort window scores (per PV).
    percentiles: tuple[float, ...] = (95.0, 99.0, 99.9)

    def __post_init__(self):
        if not all(0 < p < 100 for p in self.percentiles):
            raise ValueError("exceedance.percentiles must lie strictly between 0 and 100")
        if len(set(self.percentiles)) != len(self.percentiles):
            raise ValueError("exceedance.percentiles must not repeat")


@dataclass(frozen=True)
class EvaluationConfig:
    config_path: Path
    project_root: Path
    dataset_path: Path
    checkpoint_path: Path
    output_dir: Path
    run: RunSettings
    data: DataSettings
    model: ModelSettings
    selection: SelectionSettings
    reconstructions: ReconstructionSettings
    abort_distance: AbortDistanceSettings
    exceedance: ExceedanceSettings

    def to_dict(self) -> dict[str, Any]:
        result = {key: str(getattr(self, key)) for key in
                  ("config_path", "project_root", "dataset_path", "checkpoint_path", "output_dir")}
        for key in ("run", "data", "model", "selection", "reconstructions", "abort_distance", "exceedance"):
            result[key] = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(getattr(self, key)).items()}
        return result


def resolve_test_config(config_path) -> EvaluationConfig:
    config_path = Path(config_path).expanduser().resolve()
    raw = load_toml(config_path)
    required, optional = {"run", "data", "model"}, {"selection", "reconstructions", "abort_distance", "exceedance"}
    if not required <= set(raw) <= required | optional:
        raise ValueError(f"Configuration needs tables {sorted(required)} and may contain {sorted(optional)}; "
                         f"found {sorted(raw)}")
    run = parse_table(RunSettings, raw["run"], "run")
    data = parse_table(DataSettings, raw["data"], "data")
    model = parse_table(ModelSettings, raw["model"], "model")
    selection = parse_table(SelectionSettings, raw.get("selection", {}), "selection")
    reconstructions = parse_table(ReconstructionSettings, raw.get("reconstructions", {}), "reconstructions")
    abort_distance = parse_table(AbortDistanceSettings, raw.get("abort_distance", {}), "abort_distance")
    exceedance = parse_table(ExceedanceSettings, raw.get("exceedance", {}), "exceedance")

    root = Path(run.project_root).expanduser()
    root = (root if root.is_absolute() else config_path.parent / root).resolve()

    def existing(text, name):
        path = Path(text).expanduser()
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        return path

    checkpoint_path = existing(model.checkpoint, "model.checkpoint")
    # Results live next to the model they test: <trained-model folder>/<run.name>/
    output_dir = checkpoint_path.parent / run.name
    if output_dir.exists():
        raise FileExistsError(f"Test folder already exists: {output_dir}; choose another run.name")
    return EvaluationConfig(
        config_path=config_path, project_root=root,
        dataset_path=existing(data.dataset_path, "data.dataset_path"),
        checkpoint_path=checkpoint_path, output_dir=output_dir,
        run=run, data=data, model=model, selection=selection, reconstructions=reconstructions,
        abort_distance=abort_distance, exceedance=exceedance,
    )
