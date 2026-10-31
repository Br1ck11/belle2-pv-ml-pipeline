"""Model-independent PyTorch training for belle2-dataset HDF5 files.

Models are defined in src/models and selected by the TOML [model].name.
"""

__version__ = "0.1.0"


def train_model(config_path, *, progress=print):
    """Train one model from TOML; return the trained-model folder's Path."""
    from .train import train_model as run

    return run(config_path, progress=progress)


__all__ = ["train_model", "__version__"]
