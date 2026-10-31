"""Model-independent testing of trained models on mode='test' belle2-dataset files."""

__version__ = "0.1.0"


def evaluate_model(config_path, *, progress=print):
    """Test one model from TOML; return the results folder's Path."""
    from .evaluate import evaluate_model as run

    return run(config_path, progress=progress)


__all__ = ["evaluate_model", "__version__"]
