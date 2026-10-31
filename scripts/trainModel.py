"""Train a model: python scripts/trainModel.py --config <toml>.

Equivalent to `python -m train --config <toml>`; all settings come from the
TOML (see src/train/README.md and src/models/cnn_vae/exampleTrain.toml).
"""

from train.train import main


if __name__ == "__main__":
    raise SystemExit(main())
