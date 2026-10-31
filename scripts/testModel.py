"""Test a trained model: python scripts/testModel.py --config <toml>.

Equivalent to `python -m evaluation --config <toml>`; all settings come from
the TOML (see src/evaluation/README.md and src/evaluation/example.toml).
"""

from evaluation.evaluate import main


if __name__ == "__main__":
    raise SystemExit(main())
