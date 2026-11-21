from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path = PROJECT_ROOT / "configs/config.yaml") -> dict:
    """Load the configuration from a YAML file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config