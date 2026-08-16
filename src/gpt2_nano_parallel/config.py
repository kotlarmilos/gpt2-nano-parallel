"""TOML configuration loading."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    required = {"model", "parallel"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    return config
