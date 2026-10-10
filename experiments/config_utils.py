"""Shared strict YAML configuration loading for experiment runners."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


ConfigT = TypeVar("ConfigT")


def load_dataclass_config(
    path: str | Path,
    config_type: type[ConfigT],
    *,
    required: set[str] | None = None,
    ignored_keys: set[str] | None = None,
) -> ConfigT:
    """Load a YAML mapping into a dataclass and reject unknown/missing keys."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration must contain a YAML mapping: {config_path}")
    field_names = {field.name for field in fields(config_type)}
    ignored = ignored_keys or {"experiment"}
    unknown = set(loaded) - field_names - ignored
    if unknown:
        raise ValueError(f"unknown configuration keys in {config_path}: {sorted(unknown)}")
    required_keys = required or set()
    missing = required_keys - set(loaded)
    if missing:
        raise ValueError(f"missing required configuration keys: {sorted(missing)}")
    values = {key: value for key, value in loaded.items() if key in field_names}
    try:
        return config_type(**values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid configuration in {config_path}: {error}") from error
