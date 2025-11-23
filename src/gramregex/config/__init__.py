"""Configuration helpers for grammar loading."""

from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, Field, ValidationError

DEFAULT_CONFIG_PATH = files("gramregex") / "config" / "default.yaml"


class GrammarConfig(BaseModel):
    """Grammar configuration loaded from YAML."""

    name: str = Field(description="Configuration name")
    description: str = Field(description="Human readable description of the grammar")
    content: str = Field(description="Grammar definition content")

    @classmethod
    def from_yaml(cls, path: Path) -> "GrammarConfig":
        """Load a grammar config from the given YAML path."""
        if not path.exists():
            msg = f"Config file not found: {path}"
            raise ValueError(msg)

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:  # pragma: no cover - explicit error path
            msg = f"Failed to parse YAML config at {path}: {exc}"
            raise ValueError(msg) from exc

        if not isinstance(data, dict):
            msg = f"Config at {path} must be a mapping"
            raise TypeError(msg)

        try:
            typed_data = cast("dict[str, Any]", data)
            return cls(**typed_data)
        except ValidationError as exc:
            msg = f"Invalid grammar config at {path}: {exc}"
            raise ValueError(msg) from exc

    @classmethod
    def load_default(cls) -> "GrammarConfig":
        """Load the default packaged grammar config."""
        with as_file(DEFAULT_CONFIG_PATH) as config_path:
            return cls.from_yaml(config_path)


def load_grammar_config(config_path: Path | None) -> GrammarConfig:
    """Load grammar configuration from a path or fall back to default."""
    if config_path:
        return GrammarConfig.from_yaml(config_path)
    return GrammarConfig.load_default()

__all__ = ["DEFAULT_CONFIG_PATH", "GrammarConfig", "load_grammar_config"]
