"""Grammar utilities shared between CLI and library API."""

from pathlib import Path

from gramregex.config import load_grammar_config


def load_grammar(grammar: str | None, grammar_file: Path | None, *, config_path: Path | None) -> str:
    """Load grammar content from direct input, file, or configured defaults."""
    if grammar and grammar_file:
        msg = "--grammar と --grammar-file は同時に指定できません"
        raise ValueError(msg)

    if grammar_file:
        return grammar_file.read_text(encoding="utf-8")

    if grammar:
        return grammar

    config = load_grammar_config(config_path)
    return config.content


__all__ = ["load_grammar"]
