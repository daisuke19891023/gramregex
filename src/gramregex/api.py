"""Public Python API for grammar-constrained generation."""

from pathlib import Path

from gramregex.config import load_grammar_config
from gramregex.grammar import load_grammar
from gramregex.llm.base import GrammarSyntax, ReasoningEffort, VerbosityLevel
from gramregex.llm.factory import create_llm_client
from gramregex.settings import Settings, get_settings


def generate(
    prompt: str,
    *,
    grammar: str | None = None,
    grammar_file: Path | None = None,
    grammar_syntax: GrammarSyntax = "lark",
    verbosity: VerbosityLevel | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    model: str | None = None,
    settings: Settings | None = None,
) -> str:
    """Generate grammar-constrained text directly from Python.

    The arguments mirror the CLI options so library users can reuse the same
    feature set programmatically. Grammar can be provided directly or via a
    file; otherwise the configured default grammar is used.
    """
    active_settings = settings or get_settings()
    cfg = load_grammar(grammar, grammar_file, config_path=active_settings.grammar_config_path)
    if model:
        active_settings = active_settings.model_copy(update={"openai_model": model})

    client = create_llm_client(active_settings)
    return client.generate(
        prompt,
        grammar=cfg,
        grammar_syntax=grammar_syntax,
        verbosity=verbosity,
        reasoning_effort=reasoning_effort,
    )


__all__ = [
    "GrammarSyntax",
    "ReasoningEffort",
    "Settings",
    "VerbosityLevel",
    "generate",
    "get_settings",
    "load_grammar_config",
]
