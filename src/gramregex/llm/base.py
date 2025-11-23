"""LLM client abstractions."""

from abc import ABC, abstractmethod
from typing import Literal

GrammarSyntax = Literal["lark", "regex"]
VerbosityLevel = Literal["low", "medium", "high"]
ReasoningEffort = Literal["minimal", "medium", "high"]


class LLMClient(ABC):
    """Protocol for LLM clients supporting grammar-constrained generation."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        grammar: str,
        grammar_syntax: GrammarSyntax,
        verbosity: VerbosityLevel | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> str:
        """Generate text given a prompt and a CFG grammar."""

