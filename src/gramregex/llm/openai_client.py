"""OpenAI Responses API client implementation."""

from typing import Protocol, cast
from collections.abc import Sequence

from openai import OpenAI

from gramregex.llm.base import (
    GrammarSyntax,
    LLMClient,
    ReasoningEffort,
    VerbosityLevel,
)
from gramregex.settings import Settings


class ResponsesResource(Protocol):
    """Subset of the OpenAI responses resource used by the client."""

    def create(self, **kwargs: object) -> object:
        """Create a response using the provided model and grammar."""


class ResponsesClient(Protocol):
    """Client exposing the responses resource."""

    responses: ResponsesResource


class ResponseContent(Protocol):
    """Single text fragment returned by the model."""

    text: str


class ResponseChoice(Protocol):
    """Container holding content fragments."""

    content: Sequence[ResponseContent]


class OpenAIResponsesClient(LLMClient):
    """LLM client using the OpenAI Responses API with CFG grammar support."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the client with application settings."""
        self._settings = settings
        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._client = cast("ResponsesClient", client)

    def generate(
        self,
        prompt: str,
        *,
        grammar: str,
        grammar_syntax: GrammarSyntax,
        verbosity: VerbosityLevel | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> str:
        """Generate output using the configured model and grammar."""
        text_config: dict[str, object] = {"format": {"type": "text"}}
        if verbosity:
            text_config["verbosity"] = verbosity

        tools: list[dict[str, object]] = [
            {
                "type": "custom",
                "name": "cfg_grammar",
                "description": "Validate output against the provided grammar.",
                "format": {
                    "type": "grammar",
                    "syntax": grammar_syntax,
                    "definition": grammar,
                },
            },
        ]

        reasoning: dict[str, str] | None = None
        if reasoning_effort:
            reasoning = {"effort": reasoning_effort}

        response_kwargs: dict[str, object] = {
            "model": self._settings.openai_model,
            "input": prompt,
            "text": text_config,
            "tools": tools,
            "parallel_tool_calls": False,
        }
        if reasoning:
            response_kwargs["reasoning"] = reasoning

        response = self._client.responses.create(**response_kwargs)
        return self._extract_output_text(response)

    @staticmethod
    def _extract_output_text(response: object) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text

        choices: Sequence[ResponseChoice] | None = getattr(response, "output", None)
        if isinstance(choices, Sequence) and choices:
            first = choices[0]
            content: Sequence[ResponseContent] | None = getattr(first, "content", None)
            if isinstance(content, Sequence) and content:
                text: str | None = getattr(content[0], "text", None)
                if isinstance(text, str):
                    return text

        message = "The response did not contain text output"
        raise ValueError(message)
