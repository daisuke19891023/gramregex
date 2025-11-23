from types import SimpleNamespace


import pytest

from gramregex.llm.openai_client import OpenAIResponsesClient
from gramregex.settings import Settings


class DummyResponses:
    """Stub responses client for assertions."""

    def __init__(self) -> None:
        """Initialize captured call store."""
        self.create_called_with: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:  # pragma: no cover - passthrough
        """Capture arguments and return canned response."""
        self.create_called_with = kwargs
        return SimpleNamespace(output_text="result text")


def test_openai_client_calls_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAIResponsesClient が responses.create を正しく呼ぶ."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    settings = Settings()

    dummy_responses = DummyResponses()
    dummy_client = SimpleNamespace(responses=dummy_responses)

    def fake_openai_client(**_: object) -> SimpleNamespace:
        return dummy_client

    monkeypatch.setattr("gramregex.llm.openai_client.OpenAI", fake_openai_client)

    client = OpenAIResponsesClient(settings)
    output = client.generate(
        "hello",
        grammar="root ::= 'hello'",
        grammar_syntax="lark",
        verbosity="medium",
        reasoning_effort="minimal",
    )

    assert output == "result text"
    assert dummy_responses.create_called_with == {
        "model": settings.openai_model,
        "input": "hello",
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tools": [
            {
                "type": "custom",
                "name": "cfg_grammar",
                "description": "Validate output against the provided grammar.",
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": "root ::= 'hello'",
                },
            },
        ],
        "parallel_tool_calls": False,
        "reasoning": {"effort": "minimal"},
    }


def test_openai_client_extracts_from_output_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Output 配列からテキストを取り出せる."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    settings = Settings()

    class DummyOutput:
        """Container holding response text."""

        def __init__(self) -> None:
            """Prepare a content list with a text field."""
            self.content = [SimpleNamespace(text="list text")]

    class DummyResponsesWithList(DummyResponses):
        """Responses mock returning content list."""

        def create(self, **kwargs: object) -> object:  # pragma: no cover
            """Return structured output mimicking the API."""
            super().create(**kwargs)
            return SimpleNamespace(output=[DummyOutput()])

    dummy_responses = DummyResponsesWithList()
    dummy_client = SimpleNamespace(responses=dummy_responses)

    def fake_openai_client(**_: object) -> SimpleNamespace:
        return dummy_client

    monkeypatch.setattr("gramregex.llm.openai_client.OpenAI", fake_openai_client)

    client = OpenAIResponsesClient(settings)
    output = client.generate("hello", grammar="grammar", grammar_syntax="lark")

    assert output == "list text"
