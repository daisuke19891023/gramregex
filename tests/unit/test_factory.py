import pytest

from gramregex.llm.base import GrammarSyntax, ReasoningEffort, VerbosityLevel
from gramregex.llm.factory import create_llm_client
from gramregex.settings import Settings


class DummyClient:
    """Simple stub client to inspect factory output."""

    def __init__(self, settings: Settings) -> None:
        """Store passed settings for assertions."""
        self.settings = settings

    def generate(
        self,
        prompt: str,
        *,
        grammar: str,
        grammar_syntax: GrammarSyntax,
        verbosity: VerbosityLevel | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> str:  # pragma: no cover - not used here
        """Return canned content for compatibility."""
        return f"{prompt}:{grammar}:{grammar_syntax}:{verbosity}:{reasoning_effort}"


def test_create_llm_client_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory は openai プロバイダで OpenAIResponsesClient を返す."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    created_settings = Settings()

    def fake_client(settings: Settings) -> DummyClient:
        return DummyClient(settings)

    monkeypatch.setattr("gramregex.llm.factory.OpenAIResponsesClient", fake_client)

    client = create_llm_client(created_settings)

    assert isinstance(client, DummyClient)
    assert client.settings is created_settings


def test_create_llm_client_unsupported_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """未知のプロバイダでは ValueError が投げられる."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    settings = Settings(provider="unknown", openai_api_key="dummy", openai_model="model")

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client(settings)
