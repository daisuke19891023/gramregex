from pathlib import Path

from collections.abc import Iterator

import pytest

from gramregex import api
from gramregex import settings as settings_module
from gramregex.api import generate
from gramregex.config import load_grammar_config
from gramregex.settings import Settings


class DummyClient:
    """Lightweight stand-in for the real LLM client."""

    def __init__(self, _: object) -> None:
        """Initialize stub storage."""
        self.generate_called_with: dict[str, object] | None = None

    def generate(
        self,
        prompt: str,
        *,
        grammar: str,
        grammar_syntax: str,
        verbosity: str | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """Record the call and return canned text."""
        self.generate_called_with = {
            "prompt": prompt,
            "grammar": grammar,
            "grammar_syntax": grammar_syntax,
            "verbosity": verbosity,
            "reasoning_effort": reasoning_effort,
        }
        return "library-output"


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """Ensure settings cache does not leak between tests."""
    settings_module.get_settings.cache_clear()
    yield
    settings_module.get_settings.cache_clear()


def test_generate_uses_factory_and_returns_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ライブラリ API がファクトリ経由で出力を返す."""
    grammar_path = tmp_path / "grammar.cfg"
    grammar_path.write_text("root ::= 'lib'", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    dummy_client = DummyClient(None)

    def fake_create_client(settings: Settings) -> DummyClient:
        fake_create_client.captured_settings = settings  # type: ignore[attr-defined]
        return dummy_client

    monkeypatch.setattr(api, "create_llm_client", fake_create_client)

    output = generate(
        "input text",
        grammar_file=grammar_path,
        grammar_syntax="regex",
        verbosity="high",
        reasoning_effort="medium",
    )

    assert output == "library-output"
    assert dummy_client.generate_called_with == {
        "prompt": "input text",
        "grammar": "root ::= 'lib'",
        "grammar_syntax": "regex",
        "verbosity": "high",
        "reasoning_effort": "medium",
    }

    captured = getattr(fake_create_client, "captured_settings", None)
    assert isinstance(captured, Settings)
    assert captured.openai_model == Settings().openai_model


def test_generate_loads_default_grammar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grammar 未指定の場合デフォルトを利用する."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    dummy_client = DummyClient(None)

    def fake_create_client(_: Settings) -> DummyClient:
        return dummy_client

    monkeypatch.setattr(api, "create_llm_client", fake_create_client)

    output = generate("input text")

    assert output == "library-output"
    assert dummy_client.generate_called_with is not None
    default_grammar = load_grammar_config(None).content
    assert dummy_client.generate_called_with["grammar"] == default_grammar


def test_generate_overrides_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model 引数で Settings を上書きできる."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    dummy_client = DummyClient(None)

    def fake_create_client(settings: Settings) -> DummyClient:
        fake_create_client.captured_model = settings.openai_model  # type: ignore[attr-defined]
        return dummy_client

    monkeypatch.setattr(api, "create_llm_client", fake_create_client)

    generate("input", model="custom-model", grammar="root ::= 'x'")

    captured_model = getattr(fake_create_client, "captured_model", None)
    assert captured_model == "custom-model"


def test_generate_rejects_conflicting_grammar_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Grammar と grammar_file の同時指定は例外になる."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    grammar_path = tmp_path / "grammar.cfg"

    with pytest.raises(ValueError, match="--grammar と --grammar-file は同時に指定できません"):
        generate("input", grammar="root ::= 'x'", grammar_file=grammar_path)
