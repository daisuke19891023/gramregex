from pathlib import Path


import pytest
from typer.testing import CliRunner

from gramregex import cli
from gramregex.llm.base import GrammarSyntax, ReasoningEffort, VerbosityLevel


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
        grammar_syntax: GrammarSyntax,
        verbosity: VerbosityLevel | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> str:
        """Record the call and return canned text."""
        self.generate_called_with = {
            "prompt": prompt,
            "grammar": grammar,
            "grammar_syntax": grammar_syntax,
            "verbosity": verbosity,
            "reasoning_effort": reasoning_effort,
        }
        return "grammar-output"


def test_cli_uses_factory_and_outputs_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLI がファクトリ経由でクライアントを呼び出すことを検証する."""
    runner = CliRunner()
    grammar_path = tmp_path / "grammar.cfg"
    grammar_path.write_text("root ::= 'x'", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    dummy_client = DummyClient(None)

    def fake_create_client(_: object) -> DummyClient:
        return dummy_client

    monkeypatch.setattr(cli, "create_llm_client", fake_create_client)

    result = runner.invoke(
        cli.app,
        [
            "--grammar-file",
            str(grammar_path),
            "--verbosity",
            "high",
            "--reasoning-effort",
            "medium",
            "--grammar-syntax",
            "regex",
            "input text",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "grammar-output" in result.stdout
    assert dummy_client.generate_called_with == {
        "prompt": "input text",
        "grammar": "root ::= 'x'",
        "grammar_syntax": "regex",
        "verbosity": "high",
        "reasoning_effort": "medium",
    }


def test_cli_requires_grammar(monkeypatch: pytest.MonkeyPatch) -> None:
    """CFG 未指定の場合にエラーとなることを検証する."""
    runner = CliRunner()
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    result = runner.invoke(cli.app, ["input text"])

    assert result.exit_code != 0
    assert "CFG 構文" in result.stdout


def test_cli_defaults_grammar_syntax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grammar syntax が未指定の場合 lark が使われる."""
    runner = CliRunner()
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    dummy_client = DummyClient(None)

    def fake_create_client(_: object) -> DummyClient:
        return dummy_client

    monkeypatch.setattr(cli, "create_llm_client", fake_create_client)

    result = runner.invoke(
        cli.app,
        [
            "--grammar",
            "root ::= 'a'",
            "--verbosity",
            "low",
            "input text",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert dummy_client.generate_called_with is not None
    assert dummy_client.generate_called_with["grammar_syntax"] == "lark"
