"""Command line interface for gramregex."""

from pathlib import Path
from typing import Annotated, Literal

import typer

from gramregex.llm.factory import create_llm_client
from gramregex.grammar import load_grammar
from gramregex.settings import get_settings

app = typer.Typer(add_completion=False, help="Generate grammar-constrained responses using OpenAI Responses API.")


@app.command(name="generate")
def generate(
    input_text: Annotated[str, typer.Argument(..., help="LLMへ送る入力テキスト")],
    grammar: Annotated[str | None, typer.Option("--grammar", "-g", help="CFG 文字列")] = None,
    grammar_file: Annotated[
        Path | None,
        typer.Option(
            "--grammar-file",
            "-f",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="CFGファイルのパス",
        ),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", help="上書きするモデル名")] = None,
    grammar_syntax: Annotated[
        Literal["lark", "regex"],
        typer.Option(
            "--grammar-syntax",
            help="grammar ツールの syntax (lark もしくは regex)",
            show_default=True,
        ),
    ] = "lark",
    verbosity: Annotated[
        Literal["low", "medium", "high"] | None,
        typer.Option("--verbosity", help="応答の詳細度 (low/medium/high)"),
    ] = None,
    reasoning_effort: Annotated[
        Literal["minimal", "medium", "high"] | None,
        typer.Option("--reasoning-effort", help="推論ステップの強度 (minimal/medium/high)"),
    ] = None,
) -> None:
    """Generate output constrained by the given CFG grammar."""
    settings = get_settings()
    try:
        cfg = load_grammar(grammar, grammar_file, config_path=settings.grammar_config_path)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if model:
        settings = settings.model_copy(update={"openai_model": model})

    client = create_llm_client(settings)
    output = client.generate(
        input_text,
        grammar=cfg,
        grammar_syntax=grammar_syntax,
        verbosity=verbosity,
        reasoning_effort=reasoning_effort,
    )
    typer.echo(output)


def main() -> None:
    """Entrypoint for console script."""
    app()


if __name__ == "__main__":
    main()
