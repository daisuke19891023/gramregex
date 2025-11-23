from pathlib import Path
import re

import pytest

from gramregex.config import GrammarConfig, load_grammar_config


def test_loads_config_from_yaml(tmp_path: Path) -> None:
    """YAML から GrammarConfig を生成する."""
    config_path = tmp_path / "grammar.yaml"
    config_path.write_text(
        """
name: sample
description: sample config
a: 1
content: |
  root ::= "from-file"
""".strip(),
        encoding="utf-8",
    )

    config = GrammarConfig.from_yaml(config_path)

    assert config.name == "sample"
    assert config.description == "sample config"
    assert config.content == 'root ::= "from-file"'


def test_loads_default_config() -> None:
    """デフォルトの config が読み込まれる."""
    config = load_grammar_config(None)

    assert config.content
    assert config.name
    assert config.description


@pytest.mark.parametrize(
    ("body", "expected_exception", "expected_message"),
    [
        ("invalid: [", ValueError, "Failed to parse YAML config"),
        ("- list\n- entries", TypeError, "Config at"),
        ("{}", ValueError, "Invalid grammar config"),
    ],
)
def test_invalid_config_raises(
    tmp_path: Path, body: str, expected_exception: type[Exception], expected_message: str,
) -> None:
    """不正な YAML では例外を送出する."""
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(body, encoding="utf-8")

    with pytest.raises(expected_exception, match=re.escape(expected_message)) as excinfo:
        GrammarConfig.from_yaml(config_path)

    assert expected_message in str(excinfo.value)
