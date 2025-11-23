import pytest
from gramregex.settings import Settings


def test_settings_loads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """環境変数から設定が読み込まれることを確認する."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "example-model")

    settings = Settings()

    assert settings.openai_api_key == "test-key"
    assert settings.provider == "openai"
    assert settings.openai_model == "example-model"
