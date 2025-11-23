"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration values for LLM access and defaults."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    provider: str = Field(default="openai", description="LLM provider identifier")
    openai_api_key: str = Field(default="", description="API key for OpenAI-compatible endpoints")
    openai_base_url: str | None = Field(
        default=None, description="Optional base URL for OpenAI-compatible endpoints",
    )
    openai_model: str = Field(default="gpt-4.1-mini", description="Default OpenAI model name")

    @model_validator(mode="after")
    def validate_api_key(self) -> "Settings":
        """Ensure API key is provided."""
        if not self.openai_api_key:
            msg = "OPENAI_API_KEY is required"
            raise ValueError(msg)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
