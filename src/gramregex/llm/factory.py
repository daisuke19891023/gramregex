"""Factory for constructing LLM clients."""

from gramregex.llm.base import LLMClient
from gramregex.llm.openai_client import OpenAIResponsesClient
from gramregex.settings import Settings


def create_llm_client(settings: Settings) -> LLMClient:
    """Return an LLM client based on provider settings."""
    provider = settings.provider.lower()
    if provider == "openai":
        return OpenAIResponsesClient(settings)

    message = f"Unsupported LLM provider: {settings.provider}"
    raise ValueError(message)
