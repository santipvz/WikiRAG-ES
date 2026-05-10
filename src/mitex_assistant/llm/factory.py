from __future__ import annotations

from mitex_assistant.config.settings import Settings
from mitex_assistant.llm.base import LLMClient
from mitex_assistant.llm.ollama import OllamaClient


def create_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "ollama":
        return OllamaClient(settings.ollama_model, settings.ollama_url)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
