from __future__ import annotations

from wikirag_es.config.settings import Settings
from wikirag_es.llm.base import LLMClient
from wikirag_es.llm.ollama import OllamaClient


def create_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "ollama":
        return OllamaClient(settings.ollama_model, settings.ollama_url)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
