from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path.cwd()
    embedding_model: str = os.getenv(
        "MITEX_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    llm_provider: str = os.getenv("MITEX_LLM_PROVIDER", "ollama")
    ollama_model: str = os.getenv("MITEX_OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
    ollama_url: str = os.getenv("MITEX_OLLAMA_URL", "http://localhost:11434")
    top_k: int = int(os.getenv("MITEX_TOP_K", "8"))
    rerank_top_k: int = int(os.getenv("MITEX_RERANK_TOP_K", "5"))
    chunk_size: int = int(os.getenv("MITEX_CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("MITEX_CHUNK_OVERLAP", "150"))


def get_settings() -> Settings:
    return Settings()
