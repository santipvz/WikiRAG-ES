from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on", "si", "sí"}


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path.cwd()
    embedding_model: str = os.getenv(
        "MITEX_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    llm_provider: str = os.getenv("MITEX_LLM_PROVIDER", "ollama")
    ollama_model: str = os.getenv("MITEX_OLLAMA_MODEL", "gemma4:e4b")
    ollama_url: str = os.getenv("MITEX_OLLAMA_URL", "http://localhost:11434")
    top_k: int = int(os.getenv("MITEX_TOP_K", "8"))
    rerank_top_k: int = int(os.getenv("MITEX_RERANK_TOP_K", "5"))
    chunk_size: int = int(os.getenv("MITEX_CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("MITEX_CHUNK_OVERLAP", "150"))
    # Reranking (cross-encoder) sobre los candidatos fusionados por RRF.
    use_reranker: bool = _as_bool(os.getenv("MITEX_USE_RERANKER", "false"))
    reranker_model: str = os.getenv(
        "MITEX_RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    # Enrutado del agente: usar el LLM para desambiguar la herramienta.
    use_llm_router: bool = _as_bool(os.getenv("MITEX_USE_LLM_ROUTER", "true"))
    # Tipo de indice FAISS: flat (exacto) | hnsw (ANN aproximado).
    index_type: str = os.getenv("MITEX_INDEX_TYPE", "flat")
    hnsw_m: int = int(os.getenv("MITEX_HNSW_M", "32"))
    # Busqueda web complementaria.
    web_search_results: int = int(os.getenv("MITEX_WEB_SEARCH_RESULTS", "4"))
    web_search_timeout: int = int(os.getenv("MITEX_WEB_SEARCH_TIMEOUT", "10"))


def get_settings() -> Settings:
    return Settings()
