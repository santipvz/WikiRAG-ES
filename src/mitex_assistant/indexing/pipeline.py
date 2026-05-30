from __future__ import annotations

from pathlib import Path

from mitex_assistant.config.settings import Settings
from mitex_assistant.data.jsonl import read_jsonl, write_jsonl
from mitex_assistant.data.models import Document
from mitex_assistant.indexing.chunking import chunk_documents
from mitex_assistant.indexing.embeddings import EmbeddingModel
from mitex_assistant.indexing.vector_store import build_hybrid_index


def build_index(
    corpus_path: Path, index_dir: Path, settings: Settings, index_type: str | None = None
) -> int:
    documents = [Document(**row) for row in read_jsonl(corpus_path)]
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    write_jsonl(index_dir / "chunks.raw.jsonl", (chunk.to_dict() for chunk in chunks))
    embedding_model = EmbeddingModel(settings.embedding_model)
    embeddings = embedding_model.encode([chunk.text for chunk in chunks])
    build_hybrid_index(
        chunks,
        embeddings,
        index_dir,
        index_type=index_type or settings.index_type,
        hnsw_m=settings.hnsw_m,
    )
    return len(chunks)
