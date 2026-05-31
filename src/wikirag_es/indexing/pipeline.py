from __future__ import annotations

from pathlib import Path

from wikirag_es.config.settings import Settings
from wikirag_es.data.jsonl import read_jsonl, write_jsonl
from wikirag_es.data.models import Document
from wikirag_es.indexing.chunking import chunk_documents
from wikirag_es.indexing.embeddings import EmbeddingModel
from wikirag_es.indexing.vector_store import build_hybrid_index


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
