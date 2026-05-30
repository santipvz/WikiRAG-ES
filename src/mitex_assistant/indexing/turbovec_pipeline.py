from __future__ import annotations

from pathlib import Path

from mitex_assistant.config.settings import Settings
from mitex_assistant.data.jsonl import read_jsonl
from mitex_assistant.data.models import Document
from mitex_assistant.indexing.chunking import chunk_documents
from mitex_assistant.indexing.embeddings import EmbeddingModel
from mitex_assistant.indexing.turbovec_store import build_turbovec_index


def build_turbovec_index_from_corpus(
    corpus_path: Path, index_dir: Path, settings: Settings, bit_width: int = 4
) -> int:
    documents = [Document(**row) for row in read_jsonl(corpus_path)]
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    embedding_model = EmbeddingModel(settings.embedding_model)
    embeddings = embedding_model.encode([chunk.text for chunk in chunks])
    build_turbovec_index(chunks, embeddings, index_dir, bit_width=bit_width)
    return len(chunks)
