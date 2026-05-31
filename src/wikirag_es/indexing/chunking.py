from __future__ import annotations

from wikirag_es.data.models import Chunk, Document
from wikirag_es.utils.ids import stable_id


def chunk_document(document: Document, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be lower than chunk_size")
    text = document.text
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = stable_id(document.doc_id, str(start), str(end), chunk_text[:128])
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    text=chunk_text,
                    source_path=document.source_path,
                    start_char=start,
                    end_char=end,
                    metadata={"title": document.title, **document.metadata},
                )
            )
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_documents(documents: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks
