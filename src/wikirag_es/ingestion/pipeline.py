from __future__ import annotations

from pathlib import Path

from wikirag_es.data.jsonl import write_jsonl
from wikirag_es.data.models import Document
from wikirag_es.ingestion.loaders import load_path
from wikirag_es.utils.text import normalize_text, strip_prompt_injection


def ingest(input_path: Path, output_path: Path, min_chars: int = 200) -> list[Document]:
    raw_documents = load_path(input_path)
    documents: list[Document] = []
    seen_texts: set[str] = set()
    for document in raw_documents:
        text = strip_prompt_injection(normalize_text(document.text))
        if len(text) < min_chars or text in seen_texts:
            continue
        seen_texts.add(text)
        documents.append(
            Document(
                doc_id=document.doc_id,
                text=text,
                source_path=document.source_path,
                title=document.title,
                metadata=document.metadata,
            )
        )
    write_jsonl(output_path, (document.to_dict() for document in documents))
    return documents
