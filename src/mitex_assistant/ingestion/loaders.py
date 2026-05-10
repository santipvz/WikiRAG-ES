from __future__ import annotations

from pathlib import Path

import fitz
from bs4 import BeautifulSoup

from mitex_assistant.data.models import Document
from mitex_assistant.utils.ids import stable_id
from mitex_assistant.utils.text import normalize_text


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}


def load_path(path: Path) -> list[Document]:
    if path.is_dir():
        documents: list[Document] = []
        for file_path in sorted(path.rglob("*")):
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                documents.extend(load_file(file_path))
        return documents
    return load_file(path)


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return [_load_pdf(path)]
    if suffix in {".txt", ".md"}:
        return [_load_text(path)]
    if suffix in {".html", ".htm"}:
        return [_load_html(path)]
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def _load_pdf(path: Path) -> Document:
    doc = fitz.open(path)
    pages: list[str] = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append(f"[Pagina {page_number}]\n{text}")
    joined = normalize_text("\n\n".join(pages))
    return Document(
        doc_id=stable_id(str(path), joined[:512]),
        text=joined,
        source_path=str(path),
        title=path.stem,
        metadata={"file_type": "pdf", "pages": len(doc)},
    )


def _load_text(path: Path) -> Document:
    text = normalize_text(path.read_text(encoding="utf-8", errors="replace"))
    return Document(
        doc_id=stable_id(str(path), text[:512]),
        text=text,
        source_path=str(path),
        title=path.stem,
        metadata={"file_type": path.suffix.lower().lstrip(".")},
    )


def _load_html(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for node in soup(["script", "style", "nav", "footer", "header"]):
        node.decompose()
    text = normalize_text(soup.get_text(" "))
    return Document(
        doc_id=stable_id(str(path), text[:512]),
        text=text,
        source_path=str(path),
        title=soup.title.string.strip() if soup.title and soup.title.string else path.stem,
        metadata={"file_type": "html"},
    )
