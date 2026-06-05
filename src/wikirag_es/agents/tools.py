from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from wikirag_es.agents.web_search import WebSearchTool
from wikirag_es.indexing.embeddings import EmbeddingModel
from wikirag_es.retrieval.hybrid import HybridRetriever
from wikirag_es.summarization.extractive import textrank_summary

__all__ = ["RagTool", "SummaryTool", "TopicTool", "DateTool", "WebSearchTool"]

WEEKDAYS_ES = [
    "lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo",
]
MONTHS_ES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]


class RagTool:
    def __init__(self, retriever: HybridRetriever, top_k: int) -> None:
        self.retriever = retriever
        self.top_k = top_k

    def run(self, query: str) -> tuple[str, list[str]]:
        results = self.retriever.retrieve(query, top_k=self.top_k)
        context = self.retriever.build_context(results)
        sources = [f"[Fuente {result.rank}] {result.chunk.source_path}" for result in results]
        return context, sources


class SummaryTool:
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model

    def run_text(self, text: str, sentences: int = 4, query: str | None = None) -> str:
        text = _extract_plain_context_text(text)
        return " ".join(
            textrank_summary(text, self.embedding_model, sentences=sentences, query=query)
        )

    def run_file(self, path: Path, sentences: int = 4, query: str | None = None) -> str:
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.run_text(text, sentences=sentences, query=query)


class TopicTool:
    def __init__(self, report_path: Path) -> None:
        self.report_path = report_path

    def run(self, max_topics: int = 20) -> str:
        payload = json.loads(self.report_path.read_text(encoding="utf-8"))
        method = payload.get("method", "unknown")
        num_documents = payload.get("num_documents", "?")
        topic_info = payload.get("topic_info", [])
        outliers = [
            topic for topic in topic_info if topic.get("Topic", topic.get("topic")) == -1
        ]
        real_topics = [
            topic for topic in topic_info if topic.get("Topic", topic.get("topic")) != -1
        ]
        real_topics = sorted(
            real_topics,
            key=lambda topic: (-_topic_count(topic), str(topic.get("Topic", topic.get("topic", "")))),
        )
        shown_topics = real_topics[:max_topics]
        outlier_count = (
            outliers[0].get("Count", outliers[0].get("count", 0)) if outliers else 0
        )
        lines = [
            f"Modelado offline de topicos. Metodo: {method}. Documentos: {num_documents}.",
            f"Clusters detectados: {len(real_topics)} "
            f"(+ {outlier_count} documentos outlier/sin cluster).",
            f"Mostrando los {len(shown_topics)} clusters mas grandes:",
        ]
        for topic in shown_topics:
            topic_id = topic.get("Topic", topic.get("topic", "?"))
            count = topic.get("Count", topic.get("count", "?"))
            representation = topic.get("Representation") or topic.get("Name") or topic.get("Name", "")
            if isinstance(representation, list):
                representation = ", ".join(str(item) for item in representation[:8])
            lines.append(f"- Topico {topic_id} ({count} docs): {representation}")
        return "\n".join(lines)


def _topic_count(topic: dict) -> int:
    try:
        return int(topic.get("Count", topic.get("count", 0)))
    except (TypeError, ValueError):
        return 0


_CONTEXT_TEXT_RE = re.compile(
    r"(?:^|\n)Texto:\s*(.*?)(?=\n\n\[Fuente \d+\]|\Z)", re.DOTALL
)


def _extract_plain_context_text(text: str) -> str:
    """Extrae solo el contenido textual de bloques RAG con metadatos."""
    matches = [match.strip() for match in _CONTEXT_TEXT_RE.findall(text) if match.strip()]
    if matches:
        return "\n".join(matches)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[Fuente ") or stripped.startswith("Ruta:"):
            continue
        if stripped.startswith("Texto:"):
            stripped = stripped.removeprefix("Texto:").strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


class DateTool:
    def run(self) -> str:
        today = date.today()
        weekday = WEEKDAYS_ES[today.weekday()]
        month = MONTHS_ES[today.month - 1]
        return f"Hoy es {weekday}, {today.day} de {month} de {today.year}."
