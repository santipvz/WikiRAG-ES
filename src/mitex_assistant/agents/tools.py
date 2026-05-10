from __future__ import annotations

from pathlib import Path

from mitex_assistant.indexing.embeddings import EmbeddingModel
from mitex_assistant.retrieval.hybrid import HybridRetriever
from mitex_assistant.summarization.extractive import textrank_summary


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

    def run_text(self, text: str, sentences: int = 4) -> str:
        return " ".join(textrank_summary(text, self.embedding_model, sentences=sentences))

    def run_file(self, path: Path, sentences: int = 4) -> str:
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.run_text(text, sentences=sentences)
