from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from wikirag_es.data.models import RetrievalResult
from wikirag_es.indexing.embeddings import EmbeddingModel
from wikirag_es.utils.text import strip_prompt_injection

if TYPE_CHECKING:
    from wikirag_es.retrieval.reranking import CrossEncoderReranker


def reciprocal_rank_fusion(
    result_sets: list[list[RetrievalResult]], top_k: int, k: int = 60
) -> list[RetrievalResult]:
    by_chunk: dict[str, RetrievalResult] = {}
    scores: dict[str, float] = {}
    for results in result_sets:
        for result in results:
            by_chunk[result.chunk.chunk_id] = result
            scores[result.chunk.chunk_id] = scores.get(result.chunk.chunk_id, 0.0) + 1.0 / (k + result.rank)
    ranked_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    fused: list[RetrievalResult] = []
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        base = by_chunk[chunk_id]
        fused.append(
            RetrievalResult(
                chunk=base.chunk,
                score=float(scores[chunk_id]),
                source="rrf",
                rank=rank,
            )
        )
    return fused


class HybridRetriever:
    def __init__(
        self,
        index: Any,
        embedding_model: EmbeddingModel,
        reranker: "CrossEncoderReranker | None" = None,
        rerank_top_k: int | None = None,
    ) -> None:
        self.index = index
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        # Con reranker recuperamos mas candidatos para darle margen de reordenado.
        candidate_k = max(top_k, (self.rerank_top_k or top_k) * 2) if self.reranker else top_k
        query_embedding = self.embedding_model.encode([query])
        semantic = self.index.semantic_search(np.asarray(query_embedding), top_k=candidate_k)
        if hasattr(self.index, "lexical_search"):
            lexical = self.index.lexical_search(query, top_k=candidate_k)
            fused = reciprocal_rank_fusion([semantic, lexical], top_k=candidate_k)
        else:
            fused = semantic
        if self.reranker is None:
            return fused[:top_k]
        final_k = self.rerank_top_k or top_k
        return self.reranker.rerank(query, fused, top_k=final_k)

    def build_context(self, results: list[RetrievalResult], max_chars: int = 6000) -> str:
        blocks: list[str] = []
        total = 0
        for result in results:
            text = strip_prompt_injection(result.chunk.text)
            block = (
                f"[Fuente {result.rank}] {result.chunk.metadata.get('title') or result.chunk.source_path}\n"
                f"Ruta: {result.chunk.source_path}\n"
                f"Texto: {text}"
            )
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block)
        return "\n\n".join(blocks)
