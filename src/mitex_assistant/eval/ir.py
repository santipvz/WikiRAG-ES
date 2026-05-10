from __future__ import annotations

from dataclasses import dataclass

from mitex_assistant.data.models import RetrievalResult


@dataclass(frozen=True)
class IRMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float


def evaluate_retrieval(results: list[RetrievalResult], relevant_chunk_ids: set[str], k: int) -> IRMetrics:
    top_results = results[:k]
    if not top_results:
        return IRMetrics(0.0, 0.0, 0.0)
    retrieved_ids = [result.chunk.chunk_id for result in top_results]
    hits = [chunk_id in relevant_chunk_ids for chunk_id in retrieved_ids]
    precision = sum(hits) / len(top_results)
    recall = sum(hits) / len(relevant_chunk_ids) if relevant_chunk_ids else 0.0
    reciprocal_rank = 0.0
    for index, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1.0 / index
            break
    return IRMetrics(precision, recall, reciprocal_rank)
