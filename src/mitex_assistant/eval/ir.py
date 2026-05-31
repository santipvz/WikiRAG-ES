from __future__ import annotations

from dataclasses import dataclass

from mitex_assistant.data.models import RetrievalResult


@dataclass(frozen=True)
class IRMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float


def evaluate_retrieval(results: list[RetrievalResult], relevant_chunk_ids: set[str], k: int) -> IRMetrics:
    return evaluate_ids([result.chunk.chunk_id for result in results], relevant_chunk_ids, k)


def evaluate_retrieval_by_doc(results: list[RetrievalResult], relevant_doc_ids: set[str], k: int) -> IRMetrics:
    return evaluate_ids([result.chunk.doc_id for result in results], relevant_doc_ids, k)


def evaluate_ids(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> IRMetrics:
    if k <= 0:
        raise ValueError("k must be positive")
    top_ids = _unique_ids(retrieved_ids)[:k]
    if not top_ids or not relevant_ids:
        return IRMetrics(0.0, 0.0, 0.0)
    hits = [retrieved_id in relevant_ids for retrieved_id in top_ids]
    precision = sum(hits) / k
    recall = sum(hits) / len(relevant_ids)
    reciprocal_rank = 0.0
    for index, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1.0 / index
            break
    return IRMetrics(precision, recall, reciprocal_rank)


def _unique_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in ids:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def average_metrics(metrics: list[IRMetrics]) -> IRMetrics:
    if not metrics:
        return IRMetrics(0.0, 0.0, 0.0)
    return IRMetrics(
        precision_at_k=sum(metric.precision_at_k for metric in metrics) / len(metrics),
        recall_at_k=sum(metric.recall_at_k for metric in metrics) / len(metrics),
        mrr=sum(metric.mrr for metric in metrics) / len(metrics),
    )
