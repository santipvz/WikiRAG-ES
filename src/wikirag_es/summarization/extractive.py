from __future__ import annotations

import re

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from wikirag_es.indexing.embeddings import EmbeddingModel


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if len(part.strip()) > 20]


def textrank_summary(
    text: str,
    embedding_model: EmbeddingModel,
    sentences: int = 4,
    threshold: float = 0.45,
    diversity: float = 0.35,
    query: str | None = None,
    query_weight: float = 0.5,
) -> list[str]:
    sentence_list = split_sentences(text)
    if len(sentence_list) <= sentences:
        return sentence_list
    embeddings = embedding_model.encode(sentence_list)
    similarity = cosine_similarity(embeddings)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(sentence_list)))
    for i in range(len(sentence_list)):
        for j in range(i + 1, len(sentence_list)):
            if similarity[i, j] >= threshold:
                graph.add_edge(i, j, weight=float(similarity[i, j]))
    scores = nx.pagerank(graph, alpha=0.85, weight="weight") if graph.number_of_edges() else {i: 1.0 for i in range(len(sentence_list))}
    if query:
        scores = _bias_scores_with_query(
            scores, sentence_list, embeddings, embedding_model, query, query_weight
        )
    selected = _mmr_select(scores, similarity, sentences, diversity)
    return [sentence_list[index] for index in sorted(selected)]


def _bias_scores_with_query(
    scores: dict[int, float],
    sentence_list: list[str],
    embeddings: np.ndarray,
    embedding_model: EmbeddingModel,
    query: str,
    query_weight: float,
) -> dict[int, float]:
    """Mezcla la centralidad TextRank con la relevancia a la consulta (QFS)."""
    query_embedding = embedding_model.encode([query])
    relevance = cosine_similarity(embeddings, query_embedding).reshape(-1)
    relevance = np.clip(relevance, 0.0, None)
    centrality = np.array([scores[i] for i in range(len(sentence_list))], dtype=float)
    centrality = _normalize(centrality)
    relevance = _normalize(relevance)
    blended = (1 - query_weight) * centrality + query_weight * relevance
    return {i: float(blended[i]) for i in range(len(sentence_list))}


def _normalize(values: np.ndarray) -> np.ndarray:
    span = float(values.max() - values.min())
    if span <= 0:
        return np.zeros_like(values)
    return (values - values.min()) / span


def _mmr_select(scores: dict[int, float], similarity: np.ndarray, k: int, diversity: float) -> list[int]:
    candidates = sorted(scores.keys())
    selected: list[int] = []
    while candidates and len(selected) < k:
        best = max(
            candidates,
            key=lambda idx: (_mmr_score(idx, selected, scores, similarity, diversity), -idx),
        )
        selected.append(best)
        candidates.remove(best)
    return selected


def _mmr_score(
    index: int,
    selected: list[int],
    scores: dict[int, float],
    similarity: np.ndarray,
    diversity: float,
) -> float:
    redundancy = max((similarity[index, other] for other in selected), default=0.0)
    return (1 - diversity) * scores[index] - diversity * redundancy
