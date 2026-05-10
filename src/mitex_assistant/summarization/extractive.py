from __future__ import annotations

import re

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from mitex_assistant.indexing.embeddings import EmbeddingModel


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if len(part.strip()) > 20]


def textrank_summary(
    text: str,
    embedding_model: EmbeddingModel,
    sentences: int = 4,
    threshold: float = 0.45,
    diversity: float = 0.35,
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
    selected = _mmr_select(scores, similarity, sentences, diversity)
    return [sentence_list[index] for index in sorted(selected)]


def _mmr_select(scores: dict[int, float], similarity: np.ndarray, k: int, diversity: float) -> list[int]:
    candidates = set(scores.keys())
    selected: list[int] = []
    while candidates and len(selected) < k:
        best = max(candidates, key=lambda idx: _mmr_score(idx, selected, scores, similarity, diversity))
        selected.append(best)
        candidates.remove(best)
    return selected


def _mmr_score(index: int, selected: list[int], scores: dict[int, float], similarity: np.ndarray, diversity: float) -> float:
    redundancy = max((similarity[index, other] for other in selected), default=0.0)
    return (1 - diversity) * scores[index] - diversity * redundancy
