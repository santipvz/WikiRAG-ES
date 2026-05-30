from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from mitex_assistant.data.jsonl import read_jsonl
from mitex_assistant.eval.ir import average_metrics, evaluate_retrieval_by_doc
from mitex_assistant.indexing.embeddings import EmbeddingModel
from mitex_assistant.indexing.turbovec_store import TURBOVEC_INDEX_FILE, TurboVecIndex
from mitex_assistant.indexing.vector_store import HybridIndex


def benchmark_turbovec_against_faiss(
    faiss_index_dir: Path,
    turbovec_index_dir: Path,
    queries_path: Path,
    embedding_model: EmbeddingModel,
    k: int = 10,
    limit: int | None = 100,
) -> dict:
    rows = list(read_jsonl(queries_path))
    if limit is not None:
        rows = rows[:limit]
    queries = [str(row["query"]) for row in rows]
    query_embeddings = embedding_model.encode(queries) if queries else np.empty((0, 0), dtype="float32")
    faiss_index = HybridIndex.load(faiss_index_dir)
    turbovec_index = TurboVecIndex.load(turbovec_index_dir)

    faiss_metrics = []
    turbovec_metrics = []
    faiss_seconds = 0.0
    turbovec_seconds = 0.0
    for row, embedding in zip(rows, query_embeddings):
        relevant_doc_ids = set(row.get("relevant_doc_ids", []))
        query = np.asarray([embedding], dtype="float32")

        start = time.perf_counter()
        faiss_results = faiss_index.semantic_search(query, top_k=k)
        faiss_seconds += time.perf_counter() - start

        start = time.perf_counter()
        turbovec_results = turbovec_index.semantic_search(query, top_k=k)
        turbovec_seconds += time.perf_counter() - start

        faiss_metrics.append(evaluate_retrieval_by_doc(faiss_results, relevant_doc_ids, k))
        turbovec_metrics.append(evaluate_retrieval_by_doc(turbovec_results, relevant_doc_ids, k))

    return {
        "k": k,
        "queries": len(rows),
        "faiss": {
            "average": average_metrics(faiss_metrics).__dict__,
            "search_seconds": faiss_seconds,
            "avg_search_ms": _avg_ms(faiss_seconds, len(rows)),
            "vector_bytes": _faiss_vector_bytes(faiss_index_dir),
        },
        "turbovec": {
            "average": average_metrics(turbovec_metrics).__dict__,
            "search_seconds": turbovec_seconds,
            "avg_search_ms": _avg_ms(turbovec_seconds, len(rows)),
            "vector_bytes": _file_size(turbovec_index_dir / TURBOVEC_INDEX_FILE),
        },
    }


def _avg_ms(seconds: float, count: int) -> float:
    return (seconds / count * 1000.0) if count else 0.0


def _faiss_vector_bytes(index_dir: Path) -> int:
    return _file_size(index_dir / "faiss.index") + _file_size(index_dir / "embeddings.npy")


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0
