from __future__ import annotations

import json
from pathlib import Path

from mitex_assistant.data.jsonl import read_jsonl
from mitex_assistant.eval.ir import IRMetrics, average_metrics, evaluate_retrieval_by_doc
from mitex_assistant.indexing.embeddings import EmbeddingModel
from mitex_assistant.indexing.vector_store import HybridIndex
from mitex_assistant.retrieval.hybrid import HybridRetriever


def evaluate_ir_file(
    index_dir: Path,
    queries_path: Path,
    embedding_model: EmbeddingModel | None,
    mode: str = "hybrid",
    k: int = 10,
    limit: int | None = None,
    output_path: Path | None = None,
    reranker=None,
    rerank_top_k: int | None = None,
) -> dict:
    if mode not in {"hybrid", "semantic", "lexical"}:
        raise ValueError("mode must be one of: hybrid, semantic, lexical")
    if mode in {"hybrid", "semantic"} and embedding_model is None:
        raise ValueError("embedding_model is required for hybrid and semantic modes")

    index = HybridIndex.load(
        index_dir,
        load_chunk_text=reranker is not None,
        load_embeddings=False,
        load_faiss=mode in {"hybrid", "semantic"},
        load_bm25=mode in {"hybrid", "lexical"},
    )
    retriever = (
        HybridRetriever(index, embedding_model, reranker=reranker, rerank_top_k=rerank_top_k)
        if mode == "hybrid"
        else None
    )
    metrics: list[IRMetrics] = []
    per_query: list[dict] = []
    for row_number, row in enumerate(read_jsonl(queries_path), start=1):
        if limit is not None and row_number > limit:
            break
        query = str(row["query"])
        relevant_doc_ids = set(row.get("relevant_doc_ids", []))
        if mode == "hybrid":
            assert retriever is not None
            results = retriever.retrieve(query, top_k=k)
        elif mode == "semantic":
            assert embedding_model is not None
            query_embedding = embedding_model.encode([query])
            results = index.semantic_search(query_embedding, top_k=k)
        else:
            results = index.lexical_search(query, top_k=k)
        metric = evaluate_retrieval_by_doc(results, relevant_doc_ids, k)
        metrics.append(metric)
        per_query.append(
            {
                "query": query,
                "relevant_doc_ids": sorted(relevant_doc_ids),
                "retrieved_doc_ids": [result.chunk.doc_id for result in results[:k]],
                "metrics": metric.__dict__,
            }
        )

    summary = average_metrics(metrics)
    payload = {
        "mode": mode,
        "k": k,
        "queries": len(metrics),
        "average": summary.__dict__,
        "per_query": per_query,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
