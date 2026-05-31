"""Benchmark de latencia y recall del índice ANN HNSW frente a FAISS Flat."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=Path("data/reports/miracl_es_eval.jsonl"))
    parser.add_argument("--flat-index", type=Path, default=Path("data/indexes/default/faiss.index"))
    parser.add_argument(
        "--hnsw-index", type=Path, default=Path("data/indexes/default_hnsw/faiss.index")
    )
    parser.add_argument("--output", type=Path, default=Path("data/reports/ann_hnsw_vs_flat.json"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def read_queries(path: Path, limit: int) -> list[str]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line_number > limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return [str(row["query"]) for row in rows]


def bench(index: faiss.Index, query_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    index.search(query_embeddings[: min(5, len(query_embeddings))], k)
    start = time.perf_counter()
    _, indexes = index.search(query_embeddings, k)
    seconds = time.perf_counter() - start
    return indexes, seconds


def main() -> None:
    args = parse_args()
    missing = [path for path in (args.queries, args.flat_index, args.hnsw_index) if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(
            "Faltan artefactos para el benchmark HNSW:\n"
            f"{missing_text}\n"
            "Construye el índice HNSW con: "
            "wikirag build-index --corpus data/processed/miracl_es_corpus_1M.jsonl "
            "--index-dir data/indexes/default_hnsw --index-type hnsw"
        )

    queries = read_queries(args.queries, args.limit)
    model = SentenceTransformer(args.model)
    query_embeddings = model.encode(
        queries, normalize_embeddings=True, show_progress_bar=False
    ).astype("float32")

    flat = faiss.read_index(str(args.flat_index))
    hnsw = faiss.read_index(str(args.hnsw_index))
    flat_indexes, flat_seconds = bench(flat, query_embeddings, args.k)
    hnsw_indexes, hnsw_seconds = bench(hnsw, query_embeddings, args.k)

    overlap = np.mean(
        [
            len(set(flat_indexes[index]) & set(hnsw_indexes[index])) / args.k
            for index in range(len(queries))
        ]
    )
    top1_match = np.mean(
        [flat_indexes[index][0] == hnsw_indexes[index][0] for index in range(len(queries))]
    )

    result = {
        "queries": len(queries),
        "k": args.k,
        "flat": {
            "total_s": round(flat_seconds, 4),
            "avg_ms": round(flat_seconds / len(queries) * 1000, 3),
        },
        "hnsw": {
            "total_s": round(hnsw_seconds, 4),
            "avg_ms": round(hnsw_seconds / len(queries) * 1000, 3),
        },
        "speedup": round(flat_seconds / hnsw_seconds, 2),
        "ann_recall@10_vs_exact": round(float(overlap), 4),
        "ann_top1_match": round(float(top1_match), 4),
    }
    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
