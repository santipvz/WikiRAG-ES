from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import faiss
import numpy as np

from wikirag_es.data.jsonl import read_jsonl
from wikirag_es.data.models import Chunk
from wikirag_es.indexing.embeddings import EmbeddingModel
from wikirag_es.indexing.turbovec_store import TurboVecIndex
from wikirag_es.indexing.vector_store import build_faiss_index


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara latencia por consulta de Flat, HNSW y TurboVec."
    )
    parser.add_argument("--index-dir", type=Path, default=Path("data/indexes/default"))
    parser.add_argument("--queries", type=Path, default=Path("data/reports/miracl_es_eval.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/bench_scaling_techniques.json")
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--engines",
        default="flat,hnsw,turbovec",
        help="Motores a medir separados por coma: flat,hnsw,turbovec.",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    return parser.parse_args()


def bench_faiss(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    k: int,
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    return _bench_query_by_query(
        lambda row: index.search(row.reshape(1, -1), k),
        query_embeddings,
        repeats,
        warmup,
    )


def bench_turbovec(
    index: TurboVecIndex,
    query_embeddings: np.ndarray,
    k: int,
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    return _bench_query_by_query(
        lambda row: index.semantic_search(row, k),
        query_embeddings,
        repeats,
        warmup,
    )


def _bench_query_by_query(
    search_one: Callable[[np.ndarray], object],
    query_embeddings: np.ndarray,
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if len(query_embeddings) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    for _ in range(warmup):
        for row in query_embeddings:
            search_one(row)

    per_query_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for row in query_embeddings:
            search_one(row)
        seconds = time.perf_counter() - start
        per_query_ms.append(seconds / len(query_embeddings) * 1000.0)

    values = np.asarray(per_query_ms, dtype="float64")
    return {
        "mean": round(float(values.mean()), 3),
        "median": round(float(np.median(values)), 3),
        "std": round(float(values.std(ddof=0)), 3),
        "min": round(float(values.min()), 3),
        "max": round(float(values.max()), 3),
    }


def run() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("k must be positive")
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    engines = {engine.strip() for engine in args.engines.split(",") if engine.strip()}
    unknown = engines - {"flat", "hnsw", "turbovec"}
    if unknown:
        raise ValueError(f"unknown engines: {', '.join(sorted(unknown))}")

    faiss.omp_set_num_threads(args.threads)
    print(f"FAISS threads: {args.threads}", flush=True)
    print(f"Engines: {', '.join(sorted(engines))}", flush=True)

    print("Cargando embeddings...", flush=True)
    embeddings = np.load(args.index_dir / "embeddings.npy")
    chunks: list[Chunk] = []
    if "turbovec" in engines:
        print("Cargando chunks para TurboVec...", flush=True)
        with (args.index_dir / "chunks.jsonl").open("r", encoding="utf-8") as file:
            for line in file:
                chunks.append(Chunk(**json.loads(line)))

    queries = [str(row["query"]) for row in read_jsonl(args.queries)][: args.limit]

    print("Codificando queries...", flush=True)
    model = EmbeddingModel(args.model)
    query_embeddings = model.encode(queries)

    sizes = args.sizes or [100_000, 500_000, len(embeddings)]
    results = []

    for size in sizes:
        print(f"\n--- Evaluando tamano: {size} vectores ---", flush=True)
        sliced_embeddings = embeddings[:size]
        latency_stats = {}

        if "flat" in engines:
            print("Construyendo FAISS Flat...", flush=True)
            flat_index = build_faiss_index(sliced_embeddings, "flat")
            latency_stats["flat"] = bench_faiss(
                flat_index, query_embeddings, args.k, args.repeats, args.warmup
            )

        if "hnsw" in engines:
            print("Construyendo FAISS HNSW...", flush=True)
            hnsw_index = build_faiss_index(sliced_embeddings, "hnsw")
            latency_stats["hnsw"] = bench_faiss(
                hnsw_index, query_embeddings, args.k, args.repeats, args.warmup
            )

        if "turbovec" in engines:
            print("Construyendo TurboVec...", flush=True)
            turbovec_index = TurboVecIndex(chunks[:size], sliced_embeddings, bit_width=4)
            latency_stats["turbovec"] = bench_turbovec(
                turbovec_index, query_embeddings, args.k, args.repeats, args.warmup
            )

        latency_ms = {engine: stats["median"] for engine, stats in latency_stats.items()}
        speedup_vs_flat = {}
        if "flat" in latency_stats:
            flat_latency = latency_stats["flat"]["median"]
            speedup_vs_flat = {
                engine: round(flat_latency / stats["median"], 1)
                for engine, stats in latency_stats.items()
                if engine != "flat" and stats["median"] > 0
            }

        res = {
            "vectors": size,
            "queries": len(queries),
            "k": args.k,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "threads": args.threads,
            "measurement": "query_by_query_ms_per_query",
            "latency_ms": latency_ms,
            "latency_stats_ms": latency_stats,
            "speedup_vs_flat": speedup_vs_flat,
        }
        results.append(res)
        print(f"Resultados para {size}: {res['latency_ms']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados guardados en {args.output}")


if __name__ == "__main__":
    run()
