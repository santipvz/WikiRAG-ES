from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import numpy as np

from wikirag_es.data.models import Chunk
from wikirag_es.indexing.embeddings import EmbeddingModel
from wikirag_es.indexing.turbovec_store import TurboVecIndex
from wikirag_es.indexing.vector_store import build_faiss_index
from wikirag_es.data.jsonl import read_jsonl

def bench_faiss(index: faiss.Index, query_embeddings: np.ndarray, k: int) -> float:
    # warmup
    index.search(query_embeddings[: min(5, len(query_embeddings))], k)
    start = time.perf_counter()
    index.search(query_embeddings, k)
    return time.perf_counter() - start

def bench_turbovec(index: TurboVecIndex, query_embeddings: np.ndarray, k: int) -> float:
    # warmup
    index.semantic_search(query_embeddings[: min(5, len(query_embeddings))], k)
    start = time.perf_counter()
    # turbovec semantic search takes single queries in a loop internally normally, but it supports batched arrays
    for row in query_embeddings:
        index.semantic_search(row, k)
    return time.perf_counter() - start

def run():
    index_dir = Path("data/indexes/default")
    queries_path = Path("data/reports/miracl_es_eval.jsonl")
    output_path = Path("data/reports/bench_scaling_techniques.json")

    print("Cargando embeddings y chunks...")
    embeddings = np.load(index_dir / "embeddings.npy")
    chunks = []
    with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))
    
    queries = [str(row["query"]) for row in read_jsonl(queries_path)]
    
    print("Codificando queries...")
    model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    query_embeddings = model.encode(queries)
    
    sizes = [100_000, 500_000, len(embeddings)]
    results = []
    
    for size in sizes:
        print(f"\n--- Evaluando tamano: {size} vectores ---")
        sliced_embeddings = embeddings[:size]
        sliced_chunks = chunks[:size]
        
        # 1. Flat FAISS
        print("Construyendo FAISS Flat...")
        flat_index = build_faiss_index(sliced_embeddings, "flat")
        flat_time = bench_faiss(flat_index, query_embeddings, 10)
        flat_ms = (flat_time / len(queries)) * 1000
        
        # 2. HNSW FAISS
        print("Construyendo FAISS HNSW...")
        hnsw_index = build_faiss_index(sliced_embeddings, "hnsw")
        hnsw_time = bench_faiss(hnsw_index, query_embeddings, 10)
        hnsw_ms = (hnsw_time / len(queries)) * 1000
        
        # 3. TurboVec
        print("Construyendo TurboVec...")
        turbovec_index = TurboVecIndex(sliced_chunks, sliced_embeddings, bit_width=4)
        turbovec_time = bench_turbovec(turbovec_index, query_embeddings, 10)
        turbovec_ms = (turbovec_time / len(queries)) * 1000
        
        res = {
            "vectors": size,
            "queries": len(queries),
            "latency_ms": {
                "flat": round(flat_ms, 3),
                "hnsw": round(hnsw_ms, 3),
                "turbovec": round(turbovec_ms, 3)
            }
        }
        results.append(res)
        print(f"Resultados para {size}: {res['latency_ms']}")
            
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados guardados en {output_path}")

if __name__ == "__main__":
    run()
