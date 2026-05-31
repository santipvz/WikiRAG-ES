from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from wikirag_es.data.models import Chunk
from wikirag_es.eval.batch import evaluate_ir_file
from wikirag_es.indexing.embeddings import EmbeddingModel
from wikirag_es.indexing.vector_store import build_hybrid_index

def run():
    index_dir = Path("data/indexes/default")
    queries_path = Path("data/reports/miracl_es_eval.jsonl")
    output_path = Path("data/reports/bench_scaling.json")

    print("Cargando embeddings y chunks base...")
    embeddings = np.load(index_dir / "embeddings.npy")
    chunks = []
    with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))
    
    print(f"Total chunks disponibles: {len(chunks)}")
    
    import tempfile
    
    sizes = [100_000, 500_000, len(chunks)]
    results = []
    
    print("Cargando modelo de embeddings...")
    model = EmbeddingModel("paraphrase-multilingual-MiniLM-L12-v2")
    
    for size in sizes:
        print(f"\n--- Evaluando tamano: {size} chunks ---")
        sliced_chunks = chunks[:size]
        sliced_embeddings = embeddings[:size]
        
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            print("Construyendo indice hibrido en memoria/disco...")
            t0 = time.time()
            build_hybrid_index(sliced_chunks, sliced_embeddings, temp_dir)
            print(f"Indice construido en {time.time() - t0:.2f}s")
            
            print("Evaluando queries (hibrido)...")
            t0 = time.time()
            # limitamos a 50 queries tal y como dice el informe para que sea rapido
            report = evaluate_ir_file(
                index_dir=temp_dir,
                queries_path=queries_path,
                embedding_model=model,
                mode="hybrid",
                k=10,
                limit=50
            )
            eval_time = time.time() - t0
            
            queries_count = report["queries"]
            latency_ms = (eval_time / queries_count) * 1000
            
            res = {
                "chunks": size,
                "latency_ms": round(latency_ms, 2),
                "precision_at_k": round(report["average"]["precision_at_k"], 3),
                "recall_at_k": round(report["average"]["recall_at_k"], 3),
                "mrr": round(report["average"]["mrr"], 3),
            }
            results.append(res)
            print(f"Resultados para {size}: {res}")
            
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados guardados en {output_path}")

if __name__ == "__main__":
    run()
