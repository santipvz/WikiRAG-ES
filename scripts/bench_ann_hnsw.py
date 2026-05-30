"""Benchmark de latencia y recall del índice ANN HNSW frente al FAISS Flat exacto.

Requiere haber construido ambos índices sobre los mismos vectores:
    mitex build-index --corpus <corpus> --index-dir data/indexes/default_100k --index-type flat
    # y un índice hnsw con los mismos embeddings (ver informe / README)

Uso: PYTHONPATH=src python scripts/bench_ann_hnsw.py
"""
import json, time
from pathlib import Path
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

Q = Path("data/reports/miracl_es_eval_100k.jsonl")
rows = [json.loads(l) for l in Q.read_text(encoding="utf-8").splitlines() if l.strip()]
queries = [r["query"] for r in rows]

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
qemb = model.encode(queries, normalize_embeddings=True, show_progress_bar=False).astype("float32")

flat = faiss.read_index("data/indexes/default_100k/faiss.index")
hnsw = faiss.read_index("data/indexes/default_100k_hnsw/faiss.index")
K = 10

def bench(index, name):
    # warmup
    index.search(qemb[:5], K)
    t0 = time.perf_counter()
    D, I = index.search(qemb, K)
    dt = time.perf_counter() - t0
    return I, dt

flat_I, flat_dt = bench(flat, "flat")
hnsw_I, hnsw_dt = bench(hnsw, "hnsw")

# Recall del ANN respecto al exacto (solape de top-10)
overlap = np.mean([len(set(flat_I[i]) & set(hnsw_I[i])) / K for i in range(len(queries))])
top1_match = np.mean([flat_I[i][0] == hnsw_I[i][0] for i in range(len(queries))])

n = len(queries)
result = {
    "queries": n, "k": K,
    "flat": {"total_s": round(flat_dt,4), "avg_ms": round(flat_dt/n*1000,3)},
    "hnsw": {"total_s": round(hnsw_dt,4), "avg_ms": round(hnsw_dt/n*1000,3)},
    "speedup": round(flat_dt/hnsw_dt, 2),
    "ann_recall@10_vs_exact": round(float(overlap), 4),
    "ann_top1_match": round(float(top1_match), 4),
}
print(json.dumps(result, indent=2))
Path("data/reports/ann_hnsw_vs_flat_100k.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
