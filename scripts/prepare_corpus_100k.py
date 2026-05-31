"""Genera el corpus de 100k y su evaluación alineada a partir de un volcado local.

Toma los primeros 100 000 documentos del corpus MIRACL-es y, además, garantiza que
todos los documentos relevantes de las consultas (qrels) estén incluidos, de modo que
las métricas IR midan el recuperador y no la cobertura. Alternativa equivalente desde
cero (descargando de HuggingFace):

    mitex acquire-corpus --limit 100000 --eval-limit 200

Uso: python scripts/prepare_corpus_100k.py
"""

from __future__ import annotations

import json
from pathlib import Path

SRC = Path("data/raw/MITEX_DATA/data/processed/miracl_es_corpus_1M.jsonl")
EVAL_SRC = Path("data/raw/MITEX_DATA/data/reports/miracl_es_eval.jsonl")
CORPUS_OUT = Path("data/processed/miracl_es_corpus_100k.jsonl")
EVAL_OUT = Path("data/reports/miracl_es_eval_100k.jsonl")
LIMIT = 100_000

CORPUS_OUT.parent.mkdir(parents=True, exist_ok=True)
EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)

# Conjunto de docs relevantes de la evaluacion (para garantizar qrels en el corpus)
eval_rows = [
    json.loads(line)
    for line in EVAL_SRC.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
wanted = {doc_id for row in eval_rows for doc_id in row["relevant_doc_ids"]}

included = set()
regular = 0
extra_eval = 0
with SRC.open("r", encoding="utf-8") as fin, CORPUS_OUT.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        doc = json.loads(line)
        doc_id = doc["doc_id"]
        keep_regular = regular < LIMIT
        keep_eval = doc_id in wanted and doc_id not in included
        if keep_regular:
            fout.write(line)
            included.add(doc_id)
            regular += 1
        elif keep_eval:
            fout.write(line)
            included.add(doc_id)
            extra_eval += 1
        # parar cuando tenemos los 100k regulares y ya pasamos a recoger evals;
        # seguimos escaneando hasta agotar wanted para no perder qrels
        if regular >= LIMIT and wanted.issubset(included):
            break

# Reconstruir eval alineada al subconjunto 100k
kept = []
for row in eval_rows:
    relevant_doc_ids = [doc_id for doc_id in row["relevant_doc_ids"] if doc_id in included]
    if relevant_doc_ids:
        kept.append({**row, "relevant_doc_ids": relevant_doc_ids})
with EVAL_OUT.open("w", encoding="utf-8") as file:
    for row in kept:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"corpus_docs={regular + extra_eval} (regular={regular}, extra_eval={extra_eval})")
print(f"eval_queries_kept={len(kept)} / {len(eval_rows)}")
print(f"corpus -> {CORPUS_OUT}")
print(f"eval   -> {EVAL_OUT}")
