# Guía de ejecución — MITEX Assistant

Guía práctica para levantar y probar el asistente RAG sobre MIRACL-es (Wikipedia en
español). Para la arquitectura y la cobertura del enunciado, ver `README.md`.

---

## 1. Requisitos

- **Python 3.12** (PyTorch no publica *wheels* para 3.14; usa 3.12).
- **Ollama** en marcha con el modelo `gemma4:e4b`.
- GPU NVIDIA opcional (acelera el indexado; funciona también en CPU).

```bash
# Modelo LLM local
ollama serve &              # arranca el servidor (si no está ya activo)
ollama pull gemma4:e4b      # descarga el modelo (~9.6 GB)
```

---

## 2. Instalación

```bash
# Crear entorno con Python 3.12
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -e .

# (opcional) extra de cuantización para el benchmark TurboVec
pip install turbovec
```

> Si usas `mise`: `mise install python@3.12` y luego
> `~/.local/share/mise/installs/python/3.12.13/bin/python -m venv .venv312`.

---

## 3. Inicio rápido (índice ya construido)

Si ya existe `data/indexes/default_100k/` (corrida documentada en el informe), puedes
chatear directamente:

```bash
source .venv312/bin/activate
PYTHONPATH=src mitex chat \
  --index-dir data/indexes/default_100k \
  --topics-report data/reports/topics_100k.json
```

En el chat: escribe `nuevo` para reiniciar la memoria conversacional y `salir` para terminar.

---

## 4. Construir todo desde cero (100 000 documentos)

```bash
source .venv312/bin/activate

# 1) Descargar corpus + evaluación (Wikipedia ES, qrels expertos)
PYTHONPATH=src mitex acquire-corpus \
  --corpus-output data/processed/miracl_es_corpus_100k.jsonl \
  --eval-output data/reports/miracl_es_eval_100k.jsonl \
  --limit 100000 --eval-limit 200

# 2) Construir índice FAISS (exacto) + BM25
PYTHONPATH=src mitex build-index \
  --corpus data/processed/miracl_es_corpus_100k.jsonl \
  --index-dir data/indexes/default_100k \
  --index-type flat

#    (opcional) índice ANN aproximado HNSW, ~16x más rápido
PYTHONPATH=src mitex build-index \
  --corpus data/processed/miracl_es_corpus_100k.jsonl \
  --index-dir data/indexes/default_100k_hnsw \
  --index-type hnsw

# 3) Modelado de tópicos offline (BERTopic)
PYTHONPATH=src mitex topics \
  --corpus data/processed/miracl_es_corpus_100k.jsonl \
  --output data/reports/topics_100k.json

# 4) Chatear
PYTHONPATH=src mitex chat \
  --index-dir data/indexes/default_100k \
  --topics-report data/reports/topics_100k.json
```

---

## 5. Consulta puntual (sin entrar al chat)

```bash
PYTHONPATH=src mitex ask \
  --index-dir data/indexes/default_100k \
  --topics-report data/reports/topics_100k.json \
  "¿Cómo se convirtió Egipto en musulmán?"
```

---

## 6. Benchmarks de validación

```bash
IDX=data/indexes/default_100k
Q=data/reports/miracl_es_eval_100k.jsonl

# IR: léxico vs semántico vs híbrido vs híbrido+reranking (200 queries, k=10)
for mode in lexical semantic hybrid; do
  PYTHONPATH=src mitex eval-ir-file --index-dir $IDX --queries $Q \
    --mode $mode --k 10 --limit 200 --output data/reports/eval_100k_$mode.json
done
PYTHONPATH=src mitex eval-ir-file --index-dir $IDX --queries $Q \
  --mode hybrid --k 10 --limit 200 --rerank \
  --output data/reports/eval_100k_hybrid_rerank.json

# Benchmark ANN HNSW vs Flat (latencia + recall)
PYTHONPATH=src python scripts/bench_ann_hnsw.py

# Benchmark de cuantización TurboVec (memoria)
PYTHONPATH=src mitex build-turbovec-index \
  --corpus data/processed/miracl_es_corpus_100k.jsonl \
  --index-dir data/indexes/default_100k_turbovec --bit-width 4
PYTHONPATH=src mitex bench-turbovec \
  --faiss-index-dir $IDX \
  --turbovec-index-dir data/indexes/default_100k_turbovec \
  --queries $Q --limit 200 --k 10 \
  --output data/reports/turbovec_benchmark_100k.json
```

---

## 7. Tests

```bash
PYTHONPATH=src pytest -q
```

---

## 8. Variables de entorno útiles

| Variable | Por defecto | Para qué |
|----------|-------------|----------|
| `MITEX_OLLAMA_MODEL` | `gemma4:e4b` | Modelo LLM de Ollama |
| `MITEX_USE_RERANKER` | `false` | Activa el cross-encoder en chat/ask |
| `MITEX_USE_LLM_ROUTER` | `true` | El LLM elige herramienta ante ambigüedad |
| `MITEX_INDEX_TYPE` | `flat` | `flat` (exacto) o `hnsw` (ANN) |
| `MITEX_WEB_SEARCH_RESULTS` | `4` | Nº de resultados web a integrar |

Ejemplo con reranking activado:

```bash
MITEX_USE_RERANKER=true PYTHONPATH=src mitex chat \
  --index-dir data/indexes/default_100k \
  --topics-report data/reports/topics_100k.json
```

---

## 9. Queries de ejemplo para probar cada herramienta

Dentro de `mitex chat`:

| Tipo | Ejemplo de consulta | Herramienta activada |
|------|---------------------|----------------------|
| Factual (RAG) | `¿Cómo se convirtió Egipto en musulmán?` | RAG híbrido + citas |
| Factual (RAG) | `¿Qué fue la Revolución Industrial?` | RAG híbrido |
| Resumen (QFS) | `Resume lo que tengas sobre la mitología griega` | TextRank + MMR |
| Tópicos | `¿Qué temas o tópicos hay en el corpus?` | BERTopic offline |
| Fecha | `¿Qué día es hoy?` | DateTool (sin LLM) |
| Web | `Busca en internet el precio actual del oro` | WebSearchTool |
| Seguimiento | `¿Cuál es el país más poblado de África?` → luego `¿y su capital?` | Memoria conversacional |
