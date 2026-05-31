# MITEX Assistant: Asistente RAG sobre Wikipedia en español

Implementación modular de un asistente inteligente para el proyecto final de Minería de Textos (MITEX).

El caso de uso es un **asistente de consulta enciclopédica en español** basado en
MIRACL-es, un corpus real de recuperación sobre **Wikipedia en español** con
consultas y juicios de relevancia expertos.

## Arquitectura

- `acquisition`: descarga de MIRACL-es desde HuggingFace y generación de corpus/evaluación local.
- `ingestion`: extracción de PDF/TXT/HTML, normalización, deduplicación y filtrado de inyecciones.
- `indexing`: chunking, embeddings con Sentence Transformers, índice FAISS (exacto `flat` o ANN `hnsw`) e índice BM25.
- `indexing.turbovec_*`: índice cuantizado opcional con TurboVec/TurboQuant para comparativa.
- `topics`: modelado de tópicos offline con BERTopic o NMF como fallback ligero.
- `summarization`: resumen extractivo con TextRank y selección MMR.
- `retrieval`: recuperación híbrida FAISS+BM25 con Reciprocal Rank Fusion (RRF) y reranking opcional con cross-encoder.
- `llm`: cliente LLM configurable (Ollama por defecto).
- `agents`: router agencial (regex + desambiguación con el LLM en JSON) con herramientas RAG, resumen, tópicos, fecha, búsqueda web y memoria conversacional.
- `eval`: métricas de recuperación (Precision@k, Recall@k, MRR) y benchmark de cuantización.
- `cli`: comandos de ingesta, indexado, tópicos, chat y evaluación.

## Instalación

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

Extra opcional para TurboQuant:

```bash
pip install -e ".[turbovec]"
```

El proveedor LLM por defecto es Ollama y el modelo usado en las pruebas es `gemma4:e4b`.
Se puede cambiar con `MITEX_OLLAMA_MODEL`. El endpoint por defecto es
`MITEX_OLLAMA_URL=http://localhost:11434`.

```bash
ollama pull gemma4:e4b
```

## Evaluación Rápida (Demo)

Para probar el asistente sin reconstruir el índice, instala las dependencias y lanza el
chat sobre un índice ya compilado. Asegúrate de tener Python 3 y Ollama en marcha:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull gemma4:e4b
mitex chat \
  --index-dir data/indexes/default \
  --topics-report data/reports/topics.json
```

El dataset de entrega (`MITEX_DATA.tar.gz`) contiene un corpus local de **1 000 000**
documentos MIRACL-es, el índice `data/indexes/default` con **1 140 063 chunks**,
`data/reports/topics.json` y el paquete `data/reports/miracl_es_corpus_package.tar.gz`
con corpus, evaluación y manifiesto.

## Flujo recomendado

```bash
# 1. Descargar corpus MIRACL-es (Wikipedia en español)
mitex acquire-corpus \
  --corpus-output data/processed/miracl_es_corpus_1M.jsonl \
  --eval-output data/reports/miracl_es_eval.jsonl \
  --limit 1000000 --eval-limit 200

# 2. Construir índice FAISS + BM25 (1 000 000 documentos)
#    --index-type flat (exacto, por defecto) o hnsw (ANN aproximado, más rápido a gran escala)
mitex build-index \
  --corpus data/processed/miracl_es_corpus_1M.jsonl \
  --index-dir data/indexes/default \
  --index-type flat

# 3. Modelado de tópicos offline
#    En la entrega se proporciona topics.json generado sobre un subconjunto de 100 198 documentos.
mitex topics \
  --corpus data/processed/miracl_es_corpus_1M.jsonl \
  --output data/reports/topics.json \
  --limit 100198

# 4. Empaquetar corpus para entrega (los tópicos se entregan como JSON aparte)
mitex package-corpus \
  --corpus data/processed/miracl_es_corpus_1M.jsonl \
  --eval data/reports/miracl_es_eval.jsonl \
  --output data/reports/miracl_es_corpus_package.tar.gz

# 5. Chat interactivo
mitex chat \
  --index-dir data/indexes/default \
  --topics-report data/reports/topics.json
```

También puedes usar una consulta puntual:

```bash
mitex ask \
  --index-dir data/indexes/default \
  --topics-report data/reports/topics.json \
  "¿Cuál es el río más largo de España?"
```

## Cobertura del enunciado

| Requisito del enunciado | Módulo del proyecto | Descripción |
|-------------------------|---------------------|-------------|
| Embeddings | `indexing/embeddings.py` | Sentence Transformers multilingüe |
| Modelado de tópicos | `topics/modeling.py` | BERTopic / NMF offline sobre el corpus |
| Resumen extractivo | `summarization/extractive.py` | TextRank con selección MMR |
| LLM | `llm/ollama.py` | Ollama con gemma4:e4b |
| RAG | `retrieval/hybrid.py` | FAISS semántico + BM25 léxico + RRF |
| Agentes | `agents/router.py` | Router regex + LLM (JSON) con herramientas RAG, resumen, tópicos, fecha y búsqueda web |
| Extra: adquisición | `acquisition/miracl.py` | Descarga automática de MIRACL-es |
| Extra: limpieza | `ingestion/`, `utils/text.py` | Normalización, deduplicación, filtrado anti-inyección |
| Extra: cuantización | `indexing/turbovec_store.py` | Benchmark FAISS vs TurboVec |
| Extra: reranking | `retrieval/reranking.py` | Cross-encoder multilingüe sobre los candidatos RRF |
| Extra: ANN | `indexing/vector_store.py` | Índice FAISS HNSW aproximado para escala |
| Extra: búsqueda web | `agents/web_search.py` | Información complementaria de DuckDuckGo integrada en el prompt |

## Validación

```bash
python3 -m compileall src
PYTHONPATH=src pytest
ruff check .
```

Evaluación automática con las consultas etiquetadas del corpus:

```bash
mitex eval-ir-file --index-dir data/indexes/default \
  --queries data/reports/miracl_es_eval.jsonl \
  --mode lexical --k 10 --limit 50 \
  --output data/reports/doc_eval_lexical.json

mitex eval-ir-file --index-dir data/indexes/default \
  --queries data/reports/miracl_es_eval.jsonl \
  --mode semantic --k 10 --limit 50 \
  --output data/reports/doc_eval_semantic.json

mitex eval-ir-file --index-dir data/indexes/default \
  --queries data/reports/miracl_es_eval.jsonl \
  --mode hybrid --k 10 --limit 50 \
  --output data/reports/doc_eval_hybrid.json

# Comparativa con reranking cross-encoder sobre la fusión híbrida.
# Este comando genera el informe opcional de reranking.
mitex eval-ir-file --index-dir data/indexes/default \
  --queries data/reports/miracl_es_eval.jsonl \
  --mode hybrid --k 10 --limit 50 --rerank \
  --output data/reports/doc_eval_hybrid_rerank.json
```

## Configuración (variables de entorno)

| Variable | Por defecto | Descripción |
|----------|-------------|-------------|
| `MITEX_INDEX_TYPE` | `flat` | Tipo de índice FAISS: `flat` (exacto) o `hnsw` (ANN). |
| `MITEX_HNSW_M` | `32` | Grado del grafo HNSW cuando `index_type=hnsw`. |
| `MITEX_USE_RERANKER` | `false` | Activa el cross-encoder reranker en chat/ask. |
| `MITEX_RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Modelo cross-encoder multilingüe. |
| `MITEX_RERANK_TOP_K` | `5` | Nº de resultados finales tras el reranking. |
| `MITEX_USE_LLM_ROUTER` | `true` | El LLM desambigua la herramienta ante consultas ambiguas. |
| `MITEX_WEB_SEARCH_RESULTS` | `4` | Nº de resultados web a integrar. |
| `MITEX_OLLAMA_MODEL` | `gemma4:e4b` | Modelo LLM servido por Ollama. |

En el chat interactivo, escribe `nuevo` para reiniciar la memoria conversacional y `salir` para terminar.

## TurboQuant (benchmark opcional)

```bash
mitex build-turbovec-index \
  --corpus data/processed/miracl_es_corpus_1M.jsonl \
  --index-dir data/indexes/default_turbovec --bit-width 4

mitex bench-turbovec \
  --faiss-index-dir data/indexes/default \
  --turbovec-index-dir data/indexes/default_turbovec \
  --queries data/reports/miracl_es_eval.jsonl \
  --limit 50 --k 10 \
  --output data/reports/doc_turbovec_benchmark.json
```

## Informe

El informe está en `docs/informe_mitex.tex`. Para generar el PDF:

```bash
pdflatex -interaction=nonstopmode -output-directory docs docs/informe_mitex.tex
pdflatex -interaction=nonstopmode -output-directory docs docs/informe_mitex.tex
```
