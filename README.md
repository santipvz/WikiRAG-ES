# MITEX Assistant — Asistente RAG sobre Wikipedia en español

Implementación modular de un asistente inteligente para el proyecto final de Minería de Textos (MITEX).

El caso de uso es un **asistente de consulta enciclopédica en español** basado en
MIRACL-es, un corpus real de recuperación sobre **Wikipedia en español** con
consultas y juicios de relevancia expertos.

## Arquitectura

- `acquisition`: descarga de MIRACL-es desde HuggingFace y generación de corpus/evaluación local.
- `ingestion`: extracción de PDF/TXT/HTML, normalización, deduplicación y filtrado de inyecciones.
- `indexing`: chunking, embeddings con Sentence Transformers, índice FAISS e índice BM25.
- `indexing.turbovec_*`: índice cuantizado opcional con TurboVec/TurboQuant para comparativa.
- `topics`: modelado de tópicos offline con BERTopic o NMF como fallback ligero.
- `summarization`: resumen extractivo con TextRank y selección MMR.
- `retrieval`: recuperación híbrida FAISS+BM25 con Reciprocal Rank Fusion (RRF).
- `llm`: cliente LLM configurable (Ollama por defecto).
- `agents`: router agencial con herramientas RAG, resumen, tópicos y fecha.
- `eval`: métricas de recuperación (Precision@k, Recall@k, MRR) y benchmark de cuantización.
- `cli`: comandos de ingesta, indexado, tópicos, chat y evaluación.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m spacy download es_core_news_sm
```

Extra opcional para TurboQuant:

```bash
pip install -e .[turbovec]
```

El proveedor LLM por defecto es Ollama y el modelo usado en las pruebas es `gemma3:4b`.
Se puede cambiar con `MITEX_OLLAMA_MODEL`; el endpoint por defecto es
`MITEX_OLLAMA_URL=http://localhost:11434`.

```bash
ollama pull gemma3:4b
```

## Evaluación Rápida (Demo)

Para evaluar el proyecto sin necesidad de descargar el corpus de 1M de documentos ni indexarlo, se ha incluido un script de lanzamiento automático. Asegúrese de tener Python 3 y Ollama instalados, y ejecute:

```bash
./run_demo.sh
```

Este script creará el entorno virtual, instalará las dependencias de forma silenciosa e iniciará el Agente Router en modo chat interactivo cargando el índice ya pre-compilado de 1.14 millones de vectores.

## Flujo recomendado

```bash
# 1. Descargar corpus MIRACL-es (Wikipedia en español)
mitex acquire-corpus \
  --corpus-output data/processed/miracl_es_corpus.jsonl \
  --eval-output data/reports/miracl_es_eval.jsonl \
  --limit 100000 --eval-limit 200

# 2. Construir índice FAISS + BM25
mitex build-index \
  --corpus data/processed/miracl_es_corpus.jsonl \
  --index-dir data/indexes/default

# 3. Modelado de tópicos offline
mitex topics \
  --corpus data/processed/miracl_es_corpus.jsonl \
  --output data/reports/topics.json

# 4. Empaquetar corpus para entrega
mitex package-corpus \
  --corpus data/processed/miracl_es_corpus.jsonl \
  --eval data/reports/miracl_es_eval.jsonl \
  --topics-report data/reports/topics.json \
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
| LLM | `llm/ollama.py` | Ollama con gemma3:4b |
| RAG | `retrieval/hybrid.py` | FAISS semántico + BM25 léxico + RRF |
| Agentes | `agents/router.py` | Router con herramientas RAG, resumen, tópicos, fecha |
| Extra: adquisición | `acquisition/miracl.py` | Descarga automática de MIRACL-es |
| Extra: limpieza | `ingestion/`, `utils/text.py` | Normalización, deduplicación, filtrado |
| Extra: cuantización | `indexing/turbovec_store.py` | Benchmark FAISS vs TurboVec |

## Validación

```bash
python3 -m compileall src
PYTHONPATH=src pytest
```

Evaluación automática con las consultas etiquetadas del corpus:

```bash
mitex eval-ir-file --index-dir data/indexes/default \
  --queries data/reports/miracl_es_eval.jsonl \
  --mode lexical --k 10 --limit 200 \
  --output data/reports/eval_lexical.json

mitex eval-ir-file --index-dir data/indexes/default \
  --queries data/reports/miracl_es_eval.jsonl \
  --mode semantic --k 10 --limit 200 \
  --output data/reports/eval_semantic.json

mitex eval-ir-file --index-dir data/indexes/default \
  --queries data/reports/miracl_es_eval.jsonl \
  --mode hybrid --k 10 --limit 200 \
  --output data/reports/eval_hybrid.json
```

## TurboQuant (benchmark opcional)

```bash
mitex build-turbovec-index \
  --corpus data/processed/miracl_es_corpus.jsonl \
  --index-dir data/indexes/default_turbovec --bit-width 4

mitex bench-turbovec \
  --faiss-index-dir data/indexes/default \
  --turbovec-index-dir data/indexes/default_turbovec \
  --queries data/reports/miracl_es_eval.jsonl \
  --limit 50 --k 10 \
  --output data/reports/turbovec_benchmark.json
```

## Informe

El informe está en `docs/informe_mitex.tex`. Para generar el PDF:

```bash
pdflatex -interaction=nonstopmode -output-directory docs docs/informe_mitex.tex
pdflatex -interaction=nonstopmode -output-directory docs docs/informe_mitex.tex
```
