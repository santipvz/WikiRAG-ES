# MITEX Assistant

Implementacion modular de un asistente inteligente para el proyecto final de Mineria de Textos.

## Arquitectura

- `ingestion`: extraccion de PDFs/TXT/HTML y limpieza.
- `indexing`: chunking, embeddings, indice FAISS y BM25.
- `topics`: modelado de topicos offline con BERTopic.
- `summarization`: resumen extractivo con TextRank y MMR.
- `retrieval`: recuperacion hibrida y preparacion de contexto RAG.
- `llm`: clientes de LLM configurables.
- `agents`: router agencial y herramientas del sistema.
- `eval`: metricas simples para IR/RAG.
- `cli`: comandos de ingesta, indexado, topicos, chat y evaluacion.

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m spacy download es_core_news_sm
```

Si usas Ollama como proveedor de LLM, instala el modelo configurado en `.env` o ajustalo por otro compatible:

```bash
ollama pull gemma3:4b
```

## Flujo recomendado

```bash
mitex ingest --input Teoria --output data/processed/corpus.jsonl
mitex build-index --corpus data/processed/corpus.jsonl --index-dir data/indexes/default
mitex topics --corpus data/processed/corpus.jsonl --output data/reports/topics.json
mitex chat --index-dir data/indexes/default
```

Tambien puedes usar una consulta puntual:

```bash
mitex ask --index-dir data/indexes/default "Que pide el proyecto?"
```

## Validacion

```bash
python -m compileall src
pytest
```

Para evaluar recuperacion con etiquetas manuales:

```bash
mitex eval-ir --index-dir data/indexes/default --query "RAG y agentes" --relevant CHUNK_ID_1 --relevant CHUNK_ID_2
```

Los `chunk_id` se consultan en `data/indexes/default/chunks.jsonl`.
