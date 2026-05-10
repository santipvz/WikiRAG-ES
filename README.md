# MITEX Assistant

Implementacion modular de un asistente inteligente para el proyecto final de Mineria de Textos.

## Arquitectura

- `ingestion`: extraccion de PDFs/TXT/HTML y limpieza.
- `indexing`: chunking, embeddings, indice FAISS y BM25.
- `topics`: modelado de topicos offline con BERTopic.
- `summarization`: resumen extractivo con TextRank y MMR.
- `retrieval`: recuperacion hibrida y preparacion de contexto RAG.
- `llm`: clientes de LLM, incluyendo Ollama local.
- `agents`: router agencial y herramientas del sistema.
- `eval`: metricas simples para IR/RAG.
- `cli`: comandos de ingesta, indexado, topicos, chat y evaluacion.

## Instalacion con fish

```fish
python -m venv .venv
source .venv/bin/activate.fish
pip install -e .
python -m spacy download es_core_news_sm
```

Si usas Ollama:

```fish
ollama pull llama3.1:8b-instruct-q4_K_M
```

## Flujo recomendado

```fish
mitex ingest --input Teoria --output data/processed/corpus.jsonl
mitex build-index --corpus data/processed/corpus.jsonl --index-dir data/indexes/default
mitex topics --corpus data/processed/corpus.jsonl --output data/reports/topics.json
mitex chat --index-dir data/indexes/default
```

Tambien puedes usar una consulta puntual:

```fish
mitex ask --index-dir data/indexes/default "Que pide el proyecto?"
```

## Validacion

```fish
python -m compileall src
pytest
```

Para evaluar recuperacion con etiquetas manuales:

```fish
mitex eval-ir --index-dir data/indexes/default --query "RAG y agentes" --relevant CHUNK_ID_1 --relevant CHUNK_ID_2
```

Los `chunk_id` se consultan en `data/indexes/default/chunks.jsonl`.

## Nota sobre hardware

El equipo tiene una RTX 5060 Ti con 8 GB de VRAM. Para generacion local se recomienda un modelo cuantizado de 3B-8B mediante Ollama. Los embeddings y BERTopic pueden ejecutarse en CPU si no hay CUDA disponible para PyTorch.
