# WikiRAG-ES: Asistente RAG sobre Wikipedia en español

Asistente de consulta enciclopédica en español basado en MIRACL-es, utilizando recuperación híbrida (FAISS Semántico + BM25 Léxico) y modelos LLM en local (Ollama).

## Instalación y Requisitos

Requiere Python 3.10+ y [Ollama](https://ollama.com/) instalado.

```bash
# Entorno virtual y dependencias
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# (Opcional) Soporte para índices cuantizados TurboVec
pip install -e ".[turbovec]"

# Descargar modelo LLM por defecto
ollama pull gemma4:e4b
```

## Ejecución (Chat y Consultas)

Si ya se dispone de un índice precomputado se puede lanzar el chat interactivo directamente:

```bash
wikirag chat \
  --index-dir data/indexes/default \
  --topics-report data/reports/topics.json
```

Para una consulta puntual sin entrar al modo chat:

```bash
wikirag ask --index-dir data/indexes/default "¿Cuál es el río más largo de España?"
```

## Creación de Índices (Opciones)

El sistema soporta distintos motores de búsqueda vectorial (FAISS y TurboVec). Para construirlos a partir del corpus, usa el comando `build-index`:

### 1. Índice Flat (Exacto)

Búsqueda exhaustiva por producto interno. Es el motor por defecto. Recomendado para corpus pequeños/medianos.

```bash
wikirag build-index --corpus data/processed/miracl_es_corpus.jsonl \
  --index-dir data/indexes/default \
  --index-type flat
```

### 2. Índice HNSW (Aproximado)

Utiliza un grafo HNSW para una recuperación sublineal. Altamente recomendado para escalar a más de 1M de documentos, ya que reduce la latencia semántica drásticamente.

```bash
wikirag build-index --corpus data/processed/miracl_es_corpus.jsonl \
  --index-dir data/indexes/hnsw \
  --index-type hnsw
```

### 3. Índice TurboVec (Benchmarking / Cuantización)

Índice experimental cuantizado (2 o 4 bits) para comparar métricas de compresión de memoria. Ahora integrado en los comandos `chat` y `ask`. Puedes probar su velocidad pasándole el flag `--turbovec`. Al utilizar este flag, el sistema omite el índice híbrido BM25 y realiza **exclusivamente búsqueda semántica rápida**.

```bash
wikirag build-turbovec-index --corpus data/processed/miracl_es_corpus.jsonl \
  --index-dir data/indexes/turbovec \
  --bit-width 4

# Probar en el chat (Bypass de BM25, respuesta ultrarrápida)
wikirag chat --index-dir data/indexes/turbovec --turbovec
```

## Flujo Completo (Desde Cero)

Si quieres generar todo desde cero, sigue estos pasos:

```bash
# 1. Adquisición del corpus MIRACL
wikirag acquire-corpus --corpus-output data/processed/miracl_es_corpus.jsonl --limit 1000000

# 2. Construcción del índice híbrido (FAISS + BM25)
wikirag build-index --corpus data/processed/miracl_es_corpus.jsonl --index-dir data/indexes/default --index-type hnsw

# 3. Modelado de Tópicos
wikirag topics --corpus data/processed/miracl_es_corpus.jsonl --output data/reports/topics.json --limit 100000

# 4. Evaluación de métricas (Precision, Recall, MRR)
wikirag eval-ir-file --index-dir data/indexes/default --queries data/reports/miracl_es_eval.jsonl --mode hybrid
```

## Variables de Entorno y Configuración

| Variable                   | Por defecto    | Descripción                                          |
| -------------------------- | -------------- | ----------------------------------------------------- |
| `WIKIRAG_INDEX_TYPE`     | `flat`       | `flat` (exacto) o `hnsw` (ANN).                   |
| `WIKIRAG_OLLAMA_MODEL`   | `gemma4:e4b` | Modelo a utilizar en Ollama.                          |
| `WIKIRAG_USE_RERANKER`   | `false`      | Activa un cross-encoder para mejorar el ordenamiento. |
| `WIKIRAG_RERANK_TOP_K`   | `5`          | Resultados devueltos por el reranker.                 |
| `WIKIRAG_USE_LLM_ROUTER` | `true`       | El LLM desambigua intenciones vs regex.               |
