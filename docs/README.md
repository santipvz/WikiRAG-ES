# Documentacion MITEX

Compilar el informe final:

```bash
pdflatex -interaction=nonstopmode -output-directory docs docs/informe_mitex.tex
pdflatex -interaction=nonstopmode -output-directory docs docs/informe_mitex.tex
```

El PDF generado queda en `docs/informe_mitex.pdf`.

Los resultados incluidos en la documentacion salen de los artefactos de entrega
descomprimidos desde `MITEX_DATA.tar.gz`:

```bash
mitex eval-ir-file --index-dir data/indexes/default --queries data/reports/miracl_es_eval.jsonl --mode lexical --k 10 --limit 50 --output data/reports/doc_eval_lexical.json
mitex eval-ir-file --index-dir data/indexes/default --queries data/reports/miracl_es_eval.jsonl --mode semantic --k 10 --limit 50 --output data/reports/doc_eval_semantic.json
mitex eval-ir-file --index-dir data/indexes/default --queries data/reports/miracl_es_eval.jsonl --mode hybrid --k 10 --limit 50 --output data/reports/doc_eval_hybrid.json
mitex bench-turbovec --faiss-index-dir data/indexes/default --turbovec-index-dir data/indexes/default_turbovec --queries data/reports/miracl_es_eval.jsonl --limit 50 --k 10 --output data/reports/doc_turbovec_benchmark.json
```

El índice HNSW es opcional y no está incluido en el paquete de datos para no
duplicar varios GB. Se puede generar con:

```bash
mitex build-index --corpus data/processed/miracl_es_corpus_1M.jsonl --index-dir data/indexes/default_hnsw --index-type hnsw
python scripts/bench_ann_hnsw.py --limit 50
```
