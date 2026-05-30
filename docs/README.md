# Documentacion MITEX

Compilar el informe final:

```bash
pdflatex -interaction=nonstopmode -output-directory docs docs/informe_mitex.tex
pdflatex -interaction=nonstopmode -output-directory docs docs/informe_mitex.tex
```

El PDF generado queda en `docs/informe_mitex.pdf`.

Los resultados incluidos en la documentacion salen de:

```bash
mitex eval-ir-file --index-dir data/indexes/default --queries data/reports/miracl_es_eval.jsonl --mode lexical --k 10 --limit 50 --output data/reports/doc_eval_lexical.json
mitex eval-ir-file --index-dir data/indexes/default --queries data/reports/miracl_es_eval.jsonl --mode semantic --k 10 --limit 50 --output data/reports/doc_eval_semantic.json
mitex eval-ir-file --index-dir data/indexes/default --queries data/reports/miracl_es_eval.jsonl --mode hybrid --k 10 --limit 50 --output data/reports/doc_eval_hybrid.json
mitex bench-turbovec --faiss-index-dir data/indexes/default --turbovec-index-dir data/indexes/default_turbovec --queries data/reports/miracl_es_eval.jsonl --limit 50 --k 10 --output data/reports/doc_turbovec_benchmark.json
```
