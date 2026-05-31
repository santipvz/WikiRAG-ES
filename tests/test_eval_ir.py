import wikirag_es.eval.batch as batch
from wikirag_es.eval.ir import average_metrics, evaluate_ids


def test_evaluate_ids_computes_precision_recall_and_mrr() -> None:
    metrics = evaluate_ids(["a", "b", "c"], {"b", "d"}, k=3)

    assert metrics.precision_at_k == 1 / 3
    assert metrics.recall_at_k == 1 / 2
    assert metrics.mrr == 1 / 2


def test_evaluate_ids_precision_uses_k_denominator() -> None:
    metrics = evaluate_ids(["a"], {"a", "b"}, k=3)

    assert metrics.precision_at_k == 1 / 3
    assert metrics.recall_at_k == 1 / 2


def test_evaluate_ids_rejects_non_positive_k() -> None:
    try:
        evaluate_ids(["a"], {"a"}, k=0)
    except ValueError:
        return
    raise AssertionError("Se esperaba ValueError cuando k <= 0")


def test_average_metrics_handles_empty_input() -> None:
    metrics = average_metrics([])

    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k == 0.0
    assert metrics.mrr == 0.0


def test_evaluate_ir_file_loads_chunk_text_when_reranking(monkeypatch, tmp_path) -> None:
    load_kwargs = {}

    def fake_load(index_dir, **kwargs):
        load_kwargs.update(kwargs)
        return object()

    class FakeRetriever:
        def __init__(self, index, embedding_model, reranker=None, rerank_top_k=None) -> None:
            self.reranker = reranker

        def retrieve(self, query: str, top_k: int):
            return []

    monkeypatch.setattr(batch.HybridIndex, "load", fake_load)
    monkeypatch.setattr(batch, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(
        batch,
        "read_jsonl",
        lambda path: iter([{"query": "consulta", "relevant_doc_ids": []}]),
    )

    batch.evaluate_ir_file(
        tmp_path,
        tmp_path / "queries.jsonl",
        embedding_model=object(),
        mode="hybrid",
        k=10,
        reranker=object(),
    )

    assert load_kwargs["load_chunk_text"] is True
