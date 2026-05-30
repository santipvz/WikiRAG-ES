from mitex_assistant.eval.ir import average_metrics, evaluate_ids


def test_evaluate_ids_computes_precision_recall_and_mrr() -> None:
    metrics = evaluate_ids(["a", "b", "c"], {"b", "d"}, k=3)

    assert metrics.precision_at_k == 1 / 3
    assert metrics.recall_at_k == 1 / 2
    assert metrics.mrr == 1 / 2


def test_average_metrics_handles_empty_input() -> None:
    metrics = average_metrics([])

    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k == 0.0
    assert metrics.mrr == 0.0
