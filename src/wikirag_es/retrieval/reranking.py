from __future__ import annotations

from wikirag_es.data.models import RetrievalResult


class CrossEncoderReranker:
    """Reordena los candidatos fusionados con un cross-encoder multilingue.

    A diferencia del bi-encoder usado para la busqueda densa, el cross-encoder
    procesa la pareja (consulta, fragmento) conjuntamente, lo que suele mejorar
    Precision@k y MRR a costa de mayor latencia. Por eso se aplica solo sobre el
    top-k ya fusionado por RRF.
    """

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not results:
            return []
        pairs = [(query, result.chunk.text) for result in results]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        order = sorted(range(len(results)), key=lambda i: float(scores[i]), reverse=True)
        reranked: list[RetrievalResult] = []
        for rank, index in enumerate(order[:top_k], start=1):
            base = results[index]
            reranked.append(
                RetrievalResult(
                    chunk=base.chunk,
                    score=float(scores[index]),
                    source="rerank",
                    rank=rank,
                )
            )
        return reranked
