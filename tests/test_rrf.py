from mitex_assistant.data.models import Chunk, RetrievalResult
from mitex_assistant.retrieval.hybrid import reciprocal_rank_fusion


def _result(chunk_id: str, rank: int, source: str) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        text=f"texto {chunk_id}",
        source_path="mem",
        start_char=0,
        end_char=10,
    )
    return RetrievalResult(chunk=chunk, score=1.0, source=source, rank=rank)


def test_rrf_rewards_documents_present_in_both_lists() -> None:
    semantic = [_result("a", 1, "semantic"), _result("b", 2, "semantic")]
    lexical = [_result("b", 1, "lexical"), _result("c", 2, "lexical")]

    fused = reciprocal_rank_fusion([semantic, lexical], top_k=3)

    # "b" aparece en ambas listas, debe quedar primero.
    assert fused[0].chunk.chunk_id == "b"
    assert {result.chunk.chunk_id for result in fused} == {"a", "b", "c"}
    assert [result.rank for result in fused] == [1, 2, 3]


def test_rrf_respects_top_k() -> None:
    semantic = [_result(chr(97 + i), i + 1, "semantic") for i in range(5)]
    fused = reciprocal_rank_fusion([semantic], top_k=2)
    assert len(fused) == 2
