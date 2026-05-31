from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from mitex_assistant.data.models import Chunk, RetrievalResult
from mitex_assistant.utils.text import simple_tokenize


def build_faiss_index(
    embeddings: np.ndarray, index_type: str = "flat", hnsw_m: int = 32
) -> "faiss.Index":
    """Construye el indice FAISS sobre vectores normalizados (producto interno).

    - ``flat``: busqueda exacta (IndexFlatIP), exhaustiva y O(N) por consulta.
    - ``hnsw``: grafo HNSW aproximado (ANN), latencia sublineal a gran escala.
    """
    dimension = embeddings.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
    else:
        raise ValueError("index_type must be one of: flat, hnsw")
    index.add(embeddings)
    return index


class HybridIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        index_type: str = "flat",
        hnsw_m: int = 32,
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings.astype("float32")
        self.faiss_index = build_faiss_index(self.embeddings, index_type, hnsw_m)
        self.bm25 = BM25Okapi([simple_tokenize(chunk.text) for chunk in chunks])

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(index_dir / "faiss.index"))
        np.save(index_dir / "embeddings.npy", self.embeddings)
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as file:
            for chunk in self.chunks:
                file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        with (index_dir / "bm25.pkl").open("wb") as file:
            pickle.dump(self.bm25, file)

    @classmethod
    def load(
        cls,
        index_dir: Path,
        *,
        load_chunk_text: bool = True,
        load_embeddings: bool = False,
        load_faiss: bool = True,
        load_bm25: bool = True,
    ) -> "HybridIndex":
        chunks: list[Chunk] = []
        with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                if not load_chunk_text:
                    row = {
                        **row,
                        "text": "",
                        "source_path": row.get("source_path", ""),
                        "metadata": {},
                    }
                chunks.append(Chunk(**row))
        instance = cls.__new__(cls)
        instance.chunks = chunks
        instance.embeddings = (
            np.load(index_dir / "embeddings.npy").astype("float32") if load_embeddings else None
        )
        instance.faiss_index = (
            faiss.read_index(str(index_dir / "faiss.index")) if load_faiss else None
        )
        if load_bm25:
            with (index_dir / "bm25.pkl").open("rb") as file:
                instance.bm25 = pickle.load(file)
        else:
            instance.bm25 = None
        return instance

    def semantic_search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievalResult]:
        if self.faiss_index is None:
            raise RuntimeError("FAISS index was not loaded")
        scores, indexes = self.faiss_index.search(query_embedding.astype("float32"), top_k)
        results: list[RetrievalResult] = []
        for rank, (idx, score) in enumerate(zip(indexes[0], scores[0]), start=1):
            if idx == -1:
                continue
            results.append(
                RetrievalResult(
                    chunk=self.chunks[int(idx)], score=float(score), source="semantic", rank=rank
                )
            )
        return results

    def lexical_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index was not loaded")
        scores = self.bm25.get_scores(simple_tokenize(query))
        if top_k >= len(scores):
            ranked = np.argsort(scores)[::-1]
        else:
            candidates = np.argpartition(scores, -top_k)[-top_k:]
            ranked = candidates[np.argsort(scores[candidates])[::-1]]
        return [
            RetrievalResult(
                chunk=self.chunks[int(idx)], score=float(scores[idx]), source="lexical", rank=rank
            )
            for rank, idx in enumerate(ranked, start=1)
            if scores[idx] > 0
        ]


def build_hybrid_index(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    index_dir: Path,
    index_type: str = "flat",
    hnsw_m: int = 32,
) -> HybridIndex:
    index = HybridIndex(chunks, embeddings, index_type=index_type, hnsw_m=hnsw_m)
    index.save(index_dir)
    return index
