from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mitex_assistant.data.models import Chunk, RetrievalResult


TURBOVEC_INDEX_FILE = "turbovec.tvim"


class TurboVecIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, bit_width: int = 4) -> None:
        try:
            from turbovec import IdMapIndex
        except ImportError as exc:
            raise ImportError(
                "Install optional TurboVec dependencies with: pip install -e .[turbovec]"
            ) from exc

        self.chunks = chunks
        self.bit_width = bit_width
        vectors = np.ascontiguousarray(np.asarray(embeddings, dtype="float32"))
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        self.index = IdMapIndex(dim=vectors.shape[1], bit_width=bit_width)
        ids = np.ascontiguousarray(np.arange(1, len(chunks) + 1, dtype=np.uint64))
        self.index.add_with_ids(vectors, ids)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        self.index.write(str(index_dir / TURBOVEC_INDEX_FILE))
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as file:
            for chunk in self.chunks:
                file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        metadata = {"bit_width": self.bit_width, "vectors": len(self.chunks)}
        (index_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path, *, load_chunk_text: bool = True) -> "TurboVecIndex":
        try:
            from turbovec import IdMapIndex
        except ImportError as exc:
            raise ImportError(
                "Install optional TurboVec dependencies with: pip install -e .[turbovec]"
            ) from exc

        chunks: list[Chunk] = []
        with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    row = json.loads(line)
                    if not load_chunk_text:
                        row = {
                            **row,
                            "text": "",
                            "source_path": row.get("source_path", ""),
                            "metadata": {},
                        }
                    chunks.append(Chunk(**row))
        metadata_path = index_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        instance = cls.__new__(cls)
        instance.chunks = chunks
        instance.bit_width = int(metadata.get("bit_width", 4))
        instance.index = IdMapIndex.load(str(index_dir / TURBOVEC_INDEX_FILE))
        return instance

    def semantic_search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievalResult]:
        query = np.asarray(query_embedding, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = np.ascontiguousarray(query)
        scores, ids = self.index.search(query, k=top_k)
        scores = np.asarray(scores).reshape(-1)
        ids = np.asarray(ids).reshape(-1)
        results: list[RetrievalResult] = []
        for rank, (score, external_id) in enumerate(zip(scores, ids), start=1):
            chunk_index = int(external_id) - 1
            if chunk_index < 0 or chunk_index >= len(self.chunks):
                continue
            results.append(
                RetrievalResult(
                    chunk=self.chunks[chunk_index],
                    score=float(score),
                    source="turbovec",
                    rank=rank,
                )
            )
        return results


def build_turbovec_index(
    chunks: list[Chunk], embeddings: np.ndarray, index_dir: Path, bit_width: int = 4
) -> TurboVecIndex:
    index = TurboVecIndex(chunks, embeddings, bit_width=bit_width)
    index.save(index_dir)
    return index
