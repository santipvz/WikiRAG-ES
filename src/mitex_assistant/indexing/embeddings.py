from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")
