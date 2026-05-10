from __future__ import annotations

import json
from pathlib import Path

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from mitex_assistant.data.jsonl import read_jsonl


SPANISH_STOPWORDS = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para",
    "con", "no", "una", "su", "al", "lo", "es", "como", "mas", "pero", "sus", "le", "ya",
]


def run_topic_modeling(corpus_path: Path, output_path: Path) -> dict:
    docs = [row["text"] for row in read_jsonl(corpus_path)]
    if len(docs) < 5:
        raise ValueError("BERTopic needs at least a few documents to produce meaningful topics")

    umap_model = UMAP(n_neighbors=min(15, max(2, len(docs) - 1)), n_components=5, min_dist=0.0, metric="cosine")
    hdbscan_model = HDBSCAN(min_cluster_size=max(2, min(15, len(docs) // 4)), min_samples=1, prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words=SPANISH_STOPWORDS, ngram_range=(1, 2))
    model = BERTopic(
        language="spanish",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(docs)
    topic_info = model.get_topic_info().to_dict(orient="records")
    payload = {"num_documents": len(docs), "topics": topics, "topic_info": topic_info}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
