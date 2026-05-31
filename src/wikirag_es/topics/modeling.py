from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF

from wikirag_es.data.jsonl import read_jsonl


SPANISH_STOPWORDS = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para",
    "con", "no", "una", "su", "al", "lo", "es", "como", "mas", "pero", "sus", "le", "ya",
]


def run_topic_modeling(
    corpus_path: Path, output_path: Path, method: str = "auto", limit: int | None = None
) -> dict:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    docs: list[str] = []
    for row_number, row in enumerate(read_jsonl(corpus_path), start=1):
        if limit is not None and row_number > limit:
            break
        docs.append(row["text"])
    if len(docs) < 2:
        raise ValueError("Topic modeling needs at least two documents")

    if method not in {"auto", "bertopic", "nmf"}:
        raise ValueError("method must be one of: auto, bertopic, nmf")

    if method in {"auto", "bertopic"}:
        try:
            payload = _run_bertopic(docs)
        except Exception:
            if method == "bertopic":
                raise
            payload = _run_nmf(docs)
    else:
        payload = _run_nmf(docs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _run_bertopic(docs: list[str]) -> dict:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

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
    return {"method": "bertopic", "num_documents": len(docs), "topics": topics, "topic_info": topic_info}


def _run_nmf(docs: list[str]) -> dict:
    vectorizer = CountVectorizer(stop_words=SPANISH_STOPWORDS, ngram_range=(1, 2), min_df=1)
    document_term = vectorizer.fit_transform(docs)
    n_topics = min(5, max(2, len(docs) // 2), document_term.shape[1])
    model = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=500)
    weights = model.fit_transform(document_term)
    terms = vectorizer.get_feature_names_out()
    topics = weights.argmax(axis=1).astype(int).tolist()
    topic_info = []
    for topic_id, components in enumerate(model.components_):
        top_indexes = components.argsort()[::-1][:10]
        topic_info.append(
            {
                "Topic": topic_id,
                "Count": topics.count(topic_id),
                "Name": ", ".join(terms[index] for index in top_indexes[:5]),
                "Representation": [terms[index] for index in top_indexes],
            }
        )
    return {"method": "nmf", "num_documents": len(docs), "topics": topics, "topic_info": topic_info}
