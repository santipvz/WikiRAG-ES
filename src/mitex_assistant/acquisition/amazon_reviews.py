from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import requests

from mitex_assistant.data.jsonl import write_jsonl
from mitex_assistant.data.models import Document
from mitex_assistant.utils.text import normalize_text, strip_prompt_injection


DATASET = "McAuley-Lab/Amazon-Reviews-2023"
LICENSE = "research dataset; see dataset card and original terms"
REVIEW_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/review_categories/{category}.jsonl"
)
CATEGORY_STATS = {
    "Video_Games": {"reviews": 4_600_000, "review_tokens": "347.9M"},
    "Software": {"reviews": 4_900_000, "review_tokens": "179.4M"},
    "Movies_and_TV": {"reviews": 17_300_000, "review_tokens": "1.0B"},
    "Books": {"reviews": 29_500_000, "review_tokens": "2.9B"},
    "Electronics": {"reviews": 43_900_000, "review_tokens": "2.7B"},
}


def acquire_amazon_reviews(
    corpus_path: Path,
    eval_path: Path | None = None,
    category: str = "Video_Games",
    limit: int = 100_000,
    eval_limit: int = 200,
    min_chars: int = 40,
) -> dict[str, int | str | None]:
    """Stream Amazon Reviews 2023 into the project JSONL document format."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if eval_limit < 0:
        raise ValueError("eval_limit cannot be negative")

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    eval_rows: list[dict] = []
    documents = 0
    scanned = 0

    with corpus_path.open("w", encoding="utf-8") as file:
        for scanned, row in enumerate(_iter_review_rows(category), start=1):
            title = normalize_text(str(row.get("title", "")))
            review = strip_prompt_injection(normalize_text(str(row.get("text", ""))))
            if not review or len(review) < min_chars:
                continue

            asin = str(row.get("asin") or row.get("parent_asin") or "unknown").strip()
            parent_asin = str(row.get("parent_asin") or asin).strip()
            rating = row.get("rating")
            helpful_vote = row.get("helpful_vote")
            verified_purchase = row.get("verified_purchase")
            doc_id = f"amazon-{category}-{scanned}"
            text_parts = [f"Categoria: {category}."]
            if title:
                text_parts.append(f"Titulo de la resena: {title}.")
            if rating is not None:
                text_parts.append(f"Puntuacion: {rating}/5.")
            text_parts.append(f"Resena: {review}")
            text = normalize_text(" ".join(text_parts))

            document = Document(
                doc_id=doc_id,
                text=text,
                source_path=f"hf://{DATASET}/{category}/{doc_id}",
                title=title or f"Review {asin}",
                metadata={
                    "dataset": DATASET,
                    "category": category,
                    "license": LICENSE,
                    "asin": asin,
                    "parent_asin": parent_asin,
                    "rating": rating,
                    "helpful_vote": helpful_vote,
                    "verified_purchase": verified_purchase,
                },
            )
            file.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")
            documents += 1

            if eval_path is not None and len(eval_rows) < eval_limit and title:
                eval_rows.append(
                    {
                        "query_id": f"amazon-{category}-{scanned}",
                        "query": f"Opiniones sobre: {title}",
                        "relevant_doc_ids": [doc_id],
                    }
                )
            if documents >= limit:
                break

    if eval_path is not None:
        write_jsonl(eval_path, eval_rows)

    stats = CATEGORY_STATS.get(category, {})
    return {
        "dataset": DATASET,
        "category": category,
        "license": LICENSE,
        "available_reviews_estimate": stats.get("reviews"),
        "review_tokens_estimate": stats.get("review_tokens"),
        "scanned_reviews": scanned,
        "local_documents": documents,
        "eval_queries": len(eval_rows) if eval_path is not None else 0,
        "corpus_output": str(corpus_path),
        "eval_output": str(eval_path) if eval_path is not None else None,
    }


def _iter_review_rows(category: str) -> Iterator[dict]:
    url = REVIEW_URL.format(category=category)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        for raw_line in response.iter_lines(decode_unicode=True):
            line = raw_line.strip() if raw_line else ""
            if line:
                yield json.loads(line)
