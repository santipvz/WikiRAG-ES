from __future__ import annotations

import json
import gzip
from pathlib import Path
from typing import Iterator

import requests

from wikirag_es.data.jsonl import write_jsonl
from wikirag_es.data.models import Document
from wikirag_es.utils.text import normalize_text, strip_prompt_injection


CORPUS_DATASET = "miracl/miracl-corpus"
TOPICS_DATASET = "miracl/miracl"
LANGUAGE = "es"
LICENSE = "apache-2.0"
AVAILABLE_DOCUMENTS = 10_400_000
TOPICS_URL = (
    "https://huggingface.co/datasets/miracl/miracl/resolve/main/"
    "miracl-v1.0-es/topics/topics.miracl-v1.0-es-{split}.tsv"
)
QRELS_URL = (
    "https://huggingface.co/datasets/miracl/miracl/resolve/main/"
    "miracl-v1.0-es/qrels/qrels.miracl-v1.0-es-{split}.tsv"
)
CORPUS_TREE_URL = (
    "https://huggingface.co/api/datasets/miracl/miracl-corpus/tree/main/"
    "miracl-corpus-v1.0-es"
)
CORPUS_FILE_URL = "https://huggingface.co/datasets/miracl/miracl-corpus/resolve/main/{path}"


def acquire_miracl_spanish(
    corpus_path: Path,
    eval_path: Path,
    limit: int = 100_000,
    eval_split: str = "dev",
    eval_limit: int = 200,
    max_scan: int | None = None,
    min_chars: int = 80,
) -> dict[str, int | str]:
    """Stream MIRACL Spanish into the local corpus/evaluation JSONL format.

    MIRACL is a single retrieval benchmark. The local RAG corpus is built from
    ``miracl/miracl-corpus`` and the evaluation queries/qrels from ``miracl/miracl``.
    Both reference the same Spanish Wikipedia collection.
    """
    if eval_split not in {"dev", "train"}:
        raise ValueError("eval_split must be one of: dev, train")
    if limit <= 0:
        raise ValueError("limit must be positive")

    topics = _load_topics(eval_split)
    qrels = _load_qrels(eval_split)
    wanted_eval_doc_ids = {doc_id for doc_ids in qrels.values() for doc_id in doc_ids}
    scan_budget = max_scan if max_scan is not None else max(limit, limit * 5)

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    included_doc_ids: set[str] = set()
    documents = 0
    scanned = 0

    with corpus_path.open("w", encoding="utf-8") as file:
        for scanned, row in enumerate(_iter_corpus_rows(), start=1):
            doc_id = str(row.get("docid", "")).strip()
            if not doc_id:
                continue
            title = normalize_text(str(row.get("title", "")))
            text = strip_prompt_injection(normalize_text(str(row.get("text", ""))))
            joined = normalize_text(f"{title}. {text}" if title else text)
            if len(joined) < min_chars:
                continue

            keep_regular_document = documents < limit
            keep_eval_document = doc_id in wanted_eval_doc_ids and _eval_rows_count(
                qrels, included_doc_ids, limit=eval_limit
            ) < eval_limit
            if not keep_regular_document and not keep_eval_document:
                if scanned >= scan_budget:
                    break
                continue

            document = Document(
                doc_id=doc_id,
                text=joined,
                source_path=f"hf://{CORPUS_DATASET}/{LANGUAGE}/{doc_id}",
                title=title or doc_id,
                metadata={
                    "dataset": CORPUS_DATASET,
                    "topics_dataset": TOPICS_DATASET,
                    "language": LANGUAGE,
                    "license": LICENSE,
                    "miracl_docid": doc_id,
                },
            )
            file.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")
            included_doc_ids.add(doc_id)
            documents += 1

            if documents >= limit and _eval_rows_count(qrels, included_doc_ids, eval_limit) >= eval_limit:
                break
            if scanned >= scan_budget and documents >= limit:
                break

    eval_rows = _build_eval_rows(topics, qrels, included_doc_ids, limit=eval_limit)
    write_jsonl(eval_path, eval_rows)
    return {
        "dataset": CORPUS_DATASET,
        "topics_dataset": TOPICS_DATASET,
        "language": LANGUAGE,
        "license": LICENSE,
        "available_documents": AVAILABLE_DOCUMENTS,
        "scanned_documents": scanned,
        "local_documents": documents,
        "eval_queries": len(eval_rows),
        "corpus_output": str(corpus_path),
        "eval_output": str(eval_path),
    }


def _load_topics(split: str) -> dict[str, str]:
    response = requests.get(TOPICS_URL.format(split=split), timeout=120)
    response.raise_for_status()
    topics: dict[str, str] = {}
    for line in response.text.splitlines():
        parts = line.rstrip("\n").split("\t", maxsplit=1)
        if len(parts) == 2:
            topics[parts[0]] = normalize_text(parts[1])
    return topics


def _iter_corpus_rows() -> Iterator[dict]:
    for path in _list_corpus_files():
        url = CORPUS_FILE_URL.format(path=path)
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            with gzip.GzipFile(fileobj=response.raw) as compressed:
                for raw_line in compressed:
                    line = raw_line.decode("utf-8").strip()
                    if line:
                        yield json.loads(line)


def _list_corpus_files() -> list[str]:
    response = requests.get(CORPUS_TREE_URL, timeout=120)
    response.raise_for_status()
    files = [row["path"] for row in response.json() if row.get("path", "").endswith(".jsonl.gz")]
    return sorted(files, key=_corpus_file_order)


def _corpus_file_order(path: str) -> int:
    filename = path.rsplit("/", maxsplit=1)[-1]
    number = filename.removeprefix("docs-").removesuffix(".jsonl.gz")
    return int(number)


def _load_qrels(split: str) -> dict[str, set[str]]:
    response = requests.get(QRELS_URL.format(split=split), timeout=120)
    response.raise_for_status()
    qrels: dict[str, set[str]] = {}
    for line in response.text.splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        query_id, doc_id, relevance = parts[0], parts[2], int(parts[3])
        if relevance > 0:
            qrels.setdefault(query_id, set()).add(doc_id)
    return qrels


def _build_eval_rows(
    topics: dict[str, str], qrels: dict[str, set[str]], included_doc_ids: set[str], limit: int
) -> list[dict]:
    if limit <= 0:
        return []
    rows: list[dict] = []
    for query_id, relevant_doc_ids in qrels.items():
        available_relevant_ids = sorted(relevant_doc_ids & included_doc_ids)
        query = topics.get(query_id)
        if query and available_relevant_ids:
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "relevant_doc_ids": available_relevant_ids,
                }
            )
            if len(rows) >= limit:
                break
    return rows


def _eval_rows_count(qrels: dict[str, set[str]], included_doc_ids: set[str], limit: int) -> int:
    if limit <= 0:
        return 0
    count = 0
    for relevant_doc_ids in qrels.values():
        if relevant_doc_ids & included_doc_ids:
            count += 1
            if count >= limit:
                break
    return count
