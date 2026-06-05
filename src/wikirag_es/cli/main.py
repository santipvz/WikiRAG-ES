from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

import click

from wikirag_es.config.settings import get_settings


@click.group()
def main() -> None:
    """CLI del asistente WikiRAG-ES."""


@main.command()
@click.option(
    "--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
def ingest_cmd(input_path: Path, output_path: Path) -> None:
    """Extrae, limpia y guarda documentos en JSONL."""
    from wikirag_es.ingestion.pipeline import ingest

    documents = ingest(input_path, output_path)
    click.echo(f"Documentos guardados: {len(documents)} en {output_path}")


main.add_command(ingest_cmd, name="ingest")


@main.command("acquire-corpus")
@click.option(
    "--corpus-output",
    type=click.Path(path_type=Path),
    default=Path("data/processed/miracl_es_corpus.jsonl"),
    show_default=True,
)
@click.option(
    "--eval-output",
    type=click.Path(path_type=Path),
    default=Path("data/reports/miracl_es_eval.jsonl"),
    show_default=True,
)
@click.option("--limit", type=int, default=100_000, show_default=True)
@click.option("--eval-split", type=click.Choice(["dev", "train"]), default="dev", show_default=True)
@click.option("--eval-limit", type=int, default=200, show_default=True)
@click.option("--max-scan", type=int, default=None, help="Maximo de documentos MIRACL a escanear.")
@click.option("--min-chars", type=int, default=80, show_default=True)
def acquire_corpus_cmd(
    corpus_output: Path,
    eval_output: Path,
    limit: int,
    eval_split: str,
    eval_limit: int,
    max_scan: int | None,
    min_chars: int,
) -> None:
    """Descarga MIRACL en espanol y genera corpus/evaluacion locales."""
    from wikirag_es.acquisition.miracl import acquire_miracl_spanish

    payload = acquire_miracl_spanish(
        corpus_output,
        eval_output,
        limit=limit,
        eval_split=eval_split,
        eval_limit=eval_limit,
        max_scan=max_scan,
        min_chars=min_chars,
    )
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))



@main.command("package-corpus")
@click.option("--corpus", "corpus_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--eval", "eval_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--topics-report", type=click.Path(exists=True, path_type=Path), default=None
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=Path("data/reports/miracl_es_corpus_package.tar.gz"),
    show_default=True,
)
def package_corpus_cmd(
    corpus_path: Path, eval_path: Path, topics_report: Path | None, output_path: Path
) -> None:
    """Crea el entregable comprimido del corpus local con manifest."""
    corpus_metadata = _infer_corpus_metadata(corpus_path)
    manifest = {
        **corpus_metadata,
        "corpus_file": corpus_path.name,
        "eval_file": eval_path.name,
        "topics_report": topics_report.name if topics_report else None,
        "corpus_documents": _count_jsonl_rows(corpus_path),
        "eval_queries": _count_jsonl_rows(eval_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        with tarfile.open(output_path, "w:gz") as archive:
            archive.add(corpus_path, arcname=corpus_path.name)
            archive.add(eval_path, arcname=eval_path.name)
            if topics_report is not None:
                archive.add(topics_report, arcname=topics_report.name)
            archive.add(manifest_path, arcname="manifest.json")
    click.echo(json.dumps({"output": str(output_path), **manifest}, ensure_ascii=False, indent=2))


@main.command("build-index")
@click.option(
    "--corpus", "corpus_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--index-dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--index-type",
    type=click.Choice(["flat", "hnsw"]),
    default=None,
    help="flat (exacto) o hnsw (ANN aproximado). Por defecto WIKIRAG_INDEX_TYPE.",
)
def build_index_cmd(corpus_path: Path, index_dir: Path, index_type: str | None) -> None:
    """Construye chunks, embeddings, FAISS y BM25."""
    from wikirag_es.indexing.pipeline import build_index

    settings = get_settings()
    total = build_index(corpus_path, index_dir, settings, index_type=index_type)
    click.echo(f"Indice construido con {total} chunks en {index_dir}")


@main.command("build-turbovec-index")
@click.option(
    "--corpus", "corpus_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--index-dir", type=click.Path(path_type=Path), required=True)
@click.option("--bit-width", type=click.Choice(["2", "4"]), default="4", show_default=True)
def build_turbovec_index_cmd(corpus_path: Path, index_dir: Path, bit_width: str) -> None:
    """Construye un indice TurboVec/TurboQuant opcional."""
    from wikirag_es.indexing.turbovec_pipeline import build_turbovec_index_from_corpus

    settings = get_settings()
    total = build_turbovec_index_from_corpus(
        corpus_path, index_dir, settings, bit_width=int(bit_width)
    )
    click.echo(f"Indice TurboVec construido con {total} chunks en {index_dir}")


@main.command("topics")
@click.option(
    "--corpus", "corpus_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--method", type=click.Choice(["auto", "bertopic", "nmf"]), default="auto", show_default=True
)
@click.option("--limit", type=int, default=None, help="Maximo de documentos para modelado.")
def topics_cmd(corpus_path: Path, output_path: Path, method: str, limit: int | None) -> None:
    """Ejecuta modelado de topicos offline."""
    from wikirag_es.topics.modeling import run_topic_modeling

    payload = run_topic_modeling(corpus_path, output_path, method=method, limit=limit)
    click.echo(
        f"Topicos guardados en {output_path}. Metodo: {payload['method']}. "
        f"Documentos: {payload['num_documents']}"
    )


@main.command("ask")
@click.option("--index-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--topics-report", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--turbovec", is_flag=True, default=False, help="Usar indice TurboVec en lugar de FAISS+BM25.")
@click.argument("query")
def ask_cmd(index_dir: Path, topics_report: Path | None, turbovec: bool, query: str) -> None:
    """Responde una consulta puntual."""
    from wikirag_es.agents.router import deterministic_route
    from wikirag_es.agents.tools import DateTool

    if deterministic_route(query) == "date":
        click.echo(DateTool().run())
        return
    agent = _build_agent(index_dir, topics_report=topics_report, is_turbovec=turbovec)
    response = agent.answer(query, on_token=_echo_stream_chunk)
    click.echo()
    _echo_sources(response.sources)


@main.command("chat")
@click.option("--index-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--topics-report", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--turbovec", is_flag=True, default=False, help="Usar indice TurboVec en lugar de FAISS+BM25.")
def chat_cmd(index_dir: Path, topics_report: Path | None, turbovec: bool) -> None:
    """Abre un chat interactivo con el asistente."""
    click.echo("Cargando modelo de embeddings e indice (puede tardar unos segundos)...")
    agent = _build_agent(index_dir, topics_report=topics_report, is_turbovec=turbovec)
    click.echo(
        "\nChat WikiRAG-ES listo. Escribe tu pregunta y pulsa Enter.\n"
        "Comandos: 'nuevo' reinicia la conversacion, 'salir' termina."
    )
    while True:
        query = click.prompt("Usuario", type=str)
        normalized = query.lower().strip()
        if normalized in {"salir", "exit", "quit"}:
            break
        if normalized in {"nuevo", "reset", "reiniciar"}:
            agent.reset()
            click.echo("\n[Memoria conversacional reiniciada]\n")
            continue

        def echo_action(action: str) -> None:
            click.echo(f"\nAsistente ({action}):\n", nl=False)

        response = agent.answer(query, on_token=_echo_stream_chunk, on_action=echo_action)
        click.echo()
        _echo_sources(response.sources)
        click.echo()


@main.command("eval-ir")
@click.option("--index-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--query", required=True)
@click.option("--relevant", multiple=True, help="IDs de chunks relevantes esperados.")
@click.option("--k", default=5, show_default=True)
def eval_ir_cmd(index_dir: Path, query: str, relevant: tuple[str, ...], k: int) -> None:
    """Evalua una consulta de recuperacion con chunk IDs relevantes."""
    from wikirag_es.eval.ir import evaluate_retrieval
    from wikirag_es.indexing.embeddings import EmbeddingModel
    from wikirag_es.indexing.vector_store import HybridIndex
    from wikirag_es.retrieval.hybrid import HybridRetriever

    settings = get_settings()
    embedding_model = EmbeddingModel(settings.embedding_model)
    retriever = HybridRetriever(HybridIndex.load(index_dir), embedding_model)
    results = retriever.retrieve(query, top_k=k)
    metrics = evaluate_retrieval(results, set(relevant), k)
    click.echo(json.dumps(metrics.__dict__, indent=2))


@main.command("eval-ir-file")
@click.option("--index-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--queries", "queries_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "semantic", "lexical"]),
    default="hybrid",
    show_default=True,
)
@click.option("--k", default=10, show_default=True)
@click.option("--limit", type=int, default=None)
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.option("--rerank", is_flag=True, default=False, help="Aplica el cross-encoder reranker.")
def eval_ir_file_cmd(
    index_dir: Path,
    queries_path: Path,
    mode: str,
    k: int,
    limit: int | None,
    output_path: Path | None,
    rerank: bool,
) -> None:
    """Evalua recuperacion con un fichero de consultas etiquetadas."""
    from wikirag_es.eval.batch import evaluate_ir_file

    settings = get_settings()
    embedding_model = None
    if mode in {"hybrid", "semantic"}:
        from wikirag_es.indexing.embeddings import EmbeddingModel

        embedding_model = EmbeddingModel(settings.embedding_model)
    reranker = None
    rerank_top_k = None
    if rerank:
        if mode != "hybrid":
            raise click.ClickException("--rerank solo es compatible con --mode hybrid.")
        from wikirag_es.retrieval.reranking import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings.reranker_model)
        rerank_top_k = k
    payload = evaluate_ir_file(
        index_dir,
        queries_path,
        embedding_model,
        mode,
        k,
        limit,
        output_path,
        reranker=reranker,
        rerank_top_k=rerank_top_k,
    )
    summary = {key: value for key, value in payload.items() if key != "per_query"}
    click.echo(json.dumps(summary, ensure_ascii=False, indent=2))


@main.command("bench-turbovec")
@click.option("--faiss-index-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--turbovec-index-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--queries", "queries_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--k", default=10, show_default=True)
@click.option("--limit", type=int, default=100, show_default=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None)
def bench_turbovec_cmd(
    faiss_index_dir: Path,
    turbovec_index_dir: Path,
    queries_path: Path,
    k: int,
    limit: int | None,
    output_path: Path | None,
) -> None:
    """Compara FAISS y TurboVec en busqueda semantica."""
    from wikirag_es.eval.turbovec_benchmark import benchmark_turbovec_against_faiss
    from wikirag_es.indexing.embeddings import EmbeddingModel

    settings = get_settings()
    embedding_model = EmbeddingModel(settings.embedding_model)
    payload = benchmark_turbovec_against_faiss(
        faiss_index_dir,
        turbovec_index_dir,
        queries_path,
        embedding_model,
        k=k,
        limit=limit,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_agent(index_dir: Path, topics_report: Path | None = None, is_turbovec: bool = False):
    from wikirag_es.agents.router import AssistantAgent
    from wikirag_es.agents.tools import (
        DateTool,
        RagTool,
        SummaryTool,
        TopicTool,
        WebSearchTool,
    )
    from wikirag_es.indexing.embeddings import EmbeddingModel
    from wikirag_es.indexing.vector_store import HybridIndex
    from wikirag_es.llm.factory import create_llm
    from wikirag_es.retrieval.hybrid import HybridRetriever

    settings = get_settings()
    embedding_model = EmbeddingModel(settings.embedding_model)
    
    if is_turbovec:
        from wikirag_es.indexing.turbovec_store import TurboVecIndex
        index = TurboVecIndex.load(index_dir, load_chunk_text=True)
    else:
        index = HybridIndex.load(index_dir, load_embeddings=False, load_faiss=True, load_bm25=True)
    reranker = None
    if settings.use_reranker:
        from wikirag_es.retrieval.reranking import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings.reranker_model)
    retriever = HybridRetriever(
        index,
        embedding_model,
        reranker=reranker,
        rerank_top_k=settings.rerank_top_k if reranker else None,
    )
    llm = create_llm(settings)
    return AssistantAgent(
        llm=llm,
        rag_tool=RagTool(retriever, settings.top_k),
        summary_tool=SummaryTool(embedding_model),
        topic_tool=TopicTool(topics_report) if topics_report is not None else None,
        date_tool=DateTool(),
        web_tool=WebSearchTool(settings.web_search_results, settings.web_search_timeout),
        use_llm_router=settings.use_llm_router,
    )


def _echo_stream_chunk(chunk: str) -> None:
    click.echo(chunk, nl=False)
    click.get_text_stream("stdout").flush()


def _echo_sources(sources: list[str]) -> None:
    if not sources:
        return
    click.echo("\nFuentes:")
    for source in sources:
        click.echo(f"- {source}")


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def _infer_corpus_metadata(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            return {
                "dataset": metadata.get("dataset"),
                "dataset_config": metadata.get("dataset_config"),
                "topics_dataset": metadata.get("topics_dataset"),
                "category": metadata.get("category"),
                "language": metadata.get("language"),
                "license": metadata.get("license"),
                "source": metadata.get("source"),
            }
    return {
        "dataset": None,
        "dataset_config": None,
        "topics_dataset": None,
        "category": None,
        "language": None,
        "license": None,
        "source": None,
    }


if __name__ == "__main__":
    main()
