from __future__ import annotations

import json
from pathlib import Path

import click

from mitex_assistant.agents.router import AssistantAgent
from mitex_assistant.agents.tools import RagTool, SummaryTool
from mitex_assistant.config.settings import get_settings
from mitex_assistant.eval.ir import evaluate_retrieval
from mitex_assistant.indexing.embeddings import EmbeddingModel
from mitex_assistant.indexing.pipeline import build_index
from mitex_assistant.indexing.vector_store import HybridIndex
from mitex_assistant.ingestion.pipeline import ingest
from mitex_assistant.llm.factory import create_llm
from mitex_assistant.retrieval.hybrid import HybridRetriever
from mitex_assistant.topics.modeling import run_topic_modeling


@click.group()
def main() -> None:
    """CLI del asistente MITEX."""


@main.command()
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
def ingest_cmd(input_path: Path, output_path: Path) -> None:
    """Extrae, limpia y guarda documentos en JSONL."""
    documents = ingest(input_path, output_path)
    click.echo(f"Documentos guardados: {len(documents)} en {output_path}")


main.add_command(ingest_cmd, name="ingest")


@main.command("build-index")
@click.option("--corpus", "corpus_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--index-dir", type=click.Path(path_type=Path), required=True)
def build_index_cmd(corpus_path: Path, index_dir: Path) -> None:
    """Construye chunks, embeddings, FAISS y BM25."""
    settings = get_settings()
    total = build_index(corpus_path, index_dir, settings)
    click.echo(f"Indice construido con {total} chunks en {index_dir}")


@main.command("topics")
@click.option("--corpus", "corpus_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
def topics_cmd(corpus_path: Path, output_path: Path) -> None:
    """Ejecuta modelado de topicos offline."""
    payload = run_topic_modeling(corpus_path, output_path)
    click.echo(f"Topicos guardados en {output_path}. Documentos: {payload['num_documents']}")


@main.command("ask")
@click.option("--index-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.argument("query")
def ask_cmd(index_dir: Path, query: str) -> None:
    """Responde una consulta puntual."""
    agent = _build_agent(index_dir)
    response = agent.answer(query)
    click.echo(response.answer)
    if response.sources:
        click.echo("\nFuentes:")
        for source in response.sources:
            click.echo(f"- {source}")


@main.command("chat")
@click.option("--index-dir", type=click.Path(exists=True, path_type=Path), required=True)
def chat_cmd(index_dir: Path) -> None:
    """Abre un chat interactivo con el asistente."""
    agent = _build_agent(index_dir)
    click.echo("Chat MITEX. Escribe 'salir' para terminar.")
    while True:
        query = click.prompt("Usuario", type=str)
        if query.lower().strip() in {"salir", "exit", "quit"}:
            break
        response = agent.answer(query)
        click.echo(f"\nAsistente ({response.action}):\n{response.answer}\n")


@main.command("eval-ir")
@click.option("--index-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--query", required=True)
@click.option("--relevant", multiple=True, help="IDs de chunks relevantes esperados.")
@click.option("--k", default=5, show_default=True)
def eval_ir_cmd(index_dir: Path, query: str, relevant: tuple[str, ...], k: int) -> None:
    """Evalua una consulta de recuperacion con chunk IDs relevantes."""
    settings = get_settings()
    embedding_model = EmbeddingModel(settings.embedding_model)
    retriever = HybridRetriever(HybridIndex.load(index_dir), embedding_model)
    results = retriever.retrieve(query, top_k=k)
    metrics = evaluate_retrieval(results, set(relevant), k)
    click.echo(json.dumps(metrics.__dict__, indent=2))


def _build_agent(index_dir: Path) -> AssistantAgent:
    settings = get_settings()
    embedding_model = EmbeddingModel(settings.embedding_model)
    index = HybridIndex.load(index_dir)
    retriever = HybridRetriever(index, embedding_model)
    llm = create_llm(settings)
    return AssistantAgent(
        llm=llm,
        rag_tool=RagTool(retriever, settings.top_k),
        summary_tool=SummaryTool(embedding_model),
    )


if __name__ == "__main__":
    main()
