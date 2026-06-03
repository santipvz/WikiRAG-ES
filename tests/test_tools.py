import json

from wikirag_es.agents.tools import DateTool, MONTHS_ES, TopicTool, WEEKDAYS_ES
from wikirag_es.agents.web_search import WebSearchTool
from wikirag_es.summarization.extractive import split_sentences, textrank_summary


class FakeEmbeddingModel:
    def encode(self, texts: list[str], batch_size: int = 32):
        import numpy as np

        return np.ones((len(texts), 3), dtype="float32")


def test_date_tool_returns_spanish_format() -> None:
    output = DateTool().run()
    assert output.startswith("Hoy es ")
    assert any(day in output for day in WEEKDAYS_ES)
    assert any(month in output for month in MONTHS_ES)


def test_web_search_parses_results_html() -> None:
    html = """
    <div class="result">
      <a class="result__a" href="https://ejemplo.com/a">Titulo A</a>
      <div class="result__snippet">Fragmento A relevante.</div>
    </div>
    <div class="result">
      <a class="result__a" href="https://ejemplo.com/b">Titulo B</a>
      <div class="result__snippet">Fragmento B relevante.</div>
    </div>
    """
    tool = WebSearchTool(max_results=2)
    results = tool._parse(html)
    assert len(results) == 2
    assert results[0].title == "Titulo A"
    assert results[0].url == "https://ejemplo.com/a"
    assert "Fragmento A" in results[0].snippet


def test_web_search_parse_empty_html() -> None:
    tool = WebSearchTool()
    assert tool._parse("<html></html>") == []


def test_topic_tool_reports_real_cluster_count(tmp_path) -> None:
    report = tmp_path / "topics.json"
    report.write_text(
        json.dumps(
            {
                "method": "bertopic",
                "num_documents": 12,
                "topic_info": [
                    {"Topic": -1, "Count": 4, "Representation": ["outlier"]},
                    {"Topic": 0, "Count": 5, "Representation": ["historia"]},
                    {"Topic": 1, "Count": 3, "Representation": ["ciencia"]},
                    {"Topic": 2, "Count": 8, "Representation": ["transporte"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    output = TopicTool(report).run(max_topics=1)

    assert "Clusters detectados: 3" in output
    assert "4 documentos outlier" in output
    assert "Topico 2" in output
    assert "Topico 0" not in output
    assert "Topico -1" not in output


def test_split_sentences_filters_short_fragments() -> None:
    text = "Hola. Esta es una frase suficientemente larga para contar. Si."
    sentences = split_sentences(text)
    assert "Esta es una frase suficientemente larga para contar." in sentences
    assert "Si." not in sentences  # se descartan fragmentos muy cortos


def test_textrank_summary_breaks_ties_by_sentence_order() -> None:
    text = (
        "Primera frase suficientemente larga para entrar en el resumen. "
        "Segunda frase suficientemente larga para entrar en el resumen. "
        "Tercera frase suficientemente larga para entrar en el resumen. "
        "Cuarta frase suficientemente larga para entrar en el resumen."
    )

    summary = textrank_summary(text, FakeEmbeddingModel(), sentences=2)

    assert summary == [
        "Primera frase suficientemente larga para entrar en el resumen.",
        "Segunda frase suficientemente larga para entrar en el resumen.",
    ]
