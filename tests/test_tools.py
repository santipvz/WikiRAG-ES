from wikirag_es.agents.tools import DateTool, MONTHS_ES, WEEKDAYS_ES
from wikirag_es.agents.web_search import WebSearchTool
from wikirag_es.summarization.extractive import split_sentences


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


def test_split_sentences_filters_short_fragments() -> None:
    text = "Hola. Esta es una frase suficientemente larga para contar. Si."
    sentences = split_sentences(text)
    assert "Esta es una frase suficientemente larga para contar." in sentences
    assert "Si." not in sentences  # se descartan fragmentos muy cortos
