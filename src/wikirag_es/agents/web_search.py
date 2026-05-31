from __future__ import annotations

from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from wikirag_es.utils.text import normalize_text, strip_prompt_injection

# Endpoint HTML de DuckDuckGo: no requiere API key, devuelve resultados
# organicos parseables. Es la opcion de "buscador web" del enunciado sin
# depender de credenciales de Google ni de librerias adicionales.
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass(frozen=True)
class WebResult:
    title: str
    snippet: str
    url: str


class WebSearchTool:
    """Busqueda web complementaria a la informacion local del corpus.

    Recupera titulos y fragmentos de DuckDuckGo, los normaliza y filtra
    posibles inyecciones de prompt antes de integrarlos en el contexto del LLM.
    Si la red no esta disponible devuelve un resultado vacio de forma controlada.
    """

    def __init__(self, max_results: int = 4, timeout: int = 10) -> None:
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str) -> list[WebResult]:
        try:
            response = requests.post(
                DUCKDUCKGO_HTML_URL,
                data={"q": query},
                headers={"User-Agent": USER_AGENT},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException:
            return []
        return self._parse(response.text)

    def _parse(self, html: str) -> list[WebResult]:
        soup = BeautifulSoup(html, "html.parser")
        results: list[WebResult] = []
        for node in soup.select("div.result")[: self.max_results * 2]:
            title_node = node.select_one("a.result__a")
            snippet_node = node.select_one(".result__snippet")
            if title_node is None:
                continue
            title = normalize_text(title_node.get_text(" "))
            snippet = normalize_text(snippet_node.get_text(" ")) if snippet_node else ""
            url = title_node.get("href", "")
            if not title:
                continue
            results.append(
                WebResult(
                    title=strip_prompt_injection(title),
                    snippet=strip_prompt_injection(snippet),
                    url=url,
                )
            )
            if len(results) >= self.max_results:
                break
        return results

    def run(self, query: str) -> tuple[str, list[str]]:
        """Devuelve (contexto, fuentes) listo para inyectar en el prompt."""
        results = self.search(query)
        if not results:
            return "", []
        blocks: list[str] = []
        sources: list[str] = []
        for rank, result in enumerate(results, start=1):
            blocks.append(
                f"[Web {rank}] {result.title}\n"
                f"URL: {result.url}\n"
                f"Resumen: {result.snippet}"
            )
            sources.append(f"[Web {rank}] {result.url or result.title}")
        return "\n\n".join(blocks), sources
