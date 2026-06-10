from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from wikirag_es.agents.prompts import (
    RAG_PROMPT_TEMPLATE,
    REWRITE_PROMPT_TEMPLATE,
    ROUTER_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    WEB_PROMPT_TEMPLATE,
)
from wikirag_es.agents.tools import DateTool, RagTool, SummaryTool, TopicTool, WebSearchTool
from wikirag_es.llm.base import LLMClient

VALID_ACTIONS = {"rag", "web", "summary", "rag_topics", "date"}
HISTORY_TURNS = 4


def deterministic_route(query: str) -> str | None:
    lowered = query.lower()
    if re.search(
        r"\b(res[uú]m\w*|sintetiza\w*|s[ií]ntesis|ideas principales|sumario)\b",
        lowered,
    ):
        return "summary"
    if re.search(
        r"\b(que dia es hoy|qué dia es hoy|qué día es hoy|que fecha es hoy|"
        r"qué fecha es hoy|fecha actual|fecha de hoy|dia actual|día actual)\b",
        lowered,
    ):
        return "date"
    if re.search(
        r"\b(busca\w*|consulta\w*|mira|investiga\w*|encuentra\w*|googlea\w*)\b"
        r".*\b(internet|web|online|en linea|en línea)\b",
        lowered,
    ) or re.search(
        r"\b(en internet|en la web|online|en linea|en línea|noticias|actualidad|"
        r"[uú]ltima hora|reciente|actualizad[oa]s?|precio actual|cotizaci[oó]n)\b",
        lowered,
    ):
        return "web"
    topic_term = re.search(r"\b(t[oó]picos?|temas|clusters?)\b", lowered)
    topic_scope = re.search(
        r"\b(corpus|offline|detectad[oa]s?|modelado|lista|listado|"
        r"hay|tenemos|reporte|informe)\b",
        lowered,
    )
    if topic_term and topic_scope:
        return "rag_topics"
    return None


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    action: str
    sources: list[str]


class AssistantAgent:
    def __init__(
        self,
        llm: LLMClient,
        rag_tool: RagTool,
        summary_tool: SummaryTool,
        topic_tool: TopicTool | None = None,
        date_tool: DateTool | None = None,
        web_tool: WebSearchTool | None = None,
        use_llm_router: bool = True,
    ) -> None:
        self.llm = llm
        self.rag_tool = rag_tool
        self.summary_tool = summary_tool
        self.topic_tool = topic_tool
        self.date_tool = date_tool or DateTool()
        self.web_tool = web_tool
        self.use_llm_router = use_llm_router
        self.history: list[tuple[str, str]] = []

    def reset(self) -> None:
        """Limpia la memoria conversacional."""
        self.history = []

    def answer(
        self,
        query: str,
        on_token: Callable[[str], None] | None = None,
        on_action: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        # Los disparadores explicitos del usuario mandan sobre la reescritura
        # y el router LLM.
        deterministic = deterministic_route(query)
        # Memoria conversacional: reescribe seguimientos como pregunta autonoma.
        needs_rewrite = self.history and (
            deterministic is None or (deterministic == "web" and self._references_history(query))
        )
        search_query = self._rewrite_with_history(query) if needs_rewrite else query
        action = deterministic or self._route(search_query)
        if action == "web" and self.web_tool is None:
            action = "rag"
        if action == "summary":
            search_query = self._summary_search_query(search_query)
        if on_action is not None:
            on_action(action)
        response = self._dispatch(action, query, search_query, on_token=on_token)
        self.history.append((query, response.answer))
        self.history = self.history[-HISTORY_TURNS:]
        return response

    def summarize_file(self, path: Path, sentences: int = 4) -> AgentResponse:
        summary = self.summary_tool.run_file(path, sentences=sentences)
        return AgentResponse(answer=summary, action="summary_file", sources=[str(path)])

    # ------------------------------------------------------------------
    # Despacho de herramientas
    # ------------------------------------------------------------------
    def _dispatch(
        self,
        action: str,
        query: str,
        search_query: str,
        on_token: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        if action == "date":
            answer = self.date_tool.run()
            if on_token is not None:
                self._stream_text(answer, on_token)
            return AgentResponse(answer, action, [])

        if action == "summary":
            if self.history and self._references_history(query):
                history_text = "\n".join(answer for _, answer in self.history)
                summary = self.summary_tool.run_text(history_text, sentences=5, query=query)
                if on_token is not None:
                    self._stream_text(summary, on_token)
                return AgentResponse(summary, action, [])
            context, sources = self.rag_tool.run(search_query)
            summary = (
                self.summary_tool.run_text(context, sentences=5, query=search_query)
                if context
                else "No hay contexto suficiente."
            )
            if on_token is not None:
                self._stream_text(summary, on_token)
            return AgentResponse(summary, action, sources)

        if action == "web" and self.web_tool is not None:
            context, sources = self.rag_tool.run(search_query)
            web_context, web_sources = self.web_tool.run(search_query)
            if not web_context:
                web_context = "Sin resultados web (sin conexion o sin coincidencias)."
            prompt = WEB_PROMPT_TEMPLATE.format(
                query=query,
                context=context or "Sin resultados relevantes.",
                web_context=web_context,
            )
            answer = self._generate_answer(prompt, SYSTEM_PROMPT, on_token=on_token)
            return AgentResponse(
                answer=answer,
                action="web",
                sources=self._filter_sources_by_citations(answer, sources + web_sources),
            )

        context, sources = self.rag_tool.run(search_query)
        if action == "rag_topics" and self.topic_tool is not None:
            topic_context = self.topic_tool.run()
            context = f"Topicos offline del corpus:\n{topic_context}\n\nContexto RAG:\n{context}"
        prompt = RAG_PROMPT_TEMPLATE.format(
            query=query, context=context or "Sin resultados relevantes."
        )
        answer = self._generate_answer(prompt, SYSTEM_PROMPT, on_token=on_token)
        return AgentResponse(
            answer=answer,
            action=action,
            sources=self._filter_sources_by_citations(answer, sources),
        )

    def _generate_answer(
        self,
        prompt: str,
        system: str | None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        if on_token is None:
            return self.llm.generate(prompt=prompt, system=system)

        chunks: list[str] = []
        for chunk in self.llm.stream_generate(prompt=prompt, system=system):
            if not chunk:
                continue
            on_token(chunk)
            chunks.append(chunk)
        return "".join(chunks).strip()

    # ------------------------------------------------------------------
    # Enrutado
    # ------------------------------------------------------------------
    def _route(self, query: str) -> str:
        # Atajos deterministas para disparadores explicitos (rapidos y sin coste LLM).
        deterministic = deterministic_route(query)
        if deterministic is not None:
            return deterministic
        # Caso ambiguo: el LLM elige entre las herramientas restantes.
        if self.use_llm_router:
            return self._llm_route(query)
        return "rag"

    def _llm_route(self, query: str) -> str:
        prompt = ROUTER_PROMPT_TEMPLATE.format(query=query)
        try:
            raw = self.llm.generate(prompt=prompt, system="Responde solo con JSON valido.")
        except Exception:
            return "rag"
        tool = self._parse_tool(raw)
        if tool == "topics":
            return "rag_topics" if self.topic_tool is not None else "rag"
        if tool == "web" and self.web_tool is None:
            return "rag"
        if tool in VALID_ACTIONS:
            return tool
        return "rag"

    @staticmethod
    def _parse_tool(raw: str) -> str | None:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                value = str(payload.get("tool", "")).strip().lower()
                if value:
                    return value
            except (json.JSONDecodeError, AttributeError):
                pass
        # Fallback: busca un nombre de herramienta conocido en el texto plano.
        for candidate in ("topics", "summary", "date", "web", "rag"):
            if candidate in raw.lower():
                return candidate
        return None

    def _rewrite_with_history(self, query: str) -> str:
        history_text = "\n".join(
            f"Usuario: {user}\nAsistente: {answer}" for user, answer in self.history
        )
        prompt = REWRITE_PROMPT_TEMPLATE.format(history=history_text, query=query)
        try:
            rewritten = self.llm.generate(prompt=prompt, system="Reescribe consultas.")
        except Exception:
            return query
        rewritten = rewritten.strip().splitlines()[0].strip() if rewritten.strip() else ""
        return rewritten or query

    @staticmethod
    def _summary_search_query(query: str) -> str:
        cleaned = re.sub(
            r"\b(hazme|haznos|haz|dame|realiza|genera|escribe|prepara)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b(un|una)?\s*(res[uú]m\w*|sintetiza\w*|s[ií]ntesis|sumario)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b(de|del|sobre|acerca de|acerca del|acerca de la)\b",
            " ",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;¿?¡!")
        return cleaned or query

    @staticmethod
    def _references_history(query: str) -> bool:
        lowered = query.lower()
        if re.search(
            r"\b(eso|esto|anterior|previo|previa|[uú]ltim[oa]|mencionad[oa]|dicho)\b",
            lowered,
        ):
            return True
        return bool(
            re.search(
                r"\b(?:busca|consulta|mira|investiga|encuentra|resume|res[uú]me|haz|"
                r"dime|explica|ampl[ií]a)\w*(?:lo|la|los|las)\b",
                lowered,
            )
        )

    @staticmethod
    def _filter_sources_by_citations(answer: str, sources: list[str]) -> list[str]:
        citations = {
            (label.lower(), number)
            for label, number in re.findall(
                r"\[(Fuente|Web)\s+(\d+)\]", answer, flags=re.IGNORECASE
            )
        }
        if not citations:
            return []
        filtered = []
        for source in sources:
            match = re.match(r"\[(Fuente|Web)\s+(\d+)\]", source)
            if match and (match.group(1).lower(), match.group(2)) in citations:
                filtered.append(source)
        return filtered

    @staticmethod
    def _stream_text(text: str, on_token: Callable[[str], None]) -> None:
        import time
        chunks = re.findall(r"\S+\s*|\s+", text)
        for chunk in chunks:
            if chunk:
                on_token(chunk)
                time.sleep(0.025)
