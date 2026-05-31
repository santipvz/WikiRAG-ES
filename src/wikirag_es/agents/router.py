from __future__ import annotations

import json
import re
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
    if re.search(r"\b(resume|resumen|sintetiza|ideas principales)\b", lowered):
        return "summary"
    if re.search(
        r"\b(que dia es hoy|qué dia es hoy|qué día es hoy|que fecha es hoy|"
        r"qué fecha es hoy|fecha actual|fecha de hoy|dia actual|día actual)\b",
        lowered,
    ):
        return "date"
    if re.search(r"\b(t[oó]pico|t[oó]picos|tema|temas|cluster|clusters)\b", lowered):
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

    def answer(self, query: str) -> AgentResponse:
        # Memoria conversacional: reescribe seguimientos como pregunta autonoma.
        search_query = self._rewrite_with_history(query) if self.history else query
        action = self._route(search_query)
        response = self._dispatch(action, query, search_query)
        self.history.append((query, response.answer))
        self.history = self.history[-HISTORY_TURNS:]
        return response

    def summarize_file(self, path: Path, sentences: int = 4) -> AgentResponse:
        summary = self.summary_tool.run_file(path, sentences=sentences)
        return AgentResponse(answer=summary, action="summary_file", sources=[str(path)])

    # ------------------------------------------------------------------
    # Despacho de herramientas
    # ------------------------------------------------------------------
    def _dispatch(self, action: str, query: str, search_query: str) -> AgentResponse:
        if action == "date":
            return AgentResponse(self.date_tool.run(), action, [])

        if action == "summary":
            context, sources = self.rag_tool.run(search_query)
            summary = (
                self.summary_tool.run_text(context, sentences=5, query=search_query)
                if context
                else "No hay contexto suficiente."
            )
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
            answer = self.llm.generate(prompt=prompt, system=SYSTEM_PROMPT)
            return AgentResponse(answer=answer, action="web", sources=sources + web_sources)

        context, sources = self.rag_tool.run(search_query)
        if action == "rag_topics" and self.topic_tool is not None:
            topic_context = self.topic_tool.run()
            context = f"Topicos offline del corpus:\n{topic_context}\n\nContexto RAG:\n{context}"
        prompt = RAG_PROMPT_TEMPLATE.format(
            query=query, context=context or "Sin resultados relevantes."
        )
        answer = self.llm.generate(prompt=prompt, system=SYSTEM_PROMPT)
        return AgentResponse(answer=answer, action=action, sources=sources)

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
