from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import re

from mitex_assistant.agents.prompts import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT
from mitex_assistant.agents.tools import DateTool, RagTool, SummaryTool, TopicTool
from mitex_assistant.llm.base import LLMClient


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
    ) -> None:
        self.llm = llm
        self.rag_tool = rag_tool
        self.summary_tool = summary_tool
        self.topic_tool = topic_tool
        self.date_tool = date_tool or DateTool()

    def answer(self, query: str) -> AgentResponse:
        action = self._route(query)
        if action == "summary":
            context, sources = self.rag_tool.run(query)
            summary = self.summary_tool.run_text(context, sentences=5) if context else "No hay contexto suficiente."
            return AgentResponse(summary, action, sources)
        if action == "date":
            return AgentResponse(self.date_tool.run(), action, [])

        context, sources = self.rag_tool.run(query)
        if action == "rag_topics" and self.topic_tool is not None:
            topic_context = self.topic_tool.run()
            context = f"Topicos offline del corpus:\n{topic_context}\n\nContexto RAG:\n{context}"
        prompt = RAG_PROMPT_TEMPLATE.format(query=query, context=context or "Sin resultados relevantes.")
        answer = self.llm.generate(prompt=prompt, system=SYSTEM_PROMPT)
        return AgentResponse(answer=answer, action=action, sources=sources)

    def summarize_file(self, path: Path, sentences: int = 4) -> AgentResponse:
        summary = self.summary_tool.run_file(path, sentences=sentences)
        return AgentResponse(answer=summary, action="summary_file", sources=[str(path)])

    def _route(self, query: str) -> str:
        lowered = query.lower()
        if re.search(r"\b(resume|resumen|sintetiza|ideas principales)\b", lowered):
            return "summary"
        if re.search(r"\b(que dia es hoy|qué dia es hoy|qué día es hoy|que fecha es hoy|qué fecha es hoy|fecha actual|fecha de hoy|dia actual|día actual)\b", lowered):
            return "date"
        if re.search(r"\b(topico|topicos|tema|temas|cluster)\b", lowered):
            return "rag_topics"
        return "rag"
