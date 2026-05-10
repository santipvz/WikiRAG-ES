from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import re

from mitex_assistant.agents.prompts import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT
from mitex_assistant.agents.tools import RagTool, SummaryTool
from mitex_assistant.llm.base import LLMClient


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    action: str
    sources: list[str]


class AssistantAgent:
    def __init__(self, llm: LLMClient, rag_tool: RagTool, summary_tool: SummaryTool) -> None:
        self.llm = llm
        self.rag_tool = rag_tool
        self.summary_tool = summary_tool

    def answer(self, query: str) -> AgentResponse:
        action = self._route(query)
        if action == "summary":
            context, sources = self.rag_tool.run(query)
            summary = self.summary_tool.run_text(context, sentences=5) if context else "No hay contexto suficiente."
            return AgentResponse(summary, action, sources)

        context, sources = self.rag_tool.run(query)
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
        if re.search(r"\b(topico|topicos|tema|temas|cluster)\b", lowered):
            return "rag_topics"
        return "rag"
