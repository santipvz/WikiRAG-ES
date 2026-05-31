from wikirag_es.agents.router import AssistantAgent
from wikirag_es.agents.web_search import WebSearchTool


class FakeLLM:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeRag:
    def run(self, query: str):
        return f"contexto de {query}", ["[Fuente 1] mem"]


class FakeSummary:
    def run_text(self, text: str, sentences: int = 4, query=None) -> str:
        return f"resumen:{text}"


def _agent(llm, web_tool=None, use_llm_router=True) -> AssistantAgent:
    return AssistantAgent(
        llm=llm,
        rag_tool=FakeRag(),
        summary_tool=FakeSummary(),
        web_tool=web_tool,
        use_llm_router=use_llm_router,
    )


def test_regex_route_detects_summary_and_date() -> None:
    agent = _agent(FakeLLM())
    assert agent._route("Resume este documento") == "summary"
    assert agent._route("¿Qué día es hoy?") == "date"
    assert agent._route("¿Qué tópicos hay en el corpus?") == "rag_topics"


def test_llm_route_picks_web_when_tool_available() -> None:
    agent = _agent(FakeLLM('{"tool": "web"}'), web_tool=WebSearchTool())
    assert agent._route("¿Cuál es el precio actual del bitcoin?") == "web"


def test_llm_route_falls_back_to_rag_when_web_unavailable() -> None:
    agent = _agent(FakeLLM('{"tool": "web"}'), web_tool=None)
    assert agent._route("noticia de hoy") == "rag"


def test_llm_route_disabled_defaults_to_rag() -> None:
    agent = _agent(FakeLLM('{"tool": "web"}'), use_llm_router=False)
    assert agent._route("algo ambiguo") == "rag"


def test_parse_tool_handles_noisy_json() -> None:
    assert AssistantAgent._parse_tool('Claro: {"tool": "summary"} listo') == "summary"
    assert AssistantAgent._parse_tool("creo que la herramienta rag") == "rag"


def test_date_action_uses_date_tool_without_llm() -> None:
    llm = FakeLLM("no deberia llamarse")
    agent = _agent(llm)
    response = agent.answer("¿qué fecha es hoy?")
    assert response.action == "date"
    assert "Hoy es" in response.answer
    assert llm.prompts == []  # ruta determinista, sin coste LLM


def test_conversational_memory_rewrites_followup() -> None:
    llm = FakeLLM("consulta reescrita")
    agent = _agent(llm, use_llm_router=False)
    agent.history.append(("¿Capital de Francia?", "París."))
    agent.answer("¿y su poblacion?")
    # El primer prompt enviado al LLM es la reescritura con historial.
    assert "Historial reciente" in llm.prompts[0]
