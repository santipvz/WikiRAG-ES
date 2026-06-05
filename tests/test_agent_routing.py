from wikirag_es.agents.router import AssistantAgent, deterministic_route
from wikirag_es.agents.web_search import WebSearchTool


class FakeLLM:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.response

    def stream_generate(self, prompt: str, system: str | None = None):
        self.prompts.append(prompt)
        for chunk in self.response.split("|"):
            if chunk:
                yield chunk


class FakeRag:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, query: str):
        self.queries.append(query)
        return f"contexto de {query}", ["[Fuente 1] mem"]


class FakeSummary:
    def run_text(self, text: str, sentences: int = 4, query=None) -> str:
        return f"resumen:{text}"


class FakeWeb:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, query: str):
        self.queries.append(query)
        return f"web de {query}", ["[Web 1] https://example.com"]


def _agent(llm, web_tool=None, use_llm_router=True, rag_tool=None) -> AssistantAgent:
    return AssistantAgent(
        llm=llm,
        rag_tool=rag_tool or FakeRag(),
        summary_tool=FakeSummary(),
        web_tool=web_tool,
        use_llm_router=use_llm_router,
    )


def test_regex_route_detects_summary_and_date() -> None:
    agent = _agent(FakeLLM())
    assert agent._route("Resume este documento") == "summary"
    assert deterministic_route("Resúmeme las ideas principales") == "summary"
    assert agent._route("¿Qué día es hoy?") == "date"
    assert deterministic_route("Busca en internet noticias de inteligencia artificial") == "web"
    assert agent._route("¿Qué tópicos hay en el corpus?") == "rag_topics"
    assert agent._route("Dame la lista de clusters detectados") == "rag_topics"


def test_topic_route_does_not_capture_generic_theme_questions() -> None:
    agent = _agent(FakeLLM('{"tool": "rag"}'))

    assert agent._route("¿Cuál es el tema principal de Don Quijote?") == "rag"
    assert agent._route("¿Qué es un cluster en Wikipedia?") == "rag"


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


def test_explicit_summary_skips_history_rewrite() -> None:
    llm = FakeLLM("consulta reescrita sin palabra resumen")
    agent = _agent(llm)
    agent.history.append(("¿Qué sabes de la mitología griega?", "Respuesta previa."))

    response = agent.answer("Resume lo anterior")

    assert response.action == "summary"
    assert response.answer == "resumen:Respuesta previa."
    assert llm.prompts == []


def test_summary_search_query_removes_instruction_words() -> None:
    assert (
        AssistantAgent._summary_search_query("Hazme un resumen de la revolucion industrial")
        == "la revolucion industrial"
    )


def test_conversational_memory_rewrites_followup() -> None:
    llm = FakeLLM("consulta reescrita")
    agent = _agent(llm, use_llm_router=False)
    agent.history.append(("¿Capital de Francia?", "París."))
    agent.answer("¿y su poblacion?")
    # El primer prompt enviado al LLM es la reescritura con historial.
    assert "Historial reciente" in llm.prompts[0]


def test_answer_streams_final_rag_response_and_updates_history() -> None:
    llm = FakeLLM("hola| mundo")
    agent = _agent(llm, use_llm_router=False)
    chunks: list[str] = []
    actions: list[str] = []

    response = agent.answer("pregunta", on_token=chunks.append, on_action=actions.append)

    assert actions == ["rag"]
    assert chunks == ["hola", " mundo"]
    assert response.answer == "hola mundo"
    assert agent.history == [("pregunta", "hola mundo")]


def test_summary_route_streams_without_llm_router() -> None:
    llm = FakeLLM("no deberia llamarse")
    agent = _agent(llm)
    agent.history.append(("previa", "respuesta previa"))
    chunks: list[str] = []

    response = agent.answer("Hazme un resumen de la revolucion industrial", on_token=chunks.append)

    assert response.action == "summary"
    assert response.answer == "resumen:contexto de la revolucion industrial"
    assert chunks == ["resumen:contexto de la revolucion industrial"]
    assert llm.prompts == []


def test_explicit_web_route_skips_llm_router() -> None:
    llm = FakeLLM("respuesta web")
    web = FakeWeb()
    agent = _agent(llm, web_tool=web)
    actions: list[str] = []

    response = agent.answer(
        "Busca en internet noticias de inteligencia artificial", on_action=actions.append
    )

    assert actions == ["web"]
    assert response.action == "web"
    assert web.queries == ["Busca en internet noticias de inteligencia artificial"]
    assert len(llm.prompts) == 1  # solo generacion final, sin router previo


def test_explicit_web_falls_back_to_rag_when_unavailable() -> None:
    llm = FakeLLM("respuesta rag")
    rag = FakeRag()
    agent = _agent(llm, web_tool=None, rag_tool=rag)

    response = agent.answer("Busca en internet noticias de inteligencia artificial")

    assert response.action == "rag"
    assert rag.queries == ["Busca en internet noticias de inteligencia artificial"]


def test_rag_response_without_citations_hides_retrieved_sources() -> None:
    agent = _agent(FakeLLM("No se encuentra en el contexto."), use_llm_router=False)

    response = agent.answer("Qué es in silbuturomaceduro?")

    assert response.action == "rag"
    assert response.sources == []


def test_rag_response_keeps_only_cited_sources() -> None:
    agent = _agent(FakeLLM("Respuesta basada en el corpus. [Fuente 1]"), use_llm_router=False)

    response = agent.answer("Pregunta local")

    assert response.sources == ["[Fuente 1] mem"]


def test_web_response_keeps_only_cited_web_sources() -> None:
    agent = _agent(FakeLLM("Respuesta basada en internet. [Web 1]"), web_tool=FakeWeb())

    response = agent.answer("Busca en internet noticias de inteligencia artificial")

    assert response.action == "web"
    assert response.sources == ["[Web 1] https://example.com"]
