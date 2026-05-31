from __future__ import annotations

import requests

from wikirag_es.llm.base import LLMClient


class OllamaClient(LLMClient):
    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: int = 120) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": system or "",
                "stream": False,
                "options": {"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
