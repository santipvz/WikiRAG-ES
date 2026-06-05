from __future__ import annotations

import json
from collections.abc import Iterator

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
            json=self._payload(prompt=prompt, system=system, stream=False),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def stream_generate(self, prompt: str, system: str | None = None) -> Iterator[str]:
        with requests.post(
            f"{self.base_url}/api/generate",
            json=self._payload(prompt=prompt, system=system, stream=True),
            timeout=self.timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                payload = json.loads(line)
                chunk = payload.get("response", "")
                if chunk:
                    yield chunk
                if payload.get("done"):
                    break

    def _payload(self, prompt: str, system: str | None, stream: bool) -> dict[str, object]:
        return {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": stream,
            "options": {"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1},
        }
