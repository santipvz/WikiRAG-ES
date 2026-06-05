from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError

    def stream_generate(self, prompt: str, system: str | None = None) -> Iterator[str]:
        answer = self.generate(prompt=prompt, system=system)
        if answer:
            yield answer
