from __future__ import annotations

import re
import unicodedata

try:
    import ftfy
except ImportError:  # pragma: no cover - fallback for minimal environments
    ftfy = None


INVISIBLE_CHARS = re.compile(r"[\u200b\u200c\u200d\ufeff]")
WHITESPACE = re.compile(r"\s+")
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignora\s+(todas\s+)?las\s+instrucciones", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
]


def normalize_text(text: str) -> str:
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLE_CHARS.sub("", text)
    text = WHITESPACE.sub(" ", text)
    return text.strip()


def strip_prompt_injection(text: str) -> str:
    cleaned = text
    for pattern in PROMPT_INJECTION_PATTERNS:
        cleaned = pattern.sub("[INSTRUCCION EXTERNA FILTRADA]", cleaned)
    return cleaned


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)
