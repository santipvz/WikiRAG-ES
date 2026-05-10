from __future__ import annotations

import hashlib


def stable_id(*parts: str, length: int = 16) -> str:
    payload = "::".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]
