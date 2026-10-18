from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def looks_like_english(text: str, *, min_chars: int = 80, max_chars: int = 12_000) -> bool:
    text = text.strip()
    n = len(text)
    if n < min_chars or n > max_chars:
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters / n < 0.58:
        return False
    if text.count(" ") < 12:
        return False
    if text.count("http") > 3 or text.count("{") > 8:
        return False
    return True


def chunk_document(text: str, target: int = 1400) -> list[str]:
    text = _WS.sub(" ", text.strip())
    if len(text) <= target:
        return [text] if looks_like_english(text, min_chars=80, max_chars=target + 200) else []
    parts: list[str] = []
    buf: list[str] = []
    n = 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if n + len(sent) > target and buf:
            blob = " ".join(buf).strip()
            if looks_like_english(blob, min_chars=80, max_chars=target + 400):
                parts.append(blob)
            buf, n = [sent], len(sent)
        else:
            buf.append(sent)
            n += len(sent) + 1
    if buf:
        blob = " ".join(buf).strip()
        if looks_like_english(blob, min_chars=80, max_chars=target + 400):
            parts.append(blob)
    return parts
