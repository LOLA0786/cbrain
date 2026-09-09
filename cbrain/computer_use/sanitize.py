"""Shared secret-shape sanitization for computer observations."""

from __future__ import annotations

import hashlib
import re
from typing import Final

_SECRET_MARKERS: Final[tuple[str, ...]] = (
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "token",
    "bearer ",
    "authorization:",
)

# Obvious credential-looking assignments in accessibility text.
_ASSIGNMENT = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|token|authorization)\b\s*[:=]\s*\S+"
)


def looks_like_secret(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SECRET_MARKERS)


def sanitize_excerpt(text: str) -> str:
    """Strip secret-shaped lines before model context."""

    lines: list[str] = []
    for line in text.splitlines() or [text]:
        if looks_like_secret(line) or _ASSIGNMENT.search(line):
            lines.append("[redacted]")
        else:
            lines.append(line)
    return "\n".join(lines)


def screenshot_digest(png_bytes: bytes) -> str:
    """Evidence handle for a screenshot. Raw bytes never enter model context."""

    return "sha256:" + hashlib.sha256(png_bytes).hexdigest()


__all__ = [
    "looks_like_secret",
    "sanitize_excerpt",
    "screenshot_digest",
]
