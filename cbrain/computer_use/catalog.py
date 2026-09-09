"""Operator-owned page aliases. Models never see raw URLs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final
from urllib.parse import urlparse


class PageCatalogError(ValueError):
    """Page catalog configuration or resolution failed."""


@dataclass(frozen=True, slots=True)
class PageAlias:
    """A named destination the model may propose; URL stays operator-owned."""

    alias: str
    url: str
    capability: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.alias, str) or not self.alias.strip():
            raise PageCatalogError("alias must be non-empty")
        if not isinstance(self.capability, str) or not self.capability.strip():
            raise PageCatalogError("capability must be non-empty")
        if not isinstance(self.description, str) or not self.description.strip():
            raise PageCatalogError("description must be non-empty")
        _require_https_or_loopback(self.url)


def _require_https_or_loopback(url: str) -> None:
    if not isinstance(url, str) or not url.strip():
        raise PageCatalogError("url must be non-empty")
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}:
        return
    raise PageCatalogError(
        "page URL must be https (or http://localhost / http://127.0.0.1 for dev)"
    )


class PageCatalog:
    """Immutable alias→URL map. Models receive aliases + descriptions only."""

    def __init__(self, pages: Iterable[PageAlias]) -> None:
        values: dict[str, PageAlias] = {}
        for page in pages:
            if page.alias in values:
                raise PageCatalogError(f"duplicate page alias {page.alias!r}")
            values[page.alias] = page
        self._pages: Mapping[str, PageAlias] = MappingProxyType(values)

    def resolve(self, alias: str) -> PageAlias:
        try:
            return self._pages[alias]
        except KeyError as exc:
            raise PageCatalogError(f"unknown page alias {alias!r}") from exc

    def model_view(self) -> tuple[Mapping[str, str], ...]:
        """What the model may see: alias, capability, description — never URL."""

        return tuple(
            MappingProxyType(
                {
                    "alias": page.alias,
                    "capability": page.capability,
                    "description": page.description,
                }
            )
            for page in sorted(self._pages.values(), key=lambda p: p.alias)
        )

    def contains(self, alias: str) -> bool:
        return alias in self._pages


# Marker for audits: catalog URLs are operator config, not model output.
OPERATOR_OWNED_URLS: Final = True

__all__ = [
    "OPERATOR_OWNED_URLS",
    "PageAlias",
    "PageCatalog",
    "PageCatalogError",
]
