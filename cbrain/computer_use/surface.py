"""Browser backend seam. Production must not dial targets from the agent NS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


class ComputerBackendError(RuntimeError):
    """The computer backend could not complete an observation or act."""


class ProductionComputerRequired(ComputerBackendError):
    """Production refused a DEV_ONLY in-process browser backend."""


@dataclass(frozen=True, slots=True)
class BackendObservation:
    title: str
    url_alias: str
    accessibility_tree: str
    text_excerpt: str


@dataclass(frozen=True, slots=True)
class BackendActResult:
    ok: bool
    detail: str
    observation: BackendObservation | None = None


class ComputerBackend(Protocol):
    """Seam for a desktop/browser worker outside the agent process."""

    DEV_ONLY: bool

    def observe(self, *, url_alias: str) -> BackendObservation:
        """Read-only snapshot of the current page for ``url_alias``."""

    def navigate(self, *, url_alias: str, url: str) -> BackendActResult:
        """Open an operator-resolved URL under ``url_alias``."""

    def click(self, *, url_alias: str, selector: str) -> BackendActResult:
        """Click an element identified by a stable selector."""

    def type_text(
        self, *, url_alias: str, selector: str, text: str
    ) -> BackendActResult:
        """Type into a field. Credentials must never be supplied by the model."""

    def submit(self, *, url_alias: str, selector: str) -> BackendActResult:
        """Submit a form or confirm a primary action."""


@dataclass
class DevOnlyBrowserBackend:
    """In-memory browser stub for unit tests. DEV_ONLY — never production.

    Stores pages keyed by alias. Does not dial the network. Label is deliberate:
    if an engineer can curl the CRM from the agent container, you do not have a
    product; this backend cannot be the path that makes that true.
    """

    pages: dict[str, dict[str, Any]] = field(default_factory=dict)
    DEV_ONLY: bool = True

    def observe(self, *, url_alias: str) -> BackendObservation:
        page = self.pages.get(url_alias)
        if page is None:
            raise ComputerBackendError(f"no open page for alias {url_alias!r}")
        return BackendObservation(
            title=str(page.get("title", "")),
            url_alias=url_alias,
            accessibility_tree=str(page.get("accessibility_tree", "")),
            text_excerpt=_sanitize_excerpt(str(page.get("text_excerpt", ""))),
        )

    def navigate(self, *, url_alias: str, url: str) -> BackendActResult:
        # URL is recorded only for test assertions; never returned to the model.
        self.pages[url_alias] = {
            "title": f"page:{url_alias}",
            "url": url,
            "accessibility_tree": f"document[alias={url_alias}]",
            "text_excerpt": f"Opened {url_alias}",
        }
        return BackendActResult(
            ok=True,
            detail="navigated",
            observation=self.observe(url_alias=url_alias),
        )

    def click(self, *, url_alias: str, selector: str) -> BackendActResult:
        self._require_open(url_alias)
        page = self.pages[url_alias]
        page["text_excerpt"] = f"clicked:{selector}"
        return BackendActResult(
            ok=True,
            detail=f"clicked:{selector}",
            observation=self.observe(url_alias=url_alias),
        )

    def type_text(
        self, *, url_alias: str, selector: str, text: str
    ) -> BackendActResult:
        self._require_open(url_alias)
        if _looks_like_secret(text):
            raise ComputerBackendError(
                "refusing to type secret-shaped text from model context"
            )
        page = self.pages[url_alias]
        page["text_excerpt"] = f"typed:{selector}"
        return BackendActResult(
            ok=True,
            detail=f"typed:{selector}",
            observation=self.observe(url_alias=url_alias),
        )

    def submit(self, *, url_alias: str, selector: str) -> BackendActResult:
        self._require_open(url_alias)
        page = self.pages[url_alias]
        page["text_excerpt"] = f"submitted:{selector}"
        return BackendActResult(
            ok=True,
            detail=f"submitted:{selector}",
            observation=self.observe(url_alias=url_alias),
        )

    def _require_open(self, url_alias: str) -> None:
        if url_alias not in self.pages:
            raise ComputerBackendError(f"no open page for alias {url_alias!r}")


def require_production_computer_backend(backend: ComputerBackend) -> None:
    """Fail closed if assembly tries to ship a DEV_ONLY browser as production."""

    if getattr(backend, "DEV_ONLY", False) is True:
        raise ProductionComputerRequired(
            "DevOnlyBrowserBackend is DEV_ONLY; production requires an "
            "out-of-process computer worker reached via sole-egress sidecar"
        )


_SECRET_MARKERS = (
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "token",
    "bearer ",
)


def _looks_like_secret(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SECRET_MARKERS)


def _sanitize_excerpt(text: str) -> str:
    """Strip obvious secret-shaped lines before model context."""

    lines = []
    for line in text.splitlines() or [text]:
        if _looks_like_secret(line):
            lines.append("[redacted]")
        else:
            lines.append(line)
    return "\n".join(lines)


__all__ = [
    "BackendActResult",
    "BackendObservation",
    "ComputerBackend",
    "ComputerBackendError",
    "DevOnlyBrowserBackend",
    "ProductionComputerRequired",
    "require_production_computer_backend",
]
