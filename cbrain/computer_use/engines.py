"""Browser engines that run only inside the computer worker process."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .sanitize import looks_like_secret, sanitize_excerpt, screenshot_digest
from .surface import BackendActResult, BackendObservation, ComputerBackendError


class BrowserEngine(Protocol):
    """Real browser or stub. Lives only in the worker process."""

    def observe(self, *, url_alias: str) -> BackendObservation: ...

    def navigate(self, *, url_alias: str, url: str) -> BackendActResult: ...

    def click(self, *, url_alias: str, selector: str) -> BackendActResult: ...

    def type_text(
        self, *, url_alias: str, selector: str, text: str
    ) -> BackendActResult: ...

    def submit(self, *, url_alias: str, selector: str) -> BackendActResult: ...


@dataclass
class StubBrowserEngine:
    """Deterministic in-worker engine for CI and isolation probes.

    Does not dial the network. Production assemblies that need a real browser
    set ``CBRAIN_COMPUTER_ENGINE=playwright`` on the worker.
    """

    pages: dict[str, dict[str, Any]] = field(default_factory=dict)
    include_screenshot_digest: bool = True

    def observe(self, *, url_alias: str) -> BackendObservation:
        page = self._require(url_alias)
        digest = None
        if self.include_screenshot_digest:
            fake_png = f"stub-screenshot:{url_alias}".encode()
            digest = screenshot_digest(fake_png)
        return BackendObservation(
            title=str(page.get("title", "")),
            url_alias=url_alias,
            accessibility_tree=sanitize_excerpt(
                str(page.get("accessibility_tree", ""))
            ),
            text_excerpt=sanitize_excerpt(str(page.get("text_excerpt", ""))),
            screenshot_sha256=digest,
            screenshot_media_type="image/png" if digest else None,
        )

    def navigate(self, *, url_alias: str, url: str) -> BackendActResult:
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
        page = self._require(url_alias)
        page["text_excerpt"] = f"clicked:{selector}"
        page["accessibility_tree"] = f"document[alias={url_alias}] focus={selector}"
        return BackendActResult(
            ok=True,
            detail=f"clicked:{selector}",
            observation=self.observe(url_alias=url_alias),
        )

    def type_text(
        self, *, url_alias: str, selector: str, text: str
    ) -> BackendActResult:
        if looks_like_secret(text):
            raise ComputerBackendError(
                "refusing to type secret-shaped text from model context"
            )
        page = self._require(url_alias)
        page["text_excerpt"] = f"typed:{selector}"
        return BackendActResult(
            ok=True,
            detail=f"typed:{selector}",
            observation=self.observe(url_alias=url_alias),
        )

    def submit(self, *, url_alias: str, selector: str) -> BackendActResult:
        page = self._require(url_alias)
        page["text_excerpt"] = f"submitted:{selector}"
        return BackendActResult(
            ok=True,
            detail=f"submitted:{selector}",
            observation=self.observe(url_alias=url_alias),
        )

    def _require(self, url_alias: str) -> dict[str, Any]:
        page = self.pages.get(url_alias)
        if page is None:
            raise ComputerBackendError(f"no open page for alias {url_alias!r}")
        return page


class PlaywrightBrowserEngine:
    """Playwright-backed engine. Optional dependency ``computer`` extra.

    Runs only in the worker process. The agent never imports Playwright.
    """

    def __init__(self, *, headless: bool = True, timeout_ms: int = 15_000) -> None:
        try:
            import importlib

            sync_playwright = importlib.import_module(
                "playwright.sync_api"
            ).sync_playwright
        except ImportError as exc:  # pragma: no cover - exercised when extra missing
            raise ComputerBackendError(
                "playwright is not installed; pip install with the 'computer' extra "
                "or set CBRAIN_COMPUTER_ENGINE=stub"
            ) from exc

        self._timeout_ms = timeout_ms
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=headless)
        self._contexts: dict[str, Any] = {}
        self._pages: dict[str, Any] = {}

    def close(self) -> None:
        for context in self._contexts.values():
            context.close()
        self._browser.close()
        self._playwright.stop()

    def observe(self, *, url_alias: str) -> BackendObservation:
        page = self._require_page(url_alias)
        tree = page.locator("body").inner_text(timeout=self._timeout_ms)
        png = page.screenshot(type="png", full_page=False)
        return BackendObservation(
            title=page.title() or "",
            url_alias=url_alias,
            accessibility_tree=sanitize_excerpt(tree[:8_000]),
            text_excerpt=sanitize_excerpt(tree[:2_000]),
            screenshot_sha256=screenshot_digest(png),
            screenshot_media_type="image/png",
        )

    def navigate(self, *, url_alias: str, url: str) -> BackendActResult:
        page = self._page_for(url_alias)
        page.goto(url, wait_until="domcontentloaded", timeout=self._timeout_ms)
        return BackendActResult(
            ok=True,
            detail="navigated",
            observation=self.observe(url_alias=url_alias),
        )

    def click(self, *, url_alias: str, selector: str) -> BackendActResult:
        page = self._require_page(url_alias)
        page.locator(selector).first.click(timeout=self._timeout_ms)
        return BackendActResult(
            ok=True,
            detail=f"clicked:{selector}",
            observation=self.observe(url_alias=url_alias),
        )

    def type_text(
        self, *, url_alias: str, selector: str, text: str
    ) -> BackendActResult:
        if looks_like_secret(text):
            raise ComputerBackendError(
                "refusing to type secret-shaped text from model context"
            )
        page = self._require_page(url_alias)
        page.locator(selector).first.fill(text, timeout=self._timeout_ms)
        return BackendActResult(
            ok=True,
            detail=f"typed:{selector}",
            observation=self.observe(url_alias=url_alias),
        )

    def submit(self, *, url_alias: str, selector: str) -> BackendActResult:
        page = self._require_page(url_alias)
        locator = page.locator(selector).first
        locator.click(timeout=self._timeout_ms)
        return BackendActResult(
            ok=True,
            detail=f"submitted:{selector}",
            observation=self.observe(url_alias=url_alias),
        )

    def _page_for(self, url_alias: str) -> Any:
        if url_alias not in self._pages:
            context = self._browser.new_context()
            self._contexts[url_alias] = context
            self._pages[url_alias] = context.new_page()
        return self._pages[url_alias]

    def _require_page(self, url_alias: str) -> Any:
        page = self._pages.get(url_alias)
        if page is None:
            raise ComputerBackendError(f"no open page for alias {url_alias!r}")
        return page


def build_engine(name: str) -> BrowserEngine:
    normalized = name.strip().lower()
    if normalized in {"stub", "dev", "test"}:
        return StubBrowserEngine()
    if normalized == "playwright":
        return PlaywrightBrowserEngine()
    raise ComputerBackendError(f"unknown computer engine {name!r}")


__all__ = [
    "BrowserEngine",
    "PlaywrightBrowserEngine",
    "StubBrowserEngine",
    "build_engine",
]
