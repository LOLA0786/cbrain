"""FoundationAgent tool definitions and handlers for governed computer use.

FoundationAgent already wraps every tool in ``GovernedRuntime``. These handlers
must NOT call ``GovernedComputerSession.act`` (that would double-authorize).
They resolve operator-owned URLs from ``PageCatalog`` and call the backend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from cbrain.agent.tools import GovernedTool

from .catalog import PageCatalog, PageCatalogError
from .session import ComputerUseError
from .surface import ComputerBackend, ComputerBackendError

_PAGE_ALIAS_SCHEMA = {
    "type": "object",
    "properties": {
        "page_alias": {
            "type": "string",
            "description": "Operator-owned page alias (never a raw URL)",
        }
    },
    "required": ["page_alias"],
    "additionalProperties": False,
}

_SELECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "page_alias": {
            "type": "string",
            "description": "Operator-owned page alias (never a raw URL)",
        },
        "selector": {
            "type": "string",
            "description": "Stable CSS/accessibility selector on the page",
        },
    },
    "required": ["page_alias", "selector"],
    "additionalProperties": False,
}

_TYPE_SCHEMA = {
    "type": "object",
    "properties": {
        "page_alias": {
            "type": "string",
            "description": "Operator-owned page alias (never a raw URL)",
        },
        "selector": {
            "type": "string",
            "description": "Stable CSS/accessibility selector on the page",
        },
        "text": {
            "type": "string",
            "description": "Non-secret text to type; credentials are refused",
        },
    },
    "required": ["page_alias", "selector", "text"],
    "additionalProperties": False,
}


def computer_tools() -> tuple[GovernedTool, ...]:
    """Canonical computer tool pack for FoundationAgent profiles."""

    return (
        GovernedTool(
            name="computer_observe",
            capability="computer.observe",
            description=(
                "Read-only snapshot of an already-open page alias "
                "(title, accessibility tree, text excerpt, screenshot digest). "
                "Never returns URLs or credentials."
            ),
            input_schema=_PAGE_ALIAS_SCHEMA,
        ),
        GovernedTool(
            name="computer_navigate",
            capability="computer.navigate",
            description=(
                "Open an operator-catalog page by alias. The model never supplies "
                "the destination URL."
            ),
            input_schema=_PAGE_ALIAS_SCHEMA,
        ),
        GovernedTool(
            name="computer_click",
            capability="computer.click",
            description="Click an element on an open page alias.",
            input_schema=_SELECTOR_SCHEMA,
        ),
        GovernedTool(
            name="computer_type",
            capability="computer.type",
            description=(
                "Type non-secret text into a field on an open page alias. "
                "Secret-shaped strings are refused."
            ),
            input_schema=_TYPE_SCHEMA,
        ),
        GovernedTool(
            name="computer_submit",
            capability="computer.submit",
            description="Submit a form or confirm a primary action on a page alias.",
            input_schema=_SELECTOR_SCHEMA,
        ),
    )


class ComputerToolBridge:
    """Handlers for FoundationAgent: FA owns decide; bridge owns catalog+backend."""

    def __init__(
        self,
        *,
        catalog: PageCatalog,
        backend: ComputerBackend,
    ) -> None:
        self._catalog = catalog
        self._backend = backend

    def handlers(self) -> dict[str, Callable[[Mapping[str, Any]], Any]]:
        return {
            "computer_observe": self._observe,
            "computer_navigate": self._navigate,
            "computer_click": self._click,
            "computer_type": self._type,
            "computer_submit": self._submit,
        }

    def pages_for_model(self) -> tuple[Mapping[str, str], ...]:
        return self._catalog.model_view()

    def _observe(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        alias = _alias(arguments)
        self._catalog.resolve(alias)
        try:
            observation = self._backend.observe(url_alias=alias)
        except ComputerBackendError as exc:
            raise ComputerUseError(str(exc)) from exc
        return MappingProxyType(_observation_payload(observation.to_wire()))

    def _navigate(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        alias = _alias(arguments)
        page = self._catalog.resolve(alias)
        try:
            result = self._backend.navigate(url_alias=alias, url=page.url)
        except ComputerBackendError as exc:
            raise ComputerUseError(str(exc)) from exc
        return MappingProxyType(_act_payload(result.to_wire()))

    def _click(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        alias = _alias(arguments)
        selector = _selector(arguments)
        self._catalog.resolve(alias)
        try:
            result = self._backend.click(url_alias=alias, selector=selector)
        except ComputerBackendError as exc:
            raise ComputerUseError(str(exc)) from exc
        return MappingProxyType(_act_payload(result.to_wire()))

    def _type(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        alias = _alias(arguments)
        selector = _selector(arguments)
        text = arguments.get("text")
        if not isinstance(text, str):
            raise ComputerUseError("text must be a string")
        self._catalog.resolve(alias)
        try:
            result = self._backend.type_text(
                url_alias=alias, selector=selector, text=text
            )
        except ComputerBackendError as exc:
            raise ComputerUseError(str(exc)) from exc
        return MappingProxyType(_act_payload(result.to_wire()))

    def _submit(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        alias = _alias(arguments)
        selector = _selector(arguments)
        self._catalog.resolve(alias)
        try:
            result = self._backend.submit(url_alias=alias, selector=selector)
        except ComputerBackendError as exc:
            raise ComputerUseError(str(exc)) from exc
        return MappingProxyType(_act_payload(result.to_wire()))


def _alias(arguments: Mapping[str, Any]) -> str:
    alias = arguments.get("page_alias")
    if not isinstance(alias, str) or not alias.strip():
        raise PageCatalogError("page_alias must be non-empty")
    return alias


def _selector(arguments: Mapping[str, Any]) -> str:
    selector = arguments.get("selector")
    if not isinstance(selector, str) or not selector.strip():
        raise ComputerUseError("selector must be non-empty")
    return selector


def _observation_payload(wire: Mapping[str, Any]) -> dict[str, Any]:
    # Never forward a URL field even if a buggy worker includes one.
    return {
        key: value
        for key, value in wire.items()
        if key
        in {
            "title",
            "url_alias",
            "accessibility_tree",
            "text_excerpt",
            "screenshot_sha256",
            "screenshot_media_type",
        }
    }


def _act_payload(wire: Mapping[str, Any]) -> dict[str, Any]:
    observation = wire.get("observation")
    payload: dict[str, Any] = {
        "ok": bool(wire.get("ok", False)),
        "detail": str(wire.get("detail", "")),
    }
    if isinstance(observation, Mapping):
        payload["observation"] = _observation_payload(observation)
    return payload


__all__ = ["ComputerToolBridge", "computer_tools"]
