"""Session that splits observe (read) from act (governed)."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from cbrain.contracts import ActionIntent, GovernedExecution
from cbrain.runtime import GovernedRuntime

from .catalog import PageCatalog, PageCatalogError
from .surface import (
    BackendObservation,
    ComputerBackend,
    ComputerBackendError,
)

ActKind = Literal["navigate", "click", "type", "submit"]


class ComputerUseError(ValueError):
    """Computer-use session input is invalid."""


@dataclass(frozen=True, slots=True)
class ComputerObservation:
    """Sanitized page snapshot safe for model context (no URL, no secrets)."""

    url_alias: str
    title: str
    accessibility_tree: str
    text_excerpt: str

    def to_model_payload(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "url_alias": self.url_alias,
                "title": self.title,
                "accessibility_tree": self.accessibility_tree,
                "text_excerpt": self.text_excerpt,
            }
        )


@dataclass(frozen=True, slots=True)
class ComputerAct:
    """Consequential browser act. Destination comes from catalog, not model."""

    kind: ActKind
    page_alias: str
    selector: str | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"navigate", "click", "type", "submit"}:
            raise ComputerUseError(f"unsupported act kind {self.kind!r}")
        if not isinstance(self.page_alias, str) or not self.page_alias.strip():
            raise ComputerUseError("page_alias must be non-empty")
        if self.kind in {"click", "type", "submit"} and not (
            isinstance(self.selector, str) and self.selector.strip()
        ):
            raise ComputerUseError(f"{self.kind} requires a non-empty selector")
        if self.kind == "type" and not isinstance(self.text, str):
            raise ComputerUseError("type requires text")
        if self.kind == "navigate" and self.selector is not None:
            raise ComputerUseError("navigate does not take a selector")


class GovernedComputerSession:
    """Personal-computer surface for agents: observe freely, act under PV.

    Observation never dials a new destination; it only reads an already-open
    alias. Navigate / click / type / submit each become an ``ActionIntent`` and
    must pass ``GovernedRuntime`` before the backend mutates the page.
    """

    FRAMEWORK = "cbrain.computer_use"

    def __init__(
        self,
        *,
        agent_id: str,
        runtime: GovernedRuntime,
        catalog: PageCatalog,
        backend: ComputerBackend,
        request_id_factory: Any | None = None,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ComputerUseError("agent_id must be non-empty")
        self._agent_id = agent_id
        self._runtime = runtime
        self._catalog = catalog
        self._backend = backend
        self._request_id_factory = request_id_factory or (
            lambda: f"computer-{uuid.uuid4()}"
        )

    def observe(self, page_alias: str) -> ComputerObservation:
        """Read-only. Does not authorize; does not navigate."""

        if not self._catalog.contains(page_alias):
            raise PageCatalogError(f"unknown page alias {page_alias!r}")
        try:
            raw = self._backend.observe(url_alias=page_alias)
        except ComputerBackendError as exc:
            raise ComputerUseError(str(exc)) from exc
        return _to_observation(raw)

    def act(self, act: ComputerAct) -> GovernedExecution:
        """Consequential act: catalog resolve → ActionIntent → GovernedRuntime."""

        page = self._catalog.resolve(act.page_alias)
        arguments = _act_arguments(act, page.alias)
        action = ActionIntent.capture(
            request_id=str(self._request_id_factory()),
            idempotency_key=f"computer:{act.kind}:{page.alias}:{uuid.uuid4()}",
            agent_id=self._agent_id,
            framework=self.FRAMEWORK,
            tool_name=f"computer.{act.kind}",
            capability=page.capability,
            arguments=arguments,
            context={"page_alias": page.alias, "act_kind": act.kind},
            evidence={},
            timestamp=time.time(),
        )

        def handler(args: Mapping[str, Any]) -> Mapping[str, Any]:
            # Handler only sees catalog-resolved alias; URL injected here from
            # operator catalog, never from model arguments.
            return _run_backend_act(
                self._backend,
                act=act,
                resolved_url=page.url,
            )

        return self._runtime.execute(action, handler)

    def pages_for_model(self) -> tuple[Mapping[str, str], ...]:
        return self._catalog.model_view()


def _act_arguments(act: ComputerAct, page_alias: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": act.kind,
        "page_alias": page_alias,
    }
    if act.selector is not None:
        payload["selector"] = act.selector
    if act.kind == "type" and act.text is not None:
        payload["text"] = act.text
    return payload


def _run_backend_act(
    backend: ComputerBackend,
    *,
    act: ComputerAct,
    resolved_url: str,
) -> Mapping[str, Any]:
    if act.kind == "navigate":
        result = backend.navigate(url_alias=act.page_alias, url=resolved_url)
    elif act.kind == "click":
        assert act.selector is not None
        result = backend.click(url_alias=act.page_alias, selector=act.selector)
    elif act.kind == "type":
        assert act.selector is not None and act.text is not None
        result = backend.type_text(
            url_alias=act.page_alias,
            selector=act.selector,
            text=act.text,
        )
    elif act.kind == "submit":
        assert act.selector is not None
        result = backend.submit(url_alias=act.page_alias, selector=act.selector)
    else:
        raise ComputerUseError(f"unsupported act kind {act.kind!r}")

    observation = (
        _to_observation(result.observation).to_model_payload()
        if result.observation is not None
        else {}
    )
    return MappingProxyType(
        {
            "ok": result.ok,
            "detail": result.detail,
            "observation": dict(observation),
        }
    )


def _to_observation(raw: BackendObservation) -> ComputerObservation:
    return ComputerObservation(
        url_alias=raw.url_alias,
        title=raw.title,
        accessibility_tree=raw.accessibility_tree,
        text_excerpt=raw.text_excerpt,
    )


__all__ = [
    "ComputerAct",
    "ComputerObservation",
    "ComputerUseError",
    "GovernedComputerSession",
]
