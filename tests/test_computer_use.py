"""Governed computer-use: observe vs act, no model URLs, DEV_ONLY backends."""

from __future__ import annotations

from typing import Any

import pytest

from cbrain.computer_use import (
    ComputerAct,
    DevOnlyBrowserBackend,
    GovernedComputerSession,
    PageAlias,
    PageCatalog,
    PageCatalogError,
    require_production_computer_backend,
)
from cbrain.computer_use.surface import ProductionComputerRequired
from cbrain.contracts import ActionIntent, ExecutionStatus, GovernedExecution
from cbrain.ports import PrivateVaultGateway
from cbrain.runtime import GovernedRuntime


class AllowingGateway:
    """Test double: always ALLOW and invoke the handler once."""

    independent_execution = False

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler,
    ) -> GovernedExecution:
        output = handler(action.arguments)
        return GovernedExecution(
            status=ExecutionStatus.EXECUTED,
            request_id=action.request_id,
            tool_executed=True,
            reason="allow",
            decision_id="dec-computer-1",
            output=output,
        )


class BlockingGateway:
    independent_execution = False

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler,
    ) -> GovernedExecution:
        return GovernedExecution(
            status=ExecutionStatus.BLOCKED,
            request_id=action.request_id,
            tool_executed=False,
            reason="policy_block",
            decision_id="dec-block-1",
        )


def _catalog() -> PageCatalog:
    return PageCatalog(
        [
            PageAlias(
                alias="banking.portal",
                url="https://banking.example/portal",
                capability="computer.navigate.banking",
                description="Retail banking portal home",
            )
        ]
    )


def test_model_view_hides_urls() -> None:
    view = _catalog().model_view()
    assert len(view) == 1
    assert "url" not in view[0]
    assert view[0]["alias"] == "banking.portal"


def test_http_non_loopback_url_refused() -> None:
    with pytest.raises(PageCatalogError, match="https"):
        PageAlias(
            alias="bad",
            url="http://banking.example/portal",
            capability="computer.navigate",
            description="bad",
        )


def test_observe_does_not_require_authorization() -> None:
    backend = DevOnlyBrowserBackend()
    backend.pages["banking.portal"] = {
        "title": "Portal",
        "accessibility_tree": "main",
        "text_excerpt": "Welcome",
    }
    session = GovernedComputerSession(
        agent_id="desk-agent",
        runtime=GovernedRuntime(AllowingGateway()),  # type: ignore[arg-type]
        catalog=_catalog(),
        backend=backend,
    )
    obs = session.observe("banking.portal")
    assert obs.title == "Portal"
    assert "url" not in obs.to_model_payload()


def test_navigate_enters_governed_runtime_and_resolves_catalog_url() -> None:
    backend = DevOnlyBrowserBackend()
    seen: dict[str, Any] = {}

    class CaptureGateway(AllowingGateway):
        def decide_and_execute(self, action, handler):
            seen["tool_name"] = action.tool_name
            seen["capability"] = action.capability
            seen["arguments"] = action.arguments
            return super().decide_and_execute(action, handler)

    session = GovernedComputerSession(
        agent_id="desk-agent",
        runtime=GovernedRuntime(CaptureGateway()),  # type: ignore[arg-type]
        catalog=_catalog(),
        backend=backend,
        request_id_factory=lambda: "req-computer-1",
    )
    result = session.act(ComputerAct(kind="navigate", page_alias="banking.portal"))
    assert result.status is ExecutionStatus.EXECUTED
    assert seen["tool_name"] == "computer.navigate"
    assert seen["capability"] == "computer.navigate.banking"
    assert seen["arguments"]["page_alias"] == "banking.portal"
    assert "url" not in seen["arguments"]
    assert backend.pages["banking.portal"]["url"] == "https://banking.example/portal"


def test_blocked_act_never_mutates_page() -> None:
    backend = DevOnlyBrowserBackend()
    session = GovernedComputerSession(
        agent_id="desk-agent",
        runtime=GovernedRuntime(BlockingGateway()),  # type: ignore[arg-type]
        catalog=_catalog(),
        backend=backend,
    )
    result = session.act(ComputerAct(kind="navigate", page_alias="banking.portal"))
    assert result.status is ExecutionStatus.BLOCKED
    assert "banking.portal" not in backend.pages


def test_unknown_alias_refused() -> None:
    session = GovernedComputerSession(
        agent_id="desk-agent",
        runtime=GovernedRuntime(AllowingGateway()),  # type: ignore[arg-type]
        catalog=_catalog(),
        backend=DevOnlyBrowserBackend(),
    )
    with pytest.raises(PageCatalogError):
        session.act(ComputerAct(kind="navigate", page_alias="unknown"))


def test_secret_shaped_type_refused() -> None:
    backend = DevOnlyBrowserBackend()
    backend.navigate(
        url_alias="banking.portal",
        url="https://banking.example/portal",
    )
    session = GovernedComputerSession(
        agent_id="desk-agent",
        runtime=GovernedRuntime(AllowingGateway()),  # type: ignore[arg-type]
        catalog=_catalog(),
        backend=backend,
    )
    result = session.act(
        ComputerAct(
            kind="type",
            page_alias="banking.portal",
            selector="#password",
            text="password=hunter2",
        )
    )
    # Handler raises → CONTROL_FAILURE / INDETERMINATE path via runtime
    assert result.status is not ExecutionStatus.EXECUTED
    assert result.tool_executed is not True


def test_production_refuses_dev_only_backend() -> None:
    with pytest.raises(ProductionComputerRequired, match="DEV_ONLY"):
        require_production_computer_backend(DevOnlyBrowserBackend())


def test_gateway_protocol_satisfied() -> None:
    gateway: PrivateVaultGateway = AllowingGateway()  # type: ignore[assignment]
    assert gateway is not None
