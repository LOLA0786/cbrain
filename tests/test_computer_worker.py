"""Out-of-process computer worker + remote backend + FoundationAgent tools."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest

from cbrain.agent import (
    AgentProfile,
    FoundationAgent,
    RunInput,
    RunStatus,
    ToolRegistry,
)
from cbrain.computer_use import (
    ComputerToolBridge,
    PageAlias,
    PageCatalog,
    RemoteComputerBackend,
    computer_tools,
    require_production_computer_backend,
)
from cbrain.computer_use.engines import StubBrowserEngine
from cbrain.computer_use.surface import ComputerBackendError
from cbrain.computer_use.worker import serve_computer_worker
from cbrain.contracts import ActionIntent, ExecutionStatus, GovernedExecution
from cbrain.models import CompletionRequest, ModelRouter, TextOutput, ToolCall
from cbrain.runtime import GovernedRuntime


@pytest.fixture
def worker_server() -> Iterator[tuple[str, str]]:
    token = "test-computer-token"
    engine = StubBrowserEngine()
    server = serve_computer_worker(
        host="127.0.0.1",
        port=0,
        token=token,
        engine=engine,
    )
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1/computer", token
    finally:
        server.shutdown()
        server.server_close()


class SequenceModel:
    def __init__(self, outputs: list[Any]) -> None:
        self._outputs = list(outputs)

    @property
    def provider(self) -> str:
        return "sequence"

    @property
    def model(self) -> str:
        return "sequence-v1"

    def complete(self, request: CompletionRequest) -> TextOutput | ToolCall:
        if not self._outputs:
            raise RuntimeError("no scripted outputs remain")
        return self._outputs.pop(0)


class AllowGateway:
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
            reason="test_allow",
            output=output,
        )


def test_remote_backend_is_production_grade(worker_server: tuple[str, str]) -> None:
    endpoint, token = worker_server
    backend = RemoteComputerBackend(
        endpoint=endpoint,
        token=token,
        allow_http_internal=True,
    )
    require_production_computer_backend(backend)
    assert backend.DEV_ONLY is False

    result = backend.navigate(
        url_alias="portal",
        url="https://banking.example/portal",
    )
    assert result.ok is True
    assert result.observation is not None
    assert result.observation.screenshot_sha256 is not None
    assert result.observation.screenshot_sha256.startswith("sha256:")
    obs = backend.observe(url_alias="portal")
    assert "url" not in obs.to_wire()
    assert obs.url_alias == "portal"


def test_remote_backend_refuses_missing_token() -> None:
    with pytest.raises(ComputerBackendError, match="token"):
        RemoteComputerBackend(
            endpoint="http://127.0.0.1:9/v1/computer",
            token="",
            allow_http_internal=True,
        )


def test_unauthorized_worker_call_fails(worker_server: tuple[str, str]) -> None:
    endpoint, _token = worker_server
    backend = RemoteComputerBackend(
        endpoint=endpoint,
        token="wrong-token",
        allow_http_internal=True,
    )
    with pytest.raises(ComputerBackendError):
        backend.navigate(url_alias="portal", url="https://example.com")


def test_foundation_tools_hide_urls_and_use_catalog(
    worker_server: tuple[str, str],
) -> None:
    endpoint, token = worker_server
    backend = RemoteComputerBackend(
        endpoint=endpoint,
        token=token,
        allow_http_internal=True,
    )
    catalog = PageCatalog(
        [
            PageAlias(
                alias="banking.portal",
                url="https://banking.example/portal",
                capability="computer.navigate.banking",
                description="Banking portal",
            )
        ]
    )
    bridge = ComputerToolBridge(catalog=catalog, backend=backend)
    tools = computer_tools()
    assert {tool.name for tool in tools} == {
        "computer_observe",
        "computer_navigate",
        "computer_click",
        "computer_type",
        "computer_submit",
    }

    out = bridge.handlers()["computer_navigate"]({"page_alias": "banking.portal"})
    assert out["ok"] is True
    assert "url" not in out
    assert out["observation"]["url_alias"] == "banking.portal"
    assert "screenshot_sha256" in out["observation"]


def test_foundation_agent_runs_computer_navigate(
    worker_server: tuple[str, str],
) -> None:
    endpoint, token = worker_server
    backend = RemoteComputerBackend(
        endpoint=endpoint,
        token=token,
        allow_http_internal=True,
    )
    catalog = PageCatalog(
        [
            PageAlias(
                alias="banking.portal",
                url="https://banking.example/portal",
                capability="computer.navigate.banking",
                description="Banking portal",
            )
        ]
    )
    bridge = ComputerToolBridge(catalog=catalog, backend=backend)
    tools = ToolRegistry(computer_tools())
    profile = AgentProfile(
        agent_id="desk-agent",
        instructions="Use computer tools on page aliases only.",
        model_route="local",
        permitted_tools=frozenset({"computer_navigate", "computer_observe"}),
        max_model_turns=4,
        max_tool_calls=2,
        timeout_seconds=30.0,
    )
    model = SequenceModel(
        [
            ToolCall.capture(
                call_id="call-1",
                name="computer_navigate",
                arguments={"page_alias": "banking.portal"},
            ),
            TextOutput(text="Opened the portal."),
        ]
    )
    agent = FoundationAgent(
        profile=profile,
        runtime=GovernedRuntime(AllowGateway()),
        model_router=ModelRouter({profile.model_route: model}),
        tools=tools,
        handlers={name: bridge.handlers()[name] for name in profile.permitted_tools},
        clock=lambda: 1_700_000_000.0,
        run_id_factory=lambda: "run-computer-001",
        request_id_factory=lambda run_id, step: f"{run_id}-step-{step}",
    )
    result = agent.run(RunInput(task="Open the banking portal"))
    assert result.status is RunStatus.COMPLETED
