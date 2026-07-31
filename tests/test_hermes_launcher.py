from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from typing import Any

import pytest

from cbrain import hermes_launcher
from cbrain.adapters.hermes import (
    HermesCapabilityMap,
    HermesPreToolDecisionHook,
)
from cbrain.adapters.privatevault import PrivateVaultDecision
from cbrain.contracts import ActionIntent


class NeverDecisionClient:
    def decide(self, action: ActionIntent) -> PrivateVaultDecision:
        raise AssertionError("decision client must not be called during startup")


class FakeManager:
    def __init__(
        self,
        *,
        plugins: dict[str, object],
        hooks: dict[str, list[Callable[..., Any]]],
    ) -> None:
        self._plugins = plugins
        self._hooks = hooks


class FakePluginsModule:
    def __init__(self, manager: FakeManager) -> None:
        self.manager = manager
        self.discovered_with_force = False

    def discover_plugins(self, *, force: bool = False) -> None:
        self.discovered_with_force = force

    def get_plugin_manager(self) -> FakeManager:
        return self.manager


class FakeMainModule:
    def __init__(self, result: int | None = None) -> None:
        self.result = result
        self.called = False

    def main(self) -> int | None:
        self.called = True
        return self.result


def cbrain_callback() -> Callable[..., Any]:
    hook = HermesPreToolDecisionHook(
        client=NeverDecisionClient(),
        agent_id="agent-1",
        capabilities=HermesCapabilityMap({"terminal": "system.command.execute"}),
    )
    return hook.pre_tool_call


def install_fake_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    plugins_module: FakePluginsModule,
    main_module: FakeMainModule | None = None,
) -> None:
    def import_module(name: str) -> Any:
        if name == "hermes_cli.plugins":
            return plugins_module
        if name == "hermes_cli.main" and main_module is not None:
            return main_module
        raise ImportError(name)

    monkeypatch.setattr(
        importlib,
        "import_module",
        import_module,
    )


def valid_plugins_module() -> FakePluginsModule:
    return FakePluginsModule(
        FakeManager(
            plugins={"cbrain_guard": object()},
            hooks={"pre_tool_call": [cbrain_callback()]},
        )
    )


def test_verifies_cbrain_is_loaded_and_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugins_module = valid_plugins_module()
    install_fake_modules(
        monkeypatch,
        plugins_module=plugins_module,
    )

    hermes_launcher.verify_required_hook()

    assert plugins_module.discovered_with_force is True


def test_missing_plugin_refuses_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugins_module = FakePluginsModule(
        FakeManager(
            plugins={},
            hooks={"pre_tool_call": [cbrain_callback()]},
        )
    )
    install_fake_modules(
        monkeypatch,
        plugins_module=plugins_module,
    )

    with pytest.raises(
        hermes_launcher.HermesStartupError,
        match="mandatory cbrain_guard plugin is not loaded",
    ):
        hermes_launcher.verify_required_hook()


def test_cbrain_not_first_refuses_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugins_module = FakePluginsModule(
        FakeManager(
            plugins={"cbrain_guard": object()},
            hooks={
                "pre_tool_call": [
                    lambda **_: None,
                    cbrain_callback(),
                ]
            },
        )
    )
    install_fake_modules(
        monkeypatch,
        plugins_module=plugins_module,
    )

    with pytest.raises(
        hermes_launcher.HermesStartupError,
        match="CBrain is not the first pre-tool guard",
    ):
        hermes_launcher.verify_required_hook()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--safe-mode"],
        ["--ignore-rules"],
        ["--ignore-user-config"],
        ["--yolo"],
        ["plugins", "disable", "cbrain_guard"],
    ],
)
def test_runtime_bypass_arguments_are_rejected(
    arguments: list[str],
) -> None:
    with pytest.raises(hermes_launcher.HermesStartupError):
        hermes_launcher.validate_runtime_arguments(arguments)


def test_main_runs_hermes_only_after_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugins_module = valid_plugins_module()
    main_module = FakeMainModule(result=17)
    install_fake_modules(
        monkeypatch,
        plugins_module=plugins_module,
        main_module=main_module,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["cbrain-hermes", "chat"],
    )

    assert hermes_launcher.main() == 17
    assert main_module.called is True


def test_main_returns_configuration_error_without_running_hermes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plugins_module = FakePluginsModule(FakeManager(plugins={}, hooks={}))
    main_module = FakeMainModule()
    install_fake_modules(
        monkeypatch,
        plugins_module=plugins_module,
        main_module=main_module,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["cbrain-hermes", "chat"],
    )

    assert hermes_launcher.main() == 78
    assert main_module.called is False
    assert "CBRAIN STARTUP REFUSED" in capsys.readouterr().err
