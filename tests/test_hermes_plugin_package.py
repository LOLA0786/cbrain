from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from integrations.hermes import cbrain_guard


class RecordingContext:
    def __init__(self) -> None:
        self.hooks: list[tuple[str, Callable[..., Any]]] = []

    def register_hook(
        self,
        hook_name: str,
        callback: Callable[..., Any],
    ) -> None:
        self.hooks.append((hook_name, callback))


def configure_plugin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    secret = tmp_path / "privatevault-api-key"
    secret.write_text("pv_test_only\n", encoding="utf-8")

    capabilities = tmp_path / "capabilities.json"
    capabilities.write_text(
        json.dumps({"terminal": "system.command.execute"}),
        encoding="utf-8",
    )

    monkeypatch.setenv("CBRAIN_AGENT_ID", "agent-1")
    monkeypatch.setenv(
        "CBRAIN_PRIVATEVAULT_URL",
        "https://privatevault.example",
    )
    monkeypatch.setenv(
        "CBRAIN_PRIVATEVAULT_API_KEY_FILE",
        str(secret),
    )
    monkeypatch.setenv(
        "CBRAIN_HERMES_CAPABILITIES_FILE",
        str(capabilities),
    )
    return secret, capabilities


def test_registers_pre_tool_hook_with_valid_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_plugin(monkeypatch, tmp_path)
    context = RecordingContext()

    cbrain_guard.register(context)

    assert len(context.hooks) == 1
    assert context.hooks[0][0] == "pre_tool_call"

    directive = context.hooks[0][1](
        tool_name="terminal",
        args={},
        session_id="",
        tool_call_id="",
    )
    assert directive == {
        "action": "block",
        "message": "CBRAIN_CONTROL_FAILURE",
    }


def test_missing_required_configuration_refuses_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_plugin(monkeypatch, tmp_path)
    monkeypatch.delenv("CBRAIN_AGENT_ID")
    context = RecordingContext()

    with pytest.raises(RuntimeError, match="CBRAIN_AGENT_ID is required"):
        cbrain_guard.register(context)

    assert context.hooks == []


def test_relative_secret_path_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_plugin(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "CBRAIN_PRIVATEVAULT_API_KEY_FILE",
        "relative/privatevault-key",
    )

    with pytest.raises(RuntimeError, match="must be an absolute path"):
        cbrain_guard.register(RecordingContext())


def test_malformed_capability_file_refuses_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _secret, capabilities = configure_plugin(monkeypatch, tmp_path)
    capabilities.write_text("{not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unable to load"):
        cbrain_guard.register(RecordingContext())


def test_secret_provider_uses_privatevault_x_api_key_header(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "privatevault-api-key"
    secret.write_text("pv_test_only\n", encoding="utf-8")

    assert cbrain_guard._headers_provider(secret) == {"X-API-Key": "pv_test_only"}


def test_secret_provider_rejects_non_privatevault_key(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "privatevault-api-key"
    secret.write_text("wrong-prefix", encoding="utf-8")

    with pytest.raises(RuntimeError, match="API key file is invalid"):
        cbrain_guard._headers_provider(secret)
