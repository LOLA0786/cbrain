from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cbrain.adapters import (
    HermesCapabilityMap,
    HermesPreToolDecisionHook,
    PrivateVaultDecisionClient,
    PrivateVaultHttpTransport,
    register_hermes_hook,
)

_HOOK: HermesPreToolDecisionHook | None = None


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc

    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def _api_key_file() -> Path:
    path = Path(_required_env("CBRAIN_PRIVATEVAULT_API_KEY_FILE"))
    if not path.is_absolute():
        raise RuntimeError("CBRAIN_PRIVATEVAULT_API_KEY_FILE must be an absolute path")
    if not path.is_file():
        raise RuntimeError("CBRAIN_PRIVATEVAULT_API_KEY_FILE must reference a file")
    return path


def _headers_provider(path: Path) -> dict[str, str]:
    raw = path.read_text(encoding="utf-8")
    key = raw.strip()

    if not key or not key.startswith("pv_"):
        raise RuntimeError("PrivateVault API key file is invalid")
    if "\n" in key or "\r" in key or len(key) > 1024:
        raise RuntimeError("PrivateVault API key file contains invalid data")

    return {"X-API-Key": key}


def _load_capabilities() -> HermesCapabilityMap:
    path = Path(_required_env("CBRAIN_HERMES_CAPABILITIES_FILE"))
    if not path.is_absolute():
        raise RuntimeError("CBRAIN_HERMES_CAPABILITIES_FILE must be an absolute path")

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Unable to load Hermes capability map") from exc

    if not isinstance(value, dict):
        raise RuntimeError("Hermes capability map must be a JSON object")

    return HermesCapabilityMap(value)


def register(ctx: Any) -> None:
    """Hermes plugin entrypoint at the pinned upstream plugin contract."""

    global _HOOK

    agent_id = _required_env("CBRAIN_AGENT_ID")
    api_key_path = _api_key_file()
    transport = PrivateVaultHttpTransport(
        base_url=_required_env("CBRAIN_PRIVATEVAULT_URL"),
        headers_provider=lambda: _headers_provider(api_key_path),
        timeout_seconds=_positive_float(
            "CBRAIN_PRIVATEVAULT_TIMEOUT_SECONDS",
            5.0,
        ),
        allow_insecure_localhost=_enabled("CBRAIN_ALLOW_INSECURE_LOCALHOST"),
    )
    client = PrivateVaultDecisionClient(transport)
    _HOOK = HermesPreToolDecisionHook(
        client=client,
        agent_id=agent_id,
        capabilities=_load_capabilities(),
    )
    register_hermes_hook(ctx, _HOOK)
