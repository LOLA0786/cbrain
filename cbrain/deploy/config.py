"""Deployment configuration.

One file, validated at startup, that assembles a governed runtime from
operator-authored settings. Anything ambiguous fails here rather than at the
first consequential action.

Configuration is TOML-free on purpose: JSON only, so the config that produced
a deployment can be digested and recorded alongside the evidence it governs.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cbrain.execution.planner import ToolRoute


class ConfigurationError(RuntimeError):
    """The deployment configuration is unusable. Never start on this."""


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    organisation_id: str
    agent_id: str
    subject_principal: str
    subject_key_id: str
    privatevault_base_url: str
    database_url: str
    routes: Mapping[str, ToolRoute]
    witness_component_id: str
    witness_signer_key_id: str
    request_timeout_seconds: float
    config_digest: str

    @property
    def api_key(self) -> str:
        """Read from the environment, never from the config file."""
        key = os.environ.get("PV_API_KEY", "")
        if not key:
            raise ConfigurationError(
                "PV_API_KEY is not set; refusing to start unauthenticated"
            )
        return key


def load(path: str | Path) -> DeploymentConfig:
    source = Path(path)

    try:
        raw = json.loads(source.read_text())
    except OSError as exc:
        raise ConfigurationError(f"cannot read config: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"config is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError("config root must be an object")

    digest = "sha256:" + hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    routes = _load_routes(raw.get("routes"))

    if not routes:
        raise ConfigurationError(
            "no routes configured; an agent with no routes can do nothing"
        )

    base_url = _text(raw, "privatevault_base_url")

    if base_url.startswith("http://") and not _localhost(base_url):
        raise ConfigurationError(
            "privatevault_base_url must be https outside localhost"
        )

    return DeploymentConfig(
        organisation_id=_text(raw, "organisation_id"),
        agent_id=_text(raw, "agent_id"),
        subject_principal=_text(raw, "subject_principal"),
        subject_key_id=_text(raw, "subject_key_id"),
        privatevault_base_url=base_url,
        database_url=_env_or_text(raw, "database_url", "PV_DATABASE_URL"),
        routes=routes,
        witness_component_id=_text(raw, "witness_component_id"),
        witness_signer_key_id=_text(raw, "witness_signer_key_id"),
        request_timeout_seconds=float(
            raw.get("request_timeout_seconds", 5.0)
        ),
        config_digest=digest,
    )


def _load_routes(value: object) -> dict[str, ToolRoute]:
    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ConfigurationError("routes must be an object")

    routes: dict[str, ToolRoute] = {}

    for tool_name, entry in value.items():
        if not isinstance(entry, dict):
            raise ConfigurationError(f"route {tool_name!r} must be an object")

        try:
            routes[tool_name] = ToolRoute(
                tool_id=entry["tool_id"],
                capability=entry["capability"],
                destination=entry["destination"],
                operation=entry["operation"],
                credential_audience=entry["credential_audience"],
                peer_identity=entry["peer_identity"],
                allowed_parameters=tuple(entry.get("allowed_parameters", ())),
                required_parameters=tuple(
                    entry.get("required_parameters", ())
                ),
            )
        except (KeyError, ValueError) as exc:
            raise ConfigurationError(
                f"route {tool_name!r} is invalid: {exc}"
            ) from exc

    return routes


def _text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be non-empty text")
    return value


def _env_or_text(raw: Mapping[str, Any], key: str, env: str) -> str:
    """Prefer the environment, so secrets need not live in the file."""
    from_env = os.environ.get(env, "")
    if from_env:
        return from_env
    return _text(raw, key)


def _localhost(url: str) -> bool:
    return url.startswith(("http://localhost", "http://127.0.0.1"))


__all__ = ["ConfigurationError", "DeploymentConfig", "load"]
