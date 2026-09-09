"""DEV_ONLY transport and production config refuse."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbrain.deploy.config import ConfigurationError, DeploymentConfig, load
from cbrain.execution.transport import (
    InProcessDispatchTransport,
    ProductionDispatchRequired,
    WitnessIdentity,
    require_production_dispatch_transport,
)


def _minimal_route() -> dict:
    return {
        "tool_id": "payments.refund.v3",
        "capability": "payments.refund",
        "destination": "payments.example",
        "operation": "POST /v1/refunds",
        "credential_audience": "payments.example",
        "peer_identity": (
            "tls-spki-sha256:"
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        ),
        "allowed_parameters": ["amount"],
        "required_parameters": ["amount"],
    }


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "organisation_id": "acme",
        "agent_id": "agent-1",
        "subject_principal": "agent@acme",
        "subject_key_id": "agent-1",
        "privatevault_base_url": "https://privatevault.example",
        "database_url": "postgresql://cbrain@127.0.0.1/cbrain",
        "witness_component_id": "dispatcher",
        "witness_signer_key_id": "witness",
        "routes": {"payments.refund": _minimal_route()},
    }
    payload.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def test_in_process_marked_dev_only() -> None:
    assert InProcessDispatchTransport.DEV_ONLY is True


def test_require_production_refuses_in_process() -> None:
    transport = InProcessDispatchTransport(
        handler_runner=lambda _: None,
        witness_signer=lambda *a, **k: {},
        signing_key=object(),
        identity=WitnessIdentity(
            witness_component_id="w",
            signer_key_id="k",
            independent=False,
        ),
    )
    with pytest.raises(ProductionDispatchRequired, match="DEV_ONLY"):
        require_production_dispatch_transport(transport)


def test_require_production_refuses_non_independent() -> None:
    class Collocated:
        identity = WitnessIdentity(
            witness_component_id="w",
            signer_key_id="k",
            independent=False,
        )

    with pytest.raises(ProductionDispatchRequired, match="independent"):
        require_production_dispatch_transport(Collocated())  # type: ignore[arg-type]


def test_production_config_defaults_to_sidecar(tmp_path: Path) -> None:
    config = load(_write_config(tmp_path))
    assert config.environment == "production"
    assert config.dispatch_mode == "sidecar"
    assert config.requires_independent_sidecar is True


def test_production_refuses_in_process_dispatch_mode(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        environment="production",
        dispatch_mode="in_process",
    )
    with pytest.raises(ConfigurationError, match="DEV_ONLY"):
        load(path)


def test_development_allows_in_process(tmp_path: Path) -> None:
    config = load(
        _write_config(
            tmp_path,
            environment="development",
            dispatch_mode="in_process",
        )
    )
    assert config.dispatch_mode == "in_process"
    assert isinstance(config, DeploymentConfig)


def test_config_example_is_production_sidecar() -> None:
    config = load(Path("deploy/config.example.json"))
    assert config.environment == "production"
    assert config.dispatch_mode == "sidecar"
