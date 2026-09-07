"""Run the pinned PrivateVault Agent DNA HTTP server in-process for tests.

This is the real `api.server` from the commit pinned in `upstreams.lock.json`,
served through Starlette's TestClient with real API keys, a real grants file,
a real policy file and a real execution signer. Nothing here is a double of
PrivateVault; the only test-owned parts are the files the operator would
otherwise author.

Never imported by cbrain/.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from cbrain.adapters.privatevault import HttpJsonResponse

nacl_signing = pytest.importorskip("nacl.signing")
authority_v01 = pytest.importorskip("agent_dna.authority_v01")
pytest.importorskip("agent_dna.authorize_binding")
apikeys = pytest.importorskip("agent_dna.apikeys")
testclient = pytest.importorskip("fastapi.testclient")

SigningKey = nacl_signing.SigningKey


@dataclass(frozen=True, slots=True)
class ServerKeyring:
    """Ed25519 keys the trust bundle pins. The server holds only `execution`."""

    execution: Any
    witness: Any
    closure: Any

    def trust_bundle(self, organisation_id: str) -> dict[str, Any]:
        # The pinned server signs with `keys[0]`; the execution signer must
        # come first. Witness and closure keys are pinned so the returned
        # bundle can verify the whole evidence chain.
        return {
            "spec": authority_v01.TRUST_SPEC,
            "canonicalization": authority_v01.CANONICALIZATION,
            "organisation_id": organisation_id,
            "bundle_version": 1,
            "pinned_at": "2026-07-31T11:00:00Z",
            "keys": [
                {
                    "key_id": "execution-signer-01",
                    "principal": f"execution-runtime@{organisation_id}",
                    "algorithm": "ed25519",
                    "public_key": authority_v01.encode_public_key(self.execution),
                    "usages": ["execution_authorization_signer"],
                },
                {
                    "key_id": "witness-signer-01",
                    "principal": f"dispatcher@{organisation_id}",
                    "algorithm": "ed25519",
                    "public_key": authority_v01.encode_public_key(self.witness),
                    "usages": ["dispatch_witness_signer"],
                },
                {
                    "key_id": "closure-signer-01",
                    "principal": f"closer@{organisation_id}",
                    "algorithm": "ed25519",
                    "public_key": authority_v01.encode_public_key(self.closure),
                    "usages": ["closure_signer"],
                },
            ],
        }


@dataclass(frozen=True, slots=True)
class PinnedServer:
    client: Any
    server: Any
    keyring: ServerKeyring
    organisation_id: str
    api_keys: Mapping[str, str]

    def transport(self, agent_id: str) -> TestClientTransport:
        return TestClientTransport(self.client, api_key=self.api_keys[agent_id])


class TestClientTransport:
    """`JsonTransport` over the in-process server, recording every call."""

    __test__ = False

    def __init__(self, client: Any, *, api_key: str) -> None:
        self._client = client
        self._api_key = api_key
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post_json(self, path: str, payload: Mapping[str, Any]) -> HttpJsonResponse:
        body = json.loads(json.dumps(payload))
        self.calls.append((path, body))
        response = self._client.post(
            path,
            json=body,
            headers={"X-API-Key": self._api_key},
        )
        return HttpJsonResponse(status_code=response.status_code, body=response.json())

    def paths(self) -> list[str]:
        return [path for path, _ in self.calls]


@contextmanager
def pinned_privatevault_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    organisation_id: str,
    grants: Mapping[str, tuple[str, ...]],
    blocked_capabilities: tuple[str, ...] = (),
) -> Iterator[PinnedServer]:
    """Start the pinned server with one full-scope API key per agent.

    `grants` maps agent_id -> capabilities that agent is granted. Any other
    capability is not covered by a grant and the pinned engine answers
    REQUIRE_APPROVAL. `blocked_capabilities` are written into a policy file
    with outcome `block`.
    """
    keyring = ServerKeyring(
        execution=SigningKey.generate(),
        witness=SigningKey.generate(),
        closure=SigningKey.generate(),
    )

    api_keys: dict[str, str] = {}
    registry: dict[str, dict[str, str]] = {}
    for agent_id in grants:
        issued = apikeys.generate_key(agent_id, "full")
        api_keys[agent_id] = issued["key"]
        registry[issued["hash"]] = {"name": issued["name"], "scope": "full"}
    (tmp_path / "keys.json").write_text(json.dumps(registry), encoding="utf-8")

    (tmp_path / "grants.json").write_text(
        json.dumps(
            [
                {
                    "agent_id": agent_id,
                    "capability": capability,
                    "granted_by": "test-fixture",
                }
                for agent_id, capabilities in grants.items()
                for capability in capabilities
            ]
        ),
        encoding="utf-8",
    )

    (tmp_path / "policy.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "policies": [
                    {
                        "id": f"block-{index}",
                        "capability": capability,
                        "outcome": "block",
                        "reason": f"{capability} is blocked by policy",
                    }
                    for index, capability in enumerate(blocked_capabilities)
                ],
            }
        ),
        encoding="utf-8",
    )

    (tmp_path / "execution-signer.key").write_bytes(bytes(keyring.execution))
    (tmp_path / "trust-bundle.json").write_text(
        json.dumps(keyring.trust_bundle(organisation_id)),
        encoding="utf-8",
    )

    monkeypatch.setenv("PV_DB_PATH", str(tmp_path / "privatevault.db"))
    monkeypatch.setenv("PV_API_KEYS_FILE", str(tmp_path / "keys.json"))
    monkeypatch.setenv("PV_GRANTS_FILE", str(tmp_path / "grants.json"))
    monkeypatch.setenv("PV_POLICY_FILE", str(tmp_path / "policy.json"))
    monkeypatch.setenv(
        "PV_EXECUTION_SIGNER_KEY", str(tmp_path / "execution-signer.key")
    )
    monkeypatch.setenv("PV_TRUST_BUNDLE", str(tmp_path / "trust-bundle.json"))
    monkeypatch.delenv("PV_ALLOW_NO_AUTH", raising=False)

    import api.server as server

    server._pv_signer_cache.clear()
    server = importlib.reload(server)
    server._pv_signer_cache.clear()

    with testclient.TestClient(server.app) as client:
        yield PinnedServer(
            client=client,
            server=server,
            keyring=keyring,
            organisation_id=organisation_id,
            api_keys=api_keys,
        )

    server._pv_signer_cache.clear()


__all__ = [
    "PinnedServer",
    "ServerKeyring",
    "TestClientTransport",
    "pinned_privatevault_server",
]
