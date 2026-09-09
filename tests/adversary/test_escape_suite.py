"""Formal adversary suite: zero unauthorized target requests.

Each case pins a named refusal. The suite fails closed if any case is
missing from the repository. Evidence levels are recorded in docs/claims.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# (claim_id, evidence_level, test_node_id_or_path)
ADVERSARY_CASES = (
    (
        "forged_trust_bundle",
        "unit",
        "tests/test_independent_sidecar.py::"
        "test_forged_request_trust_bundle_never_reaches_credentials_or_target",
    ),
    (
        "first_mint_body_swap",
        "real-server",
        "tests/test_privatevault_http_contract.py::"
        "test_first_mint_refuses_action_wire_mismatch_via_direct_http",
    ),
    (
        "decide_inconsistent_wire",
        "real-server",
        "tests/test_privatevault_http_contract.py::"
        "test_decide_refuses_inconsistent_action_and_wire_digests",
    ),
    (
        "foreign_agent_record",
        "real-server",
        "tests/test_privatevault_http_contract.py::"
        "test_another_agents_record_cannot_mint",
    ),
    (
        "changed_dispatch_redirect",
        "real-server",
        "tests/test_privatevault_http_contract.py::"
        "test_changed_dispatch_after_decision_cannot_mint",
    ),
    (
        "unapproved_review_zero_effects",
        "real-server",
        "tests/test_privatevault_http_contract.py::"
        "test_unapproved_review_cannot_mint_and_never_calls_authorize",
    ),
    (
        "worker_without_credentials",
        "unit",
        "tests/test_isolation_boundary.py::"
        "test_worker_without_credentials_cannot_authorize_write",
    ),
    (
        "wrong_role_approval",
        "unit",
        "tests/test_durable_review.py::"
        "test_wrong_role_approval_refused_and_consume_once",
    ),
    (
        "concurrent_lease",
        "unit",
        "tests/test_durable_review.py::test_competing_workers_cannot_both_hold_lease",
    ),
)


def _node_path(node_id: str) -> Path:
    relative = node_id.split("::", 1)[0]
    return ROOT / relative


@pytest.mark.parametrize("claim_id,level,node_id", ADVERSARY_CASES)
def test_adversary_case_is_pinned(claim_id: str, level: str, node_id: str) -> None:
    path = _node_path(node_id)
    assert path.is_file(), f"{claim_id}: missing {path}"
    name = node_id.rsplit("::", 1)[-1]
    source = path.read_text(encoding="utf-8")
    assert f"def {name}(" in source, f"{claim_id}: missing test {name}"
    assert level in {
        "unit",
        "real-server",
        "independent-process",
        "ci-isolation",
        "live",
    }


def test_adversary_suite_covers_required_threats() -> None:
    ids = {claim_id for claim_id, _, _ in ADVERSARY_CASES}
    required = {
        "forged_trust_bundle",
        "first_mint_body_swap",
        "foreign_agent_record",
        "changed_dispatch_redirect",
        "worker_without_credentials",
        "wrong_role_approval",
        "concurrent_lease",
    }
    assert required <= ids
