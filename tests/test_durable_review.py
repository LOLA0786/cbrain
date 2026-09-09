"""Durable REVIEW lease/crash semantics."""

from __future__ import annotations

from cbrain import ActionIntent
from cbrain.company.approval import ApprovalInboxError, ApprovalPrincipal, ApprovalRole
from cbrain.company.durable_approval import DurableReviewStore, ReviewState


def intent(request_id: str = "req-review-1") -> ActionIntent:
    return ActionIntent.capture(
        request_id=request_id,
        idempotency_key=request_id,
        agent_id="buyer-agent",
        framework="test",
        tool_name="procurement.po.create",
        capability="procurement.po.create",
        timestamp=1_700_000_000.0,
        arguments={"vendor_id": "V-001", "quantity_tonnes": 40},
    )


def store(tmp_path, clock):
    return DurableReviewStore(
        tmp_path / "reviews.sqlite",
        organisation_id="steel.example",
        allowed_approvers={ApprovalRole.BUYER_LEAD: {"buyer-lead-1"}},
        max_ttl_seconds=3600,
        max_lease_seconds=30,
        clock=clock,
    )


def test_competing_workers_cannot_both_hold_lease(tmp_path):
    now = {"t": 100.0}
    reviews = store(tmp_path, lambda: now["t"])
    action = intent()
    reviews.park(
        action, digest="sha256:" + ("a" * 64), required_role=ApprovalRole.BUYER_LEAD
    )

    first = reviews.claim_lease(
        action.request_id, worker_id="worker-a", lease_seconds=10
    )
    assert first.state is ReviewState.LEASED
    assert first.lease_owner == "worker-a"

    try:
        reviews.claim_lease(action.request_id, worker_id="worker-b", lease_seconds=10)
        raise AssertionError("second worker must not steal an active lease")
    except ApprovalInboxError as exc:
        assert "another worker" in str(exc)

    reviews.close()


def test_lease_expiry_returns_parked_and_is_not_proof_of_non_execution(tmp_path):
    now = {"t": 100.0}
    reviews = store(tmp_path, lambda: now["t"])
    action = intent()
    digest = "sha256:" + ("b" * 64)
    reviews.park(action, digest=digest, required_role=ApprovalRole.BUYER_LEAD)
    reviews.claim_lease(action.request_id, worker_id="worker-a", lease_seconds=5)

    now["t"] = 106.0  # lease expired
    record = reviews.get(action.request_id)
    assert record is not None
    assert record.state is ReviewState.PARKED
    assert record.lease_owner is None
    events = [item["event"] for item in reviews.events(action.request_id)]
    assert "lease_expired" in events
    # History retained: park still happened.
    assert "parked" in events
    assert "leased" in events

    # Another worker may re-lease; expiry never means "never parked".
    again = reviews.claim_lease(
        action.request_id, worker_id="worker-b", lease_seconds=5
    )
    assert again.state is ReviewState.LEASED
    assert again.lease_owner == "worker-b"
    reviews.close()


def test_wrong_role_approval_refused_and_consume_once(tmp_path):
    now = {"t": 200.0}
    reviews = store(tmp_path, lambda: now["t"])
    action = intent("req-review-2")
    digest = "sha256:" + ("c" * 64)
    reviews.park(action, digest=digest, required_role=ApprovalRole.BUYER_LEAD)

    try:
        reviews.approve(
            action.request_id,
            principal=ApprovalPrincipal("counsel-1", ApprovalRole.COUNSEL),
        )
        raise AssertionError("wrong role must not approve")
    except ApprovalInboxError:
        pass

    approved = reviews.approve(
        action.request_id,
        principal=ApprovalPrincipal("buyer-lead-1", ApprovalRole.BUYER_LEAD),
        ttl_seconds=60,
    )
    assert approved.state is ReviewState.APPROVED

    assert reviews.consume_if_approved(
        action, digest=digest, required_role=ApprovalRole.BUYER_LEAD
    )
    assert not reviews.consume_if_approved(
        action, digest=digest, required_role=ApprovalRole.BUYER_LEAD
    )
    final = reviews.get(action.request_id)
    assert final is not None
    assert final.state is ReviewState.CONSUMED
    reviews.close()


def test_expired_approval_cannot_dispatch(tmp_path):
    now = {"t": 300.0}
    reviews = store(tmp_path, lambda: now["t"])
    action = intent("req-review-3")
    digest = "sha256:" + ("d" * 64)
    reviews.park(action, digest=digest, required_role=ApprovalRole.BUYER_LEAD)
    reviews.approve(
        action.request_id,
        principal=ApprovalPrincipal("buyer-lead-1", ApprovalRole.BUYER_LEAD),
        ttl_seconds=10,
    )
    now["t"] = 320.0
    assert not reviews.consume_if_approved(
        action, digest=digest, required_role=ApprovalRole.BUYER_LEAD
    )
    record = reviews.get(action.request_id)
    assert record is not None
    assert record.state is ReviewState.APPROVAL_EXPIRED
    reviews.close()
