from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from cbrain.simulators import (
    CRM_CONTACT_MERGE,
    CRM_CONTACT_READ,
    CRM_EMAIL_SEND,
    CRM_EXPORT_CONTACTS,
    Contact,
    CRMSimulator,
    EffectReceipt,
    SimulatorConflict,
)


def crm() -> CRMSimulator:
    return CRMSimulator(
        (
            Contact("contact-1", "Alice", "alice@example.test"),
            Contact("contact-2", "Alicia", "alicia@example.test"),
        )
    )


def test_contact_read_is_real_but_does_not_mutate_state() -> None:
    target = crm()

    receipt = target.execute(
        CRM_CONTACT_READ,
        request_id="request-read-1",
        idempotency_key="request-read-1",
        arguments={"contact_id": "contact-1"},
    )

    assert receipt.mutated is False
    assert receipt.state_version == 0
    assert receipt.result["contact"]["email"] == "alice@example.test"
    assert target.state_version == 0


def test_contact_merge_is_irreversible_and_idempotent() -> None:
    target = crm()
    arguments = {
        "source_contact_id": "contact-2",
        "target_contact_id": "contact-1",
    }

    first = target.execute(
        CRM_CONTACT_MERGE,
        request_id="request-merge-1",
        idempotency_key="merge-key-1",
        arguments=arguments,
    )
    replay = target.execute(
        CRM_CONTACT_MERGE,
        request_id="request-merge-1",
        idempotency_key="merge-key-1",
        arguments=arguments,
    )

    assert replay == first
    assert target.state_version == 1
    merged = target.snapshot()["contacts"][1]
    assert merged["status"] == "MERGED"
    assert merged["merged_into"] == "contact-1"

    with pytest.raises(SimulatorConflict, match="no longer active"):
        target.execute(
            CRM_CONTACT_MERGE,
            request_id="request-merge-2",
            idempotency_key="merge-key-2",
            arguments=arguments,
        )


def test_email_send_is_at_most_once_under_concurrency() -> None:
    target = crm()
    arguments = {
        "contact_id": "contact-1",
        "subject": "Hello",
        "body": "One externally visible message.",
    }

    def send(_: int) -> EffectReceipt:
        return target.execute(
            CRM_EMAIL_SEND,
            request_id="request-email-1",
            idempotency_key="email-key-1",
            arguments=arguments,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(send, range(32)))

    assert len({item.effect_id for item in receipts}) == 1
    snapshot = target.snapshot()
    assert snapshot["state_version"] == 1
    assert len(snapshot["sent_emails"]) == 1
    assert snapshot["sent_emails"][0]["body"] == arguments["body"]


def test_export_records_the_data_egress_effect() -> None:
    target = crm()

    receipt = target.execute(
        CRM_EXPORT_CONTACTS,
        request_id="request-export-1",
        idempotency_key="export-key-1",
        arguments={
            "format": "json",
            "fields": ["contact_id", "email"],
        },
    )

    assert receipt.mutated is True
    assert receipt.result["row_count"] == 2
    assert receipt.result["rows"][0] == {
        "contact_id": "contact-1",
        "email": "alice@example.test",
    }
    assert target.snapshot()["exports"] == [
        {
            "export_id": "export-1",
            "fields": ["contact_id", "email"],
            "format": "json",
            "row_count": 2,
        }
    ]

    snapshot = target.snapshot()
    snapshot["exports"][0]["fields"].append("name")
    assert target.snapshot()["exports"][0]["fields"] == ["contact_id", "email"]


def test_idempotency_key_cannot_be_rebound() -> None:
    target = crm()
    target.execute(
        CRM_CONTACT_READ,
        request_id="request-read-1",
        idempotency_key="shared-key",
        arguments={"contact_id": "contact-1"},
    )

    with pytest.raises(SimulatorConflict, match="another request"):
        target.execute(
            CRM_CONTACT_READ,
            request_id="request-read-2",
            idempotency_key="shared-key",
            arguments={"contact_id": "contact-2"},
        )
