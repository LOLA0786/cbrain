"""Mutable CRM target used by governed-execution scenarios."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any

from .contracts import (
    EffectReceipt,
    JsonObject,
    SimulatorBusinessError,
    SimulatorConflict,
    SimulatorContractError,
    SimulatorNotFound,
    StoredEffect,
    canonical_object,
    capture_request,
    effect_identifier,
    exact_fields,
    required_text,
    restore_object,
)

CRM_CONTACT_READ = "crm.contact.read"
CRM_CONTACT_MERGE = "crm.contact.merge"
CRM_EMAIL_SEND = "crm.email.send"
CRM_EXPORT_CONTACTS = "crm.export.contacts"
CRM_CAPABILITIES = frozenset(
    {
        CRM_CONTACT_READ,
        CRM_CONTACT_MERGE,
        CRM_EMAIL_SEND,
        CRM_EXPORT_CONTACTS,
    }
)


@dataclass(frozen=True, slots=True)
class Contact:
    contact_id: str
    name: str
    email: str
    status: str = "ACTIVE"
    merged_into: str | None = None

    def __post_init__(self) -> None:
        required_text(self.contact_id, "contact_id")
        required_text(self.name, "name")
        required_text(self.email, "email")
        if self.status not in {"ACTIVE", "MERGED"}:
            raise SimulatorContractError("contact status is invalid")
        if self.status == "ACTIVE" and self.merged_into is not None:
            raise SimulatorContractError("active contact cannot have merged_into")
        if self.status == "MERGED":
            required_text(self.merged_into, "merged_into")

    def to_payload(self) -> JsonObject:
        return {
            "contact_id": self.contact_id,
            "name": self.name,
            "email": self.email,
            "status": self.status,
            "merged_into": self.merged_into,
        }


class CRMSimulator:
    """Thread-safe CRM with irreversible merge, send, and export effects."""

    domain = "crm"

    def __init__(self, contacts: Sequence[Contact]) -> None:
        self._lock = RLock()
        self._contacts: dict[str, Contact] = {}
        for contact in contacts:
            if contact.contact_id in self._contacts:
                raise SimulatorConflict("duplicate contact_id")
            self._contacts[contact.contact_id] = contact
        self._sent_emails: list[JsonObject] = []
        self._exports: list[JsonObject] = []
        self._state_version = 0
        self._effects: dict[str, StoredEffect] = {}

    @property
    def state_version(self) -> int:
        with self._lock:
            return self._state_version

    def snapshot(self) -> JsonObject:
        with self._lock:
            value = {
                "domain": self.domain,
                "state_version": self._state_version,
                "contacts": [
                    self._contacts[key].to_payload() for key in sorted(self._contacts)
                ],
                "sent_emails": self._sent_emails,
                "exports": self._exports,
            }
            return restore_object(
                canonical_object(value, "CRM snapshot"), "CRM snapshot"
            )

    def execute(
        self,
        capability: str,
        *,
        request_id: str,
        idempotency_key: str,
        arguments: Mapping[str, Any],
    ) -> EffectReceipt:
        if capability not in CRM_CAPABILITIES:
            raise SimulatorContractError("unknown CRM capability")
        captured, request_digest = capture_request(
            capability=capability,
            request_id=request_id,
            idempotency_key=idempotency_key,
            arguments=arguments,
        )
        with self._lock:
            stored = self._effects.get(idempotency_key)
            if stored is not None:
                if stored.request_digest != request_digest:
                    raise SimulatorConflict(
                        "idempotency key was already used for another request"
                    )
                return stored.receipt

            result, mutated = self._apply(capability, captured)
            if mutated:
                self._state_version += 1
            receipt = EffectReceipt.capture(
                domain=self.domain,
                capability=capability,
                request_id=request_id,
                idempotency_key=idempotency_key,
                effect_id=effect_identifier(
                    domain=self.domain,
                    capability=capability,
                    idempotency_key=idempotency_key,
                    request_digest=request_digest,
                ),
                state_version=self._state_version,
                mutated=mutated,
                result=result,
            )
            self._effects[idempotency_key] = StoredEffect(request_digest, receipt)
            return receipt

    def _apply(self, capability: str, arguments: JsonObject) -> tuple[JsonObject, bool]:
        if capability == CRM_CONTACT_READ:
            return self._read(arguments), False
        if capability == CRM_CONTACT_MERGE:
            return self._merge(arguments), True
        if capability == CRM_EMAIL_SEND:
            return self._send_email(arguments), True
        if capability == CRM_EXPORT_CONTACTS:
            return self._export(arguments), True
        raise SimulatorContractError("unknown CRM capability")

    def _read(self, arguments: JsonObject) -> JsonObject:
        exact_fields(arguments, frozenset({"contact_id"}), "arguments")
        contact = self._active_contact(
            required_text(arguments["contact_id"], "arguments.contact_id")
        )
        return {"contact": contact.to_payload()}

    def _merge(self, arguments: JsonObject) -> JsonObject:
        exact_fields(
            arguments,
            frozenset({"source_contact_id", "target_contact_id"}),
            "arguments",
        )
        source_id = required_text(
            arguments["source_contact_id"], "arguments.source_contact_id"
        )
        target_id = required_text(
            arguments["target_contact_id"], "arguments.target_contact_id"
        )
        if source_id == target_id:
            raise SimulatorBusinessError("a contact cannot be merged into itself")
        source = self._active_contact(source_id)
        target = self._active_contact(target_id)
        self._contacts[source_id] = Contact(
            contact_id=source.contact_id,
            name=source.name,
            email=source.email,
            status="MERGED",
            merged_into=target.contact_id,
        )
        return {
            "source_contact_id": source_id,
            "target_contact_id": target_id,
            "status": "MERGED",
        }

    def _send_email(self, arguments: JsonObject) -> JsonObject:
        exact_fields(
            arguments,
            frozenset({"contact_id", "subject", "body"}),
            "arguments",
        )
        contact = self._active_contact(
            required_text(arguments["contact_id"], "arguments.contact_id")
        )
        subject = required_text(arguments["subject"], "arguments.subject")
        body = required_text(arguments["body"], "arguments.body")
        message_id = f"message-{len(self._sent_emails) + 1}"
        message = {
            "message_id": message_id,
            "contact_id": contact.contact_id,
            "recipient": contact.email,
            "subject": subject,
            "body": body,
            "delivery_state": "SENT",
        }
        self._sent_emails.append(message)
        return {key: value for key, value in message.items() if key != "body"}

    def _export(self, arguments: JsonObject) -> JsonObject:
        exact_fields(arguments, frozenset({"fields", "format"}), "arguments")
        export_format = required_text(arguments["format"], "arguments.format")
        if export_format != "json":
            raise SimulatorBusinessError("only json exports are supported")
        fields = arguments["fields"]
        if not isinstance(fields, list) or not fields:
            raise SimulatorContractError("arguments.fields must be a non-empty list")
        allowed_fields = {"contact_id", "name", "email", "status"}
        if any(
            not isinstance(item, str) or item not in allowed_fields for item in fields
        ):
            raise SimulatorContractError("arguments.fields contains an unknown field")
        if len(set(fields)) != len(fields):
            raise SimulatorContractError("arguments.fields must not contain duplicates")
        rows = []
        for contact_id in sorted(self._contacts):
            contact = self._contacts[contact_id].to_payload()
            rows.append({field: contact[field] for field in fields})
        export_id = f"export-{len(self._exports) + 1}"
        self._exports.append(
            {
                "export_id": export_id,
                "format": export_format,
                "fields": list(fields),
                "row_count": len(rows),
            }
        )
        return {
            "export_id": export_id,
            "format": export_format,
            "row_count": len(rows),
            "rows": rows,
        }

    def _active_contact(self, contact_id: str) -> Contact:
        contact = self._contacts.get(contact_id)
        if contact is None:
            raise SimulatorNotFound("contact does not exist")
        if contact.status != "ACTIVE":
            raise SimulatorConflict("contact is no longer active")
        return contact


__all__ = [
    "CRM_CAPABILITIES",
    "CRM_CONTACT_MERGE",
    "CRM_CONTACT_READ",
    "CRM_EMAIL_SEND",
    "CRM_EXPORT_CONTACTS",
    "CRMSimulator",
    "Contact",
]
