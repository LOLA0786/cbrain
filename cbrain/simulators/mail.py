"""Exact-byte mail dispatch target for governed-execution scenarios."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any

from .contracts import (
    EffectReceipt,
    JsonObject,
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

PROCUREMENT_MAIL_SEND = "procurement.mail.send"
MAIL_CAPABILITIES = frozenset({PROCUREMENT_MAIL_SEND})
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class MailMessage:
    message_id: str
    to: str
    subject: str
    body: bytes
    body_sha256: str

    def public_payload(self) -> JsonObject:
        return {
            "message_id": self.message_id,
            "to": self.to,
            "subject": self.subject,
            "body_sha256": self.body_sha256,
            "byte_length": len(self.body),
            "status": "SENT",
        }


class MailSimulator:
    """Thread-safe mail target that stores exact body bytes and never logs them."""

    domain = "procurement"

    def __init__(self, *, credential_digest: str) -> None:
        if _SHA256_HEX.fullmatch(credential_digest) is None:
            raise SimulatorContractError(
                "credential_digest must be 64 lowercase hex characters"
            )
        self._lock = RLock()
        self._credential_digest = credential_digest
        self._messages: dict[str, MailMessage] = {}
        self._state_version = 0
        self._effects: dict[str, StoredEffect] = {}

    @property
    def credential_digest(self) -> str:
        return self._credential_digest

    @property
    def state_version(self) -> int:
        with self._lock:
            return self._state_version

    def snapshot(self) -> JsonObject:
        with self._lock:
            value = {
                "domain": self.domain,
                "state_version": self._state_version,
                "credential_digest": self._credential_digest,
                "messages": [
                    self._messages[key].public_payload()
                    for key in sorted(self._messages)
                ],
            }
            return restore_object(
                canonical_object(value, "mail snapshot"), "mail snapshot"
            )

    def stored_body(self, message_id: str) -> bytes:
        with self._lock:
            message = self._messages.get(message_id)
            if message is None:
                raise SimulatorNotFound("mail message does not exist")
            return message.body

    def execute(
        self,
        capability: str,
        *,
        request_id: str,
        idempotency_key: str,
        arguments: Mapping[str, Any],
    ) -> EffectReceipt:
        if capability not in MAIL_CAPABILITIES:
            raise SimulatorContractError("unknown mail capability")
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
        if capability == PROCUREMENT_MAIL_SEND:
            return self._send(arguments), True
        raise SimulatorContractError("unknown mail capability")

    def _send(self, arguments: JsonObject) -> JsonObject:
        exact_fields(arguments, frozenset({"to", "subject", "body"}), "arguments")
        body_text = arguments["body"]
        if not isinstance(body_text, str):
            raise SimulatorContractError("arguments.body must be text")
        body = body_text.encode("utf-8")
        message_id = f"mail-{len(self._messages) + 1:04d}"
        message = MailMessage(
            message_id=message_id,
            to=required_text(arguments["to"], "arguments.to"),
            subject=required_text(arguments["subject"], "arguments.subject"),
            body=body,
            body_sha256=f"sha256:{hashlib.sha256(body).hexdigest()}",
        )
        self._messages[message_id] = message
        return message.public_payload()


__all__ = [
    "MAIL_CAPABILITIES",
    "PROCUREMENT_MAIL_SEND",
    "MailMessage",
    "MailSimulator",
]
