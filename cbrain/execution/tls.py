"""TLS peer identity pins.

Canonical form is ``tls-spki-sha256:<64 lowercase hex>``: SHA-256 of the
peer certificate SubjectPublicKeyInfo DER. The pin survives leaf rotation
when the public key is unchanged.
"""

from __future__ import annotations

import hashlib
import re

PEER_IDENTITY_PATTERN = re.compile(r"^tls-spki-sha256:[0-9a-f]{64}$")


class PeerIdentityError(ValueError):
    """A peer identity pin is missing, malformed, or not extractable."""


def require_peer_identity(value: object) -> str:
    if not isinstance(value, str) or PEER_IDENTITY_PATTERN.fullmatch(value) is None:
        raise PeerIdentityError(
            "peer_identity must match tls-spki-sha256:<64 lowercase hex>"
        )
    return value


def peer_identity_from_der_certificate(certificate_der: bytes) -> str:
    spki = subject_public_key_info_der(certificate_der)
    return f"tls-spki-sha256:{hashlib.sha256(spki).hexdigest()}"


def subject_public_key_info_der(certificate_der: bytes) -> bytes:
    certificate = _unwrap_sequence(certificate_der)
    tbs_tag, tbs, _end = _read_tlv(certificate, 0)
    if tbs_tag != 0x30:
        raise PeerIdentityError("certificate is missing TBSCertificate")
    tag, _value, offset = _read_tlv(tbs, 0)
    remaining_before_spki = 4 if tag != 0xA0 else 5
    for _ in range(remaining_before_spki):
        _tag, _value, offset = _read_tlv(tbs, offset)
    start = offset
    tag, _value, offset = _read_tlv(tbs, offset)
    if tag != 0x30:
        raise PeerIdentityError("certificate is missing SubjectPublicKeyInfo")
    return tbs[start:offset]


def _unwrap_sequence(data: bytes) -> bytes:
    tag, value, end = _read_tlv(data, 0)
    if tag != 0x30 or end != len(data):
        raise PeerIdentityError("certificate DER is not a complete SEQUENCE")
    return value


def _read_length(data: bytes, offset: int) -> tuple[int, int]:
    if offset >= len(data):
        raise PeerIdentityError("truncated DER length")
    first = data[offset]
    offset += 1
    if first < 0x80:
        return first, offset
    count = first & 0x7F
    if count == 0 or count > 4 or offset + count > len(data):
        raise PeerIdentityError("invalid DER length")
    length = int.from_bytes(data[offset : offset + count], "big")
    if length < 128:
        raise PeerIdentityError("non-canonical DER length")
    return length, offset + count


def _read_tlv(data: bytes, offset: int) -> tuple[int, bytes, int]:
    if offset >= len(data):
        raise PeerIdentityError("truncated DER")
    tag = data[offset]
    length, offset = _read_length(data, offset + 1)
    end = offset + length
    if end > len(data):
        raise PeerIdentityError("truncated DER value")
    return tag, data[offset:end], end
