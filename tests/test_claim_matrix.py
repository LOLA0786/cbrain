"""Claim matrix integrity: every docs/claims.md pin must exist."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "docs" / "claims.md"
FORBIDDEN = (
    "exactly-once purchase",
    "exactly once purchase",
)

PIN = re.compile(r"`(tests/[^`]+|deploy/[^`]+)`")


def test_claims_file_exists() -> None:
    assert CLAIMS.is_file()


def test_every_test_pin_in_claims_exists() -> None:
    text = CLAIMS.read_text(encoding="utf-8")
    missing: list[str] = []
    for match in PIN.finditer(text):
        pin = match.group(1)
        if "::" in pin:
            path_text, name = pin.split("::", 1)
            path = ROOT / path_text
            if not path.is_file():
                missing.append(pin)
                continue
            if f"def {name}(" not in path.read_text(encoding="utf-8"):
                missing.append(pin)
        else:
            path = ROOT / pin
            if not path.exists():
                missing.append(pin)
    assert missing == [], f"broken claim pins: {missing}"


def test_readme_does_not_claim_exactly_once_purchase() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for phrase in FORBIDDEN:
        assert phrase not in readme
