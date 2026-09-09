"""Computer worker isolation claim gate."""

from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("CBRAIN_ISOLATION") != "1",
    reason="CBRAIN_ISOLATION is not set",
)
def test_compose_computer_isolation_probe_passed() -> None:
    assert os.environ.get("CBRAIN_COMPUTER_ISOLATION_PROBE") == "passed"
