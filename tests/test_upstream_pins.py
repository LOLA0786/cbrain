"""README pin SHAs must match upstreams.lock.json and never drift."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "upstreams.lock.json"
README_PATH = ROOT / "README.md"
COMMIT_SHA = re.compile(r"\b[0-9a-f]{40}\b")


def test_readme_commit_shas_match_upstreams_lock() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    pinned = {str(component["commit"]) for component in lock["components"]}
    readme = README_PATH.read_text(encoding="utf-8")
    found = set(COMMIT_SHA.findall(readme))
    assert found, "README must cite pinned upstream commit SHAs"
    unknown = found - pinned
    assert unknown == set(), f"README SHAs missing from upstreams.lock.json: {unknown}"
    missing = pinned - found
    assert missing == set(), f"upstreams.lock.json SHAs missing from README: {missing}"
