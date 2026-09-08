#!/usr/bin/env python3
"""Worker probe: must fail direct write; may succeed via sidecar control plane."""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

WRITE_URL = os.environ["WRITE_URL"]
SIDECAR_URL = os.environ["SIDECAR_URL"]


def try_url(url: str) -> str:
    req = urllib.request.Request(
        url,
        data=b'{"quantity_tonnes":40}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            return f"ok:{response.status}"
    except urllib.error.URLError as exc:
        return f"denied:{type(exc.reason).__name__ if hasattr(exc, 'reason') else type(exc).__name__}"
    except Exception as exc:  # noqa: BLE001
        return f"denied:{type(exc).__name__}"


def main() -> int:
    direct = try_url(WRITE_URL)
    via_sidecar = try_url(SIDECAR_URL)
    print(f"direct={direct}")
    print(f"sidecar={via_sidecar}")
    if not direct.startswith("denied:"):
        print("FAIL: worker reached write target directly")
        return 1
    if via_sidecar != "ok:200":
        print(f"FAIL: sidecar path did not succeed: {via_sidecar}")
        return 1
    if os.environ.get("WRITE_SECRET"):
        print("FAIL: worker must not hold WRITE_SECRET")
        return 1
    print("PROBE_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
