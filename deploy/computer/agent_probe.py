"""Isolation probe: agent can reach computer-worker, not the web fixture."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    worker = os.environ["COMPUTER_WORKER_URL"]
    token = os.environ["CBRAIN_COMPUTER_TOKEN"]
    web = os.environ["WEB_FIXTURE_URL"]

    # Agent must NOT reach the business web fixture directly.
    try:
        urllib.request.urlopen(web, timeout=2)
        print("PROBE_FAILED agent reached web fixture", file=sys.stderr)
        return 1
    except Exception:
        pass

    # Agent MUST reach the computer worker on the control network.
    body = json.dumps(
        {
            "schema": "cbrain-computer-worker/request-v1",
            "method": "navigate",
            "params": {
                "url_alias": "portal",
                "url": web,
            },
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    request = urllib.request.Request(
        worker,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.URLError as exc:
        print(f"PROBE_FAILED worker unreachable:{exc}", file=sys.stderr)
        return 1

    if not payload.get("ok"):
        print(f"PROBE_FAILED worker error:{payload}", file=sys.stderr)
        return 1

    print("PROBE_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
