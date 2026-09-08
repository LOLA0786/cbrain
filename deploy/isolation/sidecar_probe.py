#!/usr/bin/env python3
"""Sidecar stand-in: holds write credentials; accepts control-plane dispatch."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WRITE_URL = os.environ["WRITE_URL"]
WRITE_SECRET = os.environ["WRITE_SECRET"]


def forward() -> tuple[int, bytes]:
    req = urllib.request.Request(
        WRITE_URL,
        data=b'{"quantity_tonnes":40}',
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {WRITE_SECRET}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        status, body = forward()
        self.send_response(status)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


if __name__ == "__main__":
    if sys.argv[-1] == "serve":
        ThreadingHTTPServer(("0.0.0.0", 8090), Handler).serve_forever()
    status, body = forward()
    print(status, body.decode())
    raise SystemExit(0 if status == 200 else 1)
