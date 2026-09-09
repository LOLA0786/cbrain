"""Computer worker entry for isolation compose (stub engine, no package install).

Copies the protocol surface inline so the slim image need not pip-install CBrain.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

TOKEN = os.environ["CBRAIN_COMPUTER_TOKEN"]
HOST = os.environ.get("CBRAIN_COMPUTER_HOST", "0.0.0.0")
PORT = int(os.environ.get("CBRAIN_COMPUTER_PORT", "8765"))
REQUEST_SCHEMA = "cbrain-computer-worker/request-v1"
RESPONSE_SCHEMA = "cbrain-computer-worker/response-v1"

_PAGES: dict[str, dict[str, str]] = {}


def _response(ok: bool, result=None, error=None) -> dict:
    payload = {"schema": RESPONSE_SCHEMA, "ok": ok}
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error
    return payload


def _handle(method: str, params: dict) -> dict:
    if method == "health":
        return _response(True, {"status": "ok"})
    alias = str(params.get("url_alias", ""))
    if method == "navigate":
        url = str(params.get("url", ""))
        _PAGES[alias] = {
            "title": f"page:{alias}",
            "url": url,
            "accessibility_tree": f"document[alias={alias}]",
            "text_excerpt": f"Opened {alias}",
        }
        return _response(True, _act("navigated", alias))
    if alias not in _PAGES:
        return _response(False, error=f"no open page for alias {alias!r}")
    if method == "observe":
        page = _PAGES[alias]
        return _response(
            True,
            {
                "title": page["title"],
                "url_alias": alias,
                "accessibility_tree": page["accessibility_tree"],
                "text_excerpt": page["text_excerpt"],
                "screenshot_sha256": "sha256:stub",
                "screenshot_media_type": "image/png",
            },
        )
    if method == "click":
        _PAGES[alias]["text_excerpt"] = f"clicked:{params.get('selector')}"
        return _response(True, _act(f"clicked:{params.get('selector')}", alias))
    if method == "type_text":
        _PAGES[alias]["text_excerpt"] = f"typed:{params.get('selector')}"
        return _response(True, _act(f"typed:{params.get('selector')}", alias))
    if method == "submit":
        _PAGES[alias]["text_excerpt"] = f"submitted:{params.get('selector')}"
        return _response(True, _act(f"submitted:{params.get('selector')}", alias))
    return _response(False, error=f"unsupported method {method!r}")


def _act(detail: str, alias: str) -> dict:
    page = _PAGES[alias]
    return {
        "ok": True,
        "detail": detail,
        "observation": {
            "title": page["title"],
            "url_alias": alias,
            "accessibility_tree": page["accessibility_tree"],
            "text_excerpt": page["text_excerpt"],
            "screenshot_sha256": "sha256:stub",
            "screenshot_media_type": "image/png",
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def do_GET(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/health":
            self._write(404, _response(False, error="not_found"))
            return
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self._write(401, _response(False, error="unauthorized"))
            return
        self._write(200, _response(True, {"status": "ok"}))

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/v1/computer":
            self._write(404, _response(False, error="not_found"))
            return
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self._write(401, _response(False, error="unauthorized"))
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._write(400, _response(False, error="bad_json"))
            return
        if payload.get("schema") != REQUEST_SCHEMA:
            self._write(400, _response(False, error="schema"))
            return
        method = str(payload.get("method", ""))
        params = payload.get("params") or {}
        response = _handle(method, params if isinstance(params, dict) else {})
        self._write(200 if response.get("ok") else 400, response)

    def _write(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"computer-worker on {HOST}:{PORT}", file=sys.stderr)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
