"""Out-of-process computer worker HTTP server.

This process owns the browser engine and may reach operator-allowlisted web
targets. The agent process must not. Auth is a shared bearer token from the
environment — never from model context.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .engines import BrowserEngine, build_engine
from .protocol import (
    ComputerProtocolError,
    build_response,
    parse_request,
)
from .surface import BackendActResult, ComputerBackendError

_MAX_BODY = 256 * 1024


class ComputerWorkerService:
    """Dispatch protocol methods onto a browser engine."""

    def __init__(self, engine: BrowserEngine) -> None:
        self._engine = engine

    def handle(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "health":
            return build_response(ok=True, result={"status": "ok"})

        try:
            if method == "observe":
                observation = self._engine.observe(
                    url_alias=_require_text(params, "url_alias")
                )
                return build_response(ok=True, result=observation.to_wire())

            if method == "navigate":
                result = self._engine.navigate(
                    url_alias=_require_text(params, "url_alias"),
                    url=_require_text(params, "url"),
                )
                return build_response(ok=True, result=_act_wire(result))

            if method == "click":
                result = self._engine.click(
                    url_alias=_require_text(params, "url_alias"),
                    selector=_require_text(params, "selector"),
                )
                return build_response(ok=True, result=_act_wire(result))

            if method == "type_text":
                result = self._engine.type_text(
                    url_alias=_require_text(params, "url_alias"),
                    selector=_require_text(params, "selector"),
                    text=_require_text(params, "text"),
                )
                return build_response(ok=True, result=_act_wire(result))

            if method == "submit":
                result = self._engine.submit(
                    url_alias=_require_text(params, "url_alias"),
                    selector=_require_text(params, "selector"),
                )
                return build_response(ok=True, result=_act_wire(result))
        except ComputerBackendError as exc:
            return build_response(ok=False, error=str(exc))

        return build_response(ok=False, error=f"unsupported method {method!r}")


def serve_computer_worker(
    *,
    host: str,
    port: int,
    token: str,
    engine: BrowserEngine,
) -> ThreadingHTTPServer:
    if not token.strip():
        raise ComputerBackendError("computer worker token must be non-empty")

    service = ComputerWorkerService(engine)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            # Never log request bodies (may include typed form text).
            sys.stderr.write(f"{self.address_string()} - {format % args}\n")

        def do_GET(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != "/health":
                self._write(404, build_response(ok=False, error="not_found"))
                return
            if not self._authorized():
                self._write(401, build_response(ok=False, error="unauthorized"))
                return
            self._write(200, build_response(ok=True, result={"status": "ok"}))

        def do_POST(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != "/v1/computer":
                self._write(404, build_response(ok=False, error="not_found"))
                return
            if not self._authorized():
                self._write(401, build_response(ok=False, error="unauthorized"))
                return

            length_header = self.headers.get("Content-Length", "")
            try:
                length = int(length_header)
            except ValueError:
                self._write(400, build_response(ok=False, error="bad_length"))
                return
            if length < 0 or length > _MAX_BODY:
                self._write(400, build_response(ok=False, error="body_too_large"))
                return

            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw)
                method, params = parse_request(payload)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                ComputerProtocolError,
            ) as exc:
                self._write(
                    400,
                    build_response(ok=False, error=f"protocol:{type(exc).__name__}"),
                )
                return

            response = service.handle(method, params)
            status = 200 if response.get("ok") else 400
            self._write(status, response)

        def _authorized(self) -> bool:
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {token}"

        def _write(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CBrain computer worker")
    parser.add_argument(
        "--host", default=os.environ.get("CBRAIN_COMPUTER_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CBRAIN_COMPUTER_PORT", "8765")),
    )
    parser.add_argument(
        "--engine",
        default=os.environ.get("CBRAIN_COMPUTER_ENGINE", "stub"),
    )
    args = parser.parse_args(argv)

    token = os.environ.get("CBRAIN_COMPUTER_TOKEN", "")
    if not token:
        print("CBRAIN_COMPUTER_TOKEN is required", file=sys.stderr)
        return 2

    engine = build_engine(args.engine)
    server = serve_computer_worker(
        host=args.host,
        port=args.port,
        token=token,
        engine=engine,
    )
    print(
        f"cbrain-computer-worker on {args.host}:{args.port} engine={args.engine}",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()
        server.server_close()
    return 0


def _require_text(params: dict[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ComputerBackendError(f"{key} must be non-empty text")
    return value


def _act_wire(result: BackendActResult) -> dict[str, Any]:
    return result.to_wire()


__all__ = ["ComputerWorkerService", "main", "serve_computer_worker"]
