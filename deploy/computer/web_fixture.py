"""Minimal HTTP fixture representing an allowlisted business portal."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            body = b"ok"
        else:
            body = (
                b"<html><body><h1>Portal</h1><button id='go'>Go</button></body></html>"
            )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
