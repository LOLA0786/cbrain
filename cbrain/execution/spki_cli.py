"""Print the canonical SPKI pin for a TLS peer. No hostname verification."""

from __future__ import annotations

import http.client
import ssl
import sys

from cbrain.execution.tls import peer_identity_from_der_certificate


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        sys.stderr.write("usage: cbrain-spki <host> <port>\n")
        return 2
    host, port_text = args
    try:
        port = int(port_text)
    except ValueError:
        sys.stderr.write("port must be an integer\n")
        return 2
    if port < 1 or port > 65535:
        sys.stderr.write("port is out of range\n")
        return 2

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    connection = http.client.HTTPSConnection(host, port, context=context, timeout=10.0)
    try:
        connection.connect()
        socket = connection.sock
        if socket is None:
            sys.stderr.write("TLS connection has no peer socket\n")
            return 1
        certificate = socket.getpeercert(binary_form=True)
        if not certificate:
            sys.stderr.write("TLS peer certificate is unavailable\n")
            return 1
        sys.stdout.write(peer_identity_from_der_certificate(certificate) + "\n")
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
