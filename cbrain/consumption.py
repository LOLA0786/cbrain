from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol


class AuthorizationConsumptionError(RuntimeError):
    """Consumption state could not be proven safely."""


@dataclass(frozen=True, slots=True)
class AuthorizationUse:
    organisation_id: str
    execution_authorization_id: str
    authorization_digest: str
    request_id: str
    claimed_at: str

    def __post_init__(self) -> None:
        for name in (
            "organisation_id",
            "execution_authorization_id",
            "request_id",
            "claimed_at",
        ):
            value = getattr(self, name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")

        _require_digest(
            self.authorization_digest,
            "authorization_digest",
        )


class AuthorizationConsumptionStore(Protocol):
    def is_consumed(
        self,
        authorization: AuthorizationUse,
    ) -> bool:
        """Return whether this authorization was already claimed."""

    def claim_once(
        self,
        authorization: AuthorizationUse,
    ) -> bool:
        """Atomically claim once; False means it was already claimed."""


class DatabaseCursor(Protocol):
    def execute(
        self,
        operation: str,
        parameters: tuple[object, ...],
    ) -> object:
        """Execute one parameterized database operation."""

    def fetchone(self) -> object | None:
        """Return one row or None."""

    def close(self) -> None:
        """Close the cursor."""


class DatabaseConnection(Protocol):
    def cursor(self) -> DatabaseCursor:
        """Create a cursor."""

    def commit(self) -> None:
        """Commit the transaction."""

    def rollback(self) -> None:
        """Roll back the transaction."""

    def close(self) -> None:
        """Close the connection."""


ConnectionFactory = Callable[[], DatabaseConnection]

_LOOKUP_SQL = """
SELECT 1
FROM cbrain_execution_authorization_uses
WHERE (
    organisation_id = %s
    AND execution_authorization_id = %s
) OR authorization_digest = %s
LIMIT 1
"""

_CLAIM_SQL = """
INSERT INTO cbrain_execution_authorization_uses (
    organisation_id,
    execution_authorization_id,
    authorization_digest,
    request_id,
    claimed_at
)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT DO NOTHING
RETURNING authorization_digest
"""


class PostgresAuthorizationConsumptionStore:
    """Atomic single-use ledger backed by PostgreSQL constraints."""

    def __init__(
        self,
        connection_factory: ConnectionFactory,
    ) -> None:
        self._connection_factory = connection_factory

    def is_consumed(
        self,
        authorization: AuthorizationUse,
    ) -> bool:
        connection = self._connect()
        cursor = connection.cursor()

        try:
            cursor.execute(
                _LOOKUP_SQL,
                (
                    authorization.organisation_id,
                    authorization.execution_authorization_id,
                    authorization.authorization_digest,
                ),
            )
            consumed = cursor.fetchone() is not None
            connection.commit()
            return consumed
        except Exception as exc:
            _rollback(connection)
            raise AuthorizationConsumptionError(
                "authorization consumption lookup failed"
            ) from exc
        finally:
            cursor.close()
            connection.close()

    def claim_once(
        self,
        authorization: AuthorizationUse,
    ) -> bool:
        connection = self._connect()
        cursor = connection.cursor()

        try:
            cursor.execute(
                _CLAIM_SQL,
                (
                    authorization.organisation_id,
                    authorization.execution_authorization_id,
                    authorization.authorization_digest,
                    authorization.request_id,
                    authorization.claimed_at,
                ),
            )
            claimed = cursor.fetchone() is not None
            connection.commit()
            return claimed
        except Exception as exc:
            _rollback(connection)
            raise AuthorizationConsumptionError(
                "authorization atomic claim failed"
            ) from exc
        finally:
            cursor.close()
            connection.close()

    def _connect(self) -> DatabaseConnection:
        try:
            return self._connection_factory()
        except Exception as exc:
            raise AuthorizationConsumptionError(
                "authorization consumption database unavailable"
            ) from exc


def _rollback(
    connection: DatabaseConnection,
) -> None:
    with suppress(Exception):
        connection.rollback()


def _require_digest(
    value: object,
    path: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a sha256 digest")

    prefix = "sha256:"
    hexadecimal = value.removeprefix(prefix)

    if (
        not value.startswith(prefix)
        or len(hexadecimal) != 64
        or any(character not in "0123456789abcdef" for character in hexadecimal)
    ):
        raise ValueError(f"{path} must be a sha256 digest")

    return value
