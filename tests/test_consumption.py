from __future__ import annotations

from pathlib import Path

import pytest

from cbrain.consumption import (
    AuthorizationConsumptionError,
    AuthorizationUse,
    DatabaseConnection,
    PostgresAuthorizationConsumptionStore,
)

ZERO = "sha256:" + ("0" * 64)


class FakeCursor:
    def __init__(
        self,
        *,
        row: object | None,
        execute_error: Exception | None = None,
    ) -> None:
        self.row = row
        self.execute_error = execute_error
        self.executions: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    def execute(
        self,
        operation: str,
        parameters: tuple[object, ...],
    ) -> object:
        self.executions.append((operation, parameters))

        if self.execute_error is not None:
            raise self.execute_error

        return None

    def fetchone(self) -> object | None:
        return self.row

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(
        self,
        cursor: FakeCursor,
        *,
        rollback_error: Exception | None = None,
    ) -> None:
        self.cursor_value = cursor
        self.rollback_error = rollback_error
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_value

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

        if self.rollback_error is not None:
            raise self.rollback_error

    def close(self) -> None:
        self.closed = True


def _authorization() -> AuthorizationUse:
    return AuthorizationUse(
        organisation_id="bank.example",
        execution_authorization_id="exec-auth-001",
        authorization_digest=ZERO,
        request_id="request-001",
        claimed_at="2026-07-31T12:00:30Z",
    )


def _store(
    connection: FakeConnection,
) -> PostgresAuthorizationConsumptionStore:
    def connect() -> DatabaseConnection:
        return connection

    return PostgresAuthorizationConsumptionStore(connect)


def test_lookup_finds_existing_claim() -> None:
    cursor = FakeCursor(row=(1,))
    connection = FakeConnection(cursor)

    consumed = _store(connection).is_consumed(_authorization())

    assert consumed is True
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert connection.closed is True
    assert cursor.closed is True

    operation, parameters = cursor.executions[0]

    assert "SELECT 1" in operation
    assert parameters == (
        "bank.example",
        "exec-auth-001",
        ZERO,
    )


def test_lookup_reports_unused_authorization() -> None:
    cursor = FakeCursor(row=None)
    connection = FakeConnection(cursor)

    consumed = _store(connection).is_consumed(_authorization())

    assert consumed is False
    assert connection.commits == 1


def test_atomic_claim_succeeds_once() -> None:
    cursor = FakeCursor(row=(ZERO,))
    connection = FakeConnection(cursor)

    claimed = _store(connection).claim_once(_authorization())

    assert claimed is True
    assert connection.commits == 1
    assert connection.rollbacks == 0

    operation, parameters = cursor.executions[0]

    assert "ON CONFLICT DO NOTHING" in operation
    assert "RETURNING authorization_digest" in operation
    assert parameters == (
        "bank.example",
        "exec-auth-001",
        ZERO,
        "request-001",
        "2026-07-31T12:00:30Z",
    )


def test_atomic_claim_refuses_duplicate() -> None:
    cursor = FakeCursor(row=None)
    connection = FakeConnection(cursor)

    claimed = _store(connection).claim_once(_authorization())

    assert claimed is False
    assert connection.commits == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("organisation_id", ""),
        ("execution_authorization_id", ""),
        ("request_id", ""),
        ("claimed_at", ""),
        ("authorization_digest", "invalid"),
    ],
)
def test_authorization_use_rejects_invalid_values(
    field: str,
    value: str,
) -> None:
    values = {
        "organisation_id": "bank.example",
        "execution_authorization_id": "exec-auth-001",
        "authorization_digest": ZERO,
        "request_id": "request-001",
        "claimed_at": "2026-07-31T12:00:30Z",
    }
    values[field] = value

    with pytest.raises(ValueError):
        AuthorizationUse(**values)


def test_database_execute_failure_rolls_back() -> None:
    cursor = FakeCursor(
        row=None,
        execute_error=RuntimeError("database detail that must not escape"),
    )
    connection = FakeConnection(cursor)

    with pytest.raises(
        AuthorizationConsumptionError,
        match="atomic claim failed",
    ) as raised:
        _store(connection).claim_once(_authorization())

    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed is True
    assert cursor.closed is True
    assert "database detail" not in str(raised.value)


def test_rollback_failure_does_not_hide_control_error() -> None:
    cursor = FakeCursor(
        row=None,
        execute_error=RuntimeError("insert failed"),
    )
    connection = FakeConnection(
        cursor,
        rollback_error=RuntimeError("rollback also failed"),
    )

    with pytest.raises(
        AuthorizationConsumptionError,
        match="atomic claim failed",
    ):
        _store(connection).claim_once(_authorization())

    assert connection.rollbacks == 1
    assert connection.closed is True


def test_connection_failure_is_fail_closed() -> None:
    def unavailable() -> DatabaseConnection:
        raise OSError("credentials unavailable")

    store = PostgresAuthorizationConsumptionStore(unavailable)

    with pytest.raises(
        AuthorizationConsumptionError,
        match="database unavailable",
    ) as raised:
        store.is_consumed(_authorization())

    assert "credentials unavailable" not in str(raised.value)


def test_migration_enforces_atomic_uniqueness() -> None:
    migration = Path(
        "migrations/postgres/001_execution_authorization_uses.sql"
    ).read_text(encoding="utf-8")

    assert "PRIMARY KEY" in migration
    assert "UNIQUE (authorization_digest)" in migration
    assert "TIMESTAMPTZ NOT NULL" in migration
