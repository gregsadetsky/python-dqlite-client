"""A write whose outcome is in doubt raises AmbiguousCommitError; everything else stays plain."""

from __future__ import annotations

import asyncio
import pickle
from typing import Any

import pytest

from dqliteclient import (
    AmbiguousCommitError,
    ConnectionPool,
    DqliteConnection,
    DqliteConnectionError,
    OperationalError,
    ProtocolError,
    retry_with_backoff,
)
from dqlitewire import (
    SQLITE_IOERR_LEADERSHIP_LOST,
    SQLITE_IOERR_LEADERSHIP_LOST_LEGACY,
    SQLITE_IOERR_NOT_LEADER,
)


def lost_reply() -> DqliteConnectionError:
    return DqliteConnectionError("Server read from localhost:9001 timed out after 4.0s")


class FailingProtocol:
    """Stands in for ``DqliteProtocol``; fails the next SQL request with ``fail_next``."""

    def __init__(self) -> None:
        self.is_alive = True
        self.sent: list[str] = []
        self.fail_next: BaseException | None = None

    def close(self) -> None:
        self.is_alive = False

    async def wait_closed(self) -> None:
        pass

    def _maybe_fail(self, sql: str) -> None:
        self.sent.append(sql)
        exc, self.fail_next = self.fail_next, None
        if exc is not None:
            raise exc

    async def exec_sql(self, db_id: int, sql: str, params: Any) -> tuple[int, int]:
        self._maybe_fail(sql)
        return (0, 0)

    async def query_sql(
        self, db_id: int, sql: str, params: Any
    ) -> tuple[list[str], list[list[Any]]]:
        self._maybe_fail(sql)
        return (["x"], [[1]])

    async def query_sql_typed(
        self, db_id: int, sql: str, params: Any
    ) -> tuple[list[str], list[int], list[list[int]], list[list[Any]]]:
        self._maybe_fail(sql)
        return (["x"], [1], [[1]], [[1]])


def failing() -> tuple[DqliteConnection, FailingProtocol]:
    proto = FailingProtocol()
    conn = DqliteConnection("localhost:9001")
    conn._protocol = proto  # type: ignore[assignment]
    conn._db_id = 1
    conn._state.connected = True
    return conn, proto


async def run(conn: Any, method: str, sql: str) -> Any:
    if method == "execute":
        return await conn.execute(sql)
    return await getattr(conn, method)(sql)


SQL_METHODS = ["execute", "query_raw", "query_raw_typed", "fetch", "fetchall", "fetchval"]

COMMITTING = [
    "INSERT INTO t VALUES (1)",
    "UPDATE t SET v = 2",
    "DELETE FROM t",
    "REPLACE INTO t VALUES (1)",
    "CREATE TABLE u (v)",
    "DROP TABLE u",
    "/* note */ insert into t values (1)",
    "WITH s AS (SELECT 1) INSERT INTO t SELECT * FROM s",
    "PRAGMA user_version = 42",
    "INSERT INTO t VALUES (1); INSERT INTO t VALUES (2)",
    "BEGIN; INSERT INTO t VALUES (1); COMMIT",
    "SELECT 1; INSERT INTO t VALUES (1)",
]

NOT_COMMITTING = [
    "SELECT 1",
    "select * from t where v = 'INSERT'",
    "VALUES (1)",
    "EXPLAIN INSERT INTO t VALUES (1)",
    "WITH s AS (SELECT 1) SELECT * FROM s",
    "PRAGMA user_version",
    "SELECT 1; SELECT 2",
    "BEGIN; INSERT INTO t VALUES (1)",
    "ROLLBACK",
]

LOST_SESSION = {
    "read-timeout": lost_reply,
    "closed": lambda: DqliteConnectionError("Connection closed by server"),
    "write-failed": lambda: DqliteConnectionError("Write failed: [Errno 32] Broken pipe"),
    "protocol": lambda: ProtocolError("wire decode failed: bad frame"),
}


@pytest.mark.parametrize("sql", COMMITTING)
@pytest.mark.parametrize("lost_name", LOST_SESSION)
async def test_autocommit_write_with_a_lost_session_is_ambiguous(sql: str, lost_name: str) -> None:
    conn, proto = failing()
    lost = proto.fail_next = LOST_SESSION[lost_name]()
    with pytest.raises(AmbiguousCommitError) as info:
        await conn.execute(sql)
    assert info.value.__cause__ is lost
    assert str(lost) in str(info.value)
    assert not conn.is_connected


async def test_ambiguous_lost_session_is_still_a_connection_error() -> None:
    """Handlers written for a lost connection (and the dbapi's cause check) still match."""
    conn, proto = failing()
    proto.fail_next = lost_reply()
    with pytest.raises(DqliteConnectionError) as info:
        await conn.execute("INSERT INTO t VALUES (1)")
    assert isinstance(info.value, AmbiguousCommitError)
    assert isinstance(info.value, OperationalError)
    assert info.value.code is None
    clone = pickle.loads(pickle.dumps(info.value))
    assert type(clone) is type(info.value) and str(clone) == str(info.value)


@pytest.mark.parametrize("method", SQL_METHODS)
async def test_returning_write_through_any_sql_method_is_ambiguous(method: str) -> None:
    conn, proto = failing()
    proto.fail_next = lost_reply()
    with pytest.raises(AmbiguousCommitError):
        await run(conn, method, "INSERT INTO t VALUES (1) RETURNING v")


@pytest.mark.parametrize(
    "code", [SQLITE_IOERR_LEADERSHIP_LOST, SQLITE_IOERR_LEADERSHIP_LOST_LEGACY]
)
async def test_autocommit_write_losing_leadership_is_ambiguous(code: int) -> None:
    conn, proto = failing()
    proto.fail_next = OperationalError("leadership lost", code)
    with pytest.raises(AmbiguousCommitError) as info:
        await conn.execute("INSERT INTO t VALUES (1)")
    assert info.value.code == code
    assert not isinstance(info.value, DqliteConnectionError)


@pytest.mark.parametrize("sql", ["COMMIT", "END", "RELEASE s"])
async def test_commit_or_release_with_a_lost_session_is_ambiguous(sql: str) -> None:
    conn, proto = failing()
    await conn.execute("BEGIN")
    await conn.execute("SAVEPOINT s")
    proto.fail_next = lost_reply()
    with pytest.raises(AmbiguousCommitError):
        await conn.execute(sql)
    assert conn.in_transaction is False


@pytest.mark.parametrize("lost_name", LOST_SESSION)
async def test_transaction_commit_with_a_lost_session_is_ambiguous(lost_name: str) -> None:
    conn, proto = failing()
    with pytest.raises(AmbiguousCommitError) as info:
        async with conn.transaction():
            await conn.execute("INSERT INTO t VALUES (1)")
            lost = proto.fail_next = LOST_SESSION[lost_name]()
    assert info.value.__cause__ is lost
    assert proto.sent == ["BEGIN", "INSERT INTO t VALUES (1)", "COMMIT"]


async def test_pool_execute_with_a_lost_reply_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    created: list[FailingProtocol] = []

    async def create() -> Any:
        conn, proto = failing()
        if not created:
            proto.fail_next = lost_reply()
        created.append(proto)
        return conn

    monkeypatch.setattr(pool, "_create_connection", create)
    with pytest.raises(AmbiguousCommitError):
        await pool.execute("INSERT INTO t VALUES (1)")
    await pool.execute("INSERT INTO t VALUES (2)")
    assert len(created) == 2
    await pool.close()


async def test_retry_with_backoff_does_not_retry_an_ambiguous_write() -> None:
    conn, proto = failing()
    calls = 0

    async def write() -> tuple[int, int]:
        nonlocal calls
        calls += 1
        proto.fail_next = lost_reply()
        return await conn.execute("INSERT INTO t VALUES (1)")

    with pytest.raises(AmbiguousCommitError):
        await retry_with_backoff(write, max_attempts=3, base_delay=0)
    assert calls == 1


# -- what stays a plain failure -------------------------------------------------------


@pytest.mark.parametrize("sql", NOT_COMMITTING)
@pytest.mark.parametrize("method", ["execute", "query_raw"])
async def test_statement_that_cannot_commit_keeps_the_plain_error(sql: str, method: str) -> None:
    conn, proto = failing()
    lost = proto.fail_next = lost_reply()
    with pytest.raises(DqliteConnectionError) as info:
        await run(conn, method, sql)
    assert info.value is lost


@pytest.mark.parametrize("lost_name", LOST_SESSION)
async def test_write_inside_a_transaction_keeps_the_plain_error(lost_name: str) -> None:
    """The server's transaction dies with the session, so nothing was applied."""
    conn, proto = failing()
    await conn.execute("BEGIN")
    lost = proto.fail_next = LOST_SESSION[lost_name]()
    with pytest.raises(type(lost)) as info:
        await conn.execute("INSERT INTO t VALUES (1)")
    assert info.value is lost


async def test_write_in_a_transaction_body_keeps_the_plain_error() -> None:
    conn, proto = failing()
    with pytest.raises(DqliteConnectionError) as info:
        async with conn.transaction():
            lost = proto.fail_next = lost_reply()
            await conn.execute("INSERT INTO t VALUES (1)")
    assert info.value is lost


async def test_not_leader_is_a_clean_rejection() -> None:
    conn, proto = failing()
    error = OperationalError("not leader", SQLITE_IOERR_NOT_LEADER)
    proto.fail_next = error
    with pytest.raises(OperationalError) as info:
        await conn.execute("INSERT INTO t VALUES (1)")
    assert info.value is error


async def test_write_on_a_disconnected_connection_is_not_ambiguous() -> None:
    conn, proto = failing()
    proto.fail_next = lost_reply()
    with pytest.raises(DqliteConnectionError):
        await conn.execute("SELECT 1")
    with pytest.raises(DqliteConnectionError, match="Not connected") as info:
        await conn.execute("INSERT INTO t VALUES (1)")
    assert not isinstance(info.value, AmbiguousCommitError)


async def test_cancelled_write_stays_a_cancellation() -> None:
    conn, proto = failing()
    proto.fail_next = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await conn.execute("INSERT INTO t VALUES (1)")


async def test_retry_with_backoff_still_retries_a_plain_lost_session() -> None:
    conn, proto = failing()
    calls = 0

    async def read() -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise lost_reply()
        return await conn.fetchval("SELECT 1")

    assert await retry_with_backoff(read, max_attempts=3, base_delay=0) == 1
    assert calls == 2
