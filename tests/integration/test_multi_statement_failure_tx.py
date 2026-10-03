"""A multi-statement execute that fails part way can leave a transaction open on the server.

The server runs the pieces of ``"BEGIN; INSERT ...; INSERT ..."`` one by one and stops at
the first failure; the BEGIN and anything after it stay applied. The client's
``in_transaction`` flag must reflect that, or the pool returns the connection to idle with
the transaction still open: the next autocommit write on it is acknowledged, then lost
when the connection closes.
"""

from __future__ import annotations

import pytest

from dqliteclient import ConnectionPool, DqliteConnection
from dqliteclient.exceptions import OperationalError

SQLITE_CONSTRAINT = 19

FAILING_BATCH = "BEGIN; INSERT INTO {t} VALUES (1); INSERT INTO {t} VALUES (1)"


async def _fresh_table(pool: ConnectionPool, table: str) -> None:
    await pool.execute(f"DROP TABLE IF EXISTS {table}")
    await pool.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)")


async def _rows(addresses: list[str], table: str) -> list[int]:
    observer = ConnectionPool(addresses, min_size=0, max_size=1)
    try:
        return [r[0] for r in await observer.fetchall(f"SELECT id FROM {table} ORDER BY id")]
    finally:
        await observer.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_failed_batch_that_ran_begin_reports_a_transaction(cluster_address: str) -> None:
    conn = DqliteConnection(cluster_address)
    try:
        await conn.connect()
        await conn.execute("DROP TABLE IF EXISTS test_mstx_flag")
        await conn.execute("CREATE TABLE test_mstx_flag (id INTEGER PRIMARY KEY)")
        with pytest.raises(OperationalError) as info:
            await conn.execute(FAILING_BATCH.format(t="test_mstx_flag"))
        assert info.value.code & 0xFF == SQLITE_CONSTRAINT
        assert conn.in_transaction is True
        await conn.execute("ROLLBACK")
        assert conn.in_transaction is False
    finally:
        await conn.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pooled_write_after_a_failed_batch_is_durable(cluster_addresses: list[str]) -> None:
    table = "test_mstx_pool"
    pool = ConnectionPool(cluster_addresses, min_size=0, max_size=1)
    try:
        await _fresh_table(pool, table)
        with pytest.raises(OperationalError) as info:
            await pool.execute(FAILING_BATCH.format(t=table))
        assert info.value.code & 0xFF == SQLITE_CONSTRAINT
        _, affected = await pool.execute(f"INSERT INTO {table} VALUES (2)")
        assert affected == 1
    finally:
        await pool.close()
    assert await _rows(cluster_addresses, table) == [2]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_control_same_statements_one_by_one_stay_durable(
    cluster_addresses: list[str],
) -> None:
    table = "test_mstx_control"
    pool = ConnectionPool(cluster_addresses, min_size=0, max_size=1)
    try:
        await _fresh_table(pool, table)
        async with pool.acquire() as conn:
            await conn.execute("BEGIN")
            await conn.execute(f"INSERT INTO {table} VALUES (1)")
            with pytest.raises(OperationalError):
                await conn.execute(f"INSERT INTO {table} VALUES (1)")
        _, affected = await pool.execute(f"INSERT INTO {table} VALUES (2)")
        assert affected == 1
    finally:
        await pool.close()
    assert await _rows(cluster_addresses, table) == [2]
