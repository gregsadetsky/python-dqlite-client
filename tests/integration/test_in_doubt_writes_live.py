"""Live integration: a write whose outcome is in doubt raises AmbiguousCommitError.

The faults come from dqlitetestlib's NetworkFaults (python-dqlite-dev): the leader's
replies to one client connection are dropped, or a one-way partition makes the leader
step down. Where the write is applied the error must not read as a clean failure; a
write inside a transaction and a read keep their plain error.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import pytest

from dqliteclient import (
    AmbiguousCommitError,
    ConnectionPool,
    DqliteConnection,
    DqliteConnectionError,
    create_pool,
)

if TYPE_CHECKING:
    from dqlitetestlib import NetworkFaults, TestClusterControl  # type: ignore[import-not-found]


def _port(address: str) -> int:
    return int(address.rsplit(":", 1)[1])


def _client_port(conn: DqliteConnection) -> int:
    assert conn._protocol is not None
    port: int = conn._protocol._writer.get_extra_info("sockname")[1]
    return port


@contextlib.asynccontextmanager
async def _pool(addresses: list[str], timeout: float = 3) -> AsyncIterator[ConnectionPool]:
    pool = await create_pool(
        addresses, database="in_doubt_" + uuid.uuid4().hex, min_size=0, max_size=1, timeout=timeout
    )
    try:
        async with pool.acquire() as conn:
            await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        yield pool
    finally:
        await pool.close()


async def _rows(pool: ConnectionPool) -> list[Any]:
    last: Exception | None = None
    for _ in range(60):
        try:
            return await pool.fetchall("SELECT id FROM t")
        except Exception as exc:  # noqa: BLE001 - a new leader is still being elected
            last = exc
            await asyncio.sleep(0.5)
    raise AssertionError(f"cluster did not serve reads again: {last!r}")


@pytest.mark.integration
async def test_pooled_autocommit_write_whose_reply_is_lost(
    cluster_node_addresses: list[str], network_faults: NetworkFaults
) -> None:
    async with _pool(cluster_node_addresses) as pool:
        async with pool.acquire() as conn:
            with (
                network_faults.drop_replies(_port(conn.address), _client_port(conn)),
                pytest.raises(AmbiguousCommitError) as info,
            ):
                await conn.execute("INSERT INTO t VALUES (42)")
        assert isinstance(info.value, DqliteConnectionError)
        assert await _rows(pool) == [[42]]


@pytest.mark.integration
async def test_transaction_commit_whose_reply_is_lost(
    cluster_node_addresses: list[str], network_faults: NetworkFaults
) -> None:
    async with _pool(cluster_node_addresses) as pool:
        async with pool.acquire() as conn:
            with contextlib.ExitStack() as faults, pytest.raises(AmbiguousCommitError):
                async with conn.transaction():
                    await conn.execute("INSERT INTO t VALUES (7)")
                    faults.enter_context(
                        network_faults.drop_replies(_port(conn.address), _client_port(conn))
                    )
        assert await _rows(pool) == [[7]]


@pytest.mark.integration
async def test_write_inside_a_transaction_whose_reply_is_lost_is_plain(
    cluster_node_addresses: list[str], network_faults: NetworkFaults
) -> None:
    """The server's transaction ends with the session: nothing is applied."""
    async with _pool(cluster_node_addresses) as pool:
        async with pool.acquire() as conn:
            with pytest.raises(DqliteConnectionError) as info:
                async with conn.transaction():
                    with network_faults.drop_replies(_port(conn.address), _client_port(conn)):
                        await conn.execute("INSERT INTO t VALUES (3)")
        assert not isinstance(info.value, AmbiguousCommitError)
        assert await _rows(pool) == []


@pytest.mark.integration
async def test_read_whose_reply_is_lost_is_plain(
    cluster_node_addresses: list[str], network_faults: NetworkFaults
) -> None:
    async with _pool(cluster_node_addresses) as pool:
        async with pool.acquire() as conn:
            with (
                network_faults.drop_replies(_port(conn.address), _client_port(conn)),
                pytest.raises(DqliteConnectionError) as info,
            ):
                await conn.fetchall("SELECT id FROM t")
        assert not isinstance(info.value, AmbiguousCommitError)


@pytest.mark.integration
async def test_autocommit_write_when_leadership_is_lost(
    cluster_node_addresses: list[str],
    cluster_control: TestClusterControl,
    network_faults: NetworkFaults,
) -> None:
    original = await cluster_control.current_leader_node()
    try:
        async with _pool(cluster_node_addresses, timeout=15) as pool:
            async with pool.acquire() as conn:
                with (
                    network_faults.isolate_from_followers(_port(conn.address)),
                    pytest.raises(AmbiguousCommitError),
                ):
                    await conn.execute("INSERT INTO t VALUES (5)")
            assert await _rows(pool) == [[5]]
    finally:
        for _ in range(60):
            try:
                if await cluster_control.find_leader() == original.address:
                    break
                await cluster_control.transfer_leadership_to(original.node_id)
            except Exception:  # noqa: BLE001 - mid-election
                pass
            await asyncio.sleep(0.5)
        else:
            raise AssertionError(f"leadership did not return to {original.address}")
