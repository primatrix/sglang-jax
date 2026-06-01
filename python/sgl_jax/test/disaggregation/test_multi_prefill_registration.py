"""Multi-prefill registration tests (Stage 3 multi-host RFC).

Simulates 4 P processes (one per host of a tp=16 P slice) each
calling :class:`BootstrapClient.register_prefill` with its own
``host_ip``. Asserts:

  * ``list_prefills`` returns all 4 entries with distinct host_ip.
  * ``get_prefill_info(room)`` for the same room returns the same P
    deterministically (so retries land on the same prefill peer).
  * Different rooms hash-balance across the 4 Ps.
"""

from __future__ import annotations

import socket
from collections import Counter

import pytest

from sgl_jax.srt.disaggregation.bootstrap import (
    BootstrapClient,
    BootstrapServer,
)


def _free_port() -> int:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def four_p_cluster():
    server = BootstrapServer("127.0.0.1", _free_port())
    server.start()
    client = BootstrapClient(f"http://127.0.0.1:{server.port}")
    # 4 hosts on a tp=16 P role.
    p_hosts = [
        ("p0", "10.0.0.10", 30001, 9600),
        ("p1", "10.0.0.11", 30001, 9600),
        ("p2", "10.0.0.12", 30001, 9600),
        ("p3", "10.0.0.13", 30001, 9600),
    ]
    for key, host, tp, scp in p_hosts:
        client.register_prefill(
            bootstrap_key=key, host=host,
            transfer_port=tp, side_channel_port=scp,
            tp_rank=int(key[1]), tp_size=4, system_dp_rank=0,
        )
    yield server, client, p_hosts
    server.stop()


def test_all_four_prefills_visible(four_p_cluster):
    _, client, p_hosts = four_p_cluster
    plist = client.list_prefills()
    assert len(plist) == 4
    seen_hosts = {p["host"] for p in plist}
    assert seen_hosts == {h for _, h, _, _ in p_hosts}


def test_same_room_returns_same_prefill(four_p_cluster):
    _, client, _ = four_p_cluster
    info_a = client.get_prefill_info(bootstrap_room=12345)
    info_b = client.get_prefill_info(bootstrap_room=12345)
    info_c = client.get_prefill_info(bootstrap_room=12345)
    assert info_a == info_b == info_c


def test_rooms_distribute_across_all_prefills(four_p_cluster):
    _, client, _ = four_p_cluster
    seen = Counter()
    for room in range(400):
        info = client.get_prefill_info(bootstrap_room=room)
        seen[info["bootstrap_key"]] += 1
    # All 4 P reached.
    assert set(seen) == {"p0", "p1", "p2", "p3"}
    # Uniform-ish — each at least 50 hits in 400 rooms (room % 4 is
    # uniform, but the registry hashes by sorted keys so depending on
    # ordering the bucket counts are exactly 100 each).
    for key, count in seen.items():
        assert count >= 50, f"P {key} got only {count} / 400 rooms"


def test_per_host_ports_are_preserved(four_p_cluster):
    _, client, _ = four_p_cluster
    plist = client.list_prefills()
    by_host = {p["host"]: p for p in plist}
    for key in ("p0", "p1", "p2", "p3"):
        host = "10.0.0.1" + key[1]
        assert by_host[host]["transfer_port"] == 30001
        assert by_host[host]["side_channel_port"] == 9600
        # tp_rank preserved per-host (used by D for any per-rank routing
        # logic Stage 4 may introduce).
        assert by_host[host]["tp_rank"] == int(key[1])


def test_re_register_replaces_only_that_host(four_p_cluster):
    """A P process restart re-registers with same key; the entry
    must be updated, not duplicated.
    """

    _, client, _ = four_p_cluster
    # p1 restarts and registers with a new transfer_port.
    client.register_prefill(
        bootstrap_key="p1", host="10.0.0.11",
        transfer_port=30002, side_channel_port=9601,
        tp_rank=1, tp_size=4,
    )
    plist = client.list_prefills()
    assert len(plist) == 4
    by_key = {p["bootstrap_key"]: p for p in plist}
    assert by_key["p1"]["transfer_port"] == 30002
    assert by_key["p1"]["side_channel_port"] == 9601
    # The other Ps unchanged.
    assert by_key["p0"]["transfer_port"] == 30001
