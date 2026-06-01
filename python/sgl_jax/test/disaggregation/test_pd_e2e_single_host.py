"""End-to-end PD smoke test (single host, two threads, no real model).

The full scheduler boots a model + tp_worker stack that needs a TPU;
we can't run that on CI. Instead this test exercises the PD wire
contract end-to-end:

  * real ``BootstrapServer`` (FastAPI in a background uvicorn thread)
  * real ``ZmqPullNotifier`` pair
  * real ``JaxTransferKVManager`` with mocked underlying wrapper
    (CPU jaxlib lacks ``jax.experimental.transfer.TransferConnection``)
  * deterministic "fake prefill" → KV; deterministic "fake decode" →
    output tokens
  * P registers via ``BootstrapClient``; D looks up via
    ``bootstrap_room`` and resolves the prefill peer
  * KV transfer drives the sender to SUCCESS and frees the buffer

What this does NOT cover:
  * Real model invocation (manual TPU e2e proves that)
  * ``Scheduler`` class composition (a separate unit test below
    asserts the Mixins compose cleanly)
  * ``--disaggregation-mode`` CLI parsing (covered by
    ``test_server_args_disaggregation.py``)
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import types
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.disaggregation.bootstrap import (
    BootstrapClient,
    BootstrapServer,
)
from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVManager,
    PMetadata,
)
from sgl_jax.srt.disaggregation.jax_transfer.zmq_notifier import (
    ZmqPullNotifier,
)
from sgl_jax.srt.disaggregation import jax_transfer_wrapper as jtw_mod
from sgl_jax.srt.disaggregation.jax_transfer_wrapper import JaxTransferWrapper


def _free_port() -> int:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _shim_transfer_module():
    """Inject a fake ``jax.experimental.transfer`` so wrapper.start()
    works on CPU jaxlib (the real module fails to import). The fake
    server records await_pull / supports a synchronous pull that
    returns whatever P registered for the same uuid.
    """

    pending: dict = {}

    def make_server():
        server = mock.MagicMock()

        def await_pull(uuid_int, data):
            pending[uuid_int] = data

        def connect(addr):
            link = mock.MagicMock()

            def pull(uuid_int, specs):
                # ``pending`` is process-shared because both P and D
                # use the same fake module. Return P's registered data
                # as-is; the spec is only used for the lazy-array
                # wrapper.
                data = pending[uuid_int]
                # Match real API: return list of lazy arrays.
                arr = mock.MagicMock()
                arr.is_ready = mock.MagicMock(return_value=True)
                arr.shape = data.shape
                arr.dtype = data.dtype
                arr.addressable_shards = data.addressable_shards
                arr.addressable_data = data.addressable_data
                return [arr]

            link.pull = pull
            return link

        server.await_pull.side_effect = await_pull
        server.connect.side_effect = connect
        return server

    fake_mod = types.ModuleType("jax.experimental.transfer")
    fake_mod.start_transfer_server = mock.MagicMock(side_effect=lambda *a, **k: make_server())
    fake_mod._pending = pending  # so the test can inspect
    return mock.patch.dict(sys.modules, {"jax.experimental.transfer": fake_mod})


@pytest.fixture(autouse=True)
def _reset_singleton():
    jtw_mod._reset_singleton_for_test()
    yield
    jtw_mod._reset_singleton_for_test()


def _device_sharding():
    devices = jax.local_devices()
    mesh = jax.sharding.Mesh(
        np.asarray(devices[:1]).reshape(1), axis_names=("x",)
    )
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def _fake_prefill_kv(input_ids: list[int]) -> jax.Array:
    """Deterministic 'KV' for a prompt: just a function of input_ids."""

    rng = np.random.default_rng(hash(tuple(input_ids)) & 0xFFFFFFFF)
    return jax.device_put(
        jnp.asarray(rng.integers(0, 256, size=(64,), dtype=np.int32).astype(np.float32)),
        _device_sharding(),
    )


def _fake_decode(kv: jax.Array, steps: int) -> list[int]:
    """Deterministic 'tokens' from KV: hash of KV bytes + step."""

    # Pull KV bytes via the Stage 1 slice workaround.
    n_shards = len(kv.addressable_shards)
    parts = []
    for i in range(n_shards):
        sub = kv.addressable_data(i)[: kv.shape[0] // n_shards]
        parts.append(np.asarray(jax.device_get(sub)).tobytes())
    kv_bytes = b"".join(parts)
    base = sum(kv_bytes) & 0xFF
    return [(base + step) & 0xFF for step in range(steps)]


def test_pd_wire_flow_e2e():
    """Drive the full PD wire flow with two in-process threads.

    Same process so we can share the fake ``jax.experimental.transfer``
    module (the pending dict is process-global). Two real
    ``JaxTransferKVManager`` instances backed by two real
    ``ZmqPullNotifier``s.
    """

    bootstrap_port = _free_port()
    p_transfer_port = _free_port()
    p_side_channel_port = _free_port()
    d_transfer_port = _free_port()
    d_side_channel_port = _free_port()
    bootstrap_room = 12345
    prompt_input_ids = [101, 7592, 1024]  # "hi"
    expected_tokens = _fake_decode(
        _fake_prefill_kv(prompt_input_ids), steps=4
    )

    server = BootstrapServer("127.0.0.1", bootstrap_port)
    server.start()
    try:
        bootstrap_url = f"http://127.0.0.1:{bootstrap_port}"

        with _shim_transfer_module():
            # ---------- P side ----------
            p_wrapper = JaxTransferWrapper("127.0.0.1", p_transfer_port)
            with mock.patch.object(
                jtw_mod.jax, "local_devices",
                return_value=[mock.MagicMock()],
            ):
                p_wrapper.start()
            jtw_mod._reset_singleton_for_test()  # let D get its own
            p_notifier = ZmqPullNotifier(
                "prefill", "127.0.0.1", p_side_channel_port
            )
            p_notifier.start()
            p_mgr = JaxTransferKVManager(p_wrapper, p_notifier)
            p_client = BootstrapClient(bootstrap_url)
            p_key = f"p-{p_transfer_port}"
            p_client.register_prefill(
                bootstrap_key=p_key,
                host="127.0.0.1",
                transfer_port=p_transfer_port,
                side_channel_port=p_side_channel_port,
            )

            # ---------- D side ----------
            d_wrapper = JaxTransferWrapper("127.0.0.1", d_transfer_port)
            with mock.patch.object(
                jtw_mod.jax, "local_devices",
                return_value=[mock.MagicMock()],
            ):
                d_wrapper.start()
            d_notifier = ZmqPullNotifier(
                "decode", "127.0.0.1", d_side_channel_port
            )
            d_notifier.start()
            d_mgr = JaxTransferKVManager(d_wrapper, d_notifier)
            d_client = BootstrapClient(bootstrap_url)

            try:
                # ---------- Drive one request end-to-end ----------
                req_id = f"req-{bootstrap_room}"

                # P: fake prefill, then send.
                kv = _fake_prefill_kv(prompt_input_ids)
                sender = p_mgr.create_sender(req_id)
                sender.init(kv_indices=None)
                sender.attach_payload(kv, use_d2h_staging=False)
                sender.send()

                # D: bootstrap lookup → connect → receiver pull
                p_info = d_client.get_prefill_info(bootstrap_room)
                assert p_info["bootstrap_key"] == p_key

                spec = jax.ShapeDtypeStruct(
                    (64,), jnp.float32, sharding=_device_sharding()
                )
                metadata = PMetadata(
                    remote_addr=(
                        f"{p_info['host']}:{p_info['transfer_port']}"
                    ),
                    uuid=req_id,
                    spec=spec,
                    p_side_channel_host=str(p_info["host"]),
                    p_side_channel_port=int(p_info["side_channel_port"]),
                )
                receiver = d_mgr.create_receiver(req_id)
                receiver.init(metadata)
                deadline = time.perf_counter() + 5.0
                while True:
                    state = receiver.poll()
                    if state == KVPoll.SUCCESS:
                        break
                    if state == KVPoll.FAILED:
                        pytest.fail(f"receiver state={state}")
                    if time.perf_counter() > deadline:
                        pytest.fail(
                            f"receiver stuck at {state} after 5s"
                        )
                    time.sleep(0.005)

                # P's sender should have received the ack and gone
                # SUCCESS via the ZMQ listener thread.
                deadline = time.perf_counter() + 5.0
                while sender.poll() != KVPoll.SUCCESS:
                    if time.perf_counter() > deadline:
                        pytest.fail(
                            f"sender stuck at {sender.poll()}"
                        )
                    time.sleep(0.005)

                # D: fake decode from received KV
                got_tokens = _fake_decode(receiver.result, steps=4)
                assert got_tokens == expected_tokens, (
                    f"PD decode {got_tokens} != baseline "
                    f"{expected_tokens}"
                )
            finally:
                d_notifier.stop()
                p_notifier.stop()
                p_client.unregister_prefill(p_key)
    finally:
        server.stop()


def test_scheduler_composes_disaggregation_mixins():
    """Static check: the Scheduler class declares both PD Mixins in
    its MRO, so a future scheduler instance will have the event-loop
    methods + queue installation hooks available.

    We can't instantiate Scheduler on CPU (it loads a model + tp
    worker), so we inspect the class object directly.
    """

    from sgl_jax.srt.managers.scheduler import Scheduler
    from sgl_jax.srt.disaggregation.decode import (
        SchedulerDisaggregationDecodeMixin,
    )
    from sgl_jax.srt.disaggregation.prefill import (
        SchedulerDisaggregationPrefillMixin,
    )

    assert issubclass(Scheduler, SchedulerDisaggregationPrefillMixin)
    assert issubclass(Scheduler, SchedulerDisaggregationDecodeMixin)
    assert hasattr(Scheduler, "event_loop_normal_disagg_prefill")
    assert hasattr(Scheduler, "event_loop_normal_disagg_decode")
    assert hasattr(Scheduler, "process_prefill_chunk")
    assert hasattr(Scheduler, "process_decode_queue")
    assert hasattr(Scheduler, "send_kv_chunk")
