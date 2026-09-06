"""Keep simulated transfer and chunked consumption aligned with Raiden slot lifetime."""

from queue import Queue
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from sgl_jax.srt.disaggregation.encoder.sim_transfer import _SimReceivePool
from sgl_jax.srt.model_executor.simulation import SimulatedDevice, SimulationMixin
from sgl_jax.srt.multimodal.in_model.embedding_view import PooledEmbedding


def test_receive_slot_waits_for_transfer_and_consumer(monkeypatch):
    clock = [0]
    monkeypatch.setattr("time.monotonic_ns", lambda: clock[0])
    pool = _SimReceivePool(
        (1, 3),
        jnp.float32,
        jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        parallelism=1,
        capacity=1,
        timeout_s=0.01,
        ms_per_mb=0,
        rtt_ms=1,
    )
    try:
        pool.start("first").close()
        with pytest.raises(TimeoutError):
            pool.start("second")
        clock[0] += 1_000_000
        pool.progress()
        second = pool.start("second")
        clock[0] += 1_000_000
        embedding = second.poll()
        second.close()
        with pytest.raises(TimeoutError):
            pool.start("third")
        embedding.lease.release_after(jnp.zeros(()))
        pool.progress()
        pool.start("third").close()
    finally:
        pool.close()


def test_chunks_release_embedding_after_last_consumer():
    releases = []
    embedding = PooledEmbedding(
        None, 0, (1, 1), (1, 1), SimpleNamespace(release_after=releases.append)
    )
    runner = SimpleNamespace(
        _sim_device=SimulatedDevice(),
        _sim_completions=Queue(),
        _sim_duration_s=lambda batch: 0,
        simulation_logits_output=lambda batch: None,
        server_args=SimpleNamespace(disable_overlap_schedule=False),
    )
    for has_tail in (True, False):
        task = SimpleNamespace(
            item=SimpleNamespace(precomputed_embeddings=embedding), has_unmerged_tail=has_tail
        )
        batch = SimpleNamespace(
            batch_size=1,
            forward_mode=SimpleNamespace(is_decode=lambda: False),
            multimodal_batch={"image": (task,)},
        )
        SimulationMixin.simulation_forward(runner, batch)
        assert runner._sim_completions.get(timeout=1).wait(1)
        assert len(releases) == int(not has_tail)
