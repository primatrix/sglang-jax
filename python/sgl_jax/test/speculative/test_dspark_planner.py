import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.speculative.dspark_planner import (
    allocate_dspark_verify_lens,
    compact_dspark_verify_inputs,
    scatter_dspark_compact_rows,
    select_dspark_verify_budget,
)
from sgl_jax.srt.speculative.dspark_tuned_config import (
    DSparkSPSPoint,
    DSparkSPSProfile,
)
from sgl_jax.srt.speculative.relay_buffer import (
    DSparkConfidenceRelayHost,
    create_dspark_confidence_relay_buffers,
    update_dspark_confidence_relay_buffers,
)


def test_budget_planner_maximizes_expected_tokens_per_step_time():
    profile = DSparkSPSProfile(
        context_bucket=1024,
        points=(
            DSparkSPSPoint(2, 10.0, 100.0),
            DSparkSPSPoint(4, 8.0, 125.0),
            DSparkSPSPoint(6, 20.0, 50.0),
        ),
    )
    decision = select_dspark_verify_budget(
        profile,
        np.array([[0.9, 0.5], [0.8, 0.2]], dtype=np.float32),
    )

    assert decision.token_bucket == 4
    assert decision.extra_budget == 2
    assert np.isclose(decision.expected_tokens, 3.7)


def test_budget_planner_selects_smallest_covering_request_bucket():
    profile = DSparkSPSProfile(
        context_bucket=1024,
        points=(
            DSparkSPSPoint(16, 1.0, 1000.0, request_bucket_per_dp=8),
            DSparkSPSPoint(32, 2.0, 500.0, request_bucket_per_dp=8),
            DSparkSPSPoint(16, 100.0, 10.0, request_bucket_per_dp=16),
            DSparkSPSPoint(32, 10.0, 100.0, request_bucket_per_dp=16),
            DSparkSPSPoint(64, 20.0, 50.0, request_bucket_per_dp=16),
        ),
    )

    decision = select_dspark_verify_budget(
        profile,
        np.ones((9, 3), dtype=np.float32),
    )

    assert decision.token_bucket == 32
    assert decision.extra_budget == 23


def test_budget_planner_can_force_token_bucket_for_tpu_collection():
    profile = DSparkSPSProfile(
        context_bucket=1024,
        points=(
            DSparkSPSPoint(8, 1.0, 1000.0),
            DSparkSPSPoint(16, 2.0, 500.0),
        ),
    )

    decision = select_dspark_verify_budget(
        profile,
        np.ones((2, 7), dtype=np.float32),
        forced_token_bucket=16,
    )

    assert decision.token_bucket == 16
    assert decision.extra_budget == 14


def test_forced_bucket_allows_static_padding_above_live_verify_all():
    profile = DSparkSPSProfile(
        context_bucket=1024,
        points=(DSparkSPSPoint(256, 10.0, 100.0),),
    )

    decision = select_dspark_verify_budget(
        profile,
        np.ones((31, 7), dtype=np.float32),
        forced_token_bucket=256,
    )

    assert decision.token_bucket == 256
    assert decision.extra_budget == 217
    assert decision.expected_tokens == 248


def test_budget_planner_rejects_request_count_above_2d_table():
    profile = DSparkSPSProfile(
        context_bucket=1024,
        points=(DSparkSPSPoint(16, 1.0, 1000.0, request_bucket_per_dp=8),),
    )

    assert (
        select_dspark_verify_budget(
            profile,
            np.ones((9, 7), dtype=np.float32),
        )
        is None
    )


def test_verify_len_allocation_preserves_prefix_and_tie_break():
    verify_lens = allocate_dspark_verify_lens(
        jnp.ones((2, 2), dtype=jnp.float32),
        jnp.array([True, True]),
        jnp.array([3], dtype=jnp.int32),
        dp_size=1,
    )

    # Position-major stable ordering selects both first positions before the
    # first request's second position.
    np.testing.assert_array_equal(np.asarray(verify_lens), np.array([3, 2]))


def test_compact_verify_inputs_keep_request_and_position_order():
    verify_lens = jnp.array([3, 1], dtype=jnp.int32)
    values = jnp.arange(8, dtype=jnp.int32)
    ids, positions, cache_loc, compact_to_logical, valid = compact_dspark_verify_inputs(
        values,
        values + 10,
        values + 20,
        verify_lens,
        dp_size=1,
        verify_width=4,
        per_dp_token_bucket=4,
    )

    np.testing.assert_array_equal(np.asarray(ids), np.array([0, 1, 2, 4]))
    np.testing.assert_array_equal(np.asarray(positions), np.array([10, 11, 12, 14]))
    np.testing.assert_array_equal(np.asarray(cache_loc), np.array([20, 21, 22, 24]))
    np.testing.assert_array_equal(np.asarray(compact_to_logical), np.array([0, 1, 2, 4]))
    assert np.asarray(valid).all()

    restored = scatter_dspark_compact_rows(ids + 100, compact_to_logical, logical_size=8)
    np.testing.assert_array_equal(
        np.asarray(restored),
        np.array([100, 101, 102, 0, 104, 0, 0, 0]),
    )


def test_compact_verify_inputs_keep_dp_rank_segments_and_padding():
    values = jnp.arange(16, dtype=jnp.int32)
    ids, _, cache_loc, compact_to_logical, valid = compact_dspark_verify_inputs(
        values,
        values,
        values,
        jnp.array([2, 0, 1, 2], dtype=jnp.int32),
        dp_size=2,
        verify_width=4,
        per_dp_token_bucket=3,
    )

    np.testing.assert_array_equal(np.asarray(ids), np.array([0, 1, 0, 8, 12, 13]))
    np.testing.assert_array_equal(np.asarray(cache_loc), np.array([0, 1, -1, 8, 12, 13]))
    np.testing.assert_array_equal(
        np.asarray(compact_to_logical),
        np.array([0, 1, 16, 8, 12, 13]),
    )
    np.testing.assert_array_equal(
        np.asarray(valid),
        np.array([True, True, False, True, True, True]),
    )


def test_planner_compaction_and_scatter_keep_explicit_dp_sharding():
    if len(jax.devices()) < 2:
        pytest.skip("requires two CPU or TPU devices")

    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = np.asarray(jax.devices()[:2]).reshape((2, 1))
    mesh = Mesh(
        devices,
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    vector = NamedSharding(mesh, P("data"))
    matrix = NamedSharding(mesh, P("data", None))
    confidence = jax.device_put(jnp.ones((4, 3), dtype=jnp.float32), matrix)
    active = jax.device_put(jnp.array([True, False, True, True]), vector)
    budgets = jax.device_put(jnp.array([1, 2], dtype=jnp.int32), vector)

    verify_lens = allocate_dspark_verify_lens(
        confidence,
        active,
        budgets,
        dp_size=2,
    )
    values = jax.device_put(jnp.arange(16, dtype=jnp.int32), vector)
    ids, _, _, mapping, _ = compact_dspark_verify_inputs(
        values,
        values,
        values,
        verify_lens,
        dp_size=2,
        verify_width=4,
        per_dp_token_bucket=4,
    )
    restored = scatter_dspark_compact_rows(ids, mapping, logical_size=16)

    assert verify_lens.sharding.spec == P("data")
    assert ids.sharding.spec == P("data")
    assert restored.sharding.spec == P("data")
    np.testing.assert_array_equal(np.asarray(verify_lens), np.array([2, 0, 2, 2]))


def test_confidence_relay_reads_exact_lag_two_and_rejects_reused_slot():
    from jax.sharding import Mesh

    mesh = Mesh(
        np.asarray(jax.devices()).reshape((1, -1)),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    pool = type("Pool", (), {"req_to_token": np.zeros((4, 8), dtype=np.int32)})()
    with jax.set_mesh(mesh):
        buffers = create_dspark_confidence_relay_buffers(
            mesh,
            pool,
            dp_size=1,
            gamma=2,
        )
        buffers = update_dspark_confidence_relay_buffers(
            buffers,
            jnp.array([2], dtype=jnp.int32),
            jnp.array([7], dtype=jnp.int32),
            jnp.array([1], dtype=jnp.int32),
            jnp.array([True]),
            jnp.array([[0.8, 0.5]], dtype=jnp.float32),
            dp_size=1,
        )

    host = DSparkConfidenceRelayHost(dp_size=1, capacity=4, gamma=2)
    host.publish(buffers)
    host.wait_for_pending_for_test()
    gathered, stats = host.gather_lagged_confidence(
        np.array([2]),
        np.array([7]),
        np.array([3]),
        np.array([True]),
    )
    np.testing.assert_allclose(gathered, np.array([[0.8, 0.5]], dtype=np.float32))
    assert stats["hit"] == 1

    reused, stats = host.gather_lagged_confidence(
        np.array([2]),
        np.array([8]),
        np.array([3]),
        np.array([True]),
    )
    np.testing.assert_array_equal(reused, np.ones((1, 2), dtype=np.float32))
    assert stats["stale_generation"] == 1


def test_confidence_relay_warmup_and_ring_round_tags_fall_back():
    host = DSparkConfidenceRelayHost(dp_size=1, capacity=2, gamma=2)
    confidence, stats = host.gather_lagged_confidence(
        np.array([0]),
        np.array([1]),
        np.array([1]),
        np.array([True]),
    )
    np.testing.assert_array_equal(confidence, np.ones((1, 2), dtype=np.float32))
    assert stats["stale_warmup"] == 1

    # No matching source-round tag is present even though the ring slot exists.
    confidence, stats = host.gather_lagged_confidence(
        np.array([0]),
        np.array([-1]),
        np.array([3]),
        np.array([True]),
    )
    np.testing.assert_array_equal(confidence, np.ones((1, 2), dtype=np.float32))
    assert stats["stale_not_ready"] == 1


def test_confidence_relay_update_keeps_dp_rank_segments_under_jit():
    from functools import partial

    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = np.asarray(jax.devices()[:4]).reshape((2, 2))
    mesh = Mesh(
        devices,
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    pool = type("Pool", (), {"req_to_token": np.zeros((4, 8), dtype=np.int32)})()
    vector = NamedSharding(mesh, P("data"))
    matrix = NamedSharding(mesh, P("data", None))

    @partial(jax.jit, static_argnames=["dp_size"])
    def update(buffers, indices, generations, rounds, valid, confidence, *, dp_size):
        return update_dspark_confidence_relay_buffers(
            buffers,
            indices,
            generations,
            rounds,
            valid,
            confidence,
            dp_size=dp_size,
        )

    with jax.set_mesh(mesh):
        buffers = create_dspark_confidence_relay_buffers(
            mesh,
            pool,
            dp_size=2,
            gamma=2,
        )
        buffers = update(
            buffers,
            jax.device_put(jnp.array([0, 1, 2, 3]), vector),
            jax.device_put(jnp.array([5, 5, 9, 9]), vector),
            jax.device_put(jnp.array([1, 1, 4, 4]), vector),
            jax.device_put(jnp.array([True, False, True, True]), vector),
            jax.device_put(
                jnp.array([[0.9, 0.8], [0.1, 0.1], [0.7, 0.6], [0.5, 0.4]]),
                matrix,
            ),
            dp_size=2,
        )

    host = DSparkConfidenceRelayHost(dp_size=2, capacity=4, gamma=2)
    host.publish(buffers)
    host.wait_for_pending_for_test()
    gathered, stats = host.gather_lagged_confidence(
        np.array([0, 1, 2, 3]),
        np.array([5, 5, 9, 9]),
        np.array([3, 3, 6, 6]),
        np.array([True, False, True, True]),
    )
    np.testing.assert_allclose(
        gathered,
        np.array([[0.9, 0.8], [1.0, 1.0], [0.7, 0.6], [0.5, 0.4]]),
    )
    assert stats["hit"] == 3
