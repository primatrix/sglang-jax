import argparse
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_dsa_metadata_builds_causal_candidates_for_decode_and_prefill():
    from sgl_jax.srt.layers.attention.dsa_backend import DsaAttentionBackend
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

    backend = DsaAttentionBackend(
        num_attn_heads=1,
        kv_lora_rank=3,
        qk_nope_head_dim=3,
        qk_rope_head_dim=2,
        v_head_dim=3,
        index_head_dim=2,
        index_topk=2,
        page_size=4,
        mesh=None,
    )
    decode = SimpleNamespace(
        dp_size=1,
        forward_mode=ForwardMode.DECODE,
        seq_lens=np.array([3, 2], dtype=np.int32),
        cache_loc=np.array([10, 11, 12, 0, 20, 21, 0, 0], dtype=np.int32),
        input_ids=np.array([7, 8], dtype=np.int32),
        positions=np.array([2, 1], dtype=np.int32),
        extend_seq_lens=None,
    )

    metadata = backend.get_forward_metadata(decode)

    np.testing.assert_array_equal(
        metadata.req_to_token_slots,
        np.array([[10, 11, 12, 0], [20, 21, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(metadata.query_request_indices, [0, 1])
    np.testing.assert_array_equal(metadata.query_positions, [2, 1])
    np.testing.assert_array_equal(metadata.query_offsets, [0, 1, 2])
    np.testing.assert_array_equal(metadata.request_offsets, [0, 4, 8])

    backend.mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(1, 1),
        ("data", "tensor"),
    )
    sharded_metadata = backend.get_forward_metadata(decode)
    assert sharded_metadata.req_to_token_slots.sharding.spec == jax.sharding.PartitionSpec(
        "data", None
    )

    prefill = SimpleNamespace(
        dp_size=1,
        forward_mode=ForwardMode.EXTEND,
        seq_lens=np.array([3, 2], dtype=np.int32),
        cache_loc=decode.cache_loc,
        input_ids=np.array([1, 2, 3, 4, 0], dtype=np.int32),
        positions=np.array([1, 2, 0, 1, 0], dtype=np.int32),
        extend_seq_lens=np.array([2, 2], dtype=np.int32),
    )

    metadata = backend.get_forward_metadata(prefill)

    np.testing.assert_array_equal(metadata.query_request_indices, [0, 0, 1, 1, -1])
    np.testing.assert_array_equal(metadata.query_positions, [1, 2, 0, 1, 0])
    np.testing.assert_array_equal(metadata.query_offsets, [0, 2, 4])

    for unsupported_mode in (
        ForwardMode.MIXED,
        ForwardMode.TARGET_VERIFY,
        ForwardMode.DRAFT_EXTEND,
    ):
        unsupported = SimpleNamespace(**prefill.__dict__)
        unsupported.forward_mode = unsupported_mode
        with pytest.raises(ValueError, match="supports only ordinary EXTEND and DECODE"):
            backend.get_forward_metadata(unsupported)


def test_dsa_backend_writes_index_cache_then_builds_physical_selection():
    from sgl_jax.srt.kernels.dsa.reference import write_indexer_k_cache
    from sgl_jax.srt.layers.attention.dsa_backend import (
        DsaAttentionBackend,
        DsaAttentionMetadata,
    )
    from sgl_jax.srt.mem_cache.dsa_pool import DsaIndexerKPool

    backend = DsaAttentionBackend(
        num_attn_heads=1,
        kv_lora_rank=3,
        qk_nope_head_dim=3,
        qk_rope_head_dim=2,
        v_head_dim=3,
        index_head_dim=2,
        index_topk=2,
        page_size=4,
        mesh=None,
    )
    backend.forward_metadata = DsaAttentionMetadata(
        req_to_token_slots=jnp.array([[0, 1]], dtype=jnp.int32),
        query_request_indices=jnp.array([0], dtype=jnp.int32),
        query_positions=jnp.array([1], dtype=jnp.int32),
        query_offsets=jnp.array([0, 1], dtype=jnp.int32),
        request_offsets=jnp.array([0, 2], dtype=jnp.int32),
    )
    pool = DsaIndexerKPool(
        size=8,
        page_size=4,
        index_head_dim=2,
        layer_num=1,
        layer_ids=(0,),
        mesh=None,
    )
    pool.k_buffer[0] = write_indexer_k_cache(
        pool.k_buffer[0],
        index_k=jnp.array([[1.0, 0.0]], dtype=jnp.float32),
        write_slots=jnp.array([0], dtype=jnp.int32),
        page_size=4,
        index_head_dim=2,
    )
    forward_batch = SimpleNamespace(out_cache_loc=jnp.array([1], dtype=jnp.int32))

    state, updated = backend.build_dsa_state(
        layer_id=0,
        q_index=jnp.array([[[1.0, 0.0]]], dtype=jnp.float32),
        head_weights=jnp.array([[1.0]], dtype=jnp.float32),
        index_k=jnp.array([[0.5, 0.0]], dtype=jnp.float32),
        forward_batch=forward_batch,
        indexer_k_pool=pool,
        prev_dsa_state=None,
    )

    np.testing.assert_array_equal(state.selection.logical_topk_ids, [[0, 1]])
    np.testing.assert_array_equal(state.selection.physical_slots, [[0, 1]])
    np.testing.assert_array_equal(state.selection.selected_counts, [2])
    assert state.selection.producer_layer == 0
    np.testing.assert_array_equal(np.asarray(updated).reshape(-1, 128)[1, :2], [0.5, 0.0])

    jitted_state, jitted_updated = jax.jit(
        lambda query, weights, key, slots, index_pool: backend.build_dsa_state(
            layer_id=0,
            q_index=query,
            head_weights=weights,
            index_k=key,
            forward_batch=SimpleNamespace(out_cache_loc=slots),
            indexer_k_pool=index_pool,
            prev_dsa_state=None,
        )
    )(
        jnp.array([[[1.0, 0.0]]], dtype=jnp.float32),
        jnp.array([[1.0]], dtype=jnp.float32),
        jnp.array([[0.5, 0.0]], dtype=jnp.float32),
        jnp.array([1], dtype=jnp.int32),
        pool,
    )
    np.testing.assert_array_equal(jitted_state.selection.physical_slots, [[0, 1]])
    np.testing.assert_array_equal(np.asarray(jitted_updated).reshape(-1, 128)[1, :2], [0.5, 0.0])


def test_dsa_attention_writes_main_mla_cache_before_sparse_read():
    from sgl_jax.srt.layers.attention.dsa_backend import DsaAttentionBackend
    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection, DsaTopKState

    backend = DsaAttentionBackend(
        num_attn_heads=1,
        kv_lora_rank=3,
        qk_nope_head_dim=3,
        qk_rope_head_dim=2,
        v_head_dim=3,
        index_head_dim=2,
        index_topk=2,
        page_size=4,
        mesh=None,
    )
    cache = jnp.zeros((2, 2, 2, 256), dtype=jnp.bfloat16)

    class Pool:
        def get_fused_kv_buffer(self, layer_id):
            assert layer_id == 0
            return cache

    state = DsaTopKState(
        selection=DsaSelection(
            logical_topk_ids=jnp.array([[0]], dtype=jnp.int32),
            physical_slots=jnp.array([[0]], dtype=jnp.int32),
            selected_counts=jnp.array([1], dtype=jnp.int32),
            producer_layer=0,
        ),
        query_offsets=jnp.array([0, 1], dtype=jnp.int32),
        request_offsets=jnp.array([0, 1], dtype=jnp.int32),
    )
    q = jnp.array([[[0.5, 0.25, 1.0]]], dtype=jnp.bfloat16)
    new_c = jnp.array([[[2.0, -1.0, 0.5]]], dtype=jnp.bfloat16)
    q_rope = jnp.array([[[1.0, 0.5]]], dtype=jnp.bfloat16)
    new_rope = jnp.array([[[0.25, -0.5]]], dtype=jnp.bfloat16)

    output, updated = backend(
        q,
        new_c,
        new_c,
        layer=SimpleNamespace(layer_id=0, scaling=1.0),
        forward_batch=SimpleNamespace(out_cache_loc=jnp.array([0], dtype=jnp.int32)),
        token_to_kv_pool=Pool(),
        q_rope=q_rope,
        k_rope=new_rope,
        dsa_state=state,
    )

    np.testing.assert_array_equal(output, new_c)
    flattened = np.asarray(updated, dtype=np.float32).reshape(-1, 256)
    np.testing.assert_array_equal(flattened[0, :3], [2.0, -1.0, 0.5])
    np.testing.assert_array_equal(flattened[0, 128:130], [0.25, -0.5])


def test_dsa_cli_pool_cost_and_compact_full_layer_pool():
    from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
        ModelRunnerKVCacheMixin,
        _build_dsa_indexer_k_pool,
    )
    from sgl_jax.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy", "--attention-backend", "dsa"])
    assert args.attention_backend == "dsa"

    config = SimpleNamespace(
        kv_lora_rank=3,
        qk_rope_head_dim=2,
        index_head_dim=2,
        indexer_types=["full", "shared", "full"],
    )
    runner = SimpleNamespace(
        use_mla_backend=True,
        server_args=SimpleNamespace(attention_backend="dsa", page_size=4),
        model_config=SimpleNamespace(hf_text_config=config),
        kv_cache_dtype=jnp.bfloat16,
        max_total_num_tokens=8,
        page_size=4,
        mesh=None,
        _kv_pool_layer_count=lambda: 3,
    )

    assert ModelRunnerKVCacheMixin._compute_cell_size(runner) == 2048
    pool = _build_dsa_indexer_k_pool(runner, dp_size=1)
    assert pool.layer_ids == (0, 2)
    assert len(pool.k_buffer) == 2
    with pytest.raises(IndexError, match="no Index-K storage"):
        pool.get_buffer(1)


def test_model_runner_selects_dsa_backend_and_registers_both_pools():
    from sgl_jax.srt.layers.attention.dsa_backend import DsaAttentionBackend
    from sgl_jax.srt.model_executor.model_runner import ModelRunner
    from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
        ModelRunnerKVCacheMixin,
    )

    config = SimpleNamespace(
        kv_lora_rank=3,
        qk_nope_head_dim=3,
        qk_rope_head_dim=2,
        v_head_dim=3,
        index_head_dim=2,
        index_topk=2,
        indexer_types=["full", "shared", "full"],
    )
    runner = SimpleNamespace(
        server_args=SimpleNamespace(
            attention_backend="dsa",
            device="cpu",
            page_size=4,
        ),
        use_mla_backend=True,
        model_config=SimpleNamespace(hf_text_config=config),
        num_attn_heads=1,
        page_size=4,
        mesh=None,
        linear_recurrent_config=None,
    )

    backend = ModelRunner._get_attention_backend(runner)
    assert isinstance(backend, DsaAttentionBackend)

    token_pool = object()
    runner.max_total_num_tokens = 8
    runner.kv_cache_dtype = jnp.bfloat16
    runner.req_to_token_pool = object()
    runner.token_to_kv_pool_allocator = object()
    runner.is_hybrid = False
    runner._kv_pool_layer_count = lambda: 3
    runner._maybe_wrap_hybrid_kv_pool = lambda pool_class, **kwargs: token_pool

    ModelRunnerKVCacheMixin._init_pools(runner, max_num_reqs=2, dp_size=1)

    assert runner.memory_pools.token_to_kv_pool is token_pool
    assert runner.memory_pools.indexer_k_pool is runner.indexer_k_pool
    assert runner.indexer_k_pool.layer_ids == (0, 2)


def test_single_tp_sharding_restore_handles_dual_pool_update_dict(monkeypatch):
    from sgl_jax.srt.model_executor.model_runner import (
        _restore_single_tp_pool_update_shardings,
    )

    calls = []

    def fake_device_put(value, sharding):
        calls.append((value, sharding))
        return f"{value}@{sharding}"

    monkeypatch.setattr(jax, "device_put", fake_device_put)
    pools = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(kv_sharding="mla-sharding"),
        indexer_k_pool=SimpleNamespace(k_sharding="index-sharding"),
    )

    restored = _restore_single_tp_pool_update_shardings(
        {
            "token_to_kv_pool": ["mla-0", "mla-1"],
            "indexer_k_pool": ["index-0"],
        },
        pools,
    )

    assert restored == {
        "token_to_kv_pool": ["mla-0@mla-sharding", "mla-1@mla-sharding"],
        "indexer_k_pool": ["index-0@index-sharding"],
    }
    assert set(calls) == {
        ("index-0", "index-sharding"),
        ("mla-0", "mla-sharding"),
        ("mla-1", "mla-sharding"),
    }
