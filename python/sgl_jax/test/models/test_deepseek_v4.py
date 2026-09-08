"""Actual mixed-checkpoint loading and the SWA/CSA/HCA trunk graph."""

import json
import struct
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from sgl_jax.srt.configs.deepseek_v4 import DeepseekV4Config
from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.models.deepseek_v4 import DeepseekV4ForCausalLM, build_weight_mappings


def tiny_config():
    cfg = DeepseekV4Config(
        hidden_size=512,
        num_hidden_layers=3,
        num_attention_heads=2,
        head_dim=256,
        qk_rope_head_dim=64,
        q_lora_rank=256,
        o_groups=2,
        o_lora_rank=256,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hash_layers=1,
        moe_intermediate_size=512,
        vocab_size=32,
        index_n_heads=2,
        index_head_dim=256,
        index_topk=2,
        compress_ratios=[0, 4, 128],
        max_position_embeddings=256,
        sliding_window=16,
        expert_dtype="fp4",
    )
    cfg.quantization_config = QuantizationConfig(
        moe_weight_dtype=jnp.float8_e4m3fn, weight_block_size=(128, 128), is_static_checkpoint=True
    )
    return cfg


def get_param(model, path):
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def write_fixture(path, model):
    """Write checkpoint-layout payloads, including raw I8/E8M0 expert pairs."""
    tensors = {}
    rng = np.random.default_rng(17)
    for key, mapping in build_weight_mappings(model.config).items():
        p = get_param(model, mapping.target_path).value
        shape = p.shape
        if key.endswith(".scale"):
            linear = get_param(model, mapping.target_path.rsplit(".", 1)[0])
            n, k = linear.weight_q.value.shape
            value = np.full((n // 128, k // 128), 125, np.uint8)  # E8M0: 2**-2
            kind = "F8_E8M0"
        elif key.endswith("tid2eid"):
            value = np.tile(np.array([[3, 1], [0, 2]], np.int32), (16, 1))
            kind = "I32"
        else:
            shape = shape[::-1] if mapping.transpose else shape
            value = rng.normal(0, 0.05, shape).astype(np.float32)
            kind = "F32"
            if key.endswith("norm.weight") or key == "norm.weight":
                value.fill(1)
            if key.endswith("weight") and mapping.target_path.endswith("weight_q"):
                value = value.astype(jnp.float8_e4m3fn)
                kind = "F8_E4M3"
        if kind == "F32" and not key.endswith(".scale") and "hc_" not in key and "gate" not in key:
            value = value.astype(jnp.bfloat16)
            kind = "BF16"
        tensors[key] = (value, kind)
    for layer in range(3):
        for expert in range(4):
            hidden, intermediate = model.config.hidden_size, model.config.moe_intermediate_size
            for w, (n, k) in {
                "w1": (intermediate, hidden),
                "w3": (intermediate, hidden),
                "w2": (hidden, intermediate),
            }.items():
                stem = f"layers.{layer}.ffn.experts.{expert}.{w}"
                tensors[stem + ".weight"] = (np.full((n, k // 2), 0x21, np.int8), "I8")
                tensors[stem + ".scale"] = (np.full((n, k // 32), 119, np.uint8), "F8_E8M0")
    header = {}
    payload = []
    offset = 0
    for key, (value, kind) in tensors.items():
        raw = value.tobytes()
        header[key] = {
            "dtype": kind,
            "shape": list(value.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        payload.append(raw)
        offset += len(raw)
    raw = json.dumps(header).encode()
    raw += b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(payload))


def make_model(dp=1, tp=1, cfg=None):
    if jax.device_count() < dp * tp:
        pytest.skip("requires four devices")
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[: dp * tp]).reshape(dp, tp),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    cfg = tiny_config() if cfg is None else cfg
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(lambda: DeepseekV4ForCausalLM(cfg, mesh))
    return model, mesh


@pytest.mark.parametrize("dp,tp", [(1, 1), (2, 2)])
def test_real_mixed_checkpoint_load(tmp_path, dp, tp, monkeypatch):
    model, mesh = make_model(dp, tp)
    write_fixture(tmp_path / "model.safetensors", model)
    from sgl_jax.srt.utils.quantization import mxfp4_fp8_loader

    convert = mxfp4_fp8_loader.convert_mxfp4_pair_from_safetensors
    calls = []

    def counted_convert(*args, **kwargs):
        calls.append(args[1])
        return convert(*args, **kwargs)

    monkeypatch.setattr(mxfp4_fp8_loader, "convert_mxfp4_pair_from_safetensors", counted_convert)
    with jax.set_mesh(mesh):
        model.load_weights(SimpleNamespace(model_path=str(tmp_path)))
    assert len(calls) == len(set(calls)) == 3 * 4 * 3
    for _, p in nnx.state(model, nnx.Param).flat_state():
        assert isinstance(p.value, jax.Array)
        assert np.isfinite(np.asarray(p.value, dtype=np.float32)).all()
    layer = model.model.layers[0]
    np.testing.assert_array_equal(layer.mlp.gate.tid2eid.value[:2], [[3, 1], [0, 2]])
    w = np.asarray(layer.mlp.experts.wi_0.value, dtype=np.float32)
    scale = np.asarray(layer.mlp.experts.wi_0_scale.value)
    decoded = w * scale[:, :, 0, :]
    np.testing.assert_array_equal(decoded[:, ::2, :], np.float32(0.5 * 2**-8))
    np.testing.assert_array_equal(decoded[:, 1::2, :], np.float32(1 * 2**-8))
    assert layer.hc_attn_fn.value.dtype == jnp.float32
    assert layer.self_attn.wq_b.weight_q.value.dtype == jnp.float8_e4m3fn


def test_missing_weights_fail_before_forward(tmp_path):
    model, mesh = make_model()
    from safetensors.numpy import save_file

    save_file(
        {"embed.weight": np.zeros((32, 256), np.float32)}, str(tmp_path / "model.safetensors")
    )
    with jax.set_mesh(mesh), pytest.raises(ValueError, match="missing .* trunk tensors"):
        model.load_weights(SimpleNamespace(model_path=str(tmp_path)))


def harness_for(model, dp, tp):
    from sgl_jax.srt.layers.attention.deepseek_v4_backend import (
        DeepseekV4AttentionBackend,
    )
    from sgl_jax.srt.mem_cache.deepseek_v4.pool import DeepseekV4CacheSpec
    from sgl_jax.test.model_executor.test_deepseek_v4_runtime import Harness

    h = Harness(dp=dp, tp=tp, spec=DeepseekV4CacheSpec.from_config(model.config))
    h.runner.attn_backend = DeepseekV4AttentionBackend(
        mesh=h.mesh, page_size=128, max_context_len=256, config=model.config
    )
    h.runner.bind_attention_resources()
    return h


@pytest.mark.parametrize("dp,tp", [(1, 1), (2, 2)])
def test_complete_trunk_abstract_prefill_decode(tmp_path, dp, tp):
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    model, mesh = make_model(dp, tp)
    write_fixture(tmp_path / "model.safetensors", model)
    with jax.set_mesh(mesh):
        model.load_weights(SimpleNamespace(model_path=str(tmp_path)))
        h = harness_for(model, dp, tp)
        for mode, count in [(ForwardMode.EXTEND, 128), (ForwardMode.DECODE, 1)]:
            b = h.batch([count] * dp, mode, capacity=128)
            fb = h.forward_batch(b)
            lm = LogitsMetadata(
                forward_mode=mode,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                logits_indices=jax.device_put(
                    np.array([127 + 128 * r for r in range(dp)], np.int32),
                    jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data")),
                ),
            )
            definition, state = nnx.split(model)

            def call(s, f, p, lm=lm, definition=definition):
                return nnx.merge(definition, s)(f, p, lm)

            result = jax.eval_shape(call, state, fb, h.runner.memory_pools)
            jaxpr = jax.make_jaxpr(call)(state, fb, h.runner.memory_pools)
            assert sum(np.asarray(value).nbytes for value in jaxpr.consts) < 1024 * 1024
            assert result[0].next_token_logits.shape[-1] == 32
            assert len(result[3]) == 3
            assert set(result[1]) == {"token_to_kv_pool", "compressor_state_pool"}


def compiled_trunk(model):
    # Match ModelRunner: weights are dynamic inputs, never multi-GB JIT constants.
    definition, state = nnx.split(model)
    traces = []

    @jax.jit
    def forward(model_state, batch, pools):
        traces.append(batch.forward_mode)
        return nnx.merge(definition, model_state).model(batch, pools)

    return lambda batch, pools: forward(state, batch, pools), traces


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="actual GMM and FP8 matmul require TPU")
@pytest.mark.parametrize("dp,tp", [(1, 1), (2, 2)])
def test_complete_trunk_chunk_and_decode_equivalence(tmp_path, dp, tp):
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

    model, mesh = make_model(dp, tp)
    write_fixture(tmp_path / "model.safetensors", model)
    with jax.set_mesh(mesh):
        model.load_weights(SimpleNamespace(model_path=str(tmp_path)))
        forward, traces = compiled_trunk(model)

        def run(chunks):
            h = harness_for(model, dp, tp)
            outputs = []
            for count, mode in chunks:
                b = h.batch([count] * dp, mode, capacity=256)
                for rank in range(dp):
                    start = h.lengths[rank] - count
                    b.input_ids[rank * 256 : rank * 256 + count] = (
                        np.arange(start, start + count) % 32
                    )
                fb = h.forward_batch(b)
                out, updates, ids = forward(fb, h.runner.memory_pools)
                out.block_until_ready()
                outputs.append(
                    np.stack(
                        [np.asarray(out)[r * 256 : r * 256 + count] for r in range(dp)], axis=0
                    )
                )
                h.runner.memory_pools.replace_all(updates)
            return np.concatenate(outputs, axis=1), h.runner.memory_pools

        whole, _ = run([(129, ForwardMode.EXTEND)])
        split, pools = run(
            [(63, ForwardMode.EXTEND), (65, ForwardMode.EXTEND), (1, ForwardMode.DECODE)]
        )
        np.testing.assert_allclose(
            split.astype(np.float32), whole.astype(np.float32), rtol=0.04, atol=0.04
        )
        assert len(traces) == 2
        assert np.isfinite(split.astype(np.float32)).all()
        assert np.any(np.asarray(pools.token_to_kv_pool.get_buffer("c4", 1)))
        assert np.any(np.asarray(pools.token_to_kv_pool.get_buffer("c128", 2)))


@pytest.mark.parametrize("layer_id", [0, 1, 2])
@pytest.mark.parametrize("dp,tp", [(1, 1), (2, 2)])
def test_attention_chunk_decode_with_real_pools(tmp_path, layer_id, dp, tp):
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

    model, mesh = make_model(dp, tp)
    write_fixture(tmp_path / "model.safetensors", model)
    if jax.default_backend() == "cpu":
        # The production BF16 load path dequantizes non-expert checkpoint FP8;
        # CPU cannot lower the TPU FP8 matmul. Keep the same on-disk fixture.
        cfg = tiny_config()
        cfg.quantization_config = None
        with jax.set_mesh(mesh):
            model = nnx.eval_shape(lambda: DeepseekV4ForCausalLM(cfg, mesh))
    with jax.set_mesh(mesh):
        model.load_weights(SimpleNamespace(model_path=str(tmp_path)))
        layer = model.model.layers[layer_id].self_attn
        rope = model.model.rope_plain.value if layer_id == 0 else model.model.rope_compressed.value

        def run(chunks):
            h = harness_for(model, dp, tp)
            output = []
            for count, mode in chunks:
                b = h.batch([count] * dp, mode, capacity=256)
                fb = h.forward_batch(b)
                # Same hidden activations for a position regardless of chunk boundaries.
                hidden = np.sin(
                    np.asarray(b.positions)[:, None] * 0.11 + np.arange(512)[None, :] * 0.017
                ).astype(jnp.bfloat16)
                hidden = jax.device_put(
                    hidden,
                    jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None)),
                )
                y, update = jax.jit(lambda x, f, p: layer(x, f, p, rope))(
                    hidden, fb, h.runner.memory_pools
                )
                y.block_until_ready()
                output.append(
                    np.stack([np.asarray(y)[r * 256 : r * 256 + count] for r in range(dp)], axis=0)
                )
                packed = h.runner.attn_backend.pack_pool_updates(
                    {layer_id: update},
                    h.runner.memory_pools.token_to_kv_pool,
                    h.runner.memory_pools.compressor_state_pool,
                )
                h.runner.memory_pools.replace_all(packed)
            return np.concatenate(output, axis=1)

        whole = run([(129, ForwardMode.EXTEND)])
        split = run([(63, ForwardMode.EXTEND), (65, ForwardMode.EXTEND), (1, ForwardMode.DECODE)])
        assert np.isfinite(whole.astype(np.float32)).all()
        assert np.max(np.abs(whole.astype(np.float32))) > 0.01
        np.testing.assert_allclose(
            split.astype(np.float32), whole.astype(np.float32), rtol=0.03, atol=0.0003
        )


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="Flash HCA uses Mosaic kernels")
def test_flash_attention_geometry_complete_trunk(tmp_path):
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

    cfg = DeepseekV4Config(
        num_hidden_layers=3,
        compress_ratios=[0, 4, 128],
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hash_layers=1,
        vocab_size=32,
        max_position_embeddings=256,
    )
    cfg.quantization_config = tiny_config().quantization_config
    model, mesh = make_model(1, 8, cfg)
    write_fixture(tmp_path / "model.safetensors", model)
    with jax.set_mesh(mesh):
        model.load_weights(SimpleNamespace(model_path=str(tmp_path)))
        forward, traces = compiled_trunk(model)

        def run(chunks):
            h = harness_for(model, 1, 8)
            assert h.runner.attn_backend.use_pallas_hca
            result = []
            for count, mode in chunks:
                b = h.batch([count], mode, capacity=256)
                start = h.lengths[0] - count
                b.input_ids[:count] = np.arange(start, start + count) % 32
                fb = h.forward_batch(b)
                output, updates, _ = forward(fb, h.runner.memory_pools)
                output.block_until_ready()
                result.append(np.asarray(output)[:count].astype(np.float32))
                h.runner.memory_pools.replace_all(updates)
            return np.concatenate(result)

        whole = run([(129, ForwardMode.EXTEND)])
        split = run([(63, ForwardMode.EXTEND), (65, ForwardMode.EXTEND), (1, ForwardMode.DECODE)])
        assert len(traces) == 2
        assert np.isfinite(whole).all() and np.max(np.abs(whole)) > 0.1
        np.testing.assert_allclose(split, whole, rtol=0.04, atol=0.04)


def test_csa_batched_requests_match_individual_requests(tmp_path):
    model, mesh = make_model()
    write_fixture(tmp_path / "model.safetensors", model)
    if jax.default_backend() == "cpu":
        cfg = tiny_config()
        cfg.quantization_config = None
        with jax.set_mesh(mesh):
            model = nnx.eval_shape(lambda: DeepseekV4ForCausalLM(cfg, mesh))
    with jax.set_mesh(mesh):
        model.load_weights(SimpleNamespace(model_path=str(tmp_path)))
        layer = model.model.layers[1].self_attn
        cache = model.model.rope_compressed.value
        inputs = [
            np.sin(np.arange(8)[:, None] * 0.11 + np.arange(512)[None, :] * 0.017),
            np.cos(np.arange(8)[:, None] * 0.13 + np.arange(512)[None, :] * 0.023),
        ]

        def run(items):
            h = harness_for(model, 1, 1)
            b = h.batch([8], capacity=256)
            if len(items) == 2:
                r = h.runner
                slot = r.req_to_token_pool.alloc([SimpleNamespace(req_pool_idx=None)])[0]
                loc = r.token_to_kv_pool_allocator.alloc_extend(
                    np.array([0]), np.array([8]), np.array([-1]), 8, dp_rank=0
                )
                r.req_to_token_pool.req_to_token[slot, :8] = loc
                b.seq_lens[1] = 8
                b.req_pool_indices[1] = slot
                b.positions[8:16] = np.arange(8)
                b.out_cache_loc[8:16] = loc
                b.extend_seq_lens[1] = 8
                b.extend_prefix_lens[1] = 0
                b.real_bs = 2
                b.real_bs_per_dp = [2]
                b.real_input_ids_len = 16
            fb = h.forward_batch(b)
            hidden = np.zeros((256, 512), jnp.bfloat16)
            hidden[: 8 * len(items)] = np.concatenate(items)
            hidden = jax.device_put(
                hidden, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None))
            )
            y, _ = jax.jit(lambda x, f, p: layer(x, f, p, cache))(hidden, fb, h.runner.memory_pools)
            return np.asarray(y)[: 8 * len(items)].astype(np.float32)

        batched = run(inputs)
        individual = np.concatenate([run([item]) for item in inputs])
        np.testing.assert_allclose(batched, individual, rtol=0.02, atol=0.0003)


def test_flash_grouped_projection_sharding_in_complete_graph():
    cfg = DeepseekV4Config(
        num_hidden_layers=3,
        compress_ratios=[0, 4, 128],
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hash_layers=1,
        vocab_size=32,
        max_position_embeddings=256,
    )
    cfg.quantization_config = tiny_config().quantization_config
    model, mesh = make_model(1, 4, cfg)
    with jax.set_mesh(mesh):
        h = harness_for(model, 1, 4)
        fb = h.forward_batch(h.batch([129], capacity=256))
        result = nnx.eval_shape(lambda m, b, p: m.model(b, p), model, fb, h.runner.memory_pools)
    assert result[0].shape == (256, 4096)


def test_reserved_nonexpert_e8m0_scale_is_rejected(tmp_path):
    model, mesh = make_model()
    path = tmp_path / "model.safetensors"
    write_fixture(path, model)
    raw = bytearray(path.read_bytes())
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + size])
    offset = 8 + size + header["layers.0.attn.wq_a.scale"]["data_offsets"][0]
    raw[offset] = 255
    path.write_bytes(raw)
    with jax.set_mesh(mesh), pytest.raises(ValueError, match="reserved E8M0 scale code"):
        model.load_weights(SimpleNamespace(model_path=str(tmp_path)))


def test_flash_checkpoint_headers_match_model_parameters():
    from pathlib import Path

    metadata = json.loads(
        (
            Path(__file__).parents[1] / "configs/deepseek_v4_flash_0731_tensor_metadata.json"
        ).read_text()
    )
    cfg = DeepseekV4Config()
    cfg.quantization_config = tiny_config().quantization_config
    model, _ = make_model(cfg=cfg)
    mappings = build_weight_mappings(cfg)
    for key, header in metadata["tensors"].items():
        if ".ffn.experts." in key:
            _, layer, _, _, _, stem, kind = key.split(".")
            expert = model.model.layers[int(layer)].mlp.experts
            shape = getattr(expert, {"w1": "wi_0", "w3": "wi_1", "w2": "wo"}[stem]).value.shape
            expected = (shape[2], shape[1] // (2 if kind == "weight" else 32))
        else:
            mapping = mappings[key]
            p = get_param(model, mapping.target_path).value
            if key.endswith(".scale"):
                matrix = get_param(model, mapping.target_path.rsplit(".", 1)[0]).weight_q.value
                expected = (matrix.shape[0] // 128, matrix.shape[1] // 128)
                assert header["dtype"] == "F8_E8M0"
            else:
                expected = p.shape[::-1] if mapping.transpose else p.shape
        assert tuple(header["shape"]) == expected, key


@pytest.mark.parametrize("dp,tp,ep", [(1, 1, 1), (2, 2, 1), (2, 2, 2)])
def test_static_fp8_load_matches_dynamic_without_conversion(tmp_path, dp, tp, ep, monkeypatch):
    from sgl_jax.srt.utils.quantization import mxfp4_fp8_loader
    from sgl_jax.srt.utils.quantization.deepseek_v4_static_fp8 import (
        CONFIG_KEY,
        FORMAT,
        export_checkpoint,
    )

    source, output = tmp_path / "source", tmp_path / "static"
    source.mkdir()
    cfg = tiny_config()
    cfg.ep_size = ep
    original, mesh = make_model(dp, tp, cfg)
    write_fixture(source / "model.safetensors", original)
    header = mxfp4_fp8_loader._read_safetensors_header(source / "model.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors" for key in header}})
    )
    (source / "config.json").write_text(json.dumps({"num_hidden_layers": 3, "n_routed_experts": 4}))
    with jax.set_mesh(mesh):
        original.load_weights(SimpleNamespace(model_path=str(source)))
    export_checkpoint(source, output, source_revision="fixture", converter_revision="test")
    loaded, static_mesh = make_model(dp, tp, cfg)
    setattr(loaded.config, CONFIG_KEY, FORMAT)

    def forbidden_conversion(*args, **kwargs):
        raise AssertionError("Static loader must never invoke MXFP4 conversion")

    monkeypatch.setattr(
        mxfp4_fp8_loader, "convert_mxfp4_pair_from_safetensors", forbidden_conversion
    )
    from sgl_jax.srt.utils.quantization import deepseek_v4_static_fp8

    monkeypatch.setattr(deepseek_v4_static_fp8, "read_static_pair", forbidden_conversion)
    monkeypatch.setattr(loaded, "_load_expert_weights", forbidden_conversion)
    with jax.set_mesh(static_mesh):
        loaded.load_weights(
            SimpleNamespace(
                model_path=str(output),
                hf_config=loaded.config,
                ep_size=ep,
                quantization_config=loaded.config.quantization_config,
            )
        )
    expected = dict(nnx.state(original, nnx.Param).flat_state())
    actual = dict(nnx.state(loaded, nnx.Param).flat_state())
    assert expected.keys() == actual.keys()
    for key in expected:
        a, b = np.asarray(expected[key].value), np.asarray(actual[key].value)
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8), err_msg=str(key))
