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
            value = np.full((n // 128, k // 128), 0.015625, np.float32)
            kind = "F32"
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
            for w, (n, k) in {"w1": (512, 512), "w3": (512, 512), "w2": (512, 512)}.items():
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


def make_model(dp=1, tp=1):
    if jax.device_count() < dp * tp:
        pytest.skip("requires four devices")
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[: dp * tp]).reshape(dp, tp),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    cfg = tiny_config()
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(lambda: DeepseekV4ForCausalLM(cfg, mesh))
    return model, mesh


@pytest.mark.parametrize("dp,tp", [(1, 1), (2, 2)])
def test_real_mixed_checkpoint_load(tmp_path, dp, tp):
    model, mesh = make_model(dp, tp)
    write_fixture(tmp_path / "model.safetensors", model)
    with jax.set_mesh(mesh):
        model.load_weights(SimpleNamespace(model_path=str(tmp_path)))
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
    from sgl_jax.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttentionBackend
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
    from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode

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
            result = jax.eval_shape(lambda f, p, lm=lm: model(f, p, lm), fb, h.runner.memory_pools)
            assert result[0].next_token_logits.shape[-1] == 32
            assert len(result[3]) == 3
            assert set(result[1]) == {"token_to_kv_pool", "compressor_state_pool"}


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="actual GMM and FP8 matmul require TPU")
@pytest.mark.parametrize("dp,tp", [(1, 1), (2, 2)])
def test_complete_trunk_chunk_and_decode_equivalence(tmp_path, dp, tp):
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

    model, mesh = make_model(dp, tp)
    write_fixture(tmp_path / "model.safetensors", model)
    with jax.set_mesh(mesh):
        model.load_weights(SimpleNamespace(model_path=str(tmp_path)))

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
                out, updates, ids = jax.jit(lambda f, p: model.model(f, p))(
                    fb, h.runner.memory_pools
                )
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
        assert np.isfinite(split.astype(np.float32)).all()
        assert np.any(np.asarray(pools.token_to_kv_pool.get_buffer("c4", 1)))
        assert np.any(np.asarray(pools.token_to_kv_pool.get_buffer("c128", 2)))
