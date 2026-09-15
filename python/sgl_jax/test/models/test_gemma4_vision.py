from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import AxisType, Mesh

from sgl_jax.srt.configs.gemma4 import Gemma4Config, Gemma4VisionConfig
from sgl_jax.srt.models.gemma4 import Gemma4ForConditionalGeneration
from sgl_jax.srt.models.gemma4_vision import (
    Gemma4VisionModel,
    apply_multidimensional_rope,
)
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    encoder_num_lanes,
    pack_2d_position_inputs,
    restore_encoder_output,
)


def _mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _vision_config(**overrides) -> Gemma4VisionConfig:
    values = {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "patch_size": 1,
        "pooling_kernel_size": 3,
        "position_embedding_size": 16,
        "default_output_length": 2,
        "standardize": True,
    }
    values.update(overrides)
    return Gemma4VisionConfig(**values)


def _grid(width: int, height: int) -> np.ndarray:
    y, x = np.indices((height, width))
    return np.stack((x, y), axis=-1).reshape(-1, 2).astype(np.int32)


def _item(width: int, height: int) -> MultimodalDataItem:
    positions = _grid(width, height)
    return MultimodalDataItem(
        Modality.IMAGE,
        feature=np.full((len(positions), 3), 0.5, dtype=np.float32),
        model_specific_data={"pixel_position_ids": positions},
    )


def _pack(model, items):
    num_lanes = encoder_num_lanes(model.mesh, model.vision_tp)
    patches, indices, positions, counts = pack_2d_position_inputs(
        items,
        num_lanes=num_lanes,
        buckets=model.input_buckets,
        merge_unit=model.pooling_unit,
        input_sharding=model.specs.sharding(model.specs.batch_axis),
    )
    return patches, positions, counts, indices


def _encode(model, patches, positions, counts):
    metadata = model.prepare_metadata(
        positions,
        counts,
        capacity=positions.shape[1],
        sharding=model.specs.sharding(model.specs.batch_axis),
    )
    with jax.set_mesh(model.mesh):
        return model.encode(patches, **metadata)


def test_gemma4_config_builds_typed_vision_config():
    config = Gemma4Config(
        text_config={"head_dim": 8, "num_key_value_heads": 1},
        vision_config={"hidden_size": 32, "patch_size": 8},
    )

    assert isinstance(config.vision_config, Gemma4VisionConfig)
    assert config.vision_config.hidden_size == 32
    assert config.vision_config.patch_size == 8
    assert config.vision_config.pooling_kernel_size == 3


def test_multidimensional_rope_rotates_each_axis_independently():
    inputs = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)
    positions = jnp.asarray([[[0, 0], [1, 0]]], dtype=jnp.int32)

    output = apply_multidimensional_rope(inputs, positions, base_frequency=100.0)

    np.testing.assert_allclose(output[0, 0], inputs[0, 0])
    assert not np.allclose(output[0, 1, 0, :2], inputs[0, 1, 0, :2])
    np.testing.assert_allclose(output[0, 1, 0, 2:], inputs[0, 1, 0, 2:])


def test_pool_indices_follow_two_dimensional_windows():
    indices = Gemma4VisionModel._pool_indices(_grid(6, 3), kernel_size=3)

    np.testing.assert_array_equal(np.bincount(indices), [9, 9])
    np.testing.assert_array_equal(indices.reshape(3, 6)[:, :3], 0)
    np.testing.assert_array_equal(indices.reshape(3, 6)[:, 3:], 1)


def test_patch_position_embedding_supports_jitted_dynamic_indices():
    mesh = _mesh()
    patches = jnp.full((1, 4, 3), 0.5, dtype=jnp.float32)
    positions = jnp.asarray([[[0, 0], [1, 0], [0, 1], [-1, -1]]], dtype=jnp.int32)
    with jax.set_mesh(mesh):
        model = Gemma4VisionModel(
            _vision_config(),
            text_hidden_size=12,
            dtype=jnp.float32,
            rngs=None,
            mesh=mesh,
            vision_tp=False,
            input_buckets=(4,),
        )
        eager = model.patch_embedder(patches, positions)
        compiled = jax.jit(lambda x, pos: model.patch_embedder(x, pos))(patches, positions)

    np.testing.assert_allclose(compiled, eager, rtol=1e-5, atol=1e-5)


def test_lane_metadata_keeps_packed_images_as_separate_attention_segments():
    mesh = _mesh()
    with jax.set_mesh(mesh):
        model = Gemma4VisionModel(
            _vision_config(),
            text_hidden_size=12,
            dtype=jnp.float32,
            rngs=None,
            mesh=mesh,
            vision_tp=False,
            input_buckets=(18,),
        )

    _, position_ids, patch_counts, output_indices = _pack(model, [_item(3, 3), _item(3, 3)])
    metadata = model._build_metadata(position_ids, patch_counts)

    np.testing.assert_array_equal(np.asarray(metadata.attention.cu_seqlens), [[0, 9, 18]])
    np.testing.assert_array_equal(np.asarray(metadata.pool_indices), [[0] * 9 + [1] * 9])
    np.testing.assert_array_equal(output_indices, [0, 1])


def test_vision_tower_uses_varlen_backend_and_returns_item_ordered_array(monkeypatch):
    backend_options = {}

    class IdentityAttention:
        def __call__(self, query, key, value, metadata, **kwargs):
            del key, value, metadata
            return query

    def fake_backend(mesh, **kwargs):
        del mesh
        backend_options.update(kwargs)
        return IdentityAttention()

    monkeypatch.setattr(
        "sgl_jax.srt.models.gemma4_vision.make_vision_attention_backend",
        fake_backend,
    )
    mesh = _mesh()
    with jax.set_mesh(mesh):
        model = Gemma4VisionModel(
            _vision_config(num_hidden_layers=1),
            text_hidden_size=12,
            dtype=jnp.float32,
            rngs=None,
            mesh=mesh,
            vision_tp=False,
            input_buckets=(9,),
        )

    item = _item(3, 3)
    patches, position_ids, patch_counts, output_indices = _pack(model, [item])
    output = _encode(model, patches, position_ids, patch_counts)
    packed = restore_encoder_output(output, output_indices, model.specs.sharding())

    assert backend_options["use_varlen"] is True
    assert packed.shape == (1, 12)
    assert bool(jnp.all(jnp.isfinite(packed)))


def test_vision_attention_casts_projection_outputs_to_model_dtype(monkeypatch):
    attention_dtypes = {}

    class Float32Projection(nnx.Module):
        def __call__(self, inputs, *, out_sharding=None):
            del out_sharding
            return inputs.astype(jnp.float32), None

    class IdentityAttention:
        def __call__(self, query, key, value, metadata, **kwargs):
            del metadata
            attention_dtypes.update(q=query.dtype, k=key.dtype, v=value.dtype)
            return query

    monkeypatch.setattr(
        "sgl_jax.srt.models.gemma4_vision.make_vision_attention_backend",
        lambda mesh, **kwargs: IdentityAttention(),
    )
    mesh = _mesh()
    with jax.set_mesh(mesh):
        model = Gemma4VisionModel(
            _vision_config(num_hidden_layers=1),
            text_hidden_size=12,
            dtype=jnp.bfloat16,
            rngs=None,
            mesh=mesh,
            vision_tp=False,
            input_buckets=(9,),
        )

    attention = model.layers[0].self_attn
    attention.q_proj = Float32Projection()
    attention.k_proj = Float32Projection()
    attention.v_proj = Float32Projection()

    item = _item(3, 3)
    patches, position_ids, patch_counts, output_indices = _pack(model, [item])
    output = _encode(model, patches, position_ids, patch_counts)
    packed = restore_encoder_output(output, output_indices, model.specs.sharding())

    assert attention_dtypes == {"q": jnp.bfloat16, "k": jnp.bfloat16, "v": jnp.bfloat16}
    assert bool(jnp.all(jnp.isfinite(packed)))


def test_vision_weight_mappings_match_gemma4_checkpoint_layout():
    fake_model = SimpleNamespace(
        visual=SimpleNamespace(
            standardize=True,
            specs=SimpleNamespace(
                col_kernel_axes=(None, "tensor"),
                row_kernel_axes=("tensor", None),
                tensor_axis="tensor",
            ),
        ),
        root_config=SimpleNamespace(vision_config=SimpleNamespace(num_hidden_layers=1)),
    )

    mappings = Gemma4ForConditionalGeneration._create_vision_weight_mappings(fake_model)

    assert (
        mappings["model.vision_tower.patch_embedder.position_embedding_table"].target_path
        == "visual.patch_embedder.position_embedding_table"
    )
    assert (
        mappings["model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"].target_path
        == "visual.layers.0.self_attn.q_proj.weight"
    )
    assert (
        mappings["model.embed_vision.embedding_projection.weight"].target_path
        == "visual.projector.embedding_projection.weight"
    )
    assert "model.vision_tower.std_scale" in mappings


def test_shared_vision_runner_restores_images_and_precompiles():
    from sgl_jax.srt.multimodal.in_model.lane_packing import run_mrope_vision_model

    mesh = _mesh()
    with jax.set_mesh(mesh):
        model = Gemma4VisionModel(
            _vision_config(standardize=False),
            text_hidden_size=12,
            dtype=jnp.float32,
            rngs=None,
            mesh=mesh,
            vision_tp=False,
            input_buckets=(9, 18, 27),
        )
    items = [_item(3, 3), _item(6, 3)]
    items[1].feature[:] = 0.75

    def encode(items):
        return run_mrope_vision_model(
            model,
            items,
            mesh=mesh,
            num_lanes=1,
            buckets=model.input_buckets,
            merge_unit=9,
            rope_type="rope_2d_packed",
            input_sharding=model.specs.sharding(model.specs.batch_axis),
            output_sharding=model.specs.sharding(),
        )

    actual = encode(items)
    expected = jnp.concatenate([encode([item]) for item in items])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    model.precompile()


def test_gemma4_overflow_bucket_preserves_nine_patch_pooling_groups():
    from sgl_jax.srt.multimodal.in_model.lane_packing import run_mrope_vision_model

    mesh = _mesh()
    with jax.set_mesh(mesh):
        model = Gemma4VisionModel(
            _vision_config(standardize=False),
            text_hidden_size=12,
            dtype=jnp.float32,
            rngs=None,
            mesh=mesh,
            vision_tp=False,
            input_buckets=(18,),
        )
    items = [_item(6, 3), _item(6, 3)]
    items[1].feature[:] = 0.75

    def encode(items):
        return run_mrope_vision_model(
            model,
            items,
            mesh=mesh,
            num_lanes=1,
            buckets=model.input_buckets,
            merge_unit=9,
            rope_type="rope_2d_packed",
            input_sharding=model.specs.sharding(model.specs.batch_axis),
            output_sharding=model.specs.sharding(),
        )

    actual = encode(items)
    expected = jnp.concatenate([encode([item]) for item in items])
    np.testing.assert_allclose(actual[:4], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(actual[4:], 0)


def test_default_precompile_covers_two_images_per_lane_without_retracing(monkeypatch):
    from sgl_jax.srt.multimodal.in_model.lane_packing import run_mrope_vision_model

    traces = []
    original = Gemma4VisionModel._forward

    def record_trace(self, patches, metadata):
        traces.append(patches.shape)
        return original(self, patches, metadata)

    monkeypatch.setattr(Gemma4VisionModel, "_forward", record_trace)
    mesh = _mesh()
    with jax.set_mesh(mesh):
        model = Gemma4VisionModel(
            _vision_config(),
            text_hidden_size=12,
            dtype=jnp.float32,
            rngs=None,
            mesh=mesh,
            vision_tp=False,
        )
    model.precompile()
    precompiled_traces = len(traces)
    items = [_item(6, 3), _item(6, 3)]
    output = run_mrope_vision_model(
        model,
        items,
        mesh=mesh,
        num_lanes=1,
        buckets=model.input_buckets,
        merge_unit=9,
        rope_type="rope_2d_packed",
        input_sharding=model.specs.sharding(model.specs.batch_axis),
        output_sharding=model.specs.sharding(),
    )
    jax.block_until_ready(output)
    assert len(traces) == precompiled_traces, "two-image batch missed startup precompile"
    assert output.shape == (4, 12), "two-image batch should not use an oversized fallback"


def test_lane_packing_restores_distinct_images_across_devices():
    import pytest

    from sgl_jax.srt.multimodal.in_model.lane_packing import run_mrope_vision_model

    if len(jax.devices()) < 8:
        pytest.skip(
            "requires eight devices; set XLA_FLAGS=--xla_force_host_platform_device_count=8"
        )
    mesh = Mesh(
        np.asarray(jax.devices()[:8]).reshape(4, 2),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    with jax.set_mesh(mesh):
        model = Gemma4VisionModel(
            _vision_config(num_hidden_layers=1, standardize=False),
            text_hidden_size=12,
            dtype=jnp.float32,
            rngs=None,
            mesh=mesh,
            vision_tp=False,
        )
    items = [_item(3 if i % 3 == 0 else 6, 3) for i in range(13)]
    for i, item in enumerate(items):
        item.feature[:] = 0.1 + i * 0.05

    def encode(batch):
        return run_mrope_vision_model(
            model,
            batch,
            mesh=mesh,
            num_lanes=8,
            buckets=model.input_buckets,
            merge_unit=9,
            rope_type="rope_2d_packed",
            input_sharding=model.specs.sharding(model.specs.batch_axis),
            output_sharding=model.specs.sharding(),
        )

    assert not np.allclose(encode([items[1]])[0], encode([items[2]])[0])
    actual = encode(items)
    expected = jnp.concatenate([encode([item])[: len(item.feature) // 9] for item in items])
    np.testing.assert_allclose(actual[: len(expected)], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(actual[len(expected) :], 0)
