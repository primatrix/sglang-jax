import dataclasses
import inspect
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.managers import mm_utils
from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.managers.mm_utils import build_mm_embed_plan, merge_jit
from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    ScheduleBatch,
    ScheduleReqsInfo,
)
from sgl_jax.srt.model_executor import forward_batch_info
from sgl_jax.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    _device_put_embed_plan,
)
from sgl_jax.srt.models.qwen2_5_vl import (
    Qwen2_5_VisionTransformer,
    Qwen2_5_VLForConditionalGeneration,
)
from sgl_jax.srt.models.vision_metadata.qwen2_5_vl import (
    Qwen25VLVisionEncodeInputs,
    Qwen25VLVisionMetadata,
    Qwen25VLVisionMetadataBuilder,
)
from sgl_jax.srt.multimodal.common import mm_plan
from sgl_jax.srt.multimodal.common.vision_plan_builder import MergeSlice
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
    VisionFlashAttentionBackend,
)
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor
from sgl_jax.srt.server_args import apply_multimodal_model_defaults


def _build_image_items(features, grids, placeholder_ranges):
    return QwenVLProcessor._build_items(
        features,
        grids,
        placeholder_ranges,
        Modality.IMAGE,
        "image_grid_thw",
    )


def test_mm_plan_exposes_modality_batch_contract():
    merge_slice = MergeSlice(
        task_id=3,
        feature_start=4,
        token_start=9,
        length=2,
    )
    assert dataclasses.astuple(merge_slice) == (3, 4, 9, 2)

    merge = mm_plan.DeviceMergePlan(
        src_idx=np.zeros((1, 2, 4), dtype=np.int32),
        dst_idx=np.zeros((1, 2, 4), dtype=np.int32),
        mask=np.zeros((1, 2, 4), dtype=np.bool_),
    )
    batch = mm_plan.ModalityEmbedBatch(
        encode_inputs=Qwen25VLVisionEncodeInputs(
            patches=np.zeros((1, 2, 8, 3), dtype=np.float32),
            valid=np.zeros((1, 2), dtype=np.int32),
            meta=None,
        ),
        merge=merge,
    )
    plan = {Modality.IMAGE: batch}
    assert plan[Modality.IMAGE].merge is merge


def test_in_model_plan_builder_registry_resolves_by_architecture():
    from sgl_jax.srt.multimodal.common.in_model_plan_builder import (
        register_in_model_plan_builder,
        resolve_in_model_plan_builder,
    )

    class _Builder:
        def __init__(self, model_config):
            self.model_config = model_config

        def build(self, reqs_info, dp_size, per_dp_token, tp_size):
            return None

    register_in_model_plan_builder("UnitTestArchitecture", _Builder)
    config = SimpleNamespace(hf_config=SimpleNamespace(architectures=["UnitTestArchitecture"]))
    assert isinstance(resolve_in_model_plan_builder(config), _Builder)


def test_in_model_plan_builder_registry_returns_none_for_unregistered_architecture():
    from sgl_jax.srt.multimodal.common.in_model_plan_builder import (
        resolve_in_model_plan_builder,
    )

    config = SimpleNamespace(hf_config=SimpleNamespace(architectures=["UnregisteredArchitecture"]))
    assert resolve_in_model_plan_builder(config) is None


def test_importing_qwen_model_registers_its_in_model_plan_builder():
    code = """
from types import SimpleNamespace
from sgl_jax.srt.multimodal.common.in_model_plan_builder import resolve_in_model_plan_builder
import sgl_jax.srt.models.qwen2_5_vl

config = SimpleNamespace(
    hf_config=SimpleNamespace(
        architectures=["Qwen2_5_VLForConditionalGeneration"],
        vision_config=SimpleNamespace(),
    )
)
print(type(resolve_in_model_plan_builder(config)).__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "InModelVisionPlanBuilder"


def test_common_in_model_plan_modules_do_not_reference_vision_fields():
    common_sources = [
        inspect.getsource(mm_utils),
        inspect.getsource(mm_plan),
        inspect.getsource(forward_batch_info),
    ]
    forbidden = ("vision", ".patches", "patch_k")
    for source in common_sources:
        source = source.lower()
        assert not any(term in source for term in forbidden)


def test_build_mm_embed_plan_delegates_to_registered_qwen_builder(monkeypatch):
    from sgl_jax.srt.multimodal.common.vision_plan_builder import InModelVisionPlanBuilder

    sentinel = object()
    monkeypatch.setattr(InModelVisionPlanBuilder, "build", lambda self, *args: sentinel)
    monkeypatch.setattr(mm_utils, "_has_in_model_multimodal_inputs", lambda _: True)
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=SimpleNamespace(),
        )
    )
    assert build_mm_embed_plan([], 1, config, 1) is sentinel


def test_mm_plan_balances_atomic_images_across_tp_lanes():
    features = np.arange(20, dtype=np.float32).reshape(20, 1)
    items = _build_image_items(
        features,
        [(1, 1, 8), (1, 1, 6), (1, 1, 4), (1, 1, 2)],
        [(0, 7), (8, 13), (14, 17), (18, 19)],
    )
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=items),
        extend_input_len=20,
    )
    vision_config = SimpleNamespace(
        patch_size=1,
        window_size=1,
        spatial_merge_size=1,
        fullatt_block_indexes=[],
        num_heads=1,
        hidden_size=4,
        rope_theta=10000.0,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=vision_config,
        ),
    )

    plan = build_mm_embed_plan(
        reqs_info=[ScheduleReqsInfo(reqs=[req])],
        dp_size=1,
        tp_size=2,
        model_config=model_config,
        per_dp_token=20,
    )

    batch = plan[Modality.IMAGE]
    np.testing.assert_array_equal(batch.encode_inputs.valid, np.array([[10, 10]]))
    np.testing.assert_array_equal(
        batch.encode_inputs.patches[0, 0, :, 0], features[[*range(8), 18, 19], 0]
    )
    np.testing.assert_array_equal(batch.encode_inputs.patches[0, 1, :, 0], features[8:18, 0])

    merge = batch.merge
    np.testing.assert_array_equal(merge.dst_idx[0, 0], np.array([*range(8), 18, 19]))
    np.testing.assert_array_equal(merge.src_idx[0, 0], np.arange(10))
    np.testing.assert_array_equal(merge.dst_idx[0, 1], np.arange(8, 18))
    np.testing.assert_array_equal(merge.src_idx[0, 1], np.arange(10))
    assert merge.mask.all()


class _NaiveSegmentAttentionBackend:
    def __call__(self, q, k, v, segment_ids):
        q_seg = segment_ids.q
        kv_seg = segment_ids.kv
        scores = jnp.einsum("dnth,dnsh->dnts", q, k)
        mask = (q_seg[:, None, :, None] == kv_seg[:, None, None, :]) & (
            q_seg[:, None, :, None] >= 0
        )
        scores = jnp.where(mask, scores, jnp.asarray(-1e9, dtype=scores.dtype))
        probs = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum("dnts,dnsh->dnth", probs, v)


def test_get_image_feature_uses_model_neutral_batch_shape():
    class _FakeVision:
        def encode(self, patches, meta, valid):
            assert patches.shape == (2, 4, 3)
            assert valid.shape == (2,)
            assert meta.window_index.shape == (2, 4)
            return patches[..., :2]

    model = SimpleNamespace(visual=_FakeVision())
    enc = Qwen25VLVisionEncodeInputs(
        patches=jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3),
        valid=jnp.array([4, 4], dtype=jnp.int32),
        meta=Qwen25VLVisionMetadata(
            window_index=jnp.arange(8, dtype=jnp.int32).reshape(2, 4),
            cu_window_seqlens=jnp.full((2, 1), 4, dtype=jnp.int32),
            rotary_pos_emb=jnp.zeros((2, 4, 2), dtype=jnp.float32),
            cu_image_seqlens=jnp.full((2, 1), 4, dtype=jnp.int32),
        ),
    )

    features = Qwen2_5_VLForConditionalGeneration.get_image_feature(model, enc)

    assert features.shape == (2, 4, 2)
    np.testing.assert_array_equal(features, enc.patches[..., :2])


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _FakeModalityInputs:
    values: object
    lengths: object

    def tree_flatten(self):
        return ((self.values, self.lengths), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def test_embed_mm_inputs_treats_encode_inputs_as_opaque_pytree(monkeypatch):
    enc = _FakeModalityInputs(
        values=jnp.arange(24, dtype=jnp.float32).reshape(1, 2, 4, 3),
        lengths=jnp.array([[4, 4]], dtype=jnp.int32),
    )
    merge = mm_plan.DeviceMergePlan(
        src_idx=jnp.zeros((1, 2, 1), dtype=jnp.int32),
        dst_idx=jnp.zeros((1, 2, 1), dtype=jnp.int32),
        mask=jnp.zeros((1, 2, 1), dtype=jnp.bool_),
    )
    plan = {Modality.AUDIO: mm_plan.ModalityEmbedBatch(encode_inputs=enc, merge=merge)}

    class _Model:
        mesh = object()

        def get_audio_feature(self, model_inputs):
            assert model_inputs.values.shape == (2, 4, 3)
            assert model_inputs.lengths.shape == (2,)
            return model_inputs.values[..., :2]

    monkeypatch.setattr(mm_utils, "merge_jit", lambda *args: args[1])
    running = jnp.zeros((1, 2), dtype=jnp.float32)
    result = mm_utils.embed_mm_inputs(
        plan,
        jnp.array([1], dtype=jnp.int32),
        lambda _: running,
        _Model(),
    )
    np.testing.assert_array_equal(result, running)


def test_embed_mm_inputs_owns_device_batch_flatten_and_restore(monkeypatch):
    enc = Qwen25VLVisionEncodeInputs(
        patches=jnp.arange(24, dtype=jnp.float32).reshape(1, 2, 4, 3),
        valid=jnp.array([[4, 4]], dtype=jnp.int32),
        meta=Qwen25VLVisionMetadata(
            window_index=jnp.arange(8, dtype=jnp.int32).reshape(1, 2, 4),
            cu_window_seqlens=jnp.full((1, 2, 1), 4, dtype=jnp.int32),
            rotary_pos_emb=jnp.zeros((1, 2, 4, 2), dtype=jnp.float32),
            cu_image_seqlens=jnp.full((1, 2, 1), 4, dtype=jnp.int32),
        ),
    )
    merge = mm_plan.DeviceMergePlan(
        src_idx=jnp.zeros((1, 2, 1), dtype=jnp.int32),
        dst_idx=jnp.zeros((1, 2, 1), dtype=jnp.int32),
        mask=jnp.zeros((1, 2, 1), dtype=jnp.bool_),
    )
    plan = {Modality.IMAGE: mm_plan.ModalityEmbedBatch(encode_inputs=enc, merge=merge)}

    class _Model:
        mesh = object()

        def get_image_feature(self, model_enc):
            assert model_enc.patches.shape == (2, 4, 3)
            assert model_enc.valid.shape == (2,)
            assert model_enc.meta.window_index.shape == (2, 4)
            return model_enc.patches[..., :2]

    def _fake_merge(mesh, running, features, src_idx, dst_idx, mask):
        assert features.shape == (1, 2, 4, 2)
        return running

    monkeypatch.setattr(mm_utils, "merge_jit", _fake_merge)
    input_ids = jnp.array([1], dtype=jnp.int32)
    running = jnp.zeros((1, 2), dtype=jnp.float32)
    out = mm_utils.embed_mm_inputs(plan, input_ids, lambda _: running, _Model())
    np.testing.assert_array_equal(out, running)


def _two_data_devices():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two devices for real data-axis sharding")
    return np.array(devices[:2])


def test_vision_transformer_uses_default_norm_eps_when_hf_vision_config_omits_it():
    vision_config = SimpleNamespace(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=3,
        hidden_size=4,
        depth=1,
        intermediate_size=8,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=1,
        fullatt_block_indexes=[],
    )

    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    with jax.set_mesh(mesh):
        Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=None,
            norm_eps=1e-6,
        )


def test_vision_transformer_encode_jit_accepts_unhashable_vision_config():
    class UnhashableVisionConfig(SimpleNamespace):
        __hash__ = None

    vision_config = UnhashableVisionConfig(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        hidden_size=4,
        depth=0,
        intermediate_size=8,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=1,
        fullatt_block_indexes=[],
    )
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    meta = Qwen25VLVisionMetadata(
        window_index=jnp.zeros((1, 2), dtype=jnp.int32),
        cu_window_seqlens=jnp.array([[2]], dtype=jnp.int32),
        rotary_pos_emb=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        cu_image_seqlens=jnp.array([[2]], dtype=jnp.int32),
    )

    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=mesh,
            norm_eps=1e-6,
        )
        features = visual.encode_jit(
            jnp.ones((1, 2, 1), dtype=jnp.float32),
            meta,
            jnp.array([2], dtype=jnp.int32),
        )

    assert features.shape == (1, 2, 4)


def test_vision_transformer_shards_flat_lane_batch_over_data_and_tensor():
    vision_config = SimpleNamespace(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        hidden_size=4,
        depth=0,
        intermediate_size=8,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=1,
        fullatt_block_indexes=[],
    )
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    meta = Qwen25VLVisionMetadata(
        window_index=jnp.zeros((1, 2), dtype=jnp.int32),
        cu_window_seqlens=jnp.array([[2]], dtype=jnp.int32),
        rotary_pos_emb=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        cu_image_seqlens=jnp.array([[2]], dtype=jnp.int32),
    )

    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=mesh,
            norm_eps=1e-6,
        )
        features = visual.encode_jit(
            jnp.ones((1, 2, 1), dtype=jnp.float32),
            meta,
            jnp.array([2], dtype=jnp.int32),
        )

    assert features.sharding.spec == PartitionSpec(("data", "tensor"), None, None)


def test_vision_flash_attention_shards_flat_lane_batch_over_data_and_tensor():
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))
    captured = {}

    def fake_shard_map(fn, *, mesh, in_specs, out_specs, check_vma):
        captured["in_specs"] = in_specs
        captured["out_specs"] = out_specs
        return fn

    with patch(
        "sgl_jax.srt.multimodal.layers.attention.flash_attention_backend.jax.shard_map",
        side_effect=fake_shard_map,
    ):
        VisionFlashAttentionBackend(mesh)

    lane_qkv = PartitionSpec(("data", "tensor"), None, None, None)
    assert captured["in_specs"] == (
        lane_qkv,
        lane_qkv,
        lane_qkv,
        PartitionSpec(("data", "tensor"), None),
    )
    assert captured["out_specs"] == lane_qkv


def test_vision_transformer_encode_jit_uses_reshard_for_explicit_mesh(monkeypatch):
    vision_config = SimpleNamespace(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        hidden_size=4,
        depth=0,
        intermediate_size=8,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=1,
        fullatt_block_indexes=[],
    )
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    meta = Qwen25VLVisionMetadata(
        window_index=jnp.zeros((1, 2), dtype=jnp.int32),
        cu_window_seqlens=jnp.array([[2]], dtype=jnp.int32),
        rotary_pos_emb=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        cu_image_seqlens=jnp.array([[2]], dtype=jnp.int32),
    )

    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=mesh,
            norm_eps=1e-6,
        )

        def fail_with_sharding_constraint(*args, **kwargs):
            raise AssertionError("with_sharding_constraint must not be used in Qwen vision encode")

        reshard_specs = []
        original_reshard = jax.sharding.reshard

        def record_reshard(x, out_sharding):
            reshard_specs.append(tuple(out_sharding.spec))
            return original_reshard(x, out_sharding)

        monkeypatch.setattr(jax.lax, "with_sharding_constraint", fail_with_sharding_constraint)
        monkeypatch.setattr(jax.sharding, "reshard", record_reshard)

        features = visual.encode_jit(
            jnp.ones((1, 2, 1), dtype=jnp.float32),
            meta,
            jnp.array([2], dtype=jnp.int32),
        )

    assert features.shape == (1, 2, 4)
    assert reshard_specs == [
        ("data", None, None, None, None, None),
        ("data", None, None),
        ("data", None, None),
    ]


def test_vision_transformer_encode_binds_mesh_for_sharded_inputs_without_callsite_context(
    monkeypatch,
):
    vision_config = SimpleNamespace(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        hidden_size=4,
        depth=2,
        intermediate_size=16,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=2,
        fullatt_block_indexes=[1],
    )
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=mesh,
            norm_eps=1e-6,
        )
        for block in visual.blocks:
            block.attn.attn_backend = None

    def fake_vision_attention(backend, q, k, v, seg):
        return jnp.zeros_like(q)

    monkeypatch.setattr(
        "sgl_jax.srt.models.qwen2_5_vl._vision_attention",
        fake_vision_attention,
    )

    plan = {
        Modality.IMAGE: mm_plan.ModalityEmbedBatch(
            encode_inputs=Qwen25VLVisionEncodeInputs(
                patches=np.ones((1, 1, 4, 1), dtype=np.float32),
                valid=np.array([[4]], dtype=np.int32),
                meta=Qwen25VLVisionMetadata(
                    window_index=np.array([[[0]]], dtype=np.int32),
                    cu_window_seqlens=np.array([[[4]]], dtype=np.int32),
                    rotary_pos_emb=np.zeros((1, 1, 4, 2), dtype=np.float32),
                    cu_image_seqlens=np.array([[[4]]], dtype=np.int32),
                ),
            ),
            merge=mm_plan.DeviceMergePlan(
                src_idx=np.zeros((1, 1, 1), dtype=np.int32),
                dst_idx=np.zeros((1, 1, 1), dtype=np.int32),
                mask=np.zeros((1, 1, 1), dtype=np.bool_),
            ),
        )
    }
    _device_put_embed_plan(plan, mesh)
    enc = plan[Modality.IMAGE].encode_inputs
    model_enc = Qwen25VLVisionEncodeInputs(
        patches=enc.patches.reshape(1, 4, 1),
        valid=enc.valid.reshape(1),
        meta=jax.tree.map(lambda value: value.reshape(1, *value.shape[2:]), enc.meta),
    )

    features = Qwen2_5_VLForConditionalGeneration.get_image_feature(
        SimpleNamespace(visual=visual, mesh=mesh),
        model_enc,
    )

    assert features.shape == (1, 1, 4)


def test_vision_patch_embed_calls_conv_with_single_batch_dim(monkeypatch):
    vision_config = SimpleNamespace(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        hidden_size=4,
        depth=0,
        intermediate_size=8,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=1,
        fullatt_block_indexes=[],
    )
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    meta = Qwen25VLVisionMetadata(
        window_index=jnp.tile(jnp.arange(3, dtype=jnp.int32)[None, :], (2, 1)),
        cu_window_seqlens=jnp.array([[3], [3]], dtype=jnp.int32),
        rotary_pos_emb=jnp.zeros((2, 3, 2), dtype=jnp.float32),
        cu_image_seqlens=jnp.array([[3], [3]], dtype=jnp.int32),
    )

    seen_input_shapes = []
    original_call = nnx.Conv.__call__

    def record_conv_input_shape(self, inputs, *args, **kwargs):
        seen_input_shapes.append(inputs.shape)
        return original_call(self, inputs, *args, **kwargs)

    monkeypatch.setattr(nnx.Conv, "__call__", record_conv_input_shape)

    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=mesh,
            norm_eps=1e-6,
        )
        features = visual.encode_jit(
            jnp.ones((2, 3, 1), dtype=jnp.float32),
            meta,
            jnp.array([3, 3], dtype=jnp.int32),
        )

    assert features.shape == (2, 3, 4)
    assert seen_input_shapes == [(6, 1, 1, 1, 1)]


def test_vision_full_attention_keeps_packed_images_block_diagonal_on_cpu():
    vision_config = SimpleNamespace(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        hidden_size=4,
        depth=1,
        intermediate_size=8,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=1,
        fullatt_block_indexes=[0],
        window_size=1,
        rope_theta=10000.0,
    )
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=None,
            norm_eps=1e-6,
        )
    visual.blocks[0].attn.attn_backend = _NaiveSegmentAttentionBackend()
    builder = Qwen25VLVisionMetadataBuilder(
        SimpleNamespace(hf_config=SimpleNamespace(vision_config=vision_config))
    )

    packed_features = np.arange(1, 8, dtype=np.float32).reshape(7, 1)
    packed_items = _build_image_items(
        packed_features,
        [(1, 2, 2), (1, 1, 3)],
        [(0, 3), (4, 6)],
    )
    packed_meta = builder.stack_metadata(
        [builder.get_metadata(packed_items)],
        patch_k=7,
    )

    single_item = _build_image_items(
        packed_features[:4],
        [(1, 2, 2)],
        [(0, 3)],
    )[0]
    single_meta = builder.stack_metadata(
        [builder.get_metadata([single_item])],
        patch_k=4,
    )

    packed_out = visual.compute_hidden_states(
        jnp.asarray(packed_features[None, :, :]),
        jnp.asarray(packed_meta.window_index),
        jnp.asarray(packed_meta.cu_window_seqlens),
        jnp.asarray(packed_meta.rotary_pos_emb),
        jnp.asarray(packed_meta.cu_image_seqlens),
        jnp.array([7], dtype=jnp.int32),
    )
    single_out = visual.compute_hidden_states(
        jnp.asarray(packed_features[None, :4, :]),
        jnp.asarray(single_meta.window_index),
        jnp.asarray(single_meta.cu_window_seqlens),
        jnp.asarray(single_meta.rotary_pos_emb),
        jnp.asarray(single_meta.cu_image_seqlens),
        jnp.array([4], dtype=jnp.int32),
    )

    np.testing.assert_allclose(
        np.asarray(packed_out[:, :4, :]),
        np.asarray(single_out),
        rtol=1e-5,
        atol=1e-5,
    )


def test_vision_single_image_request_matches_single_image_encode_on_cpu():
    vision_config = SimpleNamespace(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        hidden_size=4,
        depth=1,
        intermediate_size=8,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=1,
        fullatt_block_indexes=[0],
        window_size=1,
        rope_theta=10000.0,
    )
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=None,
            norm_eps=1e-6,
        )
    visual.blocks[0].attn.attn_backend = _NaiveSegmentAttentionBackend()
    builder = Qwen25VLVisionMetadataBuilder(
        SimpleNamespace(hf_config=SimpleNamespace(vision_config=vision_config))
    )

    patch_features = np.arange(1, 5, dtype=np.float32).reshape(4, 1)
    item = _build_image_items(
        patch_features,
        [(1, 2, 2)],
        [(0, 3)],
    )[0]
    native_meta = builder.stack_metadata([builder._get_image_metadata(item)], patch_k=4)
    packed_meta = builder.stack_metadata(
        [builder._pack_lane_metadata([builder._get_image_metadata(item)])],
        patch_k=4,
    )

    native_out = visual.compute_hidden_states(
        jnp.asarray(patch_features[None, :, :]),
        jnp.asarray(native_meta.window_index),
        jnp.asarray(native_meta.cu_window_seqlens),
        jnp.asarray(native_meta.rotary_pos_emb),
        jnp.asarray(native_meta.cu_image_seqlens),
        jnp.array([4], dtype=jnp.int32),
    )
    packed_out = visual.compute_hidden_states(
        jnp.asarray(patch_features[None, :, :]),
        jnp.asarray(packed_meta.window_index),
        jnp.asarray(packed_meta.cu_window_seqlens),
        jnp.asarray(packed_meta.rotary_pos_emb),
        jnp.asarray(packed_meta.cu_image_seqlens),
        jnp.array([4], dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(packed_out), np.asarray(native_out), rtol=1e-5, atol=1e-5)


def test_vision_encode_runs_on_real_dp2_data_mesh(monkeypatch):
    mesh = Mesh(
        _two_data_devices().reshape(2, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    vision_config = SimpleNamespace(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        hidden_size=4,
        depth=2,
        intermediate_size=16,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=2,
        fullatt_block_indexes=[1],
    )

    def fake_vision_attention(backend, q, k, v, seg):
        return jnp.zeros_like(q)

    monkeypatch.setattr(
        "sgl_jax.srt.models.qwen2_5_vl._vision_attention",
        fake_vision_attention,
    )

    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=mesh,
            norm_eps=1e-6,
        )
        for block in visual.blocks:
            block.attn.attn_backend = None

    plan = {
        Modality.IMAGE: mm_plan.ModalityEmbedBatch(
            encode_inputs=Qwen25VLVisionEncodeInputs(
                patches=np.ones((2, 1, 8, 1), dtype=np.float32),
                valid=np.array([[8], [4]], dtype=np.int32),
                meta=Qwen25VLVisionMetadata(
                    window_index=np.array([[[1, 0]], [[0, 1]]], dtype=np.int32),
                    cu_window_seqlens=np.array([[[8]], [[4]]], dtype=np.int32),
                    rotary_pos_emb=np.zeros((2, 1, 8, 2), dtype=np.float32),
                    cu_image_seqlens=np.array([[[8]], [[4]]], dtype=np.int32),
                ),
            ),
            merge=mm_plan.DeviceMergePlan(
                src_idx=np.zeros((2, 1, 1), dtype=np.int32),
                dst_idx=np.zeros((2, 1, 1), dtype=np.int32),
                mask=np.zeros((2, 1, 1), dtype=np.bool_),
            ),
        )
    }
    _device_put_embed_plan(plan, mesh)
    enc = plan[Modality.IMAGE].encode_inputs

    features = Qwen2_5_VLForConditionalGeneration.get_image_feature(
        SimpleNamespace(visual=visual, mesh=mesh),
        enc,
    )

    assert features.shape == (2, 1, 2, 4)
    assert features.sharding.spec == PartitionSpec("data", "tensor", None, None)


def test_merge_jit_scatters_lane_features_and_preserves_text_rows():
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))
    running = jnp.array(
        [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]],
        dtype=jnp.float32,
    )
    features = jnp.array(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]],
        dtype=jnp.float32,
    )
    src_idx = jnp.array([[[0, 1, 0]]], dtype=jnp.int32)
    dst_idx = jnp.array([[[2, 0, 0]]], dtype=jnp.int32)
    mask = jnp.array([[[True, True, False]]])

    out = merge_jit(mesh, running, features, src_idx, dst_idx, mask)

    expected = np.array(
        [[4.0, 5.0, 6.0], [20.0, 21.0, 22.0], [1.0, 2.0, 3.0]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_merge_jit_uses_rank_local_features_on_real_dp2_mesh():
    mesh = Mesh(
        _two_data_devices().reshape(2, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    running = np.arange(12, dtype=np.float32).reshape(6, 2)
    features = np.array(
        [
            [[[10.0, 11.0], [20.0, 21.0]]],
            [[[100.0, 101.0], [200.0, 201.0]]],
        ],
        dtype=np.float32,
    )
    src_idx = np.array([[[1, 0]], [[0, 1]]], dtype=np.int32)
    dst_idx = np.array([[[0, 1]], [[0, 2]]], dtype=np.int32)
    mask = np.ones((2, 1, 2), dtype=np.bool_)

    running_d = jax.device_put(running, NamedSharding(mesh, PartitionSpec("data", None)))
    features_d = jax.device_put(
        features,
        NamedSharding(mesh, PartitionSpec("data", "tensor", None, None)),
    )
    lane_spec = NamedSharding(mesh, PartitionSpec("data", "tensor", None))
    src_idx_d = jax.device_put(src_idx, lane_spec)
    dst_idx_d = jax.device_put(dst_idx, lane_spec)
    mask_d = jax.device_put(mask, lane_spec)

    out = merge_jit(mesh, running_d, features_d, src_idx_d, dst_idx_d, mask_d)

    expected = running.copy()
    expected[0] = features[0, 0, 1]
    expected[1] = features[0, 0, 0]
    expected[3] = features[1, 0, 0]
    expected[5] = features[1, 0, 1]
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_merge_jit_reduces_features_from_real_tp2_lanes():
    mesh = Mesh(
        _two_data_devices().reshape(1, 2),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    running = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    features = np.array([[[[10.0, 11.0]], [[20.0, 21.0]]]], dtype=np.float32)
    src_idx = np.zeros((1, 2, 1), dtype=np.int32)
    dst_idx = np.array([[[0], [2]]], dtype=np.int32)
    mask = np.ones((1, 2, 1), dtype=np.bool_)

    running_d = jax.device_put(running, NamedSharding(mesh, PartitionSpec("data", None)))
    features_d = jax.device_put(
        features,
        NamedSharding(mesh, PartitionSpec("data", "tensor", None, None)),
    )
    lane_sharding = NamedSharding(mesh, PartitionSpec("data", "tensor", None))
    out = merge_jit(
        mesh,
        running_d,
        features_d,
        jax.device_put(src_idx, lane_sharding),
        jax.device_put(dst_idx, lane_sharding),
        jax.device_put(mask, lane_sharding),
    )

    expected = np.array([[10.0, 11.0], [3.0, 4.0], [20.0, 21.0]], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(out), expected)
    assert out.sharding.spec == PartitionSpec("data", None)


def test_device_put_embed_plan_places_qwen_metadata_on_dp_tp_lanes():
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    plan = {
        Modality.IMAGE: mm_plan.ModalityEmbedBatch(
            encode_inputs=Qwen25VLVisionEncodeInputs(
                patches=np.ones((1, 1, 4, 1), dtype=np.float32),
                valid=np.array([[4]], dtype=np.int32),
                meta=Qwen25VLVisionMetadata(
                    window_index=np.array([[[0]]], dtype=np.int32),
                    cu_window_seqlens=np.array([[[4]]], dtype=np.int32),
                    rotary_pos_emb=np.zeros((1, 1, 4, 2), dtype=np.float32),
                    cu_image_seqlens=np.array([[[4]]], dtype=np.int32),
                ),
            ),
            merge=mm_plan.DeviceMergePlan(
                src_idx=np.zeros((1, 1, 2), dtype=np.int32),
                dst_idx=np.zeros((1, 1, 2), dtype=np.int32),
                mask=np.zeros((1, 1, 2), dtype=np.bool_),
            ),
        )
    }

    _device_put_embed_plan(plan, mesh)
    batch = plan[Modality.IMAGE]
    enc = batch.encode_inputs

    assert enc.patches.sharding.spec == PartitionSpec("data", "tensor", None, None)
    assert enc.valid.sharding.spec == PartitionSpec("data", "tensor")
    assert enc.meta.window_index.sharding.spec == PartitionSpec("data", "tensor", None)
    assert enc.meta.cu_window_seqlens.sharding.spec == PartitionSpec("data", "tensor", None)
    assert enc.meta.rotary_pos_emb.sharding.spec == PartitionSpec("data", "tensor", None, None)
    assert enc.meta.cu_image_seqlens.sharding.spec == PartitionSpec("data", "tensor", None)
    assert batch.merge.src_idx.sharding.spec == PartitionSpec("data", "tensor", None)
    assert batch.merge.dst_idx.sharding.spec == PartitionSpec("data", "tensor", None)
    assert batch.merge.mask.sharding.spec == PartitionSpec("data", "tensor", None)


def test_qwen_multimodal_model_defaults_enable_chunked_prefill():
    server_args = SimpleNamespace(
        disable_radix_cache=False,
        disable_overlap_schedule=False,
        chunked_prefill_size=4096,
        enable_mixed_chunk=True,
        limit_mm_data_per_request=None,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(architectures=["Qwen2_5_VLForConditionalGeneration"]),
    )

    apply_multimodal_model_defaults(server_args, model_config)

    assert server_args.disable_radix_cache is True
    assert server_args.disable_overlap_schedule is True
    assert server_args.chunked_prefill_size == 4096
    assert server_args.enable_mixed_chunk is False
    assert server_args.limit_mm_data_per_request == {"image": 16}


def test_unsupported_multimodal_model_defaults_disable_chunked_prefill():
    server_args = SimpleNamespace(
        disable_radix_cache=False,
        disable_overlap_schedule=False,
        chunked_prefill_size=4096,
        enable_mixed_chunk=True,
        limit_mm_data_per_request=None,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(architectures=["UnsupportedMultimodalModel"]),
    )

    apply_multimodal_model_defaults(server_args, model_config)

    assert server_args.chunked_prefill_size == -1
    assert server_args.enable_mixed_chunk is False


def test_generate_req_getitem_preserves_media_fields():
    req = GenerateReqInput(
        text=["a", "b"],
        sampling_params=[{}, {}],
        rid=["r0", "r1"],
        return_logprob=[False, False],
        logprob_start_len=[-1, -1],
        top_logprobs_num=[0, 0],
        token_ids_logprob=[None, None],
        return_routed_experts=[False, False],
        image_data=[["image0"], ["image1"]],
        video_data=[["video0"], ["video1"]],
        audio_data=[["audio0"], ["audio1"]],
    )
    req.input_embeds = [["emb0"], ["emb1"]]

    item = req[1]

    assert item.image_data == ["image1"]
    assert item.video_data == ["video1"]
    assert item.audio_data == ["audio1"]
    assert item.input_embeds == ["emb1"]


def test_forward_batch_input_embedding_uses_data_axis_sharding():
    devices = np.array(jax.devices()[:1])
    mesh = Mesh(devices, ("data",))
    batch = ModelWorkerBatch(
        bid=1,
        forward_mode=ForwardMode.EXTEND,
        input_ids=np.array([1], dtype=np.int32),
        real_input_ids_len=1,
        seq_lens=np.array([1], dtype=np.int32),
        out_cache_loc=np.array([1], dtype=np.int32),
        req_pool_indices=np.array([0], dtype=np.int32),
        sampling_info=None,
        positions=np.array([0], dtype=np.int32),
        cache_loc=np.array([1], dtype=np.int32),
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_seq_lens=np.array([1], dtype=np.int32),
        extend_prefix_lens=np.array([0], dtype=np.int32),
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        logits_indices=np.array([0], dtype=np.int32),
        real_bs=1,
        real_bs_per_dp=[1],
        input_embedding=np.ones((1, 4), dtype=np.float32),
    )
    runner = SimpleNamespace(
        mesh=mesh,
        attn_backend=None,
        model_config=SimpleNamespace(
            is_embedding=False,
            hf_config=SimpleNamespace(architectures=[]),
        ),
    )
    captured_specs = []

    def fake_device_array(values, sharding):
        captured_specs.append(sharding.spec)
        return values

    with patch(
        "sgl_jax.srt.model_executor.forward_batch_info.device_array",
        side_effect=fake_device_array,
    ):
        ForwardBatch.init_new(batch, runner)

    assert PartitionSpec("data", None) in captured_specs


def test_mm_embed_plan_device_put_shards_dp_and_tp_lane_axes():
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, ("data", "tensor"))
    merge = mm_plan.DeviceMergePlan(
        src_idx=np.zeros((1, 1, 4), dtype=np.int32),
        dst_idx=np.zeros((1, 1, 4), dtype=np.int32),
        mask=np.zeros((1, 1, 4), dtype=np.bool_),
    )
    plan = {
        Modality.IMAGE: mm_plan.ModalityEmbedBatch(
            encode_inputs=Qwen25VLVisionEncodeInputs(
                patches=np.ones((1, 1, 4, 3), dtype=np.float32),
                valid=np.array([[4]], dtype=np.int32),
                meta=Qwen25VLVisionMetadata(
                    window_index=np.zeros((1, 1, 1), dtype=np.int32),
                    cu_window_seqlens=np.ones((1, 1, 1), dtype=np.int32),
                    rotary_pos_emb=np.ones((1, 1, 4, 2), dtype=np.float32),
                    cu_image_seqlens=np.array([[[4]]], dtype=np.int32),
                ),
            ),
            merge=merge,
        )
    }
    captured_specs = []

    def fake_device_array(values, sharding):
        captured_specs.append(sharding.spec)
        return values

    with patch(
        "sgl_jax.srt.model_executor.forward_batch_info.device_array",
        side_effect=fake_device_array,
    ):
        _device_put_embed_plan(plan, mesh)

    assert captured_specs == [
        PartitionSpec("data", "tensor", None, None),
        PartitionSpec("data", "tensor"),
        PartitionSpec("data", "tensor", None),
        PartitionSpec("data", "tensor", None),
        PartitionSpec("data", "tensor", None, None),
        PartitionSpec("data", "tensor", None),
        PartitionSpec("data", "tensor", None),
        PartitionSpec("data", "tensor", None),
        PartitionSpec("data", "tensor", None),
    ]


def test_mrope_positions_propagate_through_model_worker_batch():
    mrope_positions = np.array(
        [
            [0, 10, 2],
            [0, 11, 2],
            [0, 12, 2],
        ],
        dtype=np.int32,
    )
    batch = ScheduleBatch(
        reqs_info=[
            ScheduleReqsInfo(
                reqs=[
                    SimpleNamespace(
                        mm_inputs={
                            "mrope_positions": mrope_positions,
                        },
                        lora_id="0",
                    )
                ],
                input_ids=np.array([1, 151655, 2], dtype=np.int32),
                seq_lens=np.array([3], dtype=np.int32),
                out_cache_loc=np.array([1, 2, 3], dtype=np.int32),
                req_pool_indices=np.array([0], dtype=np.int32),
                prefix_lens=np.array([0], dtype=np.int32),
                extend_lens=np.array([3], dtype=np.int32),
                extend_logprob_start_lens=np.array([0], dtype=np.int32),
            )
        ],
        dp_size=1,
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
    )
    batch._merge_sampling_info = lambda per_dp_bs_size, total_bs: None
    batch._merge_cache_loc = lambda *args: np.array([1, 2, 3], dtype=np.int32)

    mwb = batch.get_model_worker_batch(
        token_paddings=[3],
        bs_paddings=[1],
        cache_loc_paddings=[3],
        page_size=1,
    )

    np.testing.assert_array_equal(mwb.mrope_positions[:, :3], mrope_positions)


def test_multimodal_data_item_get_reads_common_and_model_specific_fields():
    item = MultimodalDataItem.from_dict(
        {
            "modality": "image",
            "feature": np.ones((2, 1), dtype=np.float32),
            "placeholder_ranges": [(1, 2)],
            "image_grid_thw": np.array([[1, 2, 4]], dtype=np.int32),
        }
    )

    assert item.is_image()
    np.testing.assert_array_equal(item.get("feature"), np.ones((2, 1), dtype=np.float32))
    assert item.get("placeholder_ranges") == [(1, 2)]
    np.testing.assert_array_equal(
        item.get("image_grid_thw"),
        np.array([[1, 2, 4]], dtype=np.int32),
    )
    assert item.get("missing", "fallback") == "fallback"


def test_qwen_metadata_builder_packs_request_metadata_with_image_boundaries():
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    builder = Qwen25VLVisionMetadataBuilder(
        SimpleNamespace(hf_config=SimpleNamespace(vision_config=vision_config))
    )
    features = np.arange(272, dtype=np.float32).reshape(272, 1)
    items = _build_image_items(
        features,
        [(1, 16, 16), (1, 4, 4)],
        [(0, 63), (64, 67)],
    )

    packed = builder.get_metadata(items)

    np.testing.assert_array_equal(
        packed.cu_window_seqlens,
        np.array([64, 128, 192, 256, 272], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        packed.cu_image_seqlens,
        np.array([256, 272], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.sort(packed.window_index),
        np.arange(68, dtype=np.int32),
    )
    assert packed.rotary_pos_emb.shape[0] == 272


def test_qwen_metadata_builder_single_image_request_metadata_degenerates_to_native():
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    builder = Qwen25VLVisionMetadataBuilder(
        SimpleNamespace(hf_config=SimpleNamespace(vision_config=vision_config))
    )
    item = _build_image_items(
        np.arange(8, dtype=np.float32).reshape(8, 1),
        [(1, 2, 4)],
        [(0, 1)],
    )[0]

    native = builder._get_image_metadata(item)
    packed = builder._pack_lane_metadata([native])

    np.testing.assert_array_equal(packed.window_index, native.window_index)
    np.testing.assert_array_equal(packed.cu_window_seqlens, native.cu_window_seqlens)
    np.testing.assert_array_equal(packed.rotary_pos_emb, native.rotary_pos_emb)
    np.testing.assert_array_equal(packed.cu_image_seqlens, np.array([8], dtype=np.int32))


def test_qwen_metadata_builder_stack_metadata_pads_multi_image_and_dummy_rank():
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    builder = Qwen25VLVisionMetadataBuilder(
        SimpleNamespace(hf_config=SimpleNamespace(vision_config=vision_config))
    )
    items = _build_image_items(
        np.arange(24, dtype=np.float32).reshape(24, 1),
        [(1, 2, 4), (1, 4, 4)],
        [(0, 1), (2, 5)],
    )
    meta = builder.get_metadata(items)

    stacked = builder.stack_metadata([meta, None], patch_k=24)

    assert stacked.window_index.shape == (2, 6)
    assert stacked.cu_window_seqlens.shape == (2, 2)
    assert stacked.rotary_pos_emb.shape == (2, 24, 40)
    assert stacked.cu_image_seqlens.shape == (2, 2)
    np.testing.assert_array_equal(stacked.cu_image_seqlens[0], np.array([8, 24], dtype=np.int32))
    np.testing.assert_array_equal(stacked.cu_image_seqlens[1], np.array([24, 24], dtype=np.int32))
    np.testing.assert_array_equal(stacked.window_index[1], np.arange(6, dtype=np.int32))


def test_qwen_metadata_builder_stack_metadata_fails_fast_on_all_dummy_lanes():
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    builder = Qwen25VLVisionMetadataBuilder(
        SimpleNamespace(hf_config=SimpleNamespace(vision_config=vision_config))
    )

    with pytest.raises(ValueError, match="at least one real"):
        builder.stack_metadata([None, None], patch_k=0)


def test_qwen_metadata_builder_stack_metadata_checks_patch_bucket_divisibility():
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    builder = Qwen25VLVisionMetadataBuilder(
        SimpleNamespace(hf_config=SimpleNamespace(vision_config=vision_config))
    )
    item = _build_image_items(
        np.arange(8, dtype=np.float32).reshape(8, 1),
        [(1, 2, 4)],
        [(0, 1)],
    )[0]
    meta = builder._get_image_metadata(item)

    with pytest.raises(ValueError, match="divisible"):
        builder.stack_metadata([meta], patch_k=10)


def test_mm_embed_plan_keeps_placeholder_count_separate_from_encode_rows():
    features = np.arange(24, dtype=np.float32).reshape(24, 1)
    grids = [(1, 2, 4), (1, 4, 4)]
    placeholder_ranges = [(2, 3), (5, 8)]
    items = _build_image_items(features, grids, placeholder_ranges)
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=items),
        extend_input_len=10,
    )
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        fullatt_block_indexes=[],
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=vision_config,
        ),
    )

    plan = build_mm_embed_plan(
        reqs_info=[ScheduleReqsInfo(reqs=[req])],
        dp_size=1,
        model_config=model_config,
        per_dp_token=10,
    )

    batch = plan[items[0].modality]

    np.testing.assert_array_equal(batch.encode_inputs.valid, np.array([[24]], dtype=np.int32))
    np.testing.assert_array_equal(batch.merge.dst_idx[0, 0], np.array([2, 3, 5, 6, 7, 8]))
    np.testing.assert_array_equal(batch.merge.src_idx[0, 0], np.arange(6))
    assert batch.merge.mask.all()


@pytest.mark.parametrize(
    ("prefix_len", "expected_src", "expected_dst"),
    [
        (0, np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)),
        (4, np.array([2, 3], dtype=np.int32), np.array([0, 1], dtype=np.int32)),
    ],
)
def test_mm_embed_plan_clips_visual_merge_to_chunk_window(prefix_len, expected_src, expected_dst):
    item = _build_image_items(
        np.arange(16, dtype=np.float32).reshape(16, 1),
        [(1, 4, 4)],
        [(2, 5)],
    )[0]
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=[item]),
        extend_input_len=4,
    )
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        fullatt_block_indexes=[],
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=vision_config,
        ),
    )
    info = ScheduleReqsInfo(
        reqs=[req],
        prefix_lens=[prefix_len],
        extend_lens=[4],
        seq_lens=np.array([prefix_len + 4], dtype=np.int32),
    )

    plan = build_mm_embed_plan(
        reqs_info=[info],
        dp_size=1,
        model_config=model_config,
        per_dp_token=4,
    )

    batch = plan[Modality.IMAGE]
    np.testing.assert_array_equal(batch.encode_inputs.valid, np.array([[16]], dtype=np.int32))
    np.testing.assert_array_equal(batch.merge.src_idx[0, 0], expected_src)
    np.testing.assert_array_equal(batch.merge.dst_idx[0, 0], expected_dst)
    assert batch.merge.mask[0, 0].all()


def test_mm_embed_plan_uses_full_encoded_length_for_chunked_task_placement():
    items = _build_image_items(
        np.arange(32, dtype=np.float32).reshape(32, 1),
        [(1, 4, 4), (1, 4, 4)],
        [(2, 5), (6, 9)],
    )
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=items),
        extend_input_len=4,
    )
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        fullatt_block_indexes=[],
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=vision_config,
        ),
    )
    info = ScheduleReqsInfo(
        reqs=[req],
        prefix_lens=[4],
        extend_lens=[4],
        seq_lens=np.array([8], dtype=np.int32),
    )

    plan = build_mm_embed_plan(
        reqs_info=[info],
        dp_size=1,
        model_config=model_config,
        per_dp_token=4,
    )

    batch = plan[Modality.IMAGE]
    np.testing.assert_array_equal(batch.encode_inputs.valid, np.array([[32]], dtype=np.int32))
    np.testing.assert_array_equal(batch.merge.src_idx[0, 0], np.array([2, 3, 4, 5]))
    np.testing.assert_array_equal(batch.merge.dst_idx[0, 0], np.array([0, 1, 2, 3]))
    assert batch.merge.mask[0, 0].all()


def test_mm_embed_plan_skips_visual_items_outside_chunk_window():
    item = _build_image_items(
        np.arange(16, dtype=np.float32).reshape(16, 1),
        [(1, 4, 4)],
        [(2, 5)],
    )[0]
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=[item]),
        extend_input_len=2,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=SimpleNamespace(),
        ),
    )
    info = ScheduleReqsInfo(
        reqs=[req],
        prefix_lens=[6],
        extend_lens=[2],
        seq_lens=np.array([8], dtype=np.int32),
    )

    plan = build_mm_embed_plan(
        reqs_info=[info],
        dp_size=1,
        model_config=model_config,
        per_dp_token=2,
    )

    assert plan is None


def test_mm_embed_plan_fails_fast_on_overlapping_placeholder_ranges():
    features = np.arange(3, dtype=np.float32).reshape(3, 1)
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=features,
        placeholder_ranges=[(0, 1), (1, 1)],
        model_specific_data={"image_grid_thw": np.array([[1, 1, 3]], dtype=np.int32)},
    )
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=[item]),
        extend_input_len=3,
    )
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=1,
        fullatt_block_indexes=[],
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=vision_config,
        ),
    )

    with pytest.raises(ValueError, match="assigned more than once"):
        build_mm_embed_plan(
            reqs_info=[ScheduleReqsInfo(reqs=[req])],
            dp_size=1,
            model_config=model_config,
            per_dp_token=3,
        )


def test_mm_embed_plan_packs_per_request_with_dp_dummy_lane():
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        fullatt_block_indexes=[],
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=vision_config,
        ),
    )
    rank0_req0 = SimpleNamespace(
        mm_inputs=MultimodalInputs(
            mm_items=_build_image_items(
                np.arange(8, dtype=np.float32).reshape(8, 1),
                [(1, 2, 4)],
                [(1, 2)],
            )
        ),
        extend_input_len=4,
    )
    rank0_req1 = SimpleNamespace(
        mm_inputs=MultimodalInputs(
            mm_items=_build_image_items(
                np.arange(16, dtype=np.float32).reshape(16, 1),
                [(1, 4, 4)],
                [(0, 3)],
            )
        ),
        extend_input_len=5,
    )
    rank1_req0 = SimpleNamespace(
        mm_inputs=MultimodalInputs(
            mm_items=_build_image_items(
                np.arange(8, dtype=np.float32).reshape(8, 1),
                [(1, 2, 4)],
                [(2, 3)],
            )
        ),
        extend_input_len=4,
    )

    plan = build_mm_embed_plan(
        reqs_info=[
            ScheduleReqsInfo(reqs=[rank0_req0, rank0_req1]),
            ScheduleReqsInfo(reqs=[rank1_req0]),
        ],
        dp_size=2,
        model_config=model_config,
        per_dp_token=10,
    )

    batch = plan[Modality.IMAGE]
    np.testing.assert_array_equal(
        batch.encode_inputs.valid,
        np.array([[24], [8]], dtype=np.int32),
    )
    np.testing.assert_array_equal(batch.merge.dst_idx[0, 0], np.array([1, 2, 4, 5, 6, 7]))
    np.testing.assert_array_equal(batch.merge.dst_idx[1, 0, :2], np.array([2, 3]))
    np.testing.assert_array_equal(batch.merge.mask.sum(axis=2), np.array([[6], [2]]))
    np.testing.assert_array_equal(
        batch.encode_inputs.meta.cu_image_seqlens,
        np.array([[[8, 24]], [[8, 24]]], dtype=np.int32),
    )


def test_mm_embed_plan_pads_dp_ranks_with_uneven_multi_image_requests():
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        fullatt_block_indexes=[],
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=vision_config,
        ),
    )
    rank0_items = _build_image_items(
        np.arange(24, dtype=np.float32).reshape(24, 1),
        [(1, 2, 4), (1, 4, 4)],
        [(0, 1), (3, 6)],
    )
    rank1_items = _build_image_items(
        np.arange(32, dtype=np.float32).reshape(32, 1),
        [(1, 2, 4), (1, 2, 4), (1, 4, 4)],
        [(1, 2), (4, 5), (7, 10)],
    )
    rank0_req = SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=rank0_items),
        extend_input_len=8,
    )
    rank1_req = SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=rank1_items),
        extend_input_len=12,
    )

    plan = build_mm_embed_plan(
        reqs_info=[ScheduleReqsInfo(reqs=[rank0_req]), ScheduleReqsInfo(reqs=[rank1_req])],
        dp_size=2,
        model_config=model_config,
        per_dp_token=12,
    )

    batch = plan[Modality.IMAGE]
    np.testing.assert_array_equal(
        batch.encode_inputs.valid,
        np.array([[24], [32]], dtype=np.int32),
    )
    assert batch.encode_inputs.patches.shape == (2, 1, 32, 1)
    np.testing.assert_array_equal(
        batch.encode_inputs.meta.cu_image_seqlens,
        np.array([[[8, 24, 32]], [[8, 16, 32]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.merge.dst_idx[0, 0, :6],
        np.array([0, 1, 3, 4, 5, 6]),
    )
    np.testing.assert_array_equal(
        batch.merge.dst_idx[1, 0],
        np.array([1, 2, 4, 5, 7, 8, 9, 10]),
    )
    np.testing.assert_array_equal(batch.merge.src_idx[0, 0, :6], np.arange(6))
    np.testing.assert_array_equal(batch.merge.src_idx[1, 0], np.arange(8))


def test_qwen_metadata_builder_checks_feature_rows_match_grid():
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    builder = Qwen25VLVisionMetadataBuilder(
        SimpleNamespace(hf_config=SimpleNamespace(vision_config=vision_config))
    )
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=np.ones((7, 1), dtype=np.float32),
        placeholder_ranges=[(0, 1)],
        model_specific_data={"image_grid_thw": np.array([[1, 2, 4]], dtype=np.int32)},
    )

    with pytest.raises(ValueError, match="feature rows"):
        builder._get_image_metadata(item)


def test_qwen_metadata_builder_checks_placeholder_rows_match_grid():
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    builder = Qwen25VLVisionMetadataBuilder(
        SimpleNamespace(hf_config=SimpleNamespace(vision_config=vision_config))
    )
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=np.ones((8, 1), dtype=np.float32),
        placeholder_ranges=[(0, 0)],
        model_specific_data={"image_grid_thw": np.array([[1, 2, 4]], dtype=np.int32)},
    )

    with pytest.raises(ValueError, match="placeholder rows"):
        builder._get_image_metadata(item)


def test_mm_embed_plan_rejects_dict_mm_inputs():
    feature = np.arange(8, dtype=np.float32).reshape(8, 1)
    req = SimpleNamespace(
        mm_inputs={
            "mm_items": [
                {
                    "modality": "image",
                    "feature": feature,
                    "placeholder_ranges": [(0, 1)],
                    "image_grid_thw": np.array([[1, 2, 4]], dtype=np.int32),
                }
            ]
        },
        extend_input_len=2,
    )
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        fullatt_block_indexes=[],
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=vision_config,
        ),
    )

    with pytest.raises(TypeError, match="MultimodalInputs"):
        build_mm_embed_plan(
            reqs_info=[ScheduleReqsInfo(reqs=[req])],
            dp_size=1,
            model_config=model_config,
            per_dp_token=2,
        )


def test_mm_embed_plan_returns_none_before_resolving_builder_without_images():
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.AUDIO,
                    feature=np.ones((4, 2), dtype=np.float32),
                )
            ]
        ),
        extend_input_len=4,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["NoVisionBuilderForAudioOnly"],
            vision_config=SimpleNamespace(),
        ),
    )

    plan = build_mm_embed_plan(
        reqs_info=[ScheduleReqsInfo(reqs=[req])],
        dp_size=1,
        model_config=model_config,
        per_dp_token=4,
    )

    assert plan is None


def test_qwen_mm_embed_plan_ignores_unsupported_audio_items():
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.AUDIO,
                    feature=np.ones((4, 2), dtype=np.float32),
                )
            ]
        ),
        extend_input_len=4,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=SimpleNamespace(),
        )
    )

    plan = build_mm_embed_plan(
        reqs_info=[ScheduleReqsInfo(reqs=[req])],
        dp_size=1,
        model_config=model_config,
        per_dp_token=4,
    )

    assert plan is None


def test_qwen_mm_embed_plan_preserves_video_items_on_the_vision_embedder():
    items = QwenVLProcessor._build_items(
        np.arange(8, dtype=np.float32).reshape(8, 1),
        [(1, 2, 4)],
        [(0, 1)],
        Modality.VIDEO,
        "video_grid_thw",
    )
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=items),
        extend_input_len=2,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=SimpleNamespace(),
        )
    )

    plan = build_mm_embed_plan(
        reqs_info=[ScheduleReqsInfo(reqs=[req])],
        dp_size=1,
        model_config=model_config,
        per_dp_token=2,
    )

    assert tuple(plan) == (Modality.IMAGE,)
    np.testing.assert_array_equal(
        plan[Modality.IMAGE].encode_inputs.valid,
        np.array([[8]], dtype=np.int32),
    )


def test_mm_embed_plan_fails_fast_when_qwen_vision_config_missing():
    features = np.arange(8, dtype=np.float32).reshape(8, 1)
    items = _build_image_items(features, [(1, 2, 4)], [(0, 1)])
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=items),
        extend_input_len=2,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
        ),
    )

    with pytest.raises(ValueError, match="vision_config"):
        build_mm_embed_plan(
            reqs_info=[ScheduleReqsInfo(reqs=[req])],
            dp_size=1,
            model_config=model_config,
            per_dp_token=2,
        )
