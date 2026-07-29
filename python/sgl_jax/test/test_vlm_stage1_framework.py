from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    ScheduleBatch,
    ScheduleReqsInfo,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.models.qwen2_5_vl import (
    Qwen2_5_VisionTransformer,
    _apply_rotary_pos_emb_vision,
    _segment_ids_from_cu_seqlens,
    _vision_attention,
)
from sgl_jax.srt.models.qwen3_vl import Qwen3VLVisionModel
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.in_model import embedding_pipeline
from sgl_jax.srt.multimodal.in_model.embedding_pipeline import (
    _MergeMapping,
    _MultimodalEmbeddingCache,
    build_multimodal_batch,
)
from sgl_jax.srt.multimodal.in_model.interface import (
    InModelMultimodalContract,
    MultimodalEmbeddingOutput,
)
from sgl_jax.srt.multimodal.in_model.placement import place_on_dp
from sgl_jax.srt.multimodal.kernels.flash_attention import SegmentIds
from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
    VisionFlashAttentionBackend,
)
from sgl_jax.srt.multimodal.layers.vision_sharding import (
    VisionShardSpecs,
    schedule_vision_lanes,
)
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor
from sgl_jax.srt.server_args import apply_multimodal_model_defaults

ARCH = "Qwen2_5_VLForConditionalGeneration"


class _TestInModelModel(InModelMultimodalContract):
    def get_input_embeddings(self):
        return lambda input_ids: input_ids


class _EncoderModel(_TestInModelModel):
    def __init__(self, modality, encoder):
        self.modality = modality
        self.encoder = encoder

    def get_multimodal_embedding_funcs(self):
        return {self.modality: self.encoder}


def _vision_config(**overrides):
    values = {
        "patch_size": 1,
        "temporal_patch_size": 1,
        "in_channels": 1,
        "hidden_size": 4,
        "depth": 0,
        "intermediate_size": 8,
        "hidden_act": "silu",
        "num_heads": 1,
        "out_hidden_size": 4,
        "spatial_merge_size": 1,
        "fullatt_block_indexes": [],
        "window_size": 1,
        "rope_theta": 10000.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _qwen_config(**overrides):
    values = {
        "patch_size": 14,
        "window_size": 112,
        "spatial_merge_size": 2,
        "num_heads": 16,
        "hidden_size": 1280,
        "out_hidden_size": 1280,
    }
    values.update(overrides)
    return _vision_config(**values)


def _model_config(vision_config=None, arch=ARCH):
    return SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=[arch],
            vision_config=vision_config or _qwen_config(),
        ),
    )


def _visual(config=None, mesh=None, encoder_tp=False, input_buckets=(32,)):
    mesh = mesh or _mesh()
    with jax.set_mesh(mesh):
        return Qwen2_5_VisionTransformer(
            config or _vision_config(),
            jnp.float32,
            mesh=mesh,
            vision_tp=encoder_tp,
            input_buckets=input_buckets,
        )


def _build_items(features, grids, ranges, modality=Modality.IMAGE):
    key = "image_grid_thw" if modality == Modality.IMAGE else "video_grid_thw"
    return QwenVLProcessor._build_items(features, grids, ranges, modality, key)


def _items(grids, ranges, modality=Modality.IMAGE):
    rows = sum(int(np.prod(grid)) for grid in grids)
    features = np.arange(rows, dtype=np.float32).reshape(rows, 1)
    return _build_items(features, grids, ranges, modality)


def _req(items, extend_len):
    return SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=items),
        extend_input_len=extend_len,
        lora_id="0",
    )


def _batch(items, *, config=None, prefix=0, extend=None, per_dp_token=None):
    ends = [end for item in items for _, end in (item.placeholder_ranges or [])]
    max_end = max(ends, default=extend or 1)
    extend = max_end - prefix if extend is None else extend
    per_dp_token = extend if per_dp_token is None else per_dp_token
    info = ScheduleReqsInfo(
        reqs=[_req(items, extend)],
        prefix_lens=[prefix],
        extend_lens=[extend],
        seq_lens=np.array([prefix + extend], dtype=np.int32),
    )
    return build_multimodal_batch(
        [info],
        1,
        _model_config(config),
        per_dp_token,
    )


def _batch_dp(items_by_dp, *, config=None, per_dp_token):
    infos = []
    for items in items_by_dp:
        ends = [end for item in items for _, end in (item.placeholder_ranges or [])]
        extend = max(ends, default=1)
        infos.append(
            ScheduleReqsInfo(
                reqs=[_req(items, extend)] if items else [],
                prefix_lens=[0] if items else [],
                extend_lens=[extend] if items else [],
                seq_lens=np.asarray([extend] if items else [], dtype=np.int32),
            )
        )
    return build_multimodal_batch(
        infos,
        len(items_by_dp),
        _model_config(config),
        per_dp_token,
    )


def _mesh(dp=1, tp=1):
    count = dp * tp
    if len(jax.devices()) < count:
        pytest.skip(f"requires {count} devices")
    return Mesh(
        np.asarray(jax.devices()[:count]).reshape(dp, tp),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _schedule_batch(req, model_config=None):
    input_ids = np.arange(req.extend_input_len, dtype=np.int32)
    info = ScheduleReqsInfo(
        reqs=[req],
        input_ids=input_ids,
        seq_lens=np.array([len(input_ids)], dtype=np.int32),
        out_cache_loc=np.arange(1, len(input_ids) + 1, dtype=np.int32),
        req_pool_indices=np.array([0], dtype=np.int32),
        prefix_lens=np.array([0], dtype=np.int32),
        extend_lens=np.array([len(input_ids)], dtype=np.int32),
        extend_logprob_start_lens=np.array([0], dtype=np.int32),
    )
    batch = ScheduleBatch(
        reqs_info=[info],
        dp_size=1,
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
        model_config=model_config,
    )
    batch._merge_sampling_info = lambda *_: None
    batch._merge_cache_loc = lambda *_: info.out_cache_loc
    return batch


def test_multimodal_contract_discovers_conventional_encoders():
    class Model(InModelMultimodalContract):
        def get_input_embeddings(self):
            return lambda values: values

        def get_image_feature(self, _):
            return "image"

        def get_video_feature(self, _):
            return "video"

        def get_audio_feature(self, _):
            return "audio"

    funcs = Model().get_multimodal_embedding_funcs()
    assert funcs[Modality.IMAGE]([]) == "image"
    assert funcs[Modality.MULTI_IMAGES]([]) == "image"
    assert funcs[Modality.VIDEO]([]) == "video"
    assert funcs[Modality.AUDIO]([]) == "audio"


def test_embedding_cache_deduplicates_items_and_keys_metadata():
    def item(grid, start):
        return MultimodalDataItem(
            Modality.AUDIO,
            feature=np.ones((1, 1), dtype=np.float32),
            placeholder_ranges=[(start, start + 1)],
            model_specific_data={"grid": np.asarray(grid, dtype=np.int32)},
        )

    first = [item((1, 2, 4), 0), item((1, 2, 4), 1), item((1, 4, 2), 2)]
    calls = []

    def encode(items):
        calls.append(items)
        return [np.asarray([[value.grid[1]]], dtype=np.float32) for value in items]

    cache = _MultimodalEmbeddingCache(1024)
    output, _ = embedding_pipeline.general_mm_embed_routine(
        _batch(first, extend=3),
        jnp.zeros(3, dtype=jnp.int32),
        lambda _: jnp.zeros((3, 1), dtype=jnp.float32),
        _EncoderModel(Modality.AUDIO, encode),
        cache,
    )
    np.testing.assert_array_equal(output[:, 0], [2, 2, 4])
    assert calls == [[first[0], first[2]]]

    second = [item((1, 4, 2), 0), item((1, 2, 4), 1)]
    output, _ = embedding_pipeline.general_mm_embed_routine(
        _batch(second, extend=2),
        jnp.zeros(2, dtype=jnp.int32),
        lambda _: jnp.zeros((2, 1), dtype=jnp.float32),
        _EncoderModel(Modality.AUDIO, encode),
        cache,
    )
    np.testing.assert_array_equal(output[:, 0], [4, 2])
    assert len(calls) == 1


def test_embedding_cache_is_namespaced_by_owner_dp():
    rank0 = MultimodalDataItem(
        Modality.AUDIO,
        hash=7,
        feature=np.ones((1, 1), dtype=np.float32),
        placeholder_ranges=[(0, 1)],
    )
    rank1 = MultimodalDataItem(
        Modality.AUDIO,
        hash=7,
        feature=np.ones((1, 1), dtype=np.float32),
        placeholder_ranges=[(0, 1)],
    )
    calls = []

    def batch(rank0_items, rank1_items):
        infos = [
            ScheduleReqsInfo(reqs=[_req(rank0_items, 1)] if rank0_items else []),
            ScheduleReqsInfo(reqs=[_req(rank1_items, 1)] if rank1_items else []),
        ]
        return build_multimodal_batch(infos, 2, _model_config(), 1)

    def encode(items):
        calls.append((type(items), tuple(id(item) for item in items)))
        return [np.ones((1, 1), dtype=np.float32) for _ in items]

    cache = _MultimodalEmbeddingCache(1024)
    model = _EncoderModel(Modality.AUDIO, encode)
    for multimodal_batch in (batch([rank0], []), batch([rank0], [rank1])):
        embedding_pipeline.general_mm_embed_routine(
            multimodal_batch,
            jnp.zeros(2, dtype=jnp.int32),
            lambda _: jnp.zeros((2, 1), dtype=jnp.float32),
            model,
            cache,
        )

    assert calls == [
        (list, (id(rank0),)),
        (list, (id(rank1),)),
    ]


def test_resolve_items_calls_plain_encoder_once_for_all_owners():
    mesh = _mesh(dp=2)
    items = [
        MultimodalDataItem(
            Modality.AUDIO,
            hash=rank,
            feature=np.asarray([[10 + rank]], dtype=np.float32),
            placeholder_ranges=[(0, 1)],
        )
        for rank in range(2)
    ]
    calls = []

    def encode(values):
        calls.append(values)
        return [value.feature for value in values]

    model = _EncoderModel(Modality.AUDIO, encode)
    model.mesh = mesh
    running = jax.device_put(
        jnp.zeros((2, 1)),
        NamedSharding(mesh, PartitionSpec("data", None)),
    )
    with patch.object(
        embedding_pipeline,
        "encode_multimodal_items",
        wraps=embedding_pipeline.encode_multimodal_items,
    ) as dispatch:
        output, _ = embedding_pipeline.general_mm_embed_routine(
            _batch_dp(([items[0]], [items[1]]), per_dp_token=1),
            jnp.zeros(2, dtype=jnp.int32),
            lambda _: running,
            model,
        )

    assert calls == [items]
    assert dispatch.call_count == 1
    assert dispatch.call_args.args[0] == ((items[0],), (items[1],))
    np.testing.assert_array_equal(output[:, 0], [10, 11])


def test_precomputed_audio_embeddings_use_the_same_dp_merge_path():
    mesh = _mesh(dp=2, tp=2)
    items = [
        MultimodalDataItem(
            Modality.AUDIO,
            precomputed_embeddings=np.asarray(values, dtype=np.float32)[:, None],
            placeholder_ranges=[(0, 2)],
        )
        for values in ((10, 11), (20, 21))
    ]
    model = _EncoderModel(
        Modality.AUDIO,
        lambda _: pytest.fail("precomputed embeddings must bypass the encoder"),
    )
    model.mesh = mesh
    running = jax.device_put(
        jnp.zeros((4, 1)),
        NamedSharding(mesh, PartitionSpec("data", None)),
    )

    output, _ = embedding_pipeline.general_mm_embed_routine(
        _batch_dp(([items[0]], [items[1]]), per_dp_token=2),
        jnp.zeros(4, dtype=jnp.int32),
        lambda _: running,
        model,
    )

    np.testing.assert_array_equal(output[:, 0], [10, 11, 20, 21])


def test_item_identity_includes_encoder_metadata_and_stage():
    feature = np.ones((1, 1), dtype=np.float32)
    items = [
        MultimodalDataItem(
            Modality.IMAGE,
            feature=feature,
            model_specific_data={"grid": np.asarray(grid, dtype=np.int32)},
        )
        for grid in ((1, 2, 4), (1, 4, 2))
    ]
    items.append(
        MultimodalDataItem(
            Modality.IMAGE,
            precomputed_embeddings=feature,
            model_specific_data={"grid": np.asarray((1, 2, 4), dtype=np.int32)},
        )
    )
    for item in items:
        item.set_pad_value()

    assert len({item.hash for item in items}) == len(items)


def test_item_pad_value_is_a_full_width_synthetic_token():
    first = MultimodalDataItem(Modality.IMAGE, hash=1)
    second = MultimodalDataItem(Modality.IMAGE, hash=1 + (1 << 24))
    first.set_pad_value()
    second.set_pad_value()

    assert first.pad_value == -2
    assert second.pad_value != first.pad_value
    assert second.pad_value is not None and second.pad_value < 0


def test_embedding_cache_reuses_full_item_across_chunks():
    item = MultimodalDataItem(
        Modality.AUDIO,
        feature=np.arange(4, dtype=np.float32)[:, None],
        placeholder_ranges=[(2, 6)],
    )
    calls = 0

    def encode(_):
        nonlocal calls
        calls += 1
        return jnp.arange(10, 14, dtype=jnp.float32)[:, None]

    cache = _MultimodalEmbeddingCache(1024)
    first, _ = embedding_pipeline.general_mm_embed_routine(
        _batch([item], prefix=0, extend=4),
        jnp.zeros(4, dtype=jnp.int32),
        lambda _: jnp.zeros((4, 1), dtype=jnp.float32),
        _EncoderModel(Modality.AUDIO, encode),
        cache,
    )
    second, _ = embedding_pipeline.general_mm_embed_routine(
        _batch([item], prefix=4, extend=4),
        jnp.zeros(4, dtype=jnp.int32),
        lambda _: jnp.zeros((4, 1), dtype=jnp.float32),
        _EncoderModel(Modality.AUDIO, encode),
        cache,
    )
    np.testing.assert_array_equal(first[:, 0], [0, 0, 10, 11])
    np.testing.assert_array_equal(second[:, 0], [12, 13, 0, 0])
    assert calls == 1


def test_embedding_cache_uses_byte_lru():
    calls = []

    def encode(items):
        calls.extend(int(value.feature[0, 0]) for value in items)
        return [np.full((1, 2), value.feature[0, 0], dtype=np.float32) for value in items]

    def run(value):
        item = MultimodalDataItem(
            Modality.AUDIO,
            hash=value,
            feature=np.asarray([[value]], dtype=np.float32),
            placeholder_ranges=[(0, 1)],
        )
        embedding_pipeline.general_mm_embed_routine(
            _batch([item]),
            jnp.zeros(1, dtype=jnp.int32),
            lambda _: jnp.zeros((1, 2), dtype=jnp.float32),
            _EncoderModel(Modality.AUDIO, encode),
            cache,
        )

    cache = _MultimodalEmbeddingCache(16)
    for value in (1, 2, 1, 3, 1, 2):
        run(value)
    assert calls == [1, 2, 3, 2]


def test_embedding_cache_owns_cached_deepstack_values():
    item = MultimodalDataItem(
        Modality.IMAGE,
        feature=np.ones((2, 1), dtype=np.float32),
        placeholder_ranges=[(0, 2)],
    )
    embeddings = np.asarray([[1], [2]], dtype=np.float32)
    stacked = np.asarray([[[3], [4]]], dtype=np.float32)
    calls = 0

    def encode(_):
        nonlocal calls
        calls += 1
        return MultimodalEmbeddingOutput(embeddings, stacked)

    cache = _MultimodalEmbeddingCache(1024)
    batch = _batch([item])
    args = (
        batch,
        jnp.zeros(2, dtype=jnp.int32),
        lambda _: jnp.zeros((2, 1), dtype=jnp.float32),
        _EncoderModel(Modality.IMAGE, encode),
        cache,
    )
    first, first_deepstack = embedding_pipeline.general_mm_embed_routine(*args)
    embeddings[:] = 9
    stacked[:] = 9
    second, second_deepstack = embedding_pipeline.general_mm_embed_routine(*args)
    np.testing.assert_array_equal(first[:, 0], [1, 2])
    np.testing.assert_array_equal(second[:, 0], [1, 2])
    np.testing.assert_array_equal(first_deepstack[0, :, 0], [3, 4])
    np.testing.assert_array_equal(second_deepstack[0, :, 0], [3, 4])
    assert calls == 1


def test_deepstack_uses_the_same_chunk_slice():
    item = _items([(1, 1, 4)], [(2, 6)])[0]
    batch = _batch([item], prefix=4, extend=2, per_dp_token=2)

    result, deepstack = embedding_pipeline.general_mm_embed_routine(
        batch,
        jnp.zeros(2, dtype=jnp.int32),
        lambda _: jnp.zeros((2, 1), dtype=jnp.float32),
        _EncoderModel(
            Modality.IMAGE,
            lambda _: MultimodalEmbeddingOutput(
                jnp.arange(4, dtype=jnp.float32)[:, None],
                jnp.arange(100, 104, dtype=jnp.float32)[None, :, None],
            ),
        ),
    )
    np.testing.assert_array_equal(result[:, 0], [2, 3])
    np.testing.assert_array_equal(deepstack[0, :, 0], [102, 103])


@pytest.mark.parametrize(
    ("encode_item_groups", "with_deepstack"),
    [
        pytest.param(Qwen2_5_VisionTransformer.encode_item_groups, False, id="qwen2"),
        pytest.param(Qwen3VLVisionModel.encode_item_groups, True, id="qwen3"),
    ],
)
def test_vision_packed_output_preserves_item_order(
    encode_item_groups,
    with_deepstack,
):
    calls = []
    lengths = (2, 5, 3, 6)
    placements = ((1, 0), (0, 0), (3, 0), (2, 0))
    packed = np.zeros((4, 6, 1), dtype=np.float32)
    for index, ((lane, start), length) in enumerate(zip(placements, lengths, strict=True)):
        packed[lane, start : start + length] = index

    def batch_items(items, items_per_data_rank):
        calls.append(("batch", type(items), items_per_data_rank))
        return None, None, None, placements

    def encode(*_):
        calls.append(("encode",))
        output = jnp.asarray(packed)
        if not with_deepstack:
            return output
        return output, output[:, None] + 10

    visual = SimpleNamespace(
        _batch_items=batch_items,
        encode=encode,
        spatial_merge_unit=1,
        mesh=None,
    )
    items = [
        MultimodalDataItem(
            Modality.IMAGE,
            feature=np.full((length, 1), index, dtype=np.float32),
            model_specific_data={"image_grid_thw": np.asarray([[1, 1, length]], dtype=np.int32)},
        )
        for index, length in enumerate(lengths)
    ]

    output = encode_item_groups(visual, (tuple(items),))
    embeddings = output.embeddings if with_deepstack else output

    assert calls == [("batch", list, (len(items),)), ("encode",)]
    assert [tuple(np.asarray(value[:, 0], dtype=int)) for value in embeddings] == [
        (0, 0),
        (1, 1, 1, 1, 1),
        (2, 2, 2),
        (3, 3, 3, 3, 3, 3),
    ]
    if with_deepstack:
        assert [int(value[0, 0, 0]) for value in output.deepstack] == [10, 11, 12, 13]


@pytest.mark.parametrize(
    ("vision_tp", "expected_lanes"),
    [
        (False, [[0], [1, 2], [3], [4]]),
        (True, [[0, 1, 2], [3, 4]]),
    ],
)
def test_vision_batch_layout_is_owner_local(vision_tp, expected_lanes):
    lengths = (8, 4, 2, 7, 6)
    lanes = schedule_vision_lanes(
        lengths,
        data_size=2,
        tensor_size=2,
        vision_tp=vision_tp,
        items_per_data_rank=(3, 2),
    )
    assert lanes == expected_lanes

    fake_mesh = SimpleNamespace(axis_names=("data", "tensor"))
    expected_axis = "data" if vision_tp else ("data", "tensor")
    assert VisionShardSpecs(fake_mesh, vision_tp).batch_spec(None) == PartitionSpec(
        expected_axis,
        None,
    )


def _assert_vision_precompile(visual):
    calls = []

    def encode(patches, metadata, valid):
        calls.append(
            (
                patches.shape,
                valid.shape,
                tuple(leaf.shape[0] for leaf in jax.tree.leaves(metadata)),
            )
        )

    with patch.object(type(visual), "encode", side_effect=encode):
        visual.precompile()

    assert calls == [
        ((1, 4, 1), (1,), (1, 1, 1, 1)),
        ((1, 8, 1), (1,), (1, 1, 1, 1)),
    ]


def test_qwen2_vision_precompile_warms_configured_buckets():
    config = _vision_config(
        spatial_merge_size=2,
        window_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    _assert_vision_precompile(_visual(config=config, input_buckets=(4, 8)))


def test_qwen2_vision_rejects_unaligned_buckets():
    with pytest.raises(ValueError, match="positive multiples of 4"):
        _visual(config=_vision_config(spatial_merge_size=2), input_buckets=(3,))


@pytest.mark.parametrize("encoder_tp", [False, True])
def test_qwen2_owner_grouped_packed_spmd(encoder_tp):
    mesh = _mesh(dp=2, tp=2)
    visual = _visual(
        mesh=mesh,
        encoder_tp=encoder_tp,
        input_buckets=(8,),
    )
    items = _items(
        [(1, 1, length) for length in (8, 4, 2, 7, 6)],
        [(0, length) for length in (8, 4, 2, 7, 6)],
    )
    patches, _, valid, placements = visual._batch_items(items, (3, 2))

    if encoder_tp:
        assert placements == ((0, 0), (0, 8), (0, 12), (1, 0), (1, 7))
        expected_valid = {
            mesh.devices[0, 0]: (14,),
            mesh.devices[0, 1]: (14,),
            mesh.devices[1, 0]: (13,),
            mesh.devices[1, 1]: (13,),
        }
        expected_patches = {
            mesh.devices[0, 0]: tuple(range(14)) + (0, 0),
            mesh.devices[0, 1]: tuple(range(14)) + (0, 0),
            mesh.devices[1, 0]: tuple(range(14, 27)) + (0, 0, 0),
            mesh.devices[1, 1]: tuple(range(14, 27)) + (0, 0, 0),
        }
        expected_spec = PartitionSpec("data")
    else:
        assert placements == ((0, 0), (1, 0), (1, 4), (2, 0), (3, 0))
        expected_valid = {
            mesh.devices[0, 0]: (8,),
            mesh.devices[0, 1]: (6,),
            mesh.devices[1, 0]: (7,),
            mesh.devices[1, 1]: (6,),
        }
        expected_patches = {
            mesh.devices[0, 0]: tuple(range(8)),
            mesh.devices[0, 1]: tuple(range(8, 14)) + (0, 0),
            mesh.devices[1, 0]: tuple(range(14, 21)) + (0,),
            mesh.devices[1, 1]: tuple(range(21, 27)) + (0, 0),
        }
        expected_spec = PartitionSpec(("data", "tensor"))

    assert patches.sharding.spec[0] == expected_spec[0]
    valid_shards = {
        shard.device: tuple(int(value) for value in np.asarray(shard.data).reshape(-1))
        for shard in valid.addressable_shards
    }
    patch_shards = {
        shard.device: tuple(int(value) for value in np.asarray(shard.data).reshape(-1))
        for shard in patches.addressable_shards
    }
    assert valid_shards == expected_valid
    assert patch_shards == expected_patches


@pytest.mark.parametrize("encoder_tp", [False, True])
def test_qwen2_packed_encode_spmd(encoder_tp):
    mesh = _mesh(dp=2, tp=2)
    visual = _visual(
        mesh=mesh,
        encoder_tp=encoder_tp,
        input_buckets=(4,),
    )
    items = _items([(1, 1, 4), (1, 1, 2)], [(0, 4), (4, 6)])
    outputs = visual.encode_item_groups(((items[0],), (items[1],)))
    assert [output.shape for output in outputs] == [(4, 4), (2, 4)]
    for owner, output in enumerate(outputs):
        assert output.sharding.is_fully_replicated
        assert output.sharding.device_set == set(mesh.devices[owner])
    calls = 0

    class Model(_TestInModelModel):
        mesh = visual.mesh

        def get_multimodal_embedding_funcs(self):
            return {Modality.IMAGE: self.encode}

        def get_packed_multimodal_embedding_funcs(self):
            return {Modality.IMAGE: self.encode_groups}

        @staticmethod
        def encode(values):
            pytest.fail("owner-grouped Qwen path must use the packed encoder")

        @staticmethod
        def encode_groups(values_by_owner):
            nonlocal calls
            calls += 1
            return visual.encode_item_groups(values_by_owner)

    running = jax.device_put(
        jnp.zeros((8, 4)),
        NamedSharding(mesh, PartitionSpec("data", None)),
    )
    runtime_items = _items([(1, 1, 4), (1, 1, 2)], [(0, 4), (0, 2)])
    cache = _MultimodalEmbeddingCache(1024)
    args = (
        _batch_dp(([runtime_items[0]], [runtime_items[1]]), per_dp_token=4),
        jnp.zeros(8, dtype=jnp.int32),
        lambda _: running,
        Model(),
        cache,
    )
    first, _ = embedding_pipeline.general_mm_embed_routine(*args)
    second, _ = embedding_pipeline.general_mm_embed_routine(*args)
    np.testing.assert_array_equal(first, second)
    assert first.sharding.spec == PartitionSpec("data", None)
    assert calls == 1
    for key, value in cache._entries.items():
        owner_devices = set(mesh.devices[key[0]])
        assert value.embeddings.sharding.device_set == owner_devices
        if value.deepstack is not None:
            assert value.deepstack.sharding.device_set == owner_devices


def test_qwen3_vision_precompile_warms_configured_buckets():
    config = _vision_config(
        spatial_merge_size=2,
        window_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    mesh = _mesh()
    with jax.set_mesh(mesh):
        visual = Qwen3VLVisionModel(
            config,
            jnp.float32,
            mesh=mesh,
            input_buckets=(4, 8),
        )
    _assert_vision_precompile(visual)


def test_qwen3_vision_rejects_unaligned_buckets():
    with pytest.raises(ValueError, match="positive multiples of 4"):
        Qwen3VLVisionModel(
            _vision_config(spatial_merge_size=2),
            jnp.float32,
            mesh=_mesh(),
            input_buckets=(3,),
        )


@pytest.mark.parametrize("encoder_tp", [False, True])
def test_qwen3_packed_encode_spmd(encoder_tp):
    mesh = _mesh(dp=2, tp=2)
    config = _vision_config(
        num_position_embeddings=16,
        depth=1,
        deepstack_visual_indexes=[0],
        num_heads=2,
    )
    with jax.set_mesh(mesh):
        visual = Qwen3VLVisionModel(
            config,
            jnp.float32,
            mesh=mesh,
            tp=encoder_tp,
            input_buckets=(4,),
        )
    items = _items([(1, 1, 2), (1, 1, 4)], [(0, 2), (2, 6)])
    output = visual.encode_item_groups(((items[0],), (items[1],)))
    assert [value.shape for value in output.embeddings] == [(2, 4), (4, 4)]
    for owner, value in enumerate(output.embeddings):
        assert value.sharding.is_fully_replicated
        assert value.sharding.device_set == set(mesh.devices[owner])
    for owner, value in enumerate(output.deepstack):
        assert value.sharding.is_fully_replicated
        assert value.sharding.device_set == set(mesh.devices[owner])


def test_batch_separates_patch_and_placeholder_counts():
    items = _items([(1, 2, 4), (1, 4, 4)], [(2, 4), (5, 9)])
    tasks = _batch(items, extend=10, per_dp_token=10)[Modality.IMAGE]
    assert [task.item for task in tasks] == items
    assert [task.output_len for task in tasks] == [2, 4]
    assert [task.merge_mappings for task in tasks] == [
        (_MergeMapping(0, 2, 2),),
        (_MergeMapping(0, 5, 4),),
    ]


@pytest.mark.parametrize(
    ("prefix", "extend", "destination", "source"),
    [
        (0, 4, [2, 3], [0, 1]),
        (4, 4, [0, 1], [2, 3]),
        (6, 2, None, None),
    ],
)
def test_batch_clips_to_chunk_boundaries(prefix, extend, destination, source):
    items = _items([(1, 4, 4)], [(2, 6)])
    batch = _batch(items, prefix=prefix, extend=extend, per_dp_token=extend)
    if destination is None:
        assert batch is None
    else:
        mapping = batch[Modality.IMAGE][0].merge_mappings[0]
        np.testing.assert_array_equal(
            range(
                mapping.destination_start,
                mapping.destination_start + mapping.length,
            ),
            destination,
        )
        np.testing.assert_array_equal(
            range(mapping.source_start, mapping.source_start + mapping.length),
            source,
        )


def test_batch_preserves_encoder_offsets_across_chunks():
    items = _items([(1, 4, 4), (1, 4, 4)], [(2, 6), (6, 10)])
    tasks = _batch(items, prefix=4, extend=4)[Modality.IMAGE]
    assert tasks[0].merge_mappings == (_MergeMapping(2, 0, 2),)
    assert tasks[1].merge_mappings == (_MergeMapping(0, 2, 2),)


def test_batch_uses_global_token_indices_for_dp_ranks():
    rank0 = _req(_items([(1, 2, 4), (1, 4, 4)], [(0, 2), (3, 7)]), 8)
    rank1 = _req(
        _items([(1, 2, 4), (1, 2, 4), (1, 4, 4)], [(1, 3), (4, 6), (7, 11)]),
        12,
    )
    batch = build_multimodal_batch(
        [ScheduleReqsInfo(reqs=[rank0]), ScheduleReqsInfo(reqs=[rank1])],
        2,
        _model_config(),
        12,
    )
    tasks = batch[Modality.IMAGE]
    assert [task.owner_dp for task in tasks] == [0, 0, 1, 1, 1]
    destinations = [
        [
            token
            for mapping in task.merge_mappings
            for token in range(
                mapping.destination_start,
                mapping.destination_start + mapping.length,
            )
        ]
        for task in tasks
    ]
    assert destinations == [[0, 1], [3, 4, 5, 6], [13, 14], [16, 17], [19, 20, 21, 22]]


def test_batch_routes_video_modality():
    video = _items([(1, 2, 4)], [(0, 2)], Modality.VIDEO)
    batch = _batch(video)
    assert tuple(batch) == (Modality.VIDEO,)
    assert batch[Modality.VIDEO][0].item is video[0]


@pytest.mark.parametrize(
    ("max_segment_len", "alignment"),
    [(None, 2048), (64, 256)],
)
def test_long_vision_attention_pads_to_tuned_alignment(max_segment_len, alignment):
    class _CaptureBackend:
        def __init__(self):
            self.max_segment_len = max_segment_len
            self.shape = None

        def __call__(self, q, k, v, segment_ids):
            del k, v, segment_ids
            self.shape = q.shape
            return q

    sequence_length = 64 * 1024 + 1
    shape = (1, sequence_length, 1, 1)
    q = jnp.ones(shape, dtype=jnp.bfloat16)
    backend = _CaptureBackend()

    output = _vision_attention(
        backend,
        q,
        q,
        q,
        jnp.zeros((1, sequence_length), dtype=jnp.int32),
    )

    expected_length = ((sequence_length + alignment - 1) // alignment) * alignment
    assert backend.shape == (1, 1, expected_length, 1)
    assert output.shape == shape


def test_qwen_window_blocks_enable_bounded_segment_grid():
    config = _vision_config(
        depth=2,
        fullatt_block_indexes=[1],
        patch_size=14,
        spatial_merge_size=2,
        window_size=112,
    )
    with jax.set_mesh(_mesh()):
        visual = Qwen2_5_VisionTransformer(
            config,
            jnp.bfloat16,
            mesh=_mesh(),
            norm_eps=1e-6,
        )

    assert visual.blocks[0].attn.attn_backend.max_segment_len == 64
    assert not visual.blocks[0].attn.attn_backend.block_sparse_segments
    assert visual.blocks[1].attn.attn_backend.max_segment_len is None
    assert visual.blocks[1].attn.attn_backend.block_sparse_segments


@pytest.mark.skipif(
    "TPU" not in jax.devices()[0].device_kind or jax.device_count() < 4,
    reason="Requires a four-device TPU mesh.",
)
def test_block_sparse_vision_backend_tpu_integration():
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(1, 4),
        ("data", "tensor"),
    )
    batch_size = 4
    seq_len = 64 * 1024
    qkv_shape = (batch_size, 1, seq_len, 80)
    qkv_sharding = NamedSharding(
        mesh,
        PartitionSpec(("data", "tensor"), None, None, None),
    )
    segment_sharding = NamedSharding(
        mesh,
        PartitionSpec(("data", "tensor"), None),
    )
    qkv = jax.device_put(jnp.ones(qkv_shape, dtype=jnp.bfloat16), qkv_sharding)
    segments = jax.device_put(
        jnp.broadcast_to(
            (jnp.arange(seq_len, dtype=jnp.int32) // (16 * 1024))[None, :],
            (batch_size, seq_len),
        ),
        segment_sharding,
    )
    backend = VisionFlashAttentionBackend(
        mesh,
        block_sparse_segments=True,
    )

    output = jax.jit(
        lambda qkv, segments: backend(
            qkv,
            qkv,
            qkv,
            SegmentIds(q=segments, kv=segments),
        )
    )(qkv, segments)

    assert output.shape == qkv_shape
    assert float(output[0, 0, 0, 0]) == 1.0


def test_vision_weight_tp_specs():
    mesh = _mesh(tp=4)
    config = _vision_config(
        hidden_size=8, out_hidden_size=8, intermediate_size=16, num_heads=4, depth=1
    )
    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config,
            jnp.float32,
            mesh=mesh,
            norm_eps=1e-6,
            vision_tp=True,
        )
    block = visual.blocks[0]
    assert block.attn.q_proj.weight.value.sharding.spec == PartitionSpec(None, "tensor")
    assert block.attn.proj.weight.value.sharding.spec == PartitionSpec("tensor", None)
    assert visual.merger.mlp_fc2.weight.value.sharding.spec == PartitionSpec("tensor", None)


def test_vision_rope_preserves_dtype():
    x = jnp.arange(8, dtype=jnp.bfloat16).reshape(1, 2, 1, 4)
    rotary = jnp.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=jnp.float32)
    result = _apply_rotary_pos_emb_vision(x, rotary)
    real, imag = x.astype(jnp.float32)[..., :2], x.astype(jnp.float32)[..., 2:]
    cos, sin = jnp.cos(rotary)[:, :, None, :], jnp.sin(rotary)[:, :, None, :]
    expected = jnp.concatenate([real * cos - imag * sin, real * sin + imag * cos], axis=-1)
    assert result.dtype == jnp.bfloat16
    np.testing.assert_array_equal(result, expected.astype(jnp.bfloat16))


def test_segment_ids_from_cumulative_lengths():
    cu = jnp.array([[2, 5, 8, 8], [4, 8, 8, 8]])
    np.testing.assert_array_equal(
        _segment_ids_from_cu_seqlens(cu, 8),
        [[0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 0, 0, 1, 1, 1, 1]],
    )


def test_merge_preserves_unmasked_tokens():
    item = MultimodalDataItem(
        Modality.AUDIO,
        feature=np.ones((2, 1)),
        placeholder_ranges=[(0, 1), (2, 3)],
    )
    batch = build_multimodal_batch(
        [ScheduleReqsInfo(reqs=[_req([item], 3)])],
        1,
        _model_config(),
        3,
    )

    class Model(_TestInModelModel):
        def get_multimodal_embedding_funcs(self):
            return {Modality.AUDIO: lambda _: jnp.array([[10, 11], [20, 21]])}

    running = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.float32)
    output, _ = embedding_pipeline.general_mm_embed_routine(
        batch,
        jnp.zeros(3, dtype=jnp.int32),
        lambda _: running,
        Model(),
    )
    np.testing.assert_array_equal(output, [[10, 11], [3, 4], [20, 21]])


def test_dp_local_merge_vectorizes_unsharded_dp_lanes():
    target = jnp.arange(1, 7, dtype=jnp.float32)[:, None]
    source = jnp.asarray(
        [
            [[10], [11], [12]],
            [[20], [21], [22]],
        ],
        dtype=jnp.float32,
    )
    src_idx = jnp.asarray([[1, 0, 2], [2, 0, 1]], dtype=jnp.int32)
    mask = jnp.asarray(
        [[True, False, True], [False, True, True]],
        dtype=jnp.bool_,
    )
    batch = embedding_pipeline.DPLocalMergeBatch(source, src_idx, mask)

    merge = jax.jit(lambda value: embedding_pipeline.dp_local_merge(value, batch))
    np.testing.assert_array_equal(merge(target)[:, 0], [11, 2, 12, 4, 20, 21])

    deepstack_target = jnp.stack((target, target + 100))
    deepstack_source = jnp.stack((source, source + 100))
    deepstack_batch = embedding_pipeline.DPLocalMergeBatch(
        source,
        src_idx,
        mask,
        deepstack_source,
    )
    merge_deepstack = jax.jit(
        lambda value: embedding_pipeline.dp_local_merge(
            value,
            deepstack_batch,
            deepstack=True,
        )
    )
    np.testing.assert_array_equal(
        merge_deepstack(deepstack_target)[:, :, 0],
        [
            [11, 2, 12, 4, 20, 21],
            [111, 102, 112, 104, 120, 121],
        ],
    )


def test_lower_to_dp_merge_batch_has_explicit_owner_local_ir():
    mesh = _mesh(dp=2, tp=2)
    resolved = tuple(
        embedding_pipeline.ResolvedItem(
            embedding_pipeline.ItemTask(
                MultimodalDataItem(Modality.AUDIO),
                rank,
                2,
                (_MergeMapping(0, 2 * rank, 2),),
            ),
            embedding_pipeline._ItemEmbedding(
                place_on_dp(jnp.asarray(values)[:, None], mesh, rank),
                place_on_dp(jnp.asarray(values)[None, :, None] + 20, mesh, rank),
            ),
        )
        for rank, values in enumerate(((10, 11), (20, 21)))
    )
    target = jax.device_put(
        jnp.zeros((4, 1)),
        NamedSharding(mesh, PartitionSpec("data", None)),
    )

    batch = embedding_pipeline.lower_to_dp_merge_batch(resolved, target, mesh)

    assert batch.source.shape == (2, 2, 1)
    assert batch.source.sharding.spec == PartitionSpec("data", None, None)
    assert batch.deepstack.shape == (1, 2, 2, 1)
    assert batch.deepstack.sharding.spec == PartitionSpec(None, "data", None, None)
    np.testing.assert_array_equal(batch.source[:, :, 0], [[10, 11], [20, 21]])
    np.testing.assert_array_equal(batch.src_idx, [[0, 1], [0, 1]])
    np.testing.assert_array_equal(batch.mask, True)


@pytest.mark.parametrize("item_aligned", [False, True])
def test_merge_preserves_data_sharding(item_aligned):
    mesh = _mesh(dp=2)
    rank0 = MultimodalDataItem(
        Modality.AUDIO,
        feature=np.ones((2, 1)),
        placeholder_ranges=[(0, 2)],
    )
    rank1 = MultimodalDataItem(
        Modality.AUDIO,
        feature=np.ones((2, 1)),
        placeholder_ranges=[(0, 2)],
    )
    batch = build_multimodal_batch(
        [
            ScheduleReqsInfo(reqs=[_req([rank0], 2)]),
            ScheduleReqsInfo(reqs=[_req([rank1], 2)]),
        ],
        2,
        _model_config(),
        2,
    )

    values = ([10.0, 11.0], [20.0, 21.0])
    deepstack_values = ([30.0, 31.0], [40.0, 41.0])

    class Model(_TestInModelModel):
        def get_multimodal_embedding_funcs(self):
            def encode(items):
                ranks = [0 if item is rank0 else 1 for item in items]
                encoded = [jnp.asarray(values[rank])[:, None] for rank in ranks]
                encoded_deepstack = [
                    jnp.asarray(deepstack_values[rank])[None, :, None] for rank in ranks
                ]
                return MultimodalEmbeddingOutput(
                    encoded if item_aligned else jnp.concatenate(encoded),
                    (
                        encoded_deepstack
                        if item_aligned
                        else jnp.concatenate(encoded_deepstack, axis=1)
                    ),
                )

            return {Modality.AUDIO: encode}

    Model.mesh = mesh
    running = jax.device_put(
        jnp.zeros((4, 1)),
        NamedSharding(mesh, PartitionSpec("data", None)),
    )
    output, deepstack = embedding_pipeline.general_mm_embed_routine(
        batch,
        jnp.zeros(4, dtype=jnp.int32),
        lambda _: running,
        Model(),
    )
    np.testing.assert_array_equal(output[:, 0], [10, 11, 20, 21])
    np.testing.assert_array_equal(deepstack[0, :, 0], [30, 31, 40, 41])
    assert output.sharding.spec == PartitionSpec("data", None)
    assert deepstack.sharding.spec == PartitionSpec(None, "data", None)


@pytest.mark.parametrize(
    ("arch", "chunked", "radix", "mixed_chunk"),
    [
        (ARCH, 4096, False, True),
        ("Qwen3VLForConditionalGeneration", 4096, False, False),
        ("UnsupportedVLM", -1, True, False),
    ],
)
def test_multimodal_defaults_follow_capabilities(arch, chunked, radix, mixed_chunk):
    args = SimpleNamespace(
        disable_radix_cache=False,
        disable_overlap_schedule=False,
        chunked_prefill_size=4096,
        enable_mixed_chunk=True,
        limit_mm_data_per_request=None,
    )
    apply_multimodal_model_defaults(args, _model_config(arch=arch))
    assert (args.chunked_prefill_size, args.disable_radix_cache) == (chunked, radix)
    assert args.disable_overlap_schedule is False
    assert args.enable_mixed_chunk is mixed_chunk
    assert args.limit_mm_data_per_request == {"image": 16}


def test_generate_request_preserves_media_fields():
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
    assert (item.image_data, item.video_data, item.audio_data, item.input_embeds) == (
        ["image1"],
        ["video1"],
        ["audio1"],
        ["emb1"],
    )


def test_forward_batch_shards_input_embeddings():
    batch = ModelWorkerBatch(
        bid=1,
        forward_mode=ForwardMode.EXTEND,
        input_ids=np.array([1]),
        real_input_ids_len=1,
        seq_lens=np.array([1]),
        out_cache_loc=np.array([1]),
        req_pool_indices=np.array([0]),
        sampling_info=None,
        positions=np.array([0]),
        cache_loc=np.array([1]),
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_seq_lens=np.array([1]),
        extend_prefix_lens=np.array([0]),
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        logits_indices=np.array([0]),
        real_bs=1,
        real_bs_per_dp=[1],
        input_embedding=np.ones((1, 4)),
    )
    runner = SimpleNamespace(
        mesh=Mesh(np.asarray(jax.devices()[:1]), ("data",)),
        attn_backend=None,
        model_config=SimpleNamespace(
            is_embedding=False,
            hf_config=SimpleNamespace(architectures=[]),
        ),
    )
    specs = []
    with patch(
        "sgl_jax.srt.model_executor.forward_batch_info.device_array",
        side_effect=lambda values, sharding: specs.append(sharding.spec) or values,
    ):
        ForwardBatch.init_new(batch, runner)
    assert PartitionSpec("data", None) in specs


def test_mrope_positions_reach_worker_batch():
    positions = np.array([[0, 10, 2], [0, 11, 2], [0, 12, 2]], dtype=np.int32)
    req = SimpleNamespace(mm_inputs={"mrope_positions": positions}, extend_input_len=3, lora_id="0")
    worker_batch = _schedule_batch(req).get_model_worker_batch(
        token_paddings=[3],
        bs_paddings=[1],
        cache_loc_paddings=[3],
        page_size=1,
    )
    np.testing.assert_array_equal(worker_batch.mrope_positions[:, :3], positions)


def test_overlap_copy_rebuilds_multimodal_batch_from_requests():
    items = _items([(1, 2, 4)], [(1, 3)])
    batch = _schedule_batch(_req(items, 3), _model_config())
    worker_batch = batch.get_model_worker_batch(
        token_paddings=[3],
        bs_paddings=[1],
        cache_loc_paddings=[3],
        page_size=1,
    )
    copied = batch.copy()
    rebuilt = build_multimodal_batch(copied.reqs_info, 1, _model_config(), 3)
    assert Modality.IMAGE in worker_batch.multimodal_batch
    assert getattr(copied, "multimodal_batch", None) is None
    assert Modality.IMAGE in rebuilt


def test_multimodal_item_reads_common_and_model_fields():
    item = MultimodalDataItem.from_dict(
        {
            "modality": "image",
            "feature": np.ones((2, 1)),
            "placeholder_ranges": [(1, 2)],
            "image_grid_thw": np.array([[1, 2, 4]]),
        }
    )
    assert item.is_image()
    assert item.placeholder_ranges == [(1, 2)]
    np.testing.assert_array_equal(item.get("image_grid_thw"), [[1, 2, 4]])
    assert item.get("missing", "fallback") == "fallback"
