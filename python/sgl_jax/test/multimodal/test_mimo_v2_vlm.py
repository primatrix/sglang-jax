import asyncio
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.configs.model_config import _adapt_mimo_v2_multimodal_architecture
from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.models import mimo_v2_mm
from sgl_jax.srt.models.mimo_v2_mm import (
    MiMoAudioEncoder,
    MiMoV2ForCausalLM,
    MiMoV2ForConditionalGeneration,
    MiMoVisionTransformer,
    _encode_first_key_attention_bias,
    _MiMoV2MultimodalMixin,
)
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalInputs
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract
from sgl_jax.srt.multimodal.layers.vision_sharding import VisionShardSpecs
from sgl_jax.srt.multimodal.processors.mimo_v2 import MiMoV2Processor
from sgl_jax.srt.utils.quantization.quantization_utils import apply_linear_quantization

IMAGE_TOKEN = 151655
AUDIO_TOKEN = 151669


def _hf_config(*, vision=True, audio=False):
    processor_config = SimpleNamespace(
        fps=1.0,
        video_min_pixels=8192,
        video_max_pixels=8388608,
        video_total_max_pixels=268435456,
        max_frames=3600,
        min_frames=None,
        audio_token_id=AUDIO_TOKEN,
    )
    return SimpleNamespace(
        architectures=["MiMoV2ForConditionalGeneration"],
        vision_config=(SimpleNamespace(patch_size=16, spatial_merge_size=2) if vision else None),
        audio_config=(
            SimpleNamespace(
                audio_channels=20,
                group_size=4,
                speech_vocab_size="1280",
                speech_zeroemb_idx="1024",
            )
            if audio
            else None
        ),
        processor_config=processor_config,
        image_token_id=IMAGE_TOKEN,
        image_max_pixels=8388608,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )


def _processor(*, vision=True, audio=False):
    return MiMoV2Processor(
        _hf_config(vision=vision, audio=audio),
        SimpleNamespace(
            model_path="unused",
            precompile_vision_patch_paddings=[256, 1024],
        ),
        object(),
    )


def _tiny_mimo_model(monkeypatch, *, audio=False):
    from flax import nnx

    def fake_text_init(self, config, mesh, dtype):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.model = SimpleNamespace(embed_tokens=lambda values: values)

    monkeypatch.setattr(MiMoV2ForCausalLM, "__init__", fake_text_init)
    from sgl_jax.srt.managers.schedule_batch import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "vision_encoder_parallel", "dp")
    monkeypatch.setitem(global_server_args_dict, "precompile_vision_patch_paddings", [4])
    config = _hf_config(audio=audio)
    config.vision_config = SimpleNamespace(
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        in_chans=3,
        hidden_size=8,
        intermediate_size=16,
        out_hidden_size=8,
        num_heads=2,
        num_key_value_heads=1,
        qk_channels=4,
        depth=1,
        fullatt_block_indexes=[0],
        hidden_act="silu",
        use_sink=True,
        visual_token_window_size=4,
        vit_window_attn_types=[-1],
    )
    if audio:
        config.audio_config = SimpleNamespace(
            audio_channels=2,
            group_size=2,
            input_local_dim=4,
            input_local_intermediate_size=8,
            input_local_layers=1,
            input_local_attn_heads=1,
            input_local_head_dim=4,
            input_full_attention=True,
            out_hidden_size=8,
            projection_layers=2,
            speech_vocab_size=16,
            speech_zeroemb_idx=15,
            add_post_norm=True,
            partial_rotary_factor=1.0,
            rope_theta=10000,
        )
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(
        devices,
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(
            lambda: MiMoV2ForConditionalGeneration(config, mesh=mesh, dtype=jnp.bfloat16)
        )
    return model, mesh


def test_mimo_v25_native_architecture_routes_only_multimodal_targets():
    multimodal = SimpleNamespace(
        architectures=["MiMoV2ForCausalLM"],
        vision_config={},
        audio_config={},
    )
    _adapt_mimo_v2_multimodal_architecture(multimodal, is_draft_model=False)
    assert multimodal.architectures == ["MiMoV2ForConditionalGeneration"]

    text_only = SimpleNamespace(
        architectures=["MiMoV2ForCausalLM"],
        vision_config=None,
        audio_config=None,
    )
    _adapt_mimo_v2_multimodal_architecture(text_only, is_draft_model=False)
    assert text_only.architectures == ["MiMoV2ForCausalLM"]

    draft = SimpleNamespace(
        architectures=["MiMoV2ForCausalLM"],
        vision_config={},
        audio_config={},
    )
    _adapt_mimo_v2_multimodal_architecture(draft, is_draft_model=True)
    assert draft.architectures == ["MiMoV2ForCausalLM"]
    assert issubclass(MiMoV2ForConditionalGeneration, InModelMultimodalContract)


def test_mimo_v25_constructs_visual_tower_under_nnx_eval_shape(monkeypatch):
    model, _ = _tiny_mimo_model(monkeypatch)
    assert model.visual.input_buckets == (4,)
    metadata = model.visual._metadata_for_grid((1, 4, 4))
    np.testing.assert_array_equal(
        metadata.col_index[metadata.reverse_col_index],
        np.arange(metadata.col_index.size),
    )


def test_mimo_v25_restores_and_loads_towers_outside_text_graph(monkeypatch):
    model, mesh = _tiny_mimo_model(monkeypatch, audio=True)
    quantization_config = QuantizationConfig(
        is_static_checkpoint=True,
        linear_rules=[
            {
                "module_path": ".*",
                "weight_dtype": "float8_e4m3fn",
                "activation_dtype": None,
            }
        ],
    )
    quant_model_config = SimpleNamespace(quantization_config=quantization_config)
    model = apply_linear_quantization(quant_model_config, model, is_static_input=True)
    assert not isinstance(model.visual.merger.mlp_fc1, LinearBase)
    assert not isinstance(model.audio_encoder.proj_fc1, LinearBase)

    text_loads = []

    def fake_text_load(self, model_config):
        assert not hasattr(self, "visual")
        assert not hasattr(self, "audio_encoder")
        text_loads.append(model_config)

    monkeypatch.setattr(MiMoV2ForCausalLM, "load_weights", fake_text_load)

    tower_loads = []

    class FakeWeightLoader:
        def __init__(self, tower, model_config, loader_mesh, dtype):
            assert loader_mesh is mesh
            tower_loads.append(tower)

        def load_weights_from_safetensors(self, mappings):
            assert all(
                not target.startswith(("visual.", "audio_encoder."))
                for mapping in mappings.values()
                for target in (
                    mapping.target_path
                    if isinstance(mapping.target_path, list)
                    else [mapping.target_path]
                )
            )

    monkeypatch.setattr(mimo_v2_mm, "WeightLoader", FakeWeightLoader)
    model_config = SimpleNamespace(model_path="unused", _dummy_mode=True)
    model.load_weights(model_config)

    assert text_loads == [model_config]
    assert len(tower_loads) == 2
    assert isinstance(tower_loads[0], MiMoVisionTransformer)
    assert isinstance(tower_loads[1], MiMoAudioEncoder)
    assert isinstance(model.visual.merger.mlp_fc1, LinearBase)
    assert isinstance(model.audio_encoder.proj_fc1, LinearBase)


def test_first_key_sink_bias_matches_official_vision_logits():
    q = jnp.arange(1 * 5 * 4 * 2, dtype=jnp.float32).reshape(1, 5, 4, 2) / 10
    k = jnp.arange(1 * 5 * 2 * 2, dtype=jnp.float32).reshape(1, 5, 2, 2) / 7
    v = jnp.ones_like(k)
    cu_seqlens = jnp.asarray([[0, 2, 5, 5]], dtype=jnp.int32)
    sinks = jnp.asarray([0.1, -0.2, 0.3, -0.4], dtype=jnp.float32)
    scale = 0.25

    q_aug, k_aug, v_aug = _encode_first_key_attention_bias(q, k, v, cu_seqlens, sinks, scale)
    logits = jnp.einsum("bthd,bshd->bhts", q_aug, k_aug) * scale

    repeated_k = jnp.repeat(k, 2, axis=2)
    expected = jnp.einsum("bthd,bshd->bhts", q, repeated_k) * scale
    expected = expected.at[:, :, :, 0].add(sinks[None, :, None])
    expected = expected.at[:, :, :, 2].add(sinks[None, :, None])
    np.testing.assert_allclose(logits, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(v_aug[..., :-1], jnp.repeat(v, 2, axis=2))
    np.testing.assert_array_equal(v_aug[..., -1], 0)


def test_first_key_sink_bias_preserves_explicit_attention_sharding():
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(
        devices,
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    attention_sharding = NamedSharding(mesh, PartitionSpec("data", None, "tensor", None))
    sequence_sharding = NamedSharding(mesh, PartitionSpec("data", None))
    sink_sharding = NamedSharding(mesh, PartitionSpec("tensor"))

    with jax.set_mesh(mesh):
        q = jax.device_put(jnp.ones((1, 5, 4, 2)), attention_sharding)
        k = jax.device_put(jnp.ones((1, 5, 2, 2)), attention_sharding)
        v = jax.device_put(jnp.ones((1, 5, 2, 2)), attention_sharding)
        cu_seqlens = jax.device_put(jnp.asarray([[0, 2, 5, 5]], dtype=jnp.int32), sequence_sharding)
        sinks = jax.device_put(jnp.ones((4,)), sink_sharding)
        q_aug, k_aug, v_aug = jax.jit(
            lambda q, k, v, cu_seqlens, sinks: _encode_first_key_attention_bias(
                q,
                k,
                v,
                cu_seqlens,
                sinks,
                0.25,
                out_sharding=attention_sharding,
            )
        )(q, k, v, cu_seqlens, sinks)

    assert q_aug.sharding == attention_sharding
    assert k_aug.sharding == attention_sharding
    assert v_aug.sharding == attention_sharding


def test_multimodal_tower_weight_mappings_follow_official_names():
    class MappingHarness:
        _linear_mappings = staticmethod(_MiMoV2MultimodalMixin._linear_mappings)
        _vision_weight_mappings = _MiMoV2MultimodalMixin._vision_weight_mappings
        _audio_weight_mappings = _MiMoV2MultimodalMixin._audio_weight_mappings

    specs = SimpleNamespace(
        col_kernel_axes=(None, None),
        row_kernel_axes=(None, None),
        tensor_axis=None,
    )
    model = MappingHarness()
    model.visual = SimpleNamespace(
        specs=specs,
        blocks=[SimpleNamespace(attn=SimpleNamespace(sinks=object()))],
    )
    model.audio_encoder = SimpleNamespace(
        channels=20,
        transformer=SimpleNamespace(norm=object(), layers=[object()]),
        proj_fc2=object(),
    )

    vision = model._vision_weight_mappings()
    assert {
        "visual.patch_embed.proj.weight",
        "visual.blocks.0.attn.qkv.weight",
        "visual.blocks.0.attn.qkv.bias",
        "visual.blocks.0.attn.sinks",
        "visual.merger.mlp.2.weight",
    } <= vision.keys()
    assert "visual.merger.ln_q.bias" not in vision
    assert "visual.merger.mlp.0.bias" not in vision
    assert "visual.merger.mlp.2.bias" not in vision
    assert vision["visual.blocks.0.attn.qkv.weight"].target_path == [
        "blocks.0.attn.q_proj.weight",
        "blocks.0.attn.k_proj.weight",
        "blocks.0.attn.v_proj.weight",
    ]
    assert vision["visual.merger.mlp.0.weight"].target_path == "merger.mlp_fc1.weight"

    audio = model._audio_weight_mappings()
    assert {f"speech_embeddings.{index}.weight" for index in range(20)} <= audio.keys()
    assert {
        "audio_encoder.input_local_transformer.layers.0.self_attn.q_proj.weight",
        "audio_encoder.input_local_transformer.layers.0.self_attn.q_proj.bias",
        "audio_encoder.input_local_transformer.layers.0.mlp.down_proj.weight",
        "audio_encoder.input_local_transformer.norm.weight",
        "audio_encoder.projection.mlp.0.weight",
        "audio_encoder.projection.mlp.2.weight",
    } <= audio.keys()
    assert audio["speech_embeddings.0.weight"].target_path == "speech_embeddings.0.embedding"
    assert (
        audio["audio_encoder.input_local_transformer.layers.0.self_attn.q_proj.weight"].target_path
        == "transformer.layers.0.self_attn.q_proj.weight"
    )


def test_processor_loads_vision_inputs_and_keeps_standard_rope():
    processor = _processor()

    async def run_processor():
        async def load_images(sources):
            assert sources == ["image-source"]
            return ["loaded-image"]

        async def load_videos(sources, video_config):
            assert sources == ["video-source"]
            assert video_config["fps"] == 1.0
            assert video_config["factor"] == 32
            return ["loaded-video"]

        async def combine(input_text, images=None, videos=None, **kwargs):
            assert input_text == "prompt"
            assert images == ["loaded-image"]
            assert videos == ["loaded-video"]
            assert kwargs["images_kwargs"] == {"max_pixels": 1024 * 16**2}
            assert kwargs["videos_kwargs"]["do_sample_frames"] is False
            return MultimodalInputs(mm_items=[], input_ids=[10, 11])

        processor.load_images_async = load_images
        processor._load_videos_async = load_videos
        processor.process_and_combine_mm_data_async = combine
        return await processor.process_mm_data_async(
            "image-source",
            "prompt",
            SimpleNamespace(audio_data=None, video_data="video-source"),
        )

    try:
        output = asyncio.run(run_processor())
    finally:
        processor.shutdown()
    assert output.input_ids == [10, 11]


def test_processor_collects_vision_without_language_mrope():
    processor = _processor()
    try:
        output = processor.collect_mm_items_from_processor_output(
            {
                "input_ids": np.asarray([[1, IMAGE_TOKEN, IMAGE_TOKEN, 2]]),
                "pixel_values": np.ones((8, 6), dtype=np.float32),
                "image_grid_thw": np.asarray([[1, 2, 4]], dtype=np.int32),
            },
            images=[object()],
        )
    finally:
        processor.shutdown()

    assert output.input_ids == [1, IMAGE_TOKEN, IMAGE_TOKEN, 2]
    assert output.mrope_positions is None
    assert output.mrope_position_delta is None
    assert len(output.mm_items) == 1
    assert output.mm_items[0].modality is Modality.IMAGE
    assert output.mm_items[0].placeholder_ranges == [(1, 3)]


def test_audio_placeholder_expansion_and_lane_valid_lengths():
    processor = _processor(vision=False, audio=True)
    output = MultimodalInputs(mm_items=[], input_ids=[1, AUDIO_TOKEN, 2])
    codes = np.arange(5 * 20, dtype=np.int32).reshape(5, 20) % 100
    try:
        processor._merge_audio(output, [codes])
    finally:
        processor.shutdown()

    assert output.input_ids == [1, AUDIO_TOKEN, AUDIO_TOKEN, 2]
    assert output.audio_token_id == AUDIO_TOKEN
    assert output.mm_items[0].placeholder_ranges == [(1, 3)]
    assert output.mm_items[0].feature.shape == (8, 20)
    np.testing.assert_array_equal(output.mm_items[0].feature[5:], np.repeat(codes[-1:], 3, axis=0))

    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(
        devices,
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )

    class FakeAudioEncoder:
        encoder_tp = False
        input_buckets = (8,)
        group_size = 4

        def __init__(self):
            self.specs = VisionShardSpecs(mesh, self.encoder_tp)
            self.valid = None

        def encode(self, packed_codes, valid):
            self.valid = np.asarray(valid)
            return jnp.zeros((1, 2, 3), dtype=jnp.float32)

    encoder = FakeAudioEncoder()
    model = SimpleNamespace(mesh=mesh, audio_encoder=encoder)
    features = _MiMoV2MultimodalMixin.get_audio_feature(model, output.mm_items)
    np.testing.assert_array_equal(encoder.valid, [8])
    assert features.shape == (2, 3)

    model.visual = None
    model.get_audio_feature = lambda items: items
    encode_funcs = _MiMoV2MultimodalMixin.get_multimodal_encode_funcs(model)
    assert set(encode_funcs) == {Modality.AUDIO}
