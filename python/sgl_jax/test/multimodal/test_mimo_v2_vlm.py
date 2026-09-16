import asyncio
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from sgl_jax.srt.models import mimo_v2_mm
from sgl_jax.srt.models.registry import ModelRegistry
from sgl_jax.srt.models.mimo_v2_mm import (
    MiMoV2ForCausalLM,
    MiMoV2ForConditionalGeneration,
)
from sgl_jax.srt.models.mimo_v2_vision import (
    MiMoVisionTransformer,
    _encode_first_key_attention_bias,
)
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
)
from sgl_jax.srt.multimodal.processors.mimo_v2 import MiMoV2Processor

IMAGE_TOKEN = 151655


def _hf_config():
    processor_config = SimpleNamespace(
        fps=1.0,
        video_min_pixels=8192,
        video_max_pixels=8388608,
        video_total_max_pixels=268435456,
        max_frames=3600,
        min_frames=None,
        audio_token_id=151669,
    )
    return SimpleNamespace(
        architectures=["MiMoV2ForConditionalGeneration"],
        vision_config=SimpleNamespace(patch_size=16, spatial_merge_size=2),
        audio_config=None,
        processor_config=processor_config,
        image_token_id=IMAGE_TOKEN,
        image_max_pixels=8388608,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )


def _tiny_mimo_model(
    monkeypatch,
    *,
    materialize=False,
    windowed=False,
):
    from flax import nnx

    def fake_text_init(self, config, mesh, dtype):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.model = SimpleNamespace(embed_tokens=lambda values: values)

    monkeypatch.setattr(MiMoV2ForCausalLM, "__init__", fake_text_init)
    config = _hf_config()
    config.vision_encoder_parallel = "dp"
    config.precompile_vision_patch_paddings = [4]
    config.vision_config = SimpleNamespace(
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        in_chans=3,
        hidden_size=8,
        intermediate_size=16,
        out_hidden_size=8,
        num_heads=4,
        num_key_value_heads=2,
        qk_channels=4,
        depth=3 if windowed else 1,
        fullatt_block_indexes=[0, 2] if windowed else [0],
        hidden_act="silu",
        use_sink=True,
        visual_token_window_size=4,
        vit_window_attn_types=[-1, 1, -1] if windowed else [-1],
    )
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(
        devices,
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    with jax.set_mesh(mesh):

        def build_model():
            return MiMoV2ForConditionalGeneration(config, mesh=mesh, dtype=jnp.bfloat16)

        model = build_model() if materialize else nnx.eval_shape(build_model)
    return model, mesh


@pytest.mark.parametrize(
    "architecture, vision, audio, expected",
    [
        ("MiMoV2ForCausalLM", None, None, "MiMoV2ForCausalLM"),
        ("MiMoV2ForCausalLM", {}, None, "MiMoV2ForConditionalGeneration"),
        ("MiMoV2ForCausalLM", None, {}, "MiMoV2ForConditionalGeneration"),
        ("MiMoV2ForConditionalGeneration", {}, {}, "MiMoV2ForConditionalGeneration"),
        ("MiMoV2MTPForCausalLM", {}, {}, "MiMoV2MTPForCausalLM"),
    ],
)
def test_mimo_resolution_preserves_checkpoint_architecture(architecture, vision, audio, expected):
    config = SimpleNamespace(
        architectures=[architecture], vision_config=vision, audio_config=audio
    )
    model_cls, arch = ModelRegistry.resolve_model_cls(config.architectures, hf_config=config)
    assert model_cls.__name__ == expected
    assert arch == architecture
    assert config.architectures == [architecture]
    assert ModelRegistry.is_in_model_multimodal(
        config.architectures, hf_config=config
    ) == (expected == "MiMoV2ForConditionalGeneration")
    assert "MiMoV2ForCausalLM" in MiMoV2Processor.models


def test_mimo_v25_constructs_visual_tower_under_nnx_eval_shape(monkeypatch):
    model, _ = _tiny_mimo_model(monkeypatch)
    assert model.visual.input_buckets == (4,)
    metadata = model.visual._metadata_for_grid((1, 4, 4))
    np.testing.assert_array_equal(
        metadata["col_index"][metadata["reverse_col_index"]],
        np.arange(metadata["col_index"].size),
    )


def test_mimo_v25_loads_visual_tower_outside_text_graph(monkeypatch):
    model, mesh = _tiny_mimo_model(monkeypatch)
    visual = model.visual

    text_loads = []

    def fake_text_load(self, model_config):
        assert not hasattr(self, "visual")
        text_loads.append(model_config)

    monkeypatch.setattr(MiMoV2ForCausalLM, "load_weights", fake_text_load)

    tower_loads = []

    class FakeWeightLoader:
        def __init__(self, tower, model_config, loader_mesh, dtype):
            assert loader_mesh is mesh
            tower_loads.append(tower)

        def load_weights_from_safetensors(self, mappings):
            assert all(
                not target.startswith("visual.")
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
    assert len(tower_loads) == 1
    assert isinstance(tower_loads[0], MiMoVisionTransformer)
    assert model.visual is visual


def test_first_key_sink_bias_matches_official_vision_logits():
    q = jnp.arange(1 * 5 * 4 * 2, dtype=jnp.float32).reshape(5, 4, 2) / 10
    k = jnp.arange(1 * 5 * 2 * 2, dtype=jnp.float32).reshape(5, 2, 2) / 7
    v = jnp.ones_like(k)
    first_key_mask = jnp.asarray([True, False, True, False, False])
    sinks = jnp.asarray([0.1, -0.2, 0.3, -0.4], dtype=jnp.float32)
    scale = 0.25

    q_aug, k_aug, v_aug = _encode_first_key_attention_bias(q, k, v, first_key_mask, sinks, scale)
    logits = jnp.einsum("thd,shd->hts", q_aug, k_aug) * scale

    repeated_k = jnp.repeat(k, 2, axis=1)
    expected = jnp.einsum("thd,shd->hts", q, repeated_k) * scale
    expected = expected.at[:, :, 0].add(sinks[:, None])
    expected = expected.at[:, :, 2].add(sinks[:, None])
    np.testing.assert_allclose(logits, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(v_aug[..., :-1], jnp.repeat(v, 2, axis=1))
    np.testing.assert_array_equal(v_aug[..., -1], 0)


def test_vision_weight_mappings_follow_official_names():
    class MappingHarness:
        _linear_mappings = staticmethod(MiMoV2ForConditionalGeneration._linear_mappings)
        _vision_weight_mappings = MiMoV2ForConditionalGeneration._vision_weight_mappings

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


def test_processor_loads_vision_inputs_and_keeps_standard_rope(monkeypatch):
    from sgl_jax.srt.multimodal.processors import mimo_v2

    class FakeHFProcessor:
        def __call__(self, **kwargs):
            assert kwargs["text"] == ["prompt"]
            assert kwargs["images"] == ["loaded-image"]
            assert kwargs["videos"] == ["loaded-video"]
            assert kwargs["images_kwargs"] == {"max_pixels": 1024 * 16**2}
            assert kwargs["videos_kwargs"]["do_sample_frames"] is False
            return {
                "input_ids": np.asarray([[1, IMAGE_TOKEN, IMAGE_TOKEN, 151656, 2]]),
                "pixel_values": np.ones((8, 6), dtype=np.float32),
                "image_grid_thw": np.asarray([[1, 2, 4]], dtype=np.int32),
                "pixel_values_videos": np.ones((4, 6), dtype=np.float32),
                "video_grid_thw": np.asarray([[1, 2, 2]], dtype=np.int32),
            }

    processor = MiMoV2Processor(
        _hf_config(),
        SimpleNamespace(model_path="unused", precompile_vision_patch_paddings=[256, 1024]),
        FakeHFProcessor(),
    )

    def load_image(source):
        assert source == "image-source"
        return "loaded-image"

    def load_video(source, video_config):
        assert source == "video-source"
        assert video_config["fps"] == 1.0
        assert video_config["factor"] == 32
        return "loaded-video"

    monkeypatch.setattr(processor, "load_image", load_image)
    monkeypatch.setattr(mimo_v2, "preprocess_video", load_video)
    try:
        output = asyncio.run(
            processor.process_mm_data_async(
                "image-source",
                "prompt",
                SimpleNamespace(audio_data=None, video_data="video-source"),
            )
        )
    finally:
        processor.shutdown()
    assert output.input_ids == [1, IMAGE_TOKEN, IMAGE_TOKEN, 151656, 2]
    assert [item.modality for item in output.mm_items] == [Modality.IMAGE, Modality.VIDEO]
    assert output.mrope_positions is None
    assert output.mrope_position_delta is None


def test_vision_flat_api_matches_individual_images(monkeypatch):
    model, mesh = _tiny_mimo_model(
        monkeypatch,
        materialize=True,
        windowed=True,
    )
    visual = model.visual
    visual.input_buckets = (32,)
    rng = np.random.default_rng(42)
    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=rng.normal(size=(length, visual.patch_dim)).astype(np.float32),
            placeholder_ranges=[(0, length // 4)],
            model_specific_data={"image_grid_thw": np.asarray(grid)},
        )
        for length, grid in ((4, (1, 2, 2)), (24, (1, 4, 6)))
    ]

    expected = np.concatenate(
        [np.asarray(model.get_image_feature([item]))[: len(item.feature) // 4] for item in items]
    )
    actual = np.asarray(model.get_image_feature(items))
    np.testing.assert_allclose(actual[:7], expected, rtol=0.02, atol=0.02)
    np.testing.assert_array_equal(actual[7:], 0)
    np.testing.assert_array_equal(model.get_video_feature(items), actual)
    visual.precompile()
