import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AxisType, Mesh
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.managers.schedule_batch import ScheduleReqsInfo
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.models.mimo_v2_flash import MiMoV2Model
from sgl_jax.srt.models.mimo_v2_mm import (
    MiMoAudioEncoder,
    MiMoV2FlashForConditionalGeneration,
    MiMoV2ForConditionalGeneration,
    MiMoVisionTransformer,
)
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.in_model.encoders.mimo_v2 import MiMoV2PlanBuilder
from sgl_jax.srt.multimodal.processors.mimo_v2 import MiMoV2Processor


def _vision_config(**overrides):
    values = {
        "patch_size": 1,
        "temporal_patch_size": 1,
        "in_channels": 1,
        "hidden_size": 4,
        "depth": 1,
        "intermediate_size": 8,
        "hidden_act": "silu",
        "num_heads": 1,
        "num_key_value_heads": 1,
        "qk_channels": 4,
        "out_hidden_size": 4,
        "spatial_merge_size": 2,
        "fullatt_block_indexes": [0],
        "vit_window_attn_types": [-1],
        "visual_token_window_size": -1,
        "rope_theta": 10000.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _audio_config(**overrides):
    values = {
        "audio_channels": 2,
        "group_size": 4,
        "speech_vocab_size": "8-9",
        "speech_zeroemb_idx": "0-0",
        "input_local_dim": 4,
        "input_local_layers": 1,
        "input_local_attn_heads": 1,
        "input_local_head_dim": 4,
        "input_local_intermediate_size": 8,
        "input_full_attention": True,
        "out_hidden_size": 4,
        "projection_layers": 1,
        "partial_rotary_factor": 1.0,
        "rope_theta": 10000.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _model_config():
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["MiMoV2ForConditionalGeneration"],
            vision_config=_vision_config(),
            audio_config=_audio_config(),
        )
    )


def _mesh():
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _item(modality, feature, placeholder_range, **metadata):
    return MultimodalDataItem(
        modality=modality,
        feature=np.asarray(feature),
        placeholder_ranges=[placeholder_range],
        model_specific_data=metadata,
    )


def test_composite_plan_packs_vision_and_audio_codes():
    image = _item(
        Modality.IMAGE,
        np.arange(16, dtype=np.float32).reshape(16, 1),
        (0, 4),
        image_grid_thw=np.asarray([[1, 4, 4]]),
    )
    audio = _item(
        Modality.AUDIO,
        np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        (4, 5),
    )
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=[image, audio]),
        extend_input_len=5,
    )
    builder = MiMoV2PlanBuilder(_model_config())
    plan = builder.build(
        [ScheduleReqsInfo(reqs=[req], prefix_lens=[0], extend_lens=[5])],
        dp_size=1,
        per_dp_token=5,
        tp_size=1,
    )

    assert tuple(plan) == (Modality.IMAGE, Modality.AUDIO)
    vision = plan[Modality.IMAGE]
    np.testing.assert_array_equal(vision.encode_inputs.valid, [[16]])
    np.testing.assert_array_equal(vision.encode_inputs.meta.col_index[0, 0], [0, 2, 1, 3])
    np.testing.assert_array_equal(np.flatnonzero(vision.merge.mask[0, 0]), [0, 1, 2, 3])

    audio_batch = plan[Modality.AUDIO]
    np.testing.assert_array_equal(audio_batch.encode_inputs.valid, [[4]])
    np.testing.assert_array_equal(
        audio_batch.encode_inputs.features[0, 0, :4],
        [[1, 2], [3, 4], [5, 6], [5, 6]],
    )
    np.testing.assert_array_equal(np.flatnonzero(audio_batch.merge.mask[0, 0]), [4])
    assert builder.get_num_output_tokens(16) == 4


def test_video_uses_shared_vision_tower():
    video = _item(
        Modality.VIDEO,
        np.ones((8, 1), dtype=np.float32),
        (0, 2),
        video_grid_thw=np.asarray([[2, 2, 2]]),
    )
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=[video]),
        extend_input_len=2,
    )
    plan = MiMoV2PlanBuilder(_model_config()).build(
        [ScheduleReqsInfo(reqs=[req], prefix_lens=[0], extend_lens=[2])],
        dp_size=1,
        per_dp_token=2,
        tp_size=1,
    )
    assert tuple(plan) == (Modality.IMAGE,)
    np.testing.assert_array_equal(plan[Modality.IMAGE].encode_inputs.valid, [[8]])


def test_tiny_vision_and_audio_towers_run_on_fixed_shapes():
    mesh = _mesh()
    with jax.set_mesh(mesh):
        vision = MiMoVisionTransformer(
            _vision_config(),
            jnp.float32,
            nnx.Rngs(0),
            mesh,
            False,
        )
        vision_meta = SimpleNamespace(
            col_index=jnp.asarray([[0, 2, 1, 3]], dtype=jnp.int32),
            rotary_freqs=jnp.zeros((1, 16, 4), dtype=jnp.float32),
            segment_ids=jnp.zeros((1, 16), dtype=jnp.int32),
        )
        vision_output = vision(
            jnp.ones((1, 16, 1), dtype=jnp.float32),
            vision_meta,
            jnp.asarray([16], dtype=jnp.int32),
        )

        audio = MiMoAudioEncoder(_audio_config(), jnp.float32, mesh, False)
        audio_output = audio(
            jnp.asarray([[[1, 2], [3, 4], [5, 6], [5, 6]]]),
            jnp.asarray([4], dtype=jnp.int32),
        )

    assert vision_output.shape == (1, 4, 4)
    assert audio_output.shape == (1, 1, 4)
    assert np.isfinite(np.asarray(vision_output)).all()
    assert np.isfinite(np.asarray(audio_output)).all()


def test_audio_placeholder_expansion_refreshes_vision_ranges():
    processor = MiMoV2Processor.__new__(MiMoV2Processor)
    processor.audio_token_id = 99
    processor.group_size = 4
    processor.hf_config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    image = _item(
        Modality.IMAGE,
        np.ones((4, 1), dtype=np.float32),
        (1, 2),
        image_grid_thw=np.asarray([[1, 2, 2]]),
    )
    output = MultimodalInputs(
        mm_items=[image],
        input_ids=[99, 10],
        im_token_id=10,
    )

    processor._merge_audio(output, [np.ones((5, 2), dtype=np.int32)])

    assert output.input_ids == [99, 99, 10]
    assert image.placeholder_ranges == [(2, 3)]
    audio = output.mm_items[-1]
    assert audio.modality is Modality.AUDIO
    assert audio.feature.shape == (8, 2)
    assert audio.placeholder_ranges == [(0, 2)]


def test_backbone_consumes_materialized_embeddings_for_extend():
    mesh = _mesh()
    config = SimpleNamespace(
        vocab_size=8,
        hidden_size=4,
        num_hidden_layers=0,
        layernorm_epsilon=1e-6,
    )
    with jax.set_mesh(mesh):
        model = MiMoV2Model(config, mesh, dtype=jnp.float32)
        embeddings = jnp.asarray(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            dtype=jnp.float32,
        )
        output, _, _ = model(
            SimpleNamespace(
                input_ids=jnp.asarray([0, 1], dtype=jnp.int32),
                input_embedding=embeddings,
                forward_mode=ForwardMode.EXTEND,
            ),
            None,
        )
        expected = model.norm(embeddings)

    np.testing.assert_allclose(output, expected)


@pytest.mark.parametrize(
    "model_type",
    [MiMoV2ForConditionalGeneration, MiMoV2FlashForConditionalGeneration],
)
def test_multimodal_wrappers_compose_with_both_backbones(model_type):
    mesh = _mesh()
    config = SimpleNamespace(
        vocab_size=8,
        hidden_size=4,
        num_hidden_layers=0,
        layernorm_epsilon=1e-6,
        tie_word_embeddings=True,
        vision_config=_vision_config(depth=0, fullatt_block_indexes=[], vit_window_attn_types=[]),
        audio_config=_audio_config(input_local_layers=0),
    )
    with jax.set_mesh(mesh):
        model = model_type(config, mesh=mesh, dtype=jnp.float32)

    assert model.get_multimodal_encoder(Modality.IMAGE) == model.visual.encode
    assert model.get_multimodal_encoder(Modality.AUDIO) == model.audio_encoder.encode
    mappings = model._tower_weight_mappings()
    assert "visual.merger.ln_q.bias" in mappings
    assert "audio_encoder.projection.weight" in mappings


def test_processor_normalizes_dict_vision_config():
    config = SimpleNamespace(
        vision_config={"spatial_merge_size": 2},
        audio_config=_audio_config(),
        processor_config={},
    )
    processor = MiMoV2Processor(
        config,
        SimpleNamespace(model_path="."),
        SimpleNamespace(),
    )
    assert processor.hf_config.vision_config.spatial_merge_size == 2
    assert processor.uses_mrope is False
    assert isinstance(config.vision_config, dict)


def test_processor_skips_mrope_for_text_only_vision_request():
    config = SimpleNamespace(
        architectures=["MiMoV2ForConditionalGeneration"],
        vision_config=_vision_config(),
        audio_config=_audio_config(),
        processor_config={},
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=12,
        vision_end_token_id=13,
    )
    hf_processor = lambda **_: {"input_ids": np.asarray([[1, 2]], dtype=np.int32)}
    processor = MiMoV2Processor(
        config,
        SimpleNamespace(model_path="."),
        hf_processor,
    )
    output = asyncio.run(
        processor.process_mm_data_async(
            None,
            "hello",
            SimpleNamespace(video_data=None, audio_data=None),
        )
    )
    assert output.input_ids == [1, 2]
    assert output.mrope_positions is None
    assert output.mrope_position_delta is None


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("MiMoV2ForCausalLM", "MiMoV2ForConditionalGeneration"),
        ("MiMoV2FlashForCausalLM", "MiMoV2FlashForConditionalGeneration"),
    ],
)
def test_model_config_selects_multimodal_wrapper(source, target):
    config = PretrainedConfig()
    config.architectures = [source]
    config.model_type = "mimo_v2"
    config.vocab_size = 8
    config.hidden_size = 4
    config.num_attention_heads = 1
    config.num_key_value_heads = 1
    config.num_hidden_layers = 1
    config.max_position_embeddings = 128
    config.eos_token_id = 2
    config.vision_config = _vision_config()
    config.audio_config = _audio_config()

    with (
        patch("sgl_jax.srt.configs.model_config.get_config", return_value=config),
        patch("sgl_jax.srt.configs.model_config.get_generation_config", return_value=None),
    ):
        model_config = ModelConfig(".", dtype="float32")

    assert model_config.hf_config.architectures == [target]
    assert model_config.is_multimodal
