from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.mesh import AxisType
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def _mesh():
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    return Mesh(
        devices,
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _tiny_config(**overrides):
    cfg = dict(
        architectures=["Qwen3DSparkModel"],
        model_type="qwen3",
        vocab_size=8,
        hidden_size=4,
        target_hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        max_position_embeddings=32,
        rope_theta=1000000,
        rms_norm_eps=1e-6,
        attention_bias=False,
        block_size=3,
        target_layer_ids=[0, 1],
        mask_token_id=7,
        markov_rank=2,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        tie_word_embeddings=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_dspark_architecture_is_registered():
    from sgl_jax.srt.models.registry import ModelRegistry

    model_cls, arch = ModelRegistry.resolve_model_cls(["Qwen3DSparkModel"])
    assert arch == "Qwen3DSparkModel"
    assert model_cls.__name__ == "Qwen3DSparkModel"


def test_dspark_config_splits_gamma_and_verify_width():
    from sgl_jax.srt.speculative.dspark_util import dspark_config_from_hf

    config = dspark_config_from_hf(_tiny_config(block_size=7))
    assert config.gamma == 7
    assert config.draft_width == 7
    assert config.verify_width == 8


def test_dspark_rejects_non_vanilla_markov_stage1():
    from sgl_jax.srt.speculative.dspark_util import dspark_config_from_hf

    try:
        dspark_config_from_hf(_tiny_config(markov_head_type="gated"))
    except ValueError as exc:
        assert "only supports markov_head_type='vanilla'" in str(exc)
    else:
        raise AssertionError("stage1 must reject non-vanilla Markov heads")


def test_vanilla_markov_head_is_serial():
    from sgl_jax.srt.models.dspark import VanillaMarkovHead

    mesh = _mesh()
    with jax.set_mesh(mesh):
        head = VanillaMarkovHead(
            vocab_size=4,
            markov_rank=2,
            mesh=mesh,
            dtype=jnp.float32,
        )
    head.markov_w1.value = jnp.asarray([[1, 0], [0, 1], [1, 0], [0, 0]], dtype=jnp.float32)
    # Internal LinearBase layout is [rank, vocab].
    head.markov_w2.weight.value = jnp.asarray([[0, 5, 0, 0], [0, 0, 5, 0]], dtype=jnp.float32)

    prev = jnp.asarray([0], dtype=jnp.int32)
    step0, _ = head.apply_step_logits(jnp.zeros((1, 4)), prev)
    token0 = jnp.argmax(step0, axis=-1).astype(jnp.int32)
    step1, _ = head.apply_step_logits(jnp.zeros((1, 4)), token0)
    token1 = jnp.argmax(step1, axis=-1).astype(jnp.int32)
    assert np.asarray(token0).tolist() == [1]
    assert np.asarray(token1).tolist() == [2]


def test_confidence_head_uses_markov_embedding():
    from sgl_jax.srt.models.dspark import DSparkConfidenceHead

    mesh = _mesh()
    with jax.set_mesh(mesh):
        head = DSparkConfidenceHead(
            hidden_size=2,
            markov_rank=1,
            mesh=mesh,
            dtype=jnp.float32,
        )
    head.proj.weight.value = jnp.asarray([[0.0], [0.0], [2.0]], dtype=jnp.float32)
    head.proj.bias.value = jnp.asarray([0.0], dtype=jnp.float32)
    confidence_logits = head(
        jnp.zeros((2, 2), dtype=jnp.float32),
        jnp.asarray([[0.0], [1.0]], dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        np.asarray(confidence_logits),
        np.asarray([0.0, 2.0]),
        rtol=1e-6,
    )


def test_confidence_head_aligns_markov_embedding_sharding():
    from sgl_jax.srt.models.dspark import DSparkConfidenceHead

    mesh = _mesh()
    with jax.set_mesh(mesh):
        head = DSparkConfidenceHead(
            hidden_size=2,
            markov_rank=1,
            mesh=mesh,
            dtype=jnp.float32,
        )
    head.proj.weight.value = jnp.zeros((3, 1), dtype=jnp.float32)
    head.proj.bias.value = jnp.zeros((1,), dtype=jnp.float32)
    hidden_states = jax.device_put(
        jnp.zeros((2, 2), dtype=jnp.float32),
        NamedSharding(mesh, P(None, "tensor")),
    )
    markov_embeddings = jax.device_put(
        jnp.zeros((2, 1), dtype=jnp.float32),
        NamedSharding(mesh, P("data", None)),
    )

    raw_confidence = jax.jit(head.raw_confidence)(hidden_states, markov_embeddings)
    np.testing.assert_allclose(np.asarray(raw_confidence), 0.0)


def test_dspark_weight_mapping_matches_official_checkpoint_names():
    from sgl_jax.srt.models.dspark import DSparkDraftModel

    cfg = _tiny_config()
    mappings = DSparkDraftModel._create_weight_mappings(SimpleNamespace(config=cfg))
    assert mappings["markov_head.markov_w1.weight"].target_path == ("markov_head.markov_w1")
    assert mappings["markov_head.markov_w2.weight"].transpose is True
    assert mappings["confidence_head.proj.weight"].transpose is True
    assert mappings["confidence_head.proj.bias"].transpose is False
    assert "embed_tokens.weight" not in mappings
    assert "lm_head.weight" not in mappings


def test_dspark_markov_block_returns_gamma_tokens_and_confidence_logits():
    from sgl_jax.srt.models.dspark import DSparkDraftModel

    mesh = _mesh()
    cfg = _tiny_config()
    with jax.set_mesh(mesh):
        model = DSparkDraftModel(cfg, mesh=mesh, dtype=jnp.float32)
    model.markov_head.markov_w1.value = jnp.zeros((cfg.vocab_size, cfg.markov_rank))
    model.markov_head.markov_w2.weight.value = jnp.zeros((cfg.markov_rank, cfg.vocab_size))
    model.confidence_head.proj.weight.value = jnp.zeros((cfg.hidden_size + cfg.markov_rank, 1))
    model.confidence_head.proj.bias.value = jnp.zeros((1,))

    base_logits = jnp.zeros((2, cfg.block_size, cfg.vocab_size))
    base_logits = base_logits.at[:, :, 3].set(1.0)
    run = jax.jit(model.generate_markov_block)
    tokens, confidence_logits = run(
        base_logits,
        jnp.zeros((2, cfg.block_size, cfg.hidden_size)),
        jnp.asarray([0, 1], dtype=jnp.int32),
    )
    assert tokens.shape == (2, cfg.block_size)
    assert confidence_logits.shape == (2, cfg.block_size)
    assert np.asarray(tokens).tolist() == [[3, 3, 3], [3, 3, 3]]
    np.testing.assert_allclose(np.asarray(confidence_logits), 0.0)
