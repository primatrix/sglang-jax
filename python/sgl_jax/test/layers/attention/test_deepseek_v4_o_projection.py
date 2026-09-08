"""A1/A2 -- grouped output projection and the two RoPE tables.

Independent NumPy oracles, real Flash 0731 shapes where they are small enough to be
cheap, and the two facts that were wrong or missing before: `o_lora_rank` is
per-group, and the RoPE base is chosen per layer rather than per path.
"""

import numpy as np
import pytest

from sgl_jax.srt.configs.deepseek_v4 import DeepseekV4Config
from sgl_jax.srt.layers.attention.dsv4.o_projection import (
    group_wo_a,
    grouped_output_projection,
)
from sgl_jax.srt.layers.attention.dsv4.rope import (
    apply_dsv4_partial_rope,
    build_dsv4_rope,
    dsv4_rope_tables,
    dsv4_rope_theta,
)

G = 4  # o_groups
Hpg = 2  # heads per group
DH = 8  # head_dim
ROPE = 4
R = 3  # o_lora_rank per group
HIDDEN = 6
T = 5


# --------------------------------------------------------------------------
# A2: which RoPE base, and the shape of the mistake it fixes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ratio,expected",
    [(0, "plain"), (1, "plain"), (4, "compressed"), (128, "compressed")],
)
def test_rope_base_is_chosen_per_layer(ratio, expected):
    """`compress_ratio > 1` takes compress_rope_theta; 0 and 1 take rope_theta."""
    cfg = DeepseekV4Config()
    got = dsv4_rope_theta(cfg, ratio)
    assert got == (cfg.compress_rope_theta if expected == "compressed" else cfg.rope_theta)


def test_the_two_bases_are_actually_different():
    """Otherwise the whole per-layer selection would be decoration."""
    cfg = DeepseekV4Config()
    assert cfg.rope_theta == 10000.0
    assert cfg.compress_rope_theta == 160000.0
    assert dsv4_rope_theta(cfg, 0) != dsv4_rope_theta(cfg, 4)


def test_there_are_exactly_two_tables_not_one_per_layer():
    """43 layers, two bases. Building per layer would be 43 identical-in-pairs tables."""
    cfg = DeepseekV4Config()
    tables = dsv4_rope_tables(cfg)
    assert set(tables) == {"plain", "compressed"}
    assert tables["plain"].base == cfg.rope_theta
    assert tables["compressed"].base == cfg.compress_rope_theta


def test_negative_ratio_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        dsv4_rope_theta(DeepseekV4Config(), -1)


def test_yarn_magnitude_scaling_is_disabled():
    """V4 sets mscale and mscale_all_dim to 0. That has to come out as a factor of
    exactly 1.0, not as a zero that quietly scales cos/sin to nothing."""
    rope = build_dsv4_rope(DeepseekV4Config(), 4)
    assert rope.yarn_mscale == 0.0
    assert rope.yarn_mscale_all_dim == 0.0
    assert rope._rope_mscale == pytest.approx(1.0)


def test_rope_uses_the_config_yarn_parameters():
    cfg = DeepseekV4Config()
    rope = build_dsv4_rope(cfg, 4)
    assert rope.scaling_factor == 16.0
    assert rope.original_max_position_embeddings == 65536
    assert (rope.beta_fast, rope.beta_slow) == (32.0, 1.0)
    assert rope.rotary_dim == cfg.qk_rope_head_dim == 64


# --------------------------------------------------------------------------
# A2: the inverse rotation
# --------------------------------------------------------------------------


def _cos_sin(seed=0, tokens=T, half=ROPE // 2):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi, size=(tokens, half))
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


def test_inverse_rope_undoes_the_forward_rotation():
    """The property the output projection relies on."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(T, 3, DH)).astype(np.float32)
    cos, sin = _cos_sin(2)
    c, s = cos[:, None, :], sin[:, None, :]
    rotated = apply_dsv4_partial_rope(x, c, s, rope_head_dim=ROPE)
    back = apply_dsv4_partial_rope(rotated, c, s, rope_head_dim=ROPE, inverse=True)
    np.testing.assert_allclose(np.asarray(back), x, rtol=2e-6, atol=2e-6)


def test_inverse_rope_actually_changes_the_value():
    """Guards the round-trip test from passing because both directions are no-ops."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(T, 1, DH)).astype(np.float32)
    cos, sin = _cos_sin(4)
    c, s = cos[:, None, :], sin[:, None, :]
    assert not np.allclose(
        np.asarray(apply_dsv4_partial_rope(x, c, s, rope_head_dim=ROPE, inverse=True)), x
    )


def test_rope_leaves_the_non_rotary_features_alone():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(T, 2, DH)).astype(np.float32)
    cos, sin = _cos_sin(6)
    out = np.asarray(
        apply_dsv4_partial_rope(x, cos[:, None, :], sin[:, None, :], rope_head_dim=ROPE)
    )
    np.testing.assert_array_equal(out[..., : DH - ROPE], x[..., : DH - ROPE])


# --------------------------------------------------------------------------
# A1: the grouping convention
# --------------------------------------------------------------------------


def _wo_a(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(G * R, Hpg * DH)) * 0.2).astype(np.float32)


def test_wo_a_grouping_is_group_major():
    """Checkpoint output index is g*R + r. Reading it as r-major produces the same
    shape and silently mixes groups, so pin the layout element by element."""
    wo_a = _wo_a(7)
    grouped = np.asarray(group_wo_a(wo_a, num_groups=G))
    assert grouped.shape == (G, Hpg * DH, R)
    for g in range(G):
        for r in range(R):
            np.testing.assert_array_equal(grouped[g, :, r], wo_a[g * R + r])


def test_wo_a_grouping_rejects_an_indivisible_width():
    with pytest.raises(ValueError, match="not divisible by o_groups"):
        group_wo_a(np.zeros((G * R + 1, Hpg * DH), np.float32), num_groups=G)


# --------------------------------------------------------------------------
# A1: the projection itself
# --------------------------------------------------------------------------


def _oracle(attn_out, wo_a, wo_b, cos, sin, *, num_groups, rope_head_dim, inverse=True):
    """Per-token, per-group NumPy reference."""
    x = np.asarray(attn_out, np.float64).copy()
    if inverse:
        tail = x[..., x.shape[-1] - rope_head_dim :]
        even, odd = tail[..., 0::2].copy(), tail[..., 1::2].copy()
        c = np.asarray(cos, np.float64)[:, None, :]
        s = np.asarray(sin, np.float64)[:, None, :]
        # inverse == negated sin
        tail[..., 0::2] = even * c + odd * s
        tail[..., 1::2] = -even * s + odd * c
    tokens, heads, head_dim = x.shape
    hpg = heads // num_groups
    rank = wo_a.shape[0] // num_groups
    out = np.zeros((tokens, wo_b.shape[0]), np.float64)
    for t in range(tokens):
        lora = np.zeros((num_groups, rank), np.float64)
        for g in range(num_groups):
            vec = x[t, g * hpg : (g + 1) * hpg].reshape(-1)
            for r in range(rank):
                lora[g, r] = vec @ np.asarray(wo_a[g * rank + r], np.float64)
        out[t] = np.asarray(wo_b, np.float64) @ lora.reshape(-1)
    return out


def _run(seed=0, inverse=True):
    rng = np.random.default_rng(seed)
    attn = rng.normal(size=(T, G * Hpg, DH)).astype(np.float32)
    wo_a = _wo_a(seed + 1)
    wo_b = (rng.normal(size=(HIDDEN, G * R)) * 0.2).astype(np.float32)
    cos, sin = _cos_sin(seed + 2)
    got = np.asarray(
        grouped_output_projection(
            attn,
            wo_a=wo_a,
            wo_b=wo_b,
            cos=cos,
            sin=sin,
            num_groups=G,
            rope_head_dim=ROPE,
            apply_inverse_rope=inverse,
        )
    )
    want = _oracle(attn, wo_a, wo_b, cos, sin, num_groups=G, rope_head_dim=ROPE, inverse=inverse)
    return got, want, dict(attn=attn, wo_a=wo_a, wo_b=wo_b, cos=cos, sin=sin)


@pytest.mark.parametrize("inverse", [True, False])
def test_matches_the_per_group_oracle(inverse):
    got, want, _ = _run(seed=11, inverse=inverse)
    assert got.shape == (T, HIDDEN)
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)


def test_no_cross_group_mixing_before_wo_b():
    """Perturb one group's heads and only that group's LoRA slice may change. This is
    what distinguishes the grouped projection from a plain o_proj."""
    _, _, d = _run(seed=21)
    attn, wo_a = d["attn"], d["wo_a"]
    grouped = np.asarray(group_wo_a(wo_a, num_groups=G))
    x = attn.reshape(T, G, Hpg * DH).astype(np.float64)
    lora = np.einsum("tgd,gdr->tgr", x, grouped.astype(np.float64))

    bumped = attn.copy()
    bumped[:, :Hpg] += 3.0  # group 0's heads only
    x2 = bumped.reshape(T, G, Hpg * DH).astype(np.float64)
    lora2 = np.einsum("tgd,gdr->tgr", x2, grouped.astype(np.float64))

    assert not np.allclose(lora[:, 0], lora2[:, 0])
    np.testing.assert_allclose(lora[:, 1:], lora2[:, 1:], rtol=1e-12, atol=1e-12)


def test_inverse_rope_is_applied_by_default():
    with_rope, _, _ = _run(seed=31, inverse=True)
    without, _, _ = _run(seed=31, inverse=False)
    assert not np.allclose(with_rope, without, rtol=1e-3)


# --------------------------------------------------------------------------
# A1: shape guards, using the real Flash 0731 numbers
# --------------------------------------------------------------------------


def test_flash_0731_shapes_line_up():
    """The checkpoint has wo_a [8192, 4096] and wo_b [4096, 8192]. Derive both from
    the config so a wrong reading of `o_lora_rank` fails here rather than at load."""
    cfg = DeepseekV4Config()
    heads_per_group = cfg.num_attention_heads // cfg.o_groups
    assert heads_per_group == 8  # equals the TPU sublane count the kernel assumes
    assert cfg.o_groups * cfg.o_lora_rank == 8192  # wo_a output width
    assert heads_per_group * cfg.head_dim == 4096  # wo_a reduction
    assert cfg.hidden_size == 4096  # wo_b output


def test_rejects_heads_that_do_not_split_into_groups():
    rng = np.random.default_rng(41)
    attn = rng.normal(size=(T, G * Hpg + 1, DH)).astype(np.float32)
    cos, sin = _cos_sin(42)
    with pytest.raises(ValueError, match="do not split into"):
        grouped_output_projection(
            attn,
            wo_a=_wo_a(),
            wo_b=np.zeros((HIDDEN, G * R), np.float32),
            cos=cos,
            sin=sin,
            num_groups=G,
            rope_head_dim=ROPE,
        )


def test_rejects_a_wo_a_reduction_that_disagrees_with_the_heads():
    rng = np.random.default_rng(51)
    attn = rng.normal(size=(T, G * Hpg, DH)).astype(np.float32)
    cos, sin = _cos_sin(52)
    bad = rng.normal(size=(G * R, Hpg * DH + 8)).astype(np.float32)
    with pytest.raises(ValueError, match="wo_a reduction"):
        grouped_output_projection(
            attn,
            wo_a=bad,
            wo_b=np.zeros((HIDDEN, G * R), np.float32),
            cos=cos,
            sin=sin,
            num_groups=G,
            rope_head_dim=ROPE,
        )


def test_rejects_a_wo_b_width_that_disagrees_with_g_times_r():
    rng = np.random.default_rng(61)
    attn = rng.normal(size=(T, G * Hpg, DH)).astype(np.float32)
    cos, sin = _cos_sin(62)
    with pytest.raises(ValueError, match="wo_b input width"):
        grouped_output_projection(
            attn,
            wo_a=_wo_a(),
            wo_b=np.zeros((HIDDEN, G * R + 1), np.float32),
            cos=cos,
            sin=sin,
            num_groups=G,
            rope_head_dim=ROPE,
        )
