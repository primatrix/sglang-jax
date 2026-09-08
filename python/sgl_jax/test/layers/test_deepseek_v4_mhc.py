"""M1.2 -- the mHC layer.

The reference is `test/srt/kernels/mhc/ref.py`, the independent float64 NumPy oracle
that landed with the kernels in #341. It was written from the published semantics
rather than from the kernels, so agreement is evidence about the semantics.

Checked on CPU against that oracle: the three gates, the pre/post pair, the head
collapse, and the sequencing a decoder layer performs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from sgl_jax.srt.configs.deepseek_v4 import DeepseekV4Config, mhc_param_shapes
from sgl_jax.srt.layers.deepseek_v4_mhc import (
    DeepseekV4MHC,
    collapse_head_reference,
    expand_streams,
    mhc_gates_reference,
    post_reference,
    pre_reference,
    resolve_backend,
)

# The oracle lives under test/srt, which is not a package on the path.
_ORACLE_DIR = Path(__file__).resolve().parents[4] / "test" / "srt" / "kernels" / "mhc"
if str(_ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(_ORACLE_DIR))
ref = pytest.importorskip("ref", reason="mHC NumPy oracle not importable")

HC = 4
D = 16
T = 6
EPS = 1e-6
ITERS = 3


def _cfg(**kw):
    return DeepseekV4Config(hc_mult=HC, hidden_size=D, hc_sinkhorn_iters=ITERS, hc_eps=EPS, **kw)


def _params(seed=0, *, head=False):
    rng = np.random.default_rng(seed)
    shapes = mhc_param_shapes(_cfg())
    if head:
        return (
            (rng.normal(size=shapes["head_fn"]) * 0.2).astype(np.float32),
            (rng.normal(size=shapes["head_base"]) * 0.2).astype(np.float32),
            (1.0 + 0.1 * rng.normal(size=shapes["head_scale"])).astype(np.float32),
        )
    return (
        (rng.normal(size=shapes["fn"]) * 0.2).astype(np.float32),
        (rng.normal(size=shapes["base"]) * 0.2).astype(np.float32),
        (1.0 + 0.1 * rng.normal(size=shapes["scale"])).astype(np.float32),
    )


def _streams(seed=0, tokens=T):
    return np.random.default_rng(seed).normal(size=(tokens, HC, D)).astype(np.float32)


# --------------------------------------------------------------------------
# shapes and stream handling
# --------------------------------------------------------------------------


def test_expand_streams_replicates_the_embedding():
    hidden = np.arange(T * D, dtype=np.float32).reshape(T, D)
    out = np.asarray(expand_streams(hidden, HC))
    assert out.shape == (T, HC, D)
    for h in range(HC):
        np.testing.assert_array_equal(out[:, h], hidden)


def test_expand_streams_rejects_a_flat_input():
    with pytest.raises(ValueError, match="hidden must be"):
        expand_streams(np.zeros((D,), np.float32)[0], HC)


def test_param_shapes_come_from_the_config():
    shapes = mhc_param_shapes(_cfg())
    assert shapes["fn"] == (24, HC * D)  # (2+4)*4 = 24
    assert shapes["base"] == (24,)
    assert shapes["scale"] == (3,)
    assert shapes["head_fn"] == (HC, HC * D)


# --------------------------------------------------------------------------
# the gates, against the oracle
# --------------------------------------------------------------------------


def test_gates_match_the_oracle():
    rng = np.random.default_rng(3)
    mixes = rng.normal(size=(T, 24)).astype(np.float32)
    fn, base, scale = _params(4)
    got = [
        np.asarray(a)
        for a in mhc_gates_reference(mixes, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, eps=EPS)
    ]
    want = ref.sinkhorn_gates(mixes, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, eps=EPS)
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)


def test_pre_adds_eps_after_the_sigmoid_and_post_does_not():
    """The asymmetry the oracle calls load-bearing: pre gets +eps, post gets x2."""
    mixes = np.full((1, 24), -60.0, np.float32)  # sigmoid ~ 0
    base = np.zeros((24,), np.float32)
    scale = np.ones((3,), np.float32)
    pre_gate, post_gate, _ = mhc_gates_reference(
        mixes, scale, base, hc_mult=HC, sinkhorn_iters=1, eps=0.25
    )
    # pre floors at eps; post has no eps at all.
    np.testing.assert_allclose(np.asarray(pre_gate), 0.25, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(post_gate), 0.0, rtol=1e-5, atol=1e-6)
    # And post's ceiling is 2, not 1.
    _, big, _ = mhc_gates_reference(
        np.full((1, 24), 60.0, np.float32), scale, base, hc_mult=HC, sinkhorn_iters=1, eps=0.0
    )
    np.testing.assert_allclose(np.asarray(big), 2.0, rtol=1e-5, atol=1e-5)


def test_sinkhorn_iteration_count_changes_the_result():
    """Guards the schedule: row softmax, one column pass, then iters-1 pairs."""
    rng = np.random.default_rng(5)
    mixes = rng.normal(size=(T, 24)).astype(np.float32)
    fn, base, scale = _params(6)
    one = np.asarray(
        mhc_gates_reference(mixes, scale, base, hc_mult=HC, sinkhorn_iters=1, eps=EPS)[2]
    )
    many = np.asarray(
        mhc_gates_reference(mixes, scale, base, hc_mult=HC, sinkhorn_iters=8, eps=EPS)[2]
    )
    assert not np.allclose(one, many, rtol=1e-4)
    # The schedule always *ends* on a column pass, so columns are normalised at
    # every iteration count; rows only approach 1 as the pairs accumulate. Asserting
    # the rows at one iteration would be asserting the wrong side of the schedule.
    for arr in (one, many):
        np.testing.assert_allclose(arr.sum(axis=-2), 1.0, atol=2e-5)
    row_error_one = np.abs(one.sum(axis=-1) - 1.0).max()
    row_error_many = np.abs(many.sum(axis=-1) - 1.0).max()
    assert row_error_many < row_error_one


# --------------------------------------------------------------------------
# pre / post / head, against the oracle
# --------------------------------------------------------------------------


def test_pre_matches_the_oracle():
    x = _streams(7)
    fn, base, scale = _params(8)
    y, post_gate, comb = pre_reference(
        x, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, norm_eps=EPS, hc_eps=EPS
    )
    wy, wpost, wcomb = ref.pre(
        x, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, norm_eps=EPS, hc_eps=EPS
    )
    np.testing.assert_allclose(np.asarray(y), wy, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(post_gate), wpost, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(comb), wcomb, rtol=2e-5, atol=2e-5)


def test_post_matches_the_oracle():
    x = _streams(9)
    fn, base, scale = _params(10)
    y, post_gate, comb = ref.pre(
        x, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, norm_eps=EPS, hc_eps=EPS
    )
    sublayer_out = np.random.default_rng(11).normal(size=(T, D)).astype(np.float32)
    got = np.asarray(post_reference(sublayer_out, x, post_gate, comb))
    want = ref.post(sublayer_out, x, post_gate, comb)
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)
    assert got.shape == (T, HC, D)


def test_head_collapse_matches_the_oracle():
    x = _streams(12)
    fn, base, scale = _params(13, head=True)
    got = np.asarray(collapse_head_reference(x, fn, scale, base, norm_eps=EPS, hc_eps=EPS))
    want = ref.head_collapse(x, fn, scale, base, norm_eps=EPS, hc_eps=EPS)
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)
    assert got.shape == (T, D)


def test_head_collapse_is_not_the_same_normalisation_as_pre():
    """`pre` scales the projection; `head_collapse` scales the activation before it,
    with a bf16 rounding in between. Same-looking code, different numbers -- so a
    test has to distinguish them or the asymmetry will get 'cleaned up'."""
    x = _streams(14) * 40.0  # large enough that the bf16 rounding is visible
    fn, base, scale = _params(15, head=True)
    collapsed = np.asarray(collapse_head_reference(x, fn, scale, base, norm_eps=EPS, hc_eps=EPS))
    # Recompute with pre's ordering (scale after the projection, no rounding).
    flat = x.reshape(T, -1).astype(np.float64)
    rsq = 1.0 / np.sqrt((flat**2).mean(-1, keepdims=True) + EPS)
    wrong_mixes = (flat @ fn.astype(np.float64).T) * rsq
    wrong_gate = 1.0 / (1.0 + np.exp(-(wrong_mixes * scale[0] + base))) + EPS
    wrong = (wrong_gate[..., None] * x.astype(np.float64)).sum(-2)
    assert not np.allclose(collapsed, wrong, rtol=1e-3)


# --------------------------------------------------------------------------
# the layer object
# --------------------------------------------------------------------------


def test_backend_resolution():
    assert resolve_backend("reference") == "reference"
    assert resolve_backend("pallas") == "pallas"
    # On CPU, auto must not pick the TPU-only kernels.
    assert resolve_backend("auto") == "reference"
    with pytest.raises(ValueError, match="unknown mHC backend"):
        resolve_backend("cuda")


def test_layer_pre_post_round_trip_matches_the_oracle():
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    x = _streams(16)
    fn, base, scale = _params(17)
    collapsed, post_gate, comb = mhc.pre(x, fn, base, scale)
    out = np.asarray(mhc.post(np.asarray(collapsed) * 0.5, x, post_gate, comb))

    wy, wpost, wcomb = ref.pre(
        x, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, norm_eps=EPS, hc_eps=EPS
    )
    want = ref.post(wy * 0.5, x, wpost, wcomb)
    np.testing.assert_allclose(out, want, rtol=2e-5, atol=2e-5)


def test_wrap_sublayer_is_the_sequential_semantics():
    """The unfused form: residual = X, collapse, run F, expand against the residual."""
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    x = _streams(18)
    fn, base, scale = _params(19)
    calls = []

    def sublayer(u):
        calls.append(np.asarray(u).shape)
        return np.asarray(u) * 2.0 + 1.0

    out = np.asarray(mhc.wrap_sublayer(x, fn, base, scale, sublayer))
    assert calls == [(T, D)]  # the sublayer sees the collapsed stream
    assert out.shape == (T, HC, D)

    wy, wpost, wcomb = ref.pre(
        x, fn, scale, base, hc_mult=HC, sinkhorn_iters=ITERS, norm_eps=EPS, hc_eps=EPS
    )
    np.testing.assert_allclose(out, ref.post(wy * 2.0 + 1.0, x, wpost, wcomb), rtol=2e-5, atol=2e-5)


def test_the_residual_is_the_input_not_the_collapsed_stream():
    """`post` must mix against the original hc streams. Passing the collapsed value
    would typecheck and produce the right shape."""
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    x = _streams(20)
    fn, base, scale = _params(21)
    correct = np.asarray(mhc.wrap_sublayer(x, fn, base, scale, lambda u: np.asarray(u)))
    collapsed, post_gate, comb = mhc.pre(x, fn, base, scale)
    wrong = np.asarray(
        mhc.post(
            collapsed, np.repeat(np.asarray(collapsed)[:, None, :], HC, axis=1), post_gate, comb
        )
    )
    assert not np.allclose(correct, wrong, rtol=1e-3)


def test_two_sublayers_use_independent_gate_parameters():
    """Attention and FFN each get their own pre/post set; sharing them silently
    couples the two halves of a layer."""
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    x = _streams(22)
    attn = _params(23)
    ffn = _params(24)
    after_attn = mhc.wrap_sublayer(x, *attn, lambda u: np.asarray(u))
    a = np.asarray(mhc.wrap_sublayer(after_attn, *ffn, lambda u: np.asarray(u)))
    b = np.asarray(mhc.wrap_sublayer(after_attn, *attn, lambda u: np.asarray(u)))
    assert not np.allclose(a, b, rtol=1e-3)


# --------------------------------------------------------------------------
# parameter validation
# --------------------------------------------------------------------------


def test_gate_parameters_must_stay_float32():
    """They are Sinkhorn coefficients, not projections, so they do not follow the
    activation dtype."""
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    fn, base, scale = _params(25)
    with pytest.raises(ValueError, match="must stay float32"):
        mhc.check_params(fn.astype(np.float16), base, scale)


def test_gate_parameter_shapes_are_checked():
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    fn, base, scale = _params(26)
    with pytest.raises(ValueError, match="mHC fn fn must be"):
        mhc.check_params(fn[:, :-1], base, scale)
    with pytest.raises(ValueError, match="mHC fn base must be"):
        mhc.check_params(fn, base[:-1], scale)
    with pytest.raises(ValueError, match="mHC fn scale must be"):
        mhc.check_params(fn, base, scale[:-1])


def test_head_parameter_shapes_are_checked_separately():
    mhc = DeepseekV4MHC(_cfg(), backend="reference")
    head = _params(27, head=True)
    mhc.check_params(*head, kind="head_fn")
    layer_fn, base, scale = _params(28)
    with pytest.raises(ValueError, match="mHC head_fn fn must be"):
        mhc.check_params(layer_fn, base, scale, kind="head_fn")


def test_shipped_config_geometry():
    """hc_mult=4, hidden=4096 -> mix_hc=24 and hc_dim=16384."""
    shapes = mhc_param_shapes(DeepseekV4Config())
    assert shapes["fn"] == (24, 16384)
    assert shapes["head_fn"] == (4, 16384)
    assert DeepseekV4Config().hc_sinkhorn_iters == 20
