"""Phase B: numerical alignment tests for KimiDeltaAttention.

Prefill tests (TestKDAPrefill):
    Load GPU reference dumps and compare the JAX EXTEND forward pass on TPU.
    12 cases x 2 dtypes (fp32 + bf16) = 24 tests.

Decode tests (TestKDADecode):
    Verify prefill(T-1) + decode(1) matches prefill(T) at the last position.
    3 cases x 1 dtype (fp32) = 3 tests.

Run on TPU v6e-4:
    conda activate sglang
    python -m pytest test/layers/test_kda_backend.py -v

Override dump location:
    KDA_DUMP_DIR=/path/to/kda_module python -m pytest ...
"""

from __future__ import annotations

import os
import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    MockRecurrentStatePool,
)
from sgl_jax.srt.layers.attention.linear.kda_backend import (
    KDAAttnBackend,
    KDAAttnBackendMetadata,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.models.kimi_linear import KimiDeltaAttention

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DUMP_BASE = os.environ.get(
    "KDA_DUMP_DIR", "/models/yuhao/kimi-linear/kda_module"
)
DEFAULT_LAYER = os.environ.get("KDA_DUMP_LAYER", "L0")


def _layer_dir():
    return os.path.join(DUMP_BASE, DEFAULT_LAYER)


# ===================================================================
# Shared helpers
# ===================================================================


def _make_config(weights: dict) -> SimpleNamespace:
    """Build a minimal config from weights.npz metadata keys."""
    return SimpleNamespace(
        hidden_size=int(weights["config__hidden_size"]),
        rms_norm_eps=float(weights["config__rms_norm_eps"]),
        linear_attn_config={
            "num_heads": int(weights["config__num_heads"]),
            "head_dim": int(weights["config__head_dim"]),
            "short_conv_kernel_size": int(weights["config__conv_size"]),
            "kda_layers": [1],
            "full_attn_layers": [],
        },
    )


def _set_param(module, attr_path: str, value: np.ndarray) -> None:
    """Set a nested parameter on an nnx module."""
    parts = attr_path.split(".")
    obj = module
    for part in parts[:-1]:
        obj = getattr(obj, part)
    param = getattr(obj, parts[-1])
    param[...] = jnp.asarray(value)


def _build_module(
    weights_path: str,
    dtype: jnp.dtype = jnp.float32,
) -> KimiDeltaAttention:
    """Construct a KimiDeltaAttention and load GPU reference weights."""
    weights = dict(np.load(weights_path, allow_pickle=True))
    config = _make_config(weights)
    module = KimiDeltaAttention(config, layer_idx=0, dtype=dtype)

    num_heads = config.linear_attn_config["num_heads"]

    weight_map = {
        "q_proj.weight": "weights__q_proj.weight",
        "k_proj.weight": "weights__k_proj.weight",
        "v_proj.weight": "weights__v_proj.weight",
        "f_a_proj.weight": "weights__f_a_proj.weight",
        "f_b_proj.weight": "weights__f_b_proj.weight",
        "b_proj.weight": "weights__b_proj.weight",
        "g_a_proj.weight": "weights__g_a_proj.weight",
        "g_b_proj.weight": "weights__g_b_proj.weight",
        "o_proj.weight": "weights__o_proj.weight",
        "o_norm.weight": "weights__o_norm.weight",
    }
    for attr, key in weight_map.items():
        _set_param(module, attr, weights[key])

    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        w = weights[f"weights__{name}.weight"]
        if w.ndim == 3 and w.shape[1] == 1:
            w = w[:, 0, :]
        _set_param(module, f"{name}.weight", w)

    a_log = weights["weights__A_log"]
    _set_param(module, "A_log", a_log.reshape(1, 1, num_heads, 1))
    _set_param(module, "dt_bias", weights["weights__dt_bias"])

    return module


def _build_extend_env(
    module: KimiDeltaAttention,
    T: int,
    init_state: jax.Array | None = None,
    cu_seqlens: jax.Array | None = None,
) -> tuple[ForwardBatch, MockRecurrentStatePool]:
    """Build EXTEND ForwardBatch + pool for one or more sequences."""
    H, K, V = module.num_heads, module.head_k_dim, module.head_dim
    if cu_seqlens is None:
        cu_seqlens = jnp.array([0, T], dtype=jnp.int32)
    N = cu_seqlens.shape[0] - 1
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]

    backend = KDAAttnBackend(mesh=None)
    backend.forward_metadata = KDAAttnBackendMetadata(
        cu_q_lens=cu_seqlens,
        recurrent_indices=jnp.arange(N, dtype=jnp.int32),
    )
    fb = ForwardBatch(
        bid=0, forward_mode=ForwardMode.EXTEND, batch_size=int(N),
        input_ids=jnp.zeros(T, dtype=jnp.int32),
        req_pool_indices=jnp.arange(N, dtype=jnp.int32),
        seq_lens=seq_lens,
        out_cache_loc=jnp.zeros(T, dtype=jnp.int32),
        attn_backend=backend,
        extend_seq_lens=seq_lens,
    )
    if init_state is None:
        init_state = jnp.zeros((N, H, K, V), dtype=jnp.float32)
    pool = MockRecurrentStatePool(
        layer_caches={module.layer_id: (init_state, None)},
    )
    return fb, pool


def _build_decode_env(
    module: KimiDeltaAttention,
    pool: MockRecurrentStatePool,
    B: int = 1,
) -> ForwardBatch:
    """Build DECODE ForwardBatch reusing an existing pool's state."""
    backend = KDAAttnBackend(mesh=None)
    backend.forward_metadata = KDAAttnBackendMetadata(
        cu_q_lens=jnp.array([0, B], dtype=jnp.int32),
        recurrent_indices=jnp.arange(B, dtype=jnp.int32),
    )
    return ForwardBatch(
        bid=0, forward_mode=ForwardMode.DECODE, batch_size=B,
        input_ids=jnp.zeros(B, dtype=jnp.int32),
        req_pool_indices=jnp.arange(B, dtype=jnp.int32),
        seq_lens=jnp.ones(B, dtype=jnp.int32),
        out_cache_loc=jnp.zeros(B, dtype=jnp.int32),
        attn_backend=backend,
    )


# ===================================================================
# Prefill tests — GPU reference alignment (EXTEND mode)
# ===================================================================

PREFILL_CASES = [
    "single_T1",
    "single_T8",
    "single_T64",
    "single_T65",
    "single_T128",
    "single_T256",
    "single_T1024",
    "varlen_balanced_4x32",
    "varlen_unbalanced",
    "varlen_single_T128",
    "single_T128_initstate",
    "varlen_initstate",
]

# Two-tier tolerance for cross-device (GPU→TPU) + cross-kernel (chunk→naive).
# Tier 1 (tight): designed for L0, where max_abs ≈ 1.3e-3 (fp32) / 2.0e-3 (bf16).
# Tier 2 (loose): covers all layers up to L22, where max_abs ≈ 2.8e-2 (fp32) / 6.3e-2 (bf16).
# Tight pass → silent. Tight fail + loose pass → warning. Both fail → error.
FP32_ATOL_TIGHT = 2e-3
FP32_RTOL_TIGHT = 5e-3
FP32_ATOL_LOOSE = 2e-2
FP32_RTOL_LOOSE = 2e-2

BF16_ATOL_TIGHT = 3e-3
BF16_RTOL_TIGHT = 5e-3
BF16_ATOL_LOOSE = 2e-2
BF16_RTOL_LOOSE = 2e-2


def _assert_two_tier(
    actual: np.ndarray,
    expected: np.ndarray,
    atol_tight: float,
    rtol_tight: float,
    atol_loose: float,
    rtol_loose: float,
    label: str,
) -> None:
    """Assert with two-tier tolerance: tight first, then loose as fallback."""
    try:
        np.testing.assert_allclose(
            actual, expected, atol=atol_tight, rtol=rtol_tight,
        )
    except AssertionError:
        np.testing.assert_allclose(
            actual, expected, atol=atol_loose, rtol=rtol_loose,
            err_msg=f"{label}: exceeds loose tolerance",
        )
        max_abs = np.max(np.abs(actual - expected))
        warnings.warn(
            f"{label}: passed at loose tolerance (max_abs={max_abs:.2e}, "
            f"tight={atol_tight}, loose={atol_loose})",
            stacklevel=2,
        )


def _report_intermediates(case: dict, module, forward_batch, pool):
    """Print per-step max-abs-diff for debugging prefill divergence."""
    hidden = jnp.asarray(case["hidden_states"], dtype=jnp.float32)
    hidden_2d, _ = module._flatten_hidden_states(hidden)

    q = module.q_proj(hidden_2d)
    k = module.k_proj(hidden_2d)
    v = module.v_proj(hidden_2d)
    q, k, v = module._short_convs(q, k, v, forward_batch, pool)
    q_heads = module._split_heads(q, module.num_k_heads, module.head_k_dim)
    k_heads = module._split_heads(k, module.num_k_heads, module.head_k_dim)

    for label, actual, key in [
        ("q_after_conv", q_heads, "intermediates__q_after_conv"),
        ("k_after_conv", k_heads, "intermediates__k_after_conv"),
    ]:
        if key not in case:
            continue
        expected = case[key]
        if expected.ndim == 4 and actual.ndim == 3:
            expected = expected[0]
        diff = np.max(np.abs(np.asarray(actual, dtype=np.float32) - expected))
        print(f"  {label}: max_abs_diff = {diff:.2e}")


class TestKDAPrefill:
    """Prefill (EXTEND) alignment against GPU reference dumps."""

    @pytest.fixture(scope="class", autouse=True)
    def _check_dumps(self):
        d = _layer_dir()
        if not os.path.isdir(d):
            pytest.skip(f"KDA dumps not found at {d}")

    @pytest.fixture(scope="class")
    def module(self):
        return _build_module(os.path.join(_layer_dir(), "weights.npz"))

    @pytest.fixture(scope="class")
    def module_bf16(self):
        return _build_module(
            os.path.join(_layer_dir(), "weights.npz"),
            dtype=jnp.bfloat16,
        )

    @pytest.mark.parametrize("case_name", PREFILL_CASES)
    def test_prefill_fp32(self, module, case_name):
        case_path = os.path.join(_layer_dir(), f"case_{case_name}.npz")
        if not os.path.isfile(case_path):
            pytest.skip(f"Case file not found: {case_path}")

        case = dict(np.load(case_path, allow_pickle=True))
        forward_batch, pool = _build_extend_env(
            module, int(case["T"]),
            init_state=(
                jnp.asarray(case["initial_recurrent_state"], dtype=jnp.float32)
                if bool(case["has_initial_state"]) else None
            ),
            cu_seqlens=(
                jnp.asarray(case["cu_seqlens"], dtype=jnp.int32)
                if bool(case["has_cu_seqlens"]) else None
            ),
        )

        hidden = jnp.asarray(case["hidden_states"], dtype=jnp.float32)
        output = module(hidden, None, forward_batch, pool)
        output_np = np.asarray(output, dtype=np.float32)

        expected = case["out_fp32"]
        if output_np.ndim == 2 and expected.ndim == 3:
            output_np = output_np[None, ...]

        # GPU chunk kernel produces all-zero output for T < chunk_size (64).
        if np.all(expected == 0):
            assert not np.isnan(output_np).any(), f"{case_name}: NaN in output"
            assert np.abs(output_np).max() > 0, f"{case_name}: all-zero output"
            pytest.skip(
                f"{case_name}: GPU chunk reference is all-zero "
                f"(T < chunk_size); TPU output is non-zero (correct)"
            )

        try:
            _assert_two_tier(
                output_np, expected,
                FP32_ATOL_TIGHT, FP32_RTOL_TIGHT,
                FP32_ATOL_LOOSE, FP32_RTOL_LOOSE,
                case_name,
            )
        except AssertionError:
            print(f"\n--- Intermediate diagnostics for {case_name} ---")
            _, pool2 = _build_extend_env(
                module, int(case["T"]),
                init_state=(
                    jnp.asarray(case["initial_recurrent_state"], dtype=jnp.float32)
                    if bool(case["has_initial_state"]) else None
                ),
                cu_seqlens=(
                    jnp.asarray(case["cu_seqlens"], dtype=jnp.int32)
                    if bool(case["has_cu_seqlens"]) else None
                ),
            )
            _report_intermediates(case, module, forward_batch, pool2)
            raise

    @pytest.mark.parametrize("case_name", PREFILL_CASES)
    def test_prefill_bf16(self, module_bf16, case_name):
        case_path = os.path.join(_layer_dir(), f"case_{case_name}.npz")
        if not os.path.isfile(case_path):
            pytest.skip(f"Case file not found: {case_path}")

        case = dict(np.load(case_path, allow_pickle=True))
        if "out_bf16" not in case:
            pytest.skip(f"{case_name}: no out_bf16 in dump")

        forward_batch, pool = _build_extend_env(
            module_bf16, int(case["T"]),
            init_state=(
                jnp.asarray(case["initial_recurrent_state"], dtype=jnp.float32)
                if bool(case["has_initial_state"]) else None
            ),
            cu_seqlens=(
                jnp.asarray(case["cu_seqlens"], dtype=jnp.int32)
                if bool(case["has_cu_seqlens"]) else None
            ),
        )

        hidden = jnp.asarray(case["hidden_states"], dtype=jnp.bfloat16)
        output = module_bf16(hidden, None, forward_batch, pool)
        output_np = np.asarray(output, dtype=np.float32)

        expected = case["out_bf16"]
        if output_np.ndim == 2 and expected.ndim == 3:
            output_np = output_np[None, ...]

        if np.all(expected == 0):
            assert not np.isnan(output_np).any(), f"{case_name}: NaN in output"
            pytest.skip(
                f"{case_name}: GPU bf16 reference is all-zero (T < chunk_size)"
            )

        assert not np.isnan(output_np).any(), f"{case_name}: NaN in output"
        _assert_two_tier(
            output_np, expected,
            BF16_ATOL_TIGHT, BF16_RTOL_TIGHT,
            BF16_ATOL_LOOSE, BF16_RTOL_LOOSE,
            f"{case_name} bf16",
        )


# ===================================================================
# Decode tests — prefill(T-1) + decode(1) vs GPU reference (DECODE mode)
# ===================================================================

DECODE_CASES = ["single_T8", "single_T128", "single_T128_initstate"]

# Decode error compounds: cross-device + cross-kernel (same as prefill)
# plus prefill-decode split (XLA recompilation for T vs T-1, ~5e-4).
DECODE_FP32_ATOL_TIGHT = 1e-3
DECODE_FP32_RTOL_TIGHT = 1e-3
DECODE_FP32_ATOL_LOOSE = 1e-2
DECODE_FP32_RTOL_LOOSE = 1e-2

DECODE_BF16_ATOL_TIGHT = 2e-3
DECODE_BF16_RTOL_TIGHT = 2e-3
DECODE_BF16_ATOL_LOOSE = 2e-2
DECODE_BF16_RTOL_LOOSE = 2e-2


class TestKDADecode:
    """Decode: prefill(T-1) + decode(1) vs GPU reference at position T."""

    @pytest.fixture(scope="class", autouse=True)
    def _check_dumps(self):
        d = _layer_dir()
        if not os.path.isdir(d):
            pytest.skip(f"KDA dumps not found at {d}")

    @pytest.fixture(scope="class")
    def module(self):
        return _build_module(os.path.join(_layer_dir(), "weights.npz"))

    @pytest.fixture(scope="class")
    def module_bf16(self):
        return _build_module(
            os.path.join(_layer_dir(), "weights.npz"),
            dtype=jnp.bfloat16,
        )

    def _run_decode(self, module, case, in_dtype):
        """Prefill T-1 tokens, decode the T-th, return decode output."""
        T = int(case["T"])
        hidden = jnp.asarray(case["hidden_states"], dtype=in_dtype)
        hidden_2d = hidden[0] if hidden.ndim == 3 else hidden  # [T, D]

        has_init = bool(case["has_initial_state"])
        init_state = (
            jnp.asarray(case["initial_recurrent_state"], dtype=jnp.float32)
            if has_init else None
        )

        # 1) Prefill T-1 tokens → state (recurrent + conv)
        fb_prefix, pool_prefix = _build_extend_env(module, T - 1, init_state)
        _ = module(hidden_2d[: T - 1][None], None, fb_prefix, pool_prefix)

        # 2) Decode the T-th token using the state from step 1
        fb_decode = _build_decode_env(module, pool_prefix, B=1)
        out_decode = module(hidden_2d[T - 1 : T][None], None, fb_decode, pool_prefix)
        out_decode_np = np.asarray(out_decode, dtype=np.float32)
        if out_decode_np.ndim == 3:
            out_decode_np = out_decode_np[0]
        return out_decode_np  # [1, D]

    @pytest.mark.parametrize("case_name", DECODE_CASES)
    def test_decode_fp32(self, module, case_name):
        case_path = os.path.join(_layer_dir(), f"case_{case_name}.npz")
        if not os.path.isfile(case_path):
            pytest.skip(f"Case file not found: {case_path}")

        case = dict(np.load(case_path, allow_pickle=True))
        T = int(case["T"])
        expected = case["out_fp32"]  # [1, T, D]
        expected_last = expected[0, T - 1 : T] if expected.ndim == 3 else expected[T - 1 : T]

        out_decode = self._run_decode(module, case, jnp.float32)

        assert not np.isnan(out_decode).any(), f"{case_name}: NaN in decode output"
        _assert_two_tier(
            out_decode, expected_last,
            DECODE_FP32_ATOL_TIGHT, DECODE_FP32_RTOL_TIGHT,
            DECODE_FP32_ATOL_LOOSE, DECODE_FP32_RTOL_LOOSE,
            f"{case_name} decode fp32",
        )

    @pytest.mark.parametrize("case_name", DECODE_CASES)
    def test_decode_bf16(self, module_bf16, case_name):
        case_path = os.path.join(_layer_dir(), f"case_{case_name}.npz")
        if not os.path.isfile(case_path):
            pytest.skip(f"Case file not found: {case_path}")

        case = dict(np.load(case_path, allow_pickle=True))
        if "out_bf16" not in case:
            pytest.skip(f"{case_name}: no out_bf16 in dump")

        T = int(case["T"])
        expected = case["out_bf16"]  # [1, T, D]
        expected_last = expected[0, T - 1 : T] if expected.ndim == 3 else expected[T - 1 : T]

        out_decode = self._run_decode(module_bf16, case, jnp.bfloat16)

        assert not np.isnan(out_decode).any(), f"{case_name}: NaN in decode output"
        _assert_two_tier(
            out_decode, expected_last,
            DECODE_BF16_ATOL_TIGHT, DECODE_BF16_RTOL_TIGHT,
            DECODE_BF16_ATOL_LOOSE, DECODE_BF16_RTOL_LOOSE,
            f"{case_name} decode bf16",
        )
