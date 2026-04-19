"""Equivalence unit tests for the fused QKV weight split logic in WeightLoader.

Tests verify that _split_fused_qkv_for_test (a pure static helper) produces
the same content in both transpose=True and transpose=False modes, and that
padded-V scale rows are handled correctly.

Run with:
    JAX_PLATFORMS=cpu python -m pytest tests/srt/models/test_mimo_v2_pro_qkv_split.py -xvs
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

# Constants matching real MiMo V2 Pro checkpoint
NUM_HEADS = 128
NUM_KV_HEADS = 8
HEAD_DIM_ORIG = 192
V_HEAD_DIM_ORIG = 128
HIDDEN = 6144
BLOCK = 128

Q_OUT = NUM_HEADS * HEAD_DIM_ORIG  # 24576
K_OUT = NUM_KV_HEADS * HEAD_DIM_ORIG  # 1536
V_OUT_COMPACT = NUM_KV_HEADS * V_HEAD_DIM_ORIG  # 1024
V_OUT_PADDED = NUM_KV_HEADS * HEAD_DIM_ORIG  # 1536 (scale only)

# Scale dimensions
Q_SCALE_ROWS = Q_OUT // BLOCK  # 192
K_SCALE_ROWS = K_OUT // BLOCK  # 12
V_SCALE_ROWS_PADDED = V_OUT_PADDED // BLOCK  # 12
V_SCALE_ROWS_COMPACT = V_OUT_COMPACT // BLOCK  # 8
SCALE_COLS = HIDDEN // BLOCK  # 48


def _get_helper():
    """Import the static helper under test.

    Uses importlib to load weight_utils directly, bypassing the
    sgl_jax.srt.utils __init__.py which requires heavy server deps (zmq,
    psutil, etc.) not needed for pure weight-split math tests. Also stubs out
    sgl_jax.srt.configs.model_config (needs transformers, which is heavyweight
    and may not be installed in a bare JAX-only environment).
    """
    import importlib.util
    import sys
    import types

    def _stub(name):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # Stub the heavy sub-packages so importing weight_utils.py doesn't fail.
    for name in [
        "sgl_jax",
        "sgl_jax.srt",
        "sgl_jax.srt.utils",
        "sgl_jax.srt.configs",
        "sgl_jax.srt.configs.model_config",
    ]:
        _stub(name)

    # Provide a minimal ModelConfig stub so the module-level import resolves.
    class _ModelConfig:
        pass

    sys.modules["sgl_jax.srt.configs.model_config"].ModelConfig = _ModelConfig

    weight_utils_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "python",
        "sgl_jax",
        "srt",
        "utils",
        "weight_utils.py",
    )
    spec = importlib.util.spec_from_file_location(
        "sgl_jax.srt.utils.weight_utils",
        weight_utils_path,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sgl_jax.srt.utils.weight_utils"] = mod
    spec.loader.exec_module(mod)

    WeightLoader = mod.WeightLoader
    assert hasattr(WeightLoader, "_split_fused_qkv_for_test"), (
        "WeightLoader._split_fused_qkv_for_test does not exist — "
        "add the @staticmethod helper to weight_utils.py"
    )
    return WeightLoader._split_fused_qkv_for_test


class TestSplitWeightTransposeFalse:
    """Test 1: transpose=False preserves Q/K/V layout from HF [out, in] tensor."""

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.q = rng.standard_normal((Q_OUT, HIDDEN)).astype(np.float32)
        self.k = rng.standard_normal((K_OUT, HIDDEN)).astype(np.float32)
        self.v = rng.standard_normal((V_OUT_COMPACT, HIDDEN)).astype(np.float32)
        self.fused_w = np.concatenate([self.q, self.k, self.v], axis=0)
        assert self.fused_w.shape == (
            Q_OUT + K_OUT + V_OUT_COMPACT,
            HIDDEN,
        ), f"fused_w shape mismatch: {self.fused_w.shape}"

    def test_split_weight_transpose_false_preserves_q_layout(self):
        helper = _get_helper()
        import jax.numpy as jnp

        fused_jnp = jnp.asarray(self.fused_w)
        q_out, k_out, v_out = helper(
            fused_jnp,
            transpose=False,
            is_scale=False,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim_orig=HEAD_DIM_ORIG,
            v_head_dim_orig=V_HEAD_DIM_ORIG,
            hidden=HIDDEN,
        )

        assert q_out.shape == (Q_OUT, HIDDEN), f"q shape: {q_out.shape}"
        assert k_out.shape == (K_OUT, HIDDEN), f"k shape: {k_out.shape}"
        assert v_out.shape == (V_OUT_COMPACT, HIDDEN), f"v shape: {v_out.shape}"

        assert np.array_equal(np.asarray(q_out), self.q), "Q content mismatch"
        assert np.array_equal(np.asarray(k_out), self.k), "K content mismatch"
        assert np.array_equal(np.asarray(v_out), self.v), "V content mismatch"


class TestSplitWeightTransposeTrue:
    """Test 2: transpose=True (legacy) produces transposed-equivalent content."""

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.q = rng.standard_normal((Q_OUT, HIDDEN)).astype(np.float32)
        self.k = rng.standard_normal((K_OUT, HIDDEN)).astype(np.float32)
        self.v = rng.standard_normal((V_OUT_COMPACT, HIDDEN)).astype(np.float32)
        fused_w = np.concatenate([self.q, self.k, self.v], axis=0)
        # Pre-transpose to [in, out] as done by the legacy transpose=True path
        self.fused_w_t = fused_w.T  # [HIDDEN, Q_OUT+K_OUT+V_OUT_COMPACT]

    def test_split_weight_transpose_true_legacy_equivalent(self):
        helper = _get_helper()
        import jax.numpy as jnp

        fused_jnp_t = jnp.asarray(self.fused_w_t)
        q_out, k_out, v_out = helper(
            fused_jnp_t,
            transpose=True,
            is_scale=False,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim_orig=HEAD_DIM_ORIG,
            v_head_dim_orig=V_HEAD_DIM_ORIG,
            hidden=HIDDEN,
        )

        # Shapes should be [in, out_*]
        assert q_out.shape == (HIDDEN, Q_OUT), f"q shape: {q_out.shape}"
        assert k_out.shape == (HIDDEN, K_OUT), f"k shape: {k_out.shape}"
        assert v_out.shape == (HIDDEN, V_OUT_COMPACT), f"v shape: {v_out.shape}"

        # Transposing back should equal the originals
        assert np.array_equal(np.asarray(q_out).T, self.q), "Q content mismatch after transpose"
        assert np.array_equal(np.asarray(k_out).T, self.k), "K content mismatch after transpose"
        assert np.array_equal(np.asarray(v_out).T, self.v), "V content mismatch after transpose"


class TestSplitScalePaddedV:
    """Test 3: transpose=False scale with padded V keeps all 12 V scale rows."""

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.q_scale = rng.standard_normal((Q_SCALE_ROWS, SCALE_COLS)).astype(np.float32)
        self.k_scale = rng.standard_normal((K_SCALE_ROWS, SCALE_COLS)).astype(np.float32)
        self.v_scale_padded = rng.standard_normal((V_SCALE_ROWS_PADDED, SCALE_COLS)).astype(
            np.float32
        )
        self.fused_s = np.concatenate([self.q_scale, self.k_scale, self.v_scale_padded], axis=0)
        assert self.fused_s.shape == (
            Q_SCALE_ROWS + K_SCALE_ROWS + V_SCALE_ROWS_PADDED,
            SCALE_COLS,
        ), f"fused_s shape: {self.fused_s.shape}"

    def test_split_scale_padded_v_keeps_all_v_rows(self):
        helper = _get_helper()
        import jax.numpy as jnp

        fused_jnp = jnp.asarray(self.fused_s)
        q_out, k_out, v_out = helper(
            fused_jnp,
            transpose=False,
            is_scale=True,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim_orig=HEAD_DIM_ORIG,
            v_head_dim_orig=V_HEAD_DIM_ORIG,
            hidden=HIDDEN,
        )

        assert q_out.shape == (Q_SCALE_ROWS, SCALE_COLS), f"q_scale shape: {q_out.shape}"
        assert k_out.shape == (K_SCALE_ROWS, SCALE_COLS), f"k_scale shape: {k_out.shape}"
        # V scale keeps ALL padded rows (12), not truncated to compact (8)
        assert v_out.shape == (
            V_SCALE_ROWS_PADDED,
            SCALE_COLS,
        ), f"v_scale shape: {v_out.shape}, expected ({V_SCALE_ROWS_PADDED}, {SCALE_COLS})"

        assert np.array_equal(np.asarray(q_out), self.q_scale), "Q scale content mismatch"
        assert np.array_equal(np.asarray(k_out), self.k_scale), "K scale content mismatch"
        assert np.array_equal(
            np.asarray(v_out), self.v_scale_padded
        ), "V padded scale content mismatch"


class TestSplitScaleTransposeTrue:
    """Test 4: transpose=True scale equivalent — same content in transposed layout."""

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.q_scale = rng.standard_normal((Q_SCALE_ROWS, SCALE_COLS)).astype(np.float32)
        self.k_scale = rng.standard_normal((K_SCALE_ROWS, SCALE_COLS)).astype(np.float32)
        self.v_scale_padded = rng.standard_normal((V_SCALE_ROWS_PADDED, SCALE_COLS)).astype(
            np.float32
        )
        fused_s = np.concatenate([self.q_scale, self.k_scale, self.v_scale_padded], axis=0)
        # Pre-transpose to [in_blocks, out_blocks]
        self.fused_s_t = fused_s.T  # [SCALE_COLS, Q_SCALE_ROWS+K_SCALE_ROWS+V_SCALE_ROWS_PADDED]

    def test_split_scale_transpose_true_legacy_equivalent(self):
        helper = _get_helper()
        import jax.numpy as jnp

        fused_jnp_t = jnp.asarray(self.fused_s_t)
        q_out, k_out, v_out = helper(
            fused_jnp_t,
            transpose=True,
            is_scale=True,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim_orig=HEAD_DIM_ORIG,
            v_head_dim_orig=V_HEAD_DIM_ORIG,
            hidden=HIDDEN,
        )

        # Shapes should be [in_blocks, out_blocks]
        assert q_out.shape == (SCALE_COLS, Q_SCALE_ROWS), f"q_scale shape: {q_out.shape}"
        assert k_out.shape == (SCALE_COLS, K_SCALE_ROWS), f"k_scale shape: {k_out.shape}"
        # V scale keeps ALL padded rows (12 in out dimension)
        assert v_out.shape == (
            SCALE_COLS,
            V_SCALE_ROWS_PADDED,
        ), f"v_scale shape: {v_out.shape}, expected ({SCALE_COLS}, {V_SCALE_ROWS_PADDED})"

        # Transposing back should equal originals
        assert np.array_equal(
            np.asarray(q_out).T, self.q_scale
        ), "Q scale content mismatch after transpose"
        assert np.array_equal(
            np.asarray(k_out).T, self.k_scale
        ), "K scale content mismatch after transpose"
        assert np.array_equal(
            np.asarray(v_out).T, self.v_scale_padded
        ), "V padded scale content mismatch after transpose"


class TestSplitWeightCompactVScale:
    """Test 5: is_scale=True with compact V scale (8 rows, not padded 12)."""

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.q_scale = rng.standard_normal((Q_SCALE_ROWS, SCALE_COLS)).astype(np.float32)
        self.k_scale = rng.standard_normal((K_SCALE_ROWS, SCALE_COLS)).astype(np.float32)
        # Compact V scale: 8 rows instead of 12
        self.v_scale_compact = rng.standard_normal((V_SCALE_ROWS_COMPACT, SCALE_COLS)).astype(
            np.float32
        )
        self.fused_s = np.concatenate([self.q_scale, self.k_scale, self.v_scale_compact], axis=0)
        # Q_SCALE_ROWS + K_SCALE_ROWS + V_SCALE_ROWS_COMPACT rows
        assert self.fused_s.shape == (
            Q_SCALE_ROWS + K_SCALE_ROWS + V_SCALE_ROWS_COMPACT,
            SCALE_COLS,
        ), f"fused_s shape: {self.fused_s.shape}"

    def test_split_weight_compact_v_scale_drops_pad(self):
        helper = _get_helper()
        import jax.numpy as jnp

        fused_jnp = jnp.asarray(self.fused_s)
        q_out, k_out, v_out = helper(
            fused_jnp,
            transpose=False,
            is_scale=True,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim_orig=HEAD_DIM_ORIG,
            v_head_dim_orig=V_HEAD_DIM_ORIG,
            hidden=HIDDEN,
        )

        assert q_out.shape == (Q_SCALE_ROWS, SCALE_COLS), f"q_scale shape: {q_out.shape}"
        assert k_out.shape == (K_SCALE_ROWS, SCALE_COLS), f"k_scale shape: {k_out.shape}"
        # Compact V: 8 rows, no padding
        assert v_out.shape == (
            V_SCALE_ROWS_COMPACT,
            SCALE_COLS,
        ), f"v_scale shape: {v_out.shape}, expected ({V_SCALE_ROWS_COMPACT}, {SCALE_COLS})"

        assert np.array_equal(np.asarray(q_out), self.q_scale), "Q scale content mismatch"
        assert np.array_equal(np.asarray(k_out), self.k_scale), "K scale content mismatch"
        assert np.array_equal(
            np.asarray(v_out), self.v_scale_compact
        ), "V compact scale content mismatch"


# ---------------------------------------------------------------------------
# Test 6: MiMoV2ProForCausalLM._create_layer_mappings FP8 mapping layout
# ---------------------------------------------------------------------------


def _get_mimo_pro_class():
    """Load MiMoV2ProForCausalLM with heavy deps stubbed out.

    We only need the class definition and its use of WeightMapping; all other
    sgl_jax layers/configs are replaced with lightweight stubs.
    """
    import importlib.util
    import sys
    import types

    def _stub(name, attrs=None):
        if name not in sys.modules:
            m = types.ModuleType(name)
            if attrs:
                for k, v in attrs.items():
                    setattr(m, k, v)
            sys.modules[name] = m
        return sys.modules[name]

    # --- transformers ---
    class _PretrainedConfig:
        pass

    transformers_mod = _stub("transformers")
    transformers_mod.PretrainedConfig = _PretrainedConfig

    # --- sgl_jax top-level and sub-packages ---
    for name in [
        "sgl_jax",
        "sgl_jax.srt",
        "sgl_jax.srt.configs",
        "sgl_jax.srt.configs.model_config",
        "sgl_jax.srt.layers",
        "sgl_jax.srt.layers.embeddings",
        "sgl_jax.srt.layers.fused_moe",
        "sgl_jax.srt.layers.layernorm",
        "sgl_jax.srt.layers.linear",
        "sgl_jax.srt.layers.logits_processor",
        "sgl_jax.srt.layers.moe",
        "sgl_jax.srt.layers.radix_attention",
        "sgl_jax.srt.mem_cache",
        "sgl_jax.srt.mem_cache.memory_pool",
        "sgl_jax.srt.model_executor",
        "sgl_jax.srt.model_executor.forward_batch_info",
        "sgl_jax.srt.utils",
        "sgl_jax.srt.utils.weight_utils",
    ]:
        _stub(name)

    # Provide minimal real-looking stubs for names used at class-definition time.
    class _ModelConfig:
        pass

    class _LinearBase:
        pass

    class _LogitsMetadata:
        pass

    class _LogitsProcessor:
        pass

    class _KVCache:
        pass

    class _ForwardBatch:
        pass

    class _RMSNorm:
        pass

    class _Embed:
        pass

    class _ParallelLMHead:
        pass

    def _get_rope(*a, **kw):
        pass

    class _FusedEPMoE:
        pass

    class _EPMoE:
        pass

    class _GateLogit:
        pass

    class _TopK:
        pass

    def _create_moe_weights_mapping(*a, **kw):
        return {}

    class _RadixAttention:
        pass

    sys.modules["sgl_jax.srt.configs.model_config"].ModelConfig = _ModelConfig
    sys.modules["sgl_jax.srt.layers.linear"].LinearBase = _LinearBase
    sys.modules["sgl_jax.srt.layers.logits_processor"].LogitsMetadata = _LogitsMetadata
    sys.modules["sgl_jax.srt.layers.logits_processor"].LogitsProcessor = _LogitsProcessor
    sys.modules["sgl_jax.srt.mem_cache.memory_pool"].KVCache = _KVCache
    sys.modules["sgl_jax.srt.model_executor.forward_batch_info"].ForwardBatch = _ForwardBatch
    sys.modules["sgl_jax.srt.layers.layernorm"].RMSNorm = _RMSNorm
    sys.modules["sgl_jax.srt.layers.embeddings"].Embed = _Embed
    sys.modules["sgl_jax.srt.layers.embeddings"].ParallelLMHead = _ParallelLMHead
    sys.modules["sgl_jax.srt.layers.embeddings"].get_rope = _get_rope
    sys.modules["sgl_jax.srt.layers.fused_moe"].FusedEPMoE = _FusedEPMoE
    sys.modules["sgl_jax.srt.layers.moe"].EPMoE = _EPMoE
    sys.modules["sgl_jax.srt.layers.moe"].GateLogit = _GateLogit
    sys.modules["sgl_jax.srt.layers.moe"].TopK = _TopK
    sys.modules["sgl_jax.srt.layers.moe"].create_moe_weights_mapping = _create_moe_weights_mapping
    sys.modules["sgl_jax.srt.layers.radix_attention"].RadixAttention = _RadixAttention

    # Load weight_utils first so WeightMapping is real.
    weight_utils_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "python",
        "sgl_jax",
        "srt",
        "utils",
        "weight_utils.py",
    )
    if "sgl_jax.srt.utils.weight_utils" not in sys.modules or not hasattr(
        sys.modules["sgl_jax.srt.utils.weight_utils"], "WeightMapping"
    ):
        spec = importlib.util.spec_from_file_location(
            "sgl_jax.srt.utils.weight_utils", weight_utils_path
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["sgl_jax.srt.utils.weight_utils"] = mod
        spec.loader.exec_module(mod)

    wu_mod = sys.modules["sgl_jax.srt.utils.weight_utils"]
    sys.modules["sgl_jax.srt.utils.weight_utils"].WeightLoader = wu_mod.WeightLoader
    sys.modules["sgl_jax.srt.utils.weight_utils"].WeightMapping = wu_mod.WeightMapping

    # Now load mimo_v2.py
    mimo_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "python",
        "sgl_jax",
        "srt",
        "models",
        "mimo_v2.py",
    )
    spec = importlib.util.spec_from_file_location("sgl_jax.srt.models.mimo_v2", mimo_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sgl_jax.srt.models.mimo_v2"] = mod
    spec.loader.exec_module(mod)

    return mod.MiMoV2ProForCausalLM


class TestProFP8MappingLayout:
    """Test 6: Pro FP8 fused QKV mapping must use transpose=False, sharding=("tensor", None).

    This test calls _create_layer_mappings on a stub MiMoV2ProForCausalLM instance
    (is_fp8=True) and asserts the weight mapping has the correct layout so that
    weight_q is stored in HF [out, in] order, matching QuantizedLinear's contract.
    """

    def test_pro_fp8_qkv_mapping_transpose_false_sharding_tensor_none(self):
        """FP8 fused QKV mapping: transpose=False, sharding=("tensor", None)."""
        from unittest.mock import patch

        MiMoV2ProForCausalLM = _get_mimo_pro_class()

        # Create a bare stub instance — no __init__, just set _quant_config.
        obj = object.__new__(MiMoV2ProForCausalLM)

        class _QuantConfig:
            is_static_checkpoint = True
            ignored_layers = []

        object.__setattr__(obj, "_quant_config", _QuantConfig())

        # Stub _create_common_layer_mappings at the class level so we don't
        # need self.config etc.  Use patch to avoid flax nnx __setattr__ guards.
        with patch.object(
            MiMoV2ProForCausalLM,
            "_create_common_layer_mappings",
            return_value={},
        ):
            mappings = obj._create_layer_mappings(0)

        key = "model.layers.0.self_attn.qkv_proj.weight"
        assert key in mappings, f"Expected key '{key}' in mappings, got: {list(mappings.keys())}"

        m = mappings[key]
        assert m.transpose is False, (
            f"Expected transpose=False for FP8 QKV mapping, got transpose={m.transpose!r}. "
            "The Pro FP8 fused QKV mapping must store weight_q in HF [out, in] layout "
            "to match QuantizedLinear's kernel contract."
        )
        assert m.sharding == ("tensor", None), (
            f"Expected sharding=('tensor', None) for FP8 QKV mapping, got sharding={m.sharding!r}. "
            "With HF [out, in] layout, the output axis (axis-0) maps to 'tensor' shard."
        )
