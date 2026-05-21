"""Sequence-parallel sharding-contract tests for row-parallel projections and MoE modules.

Bucket planning decides whether a shape uses full-token or token-sharded
layouts. These tests verify that modules apply the explicit sharding contract
they are given.

The Grok wiring tests guard against regressions of the form "the flag is
plumbed through ``ServerArgs`` and the model config but doesn't actually
reach the projection that needs to scatter."
"""

import ast
import inspect
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.models.grok import Grok1Attention, Grok1DecoderLayer, Grok1MLP
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor
from sgl_jax.test.test_utils import CustomTestCase

_MESH = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
_TP_SIZE = _MESH.shape.get("tensor", 1)
_MIN_LOCAL = 128
_TOTAL_DEVICES = len(jax.devices())


def _spec_dim(sharding, dim):
    """Return the partition-axis name at `dim` (or None if unsharded).

    Tolerates ``PartitionSpec`` of any length: missing trailing entries are
    treated as None, matching JAX's own semantics.
    """
    spec = sharding.spec
    return spec[dim] if dim < len(spec) else None


def _as_fp32(x):
    """Cast to fp32 numpy for tolerance-based comparison.

    ``psum`` and ``psum_scatter`` are mathematically identical but XLA may
    pick different reduction trees, so per-element bf16 outputs can drift by
    a few ULPs. Compare in fp32 with rtol/atol sized to that drift.
    """
    return np.asarray(x).astype(np.float32)


def _to_sharding(mesh: Mesh, sharding: P | jax.sharding.Sharding | None):
    if sharding is None or isinstance(sharding, jax.sharding.Sharding):
        return sharding
    return NamedSharding(mesh, sharding)


class TestQuantizedLinearScatter(CustomTestCase):
    """``QuantizedLinear.output_sharding`` behavior."""

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_scatter_active_above_threshold(self):
        """At/above threshold, output is reduce-scattered on dim 0 over `tensor`."""
        batch = _TP_SIZE * _MIN_LOCAL  # exactly at threshold
        scatter_out, baseline_out = self._run_pair(batch)

        # Scatter path: dim 0 stripes across the data and tensor axes
        # (data axis is size 1 here so it's effectively just "tensor", but
        # the spec records both names — see TestDpSpComposition for the
        # observable dp>1 case).
        self.assertEqual(_spec_dim(scatter_out.sharding, 0), ("data", "tensor"))
        self.assertEqual(_spec_dim(baseline_out.sharding, 0), "data")

        # Same math, just different communication pattern. Tolerances cover
        # bf16 reduction-order drift over a 256-wide row-parallel sum (max
        # observed abs diff ~0.5 against mean |y| ~12).
        np.testing.assert_allclose(
            _as_fp32(scatter_out), _as_fp32(baseline_out), rtol=0.05, atol=1.0
        )

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_scatter_contract_applies_for_small_bucket(self):
        """Small buckets follow the explicit output sharding chosen upstream."""
        batch = _TP_SIZE * (_MIN_LOCAL // 2)
        scatter_out, _ = self._run_pair(batch)

        self.assertEqual(_spec_dim(scatter_out.sharding, 0), ("data", "tensor"))

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_scatter_disabled_when_dimension_is_none(self):
        """``output_sharding=None`` keeps the DP-only spec regardless of size."""
        batch = _TP_SIZE * _MIN_LOCAL  # would-be scatter size
        x_host, weight_q, weight_scale = _make_quant_linear_inputs(batch, in_dim=256, out_dim=512)

        with jax.set_mesh(_MESH):
            ql = _build_quant_linear(weight_q, weight_scale, _MESH, output_sharding=None)
            x = jax.device_put(x_host, NamedSharding(_MESH, P(None, "tensor")))
            out, _ = ql(x)

        self.assertEqual(_spec_dim(out.sharding, 0), "data")

    def _run_pair(self, batch: int):
        in_dim, out_dim = 256, 512
        x_host, weight_q, weight_scale = _make_quant_linear_inputs(batch, in_dim, out_dim)

        with jax.set_mesh(_MESH):
            ql_scatter = _build_quant_linear(
                weight_q,
                weight_scale,
                _MESH,
                output_sharding=NamedSharding(_MESH, P(("data", "tensor"), None)),
            )
            ql_baseline = _build_quant_linear(weight_q, weight_scale, _MESH, output_sharding=None)

            x = jax.device_put(x_host, NamedSharding(_MESH, P(None, "tensor")))
            out_scatter, _ = ql_scatter(x)
            out_baseline, _ = ql_baseline(x)

        return out_scatter, out_baseline


class TestLinearBaseScatter(CustomTestCase):
    """``LinearBase`` row-parallel output scatter behavior."""

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_row_parallel_scatter_active_above_threshold(self):
        batch = _TP_SIZE * _MIN_LOCAL
        in_dim, out_dim = 256, 512
        key = jax.random.PRNGKey(11)
        k_x, k_w = jax.random.split(key)
        x_host = jax.random.normal(k_x, (batch, in_dim), dtype=jnp.bfloat16)
        w_host = jax.random.normal(k_w, (in_dim, out_dim), dtype=jnp.bfloat16)

        with jax.set_mesh(_MESH):
            lin_scatter = _build_linear_base(
                w_host,
                _MESH,
                output_sharding=NamedSharding(_MESH, P(("data", "tensor"), None)),
            )
            lin_baseline = _build_linear_base(w_host, _MESH, output_sharding=None)
            x = jax.device_put(x_host, NamedSharding(_MESH, P(None, "tensor")))
            scatter_out, _ = lin_scatter(x)
            baseline_out, _ = lin_baseline(x)

        self.assertEqual(_spec_dim(scatter_out.sharding, 0), ("data", "tensor"))
        self.assertEqual(_spec_dim(baseline_out.sharding, 0), "data")
        np.testing.assert_allclose(
            _as_fp32(scatter_out), _as_fp32(baseline_out), rtol=0.05, atol=1.0
        )


def _make_quant_linear_inputs(batch: int, in_dim: int, out_dim: int):
    key = jax.random.PRNGKey(0)
    k_x, k_w = jax.random.split(key)
    x_host = jax.random.normal(k_x, (batch, in_dim), dtype=jnp.bfloat16)
    w_host = jax.random.normal(k_w, (out_dim, in_dim), dtype=jnp.bfloat16)
    weight_q, weight_scale = quantize_tensor(jnp.int8, w_host.astype(jnp.float32), axis=1)
    return x_host, weight_q, weight_scale


def _build_quant_linear(weight_q, weight_scale, mesh, *, output_sharding):
    output_sharding = _to_sharding(mesh, output_sharding)
    ql = QuantizedLinear(
        weight_q=weight_q,
        weight_scale=weight_scale,
        bias=None,
        activation_dtype=None,
        mesh=mesh,
        kernel_axes=("tensor", None),
        params_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        output_sharding=output_sharding,
    )
    # Row-parallel: weight is [out, in]; shard on the input axis.
    ql.weight_q = nnx.Param(weight_q, out_sharding=P(None, "tensor"))
    ql.weight_scale = nnx.Param(weight_scale, out_sharding=P(None))
    return ql


def _build_linear_base(weight, mesh, *, output_sharding):
    output_sharding = _to_sharding(mesh, output_sharding)
    lin = LinearBase(
        input_size=weight.shape[0],
        output_size=weight.shape[1],
        use_bias=False,
        mesh=mesh,
        kernel_axes=("tensor", None),
        params_dtype=jnp.bfloat16,
        output_sharding=output_sharding,
    )
    lin.weight = nnx.Param(weight, out_sharding=P("tensor", None))
    return lin


def _make_moe_mesh(ep_size: int, tp_size: int) -> Mesh:
    devices = np.array(jax.devices()[: ep_size * tp_size]).reshape(ep_size, tp_size)
    return Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def _make_moe_inputs(batch: int, hidden_size: int, num_experts: int):
    key = jax.random.PRNGKey(7)
    k_x, k_topk = jax.random.split(key)
    x = jax.random.normal(k_x, (batch, hidden_size), dtype=jnp.bfloat16)
    topk_weights = jnp.ones((batch, 1), dtype=jnp.bfloat16)
    topk_ids = jax.random.randint(k_topk, (batch, 1), 0, num_experts)
    return x, topk_weights, topk_ids


class TestEPMoESequenceParallel(CustomTestCase):
    """``EPMoE.output_sharding`` scatter behavior."""

    HIDDEN_SIZE = 512
    INTERMEDIATE_DIM = 1024
    NUM_EXPERTS = 4

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_seq_parallel_scatters_above_threshold(self):
        """With seq-parallel ON and a large enough batch, output is scattered
        on dim 0 over `tensor`, and matches the all-reduce baseline."""
        mesh = _make_moe_mesh(ep_size=1, tp_size=_TP_SIZE)
        batch = _TP_SIZE * _MIN_LOCAL
        x, topk_weights, topk_ids = _make_moe_inputs(batch, self.HIDDEN_SIZE, self.NUM_EXPERTS)

        with jax.set_mesh(mesh):
            moe_sp = self._build_moe(mesh, output_sharding=P(("data", "tensor"), None))
            moe_base = self._build_moe(mesh, output_sharding=None)

            with jax.set_mesh(moe_sp.moe_mesh):
                out_sp = moe_sp(x, topk_weights, topk_ids)
                out_base = moe_base(x, topk_weights, topk_ids)

        # Post-MoE reshard combines ``"data"`` (DP, size 1 here) with
        # ``"tensor"`` (SP) on dim 0. With dp=1 the data axis is just a
        # label, but the spec still records both names.
        self.assertEqual(_spec_dim(out_sp.sharding, 0), ("data", "tensor"))
        # SP off → DP-only spec on dim 0.
        self.assertEqual(_spec_dim(out_base.sharding, 0), "data")

        np.testing.assert_allclose(_as_fp32(out_sp), _as_fp32(out_base), rtol=0.1, atol=2048.0)

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_seq_parallel_contract_applies_for_small_bucket(self):
        """Small buckets follow the explicit output sharding chosen upstream."""
        mesh = _make_moe_mesh(ep_size=1, tp_size=_TP_SIZE)
        batch = _TP_SIZE
        x, topk_weights, topk_ids = _make_moe_inputs(batch, self.HIDDEN_SIZE, self.NUM_EXPERTS)

        with jax.set_mesh(mesh):
            moe_sp = self._build_moe(mesh, output_sharding=P(("data", "tensor"), None))
            with jax.set_mesh(moe_sp.moe_mesh):
                out_sp = moe_sp(x, topk_weights, topk_ids)

        self.assertEqual(_spec_dim(out_sp.sharding, 0), ("data", "tensor"))

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_seq_parallel_disabled_always_replicates(self):
        """``output_sharding=None`` keeps the DP-only spec on dim 0
        regardless of batch size — the pre-feature behavior."""
        mesh = _make_moe_mesh(ep_size=1, tp_size=_TP_SIZE)
        batch = _TP_SIZE * _MIN_LOCAL  # would otherwise scatter
        x, topk_weights, topk_ids = _make_moe_inputs(batch, self.HIDDEN_SIZE, self.NUM_EXPERTS)

        with jax.set_mesh(mesh):
            moe = self._build_moe(mesh, output_sharding=None)
            with jax.set_mesh(moe.moe_mesh):
                out = moe(x, topk_weights, topk_ids)

        self.assertEqual(_spec_dim(out.sharding, 0), "data")

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_seq_parallel_uses_model_tensor_axis_when_ep_equals_world(self):
        """EP-only MoE must still match attention o_proj's SP output contract."""
        mesh = _make_moe_mesh(ep_size=1, tp_size=_TP_SIZE)
        batch = _TP_SIZE * _MIN_LOCAL
        x, topk_weights, topk_ids = _make_moe_inputs(batch, self.HIDDEN_SIZE, _TP_SIZE)

        with jax.set_mesh(mesh):
            moe = EPMoE(
                hidden_size=self.HIDDEN_SIZE,
                num_experts=_TP_SIZE,
                num_experts_per_tok=1,
                ep_size=_TP_SIZE,
                mesh=mesh,
                intermediate_dim=self.INTERMEDIATE_DIM,
                quantization_config=None,
                output_sharding=NamedSharding(mesh, P(("data", "tensor"), None)),
            )
            with jax.set_mesh(moe.moe_mesh):
                out = moe(x, topk_weights, topk_ids)

        self.assertEqual(_spec_dim(out.sharding, 0), ("data", "tensor"))

    def _build_moe(self, mesh: Mesh, *, output_sharding: P | jax.sharding.Sharding | None) -> EPMoE:
        output_sharding = _to_sharding(mesh, output_sharding)
        return EPMoE(
            hidden_size=self.HIDDEN_SIZE,
            num_experts=self.NUM_EXPERTS,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=self.INTERMEDIATE_DIM,
            quantization_config=None,
            output_sharding=output_sharding,
        )


def _single_node_mesh() -> Mesh:
    """Mesh covering all visible devices on the ``tensor`` axis.

    Constructor wiring tests don't run forward, so a 1-device mesh is fine,
    but we use the full mesh so the test exercises the same sharding pathway
    used at runtime.
    """
    devices = np.array(jax.devices()).reshape(1, -1)
    return Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


class TestGrokLayerSequenceParallelWiring(CustomTestCase):
    """Verify ``output_sharding`` reaches the projection that needs it.

    These were the silent-failure modes called out in review: the flag exists
    on ``ServerArgs`` and propagates onto the model config, but if a layer
    forgets to thread it into its row-parallel projection, sequence parallel
    becomes a no-op for that layer.
    """

    def test_grok1_mlp_wires_scatter_when_enabled(self):
        mesh = _single_node_mesh()
        with jax.set_mesh(mesh):
            mlp = Grok1MLP(
                hidden_size=128,
                intermediate_size=256,
                layer_id=0,
                mesh=mesh,
                enable_sequence_parallel=True,
            )
        # Only down_proj (row-parallel) should scatter; gate/up are column-parallel.
        self.assertEqual(mlp.down_proj.output_sharding.spec, P(("data", "tensor"), None))
        self.assertIsNone(mlp.gate_proj.output_sharding)
        self.assertIsNone(mlp.up_proj.output_sharding)

    def test_grok1_mlp_leaves_legacy_flag_unset_by_default(self):
        mesh = _single_node_mesh()
        with jax.set_mesh(mesh):
            mlp = Grok1MLP(hidden_size=128, intermediate_size=256, layer_id=0, mesh=mesh)
        self.assertIsNone(mlp.down_proj.output_sharding)

    def test_grok1_attention_wires_scatter_when_enabled(self):
        mesh = _single_node_mesh()
        cfg = SimpleNamespace(head_dim=64)
        with jax.set_mesh(mesh):
            attn = Grok1Attention(
                config=cfg,
                hidden_size=128,
                num_heads=2,
                num_kv_heads=2,
                mesh=mesh,
                enable_sequence_parallel=True,
            )
        self.assertEqual(attn.o_proj.output_sharding.spec, P(("data", "tensor"), None))
        # q/k/v projections are column-parallel; they don't scatter on output.
        self.assertIsNone(attn.q_proj.output_sharding)
        self.assertIsNone(attn.k_proj.output_sharding)
        self.assertIsNone(attn.v_proj.output_sharding)

    def test_grok1_attention_leaves_legacy_flag_unset_by_default(self):
        mesh = _single_node_mesh()
        cfg = SimpleNamespace(head_dim=64)
        with jax.set_mesh(mesh):
            attn = Grok1Attention(
                config=cfg,
                hidden_size=128,
                num_heads=2,
                num_kv_heads=2,
                mesh=mesh,
            )
        self.assertIsNone(attn.o_proj.output_sharding)

    def test_grok1_decoder_layer_threads_flag_to_attention_and_mlp(self):
        """Static check: ``Grok1DecoderLayer.__init__`` forwards
        ``config.enable_sequence_parallel`` to both the ``Grok1Attention`` and
        the ``Grok1MLP`` constructors.

        Done via AST inspection instead of full instantiation: building a
        ``Grok1DecoderLayer`` requires a complete MoE setup (gate, experts,
        weight allocation) which is far heavier than what this assertion
        needs. The bug class we're guarding against — forgetting to pass the
        kwarg through — is structural and visible in the source.
        """
        src = inspect.getsource(Grok1DecoderLayer.__init__)
        tree = ast.parse(src.lstrip())

        kwargs_by_callee: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"Grok1Attention", "Grok1MLP"}
            ):
                kwargs_by_callee[node.func.id] = [kw.arg for kw in node.keywords]

        self.assertIn(
            "Grok1Attention",
            kwargs_by_callee,
            "Grok1DecoderLayer.__init__ no longer instantiates Grok1Attention",
        )
        self.assertIn(
            "enable_sequence_parallel",
            kwargs_by_callee["Grok1Attention"],
            "Grok1DecoderLayer must thread enable_sequence_parallel into Grok1Attention; "
            "without it, attention's o_proj never reduce-scatters even with the flag set.",
        )
        # Grok1MLP only appears in the residual-MoE branch; assert only if present.
        if "Grok1MLP" in kwargs_by_callee:
            self.assertIn(
                "enable_sequence_parallel",
                kwargs_by_callee["Grok1MLP"],
                "Grok1DecoderLayer must thread enable_sequence_parallel into Grok1MLP.",
            )


def _make_dp_tp_mesh(dp_size: int, tp_size: int) -> Mesh:
    """Make a ``(dp_size, tp_size)`` mesh with axis names ``("data", "tensor")``.

    Used to exercise the DP+SP composition path where the scatter dim must
    combine ``"data"`` (from DP) with ``"tensor"`` (from SP) instead of
    clobbering the former.
    """
    devices = np.array(jax.devices()[: dp_size * tp_size]).reshape(dp_size, tp_size)
    return Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


class TestDpSpComposition(CustomTestCase):
    """Verify scatter dim stripes across BOTH ``data`` and ``tensor`` under DP+SP.

    The dp_size=1 tests above can't catch the wrong code path
    ``out_specs[scatter_dim] = input_axis`` (clobbers ``"data"``) vs the
    correct one ``out_specs[scatter_dim] = ("data", input_axis)`` (combines).
    With dp_size>1 the difference becomes observable on the output sharding.
    """

    HIDDEN_SIZE = 512
    INTERMEDIATE_DIM = 1024
    NUM_EXPERTS = 4

    @unittest.skipIf(_TOTAL_DEVICES < 8, "Needs >=8 devices for dp=2, tp=4.")
    def test_quantized_linear_scatter_combines_data_and_tensor(self):
        """SP firing under DP shards dim 0 across ``("data", "tensor")``."""
        mesh = _make_dp_tp_mesh(dp_size=2, tp_size=4)
        # Per-device local size after both splits must be >= _MIN_LOCAL for
        # ``should_scatter`` to fire, so batch >= dp * tp * _MIN_LOCAL.
        batch = 2 * 4 * _MIN_LOCAL
        in_dim, out_dim = 256, 512
        x_host, weight_q, weight_scale = _make_quant_linear_inputs(batch, in_dim, out_dim)

        with jax.set_mesh(mesh):
            ql_scatter = _build_quant_linear(
                weight_q,
                weight_scale,
                mesh,
                output_sharding=NamedSharding(mesh, P(("data", "tensor"), None)),
            )
            ql_baseline = _build_quant_linear(weight_q, weight_scale, mesh, output_sharding=None)
            x = jax.device_put(x_host, NamedSharding(mesh, P("data", "tensor")))
            out_scatter, _ = ql_scatter(x)
            out_baseline, _ = ql_baseline(x)

        # The whole point of this test: dim 0 should stripe across BOTH axes,
        # not have "data" replaced by "tensor".
        self.assertEqual(_spec_dim(out_scatter.sharding, 0), ("data", "tensor"))
        self.assertEqual(_spec_dim(out_baseline.sharding, 0), "data")

        np.testing.assert_allclose(
            _as_fp32(out_scatter), _as_fp32(out_baseline), rtol=0.05, atol=1.0
        )

    @unittest.skipIf(_TOTAL_DEVICES < 8, "Needs >=8 devices for dp=2, tp=4.")
    def test_epmoe_seq_parallel_combines_data_and_tensor(self):
        """SP firing inside EPMoE under DP preserves the scatter post-reshard.

        With ep_size=1, ``EPMoE.tp_size`` collapses to ``world_size`` (= 8).
        The post-shard_map reshard must target ``P(("data", "tensor"), None)``
        on the original mesh; otherwise the SP scatter is all-gathered away
        at the MoE→next-layer seam.
        """
        mesh = _make_dp_tp_mesh(dp_size=2, tp_size=4)
        # EPMoE.tp_size = world_size / ep_size = 8 → SP threshold = 8 * 128.
        batch = 8 * _MIN_LOCAL
        x, topk_weights, topk_ids = _make_moe_inputs(batch, self.HIDDEN_SIZE, self.NUM_EXPERTS)

        with jax.set_mesh(mesh):
            moe_sp = self._build_moe(mesh, output_sharding=P(("data", "tensor"), None))
            moe_base = self._build_moe(mesh, output_sharding=None)
            with jax.set_mesh(moe_sp.moe_mesh):
                out_sp = moe_sp(x, topk_weights, topk_ids)
                out_base = moe_base(x, topk_weights, topk_ids)

        self.assertEqual(_spec_dim(out_sp.sharding, 0), ("data", "tensor"))
        self.assertEqual(_spec_dim(out_base.sharding, 0), "data")

        # See TestEPMoESequenceParallel for the noise-floor rationale (atol
        # sized to bf16 reduction-order drift on tens-of-thousands magnitudes).
        np.testing.assert_allclose(_as_fp32(out_sp), _as_fp32(out_base), rtol=0.1, atol=2048.0)

    def _build_moe(self, mesh: Mesh, *, output_sharding: P | jax.sharding.Sharding | None) -> EPMoE:
        output_sharding = _to_sharding(mesh, output_sharding)
        return EPMoE(
            hidden_size=self.HIDDEN_SIZE,
            num_experts=self.NUM_EXPERTS,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=self.INTERMEDIATE_DIM,
            quantization_config=None,
            output_sharding=output_sharding,
        )


if __name__ == "__main__":
    unittest.main()
