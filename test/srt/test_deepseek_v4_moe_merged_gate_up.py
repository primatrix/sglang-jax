"""EPMoE.prepare_merged_gate_up: one [w0 | w1] gmm gives the same output as two gmms."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.moe import EPMoE

sys.path.insert(0, str(Path(__file__).parent))
from test_moe_block_quant_e2e import (  # noqa: E402
    _create_test_mesh,
    _get_scale_shardings,
    _make_quant_config,
    _quantize_moe_weight,
)


def _build(quantized: bool):
    mesh = _create_test_mesh()
    hidden, inter, experts, topk = 256, 512, 4 * len(jax.devices()), 2
    quant = _make_quant_config(jnp.int8, None) if quantized else None
    with jax.set_mesh(mesh):
        moe = EPMoE(
            hidden_size=hidden,
            num_experts=experts,
            num_experts_per_tok=topk,
            ep_size=len(jax.devices()),
            mesh=mesh,
            intermediate_dim=inter,
            quantization_config=quant,
        )
    k0, k1, k2, kx, kt = jax.random.split(jax.random.PRNGKey(3), 5)
    w0 = jax.random.normal(k0, (experts, hidden, inter), jnp.bfloat16)
    w1 = jax.random.normal(k1, (experts, hidden, inter), jnp.bfloat16)
    wo = jax.random.normal(k2, (experts, inter, hidden), jnp.bfloat16)
    with jax.set_mesh(moe.moe_mesh):
        if quantized:
            (w0q, s0), (w1q, s1), (woq, so) = (
                _quantize_moe_weight(w, jnp.int8, None, "per_channel") for w in (w0, w1, wo)
            )
            sh0, sh1, sho = _get_scale_shardings("per_channel")
            moe.wi_0 = nnx.Param(w0q, out_sharding=P("expert", None, "tensor"))
            moe.wi_1 = nnx.Param(w1q, out_sharding=P("expert", None, "tensor"))
            moe.wo = nnx.Param(woq, out_sharding=P("expert", "tensor", None))
            del moe.wi_0_scale, moe.wi_1_scale, moe.wo_scale
            moe.wi_0_scale = nnx.Param(s0, out_sharding=sh0)
            moe.wi_1_scale = nnx.Param(s1, out_sharding=sh1)
            moe.wo_scale = nnx.Param(so, out_sharding=sho)
        else:
            moe.wi_0 = nnx.Param(w0, out_sharding=P("expert", None, "tensor"))
            moe.wi_1 = nnx.Param(w1, out_sharding=P("expert", None, "tensor"))
            moe.wo = nnx.Param(wo, out_sharding=P("expert", "tensor", None))
    x = jax.random.normal(kx, (24, hidden), jnp.bfloat16)
    weights = jnp.ones((24, topk), jnp.bfloat16) / topk
    ids = jax.random.randint(kt, (24, topk), 0, experts)
    return moe, x, weights, ids


@pytest.mark.parametrize("quantized", [False, True])
def test_merged_gate_up_matches_two_gmms(quantized):
    moe, x, weights, ids = _build(quantized)
    with jax.set_mesh(moe.moe_mesh):
        ref_dev = moe(x, weights, ids)
    ref = np.asarray(ref_dev).astype(np.float32)  # the output lives on the outer mesh
    with jax.set_mesh(moe.moe_mesh):
        moe.prepare_merged_gate_up()
        assert moe.merged_gate_up
        assert moe.wi_01.value.shape[-1] == 2 * moe.wi_0.value.shape[-1]
        if quantized:
            assert moe.wi_01_scale.value.shape[-1] == 2 * moe.wi_0.value.shape[-1]
        out_dev = moe(x, weights, ids)
        moe.prepare_merged_gate_up()  # idempotent
    out = np.asarray(out_dev).astype(np.float32)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, rtol=2e-2, atol=2e-2 * float(np.abs(ref).max()))
