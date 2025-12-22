# Adapted from https://github.com/vllm-project/tpu-inference/blob/main/tests/kernels/fused_moe_v1_test.py
# Copyright 2025 The tpu-inference Authors. All rights reserved.
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.sharding import Mesh

from sgl_jax.srt.kernels.fused_moe.v1.kernel import fused_ep_moe, ref_moe

jax.config.parse_flags_with_absl()


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def gen_moe_inputs(
    dtype,
    top_k,
    num_experts,
    hidden_size,
    intermediate_size,
    num_tokens,
    *,
    seed=1234,
    has_bias=False,
):
    key = jax.random.key(seed)
    k0, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)

    # Use 32.0 to prevent BF16 overflow
    SCALE_FACTOR = 32.0

    a = (
        jax.random.normal(k0, (num_tokens, hidden_size), dtype=jnp.float32).astype(dtype)
        / SCALE_FACTOR
    )

    w1 = (
        jax.random.normal(
            k1,
            (num_experts, 2, hidden_size, intermediate_size),
            dtype=jnp.float32,
        )
        / SCALE_FACTOR
    ).astype(dtype)
    w2 = (
        jax.random.normal(k2, (num_experts, intermediate_size, hidden_size), dtype=jnp.float32)
        / SCALE_FACTOR
    ).astype(dtype)

    if has_bias:
        b1 = (
            jax.random.normal(k3, (num_experts, 2, 1, intermediate_size), dtype=jnp.float32)
            / SCALE_FACTOR
        ).astype(dtype)
        b2 = (
            jax.random.normal(k4, (num_experts, 1, hidden_size), dtype=jnp.float32) / SCALE_FACTOR
        ).astype(dtype)
    else:
        b1 = b2 = None

    gating_output = (
        jax.random.normal(k5, (num_tokens, num_experts), dtype=jnp.float32)
        + jnp.arange(num_tokens * num_experts, dtype=jnp.float32).reshape(num_tokens, num_experts)
        / 100
    )

    top_k_indices = jax.random.randint(
        k6, (num_tokens, top_k), minval=0, maxval=num_experts - 1, dtype=jnp.int32
    )

    one_hot = (
        jnp.sum(
            jax.nn.one_hot(top_k_indices, num_experts, dtype=jnp.float32),
            axis=1,
        )
        * 10.0
    )

    gating_output = gating_output + one_hot

    return a, w1, w2, b1, b2, gating_output.astype(dtype)


def sub_channel_quantize(x, quant_dtype, wsz=256):
    """Quantizes x with sub-channel quantization on the 2nd minor."""
    if jnp.issubdtype(quant_dtype, jnp.floating):
        dtype_info = jnp.finfo(quant_dtype)
    else:
        dtype_info = jnp.iinfo(quant_dtype)
    dtype_max = float(dtype_info.max)
    w_lst, scale_lst = [], []
    assert len(x.shape) >= 2
    assert x.shape[-2] % wsz == 0
    for i in range(0, x.shape[-2], wsz):
        y = x[..., i : i + wsz, :]
        abs_max = jnp.abs(y).max(axis=-2, keepdims=True)
        scale = (abs_max / dtype_max).astype(jnp.float32)
        w = (y / scale).astype(quant_dtype)
        w_lst.append(w)
        scale_lst.append(scale)
    return jnp.concat(w_lst, axis=-2), jnp.expand_dims(jnp.concat(scale_lst, axis=-2), axis=-2)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class MoEKernelTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        self.mesh_devices = sorted(
            jax.devices(),
            key=lambda x: (
                x.coords[0],
                (-1 if x.coords[0] % 2 else 1) * x.coords[1],
            ),
        )
        self.mesh = Mesh(np.array(self.mesh_devices).reshape(-1, 1), axis_names=("tensor", "data"))

    def _test_moe(
        self,
        dtype,
        top_k,
        num_experts,
        hidden_size,
        intermediate_size,
        num_tokens,
        seed,
        renormalize_topk_logits,
        bt,
        bf,
        bd1,
        bd2,
        btc,
        bfc,
        bd1c,
        bd2c,
        act_fn="silu",
        w_dtype=None,
        subc_quant_wsz=None,
        has_bias=False,
        bias=None,
        atol=2e-1,
        rtol=2e-1,
        use_grouped_topk=False,
        num_groups=1,
        top_k_groups=1,
    ):
        a, w1, w2, b1, b2, gating_output = gen_moe_inputs(
            dtype,
            top_k,
            num_experts,
            hidden_size,
            intermediate_size,
            num_tokens,
            seed=seed,
            has_bias=has_bias,
        )
        w1_scale = None
        w2_scale = None
        if w_dtype is not None:
            if subc_quant_wsz is None:
                subc_quant_wsz = 256
            w1, w1_scale = sub_channel_quantize(w1, w_dtype, subc_quant_wsz)
            w2, w2_scale = sub_channel_quantize(w2, w_dtype, subc_quant_wsz)

        actual = fused_ep_moe(
            mesh=self.mesh,
            tokens=a,
            w1=w1,
            w2=w2,
            gating_output=gating_output,
            top_k=top_k,
            renormalize_topk_logits=renormalize_topk_logits,
            act_fn=act_fn,
            subc_quant_wsz=subc_quant_wsz,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            b1=b1,
            b2=b2,
            bias=bias,  # Pass routing bias
            bt=bt,
            bf=bf,
            bd1=bd1,
            bd2=bd2,
            btc=btc,
            bfc=bfc,
            bd1c=bd1c,
            bd2c=bd2c,
            ep_axis_name="tensor",
            use_grouped_topk=use_grouped_topk,
            num_groups=num_groups,
            top_k_groups=top_k_groups,
        )

        expected = ref_moe(
            a,
            w1,
            w2,
            gating_output,
            top_k,
            b1=b1,
            b2=b2,
            bias=bias,  # Pass routing bias
            renormalize_topk_logits=renormalize_topk_logits,
            act_fn=act_fn,
            subc_quant_wsz=subc_quant_wsz,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            use_grouped_topk=use_grouped_topk,
            num_groups=num_groups,
            top_k_groups=top_k_groups,
        )
        self.assertAllClose(actual, expected, atol=atol, rtol=rtol)

    @parameterized.product(
        renormalize_topk_logits=[True, False],
    )
    def test_basic(self, renormalize_topk_logits):
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=renormalize_topk_logits,
            bt=32,
            bf=1024,
            bd1=1024,
            bd2=1024,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
        )

    @parameterized.named_parameters(
        ("single_group_select", 4, 1),  # 128 experts / 4 groups = 32 experts/group.
        ("multi_group_select", 8, 2),  # 128 experts / 8 groups = 16 experts/group.
    )
    def test_grouped_topk(self, num_groups, top_k_groups):
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32

        assert num_experts % num_groups == 0

        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=True,
            bt=32,
            bf=1024,
            bd1=1024,
            bd2=1024,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
            use_grouped_topk=True,
            num_groups=num_groups,
            top_k_groups=top_k_groups,
        )

    def test_routing_bias_and_group_logic(self):
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32

        bias = jnp.zeros((num_experts,), dtype=jnp.float32)
        bias = bias.at[0].set(100.0)
        num_groups = 4
        bias = bias.at[32:64].add(50.0)

        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=True,
            bt=32,
            bf=1024,
            bd1=1024,
            bd2=1024,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
            use_grouped_topk=True,
            num_groups=num_groups,
            top_k_groups=2,
            bias=bias,
        )

    @parameterized.product(
        act_fn=["silu", "gelu", "swigluoai"],
    )
    def test_activation(self, act_fn):
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=True,
            act_fn=act_fn,
            bt=32,
            bf=512,
            bd1=512,
            bd2=512,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
        )

    def test_benchmark_qwen_235(self):
        num_experts = 128
        top_k = 8
        hidden_size = 4096
        intermediate_size = 1536
        dtype = jnp.bfloat16
        num_tokens = 8 * 64
        seed = 54321
        renormalize_topk_logits = True
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=seed,
            renormalize_topk_logits=renormalize_topk_logits,
            bt=64,
            bf=768,
            bd1=2048,
            bd2=2048,
            btc=64,
            bfc=768,
            bd1c=2048,
            bd2c=2048,
            act_fn="silu",
            atol=5e-2,
            rtol=5e-2,
        )

    def test_benchmark_qwen_30b_a3b(self):
        num_experts = 128
        top_k = 8
        hidden_size = 2048
        intermediate_size = 768
        dtype = jnp.bfloat16
        num_tokens = 512
        seed = 54321
        renormalize_topk_logits = True
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=seed,
            renormalize_topk_logits=renormalize_topk_logits,
            bt=16,
            bf=384,
            bd1=512,
            bd2=512,
            btc=16,
            bfc=384,
            bd1c=256,
            bd2c=256,
            act_fn="silu",
            atol=5e-2,
            rtol=5e-2,
        )

    @parameterized.product(
        w_dtype=[jnp.int8, jnp.float8_e5m2, jnp.float4_e2m1fn],
    )
    def test_sub_channel_quantization(self, w_dtype):
        if w_dtype in (
            jnp.float8_e5m2,
            jnp.float4_e2m1fn,
        ) and not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=False,
            w_dtype=w_dtype,
            subc_quant_wsz=256,
            bt=32,
            bf=1024,
            bd1=1024,
            bd2=1024,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
        )

    def test_ffn_bias(self):
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=False,
            has_bias=True,
            bt=32,
            bf=512,
            bd1=512,
            bd2=512,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
