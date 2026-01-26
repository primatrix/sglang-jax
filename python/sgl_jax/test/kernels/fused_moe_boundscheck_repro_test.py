# Minimal TPU repro harness for a Mosaic DMA BoundsCheck crash in fused MoE.
#
# Run on TPU (example):
#   PYTHONPATH=python python -m sgl_jax.test.kernels.fused_moe_boundscheck_repro_test
#
# Optional knobs:
#   SGLANG_FUSED_MOE_BOUNDSCHECK_SWEEP=1
#     Run a small matrix of control-variable variants (shared expert on/off, topk prepad on/off,
#     token sizes).
#   SGLANG_FUSED_MOE_BOUNDSCHECK_NUM_TOKENS=512
#   SGLANG_FUSED_MOE_BOUNDSCHECK_NUM_EXPERTS=128

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from sgl_jax.srt.kernels.fused_moe.v1.kernel import FusedMoEBlockConfig, fused_ep_moe
from sgl_jax.test.test_utils import create_device_mesh

jax.config.parse_flags_with_absl()


def _align_to(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def _gen_sparse_topk(key: jax.Array, num_tokens: int, num_experts: int, top_k: int):
    token_keys = jax.random.split(key, num_tokens)
    topk_ids = jax.vmap(lambda kk: jax.random.permutation(kk, num_experts)[:top_k])(token_keys)
    topk_ids = topk_ids.astype(jnp.int32)
    topk_weights = jax.random.uniform(
        jax.random.fold_in(key, 1), (num_tokens, top_k), minval=0.0, maxval=1.0, dtype=jnp.float32
    )
    topk_weights = topk_weights / jnp.maximum(1e-6, jnp.sum(topk_weights, axis=-1, keepdims=True))
    return topk_ids, topk_weights


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class FusedMoeBoundsCheckReproTest(jtu.JaxTestCase):
    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("TPU-only repro")
        # Mesh axes are ("data", "tensor"), matching fused_ep_moe defaults.
        self.mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

    def _run_case(
        self,
        *,
        num_tokens: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        has_shared_expert: bool,
        prepad_topk: bool,
    ):
        ep_size = int(self.mesh.shape["data"]) * int(self.mesh.shape["tensor"])
        num_tokens = _align_to(num_tokens, ep_size)
        num_experts = _align_to(num_experts, ep_size)

        dtype = jnp.bfloat16
        key = jax.random.key(0)
        k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

        tokens = (
            jax.random.normal(k0, (num_tokens, hidden_size), dtype=jnp.float32).astype(dtype) / 10
        )
        w1 = (
            jax.random.normal(
                k1, (num_experts, hidden_size, intermediate_size), dtype=jnp.float32
            ).astype(dtype)
            / 10
        )
        w2 = (
            jax.random.normal(
                k2, (num_experts, intermediate_size, hidden_size), dtype=jnp.float32
            ).astype(dtype)
            / 10
        )
        w3 = (
            jax.random.normal(
                k3, (num_experts, hidden_size, intermediate_size), dtype=jnp.float32
            ).astype(dtype)
            / 10
        )

        topk_ids, topk_weights = _gen_sparse_topk(k4, num_tokens, num_experts, top_k)
        if prepad_topk:
            padded_top_k = _align_to(top_k, 128)
            topk_ids = jnp.pad(topk_ids, ((0, 0), (0, padded_top_k - top_k)))
            topk_weights = jnp.pad(topk_weights, ((0, 0), (0, padded_top_k - top_k)))

        w1_shared = w2_shared = w3_shared = None
        if has_shared_expert:
            se_intermediate_size = 512
            w1_shared = (
                jax.random.normal(
                    k5, (hidden_size, se_intermediate_size), dtype=jnp.float32
                ).astype(dtype)
                / 10
            )
            w2_shared = (
                jax.random.normal(
                    jax.random.fold_in(k5, 1),
                    (se_intermediate_size, hidden_size),
                    dtype=jnp.float32,
                ).astype(dtype)
                / 10
            )
            w3_shared = (
                jax.random.normal(
                    jax.random.fold_in(k5, 2),
                    (hidden_size, se_intermediate_size),
                    dtype=jnp.float32,
                ).astype(dtype)
                / 10
            )

        block_config = FusedMoEBlockConfig(
            bt=32,
            bts=32,
            btc=32,
            bf=512,
            bfc=512,
            bd1=1024,
            bd1c=1024,
            bd2=1024,
            bd2c=1024,
            bse=512,
        )

        with self.mesh:
            out = fused_ep_moe(
                self.mesh,
                tokens,
                w1,
                w2,
                w3,
                gating_output=None,
                top_k=top_k,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                block_config=block_config,
                w1_shared=w1_shared,
                w2_shared=w2_shared,
                w3_shared=w3_shared,
            )
            out.block_until_ready()

            # Make sure we actually produced concrete output on all devices.
            host = np.asarray(jax.device_get(out))
            self.assertEqual(host.shape, (num_tokens, hidden_size))

    @parameterized.named_parameters(
        dict(
            testcase_name="tok512_shared_prepad0",
            num_tokens=int(os.getenv("SGLANG_FUSED_MOE_BOUNDSCHECK_NUM_TOKENS", "512")),
            num_experts=int(os.getenv("SGLANG_FUSED_MOE_BOUNDSCHECK_NUM_EXPERTS", "128")),
            hidden_size=1024,
            intermediate_size=512,
            top_k=8,
            has_shared_expert=True,
            prepad_topk=False,
        ),
    )
    def test_boundscheck_repro(self, **kwargs):
        sweep = os.getenv("SGLANG_FUSED_MOE_BOUNDSCHECK_SWEEP", "0") == "1"
        cases = [kwargs]
        if sweep:
            base = kwargs.copy()
            cases = []
            for num_tokens in (256, base["num_tokens"], 1024):
                for has_shared_expert in (False, True):
                    for prepad_topk in (False, True):
                        c = base.copy()
                        c["num_tokens"] = int(num_tokens)
                        c["has_shared_expert"] = bool(has_shared_expert)
                        c["prepad_topk"] = bool(prepad_topk)
                        cases.append(c)

        for case in cases:
            with self.subTest(
                **{k: case[k] for k in ("num_tokens", "has_shared_expert", "prepad_topk")}
            ):
                self._run_case(**case)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
