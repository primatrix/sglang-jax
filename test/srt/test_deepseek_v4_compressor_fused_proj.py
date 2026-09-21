"""The HCA compressor's fused ``[Wkv|Wgate]^T`` projection is built once at load.

``fused_projection_weight`` rebuilt it in every HCA layer on every step (an f32->bf16
convert of ``wgate``, a concatenation and a transpose).  ``prepare_fused_projection``
materialises the same array once and ``weights()`` hands it to the HCA backend.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.kernels.hca.hca import fused_projection_weight
from sgl_jax.srt.models.deepseek_v4 import DeepseekV4Compressor

HIDDEN, D = 256, 64


def _compressor(mesh):
    config = SimpleNamespace(hidden_size=HIDDEN, rms_norm_eps=1e-6)
    with jax.set_mesh(mesh):
        comp = DeepseekV4Compressor(config, D, 128, jnp.bfloat16)
        rng = np.random.default_rng(0)
        comp.wkv.value = jnp.asarray(rng.standard_normal((D, HIDDEN)), jnp.bfloat16)
        comp.wgate.value = jnp.asarray(rng.standard_normal((D, HIDDEN)), jnp.float32)
    return comp


def test_prepared_projection_matches_per_step_fusion():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    comp = _compressor(mesh)
    assert comp.weights(None).fused is None  # nothing prepared: backend fuses per step
    comp.prepare_fused_projection(mesh)
    weights = comp.weights(None)
    assert weights.fused is not None and weights.fused.shape == (HIDDEN, 2 * D)
    assert weights.fused.dtype == jnp.bfloat16
    expected = fused_projection_weight(comp.wkv.value, comp.wgate.value)
    np.testing.assert_array_equal(np.asarray(weights.fused), np.asarray(expected))
    # Handing the prepared array back is the identity.
    assert fused_projection_weight(comp.wkv.value, comp.wgate.value, weights.fused) is weights.fused
    assert comp.ratio == 128


def test_xla_projection_matches_a_numpy_reference(monkeypatch):
    import numpy as np

    from sgl_jax.srt.kernels.hca.compressor import hca_project_fused_pallas
    from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule

    head = 128  # the kernel schedule needs lane-aligned heads
    sched = get_hca_kernel_schedule(
        "TPU7x", page_size=128, max_compressed_entries=64, local_heads=8, head_dim=head
    )
    rng = np.random.default_rng(3)
    tokens = 300
    x = jnp.asarray(rng.standard_normal((tokens, HIDDEN)), jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((HIDDEN, 2 * head)) * 0.02, jnp.bfloat16)
    ape = jnp.asarray(rng.standard_normal((128, head)), jnp.float32)
    pos = jnp.asarray(rng.integers(0, 5000, tokens), jnp.int32)
    monkeypatch.setenv("DSV4_HCA_PROJECT_XLA", "1")
    got = np.asarray(hca_project_fused_pallas(x, w, ape, pos, schedule=sched, head_dim=head))
    xf, wf, apef = (np.asarray(a.astype(jnp.float32)) for a in (x, w, ape))
    ref = (xf @ wf).reshape(tokens, 2, head)
    ref[:, 1] += apef[np.asarray(pos) % 128]
    assert got.shape == (tokens, 2, head)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-4)
