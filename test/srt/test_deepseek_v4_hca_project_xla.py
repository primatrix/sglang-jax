"""The HCA state projection's XLA path (DSV4_HCA_PROJECT_XLA=1) matches a NumPy reference."""

import jax.numpy as jnp

HIDDEN, D = 256, 64


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
