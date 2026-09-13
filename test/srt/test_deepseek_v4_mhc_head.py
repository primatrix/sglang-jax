"""TPU checks for the model-level mHC head collapse.

``DeepseekV4Model`` used to call ``collapse_head_reference`` directly, which meant
the Pallas kernel was never reached, the head parameters were never validated, and
bf16 streams came back as float32.  The model now goes through
``DeepseekV4MHC.collapse_head`` behind a ``shard_map``, so the two things worth
checking on real hardware are that the kernel path agrees with the reference and
that the sharding wrapper does not change the dtype, shape or values.

Kernel-level correctness lives in ``test/srt/kernels/mhc/test_mhc.py``; this file is
about the model's use of it.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.deepseek_v4_mhc import DeepseekV4MHC, collapse_head_reference
from sgl_jax.srt.models.deepseek_v4 import DeepseekV4Model

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="mHC kernels require real Mosaic lowering"
)

# DeepSeek-V4-Flash shipped configuration.
HC, HIDDEN, ITERS, EPS = 4, 4096, 20, 1e-6
# Everything downstream of the bf16 gate projection carries bf16 error even though
# the parameters are stored fp32; the repo convention for bf16 kernels.
PROJECTED = {"rtol": 2e-2, "atol": 1e-2}

# Powers of two around the tuned tile sizes, plus counts that are a multiple of no
# block size, which exercise the kernel's pad-then-slice-back path.
TOKENS = [1, 127, 128, 1000, 2048]


def _config(hidden=HIDDEN, hc=HC):
    return SimpleNamespace(
        hc_mult=hc,
        hc_sinkhorn_iters=ITERS,
        hc_eps=EPS,
        rms_norm_eps=EPS,
        hidden_size=hidden,
    )


def _params(shape, hc=HC, hidden=HIDDEN, seed=0):
    """bf16 streams of the given leading ``shape``, plus fp32 head parameters."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    return {
        "streams": (jax.random.normal(keys[0], (*shape, hc, hidden), jnp.float32) * 0.1).astype(
            jnp.bfloat16
        ),
        "head_fn": jax.random.normal(keys[1], (hc, hc * hidden), jnp.float32) * 0.01,
        "head_base": jax.random.normal(keys[2], (hc,), jnp.float32) * 0.05,
        "head_scale": jnp.asarray([0.8], jnp.float32),
    }


def _head_args(params):
    """``collapse_head`` order: base before scale, unlike the reference function."""
    return params["head_fn"], params["head_base"], params["head_scale"]


def _stub(mhc, mesh, params):
    """The four attributes ``DeepseekV4Model._collapse_head`` reads.

    Calling the unbound method against a stub tests the shipped code rather than a
    copy of it, without building a whole model and its memory pools.
    """
    return SimpleNamespace(
        mhc=mhc,
        mesh=mesh,
        hc_head_fn=SimpleNamespace(value=params["head_fn"]),
        hc_head_base=SimpleNamespace(value=params["head_base"]),
        hc_head_scale=SimpleNamespace(value=params["head_scale"]),
    )


def _mesh():
    """A single-chip 1x1 mesh; this file does not cover multi-device shapes."""
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


# -- the kernel path -----------------------------------------------------------


def test_auto_backend_is_pallas():
    """Without this the whole fix is inert: the model would still run the reference."""
    assert DeepseekV4MHC(_config()).backend == "pallas"


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_output_follows_stream_dtype(dtype):
    mhc = DeepseekV4MHC(_config())
    params = _params((128,))
    out = mhc.collapse_head(params["streams"].astype(dtype), *_head_args(params))
    assert out.dtype == dtype
    assert out.shape == (128, HIDDEN)


@pytest.mark.parametrize("n", TOKENS)
def test_matches_reference(n):
    mhc = DeepseekV4MHC(_config())
    params = _params((n,))
    got = mhc.collapse_head(params["streams"], *_head_args(params))
    want = collapse_head_reference(
        params["streams"],
        params["head_fn"],
        params["head_scale"],
        params["head_base"],
        norm_eps=EPS,
        hc_eps=EPS,
    ).astype(jnp.bfloat16)
    np.testing.assert_allclose(
        np.asarray(got, np.float32), np.asarray(want, np.float32), **PROJECTED
    )


def test_keeps_leading_dimensions():
    """The kernel flattens ``[..., hc, d]`` internally and must restore the outer shape."""
    mhc = DeepseekV4MHC(_config())
    params = _params((3, 128))
    out = mhc.collapse_head(params["streams"], *_head_args(params))
    assert out.shape == (3, 128, HIDDEN)


@pytest.mark.parametrize("bad", ["dtype", "fn_shape", "scale_shape"])
def test_check_params_rejects_bad_head_params(bad):
    """``check_params`` must reject before lowering, not crash inside Mosaic."""
    mhc = DeepseekV4MHC(_config())
    params = _params((128,))
    fn, base, scale = _head_args(params)
    if bad == "dtype":
        fn = fn.astype(jnp.bfloat16)
    elif bad == "fn_shape":
        fn = fn[:, : HC * HIDDEN // 2]
    else:
        scale = jnp.asarray([0.8, 0.2], jnp.float32)
    with pytest.raises(ValueError):
        mhc.collapse_head(params["streams"], fn, base, scale)


# -- the shard_map wrapper -----------------------------------------------------


@pytest.mark.parametrize("n", [1, 128, 1000])
def test_collapse_head_under_shard_map(n):
    """``n=1`` is the decode shape; the ragged count hits the kernel's pad path.

    The ``data`` axis is trivial here, so this covers the specs, the dtype and
    ``out_sharding`` but not cross-device correctness.
    """
    mesh = _mesh()
    mhc = DeepseekV4MHC(_config())
    params = _params((n,))
    stub = _stub(mhc, mesh, params)
    with jax.set_mesh(mesh):
        out = DeepseekV4Model._collapse_head(stub, params["streams"])
    assert out.dtype == jnp.bfloat16
    assert out.shape == (n, HIDDEN)
    assert out.sharding == NamedSharding(mesh, P("data", None))
    # Same kernel on both sides, so the wrapper must not perturb a single bit.
    direct = mhc.collapse_head(params["streams"], *_head_args(params))
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(direct, np.float32))
