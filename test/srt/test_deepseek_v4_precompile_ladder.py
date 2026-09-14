"""Precompile context ladder for DeepSeek-V4 read-table capacity buckets (CPU-only)."""

import pytest

from sgl_jax.srt.layers.attention.deepseek_v4_backend import (
    capacity_bucket,
    precompile_capacities,
)
from sgl_jax.srt.model_executor.compilation_manager import CompilationManager


def test_capacity_bucket_matches_runtime_rule():
    # Same rule the runtime applies to live batches: power of two, minimum 128.
    assert capacity_bucket(0) == 128
    assert capacity_bucket(128) == 128
    assert capacity_bucket(129) == 256
    assert capacity_bucket(8192) == 8192
    assert capacity_bucket(8193) == 16384


def test_precompile_capacities_cover_a_request_of_that_length():
    caps = precompile_capacities(262144)
    assert caps == {4: 65536, 128: 2048}
    caps = precompile_capacities(32768)
    assert caps == {4: 8192, 128: 256}
    # A 1000-token request completes 250 groups at ratio 4 (-> 256) and 7 at ratio 128 (-> 128).
    assert precompile_capacities(1000) == {4: 256, 128: 128}


def test_context_ladder_env(monkeypatch):
    monkeypatch.delenv("SGLANG_JAX_PRECOMPILE_CONTEXT_LADDER", raising=False)
    assert CompilationManager._compute_context_ladder() == [None]
    monkeypatch.setenv("SGLANG_JAX_PRECOMPILE_CONTEXT_LADDER", "262144, 32768,131072,65536")
    assert CompilationManager._compute_context_ladder() == [32768, 65536, 131072, 262144]
    monkeypatch.setenv("SGLANG_JAX_PRECOMPILE_CONTEXT_LADDER", "0,1024")
    with pytest.raises(ValueError):
        CompilationManager._compute_context_ladder()


def test_full_context_ladder_covers_every_bucket(monkeypatch):
    monkeypatch.setenv("SGLANG_JAX_PRECOMPILE_CONTEXT_LADDER", "full")
    rungs = CompilationManager._compute_context_ladder(262144)
    # 512 .. 262144 doubling: every ratio-4 bucket 128 .. 65536 appears exactly once.
    assert rungs == [512 << i for i in range(10)]
    assert [precompile_capacities(r)[4] for r in rungs] == [128 << i for i in range(10)]
    # Non power-of-two maximum: the last rung is the maximum itself.
    rungs = CompilationManager._compute_context_ladder(100000)
    assert rungs[-1] == 100000 and rungs[-2] == 65536
    assert precompile_capacities(100000)[4] == 32768
    # Walking a chunked prefill up to any rung never meets an uncovered bucket.
    covered = {tuple(sorted(precompile_capacities(r).items())) for r in rungs}
    for ctx in range(256, 100001, 256):
        assert tuple(sorted(precompile_capacities(ctx).items())) in covered
    with pytest.raises(ValueError):
        CompilationManager._compute_context_ladder(None)


def test_set_precompile_context_only_touches_backends_that_opt_in():
    class Backend:
        precompile_context_len = None

    class Runner:
        attn_backend = Backend()

    CompilationManager._set_precompile_context(Runner(), 4096)
    assert Runner.attn_backend.precompile_context_len == 4096

    class Other:
        pass

    class Runner2:
        attn_backend = Other()

    CompilationManager._set_precompile_context(Runner2(), 4096)  # no attribute, no error
    assert not hasattr(Runner2.attn_backend, "precompile_context_len")


def test_precompile_context_len_does_not_change_the_graphdef():
    # The ladder value must not leak into the nnx graphdef (== jit cache key), otherwise
    # precompiled executables (context_len=N) never match runtime calls (context_len=None).
    from flax import nnx

    from sgl_jax.srt.layers.attention.deepseek_v4_backend import _PrecompileContextBox

    class Holder(nnx.Module):
        def __init__(self):
            self._precompile_box = _PrecompileContextBox()

    h = Holder()
    before = nnx.graphdef(h)
    h._precompile_box.context_len = 32768
    assert nnx.graphdef(h) == before
    assert hash(nnx.graphdef(h)) == hash(before)
