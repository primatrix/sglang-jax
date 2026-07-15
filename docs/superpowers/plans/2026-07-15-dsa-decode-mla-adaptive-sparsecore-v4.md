# DSA Decode MLA Adaptive SparseCore Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use both active Falcon SparseCores for the latency-critical `B=1, K=2048` DSA decode gather while preserving the established two-stage MLA contract.

**Architecture:** The public DSA dispatcher resolves `gather_block="auto"` from static shape and runtime SparseCore topology.  Only the known GLM single-request pipeline shape resolves to 64; every other auto case retains 128.  SparseCore still materializes selected rows and TensorCore still performs the contiguous selected MLA calculation.

**Tech Stack:** Python 3.12, JAX 0.8.1, Pallas TPU SparseCore, BF16, unittest, Falcon v7x-8.

---

### Task 1: Specify and test auto-window resolution

**Files:**
- Modify: `python/sgl_jax/srt/kernels/mla/dsa/gather.py`
- Modify: `python/sgl_jax/test/kernels/test_dsa_decode_mla.py`

- [ ] **Step 1: Write the failing pure-planner test**

Add this import to the test module:

```python
from sgl_jax.srt.kernels.mla.dsa.gather import (
    _plan_sparsecore_pipeline,
    resolve_sparsecore_pipeline_gather_block,
)
```

Add this test to `TestDSASelectedKVGather`:

```python
def test_auto_pipeline_uses_64_rows_only_for_single_glm_request(self):
    self.assertEqual(
        resolve_sparsecore_pipeline_gather_block(
            "auto",
            batch_size=1,
            padded_selected=2048,
            cache_width=640,
            reported_cores=4,
            num_subcores=16,
        ),
        64,
    )
    self.assertEqual(
        resolve_sparsecore_pipeline_gather_block(
            "auto",
            batch_size=2,
            padded_selected=2048,
            cache_width=640,
            reported_cores=4,
            num_subcores=16,
        ),
        128,
    )
    self.assertEqual(
        resolve_sparsecore_pipeline_gather_block(
            "auto",
            batch_size=1,
            padded_selected=512,
            cache_width=640,
            reported_cores=4,
            num_subcores=16,
        ),
        128,
    )
    self.assertEqual(
        resolve_sparsecore_pipeline_gather_block(
            128,
            batch_size=1,
            padded_selected=2048,
            cache_width=640,
            reported_cores=4,
            num_subcores=16,
        ),
        128,
    )
```

- [ ] **Step 2: Run the test and verify the expected import failure**

Run:

```bash
PYTHONPATH=python uv run --directory python python -m unittest \
  sgl_jax.test.kernels.test_dsa_decode_mla.TestDSASelectedKVGather.test_auto_pipeline_uses_64_rows_only_for_single_glm_request
```

Expected: FAIL because `resolve_sparsecore_pipeline_gather_block` is not yet exported by `gather.py`.

- [ ] **Step 3: Implement the static resolver**

Add this function after `_plan_sparsecore_pipeline`:

```python
def resolve_sparsecore_pipeline_gather_block(
    requested: int | str,
    *,
    batch_size: int,
    padded_selected: int,
    cache_width: int,
    reported_cores: int,
    num_subcores: int,
) -> int:
    if requested != "auto":
        _validate_gather_block(requested)
        return requested
    if (
        batch_size == 1
        and padded_selected == 2048
        and cache_width == 640
        and _active_sparsecore_cores(reported_cores) >= 2
        and num_subcores == 16
    ):
        candidate = 64
        _plan_sparsecore_pipeline(
            batch_size=batch_size,
            padded_selected=padded_selected,
            gather_block=candidate,
            available_cores=_active_sparsecore_cores(reported_cores),
            num_subcores=num_subcores,
        )
        return candidate
    return _DEFAULT_GATHER_BLOCK
```

Keep `materialize_selected_kv_sparsecore_pipeline_unchecked` integer-only; its
caller will pass the resolved static integer.

- [ ] **Step 4: Run the planner tests**

Run:

```bash
PYTHONPATH=python uv run --directory python python -m unittest \
  sgl_jax.test.kernels.test_dsa_decode_mla.TestDSASelectedKVGather
```

Expected: PASS on CPU with no TPU requirement.

- [ ] **Step 5: Commit the planner contract**

```bash
git add python/sgl_jax/srt/kernels/mla/dsa/gather.py \
  python/sgl_jax/test/kernels/test_dsa_decode_mla.py
git commit -m "feat(kernels): select DSA SparseCore window by shape"
```

### Task 2: Route the public DSA dispatcher through the resolver

**Files:**
- Modify: `python/sgl_jax/srt/kernels/mla/dsa/kernel.py`
- Modify: `python/sgl_jax/test/kernels/test_dsa_decode_mla.py`

- [ ] **Step 1: Write the failing dispatcher-resolution test**

Extend the existing kernel import in the test module to import the expected
`_resolve_gather_block` helper, then add this mock-free static test:

```python
def test_pipeline_dispatch_resolves_auto_and_preserves_explicit_block(self):
    self.assertEqual(
        _resolve_gather_block(
            "sparsecore-pipeline", "auto", batch_size=1,
            max_selected=2048, cache_width=640, reported_cores=4,
            num_subcores=16,
        ),
        64,
    )
    self.assertEqual(
        _resolve_gather_block(
            "sparsecore", "auto", batch_size=1,
            max_selected=2048, cache_width=640, reported_cores=4,
            num_subcores=16,
        ),
        128,
    )
    self.assertEqual(
        _resolve_gather_block(
            "sparsecore-pipeline", 128, batch_size=1,
            max_selected=2048, cache_width=640, reported_cores=4,
            num_subcores=16,
        ),
        128,
    )
```

- [ ] **Step 2: Run the test and verify the missing-helper failure**

Run:

```bash
PYTHONPATH=python uv run --directory python python -m unittest \
  sgl_jax.test.kernels.test_dsa_decode_mla.TestDSASelectedKVGather.test_pipeline_dispatch_resolves_auto_and_preserves_explicit_block
```

Expected: FAIL because `_resolve_gather_block` is absent from `kernel.py`.

- [ ] **Step 3: Implement static dispatch resolution**

Import `resolve_sparsecore_pipeline_gather_block` and use:

```python
def _resolve_gather_block(
    gather_impl: GatherImplementation,
    gather_block: int | str,
    *,
    batch_size: int,
    max_selected: int,
    cache_width: int,
    reported_cores: int,
    num_subcores: int,
) -> int:
    if gather_impl != "sparsecore-pipeline":
        return 128 if gather_block == "auto" else gather_block
    return resolve_sparsecore_pipeline_gather_block(
        gather_block,
        batch_size=batch_size,
        padded_selected=max_selected,
        cache_width=cache_width,
        reported_cores=reported_cores,
        num_subcores=num_subcores,
    )
```

Set the public `dsa_decode_mla_attention` default to `gather_block="auto"`.
After `_resolve_gather_impl`, obtain `pltpu.get_tpu_info().sparse_core` only
for the pipeline case, resolve a Python integer from static array shapes, and
pass that integer into the cached composed launcher.  Keep
`dsa_decode_mla_attention_unchecked` defaulted to integer 128 for direct
benchmark/reproduction callers.

- [ ] **Step 4: Re-run CPU tests**

Run:

```bash
PYTHONPATH=python uv run --directory python python -m unittest \
  sgl_jax.test.kernels.test_dsa_decode_mla.TestDSASelectedKVGather
```

Expected: PASS.

- [ ] **Step 5: Commit dispatcher routing**

```bash
git add python/sgl_jax/srt/kernels/mla/dsa/kernel.py \
  python/sgl_jax/test/kernels/test_dsa_decode_mla.py
git commit -m "feat(kernels): auto-size DSA pipeline gather"
```

### Task 3: Close the GLM-shape Falcon correctness gap

**Files:**
- Modify: `python/sgl_jax/test/kernels/test_dsa_decode_mla.py`
- Modify: `scripts/kernels/falcon_dsa_decode_mla_pipeline_v7x8.yaml`

- [ ] **Step 1: Write the TPU-only batch-32 GLM test**

Add a new test with `batch_size=32`, `num_heads=8`, `latent_dim=512`,
`rope_dim=64`, `top_k=2048`, `page_size=128`, and cache shape
`(16, 64, 2, 640)`.  Make each batch row a different cyclic permutation of
`(arange(2048) * 17) % 2048`; call the public DSA operation with
`gather_impl="sparsecore-pipeline"` and default auto block.  Compare it with
`reference_dsa_decode_mla_attention` at `rtol=2e-2, atol=1e-2` after checking
that all output values are finite.

- [ ] **Step 2: Verify the test is skipped locally without a TPU**

Run:

```bash
PYTHONPATH=python uv run --directory python python -m unittest -v \
  sgl_jax.test.kernels.test_dsa_decode_mla.TestDSADecodeMLAPallas.test_tpu_pipeline_composed_batch_32_glm_shape_2048_matches_reference
```

Expected: SKIP on a non-TPU workstation.

- [ ] **Step 3: Add the test to the Falcon correctness command**

Extend `pipeline_correctness.log` command in the manifest with the new exact
test name, retaining the B1 GLM test and the gather-only equality test.

- [ ] **Step 4: Commit the correct-shape gate**

```bash
git add python/sgl_jax/test/kernels/test_dsa_decode_mla.py \
  scripts/kernels/falcon_dsa_decode_mla_pipeline_v7x8.yaml
git commit -m "test(kernels): gate batched GLM DSA pipeline correctness"
```

### Task 4: Run one focused Falcon discriminator

**Files:**
- Modify: `benchmark/kernels/mla/bench_dsa_decode_mla.py`
- Modify: `scripts/kernels/falcon_dsa_decode_mla_pipeline_v7x8.yaml`

- [ ] **Step 1: Write failing benchmark-variant metadata tests**

Add `sparsecore-pipeline-64` and `sparsecore-pipeline-128` to
`BENCHMARK_VARIANTS`, then update
`python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py` to assert both names
are present and both estimate `3 * selected_bytes`.

- [ ] **Step 2: Run the metadata tests and observe failure**

Run:

```bash
PYTHONPATH=python uv run --directory python python -m unittest \
  sgl_jax.test.kernels.test_bench_dsa_decode_mla
```

Expected: FAIL because neither explicit pipeline variant exists.

- [ ] **Step 3: Implement both explicit pipeline variants**

Add benchmark callables using
`dsa_decode_mla_attention_unchecked(..., gather_impl="sparsecore-pipeline",
gather_block=64)` and the identical 128 call.  Retain legacy `sparsecore`,
`xla-gather`, `gather-only`, `attention-only`, and the labelled non-production
`dense-jax-baseline`; do not interpret the last one as an MLA-v2 serving
baseline.

- [ ] **Step 4: Change the Falcon manifest to one discriminator job**

After the correctness command, invoke:

```bash
python benchmark/kernels/mla/bench_dsa_decode_mla.py \
  --batch-size 1 --context-length 160000 --top-k 2048 \
  --slot-distribution uniform --variant all --warmup-iters 50 --iters 200 \
  --output "${RESULTS}/b1_ctx160000_adaptive_pipeline.json"
```

The benchmark JSON and log must remain in the experiment artifact directory.

- [ ] **Step 5: Run local benchmark metadata tests**

Run:

```bash
PYTHONPATH=python uv run --directory python python -m unittest \
  sgl_jax.test.kernels.test_bench_dsa_decode_mla
```

Expected: PASS on CPU; the benchmark itself remains TPU-only.

- [ ] **Step 6: Commit and push to the Falcon source remote**

```bash
git add benchmark/kernels/mla/bench_dsa_decode_mla.py \
  python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py \
  scripts/kernels/falcon_dsa_decode_mla_pipeline_v7x8.yaml
git commit -m "bench(kernels): compare adaptive DSA gather windows"
git push primatrix develop/dsa-decode-mla
```

- [ ] **Step 7: Submit the isolated v7x-8 Falcon job and collect evidence**

Run:

```bash
EXP_ID="$(falcon workflow exp submit -f scripts/kernels/falcon_dsa_decode_mla_pipeline_v7x8.yaml --output json | jq -r '.exp_id')"
test -n "${EXP_ID}"
test "${EXP_ID}" != "null"
falcon workflow exp wait "${EXP_ID}" --timeout 40m --output json
falcon exp logs "${EXP_ID}" --container task --tail 500 --output json
```

Expected: all three Falcon correctness tests pass; the benchmark reports both
pipeline window variants.  Retain adaptive-64 only if its median and p99 beat
pipeline-128.  Otherwise revert only the automatic selection while retaining
the explicit benchmark variants and correctness gate for diagnosis.
