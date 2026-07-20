# GLM-5.2 DSA Kernel Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver a stable CPU PyTorch correctness oracle and a production-derived performance shape manifest for the GLM-5.2 final sparse MLA kernel.

**Architecture:** Keep correctness and performance inputs separate while sharing one frozen ABI. The CPU package independently computes Indexer Top-K, logical-to-physical mapping, and sparse MLA golden outputs; the TPU benchmark consumes only final sparse-MLA inputs and measures precompiled calls for named production scenarios.

**Tech Stack:** Python, PyTorch CPU, NumPy `.npz`, JSON manifests, JAX/Pallas benchmark harness, pytest/unittest.

---

## File map

- Create `benchmark/kernels/mla/glm52_dsa_handoff.py`: production constants and named performance cases only; no JAX dependency.
- Create `benchmark/kernels/mla/export_glm52_dsa_golden.py`: deterministic CPU PyTorch fixture exporter using the existing independent reference functions.
- Modify `benchmark/kernels/mla/bench_dsa_decode_mla.py`: represent physical query buckets, inactive padding rows, and causal prefill selected-count patterns.
- Create `python/sgl_jax/test/kernels/test_glm52_dsa_handoff.py`: host-only contract tests for the manifest and exporter.
- Modify `python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py`: TDD coverage for padded decode and causal prefill fixtures.
- Create `note/2026-07-20-glm52-dsa-kernel-handoff.md`: operator-facing ABI, golden policy, required cases, commands, and acceptance gates.

### Task 1: Freeze production constants and named performance cases

**Files:**
- Create: `benchmark/kernels/mla/glm52_dsa_handoff.py`
- Create: `python/sgl_jax/test/kernels/test_glm52_dsa_handoff.py`

- [x] **Step 1: Write the failing manifest test**

```python
def test_production_contract_uses_tp32_local_shape():
    assert GLM52_DSA_CONTRACT.total_query_heads == 64
    assert GLM52_DSA_CONTRACT.tensor_parallel_size == 32
    assert GLM52_DSA_CONTRACT.local_query_heads == 2
    assert GLM52_DSA_CONTRACT.latent_dim == 512
    assert GLM52_DSA_CONTRACT.rope_dim == 64
    assert GLM52_DSA_CONTRACT.cache_width == 640
    assert GLM52_DSA_CONTRACT.page_size == 128
    assert GLM52_DSA_CONTRACT.index_topk == 2048
```

- [x] **Step 2: Run the test and verify RED**

Run: `python -m pytest python/sgl_jax/test/kernels/test_glm52_dsa_handoff.py -q`

Expected: collection fails because `benchmark.kernels.mla.glm52_dsa_handoff` does not exist.

- [x] **Step 3: Implement immutable contract and case records**

Define `Glm52DsaContract`, `SparseMlaPerfCase`, `GLM52_DSA_CONTRACT`, and `PERFORMANCE_CASES`. Required named cases are:

```python
("decode-bucket-a1-c512", "decode", 64, 1, 512, 2048)
("decode-bucket-a1-c1024", "decode", 64, 1, 1024, 2048)
("decode-bucket-a1-c2048", "decode", 64, 1, 2048, 2048)
("decode-bucket-a1-c4096", "decode", 64, 1, 4096, 2048)
("decode-bucket-a8-c4096", "decode", 64, 8, 4096, 2048)
("decode-bucket-a32-c4096", "decode", 64, 32, 4096, 2048)
("decode-bucket-a64-c4096", "decode", 64, 64, 4096, 2048)
("decode-long-a1-c160k", "decode", 1, 1, 160_000, 2048)
("decode-throughput-a8-c32k", "decode", 8, 8, 32_000, 2048)
("prefill-t128-start0", "prefill", 128, 128, 128, 2048)
("prefill-t128-start2048", "prefill", 128, 128, 2176, 2048)
```

For prefill cases, store `start_position` and one shared request region; for decode cases, selected counts are `min(context_length, 2048)` on active rows and request regions are disjoint. Validate names, active rows, dimensions, mode, context coverage, and `H_local=2` when records are built.

- [x] **Step 4: Run the test and verify GREEN**

Run: `python -m pytest python/sgl_jax/test/kernels/test_glm52_dsa_handoff.py -q`

Expected: production contract and case tests pass.

### Task 2: Make the benchmark fixture model static padding and causal prefill

**Files:**
- Modify: `python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py`
- Modify: `benchmark/kernels/mla/bench_dsa_decode_mla.py`

- [x] **Step 1: Add failing padded-decode and causal-prefill tests**

```python
def test_fixture_zeroes_inactive_physical_bucket_rows():
    inputs = make_benchmark_inputs(
        batch_size=64, active_batch_size=1, context_length=4096,
        top_k=2048, num_heads=2, latent_dim=512, rope_dim=64,
        page_size=128, slot_order="unsorted", valid_count_pattern="full",
    )
    assert inputs.valid_counts.tolist() == [2048] + [0] * 63
    assert np.all(inputs.topk_slots[1:] == 0)

def test_fixture_builds_causal_prefill_counts_with_static_kmax():
    inputs = make_benchmark_inputs(
        batch_size=128, active_batch_size=128, context_length=128,
        top_k=2048, num_heads=2, latent_dim=512, rope_dim=64,
        page_size=128, slot_order="unsorted", valid_count_pattern="causal",
        start_position=0,
    )
    assert inputs.topk_slots.shape == (128, 2048)
    assert inputs.valid_counts.tolist() == list(range(1, 129))
```

- [x] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py -q`

Expected: `make_benchmark_inputs()` rejects the new keyword arguments.

- [x] **Step 3: Implement the minimal fixture behavior**

Add `active_batch_size`, `valid_count_pattern`, `start_position`, `request_layout`, and optional `cache_capacity` arguments. Reserve the full `[physical_rows, K_max]` slot tensor, populate only counted prefixes, set inactive rows to count zero, generate per-query causal counts as `min(start_position + row + 1, context_length, K_max)`, and use disjoint address regions for multi-request decode. Keep invalid/padded slot contents semantically irrelevant and fill them with zero.

- [x] **Step 4: Expose the behavior through CLI and JSON**

Add `--active-batch-size`, `--valid-count-pattern {full,causal}`, and `--start-position`. Record all three fields in the emitted JSON input block. Keep compilation outside timed iterations and retain synchronized warmups.

- [x] **Step 5: Run benchmark host tests and verify GREEN**

Run: `python -m pytest python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py -q`

Expected: all tests pass, including existing cache-layout and no-closed-constant gates.

### Task 3: Export deterministic CPU PyTorch goldens

**Files:**
- Modify: `python/sgl_jax/test/kernels/test_glm52_dsa_handoff.py`
- Create: `benchmark/kernels/mla/export_glm52_dsa_golden.py`

- [x] **Step 1: Add a failing exporter round-trip test**

The test invokes `export_golden_bundle(tmp_path, candidate_lengths=(1, 129, 2049), seed=7)`, loads `manifest.json`, and verifies:

```python
assert manifest["schema_version"] == "glm52-dsa-golden-v1"
assert manifest["candidate_lengths"] == [1, 129, 2049]
assert {case["stage"] for case in manifest["cases"]} == {
    "indexer_selection", "logical_to_physical", "sparse_mla"
}
```

For every declared `.npz`, verify the file exists, its SHA-256 matches the manifest, integer outputs have exact dtypes, and the sparse MLA output is FP32.

- [x] **Step 2: Run the exporter test and verify RED**

Run: `python -m pytest python/sgl_jax/test/kernels/test_glm52_dsa_handoff.py -q`

Expected: import fails because the exporter does not exist.

- [x] **Step 3: Implement deterministic stage-separated fixtures**

Use only:

```python
torch_glm_dsa_select
torch_logical_topk_to_physical_slots
torch_dsa_sparse_mla
```

from `sgl_jax.srt.kernels.dsa.torch_reference`. Quantize floating inputs through CPU BF16 and store their exact quantized values as NumPy FP32 because NumPy has no portable BF16 `.npz` dtype. Store expected Indexer scores and sparse MLA output as FP32; IDs, slots, and counts as INT32. Use unique deterministic score margins so Top-K integer equality is well-defined, plus one realistic fixture that exercises all 32 heads, all 128 dimensions, signed head weights, and both sides of ReLU.

The default boundary lengths are:

```text
1, 127, 128, 129, 257, 2047, 2048, 2049, 3072, 4096
```

Keep correctness fixtures small in token/query count; do not serialize a 160k cache. Long-context memory footprint belongs only to the performance matrix.

- [x] **Step 4: Add manifest metadata and checksums**

Declare shapes, semantic dtypes, storage dtypes, seed, scale, cache axis names, slot decode formula, valid address range, page size, packing, dimensions, counted-prefix semantics, per-case file name, and SHA-256. Refuse to overwrite a non-empty output directory unless `--force` is passed.

- [x] **Step 5: Run the exporter test and verify GREEN**

Run: `python -m pytest python/sgl_jax/test/kernels/test_glm52_dsa_handoff.py -q`

Expected: all manifest, checksum, dtype, and stage-coverage assertions pass.

### Task 4: Write the operator-facing handoff

**Files:**
- Create: `note/2026-07-20-glm52-dsa-kernel-handoff.md`

- [x] **Step 1: Document the frozen final sparse-MLA ABI**

Document these tensor contracts exactly:

```text
q_latent        [Q_bucket, 2, 512] BF16
q_rope          [Q_bucket, 2, 64]  BF16
cache           [pages, 64, 2, 640] BF16
physical_slots  [Q_bucket, 2048] INT32
selected_counts [Q_bucket] INT32
output          [Q_bucket, 2, 512] BF16
page_size=128, packing=2, sm_scale=256^-0.5
```

State that only `physical_slots[:, :selected_counts]` is valid, slot 0 is legal, cache reads are token-granular, and the sparse forward does not own Indexer, Top-K, mapping, or cache writes.

- [x] **Step 2: Document correctness gates**

State:

- selection IDs/counts and mapped physical slots compare exactly;
- selection scores compare in FP32 with `rtol=1e-6, atol=1e-6` for deterministic no-tie fixtures;
- sparse MLA candidate output compares to PyTorch FP32 with production BF16 tolerance `rtol=2e-2, atol=1e-2`;
- when visible tokens are at most 2048, sparse output also matches dense-over-visible reference;
- when visible tokens exceed 2048, dense full attention is not a golden: CPU Indexer selection plus CPU sparse MLA is the golden.

- [x] **Step 3: Document required performance cases and measurement rules**

Mark the physical-B64 selected-count and active-row sweeps, B1/C160k latency, B8/C32k throughput, and two T128 prefill cases as required. Record an explicit synchronized compile call and `compile_ms`, at least 20 synchronized warmups, at least 100 timed calls, median/p95/p99, unsorted slots as primary, page-sorted as diagnostic, and no CPU work inside timing.

- [x] **Step 4: Add reproducible commands**

Include the exact CPU export command, host test commands, and one command per named TPU performance case. Explain that `batch_size` is physical query rows while `active_batch_size` determines counted rows.

### Task 5: Verify the complete handoff

**Files:**
- Verify all files above.

- [x] **Step 1: Run host-only correctness tests**

Run:

```bash
python -m pytest \
  python/sgl_jax/test/test_dsa_cross_framework.py \
  python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py \
  python/sgl_jax/test/kernels/test_glm52_dsa_handoff.py -q
```

Expected: all selected tests pass.

- [x] **Step 2: Generate and inspect the default golden bundle**

Run:

```bash
python benchmark/kernels/mla/export_glm52_dsa_golden.py \
  --output-dir /tmp/glm52-dsa-golden
python -m json.tool /tmp/glm52-dsa-golden/manifest.json >/dev/null
```

Expected: exporter succeeds and the manifest parses.

- [x] **Step 3: Check repository diff and commit**

Run:

```bash
git diff --check
git status --short
git add \
  benchmark/kernels/mla/glm52_dsa_handoff.py \
  benchmark/kernels/mla/export_glm52_dsa_golden.py \
  benchmark/kernels/mla/bench_dsa_decode_mla.py \
  python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py \
  python/sgl_jax/test/kernels/test_glm52_dsa_handoff.py \
  note/2026-07-20-glm52-dsa-kernel-handoff.md \
  docs/superpowers/plans/2026-07-20-glm52-dsa-kernel-handoff.md
git commit -m "feat(dsa): package GLM-5.2 kernel handoff inputs"
```

Expected: one focused commit containing only the handoff package.
