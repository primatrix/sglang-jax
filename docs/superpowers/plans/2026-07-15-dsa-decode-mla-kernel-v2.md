# DSA Decode MLA Kernel v2 Implementation Plan

> **Design:** `docs/superpowers/specs/2026-07-15-dsa-decode-mla-kernel-v2-design.md`
>
> **Execution rule:** complete each correctness gate before starting the next
> one. Do not submit a Falcon performance job until the production-layout
> gather test, selected-attention test, and composed end-to-end test all pass.

**Goal:** Replace the invalid per-slot TensorCore DMA prototype with a
SparseCore indirect-gather stage followed by a contiguous TensorCore MLA
attention stage, while preserving the current public decode API.

**Architecture:** Stage A flattens the packed paged cache to logical token rows
and uses SparseCore indirect DMA to materialize `[B, Kpad, W]`. Stage B packs
queries like MLA v2, copies one contiguous selected-KV block into VMEM, and
computes fixed-K QK, masked FP32 softmax, and PV on TensorCore. A plain JAX
gather remains an explicit baseline/fallback; the old per-slot kernel is
removed once the composed path passes.

**Runtime target:** JAX/JAXlib 0.8.1, BF16, TPU v7x. The SparseCore stage uses
`KernelType.SC_VECTOR_SUBCORE` and the JIT compiler option
`xla_tpu_use_tc_device_shape_on_sc=false`.

**Local test command prefix:**

```bash
PYTHONPATH=python /Users/jiongxuan/workspace/sglang-jax/python/.venv/bin/python
```

## Task 1: Lock production cache layout and GLM math into the oracle

**Files:**

- Modify: `python/sgl_jax/test/kernels/test_dsa_decode_mla.py`
- Modify: `python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py`
- Modify: `benchmark/kernels/mla/bench_dsa_decode_mla.py`

### Step 1: Write the failing production-layout tests

Add a fixture that builds BF16 cache pages as
`[pages, page_size // 2, 2, 640]`, not the prototype's packing-one layout. Fill
each logical token row with an identifiable value using the explicit mapping
`page, offset -> page, offset // 2, offset % 2`.

Add tests that assert:

- odd/even lanes and page-boundary slots decode correctly;
- unsorted and duplicate slots match the dense selected oracle;
- benchmark fixtures use packing factor two; and
- the GLM benchmark scale is exactly `256**-0.5`.

Run:

```bash
PYTHONPATH=python:. /Users/jiongxuan/workspace/sglang-jax/python/.venv/bin/python \
  -m unittest -v sgl_jax.test.kernels.test_dsa_decode_mla \
  sgl_jax.test.kernels.test_bench_dsa_decode_mla
```

Expected: benchmark-layout and scale assertions fail against the current
packing-one fixture and `(latent_dim + rope_dim)**-0.5` default.

### Step 2: Fix fixtures without changing the reference algorithm

Update `make_benchmark_inputs` to require an even page size for BF16, allocate
`[num_pages, page_size // 2, 2, padded_width]`, and keep physical slots in
logical token units. Add a `--sm-scale` override whose GLM default is
`256**-0.5`; record it in benchmark JSON.

Run the Task 1 tests again. Expected: pass.

### Step 3: Commit the contract correction

```bash
git add python/sgl_jax/test/kernels/test_dsa_decode_mla.py \
  python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py \
  benchmark/kernels/mla/bench_dsa_decode_mla.py
git commit -m "test: model production DSA MLA cache layout"
```

## Task 2: Add selected-slot padding and a plain-JAX materialization baseline

**Files:**

- Create: `python/sgl_jax/srt/kernels/mla/dsa/gather.py`
- Modify: `python/sgl_jax/srt/kernels/mla/dsa/__init__.py`
- Modify: `python/sgl_jax/test/kernels/test_dsa_decode_mla.py`

### Step 1: Write failing safe-padding tests

Test a pure helper with heterogeneous `valid_counts` and `K` not divisible by
128. Assert that it:

- rounds K up to the gather block;
- preserves all valid slots exactly;
- replaces every invalid or padded entry with slot zero; and
- rejects a nonpositive or unsupported gather block at the validated wrapper.

Also test `materialize_selected_kv_xla` against the explicit page/packed-row/
lane oracle for packing two, duplicates, cross-page rows, and `-1` padding.

Run only the new tests. Expected: import failure.

### Step 2: Implement the minimum pure JAX baseline

Implement:

```python
def prepare_safe_topk_slots(topk_slots, valid_counts, *, gather_block=128): ...

def materialize_selected_kv_xla(
    cache_kv, topk_slots, valid_counts, *, gather_block=128
): ...
```

The helper uses a position iota and `position < valid_count`; it never uses a
negative slot for a gather. Flatten cache dimensions 0 through 2 only after
safe-slot construction. Keep these functions JIT-compatible and avoid NumPy
conversion in unchecked code.

Run the new tests. Expected: pass.

### Step 3: Commit the baseline

```bash
git add python/sgl_jax/srt/kernels/mla/dsa/gather.py \
  python/sgl_jax/srt/kernels/mla/dsa/__init__.py \
  python/sgl_jax/test/kernels/test_dsa_decode_mla.py
git commit -m "feat: add safe selected KV materialization baseline"
```

## Task 3: Implement SparseCore indirect gather

**Files:**

- Modify: `python/sgl_jax/srt/kernels/mla/dsa/gather.py`
- Modify: `python/sgl_jax/test/kernels/test_dsa_decode_mla.py`

### Step 1: Add a TPU-gated failing Stage A test

Add `TestDSASparseCoreGather` with:

- a small BF16 packing-two cross-page case;
- a production-width `W=640`, `G=128` case;
- a `K=2048` case; and
- unsorted slots and duplicates.

Compare the SparseCore output bit-for-bit with
`materialize_selected_kv_xla`, because Stage A only copies BF16 values. Skip
when the default backend is not TPU.

### Step 2: Implement the official JAX 0.8.1 gather pattern

Add a SparseCore Pallas kernel whose cache input has HBM memory space and
whose slot block has VMEM memory space. Flatten batch and selected dimensions
at the Pallas-call boundary because JAX 0.8.1 SparseCore does not support
`None`/squeezed block dimensions. For each one-dimensional gather-block grid
program, execute the equivalent of:

```python
pltpu.sync_copy(cache_hbm_ref.at[slot_indices_ref], output_vmem_ref)
```

Give the output a VMEM block spec and let the Pallas block pipeline commit it
to the global HBM result. This mirrors JAX 0.8.1's official SparseCore gather
test and avoids a redundant explicit copy. Do not add data-dependent loops.

The unchecked `pallas_call` remains raw so it can compose with TensorCore.
Wrap the outermost eager/composed call in:

```python
jax.jit(
    call,
    compiler_options={"xla_tpu_use_tc_device_shape_on_sc": "false"},
)
```

with `CompilerParams(kernel_type=KernelType.SC_VECTOR_SUBCORE)`.

Expose `materialize_selected_kv_sparsecore_unchecked`; its inputs are already
safe and padded. Keep host validation in a separate eager wrapper.

### Step 3: Run static/local checks

The real SparseCore kernel cannot be established by CPU interpret mode. Run
all non-TPU tests and Ruff first:

```bash
PYTHONPATH=python:. /Users/jiongxuan/workspace/sglang-jax/python/.venv/bin/python \
  -m unittest -v sgl_jax.test.kernels.test_dsa_decode_mla
ruff check \
  python/sgl_jax/srt/kernels/mla/dsa \
  python/sgl_jax/test/kernels/test_dsa_decode_mla.py
```

Expected: local tests pass and SparseCore tests skip.

### Step 4: Run Falcon Gate 1 only

Update the Falcon spec to run just the tiny and production-width Stage A
tests. Submit one v7x-8 task on `tpuv7x-64-node`; do not include end-to-end or
benchmark commands.

Expected: compile succeeds and gather output is bitwise equal to XLA gather.
If it fails, classify the failure as API/lowering, layout, runtime support, or
test-contract error before editing code. Use plain XLA gather only as an
explicit functional fallback if the runtime truly lacks the feature.

### Step 5: Commit the proven gather stage

```bash
git add python/sgl_jax/srt/kernels/mla/dsa/gather.py \
  python/sgl_jax/test/kernels/test_dsa_decode_mla.py \
  scripts/kernels/falcon_dsa_decode_mla_v7x8.yaml
git commit -m "feat: gather selected MLA cache rows on SparseCore"
```

## Task 4: Implement contiguous TensorCore selected attention

**Files:**

- Create: `python/sgl_jax/srt/kernels/mla/dsa/attention.py`
- Modify: `python/sgl_jax/srt/kernels/mla/dsa/__init__.py`
- Modify: `python/sgl_jax/test/kernels/test_dsa_decode_mla.py`

### Step 1: Write failing local interpret tests

Test `selected_mla_attention` independently of Stage A. Build the selected KV
tensor directly from the reference fixture and cover:

- packing-two source semantics already resolved into `[B, Kpad, W]`;
- independently padded `L=96`, `R=64`;
- heterogeneous valid counts and poisoned padded rows;
- slot permutation and duplicate metamorphic cases; and
- the TP=8 GLM shape `B=1,H=8,L=512,R=64,K=2048` with scale
  `256**-0.5`.

Compare against a new reference helper that consumes an already materialized
selected tensor. The helper must use FP32 complete softmax.

Run the new tests. Expected: import failure.

### Step 2: Implement query packing and the smallest full-K Pallas kernel

Reuse or extract the MLA v2 query preparation rules: align each head dimension
to 128, pad head count to BF16 packing, reshape to `[B,H/2,2,D]`, and place an
optimization barrier after pad/reshape.

In one TensorCore program per batch item:

1. DMA packed Q-nope, packed Q-RoPE, and contiguous selected KV from HBM to
   VMEM.
2. Reshape packed Q back to local heads in VMEM.
3. Compute QK with `jnp.einsum(..., preferred_element_type=jnp.float32)` using
   the same operand orientation as MLA v2.
4. Apply `k < valid_count` before a numerically stable FP32 softmax.
5. Compute PV in FP32 and cast/store through a packed VMEM output.

Return only actual heads and latent width after the Pallas call. Keep
`interpret=True` available for local correctness tests.

### Step 3: Run the local Stage B suite

Run the new selected-attention tests plus Ruff. Expected: all pass. Record
max absolute error for the GLM fixture if tolerance needs investigation; do
not relax `rtol=2e-2, atol=1e-2` without explaining the reduction difference.

### Step 4: Run Falcon Gate 2 only

Run Stage B TPU tests with an XLA-materialized selected tensor, first small and
then GLM shape. Do not invoke SparseCore in this gate.

Expected: both compile and match the FP32 reference. If full-K VMEM or compile
behavior fails, replace only Stage B with aligned K blocks and the existing
MLA v2 online-softmax merge; keep Stage A unchanged.

### Step 5: Commit the attention stage

```bash
git add python/sgl_jax/srt/kernels/mla/dsa/attention.py \
  python/sgl_jax/srt/kernels/mla/dsa/__init__.py \
  python/sgl_jax/test/kernels/test_dsa_decode_mla.py \
  scripts/kernels/falcon_dsa_decode_mla_v7x8.yaml
git commit -m "feat: add contiguous selected MLA attention kernel"
```

## Task 5: Compose both stages and preserve the public API

**Files:**

- Replace: `python/sgl_jax/srt/kernels/mla/dsa/kernel.py`
- Modify: `python/sgl_jax/srt/kernels/mla/dsa/__init__.py`
- Modify: `python/sgl_jax/test/kernels/test_dsa_decode_mla.py`

### Step 1: Write failing composition tests

Keep the existing public names:

- `dsa_decode_mla_attention`
- `dsa_decode_mla_attention_unchecked`

Add an explicit `gather_impl` selection for testing (`"sparsecore"` or
`"xla"`), defaulting to SparseCore on TPU and XLA only for interpret/local
execution. Test that validation still rejects malformed slots before dispatch.

End-to-end tests must use production packing two and cover:

- dynamic slot values under JIT;
- padding and duplicate semantics;
- GLM K=2048 dimensions and scale; and
- parity between XLA-gather composition and the explicit packed-cache oracle.

### Step 2: Replace the old direct-DMA loop

The wrapper performs:

1. static/host validation when requested;
2. safe slot padding;
3. selected KV materialization via the chosen implementation;
4. contiguous selected attention; and
5. output unpack/slice.

Delete `_dsa_decode_mla_kernel` and its per-slot DMA scratch. Do not retain an
unused invalid Pallas path.

### Step 3: Run all local DSA tests

```bash
PYTHONPATH=python:. /Users/jiongxuan/workspace/sglang-jax/python/.venv/bin/python \
  -m unittest -v sgl_jax.test.kernels.test_dsa_decode_mla \
  sgl_jax.test.kernels.test_bench_dsa_decode_mla
```

Expected: all CPU/interpret tests pass and only TPU-specific tests skip.

### Step 4: Run Falcon Gate 3

Run the composed SparseCore+TensorCore path for batch 1, 8, and 32. Compare to
the explicit packed-cache reference and print max absolute/relative error.
This job contains no timing loop.

Expected: finite BF16 outputs and all numerical assertions pass.

### Step 5: Commit the composed kernel

```bash
git add python/sgl_jax/srt/kernels/mla/dsa/kernel.py \
  python/sgl_jax/srt/kernels/mla/dsa/__init__.py \
  python/sgl_jax/test/kernels/test_dsa_decode_mla.py \
  scripts/kernels/falcon_dsa_decode_mla_v7x8.yaml
git commit -m "feat: compose SparseCore gather and DSA MLA attention"
```

## Task 6: Make the benchmark measure the real architecture

**Files:**

- Modify: `benchmark/kernels/mla/bench_dsa_decode_mla.py`
- Modify: `python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py`
- Modify: `scripts/kernels/falcon_dsa_decode_mla_v7x8.yaml`

### Step 1: Write failing benchmark-contract tests

Require benchmark variants for:

- `sparsecore`: Stage A plus Stage B;
- `xla-gather`: XLA gather plus the identical Stage B;
- `gather-only`: SparseCore Stage A;
- `attention-only`: TensorCore Stage B; and
- `dense-mla-v2`: the repository's production dense MLA kernel when its
  fixture contract can be satisfied.

If production MLA v2 integration needs unrelated runtime metadata, keep the
existing dense math workload but label it `dense-jax-baseline`; never call it
the production MLA result.

Add slot distributions `uniform` and `clustered`. Assert packing-two shape,
scale, distribution, per-stage bytes, and implementation name are recorded in
JSON.

### Step 2: Implement benchmark variants and separate compilation timing

Compile each variant once and record compilation latency before warmups. Time
50 warmups and 200 device-synchronized iterations for final runs. Preserve
median, p99, mean, and minimum. Add a smaller smoke mode for remote gates.

### Step 3: Run local benchmark contract tests and commit

```bash
PYTHONPATH=python:. /Users/jiongxuan/workspace/sglang-jax/python/.venv/bin/python \
  -m unittest -v sgl_jax.test.kernels.test_bench_dsa_decode_mla
git add benchmark/kernels/mla/bench_dsa_decode_mla.py \
  python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py \
  scripts/kernels/falcon_dsa_decode_mla_v7x8.yaml
git commit -m "bench: measure staged DSA MLA kernel"
```

## Task 7: Run the Falcon performance matrix only after correctness

**Files:**

- Modify: `scripts/kernels/falcon_dsa_decode_mla_v7x8.yaml`
- Create: `docs/superpowers/reports/2026-07-15-dsa-decode-mla-v2-falcon.md`

### Step 1: Submit a short performance smoke

On `tpuv7x-64-node`, run batch 1 and 8 at context 32K with reduced iterations.
Require all variants to finish and inspect an XProf trace for unexpected VMEM
spill or dense cache work before launching the matrix.

### Step 2: Run the measurement matrix

Use K=2048, TP-local `H=8,L=512,R=64`, GLM scale, and contexts
8K/32K/128K/160K for batch 1/8/32. Run uniform and clustered slots. Add batch
128 only if the 320 MiB intermediate workspace and total device memory are
safe.

Capture separate Stage A, Stage B, end-to-end, XLA-gather, and dense baseline
latencies. Capture standard XProf for batch 1 and 32. Do not enable the LLO
flags rejected by libtpu 0.0.30.

### Step 3: Write the evidence report

Record:

- task IDs, commit SHA, cluster/device topology, JAX and libtpu versions;
- exact commands and fixture layout;
- correctness error summaries;
- compile, median, and p99 latencies;
- estimated/interpreted HBM traffic and trace observations; and
- the measured sparse-vs-dense dispatch crossover.

State clearly whether the v2 hypothesis is confirmed, partially confirmed, or
rejected. No performance conclusion may be based only on the XLA gather path.

### Step 4: Commit the report and final Falcon spec

```bash
git add scripts/kernels/falcon_dsa_decode_mla_v7x8.yaml \
  docs/superpowers/reports/2026-07-15-dsa-decode-mla-v2-falcon.md
git commit -m "docs: report Falcon DSA MLA v2 results"
```

## Task 8: Final verification

Run from a clean worktree:

```bash
PYTHONPATH=python:. /Users/jiongxuan/workspace/sglang-jax/python/.venv/bin/python \
  -m unittest -v sgl_jax.test.kernels.test_dsa_decode_mla \
  sgl_jax.test.kernels.test_bench_dsa_decode_mla
ruff check \
  python/sgl_jax/srt/kernels/mla/dsa \
  python/sgl_jax/test/kernels/test_dsa_decode_mla.py \
  python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py \
  benchmark/kernels/mla/bench_dsa_decode_mla.py
git diff --check
git status --short
```

Then rerun only the three Falcon correctness gates against the final commit.
Claim completion only when local tests, Ruff, final TPU correctness, and the
evidence report all agree.
