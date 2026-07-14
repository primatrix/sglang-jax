# DSA Decode MLA Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify an isolated Pallas TPU kernel that performs decode MLA only over preselected physical KV slots.

**Architecture:** Keep the reference implementation and Pallas implementation separate.  Both consume the current packed MLA KV cache plus `topk_slots[B, K]`; the reference gathers selected slots in JAX and the Pallas path updates online-softmax state without materializing a selected-KV HBM buffer.  No model-runner or indexer code changes are in this slice.

**Tech Stack:** Python 3.12, JAX, Pallas TPU, NumPy, pytest/unittest, XProf and Falcon.

---

## File structure

- Create: `python/sgl_jax/srt/kernels/mla/dsa/__init__.py` — public exports.
- Create: `python/sgl_jax/srt/kernels/mla/dsa/reference.py` — validation plus two independent selected-slot MLA references.
- Create: `python/sgl_jax/srt/kernels/mla/dsa/kernel.py` — TPU-only Pallas online-softmax kernel and public dispatch wrapper.
- Create: `python/sgl_jax/test/kernels/test_dsa_decode_mla.py` — CPU reference tests and TPU-gated Pallas tests.
- Create: `benchmark/kernels/mla/bench_dsa_decode_mla.py` — reproducible dense-versus-sparse decode microbenchmark.

### Task 1: Selected-slot reference and contract validation

**Files:**
- Create: `python/sgl_jax/srt/kernels/mla/dsa/reference.py`
- Create: `python/sgl_jax/srt/kernels/mla/dsa/__init__.py`
- Create: `python/sgl_jax/test/kernels/test_dsa_decode_mla.py`

- [ ] **Step 1: Write the failing reference-oracle test**

```python
def test_selected_slot_reference_matches_independent_dense_gather():
    ql_nope, q_pe, cache_kv, slots, valid_counts = _make_inputs(dtype=jnp.float32)
    got = reference_dsa_decode_mla_attention(
        ql_nope, q_pe, cache_kv, slots, valid_counts, sm_scale=1 / jnp.sqrt(256)
    )
    expected = dense_selected_mla_attention(
        ql_nope, q_pe, cache_kv, slots, valid_counts, sm_scale=1 / jnp.sqrt(256)
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
```

`_make_inputs` must create a `[3, 8, 1, 256]` packed cache, two batch rows,
two heads, slots `[[0, 7, 8, 19, -1], [23, 1, 8, -1, -1]]`, and valid counts
`[4, 3]`.  This exercises page boundaries, non-monotonic ordering, and
padding without needing a TPU.

- [ ] **Step 2: Run the new test and confirm the missing-module failure**

Run: `PYTHONPATH=python python/.venv/bin/python -m pytest python/sgl_jax/test/kernels/test_dsa_decode_mla.py::test_selected_slot_reference_matches_independent_dense_gather -q`

Expected: FAIL during collection because `sgl_jax.srt.kernels.mla.dsa.reference` does not exist.

- [ ] **Step 3: Implement the two references and static validation**

```python
def _decode_slots(cache_kv, slots, valid):
    page_size = cache_kv.shape[1] * cache_kv.shape[2]
    safe_slots = jnp.where(valid, slots, 0)
    pages, offsets = divmod(safe_slots, page_size)
    rows, cols = divmod(offsets, cache_kv.shape[2])
    return cache_kv[pages, rows, cols]


def reference_dsa_decode_mla_attention(ql_nope, q_pe, cache_kv, topk_slots, valid_counts, *, sm_scale):
    valid = jnp.arange(topk_slots.shape[1])[None, :] < valid_counts[:, None]
    selected = _decode_slots(cache_kv, topk_slots, valid)
    lkv_dim = align_to(ql_nope.shape[-1], 128)
    r_dim = align_to(q_pe.shape[-1], 128)
    key = jnp.concatenate([selected[..., :lkv_dim], selected[..., lkv_dim:lkv_dim + r_dim]], -1)
    logits = jnp.einsum("bhd,bkd->bhk", jnp.concatenate([ql_nope, q_pe], -1), key,
                        preferred_element_type=jnp.float32) * sm_scale
    probs = jax.nn.softmax(jnp.where(valid[:, None, :], logits, -jnp.inf), axis=-1)
    return jnp.einsum("bhk,bkd->bhd", probs, selected[..., :lkv_dim],
                      preferred_element_type=jnp.float32)
```

`dense_selected_mla_attention` must independently flatten the packed cache to
`[pages * page_size, dim]`, gather it with the same safe slots, and calculate
the same expression.  Both references must zero-pad `ql_nope` and `q_pe` to
their respective 128-aligned cache dimensions before concatenating Q.  This is
required for the GLM `qk_rope_head_dim=64` production shape.  `_validate_inputs` must reject non-3D Q tensors,
non-4D cache tensors, non-`int32` slots/counts, mismatched batch widths,
invalid count ranges, a negative slot in a valid position, an out-of-capacity
slot, and zero valid counts with `ValueError`.

- [ ] **Step 4: Run reference tests and confirm they pass**

Run: `PYTHONPATH=python python/.venv/bin/python -m pytest python/sgl_jax/test/kernels/test_dsa_decode_mla.py -q -k 'reference or validation'`

Expected: PASS for reference and validation tests; TPU tests are skipped on a CPU-only workstation.

- [ ] **Step 5: Commit the completed reference contract**

```bash
git add python/sgl_jax/srt/kernels/mla/dsa/__init__.py \
  python/sgl_jax/srt/kernels/mla/dsa/reference.py \
  python/sgl_jax/test/kernels/test_dsa_decode_mla.py
git commit -m "feat(kernels): add DSA decode MLA reference"
```

### Task 2: TPU Pallas decode implementation

**Files:**
- Create: `python/sgl_jax/srt/kernels/mla/dsa/kernel.py`
- Modify: `python/sgl_jax/srt/kernels/mla/dsa/__init__.py`
- Modify: `python/sgl_jax/test/kernels/test_dsa_decode_mla.py`

- [ ] **Step 1: Write the TPU-only failing Pallas comparison test**

```python
@pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="requires TPU Pallas")
def test_pallas_decode_matches_selected_slot_reference():
    from sgl_jax.srt.kernels.mla.dsa.kernel import dsa_decode_mla_attention
    ql_nope, q_pe, cache_kv, slots, valid_counts = _make_inputs(dtype=jnp.bfloat16)
    got = dsa_decode_mla_attention(
        ql_nope, q_pe, cache_kv, slots, valid_counts, sm_scale=1 / jnp.sqrt(256)
    )
    expected = reference_dsa_decode_mla_attention(
        ql_nope, q_pe, cache_kv, slots, valid_counts, sm_scale=1 / jnp.sqrt(256)
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=2e-2, atol=1e-2)
```

- [ ] **Step 2: Run on a Falcon TPU worker and verify the expected missing-module failure**

Run: `PYTHONPATH=python python/.venv/bin/python -m pytest python/sgl_jax/test/kernels/test_dsa_decode_mla.py::test_pallas_decode_matches_selected_slot_reference -q`

Expected: FAIL during test execution because `kernel.py` does not exist.

- [ ] **Step 3: Implement a one-program-per-request online-softmax kernel**

```python
def _dsa_decode_mla_kernel(ql_nope_ref, q_pe_ref, cache_kv_ref, slots_ref, counts_ref, out_ref, *, sm_scale):
    b = pl.program_id(0)
    q = jnp.concatenate([ql_nope_ref[b], q_pe_ref[b]], axis=-1).astype(jnp.float32)
    m = jnp.full((q.shape[0],), -jnp.inf, jnp.float32)
    l = jnp.zeros((q.shape[0],), jnp.float32)
    acc = jnp.zeros((q.shape[0], ql_nope_ref.shape[-1]), jnp.float32)
    for k_idx in range(slots_ref.shape[1]):
        slot = slots_ref[b, k_idx]
        valid = k_idx < counts_ref[b]
        # Decode page/packed-row/packed-column, read one selected KV vector,
        # then merge its score/value into (m, l, acc) with online softmax.
    out_ref[b] = (acc / l[:, None]).astype(ql_nope_ref.dtype)
```

Use `pl.pallas_call` with HBM `BlockSpec`s for all inputs, a grid of
`(batch_size,)`, and `pltpu.VMEM` scratch for `m`, `l` and `acc`.  The valid branch
must perform the selected packed-cache read; the invalid branch must update no
online-softmax state.  The public wrapper calls `_validate_inputs`, rejects a
non-TPU backend, preserves static `K` in the compilation cache key, and
returns `[B, H, kv_lora_rank]`.  It exposes `validate=True` for direct/tests;
the benchmark uses `validate=False` after fixture construction so host-side
value validation is not included in the timed kernel path.

- [ ] **Step 4: Re-run the targeted TPU numerical test**

Run: `PYTHONPATH=python python/.venv/bin/python -m pytest python/sgl_jax/test/kernels/test_dsa_decode_mla.py::test_pallas_decode_matches_selected_slot_reference -q`

Expected: PASS with BF16 output within `rtol=2e-2`, `atol=1e-2`.

- [ ] **Step 5: Add and run boundary/ordering TPU tests**

```python
@pytest.mark.parametrize("page_size", [8, 16, 32, 64])
def test_pallas_handles_boundaries_duplicates_and_padding(page_size):
    ql_nope, q_pe, cache_kv, slots, counts = _make_inputs(
        dtype=jnp.bfloat16, page_size=page_size, duplicate_slots=True
    )
    got = dsa_decode_mla_attention(ql_nope, q_pe, cache_kv, slots, counts, sm_scale=1 / jnp.sqrt(256))
    expected = reference_dsa_decode_mla_attention(ql_nope, q_pe, cache_kv, slots, counts, sm_scale=1 / jnp.sqrt(256))
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=2e-2, atol=1e-2)
```

Run: `PYTHONPATH=python python/.venv/bin/python -m pytest python/sgl_jax/test/kernels/test_dsa_decode_mla.py -q`

Expected: PASS on TPU; all Pallas tests are skipped rather than emulated on CPU.

- [ ] **Step 6: Commit the Pallas kernel**

```bash
git add python/sgl_jax/srt/kernels/mla/dsa/kernel.py \
  python/sgl_jax/srt/kernels/mla/dsa/__init__.py \
  python/sgl_jax/test/kernels/test_dsa_decode_mla.py
git commit -m "feat(kernels): add sparse DSA MLA decode kernel"
```

### Task 3: Reproducible Falcon microbenchmark

**Files:**
- Create: `benchmark/kernels/mla/bench_dsa_decode_mla.py`

- [ ] **Step 1: Write the benchmark smoke test first**

```python
def test_sparse_benchmark_case_returns_latency():
    result = benchmark_case(batch_size=1, context_len=8192, topk=2048, tries=3)
    assert result.sparse_ms > 0
    assert result.dense_ms > 0
```

The test is TPU-only and uses the same deterministic packed-cache fixture as
the kernel tests.

- [ ] **Step 2: Run the smoke test on Falcon and verify it fails because the benchmark module is missing**

Run: `PYTHONPATH=python python/.venv/bin/python -m pytest benchmark/kernels/mla/bench_dsa_decode_mla.py -q`

Expected: FAIL during collection because the benchmark file does not exist.

- [ ] **Step 3: Implement sparse/dense timing with separate compilation accounting**

```python
@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    compile_ms: float
    sparse_ms: float
    dense_ms: float
    sparse_p99_ms: float
    dense_p99_ms: float


def benchmark_case(*, batch_size: int, context_len: int, topk: int, tries: int = 200) -> BenchmarkResult:
    sparse = functools.partial(dsa_decode_mla_attention, sm_scale=SM_SCALE, validate=False)
    dense = functools.partial(mla_ragged_paged_attention, sm_scale=SM_SCALE)
    # Block once for compile, execute 50 warm-ups, then record `tries` blocked calls.
```

Build the dense invocation with `kv_lens=[context_len] * B`, `cu_q_lens` for
one decode query per request, contiguous ragged `page_indices`, page-aligned
`cu_kv_lens`, and zero `new_kv_c/new_k_pe`; it must call the existing
`mla_ragged_paged_attention` kernel.  Use
`multiple_iteration_timeit_from_trace` for timed iterations and print a single
JSON line per case containing device name, JAX version, dimensions and all five
measured values.  Dense and sparse outputs are intentionally not compared by
the benchmark because their KV sets differ; correctness remains in Task 2.

- [ ] **Step 4: Run the Falcon benchmark matrix**

Run: `PYTHONPATH=python python/.venv/bin/python benchmark/kernels/mla/bench_dsa_decode_mla.py --batch-sizes 1,8,32,128 --context-lens 8192,32768,131072,163840 --topk 2048 --tries 200`

Expected: every case emits a JSON measurement and exits 0.  Capture XProf for
`(B, context) = (1, 160K)` and `(32, 160K)` in a separate output directory.

- [ ] **Step 5: Commit benchmark support**

```bash
git add benchmark/kernels/mla/bench_dsa_decode_mla.py
git commit -m "bench(kernels): add DSA MLA decode benchmark"
```

### Task 4: End-to-end verification and report

**Files:**
- Modify: `docs/superpowers/specs/2026-07-14-dsa-decode-mla-kernel-design.md` — record actual Falcon command, SKU, test result and benchmark artifact paths.

- [ ] **Step 1: Run the complete local reference suite**

Run: `PYTHONPATH=python python/.venv/bin/python -m pytest python/sgl_jax/test/kernels/test_dsa_decode_mla.py -q`

Expected: all reference/validation tests pass; TPU-only tests are skipped if this machine has no TPU.

- [ ] **Step 2: Run the complete Falcon TPU suite**

Run: `PYTHONPATH=python python/.venv/bin/python -m pytest python/sgl_jax/test/kernels/test_dsa_decode_mla.py -q`

Expected: all tests pass with no skipped TPU test.

- [ ] **Step 3: Inspect the two XProf traces**

Run: `xprof --logdir /tmp/dsa-decode-mla-xprof`

Expected: the report contains device HBM traffic and no VMEM spill warning for the sparse kernel.  Record observed values and trace paths in the design document.

- [ ] **Step 4: Commit verification metadata**

```bash
git add docs/superpowers/specs/2026-07-14-dsa-decode-mla-kernel-design.md
git commit -m "docs: record DSA decode kernel verification"
```
