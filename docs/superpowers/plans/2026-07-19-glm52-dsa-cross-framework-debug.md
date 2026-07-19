# GLM-5.2 DSA Cross-Framework Golden and Layer Debug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish an independent PyTorch CPU golden for GLM-5.2 DSA selection and sparse MLA, add opt-in layerwise JAX dumps, and use both to diagnose short-context DSA/FA drift and validate true sparse contexts on Falcon v7x-32.

**Architecture:** Keep the numerical oracle independent from JAX/Pallas: deterministic fixtures flow through pure PyTorch CPU selection, logical-to-physical mapping, and FP32 sparse MLA before being compared with JAX and TPU Pallas. Add PR-1062-style `jax.debug.callback` dumps behind disabled-by-default environment gates, with semantic metadata that lets multi-host output be aligned by component, layer, mode, and occurrence. Reuse the real-weight runner for short and boundary profiles, then distinguish selection failures, kernel failures, and accumulated model-level drift.

**Tech Stack:** Python 3.12, PyTorch CPU, JAX/Flax, NumPy, pytest, Pallas TPU, Bash, Falcon v7x-32.

---

## File map

- Create `python/sgl_jax/srt/kernels/dsa/torch_reference.py`: pure PyTorch CPU selection, logical-to-physical mapping, and sparse MLA oracle; never imported by serving code.
- Create `python/sgl_jax/test/test_dsa_cross_framework.py`: deterministic cross-framework length matrix and exact/numerical gates.
- Modify `python/sgl_jax/srt/utils/debug_utils.py`: disabled-by-default JAX array dump callback and JSONL manifest.
- Create `python/sgl_jax/test/test_debug_tensor_dump.py`: dump enable/filter/file/manifest tests.
- Modify `python/sgl_jax/srt/models/glm5_moe.py`: Indexer, decoder-layer, final-hidden, and logits dump points.
- Modify `python/sgl_jax/srt/layers/attention/dsa_backend.py`: selection and sparse-MLA boundary dump points.
- Modify `python/sgl_jax/test/test_dsa_backend.py`: verify DSA semantic dump calls without filesystem I/O.
- Modify `python/sgl_jax/test/test_dsa_glm52.py`: verify model-level semantic dump calls.
- Create `scripts/kernels/compare_debug_tensor_dumps.py`: align dump manifests and report shape/dtype/max/mean/p99/cosine/top-k metrics.
- Create `python/sgl_jax/test/test_debug_tensor_compare.py`: comparator tests, including mismatch failures.
- Modify `scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh`: dump directory propagation and `smoke`/`boundary` request profiles.
- Modify `python/sgl_jax/test/test_glm52_e2e_compare.py`: runner profile and dump environment assertions.
- Modify `note/2026-07-18-glm52-dsa-falcon-results.md`: commands, artifacts, cross-framework gates, root cause, and remaining limits.

### Task 1: Pure PyTorch CPU Selection and Sparse-MLA Golden

**Files:**
- Create: `python/sgl_jax/srt/kernels/dsa/torch_reference.py`
- Create: `python/sgl_jax/test/test_dsa_cross_framework.py`

- [x] **Step 1: Write the failing PyTorch selection test**

Add a deterministic test that imports `torch_glm_dsa_select` and compares score, logical IDs, and counts with `GlmDsaIndexer.select_topk` for candidate lengths `1, 127, 128, 129, 257, 2047, 2048, 2049, 3072, 4096`. Quantize source inputs to BF16 first, then compute both references in FP32. Require score `allclose`, exact selected counts, and exact IDs for fixtures whose Top-K boundary has no tie.

```python
@pytest.mark.parametrize(
    "candidate_len", [1, 127, 128, 129, 257, 2047, 2048, 2049, 3072, 4096]
)
def test_torch_cpu_selection_matches_jax_across_context_boundaries(candidate_len):
    fixture = make_selection_fixture(candidate_len, index_topk=2048)
    torch_result = torch_glm_dsa_select(**fixture.torch_inputs, index_topk=2048)
    jax_ids, jax_counts = GlmDsaIndexer.select_topk(
        **fixture.jax_inputs, index_topk=2048
    )
    np.testing.assert_allclose(torch_result.scores.numpy(), fixture.jax_scores(), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(torch_result.selected_counts.numpy(), jax_counts)
    np.testing.assert_array_equal(torch_result.logical_topk_ids.numpy(), jax_ids)
```

- [x] **Step 2: Run the selection test and verify RED**

Run:

```bash
JAX_PLATFORMS=cpu ../../.venv/bin/python -m pytest -q \
  python/sgl_jax/test/test_dsa_cross_framework.py::test_torch_cpu_selection_matches_jax_across_context_boundaries
```

Expected: collection/import failure because `torch_reference.py` does not exist.

- [x] **Step 3: Implement the minimal independent selection oracle**

Implement the score formula with pure Torch operations and explicit FP32:

```python
logits = torch.einsum("thd,tcd->tch", q_index.float(), candidate_keys.float())
scores = (torch.relu(logits) * head_weights.float()[:, None, :]).sum(dim=-1)
scores *= q_index.shape[2] ** -0.5 * q_index.shape[1] ** -0.5
scores = scores.masked_fill(~candidate_valid, -torch.inf)
values, offsets = torch.topk(scores, k=index_topk, dim=-1, sorted=True)
logical_ids = torch.gather(padded_logical_ids, 1, offsets)
```

Return a frozen `TorchDsaSelectionResult(scores, logical_topk_ids, selected_counts)`. Validate ranks, dtypes, dimensions, positive Top-K, and empty-candidate behavior independently of JAX.

- [x] **Step 4: Run the selection test and verify GREEN**

Run the command from Step 2. Expected: all parameterized cases pass.

- [x] **Step 5: Write failing physical-slot and sparse-MLA tests**

Add tests that send the Torch-selected logical IDs through both mapping implementations and then through both attention implementations. Cover shuffled physical pages, causal removal, duplicates, padding, ragged counts, and selected-count boundaries `0, 1, 127, 128, 129, 2047, 2048`.

```python
torch_mapping = torch_logical_topk_to_physical_slots(
    logical_topk_ids=torch_result.logical_topk_ids,
    selected_counts=torch_result.selected_counts,
    req_to_token_slots=torch_mapping_table,
    query_request_indices=torch.tensor([0], dtype=torch.int32),
    query_positions=torch.tensor([candidate_len - 1], dtype=torch.int32),
)
jax_mapping = logical_topk_to_physical_slots(...)
torch.testing.assert_close(torch_mapping.physical_slots, torch.from_numpy(np.asarray(jax_mapping.physical_slots)), rtol=0, atol=0)
torch.testing.assert_close(torch_mapping.selected_counts, torch.from_numpy(np.asarray(jax_mapping.selected_counts)), rtol=0, atol=0)

torch_output = torch_dsa_sparse_mla(...)
jax_output = dsa_sparse_mla_reference(...)
np.testing.assert_allclose(torch_output.numpy(), np.asarray(jax_output), rtol=1e-5, atol=1e-5)
```

- [x] **Step 6: Run the new tests and verify RED**

Expected: missing `torch_logical_topk_to_physical_slots` and `torch_dsa_sparse_mla` imports.

- [x] **Step 7: Implement mapping and sparse MLA oracles**

Use straightforward CPU loops for validity/duplicate-preserving compaction. Flatten packed cache token rows, gather counted physical slots, concatenate latent and RoPE keys, and use FP32 `torch.softmax` and einsum. `selected_counts` is the sole padding authority; reject invalid counted slots.

- [x] **Step 8: Verify Task 1**

Run:

```bash
JAX_PLATFORMS=cpu ../../.venv/bin/python -m pytest -q \
  python/sgl_jax/test/test_dsa_cross_framework.py \
  python/sgl_jax/test/test_dsa_reference.py \
  python/sgl_jax/test/test_dsa_glm52.py
```

Expected: all tests pass; target-shape selection and sparse MLA produce finite output.

- [x] **Step 9: Commit Task 1**

```bash
git add python/sgl_jax/srt/kernels/dsa/torch_reference.py \
  python/sgl_jax/test/test_dsa_cross_framework.py
git commit -m "test(dsa): add PyTorch CPU cross-framework golden"
```

### Task 2: Opt-in JAX Tensor Dump Utility

**Files:**
- Modify: `python/sgl_jax/srt/utils/debug_utils.py`
- Create: `python/sgl_jax/test/test_debug_tensor_dump.py`

- [x] **Step 1: Write failing disabled/filter tests**

Monkeypatch `jax.debug.callback` and assert no callback occurs by default. Enable `SGLANG_JAX_DEBUG_DUMP=1`, then verify component, layer, and process filters independently allow or reject a callback.

```python
def test_debug_dump_is_disabled_by_default(monkeypatch):
    calls = []
    monkeypatch.delenv("SGLANG_JAX_DEBUG_DUMP", raising=False)
    monkeypatch.setattr(jax.debug, "callback", lambda *args, **kwargs: calls.append(args))
    maybe_dump_jax_array(jnp.ones((2,)), component="dsa", name="q", layer_id=3)
    assert calls == []
```

- [x] **Step 2: Run tests and verify RED**

Expected: import failure for `maybe_dump_jax_array`.

- [x] **Step 3: Implement disabled-by-default gating**

Port the PR-1062 pattern with these environment variables:

```text
SGLANG_JAX_DEBUG_DUMP=0|1
SGLANG_JAX_DEBUG_DUMP_DIR=<directory>
SGLANG_JAX_DEBUG_DUMP_COMPONENTS=a,b
SGLANG_JAX_DEBUG_DUMP_LAYERS=3,39,75
SGLANG_JAX_DEBUG_DUMP_PROCESSES=0,1
```

Use sanitized semantic filenames containing process/component/layer/mode/name/occurrence. Capture the process ID before entering the host callback.

- [x] **Step 4: Verify gating tests GREEN**

Run the focused test file. Expected: all gating tests pass.

- [x] **Step 5: Write failing file and manifest test**

Call the captured callback with a NumPy array and require one `.npy` plus one `manifest-pNNNNN.jsonl` row containing filename, process, component, layer, forward mode, occurrence, shape, and dtype.

- [x] **Step 6: Implement atomic dump and manifest writing**

Use a process-local lock, `np.save`, and one JSON object per line. Never perform filesystem work unless dumping is enabled. Ensure callback arguments are host values and filenames cannot escape the dump directory.

- [x] **Step 7: Verify and commit Task 2**

```bash
JAX_PLATFORMS=cpu ../../.venv/bin/python -m pytest -q \
  python/sgl_jax/test/test_debug_tensor_dump.py
git add python/sgl_jax/srt/utils/debug_utils.py \
  python/sgl_jax/test/test_debug_tensor_dump.py
git commit -m "feat(debug): add opt-in JAX tensor dumps"
```

### Task 3: DSA Semantic Instrumentation

**Files:**
- Modify: `python/sgl_jax/srt/models/glm5_moe.py`
- Modify: `python/sgl_jax/srt/layers/attention/dsa_backend.py`
- Modify: `python/sgl_jax/test/test_dsa_backend.py`
- Modify: `python/sgl_jax/test/test_dsa_glm52.py`

- [x] **Step 1: Write failing backend instrumentation test**

Monkeypatch the module-level dumper and run the existing tiny backend fixture. Assert semantic names include `q_index`, `head_weights`, `index_k`, `logical_topk_ids`, `selected_counts`, `physical_slots`, `q_latent`, `q_rope`, and `o_latent`, each with the producing layer ID and forward mode.

- [x] **Step 2: Verify RED**

Expected: dump call list is empty.

- [x] **Step 3: Add minimal backend dump points**

Insert dumps immediately after Indexer projection, after selection/mapping, before sparse MLA dispatch, and after output. Do not dump the full Index-K or MLA cache; dump only current write inputs and selection metadata.

- [x] **Step 4: Verify backend test GREEN**

Run focused backend and GLM tests.

- [x] **Step 5: Write failing model instrumentation test**

Use the existing tiny decoder/model fakes and assert dumps for embedding output, `attention_output`, `residual_post_attention`, `mlp_output`, `hidden_states_post_mlp`, final normalized hidden state, and `next_token_logits`.

- [x] **Step 6: Add model dump points**

Follow PR #1062 component naming, passing `forward_batch.forward_mode` and exact layer IDs. Dump `output.next_token_logits` after `LogitsProcessor`; do not attempt to dump the output dataclass itself.

- [x] **Step 7: Verify and commit Task 3**

```bash
JAX_PLATFORMS=cpu ../../.venv/bin/python -m pytest -q \
  python/sgl_jax/test/test_dsa_backend.py \
  python/sgl_jax/test/test_dsa_glm52.py \
  python/sgl_jax/test/test_debug_tensor_dump.py
git add python/sgl_jax/srt/models/glm5_moe.py \
  python/sgl_jax/srt/layers/attention/dsa_backend.py \
  python/sgl_jax/test/test_dsa_backend.py \
  python/sgl_jax/test/test_dsa_glm52.py
git commit -m "feat(debug): instrument GLM DSA layer boundaries"
```

### Task 4: Dump Comparator and Real-E2E Profiles

**Files:**
- Create: `scripts/kernels/compare_debug_tensor_dumps.py`
- Create: `python/sgl_jax/test/test_debug_tensor_compare.py`
- Modify: `scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh`
- Modify: `python/sgl_jax/test/test_glm52_e2e_compare.py`

- [x] **Step 1: Write failing comparator tests**

Create two temporary dump directories with manifests and `.npy` files. Require alignment by `(component, layer, mode, name, occurrence, process)`, shape/dtype validation, and numerical fields `max_abs`, `mean_abs`, `p99_abs`, `cosine`, `topk_overlap` where applicable. Missing or duplicate semantic keys must fail.

- [x] **Step 2: Verify comparator tests RED**

Expected: script module cannot be loaded.

- [x] **Step 3: Implement comparator**

Read JSONL manifests, load matching arrays, promote floating values to FP32, and emit a deterministic JSON report. CLI thresholds are optional filters; the report must always retain raw metrics and the first failing key.

- [x] **Step 4: Verify comparator tests GREEN**

Run the focused comparator tests.

- [x] **Step 5: Write failing runner-profile tests**

Assert the runner supports:

```text
GLM52_DSA_REQUEST_PROFILE=smoke     # existing 4/257/9+133 inputs
GLM52_DSA_REQUEST_PROFILE=boundary  # 2047/2048/2049/3072 inputs
GLM52_DSA_MAX_NEW_TOKENS=1
```

When dumping is enabled, each rank must export `SGLANG_JAX_DEBUG_DUMP_DIR=$OUT/debug_dumps`. Boundary requests use one generated token, `ignore_eos=true`, and input IDs within vocabulary range.

- [x] **Step 6: Verify runner tests RED**

Expected: missing profile strings and dump directory export.

- [x] **Step 7: Implement request profiles and dump propagation**

Pass profile and max-new-token values into the Python request generator. Derive expected prompt/completion counts from the generated request files instead of hardcoding smoke-only values. Keep current defaults unchanged.

- [x] **Step 8: Verify and commit Task 4**

```bash
JAX_PLATFORMS=cpu ../../.venv/bin/python -m pytest -q \
  python/sgl_jax/test/test_debug_tensor_compare.py \
  python/sgl_jax/test/test_glm52_e2e_compare.py
bash -n scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh
git add scripts/kernels/compare_debug_tensor_dumps.py \
  scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh \
  python/sgl_jax/test/test_debug_tensor_compare.py \
  python/sgl_jax/test/test_glm52_e2e_compare.py
git commit -m "test(falcon): add GLM DSA layer-dump profiles"
```

### Task 5: Local and TPU Kernel Matrix

**Files:**
- Modify if evidence requires a fix: `python/sgl_jax/srt/kernels/mla/dsa/kernel.py`
- Modify if evidence requires a fix: `python/sgl_jax/srt/models/glm5_moe.py`
- Test: `python/sgl_jax/test/test_dsa_cross_framework.py`
- Test: `python/sgl_jax/test/kernels/test_dsa_decode_mla.py`

- [x] **Step 1: Run the complete CPU cross-framework matrix**

```bash
JAX_PLATFORMS=cpu ../../.venv/bin/python -m pytest -q \
  python/sgl_jax/test/test_dsa_cross_framework.py \
  python/sgl_jax/test/test_dsa_backend.py \
  python/sgl_jax/test/test_dsa_reference.py \
  python/sgl_jax/test/test_dsa_glm52.py \
  python/sgl_jax/test/kernels/test_dsa_decode_mla.py
```

Expected: exact selection/count/slot gates and finite sparse output across all lengths.

- [x] **Step 2: Run target-shape Pallas against PyTorch CPU fixtures on Falcon**

Run v7x-32 cases with page size 128, global/local heads 64/2, latent/RoPE 512/64, Top-K 2048, and candidate lengths `127,128,129,2047,2048,2049,3072,4096`. Save raw max/mean/p99 error; start with the existing `rtol=2e-2, atol=1e-2` gate but do not relax it to hide a failure.

- [x] **Step 3: If a gate fails, complete systematic root-cause phases**

Record the first failing tensor and case, reproduce it with a single fixture, compare Torch score/IDs/slots and Torch/JAX/Pallas attention in order, state one hypothesis, and add a failing regression test before changing production code.

- [x] **Step 4: Commit only evidence-backed fixes**

Use a focused `fix(dsa): ...` commit per confirmed root cause. If no failures occur, make no production-kernel change.

### Task 6: Real-Weight Layerwise E2E Diagnosis

**Files:**
- Modify: `note/2026-07-18-glm52-dsa-falcon-results.md`

- [x] **Step 1: Run short DSA and FA dumps**

Run the smoke profile with one generated token. First dump all `decoder_layer` outputs to find the first divergent layer; then rerun only that layer and its preceding full Indexer layer with DSA internals enabled.

Recommended filters:

```text
SGLANG_JAX_DEBUG_DUMP_COMPONENTS=embed,decoder_layer,final,logits
SGLANG_JAX_DEBUG_DUMP_LAYERS=0,1,...,77
```

and narrowed rerun:

```text
SGLANG_JAX_DEBUG_DUMP_COMPONENTS=dsa_indexer,dsa_selection,dsa_attention,decoder_layer,logits
SGLANG_JAX_DEBUG_DUMP_LAYERS=<producer>,<first-divergent>
```

- [x] **Step 2: Compare short dumps**

Generate a JSON report for DSA vs FA. For all-visible contexts, identify whether drift starts in selection ordering/Pallas `o_latent`, post-`o_proj`, or later accumulation. Do not infer root cause from final logprob alone.

- [x] **Step 3: Run true-sparse boundary profile**

Run DSA at `2047/2048/2049/3072`, verify exact `selected_counts=min(visible,2048)`, no future logical IDs, valid physical slots, and actual truncation for lengths above 2048. Compare selected fixtures with PyTorch CPU golden and Pallas output with Torch FP32 sparse MLA.

Execution note: the real-weight run used one 3072-token prompt, whose per-query dumps cover
every position from 0 through 3071, including 2046/2047/2048. The separate Torch/JAX/Pallas
matrix covered terminal candidate lengths 2047/2048/2049/3072/4096.

- [x] **Step 4: Diagnose and fix only confirmed discrepancies**

For every failure, follow Task 5 Step 3 and TDD. Keep dense-vs-sparse quality difference separate from kernel numerical error.

- [x] **Step 5: Update evidence document**

Record source revisions, exact Falcon experiments, fixture seeds, length matrix, per-layer first divergence, selection exactness, kernel max/mean/p99 errors, final-logit metrics, artifact paths, and any unresolved quality/performance limits.

- [x] **Step 6: Commit results**

```bash
git add note/2026-07-18-glm52-dsa-falcon-results.md
git commit -m "docs(falcon): record GLM DSA cross-framework diagnosis"
```

### Task 7: Final Verification and Review

**Files:** all files above.

- [x] **Step 1: Run regression suite and static checks**

```bash
JAX_PLATFORMS=cpu ../../.venv/bin/python -m pytest -q \
  python/sgl_jax/test/test_debug_tensor_dump.py \
  python/sgl_jax/test/test_debug_tensor_compare.py \
  python/sgl_jax/test/test_dsa_cross_framework.py \
  python/sgl_jax/test/test_dsa_backend.py \
  python/sgl_jax/test/test_dsa_reference.py \
  python/sgl_jax/test/test_dsa_glm52.py \
  python/sgl_jax/test/test_glm52_e2e_compare.py \
  python/sgl_jax/test/kernels/test_dsa_decode_mla.py
ruff check <all changed Python files>
bash -n scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh
git diff --check
```

- [x] **Step 2: Review spec compliance**

Confirm independent Torch CPU computation, exact Selection gates, target token-length coverage, disabled-by-default dumps, stable multi-process manifests, short layer diagnosis, and true-sparse E2E evidence.

- [x] **Step 3: Review code quality**

Check optional debug overhead, callback safety, manifest determinism, test runtime, fixture memory, error messages, and that no Torch oracle is imported by serving paths.

- [x] **Step 4: Report outcome precisely**

Separate: kernel mathematical correctness, selection correctness, short all-visible numerical drift, long sparse approximation quality, unsupported serving modes, and performance. Never call the 0.05 output-logprob gate a full-logits golden.

## Plan self-review

- Spec coverage: PR-1062-style E2E dumps, independent PyTorch CPU select+DSA golden, page/Top-K/long/ragged length matrix, Falcon execution, and systematic debugging are each assigned to explicit tasks.
- Placeholder scan: no TBD/TODO or unspecified implementation steps remain; production fixes are intentionally conditional on observed failing evidence.
- Type consistency: Torch selection returns logical IDs/counts/scores; mapping consumes logical IDs/counts and returns physical slots/counts; sparse MLA consumes physical slots/counts. Dump and comparator semantic keys use the same process/component/layer/mode/name/occurrence fields.
