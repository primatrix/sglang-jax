# GLM-5.2 DSA Real-Weight E2E Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate that the current GLM-5.2 DSA implementation loads the complete real checkpoint and produces functionally correct prefill/decode results on Falcon v7x-32, with numerical evidence against both the JAX reference and dense MLA for short all-visible sequences.

**Architecture:** Keep the existing v7x-32 debug experiment as the execution host and transfer only reviewed local files through `falcon exp cp`. A real-weight runner launches one four-rank server at a time and writes durable run context, server logs, request payloads, and responses into the Falcon artifact mount. Kernel precision is checked independently against the JAX sparse reference; model-level precision uses short sequences where `index_topk=2048` includes every causal token, so DSA and dense MLA implement the same attention set.

**Tech Stack:** Bash, Python 3.12, pytest, JAX 0.9.0, libtpu 0.0.34, SGLang-JAX native HTTP API, Falcon v7x-32 (`replica: 4`, `device_topo: 2x2x4`).

---

### Task 1: Checkpoint completion gate

**Files:**
- Modify: `note/2026-07-18-glm52-dsa-falcon-results.md`

- [ ] **Step 1: Monitor the downloader through Falcon**

Run `falcon exp get exp-q7odgo8q9x --output json` and use `falcon exp exec exp-q7odgo8q9x --rank 0 -- ...` to inspect `/models/GLM-5.2`. Do not use a provider CLI or direct bucket API.

- [ ] **Step 2: Validate the index-declared shard set**

Run a Python validator inside the Falcon pod that reads `model.safetensors.index.json`, requires exactly 282 unique shard names, requires every named file to exist and be non-empty, and reports their total byte count.

- [ ] **Step 3: Require the completion marker**

Do not start model loading until `/models/GLM-5.2/_DOWNLOAD_COMPLETE` exists and the downloader log contains `GLM52_CHECKPOINT_VALIDATED shards=282` followed by `GLM52_CHECKPOINT_DOWNLOAD_COMPLETE`.

### Task 2: Result comparison utility (TDD)

**Files:**
- Create: `python/sgl_jax/test/test_glm52_e2e_compare.py`
- Create: `scripts/kernels/compare_glm52_e2e_results.py`

- [ ] **Step 1: Write failing tests**

The tests construct native `/generate` responses and require the comparator to:

1. accept identical single and batched responses;
2. reject different output token IDs;
3. reject non-finite output logprobs;
4. calculate max absolute generated-token logprob error;
5. calculate per-step top-k token overlap while matching values by token id.

Run `../../.venv/bin/python -m pytest -q python/sgl_jax/test/test_glm52_e2e_compare.py` and verify failure because `compare_glm52_e2e_results.py` does not exist.

- [ ] **Step 2: Implement the minimum comparator**

Expose `compare_responses(candidate, baseline, *, max_logprob_abs_error, min_topk_overlap)` and a CLI taking `--candidate`, `--baseline`, `--max-logprob-abs-error`, `--min-topk-overlap`, and `--output`. The output JSON must contain `passed`, `response_count`, `output_ids_equal`, `max_output_logprob_abs_error`, and `min_topk_overlap`.

- [ ] **Step 3: Verify red-to-green**

Run the focused test, then the existing DSA test group. Expected results are zero failures.

### Task 3: Real-weight distributed runner

**Files:**
- Create: `scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh`
- Test: `python/sgl_jax/test/test_glm52_e2e_compare.py`

- [ ] **Step 1: Add runner contract assertions to the test**

Read the runner as text and assert it requires a unique `GLM52_DSA_RUN_ID`, uses `/models/GLM-5.2`, refuses to run without `_DOWNLOAD_COMPLETE`, uses `--load-format safetensors`, accepts only `dsa` or `fa` through `GLM52_ATTENTION_BACKEND`, and preserves the four-rank SUCCESS/STOP/ACK teardown protocol.

- [ ] **Step 2: Verify the new assertions fail**

Run the focused pytest target and confirm failure because the real runner is absent.

- [ ] **Step 3: Implement the runner**

Base process coordination on `run_glm52_dsa_v7x32_dummy_e2e.sh`. Launch with TP32/DP1/EP32, BF16, page size 128, context 4096, chunk size 128, and real safetensors. Rank 0 sends three deterministic requests with `temperature=0`, `return_logprob=true`, and `top_logprobs_num=20`:

1. a four-token short request with two decode tokens;
2. a 257-token request to cross three chunked-prefill buckets;
3. a ragged batch containing lengths 9 and 133 with two decode tokens each.

Require finite logprobs, non-empty generated IDs, exact prompt-token counts, and at least one completion token. Store each payload and response under the rank-0 artifact directory.

- [ ] **Step 4: Verify shell and unit tests**

Run `bash -n scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh` and the focused pytest target.

### Task 4: TPU correctness and real DSA E2E

**Files:**
- Modify: `note/2026-07-18-glm52-dsa-falcon-results.md`

- [ ] **Step 1: Transfer the candidate tree safely**

Create a patch from the known base commit and transfer it plus the two runner utilities with `falcon exp cp` to all four ranks of `exp-x9ghpgedxk`. Verify every rank reports the same candidate commit and a clean applied diff.

- [ ] **Step 2: Re-run sparse reference versus Pallas**

Run `scripts/kernels/run_glm52_dsa_v7x32_shardmap_smoke.py` on all ranks. Require four `GLM52_DSA_V7X32_REFERENCE_AND_SHARDMAP_OK` markers and exit 0.

- [ ] **Step 3: Run real DSA serving**

Execute the real runner on all four ranks with `GLM52_ATTENTION_BACKEND=dsa` and a fresh shared run id. Require successful model load, EXTEND and DECODE compilation, `/health` 200, all three request checks, rank-0 SUCCESS, and follower ACKs.

### Task 5: Dense all-visible precision baseline

**Files:**
- Modify: `note/2026-07-18-glm52-dsa-falcon-results.md`

- [ ] **Step 1: Run dense MLA serving**

After the DSA server is fully stopped, run the same real runner with `GLM52_ATTENTION_BACKEND=fa` and a fresh run id. Preserve identical input payloads and sampling parameters.

- [ ] **Step 2: Compare short all-visible responses**

Run `compare_glm52_e2e_results.py` for the four-token request. Require identical greedy output IDs, finite logprobs, maximum generated-token logprob absolute error at most `0.05`, and top-20 overlap at least `0.90`. Record actual values even if the threshold fails.

- [ ] **Step 3: Interpret longer requests correctly**

Use the 257-token and ragged runs as DSA functional evidence only because their causal visible length remains below 2048 but different prefill bucketization may amplify BF16 implementation differences. Compare and report them, but do not hide a mismatch behind the short-request gate.

### Task 6: Debug, regress, document, and clean up

**Files:**
- Modify as required by a reproduced failure, always with a failing test first.
- Modify: `note/2026-07-18-glm52-dsa-falcon-results.md`

- [ ] **Step 1: Diagnose failures systematically**

Inspect Falcon job conditions first, then rank logs. Reproduce each code bug in a failing local test before editing implementation code. Re-run the smallest failing TPU case before the full E2E.

- [ ] **Step 2: Run final regression**

Run the complete local DSA test set, YAML/shell syntax checks, v7x-32 reference/Pallas smoke, real DSA E2E, and the dense comparison gate. Read every command's exit status and output before claiming success.

- [ ] **Step 3: Record evidence and clean resources**

Record experiment ids, commit, environment, checkpoint byte count, compile times, request results, precision metrics, and known limitations. Abort the downloader after a verified marker if it has not terminated itself. Keep or abort debug reservations according to whether more work is required; never leave a serving process running inside a retained debug pod.
