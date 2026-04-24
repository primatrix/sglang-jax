# DP Merge TDD Fix Log

This document tracks the risks found while reviewing the DP merge branch and records each fix using a strict TDD loop:

1. Add a regression test that fails on the current code.
2. Implement the smallest fix.
3. Re-run the same test and related existing tests.

## Issue Queue

| Order | Issue | Impact | dp_size=1 impact | Priority | Difficulty | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Merge conflict with `upstream/main` in `schedule_batch.py` | Branch cannot be merged directly | Yes | P0 | Medium | Pending |
| 2 | `min_p` flag is dropped in `_merge_sampling_info()` | `min_p` values are copied but filtering is disabled | Yes | P0 | Low | Fixed |
| 3 | Penalty state is dropped in merged sampling info | `frequency_penalty`, `presence_penalty`, `min_new_tokens` do not affect logits | Yes | P0 | Medium | Fixed |
| 4 | Logprob options interact badly with padded DP layout | `top_logprobs_num`, `token_ids_logprob`, `return_logprob` can index wrong rows or build ragged arrays | Yes | P0 | Medium | Fixed |
| 5 | Decode OOM graceful abort from upstream is not preserved | Decode OOM may assert/crash instead of aborting requests | Yes | P0 | Medium | Pending |
| 6 | Speculative decoding still references old batch fields | Spec decode/EAGLE can fail after DP refactor | Yes when spec is enabled | P1 | Medium-high | Pending |
| 7 | Multimodal/MRoPE/deepstack are disabled in DP model worker path | Multimodal, MRoPE, deepstack regressions | Yes | P1 | Medium-high | Pending |
| 8 | Grammar masks use compact request order instead of padded DP order | JSON schema/regex/grammar rows can be misaligned | Mostly dp_size>1 | P1 | Medium | Pending |
| 9 | Logprobs tests are skipped or incomplete | Logprob behavior is not protected by CI | Yes, worse for dp_size>1 | P1 | Medium | Pending |
| 10 | Missing explicit `tp_size % dp_size == 0` validation | Bad launch config fails late/unclearly | No | P2 | Low | Pending |
| 11 | `need_top_p_sampling` and `need_top_k_sampling` are not merged | Currently low functional impact because sampler uses values directly | Low | P2 | Low | Pending |
| 12 | `repetition_penalty` and `logit_bias` do not appear wired into sampler | Parameter compatibility gap | Not clearly a DP regression | P2 | Medium | Pending |

## Detailed Clues

### 1. Merge Conflict

- `git merge-tree upstream/main HEAD` reports a conflict in `python/sgl_jax/srt/managers/schedule_batch.py`.
- `upstream/main` contains the decode OOM graceful abort change from `#944`.
- The DP branch rewrote the same scheduling path, so conflict resolution must preserve both DP layout and upstream OOM behavior.

### 2. `min_p` Flag Dropped

- `SamplingBatchInfo.from_schedule_batch()` sets `need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs)`.
- `ScheduleBatch._merge_sampling_info()` copies `min_ps` but does not copy/OR `need_min_p_sampling`.
- `SamplingMetadata.from_model_worker_batch()` reads `batch.sampling_info.need_min_p_sampling`.
- `sampler.py` only applies min-p filtering when `need_min_p_sampling` is true.

### 3. Penalty State Dropped

- Per-DP `SamplingBatchInfo` owns a `BatchedPenalizerOrchestrator`.
- The merged `ModelWorkerSamplingInfo` returned by `_merge_sampling_info()` does not carry `linear_penalty` or `penalizer_orchestrator`.
- `SamplingMetadata.from_model_worker_batch()` falls back to a zero penalty and `do_penalties=False`.
- A comment already says `TODO: @Brian fix penalty with DataParallel`.

### 4. Logprob Padded Layout

- DP `get_model_worker_batch()` expands `top_logprobs_nums` and `token_ids_logprobs` to `total_bs`, including padding rows.
- `get_top_logprobs()` and `get_token_ids_logprobs()` build `jnp.array()` from per-row lists.
- If a real row has length > 0 and a padding row has length 0, this can become a ragged array error.
- Output processing also indexes logprob arrays by compact request index in several paths, while DP logits are in padded layout.

### 5. Decode OOM Graceful Abort

- Upstream `#944` changes decode OOM behavior to graceful abort.
- DP `retract_decode` still contains assert-heavy behavior, so `dp_size=1` can still crash.

### 6. Speculative Decoding

- Some spec paths still reference old `ScheduleBatch` fields such as `batch.reqs`, `self.extend_lens`, `self.prefix_lens`, and direct `batch.seq_lens`.
- DP stores those values under `reqs_info`.

### 7. Multimodal/MRoPE/Deepstack

- DP `get_model_worker_batch()` returns `mrope_positions=None`, `input_embedding=None`, and `apply_for_deepstack=False`.
- Upstream standard path computes and forwards those fields.

### 8. Grammar Mask Alignment

- `_merge_sampling_info()` builds `grammars=[req.grammar for req in all_reqs]`.
- `ModelWorkerSamplingInfo.update_grammar_vocab_mask()` allocates masks with `batch_size=len(self.temperatures)`, which includes padded DP slots.
- For `dp_size>1`, compact grammar rows do not line up with padded per-rank row offsets.

### 9. Logprob Test Coverage

- `test/srt/run_suite.py` comments out `test_logprobs.py`.
- Existing logprob behavior is not protected well enough for DP changes.

### 10. Divisibility Validation

- `attention_tp_size = tp_size // dp_size` and mesh construction use integer division.
- There is no clear early validation that `tp_size % dp_size == 0`.

### 11. Top-p/Top-k Flags

- `_merge_sampling_info()` does not merge `need_top_p_sampling` or `need_top_k_sampling`.
- Current sampler applies top-p/top-k from the values directly, so this is lower risk today but still inconsistent metadata.

### 12. Repetition Penalty and Logit Bias

- `SamplingParams` accepts `repetition_penalty` and `logit_bias`.
- The reviewed sampler paths do not show an active implementation for either parameter.
- This needs separate confirmation against upstream expectations.

## TPU Validation Notes

- Single-host TPU pod: `tpu7x-multi-slice-4-job-xz-0-wr6rk`.
- Multi-slice TPU job: `tpu7x-multi-slice-16-job-xz`.
- Venv: `source /tmp/tpu_logs/venv/bin/activate`.
- If a dependency is missing, install it into the venv with `uv pip install <package>`.
- For code consistency, push the local branch and sync TPU with `git pull --rebase` instead of copying files manually.
- The current TPU venv did not include `pytest`, so `python -m unittest ...` was used for the completed sampler checks.
- If JAX fails with `Internal error when accessing libtpu multi-process lockfile`, inspect active processes and remove stale `/tmp/libtpu_lockfile`.

## Fix Log

### Fix 1: `min_p` flag in DP sampling merge

Status: Fixed.

TDD plan:

1. Add a regression test proving `_merge_sampling_info()` preserves `need_min_p_sampling` when any DP rank needs min-p.
2. Observe the test fail on the current code.
3. OR the flag while merging sampling info.
4. Re-run the regression test and existing DP mixed chunk test.

Result:

- Failing test before fix: `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py::TestDPSamplerRegressions::test_merge_sampling_info_preserves_min_p_flag -q`
- Failure observed: `AssertionError: False is not true`
- Fix: `_merge_sampling_info()` now ORs `need_min_p_sampling` across DP ranks and passes it into `ModelWorkerSamplingInfo`.
- Verification after fix:
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py::TestDPSamplerRegressions::test_merge_sampling_info_preserves_min_p_flag -q` passed.
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_mixed_chunk_dp.py -q` passed.

### Fix 2: penalty state in DP sampling merge

Status: Fixed.

TDD plan:

1. Add a regression test proving `_merge_sampling_info()` carries per-DP `linear_penalty` into the padded global layout.
2. Observe the test fail on the current code.
3. Merge penalty rows with the same per-DP padding layout used for sampling arrays.
4. Preserve precomputed `linear_penalty` in `ModelWorkerSamplingInfo.update_penalties()`.
5. Re-run the regression test and existing DP mixed chunk test.

Result:

- Failing tests before fix:
  - `test_merge_sampling_info_preserves_linear_penalty_layout`
  - `test_merge_sampling_info_materializes_penalty_orchestrator`
  - `test_model_worker_sampling_info_update_penalties_preserves_linear_penalty`
- Failures observed:
  - Merged `linear_penalty` was `None`.
  - `ModelWorkerSamplingInfo.update_penalties()` cleared precomputed `linear_penalty`.
- Fix:
  - `_merge_sampling_info()` now writes each DP rank's `linear_penalty` into the same padded global layout as sampling arrays.
  - If `linear_penalty` is not precomputed but a required `penalizer_orchestrator` exists, `_merge_sampling_info()` materializes it with `apply()`.
  - `ModelWorkerSamplingInfo.update_penalties()` now preserves existing `linear_penalty` and only computes from an orchestrator when needed.
- Verification after fix:
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py -q` passed.
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_mixed_chunk_dp.py -q` passed.

### Fix 3: logprob helpers and padded DP rows

Status: Fixed.

TDD plan:

1. Add regression tests proving logprob helper functions can handle rows with requested length `0` or `None` without building ragged arrays.
2. Observe the tests fail on the current code.
3. Make sampler helper outputs rectangular and slice per-request values in scheduler output processing.
4. Re-run the regression tests and existing DP mixed chunk test.

Result:

- Failing tests before fix:
  - `test_get_top_logprobs_handles_padding_rows`
  - `test_get_token_ids_logprobs_handles_padding_rows`
  - `test_decode_logprob_uses_padded_dp_row_index`
  - `test_prefill_logprob_uses_padded_dp_row_index`
- Failures observed:
  - `get_top_logprobs()` and `get_token_ids_logprobs()` built ragged JAX arrays when padded rows requested zero values.
  - Decode output processing read `next_token_logprobs[i]` where `i` was the local DP rank index, so `dp_rank > 0` read row 0 instead of `dp_rank * per_dp_bs_size + i`.
  - Prefill output processing had the same local-vs-padded row issue for next-token logprobs, top-logprobs, and token-id-logprobs.
- Fix:
  - Sampler helper functions now return rectangular arrays for padded rows.
  - Output processing slices rows back to each request's requested `top_logprobs_num` or `token_ids_logprob` length before storing results.
  - Decode and prefill output processing now use padded `global_i = dp_rank * per_dp_bs_size + i` for logits/logprob rows.
  - Prefill still uses compact request indices for `extend_input_len_per_req` and `extend_logprob_start_len_per_req`, because those arrays are compact per real request.
- Local verification after fix:
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py -q` passed.
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_mixed_chunk_dp.py -q` passed.
  - `PYTHONPATH=python python -m compileall -q python/sgl_jax/srt/layers/sampler.py python/sgl_jax/srt/managers/schedule_batch.py python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py python/sgl_jax/test/test_dp_sampler_regressions.py` passed.
- TPU verification after fix:
  - On `tpu7x-multi-slice-4-job-xz-0-wr6rk`, `source /tmp/tpu_logs/venv/bin/activate && PYTHONPATH=python python -m unittest sgl_jax.test.test_dp_sampler_regressions -v` passed.
  - `source /tmp/tpu_logs/venv/bin/activate && PYTHONPATH=python python -m unittest sgl_jax.test.test_sampler -v` initially failed due stale libtpu lockfile, then passed after removing `/tmp/libtpu_lockfile`.
