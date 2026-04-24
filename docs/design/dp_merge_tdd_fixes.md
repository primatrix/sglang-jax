# DP Merge TDD Fix Log

This document tracks the risks found while reviewing the DP merge branch and records each fix using a strict TDD loop:

1. Add a regression test that fails on the current code.
2. Implement the smallest fix.
3. Re-run the same test and related existing tests.

## Issue Queue

| Order | Issue | Impact | dp_size=1 impact | Priority | Difficulty | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Merge conflict with `upstream/main` in `schedule_batch.py` | Branch cannot be merged directly | Yes | P0 | Medium | Fixed |
| 2 | `min_p` flag is dropped in `_merge_sampling_info()` | `min_p` values are copied but filtering is disabled | Yes | P0 | Low | Fixed |
| 3 | Penalty state is dropped in merged sampling info | `frequency_penalty`, `presence_penalty`, `min_new_tokens` do not affect logits | Yes | P0 | Medium | Fixed |
| 4 | Logprob options interact badly with padded DP layout | `top_logprobs_num`, `token_ids_logprob`, `return_logprob` can index wrong rows or build ragged arrays | Yes | P0 | Medium | Fixed |
| 5 | Decode OOM graceful abort from upstream is not preserved | Decode OOM may assert/crash instead of aborting requests | Yes | P0 | Medium | Fixed |
| 6 | Speculative decoding still references old batch fields | Spec decode/EAGLE can fail after DP refactor | Yes when spec is enabled | P1 | Medium-high | Fixed |
| 7 | Multimodal/MRoPE/deepstack are disabled in DP model worker path | Multimodal, MRoPE, deepstack regressions | Yes | P1 | Medium-high | Fixed |
| 8 | Grammar masks use compact request order instead of padded DP order | JSON schema/regex/grammar rows can be misaligned | Mostly dp_size>1 | P1 | Medium | Fixed |
| 9 | Logprobs tests are skipped or incomplete | Logprob behavior is not protected by CI | Yes, worse for dp_size>1 | P1 | Medium | Pending |
| 10 | Missing explicit `tp_size % dp_size == 0` validation | Bad launch config fails late/unclearly | No | P2 | Low | Pending |
| 11 | `need_top_p_sampling` and `need_top_k_sampling` are not merged | Currently low functional impact because sampler uses values directly | Low | P2 | Low | Pending |
| 12 | `repetition_penalty` and `logit_bias` do not appear wired into sampler | Parameter compatibility gap | Not clearly a DP regression | P2 | Medium | Pending |
| 13 | Qwen2.5-VL 3B DP e2e is not part of v6e-4 CI and falls back to a non-TP4 stage config | The real multimodal DP regression can be missed by CI; 4-card CI cannot fit default ViT+AR TPU allocation | No direct runtime impact outside 3B config choice | P1 | Low | Fixed |

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

### 13. Qwen2.5-VL 3B CI Coverage

- `test/srt/multimodal/test_qwen2_5_vl_dp.py` was added but not listed in `test/srt/run_suite.py`, so PR CI would not execute it.
- The v6e-4 CI runner can run the test on a single host only if the ViT stage uses CPU and the AR stage uses all 4 TPU devices.
- `Qwen2.5-VL-3B-Instruct` was not explicitly registered, so local paths such as `/models/Qwen2.5-VL-3B-Instruct` fell back to `qwen2_5_vl_stage_config.yaml`, whose ViT and AR stages both request TPU devices.
- Existing `qwen2_5_vl_stage_config_tp4.yaml` is the intended single-host CI shape: ViT on CPU, AR on 4 TPU devices.

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

### Fix 4: decode OOM graceful abort in DP retract path

Status: Fixed.

TDD plan:

1. Add a regression test proving DP `retract_decode()` aborts the last request when it still cannot fit after all possible retractions, instead of asserting.
2. Add a scheduler regression test proving the third return value from `retract_decode()` is consumed and converted into an `AbortReq` response.
3. Observe both tests fail on the current code.
4. Port the upstream graceful-abort behavior into the DP-aware `retract_decode()` layout.
5. Re-run the new tests and the existing DP regression tests.

Result:

- Failing tests before fix:
  - `test_retract_decode_aborts_single_oom_request`
  - `test_scheduler_update_running_batch_sends_abort_req`
- Failures observed:
  - DP `retract_decode()` asserted with `[DP 0] No space for single request`.
  - `Scheduler.update_running_batch()` still unpacked two return values and raised `ValueError: too many values to unpack (expected 2)`.
- Fix:
  - `ScheduleBatch.retract_decode()` now returns `(retracted_reqs, new_estimate_ratio, reqs_to_abort)`.
  - When a DP rank's last remaining decode request still cannot fit, it marks that request with `FINISH_ABORT(..., HTTPStatus.INTERNAL_SERVER_ERROR, "InternalServerError")`, releases its KV/request-pool state, filters it out of the batch, and records it in `reqs_to_abort`.
  - The new estimate ratio uses `total_max_new_tokens + 1`, matching upstream's zero-division guard when all requests are aborted.
  - `Scheduler.update_running_batch()` now sends one `AbortReq` per aborted request through `_comm_backend` or `send_to_tokenizer`.
- Local verification after fix:
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py::TestDPSamplerRegressions::test_retract_decode_aborts_single_oom_request python/sgl_jax/test/test_dp_sampler_regressions.py::TestDPSamplerRegressions::test_scheduler_update_running_batch_sends_abort_req -q` passed.
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py -q` passed.
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_mixed_chunk_dp.py -q` passed.
- TPU verification after fix:
  - On `tpu7x-multi-slice-4-job-xz-0-wr6rk`, synced `/sglang-jax` to commit `7a4757b0`.
  - `source /tmp/tpu_logs/venv/bin/activate && PYTHONPATH=python python -m unittest sgl_jax.test.test_dp_sampler_regressions -v` passed.
  - `source /tmp/tpu_logs/venv/bin/activate && PYTHONPATH=python python -m unittest sgl_jax.test.test_sampler -v` passed.
  - `source /tmp/tpu_logs/venv/bin/activate && PYTHONPATH=python python -m unittest sgl_jax.test.test_mixed_chunk_dp -v` passed.

### Fix 5: upstream merge conflict resolution

Status: Fixed.

TDD plan:

1. Use `git merge-tree upstream/main HEAD` as the failing mergeability check.
2. Observe the conflict before attempting the merge.
3. Merge `upstream/main` and resolve conflicts while preserving the DP refactor and upstream graceful-abort behavior.
4. Re-run mergeability, compile, and DP regression tests after the merge.

Result:

- Failing check before fix:
  - `git merge-tree upstream/main HEAD` exited with conflicts in `python/sgl_jax/srt/managers/schedule_batch.py` and `python/sgl_jax/srt/managers/scheduler.py`.
- Conflict resolution:
  - `scheduler.py`: kept the three-value `retract_decode()` contract and `AbortReq` sending path; only the duplicated comment differed from upstream.
  - `schedule_batch.py`: kept the DP-aware `retract_decode()`, `release_req()`, `retract_all()`, `prepare_for_idle()`, and `_evict_swa()` structure.
  - `schedule_batch.py`: also carried over upstream's `tree_cache.disable` handling into the DP `release_req()` path, so disabled radix cache follows the same release behavior as `ChunkCache`.
  - Automatically merged upstream additions from `#941` and `#944`, including DeepSeek V3 related files and model/layer updates.
- Local verification after conflict resolution:
  - `git merge-base --is-ancestor upstream/main HEAD` passed after merge commit `bb14f1260`.
  - `git merge-tree upstream/main HEAD` exited successfully and no longer reported conflicts.
  - `PYTHONPATH=python python -m compileall -q benchmark/gsm8k/__init__.py benchmark/gsm8k/bench_sglang_jax.py python/sgl_jax/srt/configs/model_config.py python/sgl_jax/srt/layers/attention/mla.py python/sgl_jax/srt/layers/embeddings.py python/sgl_jax/srt/layers/moe.py python/sgl_jax/srt/managers/schedule_batch.py python/sgl_jax/srt/managers/scheduler.py python/sgl_jax/srt/models/deepseek_v3.py python/sgl_jax/srt/models/grok.py` passed.
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py -q` passed.
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_mixed_chunk_dp.py -q` passed.
- TPU verification after conflict resolution:
  - On `tpu7x-multi-slice-4-job-xz-0-wr6rk`, synced `/sglang-jax` to commit `d8bb8f11`.
  - `source /tmp/tpu_logs/venv/bin/activate && PYTHONPATH=python python -m unittest sgl_jax.test.test_dp_sampler_regressions -v` passed.
  - `source /tmp/tpu_logs/venv/bin/activate && PYTHONPATH=python python -m unittest sgl_jax.test.test_sampler -v` passed.
  - `source /tmp/tpu_logs/venv/bin/activate && PYTHONPATH=python python -m unittest sgl_jax.test.test_mixed_chunk_dp -v` passed.

### Fix 6: speculative decode with single-DP DP batch layout

Status: Fixed for `dp_size=1`; `dp_size>1` speculative decoding remains unsupported and now fails explicitly.

TDD plan:

1. Add a regression test that exercises the scheduler-level spec path, not individual fields: construct a DP-layout `ScheduleBatch`, enable `SpeculativeAlgorithm.EAGLE`, call `Scheduler.run_batch()` with a fake draft worker, and assert the scheduling result.
2. Observe the current code fail because the spec path still reads old single-batch fields that no longer exist after the DP refactor.
3. Add single-DP compatibility aliases so existing EAGLE code can still use the old field names when `dp_size=1`.
4. Preserve safety for `dp_size>1` by rejecting speculative decoding explicitly instead of silently using a wrong layout.
5. Re-run the scheduler-level spec regression and the DP regression suite.

Result:

- Failing behavior before fix:
  - `ScheduleBatch.get_spec_model_worker_batch()` raised `AttributeError: 'ScheduleBatch' object has no attribute 'input_ids'`.
  - `SchedulerOutputProcessorMixin._resolve_spec_decode_token_ids()` raised `AttributeError: 'ScheduleBatch' object has no attribute 'reqs'`.
  - These failures are exactly the class of issue seen when enabling spec after the DP batch refactor, including `dp_size=1`.
- Test design adjustment:
  - The first probe used field-level tests to identify the failure, but those were too brittle.
  - Per review feedback, the final checked-in regression is `test_scheduler_run_batch_spec_decode_uses_single_dp_layout`, which exercises `Scheduler.run_batch()` through the EAGLE branch and asserts scheduler-visible behavior only.
- Fix:
  - `ScheduleBatch` now exposes old single-batch aliases (`reqs`, `input_ids`, `seq_lens`, `spec_info`, etc.) only when `dp_size == 1`, backed by `reqs_info[0]`.
  - `get_spec_model_worker_batch()` now passes `real_bs_per_dp`, `dp_size`, and `per_dp_bs_size` into `ModelWorkerBatch` on the spec path.
  - `get_spec_model_worker_batch()` raises `NotImplementedError` for `dp_size > 1`, because multi-DP EAGLE still needs a real padded-layout implementation.
- Local verification after fix:
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py::TestDPSamplerRegressions::test_scheduler_run_batch_spec_decode_uses_single_dp_layout -q` passed.
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py -q` passed.
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_mixed_chunk_dp.py -q` passed.
  - `PYTHONPATH=python python -m compileall -q python/sgl_jax/srt/managers/schedule_batch.py python/sgl_jax/test/test_dp_sampler_regressions.py` passed.
- Existing speculative unit suite note:
  - `PYTHONPATH=python python -m pytest python/sgl_jax/test/speculative/test_eagle_tree_build.py -q` could not be used locally as a clean signal: without stubs it fails collection due missing `llguidance`; with stubs it reaches Pallas kernels and fails on CPU with `ValueError: Only interpret mode is supported on CPU backend`.
  - TPU validation is still required for that kernel-level suite or for a real model E2E spec run.
- TPU verification after fix:
  - On `tpu7x-multi-slice-4-job-xz-0-wr6rk`, synced `/sglang-jax` to commit `93f6ee80`.
  - `source /tmp/tpu_logs/venv/bin/activate && PYTHONPATH=python python -m unittest sgl_jax.test.test_dp_sampler_regressions -v` passed, including the scheduler-level spec regression.
  - `source /tmp/tpu_logs/venv/bin/activate && PYTHONPATH=python python -m unittest sgl_jax.test.test_sampler -v` passed.
  - `source /tmp/tpu_logs/venv/bin/activate && PYTHONPATH=python python -m unittest sgl_jax.test.test_mixed_chunk_dp -v` passed.
  - `source /tmp/tpu_logs/venv/bin/activate && PYTHONPATH=python python -m unittest sgl_jax.test.speculative.test_eagle_tree_build -v` passed on TPU.
  - A real model E2E spec run is still not recorded here; `/models` on the TPU host has target checkpoints such as `Qwen3-8B`, but no obvious local EAGLE/EAGLE3 draft checkpoint was present.

### Test Coverage Hardening: sampler/logprob behavior checks

Status: Added behavior-level tests; no production behavior change.

Reason:

- The first regression tests for fixes 1-3 were intentionally narrow so the failing fields could be localized quickly.
- Review feedback correctly pointed out that field-level tests alone are brittle: a harmless metadata rename could break the tests without proving user-visible sampler behavior regressed.
- The retained narrow tests still help pinpoint failures, but they are now backed by behavior tests that go through the practical paths used by decode.

Added tests:

| Test | Practical path exercised | Old failure it protects |
| --- | --- | --- |
| `test_dp_min_p_changes_sampled_token_after_model_worker_conversion` | `ScheduleBatch` DP merge -> `ModelWorkerBatch` -> `SamplingMetadata` -> `Sampler` | Before fix 1, the DP merge dropped `need_min_p_sampling`; the seeded request sampled token `2` even though min-p should filter it, instead of token `0`. |
| `test_dp_linear_penalty_changes_greedy_token_after_model_worker_conversion` | `ScheduleBatch` DP merge -> `ModelWorkerBatch` -> `SamplingMetadata` -> `Sampler` with greedy decoding | Before fix 2, the merged path dropped `linear_penalty`; greedy output stayed token `0` instead of switching to token `1` after penalty application. |
| `test_dp_sampler_logprobs_are_returned_to_request_in_padded_layout` | `ScheduleBatch` DP merge -> `SamplingMetadata` -> `Sampler` logprob output -> `SchedulerOutputProcessorMixin.process_batch_result_decode()` | Before fix 3, padded DP rows could either build ragged logprob arrays or return row-0/padding logprobs to a real request on `dp_rank > 0`. |

Verification:

- `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py::TestDPSamplerRegressions::test_dp_min_p_changes_sampled_token_after_model_worker_conversion python/sgl_jax/test/test_dp_sampler_regressions.py::TestDPSamplerRegressions::test_dp_linear_penalty_changes_greedy_token_after_model_worker_conversion python/sgl_jax/test/test_dp_sampler_regressions.py::TestDPSamplerRegressions::test_dp_sampler_logprobs_are_returned_to_request_in_padded_layout -q` passed.
- `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py -q` passed with 14 tests.
- First TPU run found a test-tolerance issue, not a behavior mismatch: `test_dp_sampler_logprobs_are_returned_to_request_in_padded_layout` differed from the expected float32 logprob by `8.9e-08` under `np.testing.assert_allclose()` defaults.
- The logprob behavior test now uses `rtol=1e-6, atol=1e-6` for float32 comparisons.
- TPU verification after tolerance adjustment:
  - On `tpu7x-multi-slice-4-job-xz-0-wr6rk`, synced `/sglang-jax` to commit `de57abea5`.
  - `source /tmp/tpu_logs/venv/bin/activate && PYTHONPATH=python python -m unittest sgl_jax.test.test_dp_sampler_regressions -v` passed with 14 tests.

Remaining test-quality gap:

- These tests are stronger than the original field-only checks, but they are still unit-level simulations with fake requests and logits.
- Issue 9 remains pending until a model-facing or API-facing logprob test is enabled in the regular suite.

### Fix 7: multimodal Qwen2.5-VL with DP attention

Status: Fixed.

TDD plan:

1. Add a real e2e test for Qwen2.5-VL chat completions with `--multimodal`, `--dp-size 2`, `--tp-size 4`, and a real Hugging Face checkpoint under `/models`.
2. Run the e2e on TPU before production changes and record the failure.
3. Fix only the first observed failure, rerun the same e2e, and continue until the user-visible request succeeds.
4. Re-run local DP regressions after code changes.
5. Re-run the same TPU e2e after the final code change.

Test added:

- `test/srt/multimodal/test_qwen2_5_vl_dp.py`
- Model: `SGLANG_JAX_QWEN2_5_VL_MODEL` when set; otherwise `/models/Qwen2.5-VL-3B-Instruct`.
- Request: `/v1/chat/completions` with a generated red PNG data URI and prompt `What color is this image? Answer with one word.`

Failures observed before fixes:

- First e2e run reached model forward and crashed in flash attention:
  `ValueError: Expected cu_q_lens.shape=(4,) to be (3,).`
- After fixing the mesh, the same e2e reached prefill output processing and crashed:
  `AttributeError: spec_info is only available as a compatibility alias when dp_size == 1; use reqs_info for DP batches`
- After fixing the non-spec `spec_info` write, the same e2e returned HTTP 200 but failed semantically:
  `AssertionError: 'red' not found in 'blue'`

Fixes:

- `test_qwen2_5_vl_dp.py` now reports HTTP response bodies on status failures.
- Multimodal auto-regressive stages now honor `server_args.dp_size` and `server_args.tp_size` when building the stage mesh, so the AR stage uses a real `data x tensor` mesh instead of YAML `runtime.num_tpus=1` when DP is enabled.
- Non-spec prefill output processing no longer writes `batch.spec_info = None`, avoiding the single-DP compatibility alias on normal DP requests.
- `ScheduleBatch.get_model_worker_batch()` now merges multimodal fields into the same padded per-DP token layout as `input_ids`:
  - `mrope_positions` is generated per DP token section and padded to `per_dp_token_padding`.
  - `input_embedding` is sliced by each request's extend window and written into its DP token section.
  - `deepstack_visual_embedding` is merged into a `[layers, total_padded_tokens, hidden]` layout with zeros outside visual positions.

Local verification after implementation:

- `python -m compileall -q python/sgl_jax/srt/managers/schedule_batch.py python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py python/sgl_jax/srt/multimodal/manager/stage.py test/srt/multimodal/test_qwen2_5_vl_dp.py` passed.
- `python -m ruff check python/sgl_jax/srt/managers/schedule_batch.py` passed.
- `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py -q` passed with 14 tests.
- `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_mixed_chunk_dp.py -q` passed with 2 tests.

TPU verification after fix:

- The pre-fix and intermediate TPU runs were executed on `tpu7x-multi-slice-4-job-xz-0-wr6rk` and produced the failures above.
- The first final rerun after commit `0f44f9e2c` could not start because local `kubectl` failed before reaching the pod:
  `gke-gcloud-auth-plugin failed` while `gcloud config config-helper` attempted to refresh an access token from `oauth2.googleapis.com` and hit SSL EOF/read timeout errors.
  This was a local Kubernetes credential refresh/network blocker, not a model/test failure.
- After `kubectl` recovered, synced `/sglang-jax` to commit `54739e472`.
- `source /tmp/tpu_logs/venv/bin/activate && export SGLANG_JAX_QWEN2_5_VL_MODEL=/models/Qwen2.5-VL-3B-Instruct && PYTHONPATH=python:test/srt python -m unittest test.srt.multimodal.test_qwen2_5_vl_dp -v` passed on `tpu7x-multi-slice-4-job-xz-0-wr6rk`.

### Fix 7 follow-up: Qwen2.5-VL DP e2e in v6e-4 CI

Status: Fixed.

TDD plan:

1. Add a registry regression test proving Qwen2.5-VL 3B resolves to the single-host TP4 stage config for HF IDs, short names, and `/models/...` paths.
2. Run the test before the registry fix and record the failure.
3. Add explicit 3B registry entries for `qwen2_5_vl_stage_config_tp4.yaml`.
4. Add the registry test to the regular unit suite and the real Qwen2.5-VL DP e2e to `e2e-test-tpu-v6e-4`.
5. Install Qwen VL e2e dependencies in the v6e-4 e2e CI job.
6. Re-run the registry test locally and run the real e2e on the v6e-4 TPU host.

Failure observed before fix:

- `PYTHONPATH=python python -m unittest python/sgl_jax/test/multimodal/test_stage_config_registry.py -v` failed for all tested model paths.
- Each path resolved to `qwen2_5_vl_stage_config.yaml` instead of `qwen2_5_vl_stage_config_tp4.yaml`.

Fixes:

- `StageConfigRegistry` now maps `Qwen/Qwen2.5-VL-3B-Instruct` and `Qwen2.5-VL-3B-Instruct` to `qwen2_5_vl_stage_config_tp4.yaml`.
- `python/sgl_jax/test/multimodal/test_stage_config_registry.py` guards the HF ID, short name, and `/models/...` local-path cases.
- `test/srt/run_suite.py` now includes:
  - `python/sgl_jax/test/multimodal/test_stage_config_registry.py` in `unit-test-tpu-v6e-1`.
  - `test/srt/multimodal/test_qwen2_5_vl_dp.py` in `e2e-test-tpu-v6e-4`.
- `.github/workflows/pr-test.yml` installs `python[all,qwen-vl]` for the v6e-4 e2e job.
- `python/pyproject.toml` adds a `qwen-vl` extra for `qwen-vl-utils`, `torch`, and `torchvision`.
- The e2e keeps the simple default `/models/Qwen2.5-VL-3B-Instruct`; TPU hosts that only have `/models/Qwen/Qwen2.5-VL-3B-Instruct` should sync or copy it to the flat target directory before running.

Local verification after implementation:

- `PYTHONPATH=python python -m unittest python/sgl_jax/test/multimodal/test_stage_config_registry.py -v` passed.
- `python -m compileall -q python/sgl_jax/test/multimodal/test_stage_config_registry.py python/sgl_jax/srt/multimodal/models/static_configs/yaml_registry.py test/srt/run_suite.py test/srt/multimodal/test_qwen2_5_vl_dp.py` passed.
- `e2e-test-tpu-v6e-4` auto partition includes `test/srt/multimodal/test_qwen2_5_vl_dp.py` in partition 0 when `--auto-partition-size 2`.

TPU v6e-4 verification:

- On `kb-tpu`, cloned `/home/gcpuser/sglang-jax` at commit `ec599aaf6`.
- Installed the branch into `/home/gcpuser/jax-env` with `uv pip install -e "python[all,qwen-vl]"`.
- Copied `/models/Qwen/Qwen2.5-VL-3B-Instruct` to `/models/Qwen2.5-VL-3B-Instruct` so the e2e can use the simple default path.
- `cd /home/gcpuser/sglang-jax && source /home/gcpuser/jax-env/bin/activate && PYTHONPATH=python python -m unittest python/sgl_jax/test/multimodal/test_stage_config_registry.py -v` passed.
- `cd /home/gcpuser/sglang-jax && source /home/gcpuser/jax-env/bin/activate && bash scripts/killall_sglang.sh || true && PYTHONPATH=python:test/srt python -m unittest test.srt.multimodal.test_qwen2_5_vl_dp -v` passed on v6e-4.
- The e2e logs confirmed `qwen2_5_vl_stage_config_tp4.yaml`, prefill on DP layout `#prefill per DP: [1, 0]`, HTTP 200 from `/v1/chat/completions`, and `Ran 1 test in 64.694s`.

### Fix 8: grammar masks in DP padded layout

Status: Fixed.

TDD plan:

1. Add a regression test that builds a DP batch with one unconstrained request on DP rank 0 and one grammar-constrained request on DP rank 1.
2. Run the regression before the fix and confirm the grammar mask is not aligned to the padded DP row.
3. Merge grammar objects into the same padded per-DP batch layout as sampling arrays.
4. Initialize grammar vocab masks to allow-all before filling constrained rows, so unconstrained real rows in a mixed batch are not accidentally fully masked.
5. Add a real server e2e using `/generate` with explicit `dp_rank=0` and `dp_rank=1`, one unconstrained request and one EBNF-constrained request.
6. Run the unit regression locally, then run the e2e on v6e-4 TPU.

Failure observed before fix:

- `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py::TestDPSamplerRegressions::test_dp_grammar_masks_use_padded_request_layout -q` failed.
- The first row of `vocab_mask` was `[0, 0]` instead of allow-all `[-1, -1]`.
- The compact grammar list would place the DP-rank-1 grammar at padded row 1 instead of global row 2, leaving the real grammar request row fully masked.

Fixes:

- `_merge_sampling_info()` now creates `grammars` with length `total_bs` and writes each request grammar at `dp_rank * per_dp_bs_size + local_index`.
- `ModelWorkerSamplingInfo.update_grammar_vocab_mask()` now fills the allocated mask with `-1` before applying active grammar rows.
- `test/srt/openai_server/features/test_dp_grammar.py` adds a real `/generate` e2e with concurrent requests pinned to different DP ranks.
- `test/srt/run_suite.py` includes the new e2e in `e2e-test-tpu-v6e-4`.

Local verification after implementation:

- `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py::TestDPSamplerRegressions::test_dp_grammar_masks_use_padded_request_layout -q` passed.
- `PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py -q` passed with 15 tests.
- `PYTHONPATH=python python -m ruff check python/sgl_jax/srt/managers/schedule_batch.py python/sgl_jax/test/test_dp_sampler_regressions.py test/srt/openai_server/features/test_dp_grammar.py test/srt/run_suite.py` passed.
- `python -m compileall -q python/sgl_jax/srt/managers/schedule_batch.py python/sgl_jax/test/test_dp_sampler_regressions.py test/srt/openai_server/features/test_dp_grammar.py test/srt/run_suite.py` passed.
- `e2e-test-tpu-v6e-4` auto partition includes `test/srt/openai_server/features/test_dp_grammar.py` in partition 1 when `--auto-partition-size 2`.

TPU v6e-4 verification:

- On `kb-tpu`, synced `/home/gcpuser/sglang-jax` to branch `fix/dp-attn-jimoosciuc` at commit `1fd839ee5`.
- Installed missing `pytest` into `/home/gcpuser/jax-env` with `uv pip install pytest`.
- `cd /home/gcpuser/sglang-jax && source /home/gcpuser/jax-env/bin/activate && PYTHONPATH=python python -m pytest python/sgl_jax/test/test_dp_sampler_regressions.py::TestDPSamplerRegressions::test_dp_grammar_masks_use_padded_request_layout -q` passed.
- `cd /home/gcpuser/sglang-jax && source /home/gcpuser/jax-env/bin/activate && bash scripts/killall_sglang.sh || true && PYTHONPATH=python:test/srt python -m unittest test.srt.openai_server.features.test_dp_grammar -v` passed on v6e-4.
- The e2e logs confirmed `Qwen/Qwen3-1.7B`, `load_format=dummy`, `tp_size=4`, `dp_size=2`, `grammar_backend=llguidance`, and the target mixed batch `#prefill per DP: [1, 1]`.
- The e2e reported `Ran 1 test in 60.840s` and the EBNF-constrained DP-rank-1 response matched `Hello`.
