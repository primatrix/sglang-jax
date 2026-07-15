# PD Transfer Observability Status - 2026-07-05

## Context

This note captures the current Falcon debug-pod state after adding lightweight
per-request PD timing for the MiMo SWA PD disaggregation path.

- Falcon exp: `exp-5uqgg64144`
- Rank 0: prefill + bootstrap, pod IP `10.125.130.4`
- Rank 1: decode + router, pod IP `10.125.129.4`
- Model: `/models/MiMo-V2-Flash`
- Data plane: Raiden chunked transfer, 8 chunks for 16k prompts
- Router cap used in these runs: prefill max inflight 4
- JIT cache path: `/tmp/tpu_logs/jit_cache`

Startup remains dominated by weight loading. In the latest restarts, precompile
was about 17-18s per side, while weight loading was about 4 minutes per side.

## Code State

Added lightweight request timing in:

- `python/sgl_jax/srt/disaggregation/req_time_stats.py`
- `python/sgl_jax/srt/disaggregation/prefill.py`
- `python/sgl_jax/srt/disaggregation/decode.py`
- `python/sgl_jax/srt/disaggregation/jax_transfer/conn.py`

New local test:

- `python/sgl_jax/test/disaggregation/test_pd_time_stats.py`

Local verification:

- `pytest python/sgl_jax/test/disaggregation/test_pd_time_stats.py python/sgl_jax/test/disaggregation/test_pd_internal_state.py python/sgl_jax/test/disaggregation/test_pd_router_admission.py -q`
- Result: 21 passed
- `git diff --check`: clean
- `py_compile` for changed runtime files: pass

The updated files were synced to both Falcon ranks. Rank 0 was restarted for the
latest prefill-tail split. After the no-op poll-before-forward experiment, the
final `prefill.py` was synced back to both pod disks; restart rank 0 before a
strict commit-shape rerun. Rank 1 decode stayed running, and the router was
restored to prefill inflight cap 4.

## Bench Runs

Latest focused run:

```text
/tmp/e2e_logs/tail_split_c16_1783240380.log
/tmp/e2e_logs/tail_split_c16_1783240380.jsonl
```

C16, 16k input, 128 output:

```text
duration: 24.21s
TTFT mean: 12731.67ms
TTFT p99: 22422.40ms
ITL mean: 18.66ms
successful requests: 16
```

Earlier same-code-shape C64 before the final tail split:

```text
/tmp/e2e_logs/phase_c64_1783239248.log
/tmp/e2e_logs/phase_c64_1783239248.jsonl
duration: 88.04s
TTFT mean: 44857.01ms
TTFT p99: 86118.75ms
ITL mean: 20.29ms
successful requests: 64
```

Router prefill-inflight cap tuning, C64 16k/128:

| cap | log prefix | duration | TTFT mean | TTFT p99 | ITL mean | throughput |
|---:|---|---:|---:|---:|---:|---:|
| 3 | `cap3_c64_1783241837` | 89.54s | 45582.46ms | 87502.43ms | 18.21ms | 0.71 req/s |
| 4 | `phase_c64_1783239248` | 88.04s | 44857.01ms | 86118.75ms | 20.29ms | 0.73 req/s |
| 5 | `cap5_c64_1783242035` | 88.64s | 45308.17ms | 86721.01ms | 22.76ms | 0.72 req/s |

Conclusion: cap 4 is still the best tested point for this single-prefill /
single-decode debug setup. Cap 3 underfeeds prefill slightly, while cap 5 does
not improve TTFT and increases ITL.

Poll-before-forward experiment:

```text
/tmp/e2e_logs/poll_before_forward_c16_1783241584.log
duration: 24.06s
TTFT mean: 12570.74ms
TTFT p99: 22232.78ms
ITL mean: 18.99ms
sender_done_wait mean: 326.28ms
prefill_reap_gap mean: 0.12ms
transfer_tail mean: 326.41ms
```

Conclusion: adding one extra sender poll before selecting/running the next
prefill batch does not reduce the P-side `sender_done_wait`. The experiment was
reverted locally and should not be committed as an optimization.

## Current Per-Request Findings

### Prefill, C16 Last 16

```text
forward mean: 2742.54ms
forward_chunk_sum mean: 2713.65ms
forward_chunk_count: 8
forward_chunk_avg mean: 339.19ms

first_chunk_register_wait mean: 1.36ms
chunk_register_span mean: 2409.55ms
chunk_handoff_sum mean: 10.39ms
chunk_handoff_avg mean: 1.29ms

sender_done_wait mean: 324.68ms
prefill_reap_gap mean: 0.12ms
transfer_tail mean: 324.81ms
transfer mean: 2735.72ms
```

Interpretation:

- Chunked overlap is active: P registers chunks while forward continues.
- P-side handoff overhead is small, about 10ms total for 8 chunks.
- The old `transfer_tail` is almost entirely `sender_done_wait`.
- `prefill_reap_gap` is only about 0.1ms, so Python cleanup/logging is not the
  source of the 325ms tail.
- `sender_done_wait` is close to one forward chunk, which suggests the prefill
  event loop may only observe Raiden done_sending after it finishes another
  forward chunk.

### Decode, C64 Last 64

```text
metadata_wait mean: 2579.03ms
kv_alloc mean: 0.01ms
receiver_init mean: 0.10ms
transfer_setup mean: 0.20ms
prealloc_wait mean: 2579.34ms

first_chunk_wait mean: 4.60ms
start_read_call_sum mean: 0.30ms
start_read_call_count: 8
chunk_start_span mean: 2340.25ms
transfer_tail mean: 52.26ms
enqueue_decode mean: 2.51ms
kv_wait mean: 2399.61ms
```

Interpretation:

- D-side `start_read` synchronous call overhead is negligible.
- D starts chunk reads over roughly the same span as P registers chunks, so
  P/D chunk overlap is working.
- D-side final tail after last start_read is about 50-65ms.
- The large `metadata_wait/prealloc_wait` is mostly prefill capacity/backlog
  visibility and is not the current optimization target.

## Working Conclusion

The current TTFT shape is dominated by prefill service capacity and backlog. The
transfer data plane itself is not showing large per-call CPU overhead:

- P handoff is about 1-2ms per chunk.
- D start_read is effectively sub-ms total at the current log precision.
- D final transfer tail is about 50ms.

The remaining suspicious serial gap is P-side `sender_done_wait` around one
chunk duration. This may not directly block D from producing the first token,
because D observes `done_recving` much earlier, but it does delay prefill
request completion and resource release on the prefill side. A simple extra
event-loop poll did not reduce it, so the next check should instrument Raiden
sender completion state more directly instead of adding more scheduler polls.

## Suggested Next Work Items

1. Keep router prefill inflight cap at 4 for the current 1P:1D Falcon debug
   setup. Re-test only if decode/prealloc limits or prefill count changes.

2. Keep C16 as the quick regression for observability and run C64 only after a
   promising change.

3. Add Raiden-side completion visibility for the sender path:
   when final chunk is accepted, when transfer is actually complete, and when
   `poll()` first reports success.

4. The main TTFT optimization direction remains prefill capacity/utilization:
   keep prefill at a small controlled inflight count, avoid router over-admit,
   and use multiple prefill workers per decode in production-scale setups.
