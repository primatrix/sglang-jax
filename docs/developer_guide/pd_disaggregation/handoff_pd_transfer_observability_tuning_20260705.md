# PD Transfer Observability / Tuning Handoff - 2026-07-05

## Current State

This handoff covers the lightweight PD transfer timing work and one round of
Falcon tuning on MiMo-V2-Flash SWA PD disaggregation.

- Falcon exp: `exp-5uqgg64144`
- Rank 0: prefill + bootstrap, IP `10.125.130.4`
- Rank 1: decode + router, IP `10.125.129.4`
- Model: `/models/MiMo-V2-Flash`
- Workload: random 16k input, 128 output
- Transfer: Raiden chunked path, 8 chunks for 16k prompts
- Recommended current router cap: `--pd-prefill-max-inflight-requests 4`
- Recommended router P/D request mode: prefill/decode POST overlap enabled
- JIT cache: `/tmp/tpu_logs/jit_cache`

At handoff time, both sides were healthy and router was restored to cap 4 with
prefill/decode overlap enabled:

```text
rank0: bootstrap health ok, prefill health ok
rank1: decode health ok, router health ok
router pid: 225058
router cap: 4
pd_router_prefill_decode_overlap: true
```

The final `prefill.py` was copied back to both pod disks after a reverted
poll-before-forward experiment. The running rank0 process was not restarted
after that final disk sync to avoid another multi-minute model reload; restart
rank0 before a strict final-code rerun.

## What Changed

Runtime observability:

- Added role-specific per-request phase timing in
  `python/sgl_jax/srt/disaggregation/req_time_stats.py`.
- Prefill now records forward chunk time, chunk handoff time/count, first/last
  chunk registration, sender done, and cleanup/reap gap.
- Decode now records metadata wait, KV allocation, receiver init, transfer
  setup, first/last chunk read start, done_recving, enqueue decode, and first
  token.
- Jax transfer receiver records `start_read` call time and chunk-start count.
- Router/scheduler metrics expose macro backlog/busy-shape signals for quick
  C16/C64 regression.
- Router now exposes `pd_router_prefill_decode_overlap` in `/server_info` and
  supports `--no-pd-router-prefill-decode-overlap` for A/B only. The default
  remains overlap enabled.

Scheduler host/device overlap:

- PD disaggregation no longer forcibly disables scheduler overlap when
  `--disable-overlap-schedule` is not set.
- Added PD-specific overlap event loops for prefill and decode.
- Prefill overlap resolves the previous forward result before publishing that
  batch's KV chunks, but keeps PD batches out of `running_batch`.
- Prefill overlap stores a per-batch snapshot of `chunked_reqs`. This is
  required because `scheduler.chunked_reqs` has already advanced by the time the
  previous batch is processed. Without the snapshot, a mid chunk can be
  incorrectly registered as final; Falcon reproduced this as duplicate terminal
  chunks such as `c6/7` then `c7/8`, decode starting only 7 chunks, and prefill
  waiting for Raiden `done_sending` until the 30s timeout.
- `/get_server_info` internal state now includes `enable_overlap`.

Falcon local TPU env correction:

```bash
export TPU_PROCESS_ADDRESSES=localhost:8471
export TPU_WORKER_HOSTNAMES=localhost
export TPU_PROCESS_PORT=8471
export TPU_WORKER_ID=0
export TPU_HOST_BOUNDS=1,1,1
export HOST_BOUNDS=1,1,1
export TPU_TOPOLOGY=2x2x1
```

Using Falcon's default two-rank TPU env made JAX see 16 global devices and fail
against the `(2,4)` mesh. Setting only `TPU_PROCESS_ADDRESSES=localhost:8471`
is also insufficient because libtpu still infers two workers from
`TPU_HOST_BOUNDS=1,1,2`.

Tests:

- Added `python/sgl_jax/test/disaggregation/test_pd_time_stats.py`.
- Extended PD internal-state and router-admission tests for the new metrics.

Local verification:

```text
.venv/bin/pytest python/sgl_jax/test/disaggregation/test_pd_time_stats.py \
  python/sgl_jax/test/disaggregation/test_pd_internal_state.py \
  python/sgl_jax/test/disaggregation/test_pd_router_admission.py -q
21 passed in 3.48s

.venv/bin/python -m py_compile <changed runtime files>
pass

git diff --check
clean
```

Additional scheduler-overlap verification:

```text
.venv/bin/pytest python/sgl_jax/test/disaggregation/test_pd_overlap_schedule.py \
  python/sgl_jax/test/disaggregation/test_pd_time_stats.py \
  python/sgl_jax/test/disaggregation/test_pd_internal_state.py \
  python/sgl_jax/test/disaggregation/test_pd_router_admission.py -q
28 passed in 3.51s
```

## Serial / Parallel Shape

For a single 16k request, prefill runs 8 chunks serially:

```text
P forward chunk0 -> P register chunk0
P forward chunk1 -> P register chunk1
...
P forward chunk7 -> P register chunk7
```

The transfer is chunk-overlapped, not end-of-prompt-only:

```text
P register chunk0  -> D start_read chunk0
P forward chunk1   overlaps D reading chunk0
P register chunk1  -> D start_read chunk1
...
```

Decode waits for metadata, allocates destination KV, initializes the receiver,
starts reads as chunks are published, waits for `done_recving`, then enqueues
decode and produces the first token.

## Per-Request Findings

Prefill C16 last16:

```text
forward mean:               2742.54ms
forward_chunk_sum mean:     2713.65ms
forward_chunk_count:        8
forward_chunk_avg mean:     339.19ms
first_chunk_register_wait:  1.36ms
chunk_register_span mean:   2409.55ms
chunk_handoff_sum mean:     10.39ms
chunk_handoff_avg mean:     1.29ms
sender_done_wait mean:      324.68ms
prefill_reap_gap mean:      0.12ms
transfer_tail mean:         324.81ms
```

Decode C64 last64:

```text
metadata_wait mean:         2579.03ms
kv_alloc mean:              0.01ms
receiver_init mean:         0.10ms
transfer_setup mean:        0.20ms
prealloc_wait mean:         2579.34ms
first_chunk_wait mean:      4.60ms
start_read_call_sum mean:   0.30ms
start_read_call_count:      8
chunk_start_span mean:      2340.25ms
transfer_tail mean:         52.26ms
enqueue_decode mean:        2.51ms
kv_wait mean:               2399.61ms
```

Interpretation:

- P/D chunk overlap is working.
- P handoff and D `start_read` synchronous overhead are small.
- D final transfer tail is around 50-65ms in these runs.
- P-side `sender_done_wait` is around one chunk duration, but a simple extra
  scheduler poll did not reduce it.
- TTFT is dominated by prefill service rate and backlog, not transfer API call
  CPU overhead.

## Tuning Results

C64, 16k input, 128 output:

| prefill cap | log prefix | duration | TTFT mean | TTFT p99 | ITL mean | throughput |
|---:|---|---:|---:|---:|---:|---:|
| 3 | `cap3_c64_1783241837` | 89.54s | 45582.46ms | 87502.43ms | 18.21ms | 0.71 req/s |
| 4 | `phase_c64_1783239248` | 88.04s | 44857.01ms | 86118.75ms | 20.29ms | 0.73 req/s |
| 5 | `cap5_c64_1783242035` | 88.64s | 45308.17ms | 86721.01ms | 22.76ms | 0.72 req/s |

Cap 4 remains the best tested point for this 1P:1D debug setup. Cap 3 slightly
underfeeds prefill. Cap 5 does not improve TTFT and worsens ITL.

Poll-before-forward experiment:

```text
log prefix:                 poll_before_forward_c16_1783241584
duration:                   24.06s
TTFT mean:                  12570.74ms
TTFT p99:                   22232.78ms
ITL mean:                   18.99ms
sender_done_wait mean:      326.28ms
prefill_reap_gap mean:      0.12ms
transfer_tail mean:         326.41ms
```

Conclusion: do not keep this as an optimization. The P-side wait is not fixed
by one more scheduler-level poll before forward.

Router prefill/decode overlap A/B, C16 16k/128:

| mode | log prefix | duration | throughput | TTFT mean | P sender_done_wait |
|---|---|---:|---:|---:|---:|
| overlap on | `overlap_on_c16_1783246003` | 24.15s | 0.66 req/s | 12675.81ms | hundreds of ms |
| overlap off | `overlap_off_c16_full_1783246538` | 164.96s | 0.10 req/s | streaming stat invalid | about 30000ms |

Conclusion: router-side prefill/decode POST overlap is required for good
throughput in this Raiden path. With overlap disabled, prefill waits for sender
completion while decode has not started reading yet, so P-side
`sender_done_wait/transfer_tail` inflates to the 30s pull-timeout scale. The
measured C16 throughput drops by about 6.8x.

Scheduler host/device overlap, C16 16k/128:

| mode | log prefix | duration | throughput | TTFT mean | ITL mean | P sender_done_wait |
|---|---|---:|---:|---:|---:|---:|
| router overlap only baseline | `overlap_on_c16_1783246003` | 24.15s | 0.66 req/s | 12675.81ms | 19.81ms | hundreds of ms |
| scheduler overlap bug | `pd_sched_overlap_c16_1783248833` | 135.46s | 0.12 req/s | 83610.25ms | 0.02ms | about 30000ms |
| scheduler overlap fixed | `pd_sched_overlap_chunkfix_c16_1783249614` | 22.94s | 0.70 req/s | 12382.19ms | 12.78ms | 312.24ms |

The fixed run had `chunks_registered_count=8` on prefill and
`chunks_started_count=8` on decode for all last-16 requests. Decode
`transfer_tail` averaged `51.81ms`; prefill `transfer_tail` averaged
`312.36ms`.

## Next Work

1. Keep cap 4 as the default for this debug environment.
2. Keep router prefill/decode overlap enabled. Use the disable flag only for
   explicit A/B or failure reproduction.
3. Use C16 for quick transfer/forward regression, then C64 for utilization and
   tail validation.
4. Instrument Raiden sender completion more directly:
   final chunk accepted, transport complete, first poll success.
5. If TTFT needs a real step-function improvement, scale prefill capacity
   first, e.g. multiple prefill workers per decode, then retune router cap.
6. Treat `metadata_wait/prealloc_wait` as backlog visibility unless it remains
   high after prefill capacity is scaled.
