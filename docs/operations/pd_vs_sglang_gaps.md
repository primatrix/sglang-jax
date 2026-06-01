# PD: gaps vs ../sglang reference impl

This is the actionable takeaway from a side-by-side comparison
between `sgl-jax/python/sgl_jax/srt/disaggregation/` and
`sglang/python/sglang/srt/disaggregation/` (the production-tested
reference). It is organized as a *do-list*, not a design doc —
each item links to where in sglang the pattern lives, what
problem it solves, and where in sgl-jax it lands.

Source: code-review subagent run on
`epic/pd-disaggregation @ 799b6322 + WIP _jit_gather fix`.

## Top 5 P0 items (sorted by ROI / effort)

### 1. `PrefillInfo` gains `page_size` + `kv_cache_dtype` validation fields  (~30 min)

**Why:** the only thing today that gates a P/D version mismatch is
`protocol_version`. Two engines with different `page_size` or KV
dtype silently produce wrong output on D after the transfer.

**Where in sglang:** `disaggregation/common/conn.py:709-980` —
the bootstrap `/route` endpoint returns server-info with these
fields, the client raises on mismatch via `try_ensure_parallel_info`.

**Where in sgl-jax:** add the fields in
`disaggregation/bootstrap.py:PrefillInfo` and `RegisterPrefillRequest`;
extend `BootstrapClient.get_prefill_info` to compare against the
local engine's values and raise on drift.

### 2. Per-request 7-stage `time_stats` + `compute_and_observe_kv_transfer_metrics`  (~half day)

**Why:** when a PD request is slow in production, we currently
have no way to bisect *which* stage (bootstrap query / wait queue /
prefill / transfer / D wait / D pull / completion) is slow. Stage 4
H-A metrics only show aggregate phase latency, not per-request.

**Where in sglang:** `disaggregation/utils.py`, the
`req_time_stats.set_*_time()` family + the `compute_and_observe`
function that publishes `kv_transfer_latency_ms` /
`kv_transfer_speed_gb_s` per terminal request.

**Where in sgl-jax:** add `disagg_time_stats` field on `Req`,
plant 7 `set_*_time(perf_counter())` calls at the obvious
boundaries (intake, dequeue, prefill done, transfer enqueue,
transfer success, decode enqueue, terminal), expose a single
`pd_e2e_latency_seconds{phase=...}` histogram in `metrics.py`.

### 3. `TransferBackend` enum + `FakeKVManager` + `get_kv_class` factory  (~1 day)

**Why:** the current single-backend layout makes the unit tests
mock the whole transfer stack. A pure in-process `FakeKVManager`
lets us cover edge cases (out-of-order acks, double-fail, etc.)
without touching `jax.experimental.transfer`. Also opens the
door for future TPU-NIXL or other backends.

**Where in sglang:** `disaggregation/utils.py:112-296`
(`TransferBackend` enum + `KVClassType` matrix + `get_kv_class`).

**Where in sgl-jax:** lift `base/kv_manager.py` from "ABC + one
impl" to "ABC + enum-keyed factory"; add `jax_transfer/fake.py`
backend that implements the same `KVManager` ABC with in-memory
queues for testing.

### 4. Split `PrefillBootstrapQueue` → `PrefillBootstrapQueue` + `PrefillInflightQueue` AND finish FINDING-D (~2 days)

**Why:** two unrelated wins coupled because both touch the same
files. (a) Split lets bootstrap-timeout vs transfer-timeout use
different policies + recovery. (b) FINDING-D still has the open
question of whether the new `@jit`-wrapped per-layer gather
actually shares the compile cache; if it doesn't, the next try
is a single XLA-fused `jnp.stack(layer_bufs, axis=0)[:, idx]`
that lets XLA fuse all gathers into one program.

**Where in sglang:** `disaggregation/prefill.py:87-352` for the
queue split; `disaggregation/mooncake/conn.py:579-685` for the
"layers_params batch" pattern (RDMA-based, but the idea — one
descriptor per layer all submitted at once — transfers to our
single-jit-multi-buffer-gather equivalent).

**Where in sgl-jax:** new `PrefillInflightQueue` class in
`disaggregation/prefill.py`; rework the mixin's tick to
`pop_bootstrapped() -> inflight.add()`; if `@jit`-wrap is
insufficient, replace `_extract_req_kv` with the fused-stack
version.

### 5. `MetadataBuffers` + `ReqToMetadataIdxAllocator` + `ForwardMode.PREBUILT` (~3–5 days)

**Why:** the largest correctness + latency win on this list.
sglang transfers the first decode token + bootstrap_room +
cached_tokens count in a pre-allocated aux buffer *alongside* the
KV. D reads it from the buffer, marks the req as `ForwardMode.PREBUILT`,
and **skips the first forward entirely**. We currently re-prefill
the last input token on D (`_write_kv_to_pool` § "leave 1 token
unprefilled"), costing one extra step per request. Bonus: writing
`bootstrap_room` into the metadata buffer + verifying it on D
catches index-allocator bugs that would otherwise corrupt KV
silently.

**Where in sglang:** `disaggregation/utils.py:112-296`
(`MetadataBuffers` shape) + `disaggregation/decode_schedule_batch_mixin.py:22-101`
(`prepare_for_prebuilt`) + `decode.py:_commit_transfer_to_req`
(corruption check).

**Where in sgl-jax:** new `disaggregation/metadata_buffers.py`
holding 5–7 `jax.Array` buffers + a deque-based slot allocator;
extend `Req` with `metadata_buffer_index`; rewire D-side
`_write_kv_to_pool` to also read the buffer and set
`req.forward_mode = ForwardMode.PREBUILT`; add the
`process_batch_result_prebuilt` skip path in the scheduler loop.

## P1 follow-ups (recorded, not scheduled)

| # | Item | Source in sglang |
|---|------|------------------|
| P1-A | `poll_and_all_reduce` across TP ranks so PD state stays consistent | `prefill.py:poll_and_all_reduce` |
| P1-B | D-side `_allocatable_tokens` admission control | `decode.py:241-948` |
| P1-C | `DecodeReqToTokenPool` extension with `pre_alloc_size` | `decode.py:DecodeReqToTokenPool` |
| P1-D | `_ensure_prefill_info` retry+backoff vs single-shot GET | `decode.py:_ensure_prefill_info` |
| P1-E | `addr_to_rooms_tracker` + `connection_pool` for ZMQ socket reuse | `mooncake/conn.py` |
| P1-F | `prepare_abort` + `stream_output` + `release_aborted_request` triple for failed req notify-client path | `prefill.py:prepare_abort` |
| P1-G | `compute_and_observe_kv_transfer_metrics` per-request `bytes/duration` accounting | (see #2) |
| P1-H | `CommonKVManager` base class so future backends share bootstrap/TP wiring | `common/conn.py:CommonKVManager` |
| P1-I | `/register_dp_rank` + `/query_dp_ranks` two-stage DP routing | `common/conn.py:709-980` |
| P1-J | bootstrap server `_cleanup_expired_entries` async task | `common/conn.py:Bootstrap` |
| P1-K | `disaggregation_decode_polling_interval` to throttle D-side ZMQ poll | sglang ServerArgs |

## What we already have that sglang doesn't (preserve)

| # | Item | Where |
|---|------|-------|
| 1 | shared-secret HMAC + Bearer auth on all 3 channels | `pd_auth.py`, `bootstrap.py`, `zmq_notifier.py` |
| 2 | `protocol_version` field on `PrefillInfo` + client rejection of N-2 peers | `bootstrap.py` |
| 3 | Heartbeat daemon + TTL-based registry eviction | `bootstrap.py:HeartbeatDaemon` |
| 4 | Reaper thread on `JaxTransferKVManager` with `pull_timeout`/`ack_timeout` config | `jax_transfer/conn.py` |
| 5 | `graceful_shutdown(drain_timeout)` + SIGTERM handler | `jax_transfer/conn.py`, `scheduler.py` |
| 6 | mini_lb-style fan-out router in-tree | `router.py` |

## Known open items in our PD that need fixing before we should layer the above on

| # | Item | Status |
|---|------|--------|
| **FINDING-B** | D engine dies with `JaxRuntimeError: INTERNAL: SocketServer: Connection closed recv() == 0` when a peer P is `SIGKILL`'d while D has an open transfer link. The reaper doesn't catch this — the failure is below the wrapper. | open |
| **FINDING-C** | P-side prefix-cache hit on a second PD request → stale slot indices. **FIXED** in `799b6322`: PD reqs with `output_ids` empty skip `tree_cache.match_prefix`. | closed |
| **FINDING-D** | Per-layer gather creates a separate jit_reshard cache entry per call site → cumulative HBM OOM. **Two layered mitigations shipped** (bucket seqlen + `@jit`-wrap gather in WIP commit). Real cross-layer fused gather is the long-term answer. | partial |

## Notes on TPU/GPU divergence

Things from sglang that **do not** transfer to TPU:

- NIXL/Mooncake/Ascend transfer backends — GPU-specific (NVLINK/IB/HCCS); we have one backend: `jax.experimental.transfer` over DCN.
- Direct RDMA descriptors `(src_ptr, idx*item_len, length)` — this is mooncake's secret sauce that bypasses any Python-side gather. We can't follow that because `jax.experimental.transfer.register_pull(uuid, array)` requires a concrete `jax.Array`. Our equivalent has to materialize the gathered KV; the best we can do is share the compile cache across layers (which is the FINDING-D direction).
- CUDA streams / `torch.cuda.Event` — JAX async dispatch differs; we use `array.is_ready()` polling instead.
- IB device pinning — TPU networking is `jax.distributed`/ICI/DCN-managed.

Everything in the P0/P1 lists above is control-plane and transfers cleanly.
