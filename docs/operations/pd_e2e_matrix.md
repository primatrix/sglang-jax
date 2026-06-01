# PD end-to-end test matrix

Last validated: `epic/pd-disaggregation @ c9d9c8e7` (Stage 4 review-fix + e2e harness).
Deployment: 2P × 2D on v6e-16 (single-pod debug mode each).

This doc tracks the operator-facing acceptance tests for PD. After
any refactor that touches `disaggregation/`, `mem_cache/host_kv_pool`,
or the scheduler PD mixin, re-run the matrix and update the table.

## Test table — Stage 4 e2e run (2026-05-26)

| ID | Module | Purpose | Result | Notes |
|---|---|---|---|---|
| MANUAL | curl P0 + curl D0 (room=11, "Hello, my name is", max_new=4) | Smoke: PD path delivers byte-equal output | ✅ PASS | P and D both returned `" Alex, and I"` (output_ids 8515,11,323,358). D's `cached_tokens=4` proves real KV transfer. e2e 1.48s. |
| P0-1 | `test_topology_multi_pd` | Bootstrap registry has >= 2 P; rooms 10/11/20/21 partition into both buckets; fan-out PD pairs all return non-empty P+D output | ✅ PASS | Buckets `{P0: 2, P1: 2}`. `protocol_version=1` on both registry entries (H-D wire format works). |
| P0-2 | `test_correctness_byte_equal` | 12 PD pairs (4 prompts × 3 max_new) byte-equal P↔D | ❌ BLOCKED on FINDING-C/D | Salt added so prompt prefix doesn't repeat; second blocking issue is jit_reshard recompile OOM. |
| P0-3 | `test_long_prompt` | 2k/4k/8k token KV transfer | — | Not attempted; expected to hit same OOM. |
| P0-4 | `test_concurrency` | 10/30 concurrent PD pairs; inflight gauge returns to 0 | — | Not attempted; `max_running_requests` clamps to 3 in this deploy. |
| I3 | `test_chaos_wrapper` | wraps `chaos.sh` kill_p/drop_dcn/bootstrap | SKIP | Requires `pd-role=` pod labels (the v6e-16 Job does not set them). |
| ORTH-DP | `test_orthogonal_dp` | rerun byte-equal subset under `--dp-size 2` | — | Not attempted (matrix gated on P0-2). |

## Findings exposed by this run

These are real Stage 4 bugs surfaced by running the matrix
end-to-end. Each goes into a follow-up issue; none of them
invalidates the Stage 4 code (CPU tests 176 PASS, single-prompt
e2e proven), but they need to be fixed before declaring the
production-hardening RFC fully delivered.

### FINDING-A — D's pull from a 2nd P sometimes hangs (closed)

Earlier run on the stale v6e-16 pods showed D hanging when
pulling from P1 (pod-2). After cluster rebuild the same fan-out
worked — most likely a stale TPU node on the original pod-2,
not a code bug.

### FINDING-B — D crashes when a P is hard-killed mid-pull

Repro: `pkill -9` a P while D has a pending pull from it. D
process dies with
`jax.errors.JaxRuntimeError: INTERNAL: SocketServer: Connection closed recv() == 0`.
The Stage 4 H-B reaper is meant to handle peer crash but the
underlying `jax.experimental.transfer` socket EOF is fatal —
needs a try/except around `wrapper.pull().is_ready()` (and the
ack send path) that translates that error into the standard
FAILED transition.

### FINDING-C — P-side `_extract_req_kv` crashes on prefix-cache hit

Repro: send PD request with rid A and prompt P. Then send PD
request with rid B (different) and prompt P. P's radix cache
serves the prefix to req B → prefill batch reports
`#new-token: 1, #cached-token: 4`. P's `_extract_req_kv` then
calls `fused.at[token_indices].get(out_sharding=...)` with
`token_indices` shaped (1,) — the reshard reports a sharding
mismatch (or, depending on HBM, OOMs the jit_reshard recompile).

Fix direction: either disable prefix-cache match for PD requests
on the prefill side (mirror the `_pd_skip_prefix_match` marker
that already exists on D), or make `_extract_req_kv` understand
that the KV for cached tokens lives in older slots than the
prefill batch's own output.

### FINDING-D — KV gather compile-cache cumulatively OOMs

`_extract_req_kv` (`disaggregation/prefill.py`) was a per-layer
Python loop over `kv_pool.layer_num` gathers. Stage 4 layered
three mitigations (`799b6322` → `c88c3e36` → WIP), each closing
one symptom and exposing the next.

| Layer | Fix | Result |
|---|---|---|
| 1 | Bucket seqlen to `_KV_GATHER_BUCKETS = (8, 16, 32, ..., 4096)` so the shape pool stays bounded. | Per-seqlen jit_reshard recompile stopped. |
| 2 | Drop `--mem-fraction-static` 0.88 → 0.6 → 0.4 for more headroom. | Compile OOM (1.7G) avoided for tiny prompts. |
| 3 | `@jax.jit` over the entire `list[jax.Array]` of layer buffers (pattern lifted from tpu-inference `tpu_connector.py:913-917`) + indices placed on the pool's mesh via `jax.device_put(np.array(...), NamedSharding(mesh, P(None)))` + `out_sharding=P(None, *pool_pspec[1:])` (gather axis replicated, downstream axes carried from the pool's pspec). | First PD pair on a small prompt: **PASS** — `cached_tokens=4`, P+D byte-equal `" Alex, and I"`, e2e 1.27s on cold cache. |

But the next request with a **different seqlen bucket** triggers:

```
RESOURCE_EXHAUSTED: XLA:TPU compile permanent error.
Ran out of memory in memory space hbm.
Used 45.00G of 31.25G hbm. Exceeded hbm capacity by 13.76G.
```

at the new bucket size's compile step. The gather output is
~64 KB per layer × 36 layers ≈ 2.3 MB — three orders of magnitude
less than 45 GB. XLA is materializing some internal intermediate
at the full pool-pspec size even though the public signature
returns only the gathered slice.

This is the same root issue as layer 3 (`out_sharding` forcing
pool-size materialization) but now manifest at compile-time
inside the jit instead of at runtime alloc. Our pool's
`P(attention_data_axis, None, kv_partition_axis, None, None)`
sharding combined with JAX 0.8.1's gather semantics is the
mismatch.

What works:
  - Tiny-prompt smoke (5 tokens, bucket=8) — single PD pair
    byte-equal in 1.27 s. **PD plumbing end-to-end is functional**.
  - Multi-P × D topology (P0-1) — 4 fan-out PD pairs across
    2 P, all complete.

What doesn't:
  - Any second request that lands in a different bucket size.
  - The full byte-equal matrix (12 prompts of varied length).

The honest answer here is that `_extract_req_kv` needs an
architectural change, not another reshard tweak:

  - **Most likely fix:** restructure the KV pool so the layer
    axis is part of one tensor (`(L, T, page_size, ...)` shape
    sharded as `P(None, attention_data, ..., kv_partition, ...)`).
    Then the gather is one op over a 6-D tensor, not a loop
    over 36 buffers. The pre-stacked pool means XLA sees the
    gather as a single primitive and infers the output sharding
    cleanly.
  - **Alternative:** mirror sglang's design and never materialize
    the KV in Python — push slot indices to the transfer layer.
    This requires `jax.experimental.transfer` to support a
    "register pull-by-indices" API we'd have to confirm exists.
  - **Alternative:** adopt sglang's `MetadataBuffers` +
    `ForwardMode.PREBUILT` pattern (subagent P0 item #5). The
    metadata buffer ride-along avoids the per-request KV
    extract on the P side and is a much bigger architectural
    win on its own.

Stage 4 hardening (H-A..H-F + CPU tests + sub-agent gap
analysis) and the three layered mitigations are kept in tree;
they're load-bearing for the partial functionality we have.
Anyone picking this up should start from
`docs/operations/pd_vs_sglang_gaps.md` for the broader
roadmap — FINDING-D's "real fix" is tied to P0 item #4 / #5
there.

### FINDING-A — D's pull from a 2nd P sometimes hangs (closed)

Earlier run on the stale v6e-16 pods showed D hanging when
pulling from P1 (pod-2). After cluster rebuild the same fan-out
worked — most likely a stale TPU node on the original pod-2,
not a code bug.

## Deployment shape used

```
pod-0 (10.31.173.56)   bootstrap server (8998)   +   P0 :30100 / tx 31001 / sc 31002
pod-1 (10.31.175.54)                                 D0 :30200 / tx 31001 / sc 31003
pod-2 (10.31.174.56)                                 P1 :30100 / tx 31001 / sc 31002
pod-3 (10.31.172.54)                                 D1 :30200 / tx 31001 / sc 31003
```

Each engine launched with single-pod debug TPU env
(`TPU_HOST_BOUNDS=1,1,1`, `TPU_TOPOLOGY=2x2`,
`TPU_WORKER_HOSTNAMES=localhost`, `TPU_WORKER_ID=0`) so the 4
pods don't merge into one 16-chip slice.

D engines launched with
`--disaggregation-pull-timeout-seconds 300
 --disaggregation-ack-timeout-seconds 300` to work around the
Stage 4 H-B default 30s, which is too aggressive for cold pulls.

Driver: `scripts/run_pd_e2e_matrix.sh` (env vars `PD_P_URLS`,
`PD_D_URLS`, `PD_BOOTSTRAP_URL`).

## Running

```bash
export PD_P_URLS="http://10.31.175.51:30100,http://10.31.173.53:30100"
export PD_D_URLS="http://10.31.172.51:30200,http://10.31.174.53:30200"
export PD_BOOTSTRAP_URL="http://10.31.175.51:8998"
# optional:
# export SGL_JAX_PD_SHARED_SECRET=<secret>
# export PD_E2E_OUT_DIR=/tmp/pd_e2e

bash scripts/run_pd_e2e_matrix.sh                   # all tests
bash scripts/run_pd_e2e_matrix.sh topology long     # filter by substring
```

Each test script also runs standalone:

```bash
python -m sgl_jax.srt.disaggregation.tools.e2e.test_topology_multi_pd \
    --p-url http://... --p-url http://... \
    --d-url http://... --d-url http://... \
    --bootstrap-url http://...
```

## Known operational gotchas

- **`pull_timeout_seconds=30` default** (Stage 4 H-B) was tuned for
  D2H-staged KV. On path B with channel_number > 1 the first cold
  transfer can exceed 30s; the reaper then kills the receiver and
  the test fails with `KVReceiver reached failed`. Operator
  workaround until investigated: launch engines with
  `--disaggregation-pull-timeout-seconds 300
   --disaggregation-ack-timeout-seconds 300` (or revert
  `--disaggregation-channel-number` to 1).
- **`max_running_requests` clamps to KV-budget**: on Qwen3-8B single
  v6e debug pod, the live value lands at 3. The concurrency test's
  default levels (10, 30) are higher than the engine will admit;
  excess requests queue at the router. This is expected.
- **First test in a fresh deployment hits XLA precompile** (~10 min).
  Bake an XLA cache for repeat runs.
- **pod-2/3 needed manual pip install** of orjson, fastapi, uvicorn,
  starlette, pyzmq, uvloop, msgpack, httpx, click, h11, anyio,
  sniffio, python-dotenv, watchfiles, websockets, httptools, jiter,
  llguidance, modelscope, openai, partial-json-parser, pathwaysutils,
  pybase64, python-multipart, setproctitle, tiktoken (the Stage 2.5
  pods had these from a prior install). See ops log §6.6 for the
  exact `pip install --no-deps` command.

## Updating the table

After a re-run, paste the matrix-driver final summary into the
"Last run" column (commit SHA + date + PASS/FAIL). Save the
`/tmp/pd_e2e/*.json` reports alongside as evidence.
