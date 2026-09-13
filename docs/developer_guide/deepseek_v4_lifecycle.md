# DeepSeek V4 request lifecycle (C4)

V4 uses `DeepseekV4ChunkCache` with radix reuse and scheduling overlap disabled.
The registry selects it from the C1 allocator even when chunked prefill is disabled.
SWA and history have separate capacities; global request slots index device state.
This extends the [C1 resources](deepseek_v4_resources.md) and
[C3 runtime](deepseek_v4_runtime.md) contracts.

## Ownership and completion

Only the same live request can match its computed prefix. Intermediate chunks keep
the request slot, original-token history mapping and compressor state. There is no
prefix tree or cross-request match. A fresh or retracted request starts at prefix 0.

`reclaim_completed_v4_swa` runs in prefill/decode result processing, after the
submitted forward completes. For completed length L and window W, it releases
whole pages before `floor(max(0, L-W+1)/P)*P`. It never uses the end of a prepared
chunk before execution: its early queries can still read the old window. The next
partial page, compressed history and request state remain owned. C1 clears the SWA
mapping before the freed physical pages can be reused by another request.

All finish, abort and retract exits use `release_kv_cache`. V4 releases the full
allocated extent in one call, including any partial-page tail, clears the request
mapping and prefix, and returns the request slot. Repeated cleanup sees no owner
and is inert, including after another request reuses that slot. Decode's grouped
free retains C1's deferred page release until the group closes.

State invalidation is an ownership operation, not a host-side tensor clear. A new
zero-prefix forward initializes the recycled slot to content=0 and score=-inf via
M2's `state_init_mask` inside the donated model graph. This avoids mutating an old
pool wrapper after C3 has replaced its arrays. Physical history/state contents may
remain until overwritten, but no valid request metadata exposes the old contents.

## Retraction

The existing `ScheduleBatch.release_req` and parked-chunk retraction paths release
C1 resources before `Req.reset_for_retract` and requeue. Prompt, generated token
history, grammar and streaming cursors remain intact. Prefill reconstructs
`prompt + output_ids` from position zero, in chunks; intermediate sampled results
are discarded by the existing result processor, and only the final chunk produces
a new token. Already emitted tokens are not resent.

C2 chooses victims and owns admission/allocation rollback. C4 implements their
release and restart contract. Full-model J0.2/J0.3 serving acceptance remains a
separate integration step.

## Validation

Use the focused [real-weight precision workflows](../../test/manual/deepseek_v4_precision/README.md)
for GPU/TPU module and layer comparisons. Full-model request acceptance uses
`test/manual/deepseek_v4_precision/static_fp8_smoke.py` against a published static
checkpoint. The smoke checks native-encoded greedy token IDs and normal EOS;
it does not establish long-context, concurrency or broad model-quality coverage.
