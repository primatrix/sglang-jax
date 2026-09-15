# DeepSeek V4 full-context sessions (experimental)

This opt-in path retains V4 request-owned KV between independent generate
requests. It does not build a radix tree, concatenate prompts, or share KV
between session IDs. It is not upstream SGLang's append-only session protocol.

## Configuration and API

Add to a working V4 server command:

```sh
--disable-radix-cache --enable-streaming-session \
--streaming-session-timeout 300 --max-streaming-sessions 128
```

The initial implementation requires DP=1 and no speculative decoding or
prefill/decode disaggregation. Other model/cache types are rejected. The flag
is off by default. Plain requests without `session_params` still release their
KV at completion even when the flag is enabled.

Each `/generate` request supplies **full token context**, with
`"session_params": {"id": "agent-1"}`. Use one input and `n=1`. A session permits
only one in-flight request, including requests waiting for admission. A missing
or expired session is recreated automatically. Session IDs are cache handles,
not authentication credentials; clients must use unique, unguessable IDs and
the deployment must enforce its own tenant isolation.

For example, the following are two independent HTTP requests:

```json
{"input_ids": [11, 12, 13, 14], "session_params": {"id": "agent-1"},
 "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
```

```json
{"input_ids": [11, 12, 13, 14, 15, 16], "session_params": {"id": "agent-1"},
 "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
```

Tokens shown are illustrative; supply real tokenizer IDs. There is **no
guarantee of a hit** merely because the session ID matches: the prefix for which
KV was actually computed must match exactly. In overlap mode that prefix can
include the final generated token's already-launched forward. Include generated
history in the next input if it belongs to the conversation; if it differs, the
implementation safely recomputes. Uncomputed output tokens are not part of the
cache key.

`POST /open_session` is optional and accepts
`{"request_id": "open-1", "session_id": "agent-1"}` (omitting `session_id`
generates one). `POST /close_session` accepts
`{"request_id": "close-1", "session_id": "agent-1"}`. Closing an active
session defers resource release until its request finishes. Closing a missing
session is harmless. Idle sessions expire after the configured TTL; active
requests are not timed out by the cache TTL. Capacity pressure evicts idle
sessions in LRU order. Cache flush clears idle sessions as well.

## Matching and ownership

- Match is token-exact and includes the cache namespace (`extra_key`). Edited
  or shorter cached prefixes are cold misses. There is no partial rollback or
  longest-common-prefix recovery in this version.
- A fully cached input with no remaining query is also recomputed: saved KV
  does not include reusable final logits.
- A hit transfers the **same request slot**, committed/allocated lengths and
  SWA eviction cursor to the new request. This preserves compressed history,
  SWA mappings and the slot-indexed C4/C128/indexer continuation state.
- The allocator's entire extent is retained; no partially written page is
  trimmed. In overlap mode existing result resolution waits for the next
  launch before publishing completion; subsequent forwards use the ordered
  model state. An extent extending beyond known token IDs is never retained.
- Abort releases resources; retraction releases physical state but preserves
  the active session lease so another request cannot take over mid-recompute.
- Text/base-model generation only for now: multimodal, LoRA, prompt logprobs
  and hidden-state requests are rejected rather than silently returning partial
  prompt results.

`session_store.py` handles leases, TTL and close/eviction independently of JAX.
`deepseek_v4/session_cache.py` handles V4 ownership. It subclasses the existing
chunk cache so V4 allocator and SWA dispatch remain unchanged. Scheduler idle
checks account for retained pages and request slots instead of treating them
as leaked memory.

## Validation

CPU tests use the real request pool, V4 host allocator and a small JAX
continuation-state pool:

```sh
PYTHONPATH=python pytest -q test/srt/test_session_store.py \
  test/srt/test_deepseek_v4_sessions.py test/srt/test_session_wire.py
```

Before deployment, run a TPU comparison against cold generation with fixed
token IDs and greedy sampling, both with and without overlap. Cover a chunk
boundary, C4/C128 compression boundaries, a sliding-window boundary, changed
and shortened prefixes, concurrent different sessions, and close/abort/TTL.
Compare output IDs and `meta_info.cached_tokens`, then confirm a cache flush
restores free pools without a leak warning. The local ownership tests are not
a substitute for that end-to-end numerical check or a 128K + 1K benchmark.

The included client can run against an already-started server:

```sh
python test/manual/deepseek_v4_session_smoke.py --url http://127.0.0.1:30000
# Larger run, after configuring sufficient server context length and KV capacity:
python test/manual/deepseek_v4_session_smoke.py \
  --prefix-lengths 131072 --extend-length 1024
```

It checks warm/cold greedy output IDs, prefix-hit counts, changed-prefix misses
and shortened-input misses, then closes its own sessions. It does not launch
jobs or alter a running server's settings. Exact window capacity must include
the generated tokens as well as prefix and extension lengths.
