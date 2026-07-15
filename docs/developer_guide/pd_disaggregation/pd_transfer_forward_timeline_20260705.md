# PD Transfer / Forward Timeline - 2026-07-05

## Scope

This note explains the current MiMo SWA PD disaggregation timing shape from two
angles:

1. Which prefill/decode operations are serial vs overlapped.
2. How a single request's TTFT path breaks down with the current metrics.

The numbers below are from Falcon exp `exp-5uqgg64144`, using 16k input and 128
output tokens.

## Serial And Parallel Work

### Prefill Side

For a 16k prompt, prefill currently runs 8 chunks of about 2k tokens each.

Serial work for one request:

```text
P forward chunk0
  -> P register chunk0
  -> P forward chunk1
  -> P register chunk1
  ...
  -> P forward chunk7
  -> P register chunk7
```

The forward chunks are serial for the same request. The chunk handoff happens
after that chunk's forward has produced valid KV pages.

Overlapped work:

```text
P forward chunk0 -> P register chunk0
                         D can start_read chunk0
P forward chunk1 ------------------------------ overlaps D reading chunk0
P register chunk1
                         D can start_read chunk1
...
```

So the transfer is not delayed until the whole prompt is complete. It starts
after chunk0 and overlaps later prefill chunks.

Measured prefill C16:

```text
forward total mean:          2742.54ms
forward_chunk_sum mean:      2713.65ms
forward_chunk_count:         8
forward_chunk_avg mean:      339.19ms
chunk_handoff_sum mean:      10.39ms
chunk_handoff_avg mean:      1.29ms
chunk_register_span mean:    2409.55ms
```

Interpretation:

- P-side handoff itself is small.
- Most of the request's prefill service time is model forward.
- `chunk_register_span` is close to 7 chunk intervals, which matches the
  chunked-overlap model.

### Decode Side

Serial work for one request:

```text
D waits until bootstrap metadata for chunk0 exists
  -> D allocates destination KV pages
  -> D initializes receiver
  -> D start_read chunk0
  -> D start_read newly published chunks as P publishes them
  -> D waits for all chunks done_recving
  -> D enqueue decode
  -> D first token
```

Overlapped work:

```text
D start_read chunk0/chunk1/... overlaps P's later forward chunks.
```

Measured decode C64 last64:

```text
metadata_wait mean:          2579.03ms
kv_alloc mean:               0.01ms
receiver_init mean:          0.10ms
transfer_setup mean:         0.20ms
first_chunk_wait mean:       4.60ms
start_read_call_sum mean:    0.30ms
start_read_call_count:       8
chunk_start_span mean:       2340.25ms
transfer_tail mean:          52.26ms
enqueue_decode mean:         2.51ms
kv_wait mean:                2399.61ms
```

Interpretation:

- D-side synchronous transfer setup is not the bottleneck.
- D starts reads over roughly the same time span that P registers chunks.
- The D-side final transfer tail after the last chunk start is about 50-65ms.

## Per-Request TTFT Chain

Current request path:

```text
Router admission
  -> prefill queue wait / capacity gate
  -> P forward chunk0
  -> P register chunk0
      -> D metadata visible
      -> D receiver start_read chunk0
  -> P forward chunk1
  -> P register chunk1
      -> D start_read chunk1
  ...
  -> P forward chunk7
  -> P register chunk7
      -> D start_read chunk7
  -> D all chunks done_recving
  -> D enqueue decode
  -> first token
  -> remaining decode tokens
```

The important distinction:

- P-side `sender_done_wait` is not a direct synchronous wait before P can start
  the next request. It is the time until P observes that Raiden has completed
  sending all chunks for a request and can safely release the prefill-side KV
  resources.
- D-side `done_recving` is the TTFT-critical transfer completion point for the
  decode worker.

Latest C16 prefill tail split:

```text
sender_done_wait mean:       324.68ms
prefill_reap_gap mean:       0.12ms
transfer_tail mean:          324.81ms
```

This says the old P-side `transfer_tail` is not cleanup overhead. It is almost
entirely the time until P observes Raiden `done_sending`.

## Current Problem Statement

The current TTFT profile is not primarily caused by expensive transfer API calls:

- P `send_chunk/register_read`: about 1-2ms per chunk.
- D `start_read`: about 0.3ms total per request at the current log precision.
- D final transfer tail: about 50ms.

The dominant request-service component is still prefill forward:

```text
~2.7s per 16k prompt, ~340ms per chunk.
```

At high concurrency, TTFT grows mostly because requests wait for prefill service
capacity. Router admission wait is not an optimization target by itself; it is
the visible backlog created when offered load exceeds one prefill worker's
service rate.

## Small Open Questions

1. Is P-side `sender_done_wait` real Raiden completion time, or just completion
   visibility inside the Raiden/poll path? A scheduler-level extra poll before
   forward did not reduce it.
2. If P observes sender completion earlier through a lower-level completion
   signal, does it improve resource release enough to affect C64/C128
   throughput or tail latency?
3. Does D-side `transfer_tail` stay around 50-65ms at larger concurrency, or
   does it grow under transfer pressure?
