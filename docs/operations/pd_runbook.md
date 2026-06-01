# PD Production Runbook

Stage 4 H-F deliverable. Treat this as the on-call cheat sheet:
deployment checklist → health probes → decision tree → capacity
formulae → tools index. Update when behavior changes; this file is
the source of truth, not the RFCs.

## 1. Deployment checklist

### Pod spec (required)
- `capabilities.add: [IPC_LOCK]` — pinned-host memory needs it.
- A dedicated bootstrap deployment (`deploy/pd-bootstrap`, replicas=1).
- For each engine pod, the three ports must be reachable from peer pods:
  - `disaggregation_transfer_port` (default `30001`, jax_transfer)
  - `disaggregation_side_channel_port` (default `9600`, ZMQ ack)
  - `port` (HTTP control plane, default `30000`)

### ServerArgs (required)
| flag | typical value |
|---|---|
| `--disaggregation-mode` | `prefill` or `decode` |
| `--disaggregation-bootstrap-url` | `http://pd-bootstrap.<ns>.svc:8998` |
| `--disaggregation-host-ip` | pod IP (or leave empty to auto-resolve) |
| `--disaggregation-enable-d2h` | `false` (see §8) |
| `--disaggregation-channel-number` | `4` (validated knee on v6e) |
| `--disaggregation-shared-secret` | shared across pods + bootstrap |

The shared secret can also be set via `SGL_JAX_PD_SHARED_SECRET`
env var (env wins).

### Timeout matrix (defaults in H-B)
| flag | default | when to bump |
|---|---|---|
| `--disaggregation-bootstrap-timeout-seconds` | 5 | bootstrap on a slow side-net |
| `--disaggregation-pull-timeout-seconds` | 30 | very large KV (>5 GB) |
| `--disaggregation-ack-timeout-seconds` | 60 | very large KV, or expected D pauses |
| `--disaggregation-orphan-reaper-interval-seconds` | 5 | rarely needed |

## 2. Health probes

```bash
# bootstrap liveness — open even with auth on
curl -sf http://pd-bootstrap:8998/health

# bootstrap with auth — should 200
curl -sf -H "Authorization: Bearer $SGL_JAX_PD_SHARED_SECRET" \
    http://pd-bootstrap:8998/list_prefills

# scheduler readiness — look for this line in server.log
grep "ready to roll" /tmp/server.log

# scheduler heartbeat — every ~10s the prefill scheduler logs:
grep "bootstrap heartbeat" /tmp/server.log
```

## 3. Decision tree — request 5xx

```
5xx rate spikes
├── pd_transfer_failures_total{reason=...} which one?
│   ├── timeout       → check DCN latency; bump pull/ack timeout
│   ├── bootstrap_lookup → bootstrap up? registry size > 0?
│   ├── receiver_init → check D pod logs for KVReceiver traceback
│   ├── ack_send      → side-channel port reachable D→P?
│   ├── pull_init     → transfer port reachable D→P?
│   ├── auth          → secret mismatch; rotate or align
│   └── shutdown      → expected during graceful shutdown
├── pd_bootstrap_registry_size dropped?
│   └── prefill pods dying — kubectl logs to find root cause
├── pd_host_pool_used_buffers near max?
│   └── pool exhausted — bump pool size or temporarily disable d2h
└── pd_transfer_inflight stuck non-zero with low qps?
    └── stuck transfers — check reaper logs, may need to restart
```

## 4. Capacity formulae

- Host pool size: `pool_size = max_concurrent_prefill_requests * 1.5`
- Per-buffer tokens: `max_total_num_tokens / pool_size`
- Channel count: 1 per ~6 GB/s of available DCN BW
- Bootstrap timeout: ≥ TCP RTT * 10

Example: v6e-16 pod (4× chips) with ~25 GB/s DCN BW → 4 channels;
1024-context Qwen3 GQA model → host pool 64 × ~50 MB = 3.2 GB
pinned per pod.

## 5. Tools index

| script | purpose |
|---|---|
| `python -m sgl_jax.srt.disaggregation.run_bootstrap` | standalone bootstrap server |
| `python -m sgl_jax.srt.disaggregation.router` | mini_lb-style fan-out router |
| `python -m sgl_jax.srt.disaggregation.tools.sweep_channels` | DCN throughput sweep |
| `python -m sgl_jax.srt.disaggregation.tools.stress` | qps × duration stress |
| `python/sgl_jax/srt/disaggregation/tools/chaos.sh` | three chaos scenarios |

## 6. Graceful shutdown

Engines call `JaxTransferKVManager.graceful_shutdown(timeout=30s)`
from their SIGTERM handler:

1. The scheduler stops accepting new requests (Mixin layer).
2. Bootstrap registration is dropped so the router stops sending here.
3. In-flight senders/receivers are drained, or force-failed at the
   timeout. `pd_transfer_failures_total{reason="shutdown"}` counts
   the force-fail tail.
4. The wrapper, ZMQ notifier, and orphan reaper stop.

Trigger from kubectl: standard `terminationGracePeriodSeconds: 45`
on the pod spec gives the engine the 30s drain plus 15s slack for
its own teardown.

## 7. Rolling upgrade

Order: **bootstrap → D → P** (so D is always at-or-newer than P).

The Stage 4 H-D version handshake refuses peers below
`MIN_COMPATIBLE_VERSION = PROTOCOL_VERSION - 1`. Bumping
`PROTOCOL_VERSION` in `bootstrap.py` without also bumping
`MIN_COMPATIBLE_VERSION` keeps one minor of skew tolerance.

## 8. Known limitations (Stage 4)

- **D2H staging (path A) not yet integrated.** `disaggregation_enable_d2h`
  defaults to `false`; flipping it to `true` triggers a boot-time
  `RuntimeError` because the model-specific `QueueHostKVPool` is not
  wired into the scheduler yet. The Stage 2.5 e2e validation used
  path B (HBM-direct). Wiring path A end-to-end is the next
  follow-up and required before the RFC's "default ON" promise can
  be delivered.
- Multi-channel sweep tool measures register-side throughput only —
  a paired puller harness is a follow-up.
- mTLS auth mode is not yet wired (shared-secret only).
- Transfer-pull pre-handshake HMAC (RFC §2) is deferred; the ZMQ
  ack channel is the only HMAC-protected hop today.
- Stress harness drives a single router; multi-router scale-out
  stress is operator-supplied.
