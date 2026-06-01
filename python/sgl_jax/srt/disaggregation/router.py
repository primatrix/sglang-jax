"""PD multi-host router integration notes (Stage 3).

This module is documentation-only. It explains the deployment shape
agreed in ``docs/rfc/2026-05-25-pd-multihost-routing.md`` and the
contract the engine exposes so an unmodified ``sglang_router``
(``sglang/python/sglang/srt/disaggregation/mini_lb.py``) can route
PD traffic without knowing about PD.

::

                ┌──────────────────┐
                │   sglang_router  │   (single instance or replicated)
                └────────┬─────────┘
                         │ HTTP /v1/...
            ┌────────────┼────────────┐
            │            │            │
         ┌──▼──┐      ┌──▼──┐      ┌──▼──┐
         │  D  │ ...  │  D  │      │  D  │   (sgl_jax decode engines)
         └─────┘      └─────┘      └─────┘
            │ KV transfer (DCN, see Stage 0-1)
         ┌──▼──┐      ┌──▼──┐      ┌──▼──┐
         │  P  │ ...  │  P  │      │  P  │   (sgl_jax prefill engines)
         └─────┘      └─────┘      └─────┘
                         │
                ┌────────▼─────────┐
                │ Bootstrap Server │   (centralised, Stage 2)
                └──────────────────┘

Operator contract:

* Each engine is launched with ``--disaggregation-mode {prefill,
  decode}``, ``--disaggregation-bootstrap-url
  http://<bootstrap_host>:<bootstrap_port>``, and (Stage 3)
  ``--disaggregation-host-ip <this-host>`` if auto-resolution from
  ``$HOSTNAME`` is undesired.
* The bootstrap server is one process started outside the engines
  (Stage 2 also supports an in-process launch via
  ``DISAGG_LAUNCH_BOOTSTRAP=1`` on the prefill engine for single-
  node dev deployments).
* Multi-host P (e.g. 4 hosts × 4 chips for one tp=16 prefill role)
  launches as N independent processes; each one calls
  :func:`sgl_jax.srt.disaggregation.host_ip.resolve_host_ip` to
  publish its own per-host IP. The bootstrap server ends up with N
  ``PrefillInfo`` entries, one per host.
* ``sglang_router`` only sees the D endpoints and load-balances
  HTTP requests across them. It does NOT need to know about
  bootstrap, KV transfer, or the P endpoints.

Request contract (router → D):

* The router forwards a normal OpenAI-style chat or completion
  request body to D's HTTP server.
* If the body does NOT carry ``bootstrap_host``, ``bootstrap_port``,
  ``bootstrap_room``, D's tokenizer auto-derives them (Stage 3):
  ``bootstrap_host`` and ``bootstrap_port`` come from the engine's
  ``--disaggregation-bootstrap-url``; ``bootstrap_room`` is a stable
  32-bit CRC32 of the request id, so retries of the same rid hit
  the same prefill peer. Operators may still set these explicitly
  on the request body to override.

Suggested rollout:

1. Start ``--disaggregation-mode prefill`` on each P host (4 hosts
   for a tp=16 P slice). Each registers itself with the bootstrap
   server; the heartbeat daemon keeps them alive past the 30 s TTL.
2. Start ``--disaggregation-mode decode`` on each D host.
3. Start ``sglang_router`` (or any HTTP load balancer; the router
   is plain HTTP fan-out) pointing at the D HTTP endpoints.
4. Smoke: ``curl router/v1/chat/completions -d '<prompt>'`` →
   D's tokenizer derives ``bootstrap_room`` → bootstrap lookup
   picks a P → P prefills + transfers → D decodes → token stream.

This module is currently documentation only; no code lives here.
The contract is enforced by Stage 2's tokenizer derivation +
bootstrap server + scheduler dispatch. Stage 3 extended it for
multi-host P registration and removed the ``DISAGG_HOST`` env hack.
"""

# Empty module body — see docstring above.
