"""Prefill-side scheduler Mixin for PD disaggregation.

The Mixin composes with :class:`sgl_jax.srt.managers.scheduler.Scheduler`
to add the prefill-only event loop and KV handoff machinery. The
scheduler dispatches between the normal event loop and the PD-aware
variants in ``run_scheduler_process`` based on
``ServerArgs.disaggregation_mode``.

What the Mixin owns:

* :class:`PrefillBootstrapQueue` — pending senders waiting for the
  decoder to ack the pull. Polled each tick; on SUCCESS the underlying
  ``req_to_token_pool`` slot is released. Self-pruning via
  ``KVSender._on_ack`` from the ZMQ listener thread is already in
  place (Stage 1 M7); this queue is the scheduler-side view that lets
  ``event_loop_normal_disagg_prefill`` reap finished senders without
  walking the manager's dict.
* :meth:`SchedulerDisaggregationPrefillMixin.process_prefill_chunk`:
  after the standard prefill batch completes, create a ``KVSender`` per
  request and queue it.
* :meth:`SchedulerDisaggregationPrefillMixin.send_kv_chunk`: walk the
  queue and free pool slots for any sender that reached SUCCESS /
  FAILED.

What the Mixin does NOT own:

* Choice of ``use_d2h_staging`` — read from ``server_args``.
* Bootstrap registration — that's the engine entry's job (after the
  Mixin's wrapper / notifier / manager are wired in).
* Real model invocation — the Mixin delegates to the scheduler's
  existing ``run_batch`` / ``process_batch_result``.
"""

from __future__ import annotations

from http import HTTPStatus
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import jax
import jax.numpy as jnp
from functools import partial

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVManager,
    JaxTransferKVSender,
)

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


# Stage 4 e2e FINDING-D: bucket the number of PAGES (not tokens)
# that the cross-layer gather has to materialize, so that XLA's
# per-shape compile pool is bounded. The pool's axis-0 is already
# ``num_pages`` (see ``MHATokenToKVPool._create_buffers``), so each
# request only gathers ``ceil(seqlen / page_size)`` rows. For a
# typical PD deployment with ``page_size=128`` this comes out to
# a handful of pages even for long prompts (4k tokens → 32 pages),
# and these buckets cover everything up to the kernel's
# ``max_seq_len`` (8192 tokens → 64 pages at page_size=128).
_KV_GATHER_PAGE_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


def _pad_to_page_bucket(num_pages: int) -> int:
    for b in _KV_GATHER_PAGE_BUCKETS:
        if b >= num_pages:
            return b
    return _KV_GATHER_PAGE_BUCKETS[-1]


# Stage 4 e2e FINDING-D root fix: do the per-layer gather in ONE
# jit trace over the full list[jax.Array] of layer buffers. This
# pattern is taken from tpu-inference
# (`tpu_inference/distributed/tpu_connector.py:913-917` for the
# jit'd gather, and `:694` for the indices-on-mesh placement,
# `utils.py:325` for the ``device_array`` helper). That codebase
# runs `jax.experimental.transfer` over the same JAX 0.8.1 API
# on TPU and has been validated in production PD setups.
#
# The critical assumption that lets this fit in HBM: the gather
# axis (pool axis-0) is sized ``num_pages``, not ``num_tokens``.
# With ``page_size=1`` the pool axis-0 grows to
# ``max_total_num_tokens`` (~400k) and the sharded gather
# collective intermediate exceeds HBM. We enforce
# ``page_size >= 128`` for PD in ``ServerArgs.__post_init__``.
#
# Three learnings carried over:
#   1. ``@jax.jit`` over the whole ``list[jax.Array]`` lets XLA
#      fuse the per-layer gathers into a single program. Without
#      this each ``.at[].get()`` call site got its own cache
#      entry; cumulative compile footprint OOMed HBM.
#   2. **Indices must be placed on the same mesh as the cache**
#      (``jax.device_put(np.asarray(...), NamedSharding(mesh,
#      P(None)))``). A bare ``jnp.asarray(...)`` puts them on
#      the default device; JAX 0.8.1 cannot infer the gather
#      output sharding from that.
#   3. **The ``out_sharding`` we pass describes the GATHER
#      OUTPUT** (gather axis replicated since indices are
#      replicated, other axes carried from the pool's pspec),
#      NOT the pool's full sharding. Passing the pool's full
#      sharding made JAX 0.8.1 materialize the gather output at
#      pool-size before slicing.
@partial(jax.jit, static_argnames=("out_sharding",))
def _jit_gather_all_layers(buffers, page_indices, out_sharding):
    """Gather ``page_indices`` from every per-layer buffer in one jit.

    ``buffers`` is a list of ``jax.Array``, one per KV layer; all
    share shape ``(num_pages, page_size, head_num*2/packing,
    packing, head_dim)`` / dtype / sharding. ``page_indices`` is
    a 1-D ``jax.Array`` of page ids (padded to a
    ``_KV_GATHER_PAGE_BUCKETS`` bucket, placed on the same mesh
    as ``buffers``). ``out_sharding`` is the NamedSharding for
    the gather output (gather axis replicated since indices are
    replicated, other axes inherited from the pool).
    Returns a list of the same length whose elements are the
    gathered slices, each shaped ``(num_pages_padded, page_size,
    head_num*2/packing, packing, head_dim)``.
    """

    return [
        buf.at[page_indices].get(out_sharding=out_sharding) for buf in buffers
    ]


@dataclass
class PrefillBookkeeping:
    """Per-request prefill-side state tracked by the Mixin."""

    req_id: str
    sender: JaxTransferKVSender
    # Optional callback the scheduler runs when this entry reaches a
    # terminal state — used to release ``req_to_token_pool`` and any
    # owned KV indices.
    on_terminal: Optional["object"] = None


class PrefillBootstrapQueue:
    """Tracks senders pending decoder ack.

    Lock-protected because both the scheduler thread (insert) and the
    ZMQ listener thread (the sender's ``_on_ack`` may prune the
    underlying mgr entry concurrently) may inspect it. The queue's
    own state mirrors the sender's state machine — the queue does not
    drive transitions, only observes via ``sender.poll()``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, PrefillBookkeeping] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def add(
        self,
        req_id: str,
        sender: JaxTransferKVSender,
        on_terminal=None,
    ) -> None:
        with self._lock:
            if req_id in self._entries:
                raise ValueError(
                    f"PrefillBootstrapQueue already tracks "
                    f"req_id={req_id!r}"
                )
            self._entries[req_id] = PrefillBookkeeping(
                req_id=req_id, sender=sender, on_terminal=on_terminal
            )

    def drain_terminal(self) -> List[PrefillBookkeeping]:
        """Remove and return entries whose senders reached SUCCESS or
        FAILED. The caller is expected to invoke each entry's
        ``on_terminal`` to free scheduler resources.
        """

        terminal: List[PrefillBookkeeping] = []
        with self._lock:
            for req_id, entry in list(self._entries.items()):
                state = entry.sender.poll()
                if state in (KVPoll.SUCCESS, KVPoll.FAILED):
                    terminal.append(entry)
                    del self._entries[req_id]
        return terminal

    def snapshot_states(self) -> Dict[str, KVPoll]:
        with self._lock:
            return {
                rid: entry.sender.poll()
                for rid, entry in self._entries.items()
            }


class SchedulerDisaggregationPrefillMixin:
    """Mixin glued onto :class:`Scheduler` for PD prefill mode.

    The Mixin reads attributes that the engine entry is expected to
    install before calling the event loop:

    * ``self.disagg_kv_manager`` — :class:`JaxTransferKVManager`
    * ``self.disagg_prefill_queue`` — :class:`PrefillBootstrapQueue`
    * ``self.disagg_use_d2h_staging`` — bool, from ServerArgs

    The Mixin does NOT mutate the scheduler's existing ``__init__``;
    the engine entry installs these attributes after Scheduler init
    based on ``ServerArgs.disaggregation_mode``.
    """

    disagg_kv_manager: JaxTransferKVManager
    disagg_prefill_queue: PrefillBootstrapQueue
    disagg_use_d2h_staging: bool

    def event_loop_normal_disagg_prefill(self) -> None:
        """Prefill-only event loop.

        Mirrors ``event_loop_normal`` but skips the decode side of
        ``process_batch_result`` and runs ``send_kv_chunk`` each tick
        to reap finished senders.
        """

        while True:
            recv_reqs = (
                self._comm_backend.recv_requests()  # type: ignore[attr-defined]
                if self._comm_backend is not None  # type: ignore[attr-defined]
                else self.recv_requests()  # type: ignore[attr-defined]
            )
            recv_reqs = self.select_dp_for_request(recv_reqs)  # type: ignore[attr-defined]
            self.process_input_requests(recv_reqs)  # type: ignore[attr-defined]

            if self._engine_paused:  # type: ignore[attr-defined]
                continue

            batch = self.get_next_batch_to_run()  # type: ignore[attr-defined]
            self.cur_batch = batch  # type: ignore[attr-defined]

            if batch:
                result = self.run_batch(batch)  # type: ignore[attr-defined]
                self.process_prefill_chunk(batch, result)
            else:
                # Prefill-only PD requests may have reached terminal on
                # the previous tick's side-channel ack. Drain them before
                # any idle-path bookkeeping so their KV ownership is
                # returned first.
                self.send_kv_chunk()
                # Like decode-mode PD, prefill-mode PD owns KV outside
                # the scheduler's normal running-batch accounting while
                # transfers are in flight. The generic idle leak checks
                # therefore misreport live PD transfers as leaks.
                self.new_token_ratio = self.init_new_token_ratio  # type: ignore[attr-defined]
                if self._comm_backend is not None:  # type: ignore[attr-defined]
                    self._comm_backend.wait_for_new_requests(0.001)  # type: ignore[attr-defined]

            self.send_kv_chunk()
            self.last_batch = batch  # type: ignore[attr-defined]

    def process_prefill_chunk(self, batch, result) -> None:
        """After standard prefill completes, hand each req's KV off
        through ``KVSender`` and queue it for ack.

        The Mixin must read each req's prefilled KV from the
        scheduler's existing pool. Concrete extraction of the per-req
        KV tensor depends on the model + pool layout; the Mixin
        delegates to ``_extract_req_kv`` (overridable / monkey-
        patchable for tests).
        """

        pd_reqs = [
            req
            for req in batch.reqs
            if getattr(req, "bootstrap_room", None) is not None
        ]
        if not pd_reqs:
            self.process_batch_result(batch, result)  # type: ignore[attr-defined]
            return

        # Prefill-only PD mode must NOT keep generating locally after
        # producing the prompt KV. The decode side owns continuation.
        # We still need to mark batch sampling metadata resolved so the
        # scheduler can proceed to the next request.
        self.set_next_batch_sampling_info_done(batch)  # type: ignore[attr-defined]

        for req in batch.reqs:
            if req.bootstrap_room is None:
                # null-mode req leaked through; ignore.
                continue
            req_id = req.rid
            # The req stays in the batch for the decode iters that
            # follow prefill; ``process_prefill_chunk`` fires on every
            # iter the req is in. Skip if we've already sent its KV.
            if req_id in self.disagg_prefill_queue._entries:
                continue
            try:
                device_kv = self._extract_req_kv(req)
            except Exception:
                logger.exception(
                    "failed to extract KV for req_id=%s; skipping send",
                    req_id,
                )
                continue
            self._maybe_log_prefill_extract_debug(
                req,
                device_kv,
                use_d2h_staging=self.disagg_use_d2h_staging,
            )
            sender = self.disagg_kv_manager.create_sender(req_id)
            sender.init(
                kv_indices=None,
                transfer_id=(
                    getattr(req, "disagg_transfer_id", None) or req_id
                ),
            )
            sender.attach_payload(
                device_kv,
                use_d2h_staging=self.disagg_use_d2h_staging,
            )
            sender.send()

            def _on_terminal(req_obj=req, sender_obj=sender):
                self._on_prefill_transfer_terminal(req_obj, sender_obj)

            self.disagg_prefill_queue.add(
                req_id, sender, on_terminal=_on_terminal
            )

    def send_kv_chunk(self) -> None:
        """Reap senders that reached SUCCESS / FAILED."""

        terminal = self.disagg_prefill_queue.drain_terminal()
        for entry in terminal:
            on_terminal = entry.on_terminal
            if on_terminal is None:
                continue
            try:
                on_terminal()
            except Exception:
                logger.exception(
                    "on_terminal for req_id=%s raised; continuing",
                    entry.req_id,
                )

    # ------------------------------------------------------------------
    # Overridable / test-friendly hooks
    # ------------------------------------------------------------------

    def _extract_req_kv(self, req: "Req"):
        """Return the device-side KV tensor for ``req``.

        Default implementation reads the prefilled KV out of
        :class:`MHATokenToKVPool` by gathering at the token indices
        the scheduler allocated for this req (looked up in
        ``req_to_token_pool.req_to_token[req.req_pool_idx, :seqlen]``).
        Stacks all layers into a single fused tensor so the wire
        carries one register_pull / pull pair.

        Output shape: ``(layer_num, seqlen, page_size, head_num*2/
        packing, packing, head_dim)`` matching the per-layer fused
        buffer layout. Subclass / monkey-patch for non-MHA pools.
        """

        req_to_token = self.req_to_token_pool.req_to_token  # type: ignore[attr-defined]
        # Extract KV for the INPUT positions only. After
        # process_batch_result runs, ``req.fill_ids`` includes the
        # just-sampled decode token (origin_input_ids + output_ids[0]),
        # but the KV for that token wasn't computed during prefill —
        # the decoder produces it. We want positions 0..N-1 where
        # N = len(origin_input_ids).
        kv_pool = (
            self.token_to_kv_pool_allocator.get_kvcache()  # type: ignore[attr-defined]
        )
        page_size = kv_pool.page_size
        seqlen = len(req.origin_input_ids)
        # FINDING-D: gather by PAGE, not token. ``req_to_token`` holds
        # token-slot ids; with a paged allocator those slots are
        # page-aligned (allocator emits ``out_pages[:, None] *
        # page_size + np.arange(page_size)`` — see
        # PagedTokenToKVPoolAllocator.alloc), so every consecutive
        # ``page_size`` ids in ``req_to_token`` share a page id of
        # ``slot_id // page_size``. We pick one slot id per page
        # (every ``page_size``-th entry) and divide to get page
        # ids, then gather those rows from the pool — whose axis-0
        # is num_pages.
        #
        # We include the page containing the partial tail of the
        # prompt (e.g. seqlen=5, page_size=128 → 1 page); the
        # page-internal slots past position seqlen hold stale data
        # but D never reads past seqlen so that doesn't matter.
        num_pages = (seqlen + page_size - 1) // page_size
        padded_pages = _pad_to_page_bucket(num_pages)
        # Take the first slot of each padded-page (reading past
        # seqlen is safe — req_to_token rows are pre-allocated to
        # max_total_tokens, and downstream only uses the first
        # ``num_pages`` entries).
        page_id_source = (
            req_to_token[
                req.req_pool_idx,
                : padded_pages * page_size : page_size,
            ]
        )
        # FINDING-D layer 2 fix: place indices on the same mesh as
        # the pool (replicated). A bare ``jnp.asarray(...)`` lands
        # on the default device and JAX 0.8.1 can't infer the
        # gather output sharding from that.
        import numpy as _np
        from jax.sharding import NamedSharding as _NamedSharding
        from jax.sharding import PartitionSpec as _P

        idx_sharding = _NamedSharding(kv_pool.mesh, _P(None))
        page_indices = jax.device_put(
            _np.asarray(page_id_source) // page_size,
            idx_sharding,
        )
        # FINDING-D layer 3 fix: the gather output sharding must
        # describe the GATHER OUTPUT (gather axis replicated, since
        # indices are replicated), NOT the pool's full sharding.
        # The pool's pspec is
        # ``P(attention_data_axis, None, kv_partition_axis, None, None)``;
        # we replace the first axis (was sharded on the pool's
        # ``num_pages`` dim) with ``None`` because the gathered
        # output has a fresh axis 0 of length ``padded_pages``.
        pool_pspec = kv_pool.kv_sharding.spec
        gather_pspec = _P(None, *pool_pspec[1:])
        gather_out_sharding = _NamedSharding(kv_pool.mesh, gather_pspec)
        # Single-jit gather across all layers — see
        # ``_jit_gather_all_layers`` docstring for the FINDING-D
        # background.
        layer_buffers = [
            kv_pool.get_kv_buffer(layer_id)
            for layer_id in range(
                kv_pool.start_layer,
                kv_pool.start_layer + kv_pool.layer_num,
            )
        ]
        layer_kvs = _jit_gather_all_layers(
            layer_buffers, page_indices, gather_out_sharding
        )
        # ``layer_kvs[i].shape = (padded_pages, page_size, ...)``.
        # Stack to (layers, padded_pages, page_size, ...) — D's
        # ``_write_kv_to_pool`` scatters back into its own paged
        # pool by page id, only touching the first ``num_pages``
        # entries. Stale tail-page slots past seqlen aren't read.
        return jnp.stack(layer_kvs, axis=0)

    def _release_prefill_req_resources(self, req: "Req") -> None:
        """Release ``req_to_token_pool`` and any indices held by
        ``req``. Default delegates to the scheduler's standard
        ``cache_finished_req`` path; tests stub this out.
        """

        if hasattr(self, "cache_finished_req"):
            self.cache_finished_req(req)  # type: ignore[attr-defined]

    def _on_prefill_transfer_terminal(
        self, req: "Req", sender: JaxTransferKVSender
    ) -> None:
        try:
            if sender.poll() == KVPoll.SUCCESS:
                self._finish_prefill_only_success(req)
            else:
                self._finish_prefill_only_failure(req, sender)
        finally:
            if hasattr(sender, "clear"):
                sender.clear()
            self._release_prefill_req_resources(req)

    def _finish_prefill_only_success(self, req: "Req") -> None:
        from sgl_jax.srt.managers.schedule_batch import FINISH_LENGTH

        req.finished_reason = FINISH_LENGTH(length=0)
        req.output_ids = []
        req.finished_len = 0
        if hasattr(self, "stream_output"):
            self.stream_output(  # type: ignore[attr-defined]
                [req],
                getattr(req, "return_logprob", False),
                getattr(req, "return_output_logprob_only", False),
            )

    def _finish_prefill_only_failure(
        self, req: "Req", sender: JaxTransferKVSender
    ) -> None:
        from sgl_jax.srt.managers.schedule_batch import FINISH_ABORT

        error_message = (
            f"Prefill transfer failed for req_id={getattr(req, 'rid', None)!r} "
            f"bootstrap_room={getattr(req, 'bootstrap_room', None)!r}"
        )
        try:
            sender.failure_exception()
        except Exception as exc:  # noqa: BLE001
            error_message = f"{error_message}: {exc}"
        req.finished_reason = FINISH_ABORT(
            error_message,
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "PDTransferError",
        )
        req.output_ids = []
        if hasattr(self, "stream_output"):
            self.stream_output(  # type: ignore[attr-defined]
                [req],
                getattr(req, "return_logprob", False),
                getattr(req, "return_output_logprob_only", False),
            )

    def _maybe_log_prefill_extract_debug(self, req: "Req", kv, **meta) -> None:
        from sgl_jax.srt.disaggregation.debug_utils import (
            build_kv_debug_snapshot,
            kv_debug_enabled,
        )

        if not kv_debug_enabled(getattr(req, "rid", None)):
            return

        snapshot = build_kv_debug_snapshot(kv)
        logger.warning(
            "PD-KV-DEBUG prefill_extract req_id=%s shape=%s dtype=%s "
            "sharding=%s digest=%s sample=%s meta=%s",
            req.rid,
            snapshot.shape,
            snapshot.dtype,
            snapshot.sharding,
            snapshot.global_digest,
            snapshot.sample_page_digests(),
            meta,
        )
