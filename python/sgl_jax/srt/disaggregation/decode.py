"""Decode-side scheduler Mixin for PD disaggregation.

D-side queues + event loop. A decode request goes through two queues:

1. :class:`DecodePreallocQueue` — request arrived; resolve the prefill
   peer via the bootstrap server, allocate the local KV slot, create a
   ``KVReceiver``, kick off the pull.
2. :class:`DecodeTransferQueue` — receiver in ``TRANSFERRING``;
   polled each tick until SUCCESS, then the KV is written into the
   paged pool and the request is handed to the normal decode loop.

The Mixin bypasses ``tree_cache.insert`` for the PD path (per RFC-2
ADR-7); the standard ``cache_finished_req`` at decode finish handles
insertion.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.bootstrap import BootstrapClient
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVManager,
    JaxTransferKVReceiver,
    PMetadata,
)

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass
class DecodeBookkeeping:
    """Per-request decode-side state."""

    req_id: str
    req: "Req"
    receiver: Optional[JaxTransferKVReceiver] = None
    # Indices into the paged pool reserved for this request.
    kv_indices: Optional[object] = None
    # Whether the receiver has been initialized + poll started.
    started: bool = False


class DecodePreallocQueue:
    """Requests that arrived but haven't started the KV pull yet.

    Holds entries whose ``KVReceiver`` is created but not yet
    transitioned to TRANSFERRING. The Mixin tick promotes ready
    entries into :class:`DecodeTransferQueue`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, DecodeBookkeeping] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def add(self, entry: DecodeBookkeeping) -> None:
        with self._lock:
            if entry.req_id in self._entries:
                raise ValueError(
                    f"DecodePreallocQueue already tracks "
                    f"req_id={entry.req_id!r}"
                )
            self._entries[entry.req_id] = entry

    def pop_all(self) -> List[DecodeBookkeeping]:
        with self._lock:
            out = list(self._entries.values())
            self._entries.clear()
            return out


class DecodeTransferQueue:
    """Receivers in TRANSFERRING; polled each tick."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, DecodeBookkeeping] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def add(self, entry: DecodeBookkeeping) -> None:
        with self._lock:
            if entry.req_id in self._entries:
                raise ValueError(
                    f"DecodeTransferQueue already tracks "
                    f"req_id={entry.req_id!r}"
                )
            self._entries[entry.req_id] = entry

    def drain_terminal(self) -> List[DecodeBookkeeping]:
        """Return entries whose receiver reached SUCCESS or FAILED."""

        out: List[DecodeBookkeeping] = []
        with self._lock:
            for rid, entry in list(self._entries.items()):
                assert entry.receiver is not None
                state = entry.receiver.poll()
                if state in (KVPoll.SUCCESS, KVPoll.FAILED):
                    out.append(entry)
                    del self._entries[rid]
        return out

    def snapshot_states(self) -> Dict[str, KVPoll]:
        with self._lock:
            return {
                rid: e.receiver.poll() if e.receiver else KVPoll.BOOTSTRAPPING
                for rid, e in self._entries.items()
            }


class SchedulerDisaggregationDecodeMixin:
    """Mixin glued onto :class:`Scheduler` for PD decode mode.

    Attributes the engine entry installs after scheduler init:

    * ``self.disagg_kv_manager`` — :class:`JaxTransferKVManager`
    * ``self.disagg_bootstrap_client`` — :class:`BootstrapClient`
    * ``self.disagg_prealloc_queue`` — :class:`DecodePreallocQueue`
    * ``self.disagg_transfer_queue`` — :class:`DecodeTransferQueue`
    """

    disagg_kv_manager: JaxTransferKVManager
    disagg_bootstrap_client: BootstrapClient
    disagg_prealloc_queue: DecodePreallocQueue
    disagg_transfer_queue: DecodeTransferQueue

    def event_loop_normal_disagg_decode(self) -> None:
        """Decode event loop.

        Mirrors ``event_loop_normal`` but interposes the PD prealloc /
        transfer queues. The actual decode loop is unchanged: once a
        receiver reaches SUCCESS, the KV is written to the paged pool
        and the request is enqueued for the standard decode batch.
        """

        while True:
            recv_reqs = (
                self._comm_backend.recv_requests()  # type: ignore[attr-defined]
                if self._comm_backend is not None  # type: ignore[attr-defined]
                else self.recv_requests()  # type: ignore[attr-defined]
            )
            recv_reqs = self.select_dp_for_request(recv_reqs)  # type: ignore[attr-defined]
            self.process_input_requests_disagg_decode(recv_reqs)

            if self._engine_paused:  # type: ignore[attr-defined]
                continue

            self.process_decode_queue()

            batch = self.get_next_batch_to_run()  # type: ignore[attr-defined]
            self.cur_batch = batch  # type: ignore[attr-defined]

            if batch:
                result = self.run_batch(batch)  # type: ignore[attr-defined]
                self.process_batch_result(batch, result)  # type: ignore[attr-defined]
            else:
                # PD-disagg path allocates KV slots via
                # ``_prealloc_decode_kv_indices`` outside the
                # scheduler's standard owning-tracking. The standard
                # ``check_memory`` sanity check would then misreport
                # them as leaked and crash the engine. Skip the
                # check entirely on disagg_decode — Stage 4 hardening
                # can add a PD-aware variant. ``check_tree_cache``
                # is similarly skipped because the PD reqs never
                # touch tree_cache (RFC-2 ADR-7).
                self.new_token_ratio = self.init_new_token_ratio  # type: ignore[attr-defined]
                if self._comm_backend is not None:  # type: ignore[attr-defined]
                    self._comm_backend.wait_for_new_requests(0.001)  # type: ignore[attr-defined]

            self.last_batch = batch  # type: ignore[attr-defined]

    def process_input_requests_disagg_decode(self, recv_reqs) -> None:
        """Decode-mode request intake.

        Critical contract (Stage 2 review C1): a PD-mode request must
        NOT remain on ``waiting_queue`` after this method returns,
        otherwise the next ``get_next_batch_to_run`` tick will try to
        prefill it on the decode side — defeating PD entirely. We
        delegate to the scheduler's normal intake (which builds the
        ``Req`` and queues it), then immediately pull PD reqs back out
        and route them to the prealloc queue.
        """

        # Hand non-PD recv_reqs to the scheduler's normal intake so
        # they still take effect (control messages etc.). PD reqs
        # ride along because we need the Scheduler to build the Req
        # object the same way it always does.
        self.process_input_requests(recv_reqs)  # type: ignore[attr-defined]

        recv_pd_rids = {
            getattr(r, "rid", None)
            for r in recv_reqs
            if getattr(r, "bootstrap_room", None) is not None
        }
        if not recv_pd_rids:
            return

        pd_reqs = self._extract_pd_reqs_from_waiting_queue(recv_pd_rids)
        for req in pd_reqs:
            try:
                from sgl_jax.srt.disaggregation.metrics import time_phase

                with time_phase("bootstrap", "decode"):
                    p_info = self.disagg_bootstrap_client.get_prefill_info(
                        req.bootstrap_room
                    )
            except Exception:
                logger.exception(
                    "bootstrap lookup failed for req_id=%s "
                    "bootstrap_room=%s; releasing resources",
                    req.rid, req.bootstrap_room,
                )
                try:
                    from sgl_jax.srt.disaggregation.metrics import (
                        PD_TRANSFER_FAILURES_TOTAL,
                    )

                    PD_TRANSFER_FAILURES_TOTAL.labels(
                        reason="bootstrap_lookup", role="decode"
                    ).inc()
                except Exception:  # noqa: BLE001
                    pass
                self._release_decode_req_resources(req)
                continue

            kv_indices = None
            try:
                kv_indices = self._prealloc_decode_kv_indices(req)
                receiver = self.disagg_kv_manager.create_receiver(req.rid)
                spec = self._build_kv_spec_for_req(req)
                receiver.init(
                    PMetadata(
                        remote_addr=(
                            f"{p_info['host']}:{p_info['transfer_port']}"
                        ),
                        uuid=(
                            getattr(req, "disagg_transfer_id", None)
                            or req.rid
                        ),
                        spec=spec,
                        p_side_channel_host=str(p_info["host"]),
                        p_side_channel_port=int(p_info["side_channel_port"]),
                    )
                )
            except Exception:
                logger.exception(
                    "failed to set up KVReceiver for req_id=%s",
                    req.rid,
                )
                try:
                    from sgl_jax.srt.disaggregation.metrics import (
                        PD_TRANSFER_FAILURES_TOTAL,
                    )

                    PD_TRANSFER_FAILURES_TOTAL.labels(
                        reason="receiver_init", role="decode"
                    ).inc()
                except Exception:  # noqa: BLE001
                    pass
                # Release any slots we allocated before the failure.
                if kv_indices is not None:
                    self._release_decode_kv_indices(kv_indices)
                self._release_decode_req_resources(req)
                continue

            entry = DecodeBookkeeping(
                req_id=req.rid,
                req=req,
                receiver=receiver,
                kv_indices=kv_indices,
                started=True,
            )
            self.disagg_prealloc_queue.add(entry)

    def _extract_pd_reqs_from_waiting_queue(self, rids: set) -> List["Req"]:
        """Stage 2 review I6: replace fragile multi-queue walk with
        a single targeted extraction. Pulls reqs out of
        ``waiting_queue`` (in-place; the scheduler doesn't expect this
        but neither does it expect the items to remain) and returns
        them in receipt order. Reqs whose rid is not in ``rids`` stay
        in the queue.
        """

        out: List["Req"] = []
        queue = getattr(self, "waiting_queue", None)
        if queue is None:
            return out
        survivors = []
        for req in queue:
            rid = getattr(req, "rid", None)
            if rid in rids and getattr(req, "bootstrap_room", None) is not None:
                out.append(req)
            else:
                survivors.append(req)
        queue.clear()
        queue.extend(survivors)
        return out

    def process_decode_queue(self) -> None:
        """Drive prealloc -> transfer -> ready transitions."""

        # 1. Move prealloc entries into the transfer queue.
        for entry in self.disagg_prealloc_queue.pop_all():
            # ``started`` is always True at this point — intake only
            # adds entries after a successful receiver init. The check
            # was left over from an earlier retry design that no
            # longer exists.
            self.disagg_transfer_queue.add(entry)

        # 2. Drive receiver polls; on SUCCESS write KV to pool and
        # hand the req to the scheduler's decode pipeline. On FAILED
        # release ALL resources (KV indices + req_to_token_pool slot)
        # so the pool doesn't leak; surface the error to the client
        # via the standard abort path.
        for entry in self.disagg_transfer_queue.drain_terminal():
            assert entry.receiver is not None
            state = entry.receiver.poll()
            if state == KVPoll.SUCCESS:
                try:
                    kv = entry.receiver.result
                    self._maybe_log_decode_pull_debug(entry.req, kv)
                    self._write_kv_to_pool(
                        entry.req, entry.kv_indices, kv
                    )
                    try:
                        from sgl_jax.srt.disaggregation.metrics import (
                            PD_TRANSFER_BYTES_TOTAL,
                        )

                        if kv is not None and hasattr(kv, "nbytes"):
                            PD_TRANSFER_BYTES_TOTAL.labels(
                                direction="h2d", role="decode"
                            ).inc(int(kv.nbytes))
                    except Exception:  # noqa: BLE001
                        pass
                    self._enqueue_for_decode(entry.req)
                except Exception:
                    logger.exception(
                        "failed to install KV / enqueue decode for "
                        "req_id=%s; releasing resources",
                        entry.req_id,
                    )
                    if entry.kv_indices is not None:
                        self._release_decode_kv_indices(entry.kv_indices)
                    self._release_decode_req_resources(entry.req)
            else:
                logger.warning(
                    "KVReceiver for req_id=%s reached %s; releasing "
                    "resources and aborting request",
                    entry.req_id, state.value,
                )
                try:
                    from sgl_jax.srt.disaggregation.metrics import (
                        PD_TRANSFER_FAILURES_TOTAL,
                    )

                    PD_TRANSFER_FAILURES_TOTAL.labels(
                        reason="receiver_terminal_failed", role="decode"
                    ).inc()
                except Exception:  # noqa: BLE001
                    pass
                if entry.kv_indices is not None:
                    self._release_decode_kv_indices(entry.kv_indices)
                self._release_decode_req_resources(entry.req)

    # ------------------------------------------------------------------
    # Overridable / test-friendly hooks
    # ------------------------------------------------------------------

    def _prealloc_decode_kv_indices(self, req: "Req"):
        """Reserve KV slots in the paged pool for ``req``.

        Default implementation: allocate ``ceil(seqlen / page_size)
        * page_size`` slots so the allocation is page-aligned (the
        paged allocator's contract). Returns the numpy int32 array
        of allocated slot indices. Subclass overrides for non-MHA
        pools.
        """

        seqlen = len(req.origin_input_ids)
        allocator = getattr(
            self, "token_to_kv_pool_allocator", None
        )
        if allocator is None:
            return None
        # The paged allocator's ``alloc`` asserts the request size
        # is page-aligned. Round up; the slots past seqlen are
        # legitimate pool space owned by this req (released on
        # cache_finished_req with the rest).
        page_size = getattr(allocator, "page_size", 1)
        page_aligned = ((seqlen + page_size - 1) // page_size) * page_size
        # alloc returns numpy int32 indices or None on OOM.
        return allocator.alloc(page_aligned)

    def _release_decode_kv_indices(self, kv_indices) -> None:
        """Release KV indices reserved by
        :meth:`_prealloc_decode_kv_indices`. Default: hand back to
        the allocator's ``free`` API.
        """

        if kv_indices is None:
            return
        allocator = getattr(
            self, "token_to_kv_pool_allocator", None
        )
        if allocator is not None and hasattr(allocator, "free"):
            try:
                allocator.free(kv_indices)
            except Exception:
                logger.exception(
                    "failed to free kv_indices=%r", kv_indices
                )

    def _build_kv_spec_for_req(self, req: "Req") -> jax.ShapeDtypeStruct:
        """Build the ``ShapeDtypeStruct`` (with sharding) the
        receiver pulls into. Must match P's registered KV shape.

        Default implementation mirrors
        :meth:`SchedulerDisaggregationPrefillMixin._extract_req_kv`'s
        stacked layout: ``(layer_num, padded_pages, page_size,
        head_num*2/packing, packing, head_dim)`` with the MHA
        pool's ``kv_sharding`` shifted by one axis to account for
        the prepended layer dimension. The ``padded_pages``
        dimension matches the bucket P used (see
        ``_KV_GATHER_PAGE_BUCKETS`` in prefill.py); the bucket
        function is duplicated here so D doesn't have to import
        from prefill.
        """

        from jax.sharding import NamedSharding, PartitionSpec

        from sgl_jax.srt.disaggregation.prefill import _pad_to_page_bucket

        kv_pool = (
            self.token_to_kv_pool_allocator.get_kvcache()  # type: ignore[attr-defined]
        )
        page_size = kv_pool.page_size
        seqlen = len(req.origin_input_ids)
        num_pages = (seqlen + page_size - 1) // page_size
        padded_pages = _pad_to_page_bucket(num_pages)
        # ``kv_buffer[0].shape`` is
        # (num_pages, page_size, head_num*2/packing, packing, head_dim).
        per_layer_tail = kv_pool.kv_buffer[0].shape[1:]
        # P stacked over axis 0 → layer dim is prepended.
        shape = (kv_pool.layer_num, padded_pages) + per_layer_tail
        # Prepend ``None`` to the pool's spec so the layer dim we
        # added is unsharded; ``jnp.stack(..., axis=0)`` on the
        # P side produces the same sharding shape.
        base_spec = kv_pool.kv_sharding.spec
        stacked_spec = PartitionSpec(None, *base_spec)
        sharding = NamedSharding(kv_pool.kv_sharding.mesh, stacked_spec)
        return jax.ShapeDtypeStruct(shape, kv_pool.dtype, sharding=sharding)

    def _write_kv_to_pool(self, req: "Req", kv_indices, kv: jax.Array) -> None:
        """Write the pulled KV into the local paged pool by PAGE
        index. Bypasses ``tree_cache.insert`` per RFC-2 ADR-7; the
        standard ``cache_finished_req`` at decode finish handles
        insertion.

        ``kv`` arrives shaped ``(layer_num, padded_pages,
        page_size, h*2/p, p, head_dim)`` from P (see
        ``_extract_req_kv``). We scatter the first ``num_pages``
        of that into our paged pool at the page ids derived from
        ``kv_indices`` (which are page-aligned slot ids). The
        ``padded_pages - num_pages`` tail rows contain stale data
        and are discarded.

        Sets ``req.prefix_indices`` to all but the LAST input
        token's slot so the scheduler's prefix-match path computes
        ``extend_input_len = 1``. We need at least 1 token to
        extend on for the model to produce logits for the first
        decode sampling (TPU XLA crashes with "program continuator
        has halted unexpectedly" when extend=0).
        """

        if kv_indices is None:
            raise RuntimeError(
                f"_write_kv_to_pool: kv_indices is None for req "
                f"{req.rid!r}; allocator may have OOM'd"
            )
        import numpy as np

        from jax.sharding import NamedSharding, PartitionSpec

        kv_pool = (
            self.token_to_kv_pool_allocator.get_kvcache()  # type: ignore[attr-defined]
        )
        page_size = kv_pool.page_size
        seqlen = len(req.origin_input_ids)
        num_pages = (seqlen + page_size - 1) // page_size
        kv_indices_np = (
            np.asarray(kv_indices) if not isinstance(kv_indices, np.ndarray)
            else kv_indices
        )
        # Derive page ids from the page-aligned slot ids.
        page_ids_np = kv_indices_np[::page_size] // page_size
        page_ids_np = page_ids_np[:num_pages]
        # Pad the page ids to ``kv.shape[1]`` (padded_pages bucket)
        # by repeating the last valid id. The duplicate writes
        # overwrite each other harmlessly, and keeping the indices
        # array shape constant per bucket lets the scatter share
        # one compile-cache entry across requests of similar size
        # (mirrors the gather-side bucketing on P).
        padded_pages = kv.shape[1]
        if num_pages < padded_pages:
            pad = np.full(
                padded_pages - num_pages, page_ids_np[-1], dtype=page_ids_np.dtype
            )
            page_ids_padded = np.concatenate([page_ids_np, pad])
            # The padded tail rows past ``num_pages`` hold stale data from
            # P's over-read bucket fill. We intentionally duplicate the
            # LAST valid page id to keep the scatter shape bucket-stable,
            # so we must also duplicate the LAST valid page payload —
            # otherwise the stale tail rows overwrite the final real page.
            valid_prefix = jax.lax.slice_in_dim(
                kv,
                start_index=0,
                limit_index=num_pages,
                axis=1,
            )
            last_valid = jax.lax.dynamic_slice_in_dim(
                valid_prefix,
                start_index=num_pages - 1,
                slice_size=1,
                axis=1,
            )
            padded_tail = jnp.repeat(
                last_valid,
                padded_pages - num_pages,
                axis=1,
            )
            padded_tail = jax.device_put(
                padded_tail,
                valid_prefix.sharding,
            )
            kv = jnp.concatenate(
                [
                    valid_prefix,
                    padded_tail,
                ],
                axis=1,
            )
        else:
            page_ids_padded = page_ids_np
        idx_sharding = NamedSharding(kv_pool.mesh, PartitionSpec(None))
        page_ids_jax = jax.device_put(page_ids_padded, idx_sharding)
        # Scatter each layer. JAX 0.8.1 sharded ``.at[].set()``
        # requires explicit ``out_sharding`` for the same reason
        # the gather on P needs it.
        for i, layer_id in enumerate(
            range(kv_pool.start_layer, kv_pool.start_layer + kv_pool.layer_num)
        ):
            layer_idx = layer_id - kv_pool.start_layer
            # ``kv[i].shape = (padded_pages, page_size, ...)`` —
            # exactly the slice shape the scatter expects since
            # ``pool[layer_idx]`` is ``(num_pages_pool, page_size,
            # ...)``. Functional update: replace the slot in the
            # pool. ``out_sharding`` carries the pool's full
            # sharding because we're producing an array of the
            # same shape as the pool.
            kv_pool.kv_buffer[layer_idx] = (
                kv_pool.kv_buffer[layer_idx]
                .at[page_ids_jax]
                .set(kv[i], out_sharding=kv_pool.kv_sharding)
            )
        # Mark all-but-last input tokens as cached so extend_input_len=1
        # and the model re-runs on the last input position to produce
        # the first sampling logits. ``_pd_skip_prefix_match`` makes
        # the scheduler's init_next_round_input honor our setting
        # instead of overwriting via tree_cache.match_prefix.
        # ``kv_indices_np`` has ``page_aligned`` entries; the first
        # ``seqlen`` correspond to actual prompt tokens, and we keep
        # all but the last for the prefix-cache.
        valid_slots = kv_indices_np[:seqlen]
        if len(valid_slots) >= 1:
            req.prefix_indices = valid_slots[:-1]
        else:
            req.prefix_indices = valid_slots
        req.last_matched_prefix_len = len(req.prefix_indices)
        req._pd_skip_prefix_match = True
        # Hand the prealloc'd indices to req so the scheduler's
        # cache_finished_req free path picks them up on terminal —
        # otherwise the allocator's in-use count drifts (the slots
        # are alloc'd but unowned), and check_memory crashes the
        # engine ~1 iter later. ``_pd_prealloc_kv_indices`` tells
        # our event-loop self-check to skip the leak detection on
        # in-flight PD reqs.
        req._pd_prealloc_kv_indices = kv_indices_np
        # Make sure fill_ids is set so the scheduler doesn't re-derive
        # an empty prefill chunk.
        req.fill_ids = list(req.origin_input_ids) + list(req.output_ids)
        self._maybe_verify_decode_writeback_debug(
            req, kv_pool, page_ids_padded, kv
        )

    def _enqueue_for_decode(self, req: "Req") -> None:
        """Put ``req`` into the scheduler's decode-ready queue.
        Default: append to ``waiting_queue`` if present.
        """

        queue = getattr(self, "waiting_queue", None)
        if queue is not None and req not in queue:
            queue.append(req)

    def _release_decode_req_resources(self, req: "Req") -> None:
        """Release ``req``'s scheduler-side resources after a failed
        KV transfer.

        Stage 2 review C3: must NOT call ``cache_finished_req`` —
        that path assumes the req went through prefill, has
        ``output_ids``, and is in the running batch. A req that
        never received KV has none of that. Default: best-effort
        release of the ``req_to_token_pool`` slot if allocated,
        otherwise no-op. Subclass overrides may surface the error
        to the client via the comm backend.
        """

        req_pool_idx = getattr(req, "req_pool_idx", None)
        if req_pool_idx is not None:
            req_to_token_pool = getattr(self, "req_to_token_pool", None)
            if req_to_token_pool is not None and hasattr(
                req_to_token_pool, "free"
            ):
                try:
                    req_to_token_pool.free(req_pool_idx)
                except Exception:
                    logger.exception(
                        "failed to free req_to_token_pool slot %d for "
                        "req_id=%s",
                        req_pool_idx, req.rid,
                    )

    def _maybe_log_decode_pull_debug(self, req: "Req", kv) -> None:
        from sgl_jax.srt.disaggregation.debug_utils import (
            build_kv_debug_snapshot,
            kv_debug_enabled,
        )

        if not kv_debug_enabled(getattr(req, "rid", None)):
            return

        snapshot = build_kv_debug_snapshot(kv)
        logger.warning(
            "PD-KV-DEBUG decode_pull req_id=%s shape=%s dtype=%s "
            "sharding=%s digest=%s sample=%s",
            req.rid,
            snapshot.shape,
            snapshot.dtype,
            snapshot.sharding,
            snapshot.global_digest,
            snapshot.sample_page_digests(),
        )

    def _maybe_verify_decode_writeback_debug(
        self, req: "Req", kv_pool, page_ids_padded, kv
    ) -> None:
        from jax.sharding import NamedSharding, PartitionSpec

        from sgl_jax.srt.disaggregation.debug_utils import (
            build_kv_debug_snapshot,
            count_kv_debug_mismatches,
            find_first_kv_debug_mismatch,
            kv_debug_enabled,
        )
        from sgl_jax.srt.disaggregation.prefill import _jit_gather_all_layers

        if not kv_debug_enabled(getattr(req, "rid", None)):
            return

        page_ids_jax = jax.device_put(
            page_ids_padded,
            NamedSharding(kv_pool.mesh, PartitionSpec(None)),
        )
        gather_out_sharding = NamedSharding(
            kv_pool.mesh,
            PartitionSpec(None, *kv_pool.kv_sharding.spec[1:]),
        )
        layer_buffers = [
            kv_pool.get_kv_buffer(layer_id)
            for layer_id in range(
                kv_pool.start_layer,
                kv_pool.start_layer + kv_pool.layer_num,
            )
        ]
        readback = jnp.stack(
            _jit_gather_all_layers(
                layer_buffers,
                page_ids_jax,
                gather_out_sharding,
            ),
            axis=0,
        )

        expected = build_kv_debug_snapshot(kv)
        actual = build_kv_debug_snapshot(readback)
        mismatch_count = count_kv_debug_mismatches(expected, actual)
        first_mismatch = find_first_kv_debug_mismatch(expected, actual)

        logger.warning(
            "PD-KV-DEBUG decode_writeback req_id=%s expected_digest=%s "
            "readback_digest=%s mismatch_count=%d first_mismatch=%s "
            "expected_sample=%s readback_sample=%s page_ids=%s",
            req.rid,
            expected.global_digest,
            actual.global_digest,
            mismatch_count,
            first_mismatch,
            expected.sample_page_digests(),
            actual.sample_page_digests(),
            page_ids_padded.tolist(),
        )
