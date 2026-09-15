"""Full-context session reuse for V4's request-owned chunk cache.

An idle session retains the request slot, not just history token indices. The
slot also addresses SWA/compressor/indexer continuation state. No partial
prefix rollback is possible without restoring that state, so mismatches are
deliberately cold misses.
"""

import numpy as np

from sgl_jax.srt.mem_cache.chunk_cache import DeepseekV4ChunkCache
from sgl_jax.srt.mem_cache.session_store import SessionStore


class DeepseekV4SessionCache(DeepseekV4ChunkCache):
    _TRANSFER_FIELDS = (
        "req_pool_idx",
        "dp_rank",
        "kv_committed_len",
        "kv_allocated_len",
        "kv_committed_freed",
        "kv_overallocated_freed",
        "swa_evicted_seqlen",
    )

    def __init__(self, *args, session_timeout=300.0, max_sessions=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.sessions = SessionStore(super().release_req, session_timeout, max_sessions)

    def attach(self, req, params):
        if not isinstance(params, dict) or set(params) != {"id"}:
            raise ValueError('Use session_params={"id": "..."} with full input each turn')
        # Logprob/hidden-state requests can require recomputing the prompt.
        # LoRA and multimodal continuation need their own cache identity rules.
        # Req normalizes an absent adapter to the base-model sentinel "0".
        if (
            req.lora_id not in (None, "0")
            or req.return_logprob
            or req.return_hidden_states
            or req.mm_inputs
        ):
            raise ValueError("V4 sessions currently support text-only, base-model generation")
        self.sessions.acquire(params["id"], req)
        req.session_id = params["id"]

    def match_prefix(self, params):
        req = params.req
        session_id = getattr(req, "session_id", None)
        if session_id and req.req_pool_idx is None:
            session = self.sessions.sessions[session_id]
            owner = session.owner
            if owner is not None:
                length = owner.kv_committed_len
                # Leave at least one query for logits. Never trim a retained
                # extent: V4's compression state cannot be rolled back.
                can_reuse = (
                    session.active is req
                    and len(req.origin_input_ids) >= len(session.tokens)
                    and length < len(req.origin_input_ids)
                    and tuple(req.origin_input_ids[: len(session.tokens)]) == session.tokens
                    and req.extra_key == owner.extra_key
                )
                if can_reuse:
                    for field in self._TRANSFER_FIELDS:
                        setattr(req, field, getattr(owner, field))
                    req.session_restored = True
                    owner.req_pool_idx = None
                    session.owner = None
                    session.tokens = ()
                else:
                    self.sessions.discard_cache(session)
        return super().match_prefix(params)

    def release_req(self, req):
        session_id = getattr(req, "session_id", None)
        if not session_id:
            return super().release_req(req)
        session = self.sessions.sessions.get(session_id)
        if session is not None and session.owner is req:
            return  # Idempotent completion must not free a retained owner.
        if not req.finished():
            # Retraction frees physical state but keeps this request's lease.
            return super().release_req(req)
        tokens = req.origin_input_ids + req.output_ids
        reason = req.finished_reason.to_json()["type"]
        retain = (
            reason != "abort"
            and req.req_pool_idx is not None
            and 0 < req.kv_committed_len == req.kv_allocated_len <= len(tokens)
        )
        if retain:
            # Snapshot once; idle checks must not rescan every 128K mapping.
            req.session_reserved_sizes = self._reserved_sizes(req)
        # Overlap may already have launched the final sampled token's forward.
        # Keep the entire ordered extent, including that token, not len(outputs)-1.
        # Existing result resolution waits for launch_done before this transfer.
        if not self.sessions.finish(
            session_id, req, retain=retain, tokens=tokens[: req.kv_committed_len]
        ):
            super().release_req(req)

    def cancel(self, req):
        session_id = getattr(req, "session_id", None)
        if session_id:
            self.sessions.finish(session_id, req)

    def maintenance(self):
        # Validation/grammar cancellation may terminate a leased request before
        # any slot is allocated, bypassing release_kv_cache's early return.
        for session in list(self.sessions.sessions.values()):
            req = session.active
            if req is not None and req.finished() and req.req_pool_idx is None:
                self.cancel(req)
        self.sessions.reap()

    def reserve_headroom(self, num_tokens):
        """Evict idle owners before admission; never evict an active lease.

        V4's allocator intentionally bypasses radix-tree eviction. Releasing
        complete idle sessions here also returns request slots and continuation
        state ownership, which token-only eviction would miss.
        """
        while (
            self.req_to_token_pool.available_size() == 0
            or self.token_to_kv_pool_allocator.available_size() < num_tokens
        ):
            if not self.sessions.evict_one():
                break

    def reset(self):
        self.sessions.reset()

    def held_sizes(self, dp_rank=0):
        """Reserved pages, not committed tokens, for the idle leak checker."""
        full = swa = slots = 0
        for session in self.sessions.sessions.values():
            owner = session.owner
            if owner is None or (owner.dp_rank or 0) != dp_rank:
                continue
            owner_full, owner_swa = owner.session_reserved_sizes
            full += owner_full
            swa += owner_swa
            slots += 1
        return full, swa, slots

    def _reserved_sizes(self, req):
        indices = self.req_to_token_pool.read(req.req_pool_idx, req.kv_allocated_len)
        indices = indices[indices != 0]
        mapping = self.token_to_kv_pool_allocator.full_to_swa_index_mapping
        mapped = mapping[indices]
        return (
            len(np.unique(indices // self.page_size)) * self.page_size,
            len(np.unique(mapped[mapped != 0] // self.page_size)) * self.page_size,
        )
