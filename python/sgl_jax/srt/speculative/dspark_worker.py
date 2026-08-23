from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from functools import partial

import jax
import numpy as np

from sgl_jax.srt.speculative.dflash_worker import DFlashWorker
from sgl_jax.srt.speculative.dspark_util import parse_dspark_draft_config

logger = logging.getLogger(__name__)


class DSparkWorker(DFlashWorker):
    """DSpark Markov draft with fixed or tuned compact ragged verification."""

    def __init__(self, server_args, target_worker):
        self._sts_capture_path = os.getenv("SGL_JAX_DSPARK_STS_CAPTURE_PATH")
        forced_token_bucket = os.getenv("SGL_JAX_DSPARK_FORCE_TOKEN_BUCKET_PER_DP")
        self._dspark_forced_token_bucket = (
            int(forced_token_bucket) if forced_token_bucket is not None else None
        )
        if self._dspark_forced_token_bucket is not None and self._dspark_forced_token_bucket <= 0:
            raise ValueError("SGL_JAX_DSPARK_FORCE_TOKEN_BUCKET_PER_DP must be positive.")
        self.dspark_tuned_config = None
        self._dspark_sts_temperatures = None
        self._dspark_seen_ragged_plans = set()
        self._dspark_capacity_relay_metrics = {
            "hit": 0,
            "stale_warmup": 0,
            "stale_generation": 0,
            "stale_not_ready": 0,
        }
        self._dspark_last_relay_stats = dict(self._dspark_capacity_relay_metrics)
        self._dspark_logged_first_relay_hit = False
        super().__init__(server_args, target_worker)
        self._dspark_confidence_relay_device = None
        self._dspark_confidence_relay_host = None
        if self.dspark_tuned_config is not None:
            from sgl_jax.srt.speculative.relay_buffer import (
                DSparkConfidenceRelayHost,
                create_dspark_confidence_relay_buffers,
            )

            self._dspark_confidence_relay_device = create_dspark_confidence_relay_buffers(
                self.mesh,
                self.req_to_token_pool,
                dp_size=self.server_args.dp_size,
                gamma=self.draft_width,
            )
            self._dspark_confidence_relay_host = DSparkConfidenceRelayHost(
                dp_size=self.server_args.dp_size,
                capacity=self.req_to_token_pool.req_to_token.shape[0],
                gamma=self.draft_width,
            )

    @staticmethod
    def _draft_model_class():
        from sgl_jax.srt.models.dspark import DSparkDraftModel

        return DSparkDraftModel

    @staticmethod
    def _parse_block_config(server_args):
        return parse_dspark_draft_config(
            server_args.speculative_draft_model_path,
            revision=server_args.speculative_draft_model_revision,
            trust_remote_code=server_args.trust_remote_code,
        )

    def _configure_widths(self, block_config) -> None:
        config = block_config
        if self.verify_width != config.verify_width:
            raise ValueError(
                "DSPARK internal verify width must be checkpoint block_size + 1: "
                f"got {self.verify_width}, expected {config.verify_width}."
            )
        self.draft_width = config.draft_width
        self.verify_width = config.verify_width
        self.block_size = self.verify_width
        self._proposal_hidden_start = 0
        if getattr(self.server_args, "enable_dspark_tuned_config", False):
            from sgl_jax.srt.speculative.dspark_tuned_config import (
                get_tuned_dspark_config,
                make_dspark_tuned_key,
            )
            from sgl_jax.srt.utils.jax_utils import get_device_name

            key = make_dspark_tuned_key(
                target_model=self.server_args.model_path,
                draft_model=self.server_args.speculative_draft_model_path,
                target_revision=self.server_args.revision,
                draft_revision=self.server_args.speculative_draft_model_revision,
                device_name=get_device_name(),
                device_count=self.mesh.size,
                dtype=self.server_args.dtype,
                quantization=self.server_args.quantization,
                tp_size=self.server_args.tp_size,
                dp_size=self.server_args.dp_size,
                gamma=config.gamma,
                page_size=self.server_args.page_size,
                attention_backend=self.server_args.attention_backend,
                overlap=not self.server_args.disable_overlap_schedule,
            )
            self.dspark_tuned_config = get_tuned_dspark_config(key)
            if self.dspark_tuned_config is None:
                logger.warning(
                    "DSPARK tuned-config lookup miss for key=%s; keeping fixed verify-all.", key
                )
            else:
                if self._sts_capture_path:
                    raise ValueError(
                        "DSPARK STS capture requires uncalibrated fixed verify-all execution; "
                        "omit --enable-dspark-tuned-config while collecting raw logits."
                    )
                self._dspark_sts_temperatures = self.dspark_tuned_config.sts_temperatures
                logger.info("Using DSPARK tuned config: %s", self.dspark_tuned_config.provenance)

    def verify(self, model_worker_batch, cur_allocate_lens=None, *, update_relay=False):
        plan = getattr(model_worker_batch, "_dflash_target_verify_plan", None)
        result = super().verify(
            model_worker_batch,
            cur_allocate_lens,
            update_relay=update_relay,
        )
        if self._sts_capture_path and plan is not None and plan.draft_confidence_logits is not None:
            confidence_logits, accept_lens, active_mask = jax.device_get(
                (plan.draft_confidence_logits, result.accept_lens, plan.active_mask)
            )
            confidence_logits = np.asarray(confidence_logits, dtype=np.float32)
            accepted_draft = np.asarray(accept_lens, dtype=np.int32) - 1
            active_mask = np.asarray(active_mask, dtype=np.bool_)
            with open(self._sts_capture_path, "a", encoding="utf-8") as capture:
                for row, accepted in zip(
                    confidence_logits[active_mask], accepted_draft[active_mask], strict=True
                ):
                    prefix_mask = np.arange(row.shape[0], dtype=np.int32) < int(accepted)
                    capture.write(
                        json.dumps(
                            {
                                "logits": row.tolist(),
                                "prefix_mask": prefix_mask.astype(np.int32).tolist(),
                                "accepted_draft": int(accepted),
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        return result

    def _build_target_verify_plan(
        self,
        model_worker_batch,
        draft_plan,
        draft_token,
        resolved_target_prefix_lens,
        resolved_positions,
        resolved_cache_loc,
        draft_confidence,
        draft_confidence_logits,
    ):
        plan = super()._build_target_verify_plan(
            model_worker_batch,
            draft_plan,
            draft_token,
            resolved_target_prefix_lens,
            resolved_positions,
            resolved_cache_loc,
            draft_confidence,
            draft_confidence_logits,
        )
        if self.dspark_tuned_config is None:
            return plan

        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.speculative.dspark_planner import select_dspark_verify_budget
        from sgl_jax.srt.speculative.dspark_tuned_config import select_dspark_sps_profile

        active = self._active_decode_slot_mask(model_worker_batch, draft_plan.bs)
        lagged_confidence = self._publish_and_gather_lagged_confidence(
            model_worker_batch,
            draft_confidence,
            active,
        )
        active_prefix_lens = draft_plan.target_prefix_lens[active]
        context_length = (
            int(active_prefix_lens.max()) + self.verify_width if active_prefix_lens.size else 0
        )
        profile = select_dspark_sps_profile(self.dspark_tuned_config, context_length)
        if profile is None:
            logger.warning(
                "DSPARK has no SPS profile for context_length=%d; keeping fixed verify-all.",
                context_length,
            )
            return plan

        dp_size = int(model_worker_batch.dp_size)
        per_dp_bs = draft_plan.bs // dp_size
        active_rows = active.reshape((dp_size, per_dp_bs))
        extra_budget_per_dp = np.zeros((dp_size,), dtype=np.int32)
        token_buckets = []
        for dp_rank in range(dp_size):
            num_requests = int(active_rows[dp_rank].sum())
            if num_requests == 0:
                continue
            rank_confidence = lagged_confidence.reshape((dp_size, per_dp_bs, -1))[dp_rank]
            lagged_survival = np.cumprod(rank_confidence[active_rows[dp_rank]], axis=-1)
            decision = select_dspark_verify_budget(
                profile,
                lagged_survival,
                forced_token_bucket=self._dspark_forced_token_bucket,
            )
            if decision is None:
                logger.warning(
                    "DSPARK SPS profile has no usable token bucket for rank=%d requests=%d; "
                    "keeping fixed verify-all.",
                    dp_rank,
                    num_requests,
                )
                return plan
            token_buckets.append(decision.token_bucket)
            extra_budget_per_dp[dp_rank] = decision.extra_budget

        if not token_buckets:
            return plan
        per_dp_token_bucket = max(token_buckets)
        log_key = (
            profile.context_bucket,
            per_dp_token_bucket,
            tuple(extra_budget_per_dp.tolist()),
        )
        if log_key not in self._dspark_seen_ragged_plans:
            logger.info(
                "DSPARK ragged verify plan: context_bucket=%d, token_bucket_per_dp=%d, "
                "active_requests_per_dp=%s, extra_budget_per_dp=%s, "
                "capacity_lag=2, relay_stats=%s",
                profile.context_bucket,
                per_dp_token_bucket,
                active_rows.sum(axis=1).tolist(),
                extra_budget_per_dp.tolist(),
                self._dspark_last_relay_stats,
            )
            self._dspark_seen_ragged_plans.add(log_key)
        data_sharding = NamedSharding(self.mesh, P("data"))
        return replace(
            plan,
            dspark_extra_budget_per_dp=jax.device_put(
                extra_budget_per_dp,
                data_sharding,
            ),
            dspark_per_dp_token_bucket=per_dp_token_bucket,
        )

    def _publish_and_gather_lagged_confidence(
        self,
        model_worker_batch,
        draft_confidence,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        """Publish C[t] asynchronously and gather only materialized C[t-2]."""
        total_bs = int(active_mask.size)
        fallback = np.ones((total_bs, self.draft_width), dtype=np.float32)
        relay_host = self._dspark_confidence_relay_host
        relay_device = self._dspark_confidence_relay_device
        slot_generations = model_worker_batch.req_pool_slot_generations
        decode_rounds = model_worker_batch.dspark_decode_rounds
        if (
            relay_host is None
            or relay_device is None
            or slot_generations is None
            or decode_rounds is None
        ):
            self._dspark_last_relay_stats = {
                "hit": 0,
                "stale_warmup": 0,
                "stale_generation": 0,
                "stale_not_ready": int(active_mask.sum()),
            }
            return fallback

        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.speculative.relay_buffer import (
            update_dspark_confidence_relay_buffers,
        )

        req_pool_indices = np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32)
        slot_generations = np.asarray(slot_generations, dtype=np.int32)
        decode_rounds = np.asarray(decode_rounds, dtype=np.int32)
        safe_indices = np.where(active_mask, req_pool_indices, 0).astype(np.int32)
        data_sharding = NamedSharding(self.mesh, P("data"))

        if not hasattr(self, "_jit_update_dspark_confidence_relay"):
            # Do not donate the ring: the background publisher retains this
            # immutable snapshot until its asynchronous D2H copy materializes.
            @partial(jax.jit, static_argnames=["dp_size"])
            def update(buffers, indices, generations, rounds, valid, confidence, *, dp_size):
                return update_dspark_confidence_relay_buffers(
                    buffers,
                    indices,
                    generations,
                    rounds,
                    valid,
                    confidence,
                    dp_size=dp_size,
                )

            self._jit_update_dspark_confidence_relay = update

        with jax.set_mesh(self.mesh):
            self._dspark_confidence_relay_device = self._jit_update_dspark_confidence_relay(
                relay_device,
                jax.device_put(safe_indices, data_sharding),
                jax.device_put(slot_generations, data_sharding),
                jax.device_put(decode_rounds, data_sharding),
                jax.device_put(active_mask, data_sharding),
                draft_confidence,
                dp_size=int(model_worker_batch.dp_size),
            )
        relay_host.publish(self._dspark_confidence_relay_device)
        lagged_confidence, stats = relay_host.gather_lagged_confidence(
            safe_indices,
            slot_generations,
            decode_rounds,
            active_mask,
        )
        self._dspark_last_relay_stats = stats
        for name, value in stats.items():
            self._dspark_capacity_relay_metrics[name] += value
        if stats["hit"] and not self._dspark_logged_first_relay_hit:
            logger.info(
                "DSPARK capacity relay first lag-2 hit: step_stats=%s, cumulative=%s",
                stats,
                self._dspark_capacity_relay_metrics,
            )
            self._dspark_logged_first_relay_hit = True
        return lagged_confidence
