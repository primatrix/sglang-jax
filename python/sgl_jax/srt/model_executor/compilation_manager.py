from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from sgl_jax.srt.utils.common_utils import (
    PRECOMPILE_DEFAULT_BS_PADDINGS,
    PRECOMPILE_DEFAULT_TOKEN_PADDINGS,
)

if TYPE_CHECKING:
    from sgl_jax.srt.model_executor.model_runner import ModelRunner
    from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class CompilationManager:
    """Owns bucket computation, dummy batch construction, and pre-compilation."""

    def __init__(
        self,
        server_args: ServerArgs,
        max_padded_batch_size: int,
        max_padded_num_tokens: int,
        dp_size: int,
        tp_size: int,
        page_size: int,
        max_req_len: int,
        vocab_size: int,
        multimodal: bool = False,
        has_recurrent_state: bool = False,
        moe_backend: str | None = None,
    ):
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.page_size = page_size
        self.max_req_len = max_req_len
        self.max_padded_batch_size = max_padded_batch_size
        self.max_padded_num_tokens = max_padded_num_tokens
        self.vocab_size = vocab_size
        self.multimodal = multimodal
        self.has_recurrent_state = has_recurrent_state
        # Callers pass the *effective* backend (ModelConfig.moe_backend), which
        # resolves architectures that hard-code FusedEPMoE (e.g. Qwen3.5) to
        # "fused" so the bs-bucket filter below applies. Fall back to the raw
        # server_args string for callers that don't have a ModelConfig yet.
        self.moe_backend = moe_backend if moe_backend is not None else server_args.moe_backend
        self.enable_static_lora = server_args.enable_static_lora
        self.attention_backend = server_args.attention_backend

        self.token_buckets = self._compute_token_buckets(server_args.precompile_token_paddings)
        self.bs_buckets = self._compute_bs_buckets(server_args.precompile_bs_paddings)
        self.cache_loc_buckets = self._compute_cache_loc_buckets()
        self.dsa_context_buckets = self._compute_dsa_context_buckets(
            server_args.precompile_dsa_context_paddings
        )

        self._compiled_variants: set[tuple] = set()

    def _compute_token_buckets(self, user_paddings: list[int] | None) -> list[int]:
        dp_size = self.dp_size
        if user_paddings is None:
            user_paddings = [item * dp_size for item in PRECOMPILE_DEFAULT_TOKEN_PADDINGS]

        buckets = []
        for item in user_paddings:
            if item % dp_size != 0:
                item = (item // dp_size) * dp_size
            if (
                item >= self.max_padded_batch_size
                and item <= self.max_padded_num_tokens
                and item >= dp_size
            ):
                buckets.append(item)

        buckets.sort()
        if len(buckets) == 0 or buckets[-1] < self.max_padded_num_tokens:
            buckets.append(self.max_padded_num_tokens)

        return buckets

    def _compute_bs_buckets(self, user_paddings: list[int] | None) -> list[int]:
        bs_list = user_paddings if user_paddings is not None else PRECOMPILE_DEFAULT_BS_PADDINGS
        buckets = []
        for bs in bs_list:
            if (
                bs <= self.max_padded_batch_size
                and (self.moe_backend not in ("fused", "fused_v2") or bs >= self.tp_size * 2)
                and bs >= self.dp_size
            ):
                buckets.append(bs)
        buckets.sort()
        if len(buckets) == 0 or buckets[-1] < self.max_padded_batch_size:
            buckets.append(self.max_padded_batch_size)
        return buckets

    def _compute_cache_loc_buckets(self) -> list[int]:
        pages_per_req = (self.max_req_len + self.page_size - 1) // self.page_size * self.page_size
        return [bs * pages_per_req for bs in self.bs_buckets]

    def _compute_dsa_context_buckets(self, user_paddings: list[int] | None) -> list[int]:
        if user_paddings is None:
            return []
        if self.attention_backend != "dsa":
            logger.warning(
                "Ignoring DSA context precompile paddings for attention backend %s",
                self.attention_backend,
            )
            return []
        from sgl_jax.srt.layers.attention.dsa_utils import normalize_dsa_context_buckets

        return list(
            normalize_dsa_context_buckets(
                user_paddings,
                page_size=self.page_size,
                max_context_len=self.max_req_len,
            )
        )

    # ---- Pre-compilation ----

    def precompile_all(
        self,
        forward_fn: Callable,
        model_runner: ModelRunner,
        mesh,
        prepare_lora_fn: Callable | None = None,
        future_token_ids_map=None,
    ):
        self._precompile_extend(
            forward_fn, model_runner, mesh, prepare_lora_fn, future_token_ids_map
        )
        self._precompile_decode(
            forward_fn, model_runner, mesh, prepare_lora_fn, future_token_ids_map
        )

    def _precompile_extend(
        self,
        forward_fn: Callable,
        model_runner: ModelRunner,
        mesh,
        prepare_lora_fn: Callable | None,
        future_token_ids_map,
    ):
        from sgl_jax.srt.managers.schedule_batch import ForwardMode
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
        from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata

        start_time = time.perf_counter()
        bs = self.max_padded_batch_size
        logger.info(
            "[EXTEND] Begin to precompile bs_paddings=%s token_paddings=%s "
            "dsa_context_paddings=%s",
            [bs],
            self.token_buckets,
            self.dsa_context_buckets or None,
        )

        context_buckets = self.dsa_context_buckets or [None]
        pairs = list(itertools.product([bs], self.token_buckets, context_buckets))
        with tqdm(pairs, desc="[EXTEND] PRECOMPILE", leave=False) as pbar:
            for pair in pbar:
                bs_val, num_tokens, context_bucket = pair
                pbar.set_postfix(bs=bs_val, tokens=num_tokens, context=context_bucket)
                if bs_val > num_tokens:
                    logger.warning("bs=%s > num_tokens=%s, skip this pair", bs_val, num_tokens)
                    continue
                batch = self._make_dummy_batch(
                    bs_val,
                    num_tokens,
                    ForwardMode.EXTEND,
                    self.cache_loc_buckets[-1],
                    dp_size=self.dp_size,
                    per_dp_bs_size=bs_val // self.dp_size,
                    dsa_context_len=(
                        min(context_bucket, self.max_req_len)
                        if context_bucket is not None
                        else None
                    ),
                )
                if prepare_lora_fn is not None:
                    prepare_lora_fn(batch)
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    batch, 0, mesh, self.vocab_size
                )
                batch.forward_batch = ForwardBatch.init_new(batch, model_runner)
                if future_token_ids_map is not None:
                    from sgl_jax.srt.managers.utils import resolve_future_token_ids

                    batch.forward_batch.input_ids = resolve_future_token_ids(
                        batch.forward_batch.input_ids, future_token_ids_map, mesh
                    )
                forward_fn(
                    batch,
                    launch_done=None,
                    skip_sample=False,
                    sampling_metadata=sampling_metadata,
                )
                self._compiled_variants.add(
                    (ForwardMode.EXTEND, num_tokens, bs_val, context_bucket, False)
                )

        end_time = time.perf_counter()
        logger.info("[EXTEND] Precompile finished in %.0f secs", end_time - start_time)

    def _precompile_decode(
        self,
        forward_fn: Callable,
        model_runner: ModelRunner,
        mesh,
        prepare_lora_fn: Callable | None,
        future_token_ids_map,
    ):
        from sgl_jax.srt.managers.schedule_batch import ForwardMode
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
        from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata

        start_time = time.perf_counter()
        logger.info(
            "[DECODE] Begin to precompile bs_paddings=%s dsa_context_paddings=%s",
            self.bs_buckets,
            self.dsa_context_buckets or None,
        )

        pairs = list(
            itertools.product(
                enumerate(self.bs_buckets),
                self.dsa_context_buckets or [None],
            )
        )
        with tqdm(
            pairs,
            desc="[DECODE] PRECOMPILE",
            leave=False,
            total=len(pairs),
        ) as pbar:
            for (i, bs_val), context_bucket in pbar:
                pbar.set_postfix(bs=bs_val, context=context_bucket)
                aligned_cache_loc_size = self.cache_loc_buckets[i]
                batch = self._make_dummy_batch(
                    bs_val,
                    bs_val,
                    ForwardMode.DECODE,
                    aligned_cache_loc_size,
                    dp_size=self.dp_size,
                    per_dp_bs_size=bs_val // self.dp_size,
                    dsa_context_len=(
                        min(context_bucket, self.max_req_len)
                        if context_bucket is not None
                        else None
                    ),
                )
                if prepare_lora_fn is not None:
                    prepare_lora_fn(batch)
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    batch, 0, mesh, self.vocab_size
                )
                batch.forward_batch = ForwardBatch.init_new(batch, model_runner)
                if future_token_ids_map is not None:
                    from sgl_jax.srt.managers.utils import (
                        resolve_future_token_ids,
                        set_future_token_ids,
                    )

                    batch.forward_batch.input_ids = resolve_future_token_ids(
                        batch.forward_batch.input_ids, future_token_ids_map, mesh
                    )
                result = forward_fn(
                    batch,
                    launch_done=None,
                    skip_sample=False,
                    sampling_metadata=sampling_metadata,
                )
                if future_token_ids_map is not None:
                    _, next_token_ids, _ = result
                    set_future_token_ids(future_token_ids_map, 0, next_token_ids, mesh)
                self._compiled_variants.add(
                    (ForwardMode.DECODE, bs_val, bs_val, context_bucket, False)
                )

        end_time = time.perf_counter()
        logger.info("[DECODE] Precompile finished in %.0f secs", end_time - start_time)

    # ---- Dummy batch construction ----

    def _make_dummy_batch(
        self,
        bs: int,
        num_tokens: int,
        mode,
        max_cache_loc_size: int,
        speculative_algorithm=None,
        dp_size: int = 1,
        per_dp_bs_size: int = 0,
        dsa_context_len: int | None = None,
    ):
        import jax.numpy as jnp

        from sgl_jax.srt.managers.schedule_batch import (
            ForwardMode,
            ModelWorkerBatch,
            ModelWorkerSamplingInfo,
        )
        from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        # Runtime ScheduleBatch.spec_algorithm is always SpeculativeAlgorithm
        # enum (.from_string(None) -> .NONE). Default to .NONE so the dummy
        # batch's pytree aux matches and precompile shares the cache key with
        # the no-spec runtime path.
        if speculative_algorithm is None:
            spec_algorithm_value = SpeculativeAlgorithm.NONE
        else:
            spec_algorithm_value = speculative_algorithm

        if dsa_context_len is not None:
            if dp_size != 1:
                raise ValueError("DSA context precompile currently supports dp_size=1 only")
            if dsa_context_len <= 0:
                raise ValueError(f"dsa_context_len must be positive, got {dsa_context_len}")
            active_tokens = 1 if mode == ForwardMode.DECODE else min(num_tokens, dsa_context_len)
            active_bs = 1
            prefix_len = dsa_context_len - active_tokens
            valid_input_ids = np.ones(active_tokens, dtype=jnp.int32)
            valid_out_cache_loc = np.arange(
                prefix_len + 1,
                dsa_context_len + 1,
                dtype=jnp.int32,
            )
            valid_positions = np.arange(prefix_len, dsa_context_len, dtype=jnp.int32)
            seq_lens = np.concat(
                [
                    np.array([dsa_context_len], dtype=np.int32),
                    np.zeros(bs - active_bs, dtype=np.int32),
                ]
            )
            valid_cache_loc = np.arange(1, dsa_context_len + 1, dtype=jnp.int32)
        else:
            active_tokens = bs
            active_bs = bs
            prefix_len = 0
            valid_input_ids = np.ones(bs, dtype=jnp.int32)
            valid_out_cache_loc = np.arange(1, bs + 1, dtype=jnp.int32)
            valid_positions = np.zeros(bs, dtype=jnp.int32)
            seq_lens = np.ones(bs, dtype=np.int32)
            valid_cache_loc = np.arange(bs, dtype=jnp.int32)

        invalid_input_ids = np.zeros(num_tokens - active_tokens, dtype=jnp.int32)
        invalid_out_cache_loc = np.full(num_tokens - active_tokens, -1, dtype=jnp.int32)
        invalid_positions = np.zeros(num_tokens - active_tokens, dtype=jnp.int32)
        invalid_cache_loc_size = max_cache_loc_size - len(valid_cache_loc)
        if invalid_cache_loc_size < 0:
            raise ValueError(f"padding cache_loc_size {invalid_cache_loc_size} < 0!")

        invalid_cache_loc = np.zeros(invalid_cache_loc_size, dtype=jnp.int32)
        lora_ids = ["0"] * bs

        if mode == ForwardMode.EXTEND and dsa_context_len is not None:
            extend_seq_lens = np.concat(
                [
                    np.array([active_tokens], dtype=np.int32),
                    np.zeros(bs - active_bs, dtype=np.int32),
                ]
            )
            extend_prefix_lens = np.concat(
                [
                    np.array([prefix_len], dtype=np.int32),
                    np.zeros(bs - active_bs, dtype=np.int32),
                ]
            )
            logits_indices = np.concat(
                [
                    np.array([active_tokens - 1], dtype=np.int32),
                    np.zeros(bs - active_bs, dtype=np.int32),
                ]
            )
        elif mode == ForwardMode.EXTEND:
            extend_seq_lens = np.ones(bs, dtype=np.int32)
            extend_prefix_lens = np.zeros(bs, dtype=np.int32)
            logits_indices = np.zeros(bs, dtype=np.int32)
        else:
            extend_seq_lens = None
            extend_prefix_lens = None
            logits_indices = None

        if speculative_algorithm is None:
            sampling_info = ModelWorkerSamplingInfo.generate_for_precompile(bs, self.vocab_size)
            return_output_logprob_only = True
        else:
            sampling_info = ModelWorkerSamplingInfo.generate_for_precompile_all_greedy(
                bs, self.vocab_size
            )
            sampling_info.vocab_mask = None
            return_output_logprob_only = False

        return ModelWorkerBatch(
            bid=1,
            forward_mode=mode,
            input_ids=np.concat([valid_input_ids, invalid_input_ids], axis=0),
            real_input_ids_len=active_tokens,
            real_bs=active_bs,
            req_pool_indices=np.arange(bs, dtype=np.int32),
            seq_lens=seq_lens,
            out_cache_loc=np.concat([valid_out_cache_loc, invalid_out_cache_loc], axis=0),
            return_logprob=False,
            return_output_logprob_only=return_output_logprob_only,
            sampling_info=sampling_info,
            extend_input_logprob_token_ids=None,
            positions=np.concat([valid_positions, invalid_positions], axis=0),
            cache_loc=np.concat([valid_cache_loc, invalid_cache_loc], axis=0),
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_logprob_start_lens=None,
            logits_indices=logits_indices,
            input_logprob_indices=None,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL if self.multimodal else CaptureHiddenMode.NULL
            ),
            spec_algorithm=spec_algorithm_value,
            lora_ids=lora_ids,
            dp_size=dp_size,
            per_dp_bs_size=per_dp_bs_size,
            real_bs_per_dp=(
                [active_bs] if dsa_context_len is not None else [per_dp_bs_size] * dp_size
            ),
            logits_indices_selector=np.arange(active_bs, dtype=np.int32),
            # Hybrid recurrent backends (e.g. KDA) require these per-batch
            # arrays even at precompile time; slot 0 is RecurrentStatePool's
            # per-rank dummy slot, safe to point at. Leave None otherwise so
            # non-recurrent backends are unaffected.
            recurrent_indices=(np.zeros(bs, dtype=np.int32) if self.has_recurrent_state else None),
            has_initial_state=(np.zeros(bs, dtype=np.bool_) if self.has_recurrent_state else None),
        )

    # ---- Lazy compilation tracking ----

    def register_variant_if_new(self, variant_key: tuple) -> bool:
        """Register a compilation variant and return True if it was not seen before.

        Used to detect first-time compilation of a (mode, num_tokens, bs, logprob)
        shape tuple so the caller can log or act on cold-compile events.
        TODO: add runtime consumer that warns on cache misses (issue #609).
        """
        if variant_key in self._compiled_variants:
            return False
        self._compiled_variants.add(variant_key)
        return True
