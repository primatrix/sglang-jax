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
    resolve_vision_patch_buckets,
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
        precompile_in_model_vision: bool = False,
        capture_hidden_states: bool = False,
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
        self.precompile_in_model_vision = precompile_in_model_vision
        self.capture_hidden_states = capture_hidden_states
        self.has_recurrent_state = has_recurrent_state
        # Callers pass the *effective* backend (ModelConfig.moe_backend), which
        # resolves architectures that hard-code FusedEPMoE (e.g. Qwen3.5) to
        # "fused" so the bs-bucket filter below applies. Fall back to the raw
        # server_args string for callers that don't have a ModelConfig yet.
        self.moe_backend = moe_backend if moe_backend is not None else server_args.moe_backend
        self.enable_static_lora = server_args.enable_static_lora

        self.token_buckets = self._compute_token_buckets(server_args.precompile_token_paddings)
        self.bs_buckets = self._compute_bs_buckets(server_args.precompile_bs_paddings)
        self.cache_loc_buckets = self._compute_cache_loc_buckets()
        self.vision_patch_buckets = resolve_vision_patch_buckets(
            getattr(server_args, "precompile_vision_patch_paddings", None)
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

    def _extend_variant_names(self, model_runner: ModelRunner) -> tuple[str, ...]:
        variants = ["text"]
        if self.precompile_in_model_vision and getattr(
            model_runner.model, "materialize_input_embeddings", False
        ):
            variants.append("multimodal")
        return tuple(variants)

    @staticmethod
    def _populate_dummy_multimodal_inputs(batch, model_runner: ModelRunner) -> None:
        """Populate the array leaves produced by the runtime vision merge.

        The values are irrelevant for compilation. Shapes, dtypes, and shardings
        are established by ``ForwardBatch.init_new`` to match real VLM EXTEND
        batches.
        """
        hidden_size = model_runner.model_config.hidden_size
        num_tokens = len(batch.input_ids)
        dtype = np.dtype(model_runner.model_config.dtype)
        batch.input_embedding = np.zeros((num_tokens, hidden_size), dtype=dtype)

        deepstack_layers = getattr(model_runner.model, "deepstack_visual_layers", 0)
        if isinstance(deepstack_layers, int) and deepstack_layers > 0:
            batch.deepstack_visual_embedding = np.zeros(
                (deepstack_layers, num_tokens, hidden_size),
                dtype=dtype,
            )
            # Real Qwen3-VL requests carry True. Keeping the dummy identical also
            # exercises the DeepStack addition branch while adding only zeros.
            batch.apply_for_deepstack = True

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
        if self.precompile_in_model_vision:
            self._precompile_vision(model_runner, mesh)
        self._precompile_decode(
            forward_fn, model_runner, mesh, prepare_lora_fn, future_token_ids_map
        )

    def _precompile_vision(self, model_runner: ModelRunner, mesh):
        """Warm ViT encode and token-shaped merge executables across buckets.

        Runs the exact runtime embed path (``general_mm_embed_routine`` ->
        ``embed_mm_inputs`` -> encode + merge) with a zero-filled, all-masked
        dummy plan (no token rows are touched).

        Encoder output length is derived from the patch bucket and routing is
        padded to the token bucket. Warm every patch bucket once, then every
        reachable output length against every token bucket.
        """
        from types import SimpleNamespace

        import jax
        from jax.sharding import NamedSharding, PartitionSpec

        from sgl_jax.srt.model_executor.forward_batch_info import _device_put_embed_plan
        from sgl_jax.srt.multimodal.in_model.host_orchestration import (
            general_mm_embed_routine,
        )
        from sgl_jax.srt.multimodal.in_model.registry import (
            resolve_encoder_plan_builder,
        )

        builder = resolve_encoder_plan_builder(
            model_runner.model_config,
            input_buckets=self.vision_patch_buckets,
        )
        if builder is None:
            logger.info("[VISION] No in-model plan builder; skipping vision precompile.")
            return
        if not self.vision_patch_buckets:
            logger.info("[VISION] No vision buckets configured; skipping vision precompile.")
            return

        multimodal_model = model_runner.model
        language_backbone = multimodal_model.model
        # Match the runtime encode lane count: DP-Encoder fans over the tensor
        # devices, TP-Encoder uses a single collaborative lane per DP rank.
        from sgl_jax.srt.multimodal.layers.vision_sharding import encode_lane_count

        encode_lanes = encode_lane_count(mesh, getattr(multimodal_model, "encoder_tp", False))

        # Token buckets the runtime merge will actually see (dp-aligned).
        token_buckets = self.token_buckets or [max(self.dp_size, 1)]
        min_token = token_buckets[0]

        def warm(patch_bucket, num_tokens) -> bool:
            input_ids = jax.device_put(
                np.zeros((num_tokens,), dtype=np.int32),
                NamedSharding(mesh, PartitionSpec("data")),
            )
            plan = builder.dummy_plan(
                self.dp_size,
                encode_lanes,
                patch_bucket,
                num_tokens // self.dp_size,
            )
            _device_put_embed_plan(plan, mesh)
            forward_batch = SimpleNamespace(input_embedding=None)
            try:
                general_mm_embed_routine(
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    language_model=language_backbone,
                    multimodal_model=multimodal_model,
                    mm_embed_plan=plan,
                )
                jax.block_until_ready(forward_batch.input_embedding)
                return True
            except Exception as exc:  # pragma: no cover - best-effort warmup
                logger.warning(
                    "[VISION] Skipping warmup (patch=%s, tokens=%s): %s",
                    patch_bucket,
                    num_tokens,
                    exc,
                )
                return False

        # Warm every encoder input shape, then each output length and token shape.
        input_bucket_by_output_length: dict[int, int] = {}
        for patch_bucket in self.vision_patch_buckets:
            output_length = builder.get_num_output_tokens(patch_bucket)
            input_bucket_by_output_length.setdefault(
                output_length,
                patch_bucket,
            )

        combos = [
            ("encoder", patch_bucket, min_token) for patch_bucket in self.vision_patch_buckets
        ]
        combos += [
            ("merge", patch_bucket, num_tokens)
            for patch_bucket in input_bucket_by_output_length.values()
            for num_tokens in token_buckets[1:]
        ]

        start_time = time.perf_counter()
        phase_times = {"encoder": [], "merge": []}
        logger.info("[VISION] Begin to precompile %d model-shape combos", len(combos))
        warmed = 0
        with tqdm(combos, desc="[VISION] PRECOMPILE", leave=False) as pbar:
            for phase, patch_bucket, num_tokens in pbar:
                output_length = builder.get_num_output_tokens(patch_bucket)
                pbar.set_postfix(
                    phase=phase,
                    patch=patch_bucket,
                    output=output_length,
                    tokens=num_tokens,
                )
                combo_start = time.perf_counter()
                success = warm(patch_bucket, num_tokens)
                elapsed = time.perf_counter() - combo_start
                phase_times[phase].append(elapsed)
                logger.info(
                    "[VISION] %s warmup patch=%d output=%d tokens=%d %s in %.2fs",
                    phase,
                    patch_bucket,
                    output_length,
                    num_tokens,
                    "finished" if success else "failed",
                    elapsed,
                )
                if success:
                    self._compiled_variants.add(("VISION", patch_bucket, num_tokens))
                    warmed += 1
        total_time = time.perf_counter() - start_time
        logger.info(
            "[VISION] Precompile finished: warmed %d/%d combos in %.2fs "
            "(encoder sweep %.2fs, merge sweep %.2fs)",
            warmed,
            len(combos),
            total_time,
            sum(phase_times["encoder"]),
            sum(phase_times["merge"]),
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
        variant_names = self._extend_variant_names(model_runner)
        logger.info(
            "[EXTEND] Begin to precompile variants=%s bs_paddings=%s token_paddings=%s",
            variant_names,
            [bs],
            self.token_buckets,
        )

        pairs = list(itertools.product(variant_names, [bs], self.token_buckets))
        with tqdm(pairs, desc="[EXTEND] PRECOMPILE", leave=False) as pbar:
            for pair in pbar:
                variant_name, bs_val, num_tokens = pair
                pbar.set_postfix(variant=variant_name, bs=bs_val, tokens=num_tokens)
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
                )
                if variant_name == "multimodal":
                    self._populate_dummy_multimodal_inputs(batch, model_runner)
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
                    skip_sample=variant_name == "multimodal",
                    sampling_metadata=sampling_metadata,
                )
                if variant_name == "text":
                    variant_key = (ForwardMode.EXTEND, num_tokens, bs_val, False)
                else:
                    variant_key = ("VLM_EXTEND", num_tokens, bs_val)
                self._compiled_variants.add(variant_key)

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
            "[DECODE] Begin to precompile bs_paddings=%s",
            self.bs_buckets,
        )

        with tqdm(
            enumerate(self.bs_buckets),
            desc="[DECODE] PRECOMPILE",
            leave=False,
            total=len(self.bs_buckets),
        ) as pbar:
            for i, bs_val in pbar:
                pbar.set_postfix(bs=bs_val)
                aligned_cache_loc_size = self.cache_loc_buckets[i]
                batch = self._make_dummy_batch(
                    bs_val,
                    bs_val,
                    ForwardMode.DECODE,
                    aligned_cache_loc_size,
                    dp_size=self.dp_size,
                    per_dp_bs_size=bs_val // self.dp_size,
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
                self._compiled_variants.add((ForwardMode.DECODE, bs_val, bs_val, False))

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

        valid_input_ids = np.array([1] * bs, dtype=jnp.int32)
        invalid_input_ids = np.array([0] * (num_tokens - bs), dtype=jnp.int32)
        valid_out_cache_loc = np.arange(1, bs + 1, dtype=jnp.int32)
        invalid_out_cache_loc = np.array([-1] * (num_tokens - bs), dtype=jnp.int32)
        valid_positions = np.array([0] * bs, dtype=jnp.int32)
        invalid_positions = np.array([0] * (num_tokens - bs), dtype=jnp.int32)
        invalid_cache_loc_size = max_cache_loc_size - bs
        if invalid_cache_loc_size < 0:
            raise ValueError(f"padding cache_loc_size {invalid_cache_loc_size} < 0!")

        valid_cache_loc = np.arange(bs)
        invalid_cache_loc = np.array([0] * invalid_cache_loc_size, dtype=jnp.int32)
        lora_ids = ["0"] * bs

        extend_seq_lens = np.array([1] * bs) if mode == ForwardMode.EXTEND else None
        logits_indices = np.array([0] * bs) if mode == ForwardMode.EXTEND else None

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
            real_input_ids_len=len(valid_input_ids),
            real_bs=bs,
            req_pool_indices=np.arange(bs, dtype=np.int32),
            seq_lens=np.array([1] * bs, dtype=np.int32),
            out_cache_loc=np.concat([valid_out_cache_loc, invalid_out_cache_loc], axis=0),
            return_logprob=False,
            return_output_logprob_only=return_output_logprob_only,
            sampling_info=sampling_info,
            extend_input_logprob_token_ids=None,
            positions=np.concat([valid_positions, invalid_positions], axis=0),
            cache_loc=np.concat([valid_cache_loc, invalid_cache_loc], axis=0),
            extend_prefix_lens=(np.array([0] * bs) if mode == ForwardMode.EXTEND else None),
            extend_seq_lens=extend_seq_lens,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_logprob_start_lens=None,
            logits_indices=logits_indices,
            input_logprob_indices=None,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL if self.capture_hidden_states else CaptureHiddenMode.NULL
            ),
            spec_algorithm=spec_algorithm_value,
            lora_ids=lora_ids,
            dp_size=dp_size,
            per_dp_bs_size=per_dp_bs_size,
            real_bs_per_dp=[per_dp_bs_size] * dp_size,
            logits_indices_selector=np.arange(bs, dtype=np.int32),
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
