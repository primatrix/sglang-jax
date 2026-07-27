"""Single-threaded worker for normal overlap scheduling."""

import dataclasses
import logging
from functools import partial

import jax
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.sampling.sampling_batch_info import (
    SamplingMetadata,
    _get_or_create_zero_penalty_device,
)
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.overlap_utils import (
    DecodeBatchDescriptor,
    DecodeBatchInputs,
    DecodeWorkspace,
    DecodeWorkspaceBatchSpec,
    gather_decode_batch_inputs,
    resolve_decode_inputs,
    resolve_relay_inputs,
    update_decode_result,
)

logger = logging.getLogger(__name__)
_DECODE_WORKSPACE_BACKENDS = {"FlashAttention", "MLAAttentionBackend"}


@dataclasses.dataclass
class ForwardContext:
    batch: ModelWorkerBatch
    logits_output: LogitsProcessorOutput
    sampling_metadata: SamplingMetadata
    cache_miss_count: int


class ModelWorkerOverlap(ModelWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
        model_class=None,
        precompile_params: dict | None = None,
    ):
        super().__init__(
            server_args=server_args,
            mesh=mesh,
            model_class=model_class,
            precompile_params=precompile_params,
        )
        self.need_prepare_lora_batch = False
        self.cur_sampling_info = None
        self.decode_workspace = DecodeWorkspace(
            mesh,
            self.model_runner.req_to_token_pool,
            dp_size=self.dp_size,
        )
        self._resolve_relay = jax.jit(
            partial(
                resolve_relay_inputs,
                dp_size=self.dp_size,
                relay_sharding=self.decode_workspace.relay_sharding,
                output_sharding=self.decode_workspace.input_sharding,
            ),
            out_shardings=self.decode_workspace.input_sharding,
        )
        self._resolve_decode = jax.jit(
            partial(
                resolve_decode_inputs,
                dp_size=self.dp_size,
                relay_sharding=self.decode_workspace.relay_sharding,
                output_sharding=self.decode_workspace.input_sharding,
            ),
            out_shardings=(
                self.decode_workspace.input_sharding,
                self.decode_workspace.input_sharding,
            ),
        )
        self._decode_constant_cache: dict[tuple, jax.Array] = {}
        self._gather_and_forward = self._build_gather_and_forward()
        self._sample_and_update = self._build_sample_and_update()

    def run_precompile(self, future_token_ids_map=None, only: str | None = None):
        super().run_precompile(future_token_ids_map, only)
        if only != "extend":
            self._precompile_decode_workspace()

    def _precompile_decode_workspace(self):
        if not self._decode_workspace_supported():
            return

        logger.info(
            "[DECODE WORKSPACE] Begin to precompile bs_paddings=%s",
            self.compilation_manager.bs_buckets,
        )
        for bs_index, batch_size in enumerate(self.compilation_manager.bs_buckets):
            per_dp_bs = batch_size // self.dp_size
            batch = self.compilation_manager._make_dummy_batch(
                batch_size,
                batch_size,
                ForwardMode.DECODE,
                self.compilation_manager.cache_loc_buckets[bs_index],
                dp_size=self.dp_size,
                per_dp_bs_size=per_dp_bs,
            )
            req_pool_indices = np.tile(
                np.arange(per_dp_bs, dtype=np.int32),
                self.dp_size,
            )
            batch.req_pool_indices = req_pool_indices
            batch.decode_page_indices = np.zeros(
                self.compilation_manager.cache_loc_buckets[bs_index] // self.page_size,
                dtype=np.int32,
            )
            sampling_info = batch.sampling_info
            batch.decode_workspace_spec = DecodeWorkspaceBatchSpec(
                is_all_greedy=sampling_info.is_all_greedy,
                need_min_p_sampling=sampling_info.need_min_p_sampling,
                has_sampling_seeds=sampling_info.sampling_seeds is not None,
            )
            self.decode_workspace.publish_request_state(
                req_pool_indices,
                batch.input_ids,
                batch.seq_lens,
                sampling_info.temperatures,
                sampling_info.top_ps,
                sampling_info.top_ks,
                sampling_info.min_ps,
                sampling_info.sampling_seeds,
            )
            context = self.launch_forward(batch)
            self.launch_sample(context)

        logger.info("[DECODE WORKSPACE] Precompile finished")

    def _build_gather_and_forward(self):
        gather = partial(
            gather_decode_batch_inputs,
            dp_size=self.dp_size,
            page_size=self.page_size,
            relay_sharding=self.decode_workspace.relay_sharding,
            state_sharding=self.decode_workspace.state_sharding,
            output_sharding=self.decode_workspace.input_sharding,
        )
        run_model = self.model_runner.jitted_run_model_with_memory_pools

        @partial(jax.jit, donate_argnames=["memory_pools"])
        def gather_and_forward(
            forward_batch,
            memory_pools,
            logits_metadata,
            request_state,
            descriptor,
            model_state_leaves,
        ):
            batch_inputs = gather(request_state, descriptor)
            forward_batch = dataclasses.replace(
                forward_batch,
                input_ids=batch_inputs.input_ids,
                positions=batch_inputs.positions,
                req_pool_indices=descriptor.req_pool_indices,
                seq_lens=batch_inputs.seq_lens,
            )
            forward_batch.attn_backend.forward_metadata = dataclasses.replace(
                forward_batch.attn_backend.forward_metadata,
                cu_kv_lens=batch_inputs.cu_kv_lens,
                page_indices=descriptor.page_indices,
                seq_lens=batch_inputs.seq_lens,
                distribution=batch_inputs.distribution,
            )
            output, pool_updates, aux, layers_topk_ids = run_model(
                model_state_leaves,
                forward_batch,
                memory_pools,
                logits_metadata,
            )
            return output, pool_updates, aux, layers_topk_ids, batch_inputs

        return gather_and_forward

    def _build_sample_and_update(self):
        sample = self.model_runner.jitted_sampler
        update = partial(
            update_decode_result,
            dp_size=self.dp_size,
            relay_sharding=self.decode_workspace.relay_sharding,
        )

        @jax.jit
        def sample_and_update(
            request_state,
            req_pool_indices,
            current_seq_lens,
            rng_step,
            logits_output,
            sampling_metadata,
        ):
            next_token_ids, token_logprobs, sampled_output = sample(
                rng_step,
                logits_output,
                sampling_metadata,
            )
            request_state = update(
                request_state,
                req_pool_indices,
                next_token_ids,
                current_seq_lens,
            )
            return request_state, next_token_ids, token_logprobs, sampled_output

        return sample_and_update

    def can_use_decode_workspace(self, batch) -> bool:
        if (
            not batch.forward_mode.is_decode()
            or batch.has_grammar
            or not self._decode_workspace_supported()
        ):
            return False

        req_pool_indices_per_dp = []
        for info in batch.reqs_info:
            sampling_info = info.sampling_info
            if info.reqs and sampling_info is None:
                return False
            if sampling_info is not None:
                orchestrator = sampling_info.penalizer_orchestrator
                if orchestrator is not None and orchestrator.is_required:
                    return False
                if sampling_info.linear_penalty is not None and sampling_info.linear_penalty.size:
                    return False
            req_pool_indices_per_dp.append(info.req_pool_indices)
        return self.decode_workspace.contains_request_slots(req_pool_indices_per_dp)

    def _get_decode_zeros(self, shape, dtype):
        key = (tuple(shape), np.dtype(dtype).str)
        value = self._decode_constant_cache.get(key)
        if value is None:
            value = jax.device_put(
                np.zeros(shape, dtype=dtype),
                self.decode_workspace.input_sharding,
            )
            self._decode_constant_cache[key] = value
        return value

    def _decode_workspace_supported(self) -> bool:
        attn_backend = self.model_runner.attn_backend
        return (
            type(attn_backend).__name__ in _DECODE_WORKSPACE_BACKENDS
            and not self.server_args.enable_lora
            and not self.server_args.multimodal
            and not self.is_hybrid
            and not getattr(self.server_args, "enable_custom_logit_processor", False)
            and getattr(attn_backend, "swa_index_mapping", None) is None
        )

    def _get_workspace_attention_metadata(
        self,
        batch: ModelWorkerBatch,
        descriptor: DecodeBatchDescriptor,
    ):
        backend = self.model_runner.attn_backend
        backend_name = type(backend).__name__
        if backend_name not in _DECODE_WORKSPACE_BACKENDS:
            raise RuntimeError(f"Unsupported decode workspace backend: {backend_name}")

        cu_q_lens = self._decode_constant_cache.get(
            ("cu_q_lens", batch.dp_size, batch.per_dp_bs_size)
        )
        if cu_q_lens is None:
            cu_q_lens = jax.device_put(
                np.tile(
                    np.arange(batch.per_dp_bs_size + 1, dtype=np.int32),
                    batch.dp_size,
                ),
                self.decode_workspace.input_sharding,
            )
            self._decode_constant_cache[("cu_q_lens", batch.dp_size, batch.per_dp_bs_size)] = (
                cu_q_lens
            )
        seq_lens = self._get_decode_zeros(
            (len(batch.seq_lens),),
            np.int32,
        )
        cu_kv_lens = self._get_decode_zeros(
            (batch.dp_size * (batch.per_dp_bs_size + 1),),
            np.int32,
        )
        distribution = self._get_decode_zeros(
            (batch.dp_size * 3,),
            np.int32,
        )

        if backend_name == "FlashAttention":
            from sgl_jax.srt.layers.attention.flashattention_backend import (
                FlashAttentionMetadata,
            )

            return FlashAttentionMetadata(
                cu_q_lens=cu_q_lens,
                cu_kv_lens=cu_kv_lens,
                page_indices=descriptor.page_indices,
                swa_page_indices=None,
                seq_lens=seq_lens,
                distribution=distribution,
                custom_mask=None,
            )

        from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionMetadata

        return MLAAttentionMetadata(
            cu_q_lens=cu_q_lens,
            cu_kv_lens=cu_kv_lens,
            page_indices=descriptor.page_indices,
            seq_lens=seq_lens,
            distribution=distribution,
        )

    def _get_workspace_sampling_metadata(
        self,
        batch: ModelWorkerBatch,
        batch_inputs: DecodeBatchInputs,
    ) -> SamplingMetadata:
        spec = batch.decode_workspace_spec
        linear_penalty = _get_or_create_zero_penalty_device(
            (len(batch.seq_lens), self.model_config.vocab_size),
            NamedSharding(self.mesh, P("data", "tensor")),
        )
        return SamplingMetadata(
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
            token_ids_logprobs=batch.token_ids_logprobs,
            temperatures=batch_inputs.temperatures,
            top_ps=batch_inputs.top_ps,
            top_ks=batch_inputs.top_ks,
            min_ps=batch_inputs.min_ps,
            sampling_seeds=(batch_inputs.sampling_seeds if spec.has_sampling_seeds else None),
            positions=batch_inputs.positions,
            is_all_greedy=spec.is_all_greedy,
            need_min_p_sampling=spec.need_min_p_sampling,
            linear_penalty=linear_penalty,
            do_penalties=False,
            vocab_mask=None,
            apply_vocab_mask=False,
        )

    def _get_zero_vocab_mask(self, batch_size, vocab_size):
        vocab_words = (vocab_size + 31) // 32
        return self._get_decode_zeros((batch_size, vocab_words), np.int32)

    def _update_overlap_vocab_mask(self, batch, sampling_metadata):
        if batch.decode_workspace_spec is not None:
            sampling_metadata.apply_vocab_mask = False
            sampling_metadata.vocab_mask = self._get_zero_vocab_mask(
                len(batch.seq_lens),
                self.model_config.vocab_size,
            )
            return

        sampling_info = batch.sampling_info
        if sampling_info.grammars and sampling_info.vocab_mask is None:
            sampling_info.update_grammar_vocab_mask()
        if sampling_info.vocab_mask is not None:
            sampling_metadata.apply_vocab_mask = True
            sampling_metadata.vocab_mask = sampling_info.vocab_mask
            return

        sampling_metadata.apply_vocab_mask = False
        batch_size = len(sampling_info.temperatures)
        sampling_metadata.vocab_mask = self._get_zero_vocab_mask(
            batch_size,
            sampling_info.vocab_size,
        )

    def launch_forward(
        self,
        batch: ModelWorkerBatch,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> ForwardContext:
        if batch.sampling_info is not None:
            batch.sampling_info.update_penalties()
        self.cur_sampling_info = batch.sampling_info

        if self.server_args.enable_lora:
            self.prepare_lora_batch(batch)

        use_workspace = batch.decode_workspace_spec is not None
        device_overrides = None
        if use_workspace:
            decode_descriptor = self.decode_workspace.get_descriptor(
                batch.req_pool_indices,
                batch.decode_page_indices,
            )
            decode_zeros = self._get_decode_zeros(
                (len(batch.seq_lens),),
                np.int32,
            )
            device_overrides = {
                "input_ids": decode_zeros,
                "positions": decode_zeros,
                "req_pool_indices": decode_descriptor.req_pool_indices,
                "seq_lens": decode_zeros,
            }
            device_overrides["cache_loc"] = self._get_decode_zeros(
                (len(batch.decode_page_indices) * self.page_size,),
                np.int32,
            )
        elif batch.forward_mode.is_decode():
            zero_tokens = self._get_decode_zeros(
                np.shape(batch.input_ids),
                np.asarray(batch.input_ids).dtype,
            )
            device_overrides = {
                "input_ids": zero_tokens,
                "positions": zero_tokens,
                "req_pool_indices": jax.device_put(
                    batch.req_pool_indices,
                    self.decode_workspace.input_sharding,
                ),
            }

        forward_batch = ForwardBatch.init_new(
            batch,
            self.model_runner,
            device_overrides=device_overrides,
        )
        if not use_workspace and batch.forward_mode.is_decode():
            forward_batch.input_ids, forward_batch.positions = self._resolve_decode(
                self.decode_workspace.relay_buffers,
                forward_batch.req_pool_indices,
                forward_batch.input_ids,
                forward_batch.seq_lens,
            )
        elif batch.relay_input_indices is not None:
            input_sharding = forward_batch.input_ids.sharding
            indices = jax.device_put(batch.relay_input_indices, input_sharding)
            mask = jax.device_put(batch.relay_input_mask, input_sharding)
            forward_batch.input_ids = self._resolve_relay(
                self.decode_workspace.relay_buffers,
                indices,
                mask,
                forward_batch.input_ids,
            )
        batch.forward_batch = forward_batch
        if use_workspace:
            forward_metadata = self._get_workspace_attention_metadata(
                batch,
                decode_descriptor,
            )
            self.model_runner.attn_backend.forward_metadata = forward_metadata
            logits_metadata = LogitsMetadata.from_model_worker_batch(batch, self.mesh)
            (
                logits_output,
                cache_miss_count,
                layers_topk_ids,
                decode_batch_inputs,
            ) = self.model_runner.forward_with_jitted_runner(
                self._gather_and_forward,
                forward_batch,
                logits_metadata,
                self.decode_workspace.request_state,
                decode_descriptor,
                self.model_runner.model_state_leaves,
            )
            forward_batch.input_ids = decode_batch_inputs.input_ids
            forward_batch.positions = decode_batch_inputs.positions
            forward_batch.req_pool_indices = decode_descriptor.req_pool_indices
            forward_batch.seq_lens = decode_batch_inputs.seq_lens
            sampling_metadata = self._get_workspace_sampling_metadata(
                batch,
                decode_batch_inputs,
            )
            self._finalize_forward(
                batch,
                forward_batch,
                logits_output,
                layers_topk_ids,
                batch.launch_done,
            )
        else:
            if sampling_metadata is None:
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    batch,
                    0,
                    self.mesh,
                    self.model_config.vocab_size,
                )
            sampling_metadata.positions = forward_batch.positions
            forward_metadata = self.model_runner.attn_backend.get_forward_metadata(batch)
            logits_output, _, cache_miss_count = super().forward_batch_generation(
                batch,
                batch.launch_done,
                skip_sample=True,
                sampling_metadata=None,
                forward_metadata=forward_metadata,
            )
        return ForwardContext(
            batch=batch,
            logits_output=logits_output,
            sampling_metadata=sampling_metadata,
            cache_miss_count=cache_miss_count,
        )

    def launch_sample(
        self,
        context: ForwardContext,
    ) -> tuple[LogitsProcessorOutput, jax.Array, int]:
        import jax._src.test_util as jtu

        batch = context.batch
        logits_output = context.logits_output
        sampling_metadata = context.sampling_metadata
        use_workspace = batch.decode_workspace_spec is not None

        self._update_overlap_vocab_mask(batch, sampling_metadata)
        with jtu.count_pjit_cpp_cache_miss() as count:
            if use_workspace:
                (
                    request_state,
                    next_token_ids,
                    token_logprobs,
                    sampled_output,
                ) = self._sample_and_update(
                    self.decode_workspace.request_state,
                    batch.forward_batch.req_pool_indices,
                    batch.forward_batch.seq_lens,
                    self.model_runner.next_sampler_step(),
                    logits_output,
                    sampling_metadata,
                )
                self.decode_workspace.request_state = request_state
            else:
                next_token_ids, token_logprobs, sampled_output = self.model_runner.sample(
                    logits_output,
                    sampling_metadata,
                )
            cache_miss_count = context.cache_miss_count + count()

        if batch.return_output_logprob_only:
            logprobs = self.model_runner.compute_logprobs(token_logprobs, next_token_ids)
            logits_output.next_token_logprobs = logprobs
        if sampled_output is not None:
            logits_output = sampled_output

        if not use_workspace:
            self.decode_workspace.publish_request_state(
                batch.forward_batch.req_pool_indices,
                next_token_ids,
                batch.forward_batch.seq_lens + 1,
                sampling_metadata.temperatures,
                sampling_metadata.top_ps,
                sampling_metadata.top_ks,
                sampling_metadata.min_ps,
                sampling_metadata.sampling_seeds,
            )
            self.decode_workspace.mark_initialized(
                batch.req_pool_indices,
                batch.real_bs_per_dp,
                batch.per_dp_bs_size,
            )

        output_ids = next_token_ids
        if self.dp_size > 1:
            from jax.experimental.multihost_utils import process_allgather

            output_ids = process_allgather(output_ids, tiled=True)

        output_ids.copy_to_host_async()
        for value in (
            logits_output.next_token_logprobs,
            logits_output.input_token_logprobs,
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.input_top_logprobs_val,
            logits_output.input_top_logprobs_idx,
            logits_output.input_token_ids_logprobs_val,
            logits_output.hidden_states,
        ):
            if isinstance(value, jax.Array):
                value.copy_to_host_async()

        return logits_output, output_ids, cache_miss_count

    def resolve_last_batch_result(
        self,
        logits_output: LogitsProcessorOutput,
        next_token_ids: jax.Array,
        batch: ModelWorkerBatch,
        cache_miss_count: int,
        launch_done=None,
    ) -> tuple[LogitsProcessorOutput, list[int], int]:
        next_token_ids = jax.device_get(next_token_ids).tolist()
        if batch.return_logprob or batch.return_output_logprob_only:
            self._materialize_logprobs_to_host(
                logits_output,
                batch,
                batch.logits_indices_selector,
            )
            if logits_output.next_token_logprobs is not None:
                logits_output.next_token_logprobs = np.asarray(
                    logits_output.next_token_logprobs
                ).tolist()
        if logits_output.input_token_logprobs is not None:
            logits_output.input_token_logprobs = np.asarray(
                jax.device_get(logits_output.input_token_logprobs)
            ).tolist()
        if isinstance(logits_output.hidden_states, jax.Array):
            logits_output.hidden_states = np.asarray(jax.device_get(logits_output.hidden_states))
        if launch_done is not None:
            launch_done.wait()
        return logits_output, next_token_ids, cache_miss_count
