"""A tensor parallel worker."""

import dataclasses
import logging
import signal
import threading
from queue import Queue

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.managers.utils import resolve_future_token_ids, set_future_token_ids
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, FutureEagleDraftInput
from sgl_jax.srt.speculative.eagle_worker import EAGLEWorker
from sgl_jax.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class ModelWorkerClient:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
    ):
        # Load the model
        self.worker = ModelWorker(server_args, mesh=mesh)
        # overlap mode set worker need_prepare_lora_batch to False
        self.worker.need_prepare_lora_batch = False

        self.max_running_requests = self.worker.max_running_requests
        self.device = self.worker.device

        # Init future mappings
        self.future_token_ids_ct = 0
        self.future_token_ids_limit = self.max_running_requests * 3
        self.future_token_ids_map = jnp.zeros((self.max_running_requests * 5,), dtype=jnp.int32)
        self.mesh = mesh
        sharding = NamedSharding(mesh, PartitionSpec(None))
        self.future_token_ids_map = jax.device_put(self.future_token_ids_map, sharding)
        # Launch threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        # JAX handles device execution automatically, no need for explicit streams
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
            daemon=bool(server_args.enable_single_process),
        )
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()

    def get_model_runner(self):
        return self.worker.get_model_runner()

    def get_worker_info(self):
        return self.worker.get_worker_info()

    def get_pad_input_ids_func(self):
        return self.worker.get_pad_input_ids_func()

    def get_memory_pool(self):
        return (
            self.worker.model_runner.req_to_token_pool,
            self.worker.model_runner.token_to_kv_pool_allocator,
        )

    def get_kv_cache(self):
        return self.worker.model_runner.token_to_kv_pool

    def get_max_padded_size(self):
        return self.worker.get_max_padded_size()

    def get_precompile_paddings(self):
        return self.worker.get_precompile_paddings()

    def forward_thread_func(self):
        try:
            self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error("ModelWorkerClient hit an exception: %s", traceback)
            self.parent_process.send_signal(signal.SIGQUIT)

    def forward_thread_func_(self):
        while True:
            (
                model_worker_batch,
                future_token_ids_ct,
                sampling_metadata,
                forward_metadata,
            ) = self.input_queue.get()
            if not model_worker_batch:
                break

            # Resolve future tokens in the input
            input_ids = model_worker_batch.forward_batch.input_ids
            model_worker_batch.forward_batch.input_ids = resolve_future_token_ids(
                input_ids, self.future_token_ids_map
            )

            # Run forward
            with jax.profiler.TraceAnnotation(f"forward_batch_generation {model_worker_batch.bid}"):
                logits_output, next_token_ids, cache_miss_count = (
                    self.worker.forward_batch_generation(
                        model_worker_batch,
                        model_worker_batch.launch_done,
                        sampling_metadata=sampling_metadata,
                        forward_metadata=forward_metadata,
                    )
                )

            # Update the future token ids map
            self.future_token_ids_map = set_future_token_ids(
                self.future_token_ids_map,
                future_token_ids_ct,
                next_token_ids,
            )
            self.output_queue.put((None, logits_output, next_token_ids, cache_miss_count))

    def resolve_last_batch_result(self, launch_done: threading.Event | None = None):
        """
        This function is called to resolve the last batch result and
        wait for the current batch to be launched. Used in overlap mode.
        """
        _, logits_output, next_token_ids, cache_miss_count = self.output_queue.get()
        if logits_output.next_token_logprobs is not None:
            logits_output.next_token_logprobs = jax.device_get(
                logits_output.next_token_logprobs
            ).tolist()
        if logits_output.input_token_logprobs is not None:
            logits_output.input_token_logprobs = jax.device_get(
                logits_output.input_token_logprobs
            ).tolist()
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = jax.device_get(logits_output.hidden_states)
        next_token_ids = jax.device_get(next_token_ids).tolist()

        if launch_done is not None:
            launch_done.wait()

        return logits_output, next_token_ids, cache_miss_count

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        sampling_metadata: SamplingMetadata = None,
    ) -> tuple[None, jax.Array, int]:
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            penalizer_orchestrator=None,
        )

        if sampling_metadata is None:
            sampling_metadata = SamplingMetadata.from_model_worker_batch(
                model_worker_batch,
                len(model_worker_batch.seq_lens) - model_worker_batch.real_bs,
                self.mesh,
                self.worker.model_config.vocab_size,
            )

        forward_metadata = self.worker.model_runner.attn_backend.get_forward_metadata(
            model_worker_batch
        )

        # Prepare LoRA batch if LoRA is enabled
        if self.worker.server_args.enable_lora:
            self.worker.prepare_lora_batch(model_worker_batch)

        model_worker_batch.forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.worker.get_model_runner()
        )

        # Push a new batch to the queue (JAX handles synchronization automatically)
        self.input_queue.put(
            (
                model_worker_batch,
                self.future_token_ids_ct,
                sampling_metadata,
                forward_metadata,
            )
        )

        # Allocate output future objects
        bs = len([seq_len for seq_len in model_worker_batch.seq_lens if seq_len > 0])

        future_next_token_ids = np.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=np.int32,
        )
        self.future_token_ids_ct = (self.future_token_ids_ct + bs) % self.future_token_ids_limit
        return None, future_next_token_ids, 0

    def run_precompile(self):
        self.worker.run_precompile(self.future_token_ids_map)

    @property
    def sliding_window_size(self) -> int | None:
        return self.worker.sliding_window_size

    @property
    def is_hybrid(self) -> bool:
        return self.worker.is_hybrid

    def get_tokens_per_layer_info(self):
        return self.worker.get_tokens_per_layer_info()


class EagleWorkerClient:
    """An overlap-enabled wrapper for the Eagle worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
        target_worker: ModelWorker,
    ):
        self.worker = EAGLEWorker(server_args, target_worker=target_worker)
        self.max_running_requests = target_worker.max_running_requests
        self.max_draft_tokens = server_args.speculative_num_draft_tokens
        self.topk = server_args.speculative_eagle_topk
        self.mesh = mesh

        if self.worker.model_config.dtype is None:
            hidden_dtype = jnp.bfloat16
        else:
            hidden_dtype = jnp.dtype(self.worker.model_config.dtype)

        # Init future maps
        self.future_ct = 0
        self.future_limit = self.max_running_requests * 3
        self.future_map_size = self.max_running_requests * 5
        self.future_token_ids_map = jnp.zeros(
            (self.future_map_size * self.max_draft_tokens,), dtype=jnp.int32
        )
        self.future_accept_length_map = jnp.zeros((self.future_map_size,), dtype=jnp.int32)
        self.future_hidden_states_map = jnp.zeros(
            (self.future_map_size, self.worker.model_config.hidden_size),
            dtype=hidden_dtype,
        )
        self.future_topk_p_map = jnp.zeros(
            (self.future_map_size, self.topk), dtype=jnp.bfloat16
        )
        self.future_topk_index_map = jnp.zeros(
            (self.future_map_size, self.topk), dtype=jnp.int32
        )
        self.future_verified_id_map = jnp.zeros((self.future_map_size,), dtype=jnp.int32)

        sharding_1d = NamedSharding(mesh, PartitionSpec(None))
        sharding_2d = NamedSharding(mesh, PartitionSpec(None, None))
        self.future_token_ids_map = jax.device_put(self.future_token_ids_map, sharding_1d)
        self.future_accept_length_map = jax.device_put(self.future_accept_length_map, sharding_1d)
        self.future_hidden_states_map = jax.device_put(self.future_hidden_states_map, sharding_2d)
        self.future_topk_p_map = jax.device_put(self.future_topk_p_map, sharding_2d)
        self.future_topk_index_map = jax.device_put(self.future_topk_index_map, sharding_2d)
        self.future_verified_id_map = jax.device_put(self.future_verified_id_map, sharding_1d)

        # Launch threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
            daemon=bool(server_args.enable_single_process),
        )
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()

        self.speculative_num_draft_tokens = self.worker.speculative_num_draft_tokens

    def forward_thread_func(self):
        try:
            self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error("EagleWorkerClient hit an exception: %s", traceback)
            self.parent_process.send_signal(signal.SIGQUIT)

    def forward_thread_func_(self):
        while True:
            (model_worker_batch, future_ct) = self.input_queue.get()
            if not model_worker_batch:
                break

            # Resolve future draft_input (if needed)
            if isinstance(model_worker_batch.spec_info, FutureEagleDraftInput):
                future_input = model_worker_batch.spec_info
                map_indices = self._get_future_indices(future_input)
                accept_lens = jax.device_get(self.future_accept_length_map[map_indices])
                model_worker_batch.spec_info = self._resolve_future_draft_input(future_input)
                model_worker_batch.seq_lens[: future_input.bs] = (
                    model_worker_batch.seq_lens[: future_input.bs] + accept_lens - 1
                )
                model_worker_batch.seq_lens_sum = np.sum(model_worker_batch.seq_lens)

            # Resolve future tokens in the input ids
            if model_worker_batch.input_ids is not None and model_worker_batch.input_ids.size > 0:
                resolved_input_ids = resolve_future_token_ids(
                    model_worker_batch.input_ids, self.future_token_ids_map
                )
                model_worker_batch.input_ids = np.asarray(jax.device_get(resolved_input_ids))

            # Run forward
            result = self.worker.forward_batch_speculative_generation(model_worker_batch)

            # Update future maps
            self._update_future_maps(future_ct, model_worker_batch.real_bs, result)

            self.output_queue.put(result)

    def _get_future_indices(self, future_input: FutureEagleDraftInput):
        if future_input.keep_indices is None:
            return jnp.arange(
                future_input.future_ct,
                future_input.future_ct + future_input.bs,
                dtype=jnp.int32,
            )
        return jnp.asarray(future_input.keep_indices, dtype=jnp.int32) + future_input.future_ct

    def _resolve_future_draft_input(self, future_input: FutureEagleDraftInput) -> EagleDraftInput:
        map_indices = self._get_future_indices(future_input)
        return EagleDraftInput(
            hidden_states=self.future_hidden_states_map[map_indices],
            topk_p=self.future_topk_p_map[map_indices],
            topk_index=self.future_topk_index_map[map_indices],
            verified_id=self.future_verified_id_map[map_indices],
            allocate_lens=future_input.allocate_lens,
            capture_hidden_mode=future_input.capture_hidden_mode,
        )

    def _update_future_maps(self, future_ct, bs: int, result):
        draft_input = result.next_draft_input
        accept_lens = result.accept_lens
        token_ids = result.next_token_ids

        stride = self.max_draft_tokens
        start = future_ct * stride

        token_ids_arr = jnp.asarray(token_ids, dtype=jnp.int32).reshape(-1)
        if token_ids_arr.size > 0:
            self.future_token_ids_map = self.future_token_ids_map.at[
                start : start + token_ids_arr.size
            ].set(token_ids_arr)

        if accept_lens is None:
            accept_lens_arr = jnp.ones((bs,), dtype=jnp.int32)
        else:
            accept_lens_arr = jnp.asarray(accept_lens, dtype=jnp.int32)
        self.future_accept_length_map = self.future_accept_length_map.at[
            future_ct : future_ct + bs
        ].set(accept_lens_arr[:bs])

        if draft_input is not None and draft_input.hidden_states is not None:
            self.future_hidden_states_map = self.future_hidden_states_map.at[
                future_ct : future_ct + bs
            ].set(draft_input.hidden_states[:bs])
        if draft_input is not None and draft_input.topk_p is not None:
            self.future_topk_p_map = self.future_topk_p_map.at[
                future_ct : future_ct + bs
            ].set(draft_input.topk_p[:bs])
        if draft_input is not None and draft_input.topk_index is not None:
            self.future_topk_index_map = self.future_topk_index_map.at[
                future_ct : future_ct + bs
            ].set(draft_input.topk_index[:bs])
        if draft_input is not None and draft_input.verified_id is not None:
            self.future_verified_id_map = self.future_verified_id_map.at[
                future_ct : future_ct + bs
            ].set(draft_input.verified_id[:bs])

    def resolve_last_batch_result(self, launch_done: threading.Event | None = None):
        result = self.output_queue.get()

        logits_output = result.logits_output
        if logits_output is not None:
            if logits_output.next_token_logprobs is not None:
                logits_output.next_token_logprobs = jax.device_get(
                    logits_output.next_token_logprobs
                ).tolist()
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = jax.device_get(
                    logits_output.input_token_logprobs
                ).tolist()
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = jax.device_get(logits_output.hidden_states)

        if result.next_token_ids is not None:
            result.next_token_ids = jax.device_get(result.next_token_ids).tolist()
        if result.accept_lens is not None:
            result.accept_lens = np.asarray(jax.device_get(result.accept_lens))

        if launch_done is not None:
            launch_done.wait()

        return result

    def forward_batch_speculative_generation(self, model_worker_batch: ModelWorkerBatch):
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult

        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            penalizer_orchestrator=None,
        )

        self.input_queue.put((model_worker_batch, self.future_ct))

        bs = model_worker_batch.real_bs
        stride = self.max_draft_tokens
        placeholder_len = bs * stride if model_worker_batch.forward_mode.is_decode() else bs

        future_next_token_ids = np.arange(
            -(self.future_ct * stride + 1),
            -(self.future_ct * stride + 1 + placeholder_len),
            -1,
            dtype=np.int32,
        )
        future_accept_length = np.arange(
            -(self.future_ct + 1),
            -(self.future_ct + 1 + bs),
            -1,
            dtype=np.int32,
        )

        allocate_lens = None
        if isinstance(model_worker_batch.spec_info, EagleDraftInput):
            allocate_lens = model_worker_batch.spec_info.allocate_lens
            if isinstance(allocate_lens, jax.Array):
                allocate_lens = np.asarray(jax.device_get(allocate_lens))

        future_draft_input = FutureEagleDraftInput(
            future_ct=self.future_ct,
            bs=bs,
            allocate_lens=allocate_lens,
        )

        self.future_ct = (self.future_ct + bs) % self.future_limit

        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=future_next_token_ids,
            next_draft_input=future_draft_input,
            accept_lens=future_accept_length,
            allocate_lens=allocate_lens,
            bid=model_worker_batch.bid,
            cache_miss_count=0,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )

    def run_spec_decode_precompile(self):
        self.worker.run_spec_decode_precompile()

    def __delete__(self):
        self.input_queue.put((None, None, None, None))
