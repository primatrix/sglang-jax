"""CPU timing simulation used by the serving model runner."""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable
from functools import partial

import jax
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.layers.sampler import compute_logprobs
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.multimodal.in_model.host_orchestration import (
    release_consumed_embeddings,
)


class SimulatedDevice:
    """A FIFO compute worker that preserves asynchronous host/device overlap."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="sim-device", daemon=True).start()

    def _run(self) -> None:
        while True:
            duration, done, kind, bid, batch_size, on_complete = self._queue.get()
            with jax.profiler.TraceAnnotation(
                f"sim_device_compute:{kind}:bid={bid}:batch={batch_size}"
            ) as trace:
                if trace.is_enabled():
                    trace.set_metadata(start_ns=time.time_ns())
                if duration > 0:
                    time.sleep(duration)
                if trace.is_enabled():
                    trace.set_metadata(end_ns=time.time_ns())
            if on_complete is not None:
                on_complete()
            done.set()

    def dispatch(
        self,
        duration_s: float,
        *,
        kind: str = "unknown",
        bid: int | str = "unknown",
        batch_size: int = 0,
        on_complete: Callable[[], None] | None = None,
    ) -> threading.Event:
        done = threading.Event()
        self._queue.put((duration_s, done, kind, bid, batch_size, on_complete))
        return done


class SimulationMixin:
    def _sim_duration_s(self, forward_batch: ForwardBatch) -> float:
        """Modeled device forward time (seconds), linear in batch shape."""
        if self._sim_precompiling:
            return 0.0
        args = self.server_args
        if forward_batch.forward_mode.is_extend():
            ms = (
                args.simulate_compute_prefill_base_ms
                + args.simulate_compute_prefill_ms_per_token * int(forward_batch.input_ids.shape[0])
            )
        elif forward_batch.forward_mode.is_decode():
            ms = (
                args.simulate_compute_decode_base_ms
                + args.simulate_compute_decode_ms_per_seq * int(forward_batch.batch_size)
            )
        else:
            ms = 0.0
        return ms / 1000.0

    @contextlib.contextmanager
    def simulation_precompile(self):
        """Compile simulator JAX paths without enqueueing modeled device time."""
        if self._sim_device is None:
            yield
            return

        previous = self._sim_precompiling
        self._sim_precompiling = True
        try:
            yield
        finally:
            while True:
                try:
                    done = self._sim_completions.get_nowait()
                except queue.Empty:
                    break
                done.wait()
            self._sim_precompiling = previous

    def sim_wait_next_completion(self) -> None:
        """Block until the oldest in-flight simulated forward completes.

        Called by the host at result resolution (overlap mode). Dispatch order
        equals resolution order, so a FIFO of completion events is correct.
        """
        if self._sim_device is None:
            return
        try:
            done = self._sim_completions.get_nowait()
        except queue.Empty:
            return
        with jax.profiler.TraceAnnotation("sim_device_wait"):
            done.wait()

    def simulation_logits_output(
        self,
        forward_batch: ForwardBatch,
        *,
        full_vocab: bool = False,
    ) -> LogitsProcessorOutput:
        """Return placeholder logits for ``--simulate-compute``.

        Normal simulation never consumes logits, so keep the vocab dimension at
        one.  This avoids making CPU simulation spend most of its time allocating
        and sampling an otherwise unused ``[batch, vocab_size]`` tensor.  The
        full-vocab form remains available for logprob and grammar-constrained
        requests, which still use the real sampler for compatibility.
        """
        vocab_size = self.model_config.vocab_size if full_vocab else 1
        logits = np.zeros(
            (int(forward_batch.batch_size), vocab_size),
            dtype=np.float32,
        )
        logits = jax.device_put(
            logits,
            NamedSharding(
                self.mesh,
                P("data", "tensor") if full_vocab else P("data", None),
            ),
        )
        return LogitsProcessorOutput(next_token_logits=logits)

    def simulation_sample(self, seq_lens: np.ndarray) -> jax.Array:
        """Generate cheap deterministic token ids for valid simulated rows."""
        simulated_token_id = max(min(32, self.model_config.vocab_size - 1), 0)
        next_token_ids = np.where(
            np.asarray(seq_lens) > 0,
            simulated_token_id,
            0,
        ).astype(np.int32)
        return jax.device_put(
            next_token_ids,
            NamedSharding(self.mesh, P("data")),
        )

    def initialize_simulation_jit(self):
        sampler_def, sampler_state = nnx.split(self.sampler)
        sampler_state_leaves, sampler_state_def = jax.tree_util.tree_flatten(sampler_state)
        base_rng_key = self._sampler_base_rng

        @partial(jax.jit, static_argnames=["sampler_state_def", "use_sort_for_toppk_minp"])
        def jitted_sampler(
            sampler_def,
            sampler_state_def,
            sampler_state_leaves,
            use_sort_for_toppk_minp,
            rng_step,
            *args,
        ):
            model_state = jax.tree_util.tree_unflatten(sampler_state_def, sampler_state_leaves)
            sampler = nnx.merge(sampler_def, model_state)
            rng_key = jax.random.fold_in(base_rng_key, rng_step)
            return sampler(
                *args, use_sort_for_toppk_minp=use_sort_for_toppk_minp, rng_override=rng_key
            )

        @partial(jax.jit, static_argnames=["mesh"])
        def jitted_compute_logprobs(mesh, logits, next_tokens):
            return compute_logprobs(mesh, logits, next_tokens)

        self.jitted_run_model = None
        self.jitted_sampler = partial(
            jitted_sampler,
            sampler_def,
            sampler_state_def,
            sampler_state_leaves,
            self.use_sort_for_toppk_minp,
        )
        self.jitted_compute_logprobs = partial(jitted_compute_logprobs, self.mesh)

    def simulation_forward(self, forward_batch):
        def release_embeddings() -> None:
            # Simulation skips the real gather; release its leases after modeled
            # consumption, retaining any embedding needed by a later chunk.
            for tasks in (forward_batch.multimodal_batch or {}).values():
                release_consumed_embeddings(tasks)

        kind = "decode" if forward_batch.forward_mode.is_decode() else "prefill"
        done = self._sim_device.dispatch(
            self._sim_duration_s(forward_batch),
            kind=kind,
            bid=getattr(forward_batch, "bid", "unknown"),
            batch_size=int(forward_batch.batch_size),
            on_complete=release_embeddings,
        )
        if self.server_args.disable_overlap_schedule:
            with jax.profiler.TraceAnnotation("sim_device_wait"):
                done.wait()
        else:
            self._sim_completions.put(done)
        return self.simulation_logits_output(forward_batch), 0, None
