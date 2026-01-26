"""ModelRunner runs the forward passes of the models."""

import logging
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax._src import mesh as mesh_lib
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.configs.model_config import AttentionArch, MockModelConfig, ModelConfig
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.layers.routed_experts_capturer import (
    RoutedExpertsCapturer,
    set_global_experts_capturer,
)
from sgl_jax.srt.layers.sampler import Sampler, compute_logprobs
from sgl_jax.srt.lora.context_manager import LoraBatchContext
from sgl_jax.srt.managers.schedule_batch import (
    GLOBAL_SERVER_ARGS_KEYS,
    global_server_args_dict,
)
from sgl_jax.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sgl_jax.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import get_available_device_memory
from sgl_jax.srt.utils.quantization.quantization_utils import (
    apply_moe_quantization,
    apply_qwix_quantization,
)

logger = logging.getLogger(__name__)


class ModelRunner(BaseModelRunner):
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        tp_size: int,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
        is_draft_worker: bool = False,
        req_to_token_pool: ReqToTokenPool | None = None,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator | None = None,
        rngs: nnx.Rngs = None,
        max_padding: int = 1,
    ):
        # Parse args
        self.is_draft_worker = is_draft_worker
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.mesh = mesh
        # model args
        self.num_attn_heads = model_config.num_attention_heads
        self.num_kv_heads = model_config.get_total_num_kv_heads_with_replication(tp_size)
        self.rngs = rngs

        self.tp_size = tp_size
        # For fused MoE, EP group is effectively dp*tp (full 2D mesh). Some TPU
        # deployments create a mesh whose "data" axis is derived from device count,
        # so server_args.ep_size can be inconsistent with the actual mesh.
        ep_size_mesh = int(mesh.shape.get("data", 1) * mesh.shape.get("tensor", 1))
        if server_args.moe_backend == "fused" and int(server_args.ep_size) != ep_size_mesh:
            logger.warning(
                "Overriding server_args.ep_size for fused MoE to match dp*tp mesh: requested=%d actual=%d",
                int(server_args.ep_size),
                int(ep_size_mesh),
            )
            self.ep_size = ep_size_mesh
        else:
            self.ep_size = server_args.ep_size
        self.server_args = server_args
        self.is_generation = model_config.is_generation
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid = False
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.spec_algorithm = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)

        self.forward_pass_id = 0

        # For sampling
        self.use_sort_for_toppk_minp = server_args.use_sort_for_toppk_minp

        self.max_padding = max_padding

        # Global vars
        global_server_args_dict.update(
            {k: getattr(server_args, k) for k in GLOBAL_SERVER_ARGS_KEYS}
        )

        self.model_loader = get_model_loader(
            load_config=LoadConfig(
                load_format=server_args.load_format,
                download_dir=server_args.download_dir,
            ),
            mesh=self.mesh,
        )

        # Initialize precision tracer enable state
        precision_tracer.set_enable_precision_tracer(server_args.enable_precision_tracer)

        # Online EPLB state (host-side).
        self._eplb_metadata = None

        # If it is a draft model, tp_group can be different
        self.initialize()

    def initialize(self):
        server_args = self.server_args

        # Set highest matmul precision only for GPU/CUDA to improve numerical stability.
        # Do this at runtime (not import time) to avoid initializing busy backends.
        try:
            if str(getattr(server_args, "device", "")).lower() in ("gpu", "cuda"):
                from jax import config as _jax_config

                _jax_config.update("jax_default_matmul_precision", "highest")
        except Exception:
            pass

        # Load the model
        self.sampler = Sampler(nnx.Rngs(server_args.random_seed), mesh=self.mesh)
        total_device_memory = self.get_available_device_memory()
        self.init_attention_backend()
        self.load_model()

        # Check if the model is using hybrid SWA
        if (
            not self.server_args.disable_hybrid_swa_memory
            and self.sliding_window_size is not None
            and self.sliding_window_size > 0
        ):
            self.is_hybrid = True

        # Init lora
        if server_args.enable_lora:
            self.init_lora_manager()

        if not self.is_draft_worker:
            self.initialize_jit()

        # Init memory pool and attention backends
        self.init_memory_pool(
            server_args.max_running_requests,
            server_args.max_total_tokens,
            total_device_memory,
        )

        # Init routed experts capturer
        self.init_routed_experts_capturer()

    def init_routed_experts_capturer(self):
        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                enable=self.server_args.enable_return_routed_experts,
                model_config=self.model_config,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_padding=self.max_padding,
            )
        )

    def initialize_jit(self):
        model_def, model_state = nnx.split(self.model)
        # Save for online state rewrites (e.g., EPLB weight rebalance).
        self._model_def = model_def
        # note export for external modification
        self.model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)
        self._model_state_def = model_state_def
        sampler_def, sampler_state = nnx.split(self.sampler)
        sampler_state_leaves, sampler_state_def = jax.tree_util.tree_flatten(sampler_state)

        @partial(
            jax.jit,
            donate_argnames=["token_to_kv_pool"],  # just donate KV cache
            static_argnames=["model_state_def"],
        )
        def jitted_run_model(
            model_def,
            model_state_def,
            model_state_leaves,
            forward_batch,
            token_to_kv_pool,
            logits_metadata,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            with LoraBatchContext.set_batch(forward_batch):
                return model(forward_batch, token_to_kv_pool, logits_metadata)

        @partial(jax.jit, static_argnames=["sampler_state_def", "use_sort_for_toppk_minp"])
        def jitted_sampler(
            sampler_def,
            sampler_state_def,
            sampler_state_leaves,
            use_sort_for_toppk_minp,
            *args,
        ):
            model_state = jax.tree_util.tree_unflatten(sampler_state_def, sampler_state_leaves)
            sampler = nnx.merge(sampler_def, model_state)
            return sampler(*args, use_sort_for_toppk_minp=use_sort_for_toppk_minp)

        @partial(jax.jit, static_argnames=["mesh"])
        def jitted_compute_logprobs(mesh, logits, next_tokens):
            return compute_logprobs(mesh, logits, next_tokens)

        def run_model_wrapper(forward_batch, logits_metadata):
            token_to_kv_pool = self.token_to_kv_pool

            return jitted_run_model(
                model_def,
                model_state_def,
                self.model_state_leaves,
                forward_batch,
                token_to_kv_pool,
                logits_metadata,
            )

        self.jitted_run_model = run_model_wrapper

        self.jitted_sampler = partial(
            jitted_sampler,
            sampler_def,
            sampler_state_def,
            sampler_state_leaves,
            self.use_sort_for_toppk_minp,
        )

        self.jitted_compute_logprobs = partial(jitted_compute_logprobs, self.mesh)

    def _materialize_eplb_dispatch_maps(self) -> None:
        """Materialize EPLB dispatch maps after `nnx.eval_shape` + weight loading.

        The model is instantiated under `nnx.eval_shape`, so non-checkpoint state (like
        EPLB dispatch maps) can remain as `jax.ShapeDtypeStruct` leaves unless explicitly
        replaced with concrete arrays. This must run before the first JIT'ed forward.
        """
        if not getattr(self.server_args, "enable_eplb", False):
            return
        if getattr(self.server_args, "moe_backend", None) != "fused":
            return

        init_dispatch_all = getattr(self.model_config.hf_config, "eplb_initial_dispatch_map", None)

        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.layers.moe import FusedEPMoE

        visited: set[int] = set()
        stack: list[object] = [self.model]

        while stack:
            obj = stack.pop()
            oid = id(obj)
            if oid in visited:
                continue
            visited.add(oid)

            if isinstance(obj, FusedEPMoE) and getattr(obj, "enable_eplb", False):
                if init_dispatch_all is not None:
                    dispatch = init_dispatch_all[int(obj.layer_id)]
                else:
                    dispatch = np.tile(
                        np.arange(int(obj.num_logical_experts), dtype=np.int32)[:, None],
                        (1, int(obj.ep_size)),
                    )
                obj.eplb_dispatch_map.value = jax.device_put(
                    np.asarray(dispatch, dtype=np.int32),
                    NamedSharding(self.mesh, P()),
                )
                continue

            # Treat arrays and scalars as leaves.
            if isinstance(obj, (jax.Array, np.ndarray, str, bytes, int, float, bool, type(None))):
                continue

            if isinstance(obj, (list, tuple, set, frozenset)):
                stack.extend(list(obj))
                continue
            if isinstance(obj, dict):
                stack.extend(list(obj.values()))
                continue
            if hasattr(obj, "__dict__"):
                stack.extend(list(obj.__dict__.values()))

    def apply_eplb_rebalance(self, *, new_metadata) -> None:
        """Apply a new EPLB placement online (fused MoE only).

        This updates:
          - fused MoE expert weights (w1/w2/w3) via `jax.lax.ragged_all_to_all`
          - fused MoE dispatch maps used by the external-topk path

        It rewrites `self.model_state_leaves` so subsequent forward passes use the
        new expert placement. Intended for manual TPU testing.
        """
        if not getattr(self.server_args, "enable_eplb", False):
            return

        from sgl_jax.srt.eplb.metadata import ExpertLocationMetadata
        from sgl_jax.srt.eplb.weight_rebalance import (
            build_rebalance_all_to_all_device_plan,
            compute_rebalance_sources,
            rebalance_weights_all_to_all,
        )
        from sgl_jax.srt.layers.moe import FusedEPMoE

        if not isinstance(new_metadata, ExpertLocationMetadata):
            raise TypeError(
                f"Expected new_metadata to be ExpertLocationMetadata, got {type(new_metadata)}."
            )

        ep_size = int(self.mesh.shape.get("data", 1) * self.mesh.shape.get("tensor", 1))
        if new_metadata.ep_size != ep_size:
            raise ValueError(
                f"EPLB metadata ep_size mismatch: {new_metadata.ep_size} vs {ep_size}."
            )

        if self._eplb_metadata is None:
            # Assume initial fused MoE weights are laid out by logical id.
            if new_metadata.num_physical_experts != new_metadata.num_logical_experts:
                raise ValueError(
                    "Online rebalance for redundant experts requires initial EPLB metadata at startup "
                    "(expected ModelRunner._eplb_metadata to be set)."
                )
            p2l = np.tile(
                np.arange(new_metadata.num_physical_experts, dtype=np.int32)[None, :],
                (new_metadata.num_layers, 1),
            )
            dispatch = np.tile(
                np.arange(new_metadata.num_physical_experts, dtype=np.int32)[None, :, None],
                (new_metadata.num_layers, 1, ep_size),
            )
            self._eplb_metadata = ExpertLocationMetadata(
                ep_size=ep_size,
                physical_to_logical_map=p2l,
                logical_to_rank_dispatch_physical_map=dispatch,
            )

        old_metadata = self._eplb_metadata
        if old_metadata.physical_to_logical_map.shape != new_metadata.physical_to_logical_map.shape:
            raise ValueError(
                "EPLB metadata shape change is not supported online (would require recompilation): "
                f"{old_metadata.physical_to_logical_map.shape} vs {new_metadata.physical_to_logical_map.shape}."
            )

        src_for_dst = compute_rebalance_sources(
            old_physical_to_logical_map=old_metadata.physical_to_logical_map,
            new_physical_to_logical_map=new_metadata.physical_to_logical_map,
            ep_size=ep_size,
        )

        model_state = jax.tree_util.tree_unflatten(self._model_state_def, self.model_state_leaves)
        model = nnx.merge(self._model_def, model_state)

        devices = self.mesh.devices.flatten()
        expert_mesh = jax.sharding.Mesh(devices.reshape((ep_size,)), axis_names=("expert",))

        def _device_put_plan(plan_arr: np.ndarray) -> jax.Array:
            return jax.device_put(
                jnp.asarray(plan_arr),
                NamedSharding(expert_mesh, P("expert", None)),
            )

        def rebalance_param(
            param_value: jax.Array,
            *,
            send_src_local_indices_by_rank: jax.Array,
            send_sizes_by_rank: jax.Array,
            input_offsets_by_rank: jax.Array,
            output_offsets_by_rank: jax.Array,
            recv_sizes_by_rank: jax.Array,
            recv_dst_local_indices_by_rank: jax.Array,
        ) -> jax.Array:
            original_sharding = getattr(param_value, "sharding", None)
            target_expert_sharding = NamedSharding(
                expert_mesh, P("expert", *([None] * (param_value.ndim - 1)))
            )
            value_expert = jax.device_put(param_value, target_expert_sharding)

            in_specs = (
                P("expert", *([None] * (param_value.ndim - 1))),
                P("expert", None),
                P("expert", None),
                P("expert", None),
                P("expert", None),
                P("expert", None),
                P("expert", None),
            )
            out_specs = P("expert", *([None] * (param_value.ndim - 1)))

            def _do_rebalance(
                weights_local,
                send_src_local_indices,
                send_sizes,
                input_offsets,
                output_offsets,
                recv_sizes,
                recv_dst_local_indices,
            ):
                return rebalance_weights_all_to_all(
                    weights_local=weights_local,
                    send_src_local_indices=send_src_local_indices,
                    input_offsets=input_offsets,
                    send_sizes=send_sizes,
                    output_offsets=output_offsets,
                    recv_sizes=recv_sizes,
                    recv_dst_local_indices=recv_dst_local_indices,
                    axis_name="expert",
                )

            do_rebalance = jax.shard_map(
                _do_rebalance,
                mesh=expert_mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )
            out_expert = do_rebalance(
                value_expert,
                send_src_local_indices_by_rank,
                send_sizes_by_rank,
                input_offsets_by_rank,
                output_offsets_by_rank,
                recv_sizes_by_rank,
                recv_dst_local_indices_by_rank,
            )
            if original_sharding is None:
                return out_expert
            return jax.device_put(out_expert, original_sharding)

        visited = set()

        def walk(obj):
            oid = id(obj)
            if oid in visited:
                return
            visited.add(oid)

            if isinstance(obj, FusedEPMoE):
                layer_id = int(getattr(obj, "layer_id", -1))
                if not (0 <= layer_id < new_metadata.num_layers):
                    return
                if not getattr(obj, "enable_eplb", False):
                    return
                if getattr(obj, "use_grouped_topk", False):
                    # The grouped-topk fused path currently consumes dense gating output and
                    # assumes expert ids index directly into the weights tensor.
                    return
                if int(obj.num_experts) != int(new_metadata.num_physical_experts):
                    raise ValueError(
                        "FusedEPMoE num_experts mismatch vs EPLB metadata; redundancy/shape changes "
                        "require model re-init. "
                        f"{obj.num_experts=} vs {new_metadata.num_physical_experts=}."
                    )

                layer_src = src_for_dst[layer_id].astype(np.int32)
                device_plan = build_rebalance_all_to_all_device_plan(
                    src_for_dst_physical=layer_src,
                    ep_size=ep_size,
                )
                send_src_by_rank = _device_put_plan(device_plan.send_src_local_indices)
                send_sizes_by_rank = _device_put_plan(device_plan.send_sizes)
                input_offsets_by_rank = _device_put_plan(device_plan.input_offsets)
                output_offsets_by_rank = _device_put_plan(device_plan.output_offsets)
                recv_sizes_by_rank = _device_put_plan(device_plan.recv_sizes)
                recv_dst_by_rank = _device_put_plan(device_plan.recv_dst_local_indices)

                obj.w1.value = rebalance_param(
                    obj.w1.value,
                    send_src_local_indices_by_rank=send_src_by_rank,
                    send_sizes_by_rank=send_sizes_by_rank,
                    input_offsets_by_rank=input_offsets_by_rank,
                    output_offsets_by_rank=output_offsets_by_rank,
                    recv_sizes_by_rank=recv_sizes_by_rank,
                    recv_dst_local_indices_by_rank=recv_dst_by_rank,
                )
                obj.w2.value = rebalance_param(
                    obj.w2.value,
                    send_src_local_indices_by_rank=send_src_by_rank,
                    send_sizes_by_rank=send_sizes_by_rank,
                    input_offsets_by_rank=input_offsets_by_rank,
                    output_offsets_by_rank=output_offsets_by_rank,
                    recv_sizes_by_rank=recv_sizes_by_rank,
                    recv_dst_local_indices_by_rank=recv_dst_by_rank,
                )
                obj.w3.value = rebalance_param(
                    obj.w3.value,
                    send_src_local_indices_by_rank=send_src_by_rank,
                    send_sizes_by_rank=send_sizes_by_rank,
                    input_offsets_by_rank=input_offsets_by_rank,
                    output_offsets_by_rank=output_offsets_by_rank,
                    recv_sizes_by_rank=recv_sizes_by_rank,
                    recv_dst_local_indices_by_rank=recv_dst_by_rank,
                )

                dispatch = new_metadata.logical_to_rank_dispatch_physical_map[layer_id].astype(
                    np.int32
                )
                obj.eplb_dispatch_map.value = jnp.asarray(dispatch)
                return

            if isinstance(obj, (list, tuple)):
                for x in obj:
                    walk(x)
                return
            if isinstance(obj, dict):
                for x in obj.values():
                    walk(x)
                return
            if hasattr(obj, "__dict__"):
                for x in obj.__dict__.values():
                    walk(x)

        walk(model)

        _, new_state = nnx.split(model)
        self.model_state_leaves, _ = jax.tree_util.tree_flatten(new_state)
        self._eplb_metadata = new_metadata

    def get_available_device_memory(self):
        distributed = jax.process_count() != 1
        min_available_device_memory = get_available_device_memory(
            self.device, distributed=distributed
        )

        # Check memory for tensor parallelism
        local_device_memory = get_available_device_memory(self.device)
        if self.tp_size > 1 and min_available_device_memory < local_device_memory * 0.9:
            if get_bool_env_var("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"):
                logger.warning(
                    "The memory capacity is unbalanced. min_available_device_memory=%s, local_device_memory=%s, local_device_memory*0.9=%s",
                    min_available_device_memory,
                    local_device_memory,
                    local_device_memory * 0.9,
                )
            else:
                raise ValueError(
                    f"The memory capacity is unbalanced. min_available_device_memory={min_available_device_memory}, local_device_memory={local_device_memory}, local_device_memory*0.9={local_device_memory * 0.9}"
                )

        return min_available_device_memory

    def load_model(self):
        self.model_config.validate_tensor_parallel_config(self.tp_size)
        self.model_config.configure_for_tensor_parallel(self.tp_size)
        self.model_config.log_kv_heads_info(self.tp_size)
        self.model_config.hf_config.ep_size = self.ep_size
        self.model_config.hf_config.moe_backend = self.model_config.moe_backend.value
        self.model_config.hf_config.enable_return_routed_experts = (
            self.server_args.enable_return_routed_experts
        )
        self.model_config.hf_config.enable_eplb = self.server_args.enable_eplb
        self.model_config.hf_config.eplb_window_size = self.server_args.eplb_window_size
        self.model_config.hf_config.eplb_update_interval = self.server_args.eplb_update_interval
        self.model_config.hf_config.eplb_redundant_experts = self.server_args.eplb_redundant_experts
        self.model_config.hf_config.eplb_max_redundant_experts = (
            self.server_args.eplb_max_redundant_experts
        )
        self.model_config.hf_config.eplb_seed = self.server_args.eplb_seed

        if self.server_args.enable_eplb and self.model_config.moe_backend.value == "fused":
            # Redundant experts (E_physical = E_logical + R) must be chosen before model init
            # to keep expert-weight shapes stable across online rebalances.
            num_logical_experts = None
            for attr in ("num_experts", "n_experts", "num_local_experts"):
                v = getattr(self.model_config.hf_config, attr, None)
                if v is not None:
                    num_logical_experts = int(v)
                    break

            num_layers = int(
                getattr(
                    self.model_config.hf_config,
                    "num_hidden_layers",
                    getattr(self.model_config, "num_hidden_layers", 0),
                )
            )
            if num_layers <= 0:
                num_layers = int(self.model_config.num_hidden_layers)

            if num_logical_experts is None:
                logger.warning(
                    "EPLB enabled for fused MoE, but model config does not expose num_experts; "
                    "disabling redundant experts."
                )
            else:
                from sgl_jax.srt.eplb.algorithm import rebalance_experts_greedy
                from sgl_jax.srt.eplb.metadata import choose_num_physical_experts

                num_physical_experts, actual_r = choose_num_physical_experts(
                    num_logical_experts=num_logical_experts,
                    ep_size=int(self.ep_size),
                    requested_num_redundant_experts=int(self.server_args.eplb_redundant_experts),
                    max_num_redundant_experts=int(self.server_args.eplb_max_redundant_experts),
                )
                if actual_r != int(self.server_args.eplb_redundant_experts):
                    logger.warning(
                        "Adjusting eplb_redundant_experts for divisibility: requested=%d actual=%d "
                        "(E_logical=%d ep_size=%d)",
                        int(self.server_args.eplb_redundant_experts),
                        int(actual_r),
                        int(num_logical_experts),
                        int(self.ep_size),
                    )

                self.model_config.hf_config.num_physical_experts = int(num_physical_experts)
                self.model_config.hf_config.eplb_redundant_experts = int(actual_r)

                init_meta = rebalance_experts_greedy(
                    tokens_per_logical_expert=np.ones(
                        (num_layers, int(num_logical_experts)), dtype=np.float32
                    ),
                    ep_size=int(self.ep_size),
                    num_redundant_experts=int(actual_r),
                    max_num_redundant_experts=int(self.server_args.eplb_max_redundant_experts),
                    seed=int(self.server_args.eplb_seed),
                )
                self.model_config.hf_config.eplb_initial_metadata = init_meta
                self.model_config.hf_config.eplb_initial_physical_to_logical_map = (
                    init_meta.physical_to_logical_map
                )
                self.model_config.hf_config.eplb_initial_dispatch_map = (
                    init_meta.logical_to_rank_dispatch_physical_map
                )

        self.model = self.model_loader.load_model(
            model_config=self.model_config,
        )
        if (
            getattr(self.model_config.hf_config, "enable_eplb", False)
            and getattr(self.model_config.hf_config, "moe_backend", None) == "fused"
        ):
            self._materialize_eplb_dispatch_maps()
        if (
            getattr(self.model_config.hf_config, "enable_eplb", False)
            and getattr(self.model_config.hf_config, "moe_backend", None) == "fused"
        ):
            # Track the current placement for online weight rebalance.
            self._eplb_metadata = getattr(
                self.model_config.hf_config, "eplb_initial_metadata", None
            )
        if self.is_draft_worker:
            # if draft model and target model share same safetensor files, we should hack here to avoid create redundant layer kv cache
            self.model_config.num_hidden_layers = getattr(
                self.model_config, "num_nextn_predict_layers", self.model_config.num_hidden_layers
            )

        # Apply quantization if quantization config is set
        if self.model_config.quantization_config is not None:
            # Apply MoE quantization first (before QWIX, so scales are set when QWIX runs model)
            if self.model_config.quantization_config.has_moe_quantization():
                self.model = apply_moe_quantization(self.model_config, self.model)

            # Apply qwix quantization for dense layers
            qwix_rules = self.model_config.quantization_config.get_qwix_rules()
            if qwix_rules:
                self.model = apply_qwix_quantization(self.model_config, self.model, self)

        # Parse other args
        self.sliding_window_size = self.model_config.sliding_window
        self.dtype = self.model_config.dtype
        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(self.model, "end_layer", self.model_config.num_hidden_layers)
        self.num_effective_layers = self.end_layer - self.start_layer
        if self.server_args.speculative_algorithm == "EAGLE3" and not self.is_draft_worker:
            try:
                # get the aux layer from draft model config
                eagle_config = getattr(self.model_config.hf_config, "eagle_config", None)
                eagle_aux_hidden_state_layer_ids = eagle_config["eagle_aux_hidden_state_layer_ids"]
            except Exception as e:
                logger.warning("get the aux layer from draft model config %s", e)
                # if there is no aux layer, set to None
                eagle_aux_hidden_state_layer_ids = None
            self.model.set_eagle3_layers_to_capture(eagle_aux_hidden_state_layer_ids)

    def profile_max_num_token(self, total_device_memory: int):
        """
        Profile the maximum number of tokens that can fit in memory.
        Uses tpu_info to get accurate TPU memory information.
        """
        # Get accurate memory information using TPU-specific methods
        # Use tpu_info for memory information
        available_device_memory = self.get_available_device_memory()
        available_kv_cache_bytes = available_device_memory - total_device_memory * (
            1 - self.mem_fraction_static
        )

        if available_kv_cache_bytes <= 0:
            raise RuntimeError("Not enough memory. Please try to increase --mem-fraction-static.")
        head_dim_aligned = self.model_config.head_dim
        if head_dim_aligned % 128 != 0:
            head_dim_aligned = (self.model_config.head_dim + 127) // 128 * 128
        cell_size = (
            self.model_config.get_num_kv_heads(self.tp_size)
            * head_dim_aligned
            * self.model_config.num_hidden_layers
            * 2
            * jnp.dtype(self.kv_cache_dtype).itemsize
        )

        # Calculate max tokens that can fit in available memory
        max_tokens = max(1, int(available_kv_cache_bytes // cell_size))

        logger.info(
            "TPU Memory profiling: available_device_memory=%.1fGB, available_kv_cache=%.1fGB, max_tokens=%d, cell_size=%dbytes",
            available_device_memory / (1024**3),
            available_kv_cache_bytes / (1024**3),
            max_tokens,
            cell_size,
        )

        return max_tokens

    @property
    def is_hybrid_gdn(self):
        return self.model_config.hf_config.architectures[0] in [
            "Qwen3NextForCausalLM",
            "Qwen3NextForCausalLMMTP",
        ]

    def init_memory_pool(
        self,
        max_num_reqs: int | None = None,
        max_total_tokens: int | None = None,
        total_device_memory: int | None = None,
    ):
        """Initialize memory pool for KV cache."""
        # Set KV cache data type
        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "bf16":
            self.kv_cache_dtype = jnp.bfloat16
        else:
            raise ValueError(f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}.")
        logger.info("ModelRunner kv_cache_dtype: %s", self.kv_cache_dtype)
        # Profile maximum number of tokens
        self.max_total_num_tokens = self.profile_max_num_token(total_device_memory)

        # Calculate max number of requests if not provided
        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(self.max_total_num_tokens / self.model_config.context_len * 512),
                    2048,
                ),
                4096,
            )

        # Handle CI environment variable for testing
        SGLANG_CI_SMALL_KV_SIZE = os.environ.get("SGLANG_CI_SMALL_KV_SIZE")
        if SGLANG_CI_SMALL_KV_SIZE:
            self.max_total_num_tokens = int(SGLANG_CI_SMALL_KV_SIZE)

        if self.spec_algorithm is not None and not self.spec_algorithm.is_none():
            if self.is_draft_worker:
                self.max_total_num_tokens = self.server_args.draft_runner_cache_size
                max_num_reqs = self.server_args.max_num_reqs
            else:
                # We are sharing the `token_to_kv_pool`, and both verify and draft tokens
                # can be concurrently allocated, so we should give a headroom for it.
                self.server_args.draft_runner_cache_size = (
                    self.max_total_num_tokens
                    # draft
                    + max_num_reqs
                    * self.server_args.speculative_num_steps
                    * self.server_args.speculative_eagle_topk
                    # verify
                    + max_num_reqs * self.server_args.speculative_num_draft_tokens
                    # buffer
                    + 100
                )
                # Target worker and draft worker shares the same indices for the
                # token_to_kv_pool, so we should make sure to match max_total_num_tokens.
                self.max_total_num_tokens = self.server_args.draft_runner_cache_size
                self.server_args.max_num_reqs = max_num_reqs

        # Handle max_total_tokens override
        if max_total_tokens is not None:
            if max_total_tokens > self.max_total_num_tokens:
                logger.warning(
                    "max_total_tokens=%s is larger than the profiled value %s. Use the profiled value instead.",
                    max_total_tokens,
                    self.max_total_num_tokens,
                )
            self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)

        # Align to page size
        self.max_total_num_tokens = (
            self.max_total_num_tokens // self.server_args.page_size * self.server_args.page_size
        )

        # create token size for hybrid cache
        if self.is_hybrid:
            self.set_num_token_hybrid()

        if self.max_total_num_tokens <= 0:
            raise RuntimeError("Not enough memory. Please try to increase --mem-fraction-static.")

        logger.info("ModelRunner max_total_num_tokens: %s", self.max_total_num_tokens)

        # Create request to token pool if not already created
        if self.req_to_token_pool is None:
            self.req_to_token_pool = ReqToTokenPool(
                size=max_num_reqs + 1,
                max_context_len=self.model_config.context_len + 4,
                dtype=np.int32,
            )

        # Create KV cache pool
        if self.is_hybrid:
            self.token_to_kv_pool = SWAKVPool(
                size=self.full_max_total_num_tokens,
                size_swa=self.swa_max_total_num_tokens,
                swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_total_num_kv_heads_with_replication(self.tp_size),
                head_dim=self.model_config.head_dim,
                mesh=self.mesh,
            )
        else:
            self.token_to_kv_pool = MHATokenToKVPool(
                size=self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_total_num_kv_heads_with_replication(self.tp_size),
                head_dim=(self.model_config.head_dim + 127) // 128 * 128,
                layer_num=self.model_config.num_hidden_layers,
                mesh=self.mesh,
            )

        # Create KV pool allocator
        if self.token_to_kv_pool_allocator is None:
            if self.page_size == 1:
                if self.is_hybrid:
                    self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        kvcache=self.token_to_kv_pool,
                    )
                else:
                    self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                        size=self.max_total_num_tokens,
                        kvcache=self.token_to_kv_pool,
                    )
            else:
                assert not self.is_hybrid
                self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=self.page_size,
                    kvcache=self.token_to_kv_pool,
                    debug_mode=False,
                )

    def init_attention_backend(self):
        """Init attention kernel backend."""
        self.attn_backend = self._get_attention_backend()

    def _get_attention_backend(self):
        # Fallback on CPU: FlashAttention (Pallas/Triton) does not support CPU compilation and execution
        backend = self.server_args.attention_backend
        if self.server_args.device == "cpu" and backend == "fa":
            logger.warning(
                "FlashAttention backend is not supported on CPU; falling back to native."
            )
            backend = "native"
        if backend == "native":
            from sgl_jax.srt.layers.attention.native_backend import NativeAttention

            return NativeAttention(self.num_attn_heads, self.num_kv_heads, self.mesh)
        elif backend == "fa":
            from sgl_jax.srt.layers.attention.flashattention_backend import (
                FlashAttention,
            )

            return FlashAttention(
                self.num_attn_heads,
                self.num_kv_heads,
                self.model_config.head_dim,
                page_size=self.page_size,
                mesh=self.mesh,
            )
        else:
            raise ValueError(f"Unsupported attention backend: {self.server_args.attention_backend}")

    def _forward(
        self,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
    ):
        cache_miss_count = 0
        import jax._src.test_util as jtu

        with jtu.count_pjit_cpp_cache_miss() as count:
            output, layers_kv_fused, _, layers_topk_ids = self.jitted_run_model(
                forward_batch, logits_metadata
            )
            cache_miss_count = count()
        self._set_kv_cache_after_forward(layers_kv_fused)

        # layers_topk_ids required real_bs and original_input_len which could not be stored in ForwardBatch
        return output, cache_miss_count, layers_topk_ids

    def _set_kv_cache_after_forward(self, layers_kv_fused):
        # Note: For tp_size == 1, we need to put the layers_kv_fused on the device with the target_sharding
        # because sharding P(None, 'tensor') constraint has lost and this results in cache_miss for first prefill phase.
        # Issue: https://github.com/sgl-project/sglang-jax/issues/233
        # Q: Why does not call device_put in every layer?
        # A: Because it does not work and cache_miss still happens. According to benchmark(https://github.com/sgl-project/sglang-jax/pull/234), the performance is not influenced.
        if self.tp_size == 1:
            target_sharding = NamedSharding(
                self.token_to_kv_pool.mesh,
                P(None, self.token_to_kv_pool.kv_partition_axis, None),
            )
            layers_kv_fused = [
                jax.device_put(layer_kv_fused, target_sharding)
                for layer_kv_fused in layers_kv_fused
            ]

        self.token_to_kv_pool.replace_kv_buffer(layers_kv_fused)

    def forward_idle(
        self,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
    ) -> tuple[LogitsProcessorOutput, int]:
        raise NotImplementedError("forward_idle is not implemented")

    def forward(
        self,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
    ) -> tuple[LogitsProcessorOutput, int]:
        self.forward_pass_id += 1
        precision_tracer.start_batch_trace(forward_batch.bid)
        precision_tracer.set_current_forward_pass_id(self.forward_pass_id)
        with jax.profiler.TraceAnnotation("_forward_raw"):
            ret = self._forward_raw(forward_batch, logits_metadata)
        return ret

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
    ) -> tuple[LogitsProcessorOutput, int]:
        # for compatibility, 0.6.3 need to use use_mesh. set_mesh is not have __entry__ attribute.
        # on jax >=0.7.1, we need to use set_mesh.
        try:
            ctx = jax.sharding.use_mesh(self.mesh)
        except AttributeError:
            try:
                ctx = jax.set_mesh(self.mesh)
            except AttributeError:
                ctx = self.mesh
        with ctx:
            if forward_batch.forward_mode.is_decode() or forward_batch.forward_mode.is_extend():
                ret = self._forward(forward_batch, logits_metadata)
            elif forward_batch.forward_mode.is_idle():
                ret = self.forward_idle(forward_batch, logits_metadata)
            else:
                raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

        return ret

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_metadata: SamplingMetadata,
    ) -> jax.Array:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
            positions: The positions of the tokens in the sequence.
        Returns:
            A list of next_token_ids
        """
        # Penalty application has been moved to the Sampler for better JIT performance
        return self.jitted_sampler(
            logits_output,
            sampling_metadata,
        )

    def compute_logprobs(self, logits, token_ids: jax.Array) -> jax.Array:
        return self.jitted_compute_logprobs(logits, token_ids)

    def set_num_token_hybrid(self):
        assert self.sliding_window_size is not None and self.sliding_window_size > 0
        full_attention_layer_ids = []
        swa_attention_layer_ids = []

        # Try different attribute paths to access model layers
        layers = None
        layer_access_attempts = [
            lambda: self.model.model.layers,
            lambda: self.model.language_model.model.layers,
            lambda: self.model.language_model.layers,
            lambda: self.model.transformer.layers,
        ]
        for get_layers in layer_access_attempts:
            try:
                layers = get_layers()
                break
            except AttributeError:
                continue

        if layers is None:
            self.is_hybrid = False
            return

        for layer in layers:
            if (
                layer.self_attn.attn.sliding_window_size is None
                or layer.self_attn.attn.sliding_window_size == -1
            ):
                full_attention_layer_ids.append(layer.layer_id)
            else:
                swa_attention_layer_ids.append(layer.layer_id)

        self.model_config.swa_attention_layer_ids = swa_attention_layer_ids
        self.model_config.full_attention_layer_ids = full_attention_layer_ids

        # Algorithm:
        # Existing max_total_num_tokens is per layer and assume all layers have the same number of tokens.
        # - Find total # of tokens available across layers.
        # - Calculate full_max_total_num_tokens and swa_max_total_num_tokens based on the given swa_full_tokens_ratio.
        total_tokens = self.max_total_num_tokens * self.model_config.num_hidden_layers
        full_layers_num = len(full_attention_layer_ids)
        swa_layers_num = len(swa_attention_layer_ids)
        swa_full_tokens_ratio = self.server_args.swa_full_tokens_ratio

        # Solve the equations:
        # 1. swa_max_total_num_tokens * swa_layers_num + full_max_total_num_tokens * full_layers_num == total_tokens
        # 2. full_max_total_num_tokens * swa_full_tokens_ratio == swa_max_total_num_tokens
        denominator = swa_full_tokens_ratio * swa_layers_num + full_layers_num
        self.full_max_total_num_tokens = int(total_tokens / denominator)
        self.swa_max_total_num_tokens = int(self.full_max_total_num_tokens * swa_full_tokens_ratio)
        self.max_total_num_tokens = self.full_max_total_num_tokens

        logger.info(
            "Use Sliding window memory pool. full_layer_tokens=%s, swa_layer_tokens=%s",
            self.full_max_total_num_tokens,
            self.swa_max_total_num_tokens,
        )

    def init_lora_manager(self):
        """Initialize LoRA manager for LoRA adapter support."""
        from sgl_jax.srt.lora.lora_manager import LoRAManager

        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            dtype=self.dtype,
            mesh=self.mesh,
            max_lora_rank=self.server_args.max_lora_rank,
            target_modules=self.server_args.lora_target_modules,
            lora_paths=self.server_args.lora_paths,
            server_args=self.server_args,
            model_config=self.model_config,
        )


class MockModelRunner(ModelRunner):
    def __init__(
        self,
        model_config: ModelConfig | MockModelConfig,
        rngs: nnx.Rngs = None,
        mesh: mesh_lib.Mesh = None,
        server_args: ServerArgs = None,
    ):
        self.server_args = server_args
        self.tp_size = server_args.tp_size

        if isinstance(model_config, MockModelConfig):
            self.num_kv_heads = model_config.num_kv_heads
            self.num_attn_heads = model_config.num_heads
            self.rngs = rngs
        else:
            self.num_kv_heads = model_config.get_total_num_kv_heads_with_replication(self.tp_size)
            self.num_attn_heads = model_config.num_attention_heads
            self.rngs = rngs

        self.dtype = jnp.float32
        self.mem_fraction_static = 0.8
        self.model_config = model_config
        self.max_total_num_tokens = 1 << 15
        self.kv_cache_dtype = jnp.bfloat16
        self.page_size = 1
        self.mesh = mesh

        # Validate tensor parallel configuration for MockModelRunner too
        if not isinstance(model_config, MockModelConfig):
            self.model_config.validate_tensor_parallel_config(self.tp_size)

        # If it is a draft model, tp_group can be different
        max_num_reqs = min(
            max(
                int(self.max_total_num_tokens / self.model_config.context_len * 512),
                2048,
            ),
            4096,
        )
        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs + 1,
            max_context_len=self.model_config.context_len + 4,
            dtype=np.int32,
        )

        self.token_to_kv_pool = MHATokenToKVPool(
            size=self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_total_num_kv_heads_with_replication(self.tp_size),
            head_dim=(self.model_config.head_dim + 127) // 128 * 128,
            layer_num=self.model_config.num_hidden_layers,
            mesh=mesh,
        )
