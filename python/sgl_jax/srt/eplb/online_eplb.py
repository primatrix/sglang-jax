import logging
import time
from dataclasses import dataclass

import jax
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.eplb import eplb_algorithms
from sgl_jax.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_global_expert_location_metadata,
    get_num_experts_from_config,
    set_global_expert_location_metadata,
)
from sgl_jax.srt.layers.fused_moe import FusedEPMoE
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.layers.routed_experts_capturer import _ExpertDistributionRecorder
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_MOE_LAYER_TYPES = (FusedEPMoE, EPMoE)

_on_device_permute_supported: bool | None = None
_permute_fn_cache: dict = {}


def _get_permute_fn(sharding):
    key = (id(sharding.mesh), sharding.spec)
    if key not in _permute_fn_cache:

        @jax.jit
        def fn(arr, perm):
            return arr.at[perm].get(out_sharding=sharding)

        _permute_fn_cache[key] = fn
    return _permute_fn_cache[key]


def _permute_weight_on_device(param, perm_jax, sharding):
    global _on_device_permute_supported
    if _on_device_permute_supported is False:
        return False

    try:
        fn = _get_permute_fn(sharding)
        param.value = fn(param.value, perm_jax)
        if _on_device_permute_supported is None:
            _on_device_permute_supported = True
            logger.info("On-device weight permutation: AVAILABLE")
        return True
    except Exception as e:
        if _on_device_permute_supported is None:
            _on_device_permute_supported = False
            logger.info("On-device weight permutation: NOT available (%s)", e)
        return False


def _collect_moe_layers(module, result: list, visited: set | None = None):
    if visited is None:
        visited = set()
    obj_id = id(module)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if isinstance(module, _MOE_LAYER_TYPES):
        result.append(module)
        return

    if hasattr(module, "__dict__"):
        for attr_value in module.__dict__.values():
            if isinstance(attr_value, nnx.Module):
                _collect_moe_layers(attr_value, result, visited)
            elif isinstance(attr_value, (list, tuple)):
                for item in attr_value:
                    if isinstance(item, nnx.Module):
                        _collect_moe_layers(item, result, visited)


def _permute_weight_slab_diff(param, perm):
    """Permute weight along axis 0 using per-device slab diff.

    Classifies each device's local slab as unchanged (zero-copy reuse) or
    changed (reconstruct from source shards with cross-device dedup).
    """
    arr = param.value
    sharding = arr.sharding
    shards = sorted(arr.addressable_shards, key=lambda s: s.index[0].start)
    num_devices = len(shards)
    local_experts = arr.shape[0] // num_devices

    per_device_arrays = []
    stats = {"unchanged": 0, "changed": 0}

    for d in range(num_devices):
        slab_start = d * local_experts
        local_perm = perm[slab_start : slab_start + local_experts]
        identity = np.arange(slab_start, slab_start + local_experts)

        if np.array_equal(local_perm, identity):
            per_device_arrays.append(shards[d].data)
            stats["unchanged"] += 1
            continue

        stats["changed"] += 1
        fetched_shards: dict[int, np.ndarray] = {}
        new_slab = np.empty((local_experts, *arr.shape[1:]), dtype=arr.dtype)

        for i in range(local_experts):
            src = int(local_perm[i])
            src_dev = src // local_experts
            src_off = src % local_experts

            if src_dev not in fetched_shards:
                fetched_shards[src_dev] = np.array(shards[src_dev].data)
            new_slab[i] = fetched_shards[src_dev][src_off]

        per_device_arrays.append(jax.device_put(new_slab, shards[d].device))

    param.value = jax.make_array_from_single_device_arrays(arr.shape, sharding, per_device_arrays)
    return stats


@dataclass
class _RebalancePlan:
    old_p2l: np.ndarray
    new_p2l: np.ndarray
    new_l2p: np.ndarray
    layer_chunks: list[list[int]]
    current_chunk_idx: int = 0
    t_start: float = 0.0
    num_changed_total: int = 0
    total_slots: int = 0
    logical_counts: np.ndarray | None = None
    t_algo_ms: float = 0.0


class OnlineEPLBController:
    def __init__(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_runner,
        dist_recorder: _ExpertDistributionRecorder,
    ):
        self.server_args = server_args
        self.model_config = model_config
        self.model_runner = model_runner
        self.dist_recorder = dist_recorder
        self._mesh = model_runner.mesh

        self.interval_steps = server_args.online_eplb_interval_steps
        self.min_samples = server_args.online_eplb_min_samples
        self.diff_threshold = server_args.online_eplb_diff_threshold

        if server_args.nnodes > 1 and server_args.online_eplb_layers_per_chunk > 0:
            logger.info(
                "Multi-host detected (nnodes=%d): forcing layers_per_chunk=0 "
                "to avoid cross-host JIT desync between chunks",
                server_args.nnodes,
            )
            self.layers_per_chunk = 0
        else:
            self.layers_per_chunk = server_args.online_eplb_layers_per_chunk

        self._step_counter = 0
        self._rebalance_count = 0
        self._prev_logical_counts: np.ndarray | None = None
        self._pending_plan: _RebalancePlan | None = None

        self._moe_layers: list = []
        _collect_moe_layers(self.model_runner.model, self._moe_layers)
        self._moe_layers.sort(key=lambda m: m.layer_id if m.layer_id is not None else 0)
        self._moe_layer_by_id: dict[int, FusedEPMoE | EPMoE] = {
            m.layer_id: m for m in self._moe_layers if m.layer_id is not None
        }

        self._reference_treedef = None
        _, init_state = nnx.split(self.model_runner.model)
        _, self._reference_treedef = jax.tree_util.tree_flatten(init_state)

        hf_config = model_config.hf_config
        self._num_logical_experts = get_num_experts_from_config(hf_config)
        self._num_groups = getattr(hf_config, "num_expert_group", 1)

        metadata = get_global_expert_location_metadata()
        self._num_physical_experts = metadata.num_physical_experts if metadata else 0
        self._ep_size = server_args.ep_size

        logger.info(
            "OnlineEPLBController initialized: interval=%d, min_samples=%d, "
            "diff_threshold=%.3f, layers_per_chunk=%d, moe_layers=%d, "
            "physical=%d, logical=%d",
            self.interval_steps,
            self.min_samples,
            self.diff_threshold,
            self.layers_per_chunk,
            len(self._moe_layers),
            self._num_physical_experts,
            self._num_logical_experts,
        )

    def precompile(self):
        if not self._moe_layers:
            return
        moe_layer = self._moe_layers[0]
        if isinstance(moe_layer, FusedEPMoE):
            params = [moe_layer.w1, moe_layer.w2, moe_layer.w3]
            scale_params = [moe_layer.w1_scale, moe_layer.w2_scale, moe_layer.w3_scale]
        else:
            params = [moe_layer.wi_0, moe_layer.wi_1, moe_layer.wo]
            scale_params = [moe_layer.wi_0_scale, moe_layer.wi_1_scale, moe_layer.wo_scale]

        all_params = params + [p for p in scale_params if p is not None]
        seen_specs = set()
        num_physical = all_params[0].value.shape[0]
        perm_sharding = NamedSharding(self._mesh, P())
        identity_perm = jax.device_put(np.arange(num_physical, dtype=np.int32), perm_sharding)

        t0 = time.perf_counter()
        for param in all_params:
            sharding = param.value.sharding
            spec_key = (id(sharding.mesh), sharding.spec)
            if spec_key in seen_specs:
                continue
            seen_specs.add(spec_key)
            fn = _get_permute_fn(sharding)
            result = fn(param.value, identity_perm)
            result.block_until_ready()
            param.value = fn(param.value, identity_perm)
            param.value.block_until_ready()

        logger.info(
            "Online EPLB precompile: warmed up %d permute fn(s) in %.1fms",
            len(seen_specs),
            (time.perf_counter() - t0) * 1000,
        )

    @property
    def is_rebalancing(self) -> bool:
        return self._pending_plan is not None

    def maybe_rebalance(self) -> bool:
        if self._pending_plan is not None:
            return self._execute_next_chunk()

        self._step_counter += 1
        if self._step_counter < self.interval_steps:
            return False
        self._step_counter = 0

        result = self.dist_recorder.get_logical_counts_and_reset()
        if result is None:
            return False
        logical_counts, steps = result

        if steps < self.min_samples:
            return False

        old_metadata = get_global_expert_location_metadata()
        if old_metadata is None:
            return False

        old_p2l = jax.device_get(old_metadata.physical_to_logical_map)

        if self._prev_logical_counts is not None:
            old_norm = self._prev_logical_counts / (
                self._prev_logical_counts.sum(axis=-1, keepdims=True) + 1e-12
            )
            new_norm = logical_counts / (logical_counts.sum(axis=-1, keepdims=True) + 1e-12)
            dist_diff = float(np.abs(old_norm - new_norm).mean())
            if dist_diff < self.diff_threshold:
                logger.info(
                    "Online EPLB: skipping rebalance, dist_diff=%.6f < threshold=%.4f",
                    dist_diff,
                    self.diff_threshold,
                )
                return False

        t0 = time.perf_counter()
        new_p2l, new_l2p, _ = eplb_algorithms.rebalance_experts(
            tokens_per_expert=logical_counts,
            num_physical_experts=self._num_physical_experts,
            num_local_physical_experts=self._num_physical_experts // self._ep_size,
            num_groups=self._num_groups,
            num_nodes=self.server_args.nnodes,
            algorithm=eplb_algorithms.compute_algorithm(
                raw_algorithm=getattr(self.server_args, "eplb_algorithm", "auto"),
                num_groups=self._num_groups,
                num_nodes=self.server_args.nnodes,
            ),
        )
        t_algo = time.perf_counter() - t0

        changed_mask = old_p2l != new_p2l
        num_changed = int(changed_mask.sum())
        total_slots = old_p2l.size

        changed_layer_ids = [
            m.layer_id
            for m in self._moe_layers
            if m.layer_id is not None
            and m.layer_id < old_p2l.shape[0]
            and changed_mask[m.layer_id].any()
        ]

        if not changed_layer_ids:
            logger.info("Online EPLB: no layers changed, skipping")
            self._prev_logical_counts = logical_counts.copy()
            return False

        layer_chunks = self._split_into_chunks(changed_layer_ids)
        plan = _RebalancePlan(
            old_p2l=old_p2l,
            new_p2l=new_p2l,
            new_l2p=new_l2p,
            layer_chunks=layer_chunks,
            t_start=time.perf_counter(),
            num_changed_total=num_changed,
            total_slots=total_slots,
            logical_counts=logical_counts,
            t_algo_ms=t_algo * 1000,
        )

        logger.info(
            "Online EPLB: starting rebalance, changed=%d/%d (%.1f%%), "
            "algo=%.1fms, %d chunks (%d changed layers)",
            num_changed,
            total_slots,
            num_changed / total_slots * 100 if total_slots else 0,
            t_algo * 1000,
            len(layer_chunks),
            len(changed_layer_ids),
        )

        self._pending_plan = plan

        if self.layers_per_chunk == 0:
            while self._pending_plan is not None:
                self._execute_next_chunk()
        else:
            self._execute_next_chunk()

        return True

    def _split_into_chunks(self, layer_ids: list[int]) -> list[list[int]]:
        if self.layers_per_chunk <= 0:
            return [layer_ids]
        return [
            layer_ids[i : i + self.layers_per_chunk]
            for i in range(0, len(layer_ids), self.layers_per_chunk)
        ]

    def _execute_next_chunk(self) -> bool:
        plan = self._pending_plan
        chunk_layer_ids = plan.layer_chunks[plan.current_chunk_idx]

        t0 = time.perf_counter()

        for layer_id in chunk_layer_ids:
            perm = self._compute_layer_perm(plan.old_p2l[layer_id], plan.new_p2l[layer_id])
            self._permute_layer_weights(layer_id, perm)

        t_weights = time.perf_counter()

        self._update_metadata_for_layers(chunk_layer_ids, plan)

        t_metadata = time.perf_counter()

        _, model_state = nnx.split(self.model_runner.model)
        self.model_runner.model_state_leaves, new_treedef = jax.tree_util.tree_flatten(model_state)
        if new_treedef != self._reference_treedef:
            logger.error(
                "TREEDEF MISMATCH after chunk %d! leaves=%d",
                plan.current_chunk_idx,
                len(self.model_runner.model_state_leaves),
            )

        t_chunk = (time.perf_counter() - t0) * 1000
        t_w = (t_weights - t0) * 1000
        t_m = (t_metadata - t_weights) * 1000
        t_s = t_chunk - t_w - t_m
        logger.info(
            "Online EPLB chunk %d/%d: %d layers, %.1fms "
            "(weights=%.1fms, metadata=%.1fms, split=%.1fms)",
            plan.current_chunk_idx + 1,
            len(plan.layer_chunks),
            len(chunk_layer_ids),
            t_chunk,
            t_w,
            t_m,
            t_s,
        )

        plan.current_chunk_idx += 1
        if plan.current_chunk_idx >= len(plan.layer_chunks):
            self._finalize_rebalance(plan)
            self._pending_plan = None

        return True

    def _compute_layer_perm(
        self, old_layer_p2l: np.ndarray, new_layer_p2l: np.ndarray
    ) -> np.ndarray:
        num_physical = len(old_layer_p2l)

        old_logical_to_physical = {}
        for p_idx in range(num_physical):
            l_idx = int(old_layer_p2l[p_idx])
            if l_idx not in old_logical_to_physical:
                old_logical_to_physical[l_idx] = p_idx

        perm = np.arange(num_physical, dtype=np.int32)
        changed = old_layer_p2l != new_layer_p2l
        for dst in range(num_physical):
            if not changed[dst]:
                continue
            target_logical = int(new_layer_p2l[dst])
            src = old_logical_to_physical.get(target_logical, dst)
            perm[dst] = src

        return perm

    def _permute_layer_weights(self, layer_id: int, perm: np.ndarray):
        moe_layer = self._moe_layer_by_id.get(layer_id)
        if moe_layer is None:
            return

        if isinstance(moe_layer, FusedEPMoE):
            params = [moe_layer.w1, moe_layer.w2, moe_layer.w3]
            scale_params = [moe_layer.w1_scale, moe_layer.w2_scale, moe_layer.w3_scale]
        else:
            params = [moe_layer.wi_0, moe_layer.wi_1, moe_layer.wo]
            scale_params = [moe_layer.wi_0_scale, moe_layer.wi_1_scale, moe_layer.wo_scale]

        all_params = params + [p for p in scale_params if p is not None]
        perm_sharding = NamedSharding(self._mesh, P())
        perm_jax = jax.device_put(perm, perm_sharding)

        for param in all_params:
            sharding = param.value.sharding
            if not _permute_weight_on_device(param, perm_jax, sharding):
                _permute_weight_slab_diff(param, perm)

    def _update_metadata_for_layers(self, layer_ids: list[int], plan: _RebalancePlan):
        old_metadata = get_global_expert_location_metadata()

        p2l = np.array(jax.device_get(old_metadata.physical_to_logical_map))
        l2p = np.array(jax.device_get(old_metadata.logical_to_all_physical_map))

        for lid in layer_ids:
            p2l[lid] = plan.new_p2l[lid]
            l2p[lid] = plan.new_l2p[lid]

        new_metadata = ExpertLocationMetadata._init_raw(
            server_args=self.server_args,
            physical_to_logical_map=p2l,
            logical_to_all_physical_map=l2p,
            mesh=self.model_runner.mesh,
        )
        set_global_expert_location_metadata(new_metadata)

    def _finalize_rebalance(self, plan: _RebalancePlan):
        self._rebalance_count += 1
        self._prev_logical_counts = plan.logical_counts.copy()
        t_total = (time.perf_counter() - plan.t_start) * 1000

        logger.info(
            "Online EPLB rebalance #%d complete: changed=%d/%d (%.1f%%), "
            "algo=%.1fms, total=%.1fms (%d chunks)",
            self._rebalance_count,
            plan.num_changed_total,
            plan.total_slots,
            plan.num_changed_total / plan.total_slots * 100 if plan.total_slots else 0,
            plan.t_algo_ms,
            t_total,
            len(plan.layer_chunks),
        )
