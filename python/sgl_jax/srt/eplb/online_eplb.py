import functools
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.eplb import eplb_algorithms
from sgl_jax.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_global_expert_location_metadata,
    get_num_experts_from_config,
    set_global_expert_location_metadata,
)
from sgl_jax.srt.layers.fused_moe import FusedEPMoE
from sgl_jax.srt.layers.routed_experts_capturer import _ExpertDistributionRecorder
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _collect_moe_layers(module, result: list, visited: set | None = None):
    if visited is None:
        visited = set()
    obj_id = id(module)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if isinstance(module, FusedEPMoE):
        result.append(module)
        return

    if hasattr(module, "__dict__"):
        for attr_value in module.__dict__.values():
            if isinstance(attr_value, nnx.Module):
                _collect_moe_layers(attr_value, result, visited)
            elif isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, nnx.Module):
                        _collect_moe_layers(item, result, visited)


@functools.partial(jax.jit, donate_argnums=(0,))
def _permute_weight(w, perm):
    return w[perm]


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

        self.interval_steps = server_args.online_eplb_interval_steps
        self.min_samples = server_args.online_eplb_min_samples
        self.diff_threshold = server_args.online_eplb_diff_threshold

        self._step_counter = 0
        self._rebalance_count = 0

        self._moe_layers: list[FusedEPMoE] = []
        _collect_moe_layers(self.model_runner.model, self._moe_layers)
        self._moe_layers.sort(key=lambda m: m.layer_id if m.layer_id is not None else 0)

        hf_config = model_config.hf_config
        self._num_logical_experts = get_num_experts_from_config(hf_config)
        self._num_groups = getattr(hf_config, "num_expert_group", 1)

        metadata = get_global_expert_location_metadata()
        self._num_physical_experts = metadata.num_physical_experts if metadata else 0
        self._ep_size = server_args.ep_size

        logger.info(
            "OnlineEPLBController initialized: interval=%d, min_samples=%d, "
            "diff_threshold=%.3f, moe_layers=%d, physical=%d, logical=%d",
            self.interval_steps,
            self.min_samples,
            self.diff_threshold,
            len(self._moe_layers),
            self._num_physical_experts,
            self._num_logical_experts,
        )

    def maybe_rebalance(self) -> bool:
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
        change_ratio = num_changed / total_slots if total_slots > 0 else 0.0

        if change_ratio < self.diff_threshold:
            logger.debug(
                "Online EPLB: skipping rebalance, change_ratio=%.4f < threshold=%.4f",
                change_ratio,
                self.diff_threshold,
            )
            return False

        t1 = time.perf_counter()
        self._apply_weight_permutation(old_p2l, new_p2l, changed_mask)
        t_weights = time.perf_counter() - t1

        new_metadata = ExpertLocationMetadata._init_raw(
            server_args=self.server_args,
            physical_to_logical_map=new_p2l,
            logical_to_all_physical_map=new_l2p,
        )
        set_global_expert_location_metadata(new_metadata)

        _, model_state = nnx.split(self.model_runner.model)
        self.model_runner.model_state_leaves, _ = jax.tree_util.tree_flatten(model_state)

        self._rebalance_count += 1
        logger.info(
            "Online EPLB rebalance #%d: changed=%d/%d (%.1f%%), "
            "algo=%.1fms, weights=%.1fms, samples=%d steps",
            self._rebalance_count,
            num_changed,
            total_slots,
            change_ratio * 100,
            t_algo * 1000,
            t_weights * 1000,
            steps,
        )
        return True

    def _apply_weight_permutation(
        self,
        old_p2l: np.ndarray,
        new_p2l: np.ndarray,
        changed_mask: np.ndarray,
    ):
        num_layers = old_p2l.shape[0]
        num_physical = old_p2l.shape[1]

        for moe_layer in self._moe_layers:
            layer_id = moe_layer.layer_id
            if layer_id is None or layer_id >= num_layers:
                continue
            if not changed_mask[layer_id].any():
                continue

            old_logical_to_physical = {}
            for p_idx in range(num_physical):
                l_idx = int(old_p2l[layer_id, p_idx])
                if l_idx not in old_logical_to_physical:
                    old_logical_to_physical[l_idx] = p_idx

            perm = np.arange(num_physical, dtype=np.int32)
            for dst in range(num_physical):
                if not changed_mask[layer_id, dst]:
                    continue
                target_logical = int(new_p2l[layer_id, dst])
                src = old_logical_to_physical.get(target_logical, dst)
                perm[dst] = src

            perm_jax = jnp.array(perm)
            moe_layer.w1.value = _permute_weight(moe_layer.w1.value, perm_jax)
            moe_layer.w2.value = _permute_weight(moe_layer.w2.value, perm_jax)
            moe_layer.w3.value = _permute_weight(moe_layer.w3.value, perm_jax)

            if moe_layer.w1_scale is not None:
                moe_layer.w1_scale.value = _permute_weight(moe_layer.w1_scale.value, perm_jax)
            if moe_layer.w2_scale is not None:
                moe_layer.w2_scale.value = _permute_weight(moe_layer.w2_scale.value, perm_jax)
            if moe_layer.w3_scale is not None:
                moe_layer.w3_scale.value = _permute_weight(moe_layer.w3_scale.value, perm_jax)
