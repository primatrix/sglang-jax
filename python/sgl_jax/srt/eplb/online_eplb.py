import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

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


def _permute_weight_via_host(param, perm):
    """Permute weight along axis 0 via host roundtrip.

    Uses a hybrid strategy: when most device shards change, a single bulk
    device_get/device_put is faster than per-shard transfers. When few
    shards change, only the affected shards are transferred.
    """
    arr = param.value
    sharding = arr.sharding
    addressable_shards = sorted(arr.addressable_shards, key=lambda s: s.index[0].start)
    num_devices = len(addressable_shards)
    local_experts = arr.shape[0] // num_devices

    changed = set()
    for d in range(num_devices):
        s = d * local_experts
        if not np.array_equal(perm[s : s + local_experts], np.arange(s, s + local_experts)):
            changed.add(d)

    if not changed:
        return 0

    if len(changed) > num_devices // 2:
        arr_np = np.array(arr)
        arr_np = arr_np[perm]
        param.value = jax.device_put(arr_np, sharding)
        return len(changed)

    needed_src = set()
    for d in changed:
        s = d * local_experts
        for i in range(local_experts):
            needed_src.add(int(perm[s + i]) // local_experts)

    with ThreadPoolExecutor(max_workers=len(needed_src)) as pool:
        src_data = dict(
            pool.map(lambda idx: (idx, np.array(addressable_shards[idx].data)), needed_src)
        )

    per_device = []
    for d in range(num_devices):
        if d not in changed:
            per_device.append(addressable_shards[d].data)
        else:
            s = d * local_experts
            ref = next(iter(src_data.values()))
            new_shard = np.empty_like(ref)
            for i in range(local_experts):
                src = int(perm[s + i])
                new_shard[i] = src_data[src // local_experts][src % local_experts]
            per_device.append(jax.device_put(new_shard, addressable_shards[d].device))

    param.value = jax.make_array_from_single_device_arrays(arr.shape, sharding, per_device)
    return len(changed)


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
        self._prev_logical_counts: np.ndarray | None = None

        self._moe_layers: list = []
        _collect_moe_layers(self.model_runner.model, self._moe_layers)
        self._moe_layers.sort(key=lambda m: m.layer_id if m.layer_id is not None else 0)

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

        if os.environ.get("EPLB_NOOP"):
            logger.info("EPLB_NOOP: overriding with identity mapping for rebuild test")
            new_p2l = old_p2l.copy()
            new_l2p = np.array(jax.device_get(old_metadata.logical_to_all_physical_map))

        changed_mask = old_p2l != new_p2l
        num_changed = int(changed_mask.sum())
        total_slots = old_p2l.size
        change_ratio = num_changed / total_slots if total_slots > 0 else 0.0

        t1 = time.perf_counter()
        self.model_runner.model_state_leaves = []
        if os.environ.get("EPLB_HOST_PERMUTE"):
            global _on_device_permute_supported
            _on_device_permute_supported = False
            logger.info("EPLB_HOST_PERMUTE: forcing host-based weight permutation")
        self._apply_weight_permutation(old_p2l, new_p2l, changed_mask)
        t_weights = time.perf_counter() - t1

        self._verify_weights_after_permutation(old_p2l, new_p2l, changed_mask)

        new_metadata = ExpertLocationMetadata._init_raw(
            server_args=self.server_args,
            physical_to_logical_map=new_p2l,
            logical_to_all_physical_map=new_l2p,
            mesh=self.model_runner.mesh,
        )
        set_global_expert_location_metadata(new_metadata)

        _, model_state = nnx.split(self.model_runner.model)
        self.model_runner.model_state_leaves, new_treedef = jax.tree_util.tree_flatten(model_state)
        if new_treedef != self._reference_treedef:
            logger.error(
                "TREEDEF MISMATCH after rebalance! leaves=%d",
                len(self.model_runner.model_state_leaves),
            )
        else:
            logger.info("Treedef OK: %d leaves", len(self.model_runner.model_state_leaves))

        self._verify_leaves_match_model()

        self._verify_metadata(new_metadata, new_p2l)

        self._rebalance_count += 1
        self._prev_logical_counts = logical_counts.copy()
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

        work_items = []
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

            n_unique = len(np.unique(perm))
            if n_unique != num_physical:
                logger.error(
                    "INVALID PERM layer=%d: %d unique (expected %d), dupes exist",
                    layer_id,
                    n_unique,
                    num_physical,
                )
            missing_logical = set(range(self._num_logical_experts)) - set(
                old_logical_to_physical.keys()
            )
            if missing_logical:
                logger.error(
                    "Layer %d: %d logical experts missing from old mapping: %s",
                    layer_id,
                    len(missing_logical),
                    sorted(missing_logical)[:5],
                )

            if isinstance(moe_layer, FusedEPMoE):
                weights = [moe_layer.w1, moe_layer.w2, moe_layer.w3]
                scales = [moe_layer.w1_scale, moe_layer.w2_scale, moe_layer.w3_scale]
            else:
                weights = [moe_layer.wi_0, moe_layer.wi_1, moe_layer.wo]
                scales = [moe_layer.wi_0_scale, moe_layer.wi_1_scale, moe_layer.wo_scale]

            for p in weights:
                work_items.append((p, perm))
            for p in scales:
                if p is not None:
                    work_items.append((p, perm))

        if not work_items:
            logger.info("Weight permutation: no work items")
            return

        global _on_device_permute_supported
        if _on_device_permute_supported is not False:
            first_p, first_perm = work_items[0]
            first_perm_jax = jnp.array(first_perm, dtype=jnp.int32)
            if _permute_weight_on_device(first_p, first_perm_jax, first_p.value.sharding):
                for p, perm in work_items[1:]:
                    perm_jax = jnp.array(perm, dtype=jnp.int32)
                    _permute_weight_on_device(p, perm_jax, p.value.sharding)
                logger.info("Weight permutation (on-device): %d tensors", len(work_items))
                return

        max_workers = min(12, len(work_items))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(lambda item: _permute_weight_via_host(*item), work_items))

        total_changed = sum(results)
        logger.info(
            "Weight permutation (host): %d tensors, %d shard transfers (workers=%d)",
            len(work_items),
            total_changed,
            max_workers,
        )

    def _verify_metadata(self, metadata: ExpertLocationMetadata, p2l: np.ndarray):
        l2p = jax.device_get(metadata.logical_to_rank_dispatch_physical_map)
        p2l_check = jax.device_get(metadata.physical_to_logical_map)
        num_valid = jax.device_get(metadata.logical_to_all_physical_map_num_valid)

        num_layers, num_logical = l2p.shape
        sample_layers = list(range(0, num_layers, max(1, num_layers // 5)))
        errors = 0
        zero_valid = 0
        for layer_id in sample_layers:
            for lid in range(num_logical):
                pid = int(l2p[layer_id, lid])
                if pid < 0 or pid >= p2l_check.shape[1]:
                    errors += 1
                    if errors <= 3:
                        logger.error(
                            "Metadata: layer=%d logical=%d -> invalid physical=%d",
                            layer_id,
                            lid,
                            pid,
                        )
                    continue
                roundtrip = int(p2l_check[layer_id, pid])
                if roundtrip != lid:
                    errors += 1
                    if errors <= 3:
                        logger.error(
                            "Metadata roundtrip FAIL: layer=%d logical=%d -> phys=%d -> logical=%d",
                            layer_id,
                            lid,
                            pid,
                            roundtrip,
                        )
                nv = int(num_valid[layer_id, lid])
                if nv == 0:
                    zero_valid += 1
        if errors:
            logger.error(
                "Metadata verification: %d errors across %d layers", errors, len(sample_layers)
            )
        else:
            logger.info(
                "Metadata verification OK: %d layers × %d experts checked",
                len(sample_layers),
                num_logical,
            )
        if zero_valid:
            logger.warning("Metadata: %d (layer, expert) pairs have num_valid=0", zero_valid)

    def _get_first_weight(self, layer):
        if isinstance(layer, FusedEPMoE):
            return layer.w1
        return layer.wi_0

    def _get_first_scale(self, layer):
        if isinstance(layer, FusedEPMoE):
            return layer.w1_scale
        return layer.wi_0_scale

    def _verify_weights_after_permutation(
        self, old_p2l: np.ndarray, new_p2l: np.ndarray, changed_mask: np.ndarray
    ):
        layer = self._moe_layers[0]
        lid = layer.layer_id
        if lid is None:
            return

        w_param = self._get_first_weight(layer)
        w1_np = np.array(jax.device_get(w_param.value))
        logger.info(
            "Post-perm w1 shape=%s dtype=%s, slot0_sum=%.4f, slot1_sum=%.4f",
            w1_np.shape,
            w1_np.dtype,
            float(np.sum(np.abs(w1_np[0].astype(np.float32)))),
            float(np.sum(np.abs(w1_np[1].astype(np.float32)))),
        )

        s_param = self._get_first_scale(layer)
        if s_param is not None:
            s_np = np.array(jax.device_get(s_param.value))
            logger.info(
                "Post-perm w1_scale shape=%s dtype=%s, slot0_sum=%.6f",
                s_np.shape,
                s_np.dtype,
                float(np.sum(np.abs(s_np[0].astype(np.float32)))),
            )

        all_same = True
        for p in range(min(10, w1_np.shape[0])):
            fp = float(np.sum(np.abs(w1_np[p].astype(np.float32))))
            if fp == 0.0:
                logger.warning("Post-perm w1[%d] is all zeros!", p)
                all_same = False
        if all_same:
            logger.info("Post-perm w1 first 10 slots: no zeros detected")

    def _verify_leaves_match_model(self):
        layer = self._moe_layers[0]
        model_w1 = self._get_first_weight(layer).value
        model_w1_fp = float(jnp.sum(jnp.abs(model_w1[0].astype(jnp.float32))))

        found = False
        for i, leaf in enumerate(self.model_runner.model_state_leaves):
            if leaf.shape == model_w1.shape and leaf.dtype == model_w1.dtype:
                leaf_fp = float(jnp.sum(jnp.abs(leaf[0].astype(jnp.float32))))
                if abs(leaf_fp - model_w1_fp) < 1e-3:
                    is_same = leaf is model_w1
                    logger.info(
                        "Leaf match: idx=%d, same_obj=%s, fp=%.4f",
                        i,
                        is_same,
                        leaf_fp,
                    )
                    found = True
                    break
        if not found:
            logger.error("Could not find w1 in model_state_leaves!")
