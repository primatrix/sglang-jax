# Adapted from https://github.com/sgl-project/sglang/pull/12162/files#diff-8e61cb3c05ca6a5e195f011e21ea7544f9f7e08163e3ce4ffa0bacb4b5735259.
# Copyright 2025 The SGLang Authors. All rights reserved.
# Note:
# 1. Remove _RoutedExpertsDeviceCache and _RoutedExpertsHostCache due to at.set error
# 2. The following codes are modified to Jax version according to SGLang codes.

import csv
import datetime
import logging
import os
import threading
import time
from abc import ABC, abstractmethod

import jax
import numpy as np
import pybase64

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_array_size_bytes(t: np.ndarray):
    return np.prod(t.shape) * t.dtype.itemsize


class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_tokens: int,
        max_padding: int,
        *,
        enable_balance_debug: bool = False,
        balance_segment_tokens: int = 100,
        balance_output_file: str | None = None,
    ):
        if enable or enable_balance_debug:
            return _RoutedExpertsCapturerReal(
                model_config,
                num_tokens=num_tokens,
                max_padding=max_padding,
                enable_host_buffer=enable,
                enable_balance_debug=enable_balance_debug,
                balance_segment_tokens=balance_segment_tokens,
                balance_output_file=balance_output_file,
            )
        else:
            return _RoutedExpertsCapturerNoop()

    @abstractmethod
    def _sync_fwd_experts_buffer_DtoH(
        self,
        topk_ids_cpu: list[np.ndarray],
        model_worker_batch: ModelWorkerBatch,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        bid: int,
    ):
        raise NotImplementedError

    @abstractmethod
    def on_forward_end(self, topk_ids: list[jax.Array], model_worker_batch: ModelWorkerBatch):
        raise NotImplementedError


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer"""

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_padding: int,
        *,
        enable_host_buffer: bool,
        enable_balance_debug: bool,
        balance_segment_tokens: int,
        balance_output_file: str | None,
    ):
        self.enable_host_buffer = enable_host_buffer
        self.num_hidden_layers = model_config.hf_text_config.num_hidden_layers
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok
        self.num_tokens = num_tokens
        self.max_padding = max_padding

        if self.enable_host_buffer:
            self.host_buffer = np.zeros(
                (
                    self.num_hidden_layers,
                    self.num_tokens,
                    self.num_experts_per_tok,
                ),
                dtype=np.int32,
            )
            # Note: self.dummy_expert_ids is used to models whose some of layers are not MoE, like inclusionAI/Ling-mini-2.0
            self.dummy_experts_ids = np.full(
                (self.max_padding, self.num_experts_per_tok), fill_value=-1, dtype=np.int32
            )
        else:
            self.host_buffer = None
            self.dummy_experts_ids = None

        self._balance_analyzer = None
        if enable_balance_debug:
            num_experts = getattr(model_config.hf_text_config, "num_experts", None)
            if num_experts is None:
                num_experts = getattr(model_config.hf_text_config, "num_local_experts", None)
            if num_experts is None:
                logger.warning(
                    "Expert balance debug enabled, but num_experts is missing in model config. "
                    "Disabling expert balance debug."
                )
            elif not balance_output_file:
                logger.warning(
                    "Expert balance debug enabled, but output file is not set. "
                    "Disabling expert balance debug."
                )
            else:
                self._balance_analyzer = _ExpertBalanceAnalyzer(
                    num_layers=self.num_hidden_layers,
                    num_experts=num_experts,
                    topk=self.num_experts_per_tok,
                    segment_tokens=balance_segment_tokens,
                    output_file=balance_output_file,
                )
        self._balance_missing_topk_warned = False
        self.bid = None

        """Common logging and memory usage computation for captured experts buffers."""
        if self.enable_host_buffer:
            buffer_size_GB = self.get_buffer_size_bytes() / _GB
            logger.info(
                "Routing experts host buffer allocated. #tokens: %d, size: %.2f GB",
                self.num_tokens,
                buffer_size_GB,
            )
        if self._balance_analyzer is not None:
            logger.info(
                "Expert balance debug enabled. Segment tokens: %d, output: %s",
                self._balance_analyzer.segment_tokens,
                self._balance_analyzer.output_file,
            )

    def get_buffer_size_bytes(self):
        if self.host_buffer is None:
            return 0
        return get_array_size_bytes(self.host_buffer)

    def _sync_fwd_experts_buffer_DtoH(
        self,
        topk_ids_cpu: list[np.ndarray],  # padded topk_ids
        model_worker_batch: ModelWorkerBatch,
    ):
        if not self.enable_host_buffer:
            return
        unpadded_input_len = model_worker_batch.get_original_input_len()
        valid_out_cache_loc_cpu = model_worker_batch.out_cache_loc[:unpadded_input_len]
        for layer_idx, ids_cpu in enumerate(topk_ids_cpu):
            if ids_cpu is None:
                valid_ids = self.dummy_experts_ids[:unpadded_input_len]
            else:
                valid_ids = ids_cpu[:unpadded_input_len, : self.num_experts_per_tok]
            self.host_buffer[layer_idx, valid_out_cache_loc_cpu, :] = valid_ids

        self.bid = model_worker_batch.bid

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        bid: int,
    ):
        if not self.enable_host_buffer:
            raise RuntimeError("Host buffer is disabled. enable_return_routed_experts is required.")
        cache_pool_idx = req_to_token_pool.req_to_token[req_pool_idx][: seqlen - 1]
        while True:
            if self.bid >= bid:
                return self.host_buffer[:, cache_pool_idx, :]
            else:
                time.sleep(0.001)

    def on_forward_end(self, topk_ids: list[jax.Array], model_worker_batch: ModelWorkerBatch):
        if not self.enable_host_buffer and self._balance_analyzer is None:
            return
        topk_ids_cpu = jax.device_get(topk_ids)
        if self.enable_host_buffer:
            self._sync_fwd_experts_buffer_DtoH(
                topk_ids_cpu=topk_ids_cpu,
                model_worker_batch=model_worker_batch,
            )
        if self._balance_analyzer is not None:
            unpadded_input_len = model_worker_batch.get_original_input_len()
            if topk_ids_cpu and all(ids is None for ids in topk_ids_cpu):
                if not self._balance_missing_topk_warned:
                    logger.warning(
                        "Expert balance debug is enabled, but topk_ids are None. "
                        "This usually means fused MoE is enabled; no expert balance stats will be recorded."
                    )
                    self._balance_missing_topk_warned = True
                return
            self._balance_analyzer.add_topk_ids(topk_ids_cpu, unpadded_input_len)


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def _sync_fwd_experts_buffer_DtoH(
        self,
        topk_ids_cpu: list[np.ndarray],
        model_worker_batch: ModelWorkerBatch,
    ):
        pass

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        bid: int,
    ):
        pass

    def on_forward_end(self, topk_ids: list[jax.Array], model_worker_batch: ModelWorkerBatch):
        pass


_global_expert_capturer: RoutedExpertsCapturer | None = _RoutedExpertsCapturerNoop()


def get_global_experts_capturer():
    return _global_expert_capturer


def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer


def extract_routed_experts_from_meta_info(data):
    # To solve the performance issue, we return the experts_ids in base64
    # We left this function for user to change it back to normal int32
    # See detokenizer_manager::_extract_routed_experts
    routed_experts_base64 = data["meta_info"].get("routed_experts", None)
    if routed_experts_base64 is not None:
        routed_experts = np.frombuffer(
            pybase64.b64decode(routed_experts_base64.encode("utf-8")), dtype=np.int32
        )
        return routed_experts
    return None


class _ExpertBalanceAnalyzer:
    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        topk: int,
        segment_tokens: int,
        output_file: str,
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.topk = topk
        self.segment_tokens = segment_tokens
        self.output_file = output_file

        self._counts = np.zeros((num_layers, num_experts), dtype=np.int64)
        self._segment_token_count = 0
        self._segment_idx = 0
        self._lock = threading.Lock()

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self._file = open(output_file, "w", encoding="utf-8", newline="")  # noqa: SIM115
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "timestamp",
                "segment_idx",
                "segment_tokens",
                "layer",
                "num_experts",
                "topk",
                "total_assignments",
                "active_experts",
                "mean_count",
                "std_count",
                "cv",
                "min_count",
                "max_count",
                "entropy",
                "gini",
                "hot_topk_mean_multiple",
                "cold_topk_mean_multiple",
                "has_data",
            ]
        )
        self._file.flush()

    def add_topk_ids(self, topk_ids_cpu: list[np.ndarray], num_tokens: int):
        if num_tokens <= 0:
            return
        offset = 0
        with self._lock:
            while offset < num_tokens:
                take = min(num_tokens - offset, self.segment_tokens - self._segment_token_count)
                if take <= 0:
                    break
                for layer_idx, ids_cpu in enumerate(topk_ids_cpu):
                    if ids_cpu is None:
                        continue
                    ids_chunk = ids_cpu[offset : offset + take, : self.topk]
                    if ids_chunk.size == 0:
                        continue
                    flat = ids_chunk.reshape(-1)
                    if flat.size == 0:
                        continue
                    valid = flat[flat >= 0]
                    if valid.size == 0:
                        continue
                    counts = np.bincount(valid, minlength=self.num_experts)
                    self._counts[layer_idx] += counts

                self._segment_token_count += take
                offset += take

                if self._segment_token_count >= self.segment_tokens:
                    self._flush_segment()
                    self._counts.fill(0)
                    self._segment_token_count = 0
                    self._segment_idx += 1

    def _flush_segment(self):
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        for layer_idx in range(self.num_layers):
            counts = self._counts[layer_idx]
            total = int(counts.sum())
            has_data = 1 if total > 0 else 0
            if total > 0:
                mean = total / self.num_experts
                std = float(np.std(counts))
                cv = float(std / mean) if mean > 0 else 0.0
                min_count = int(counts.min())
                max_count = int(counts.max())
                active = int(np.count_nonzero(counts))
                probs = counts.astype(np.float64) / float(total)
                probs = probs[probs > 0]
                entropy = float(-(probs * np.log(probs)).sum())
                gini = float(_gini_from_counts(counts))
                k = min(self.topk, self.num_experts)
                if k <= 0 or mean <= 0:
                    hot_multiple = 0.0
                    cold_multiple = 0.0
                else:
                    hot = np.partition(counts, -k)[-k:]
                    cold = np.partition(counts, k - 1)[:k]
                    hot_multiple = float(hot.mean() / mean)
                    cold_multiple = float(cold.mean() / mean)
            else:
                mean = std = cv = entropy = gini = 0.0
                hot_multiple = cold_multiple = 0.0
                min_count = max_count = active = 0
            self._writer.writerow(
                [
                    timestamp,
                    self._segment_idx,
                    self.segment_tokens,
                    layer_idx,
                    self.num_experts,
                    self.topk,
                    total,
                    active,
                    f"{mean:.6f}",
                    f"{std:.6f}",
                    f"{cv:.6f}",
                    min_count,
                    max_count,
                    f"{entropy:.6f}",
                    f"{gini:.6f}",
                    f"{hot_multiple:.6f}",
                    f"{cold_multiple:.6f}",
                    has_data,
                ]
            )
        self._file.flush()


def _gini_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    sorted_counts = np.sort(counts.astype(np.float64))
    n = sorted_counts.size
    cumulative = np.cumsum(sorted_counts)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(max(0.0, min(1.0, gini)))
