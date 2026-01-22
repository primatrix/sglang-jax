from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def counts_from_topk_ids(
    topk_ids: np.ndarray, *, num_logical_experts: int, top_k: int | None = None
) -> np.ndarray:
    """Compute per-expert counts from logical-space top-k ids.

    Args:
      topk_ids: (num_tokens, top_k) int array of logical expert ids.
      num_logical_experts: total logical experts (bincount length).
      top_k: if provided, only use the first `top_k` columns.
    """
    if num_logical_experts <= 0:
        raise ValueError(f"Expected {num_logical_experts=} to be > 0.")
    ids = np.asarray(topk_ids)
    if ids.ndim != 2:
        raise ValueError(f"Expected topk_ids to be 2D, got {ids.ndim=}")
    if top_k is not None:
        ids = ids[:, : int(top_k)]
    if ids.size == 0:
        return np.zeros((num_logical_experts,), dtype=np.int64)
    flat = ids.reshape(-1)
    if np.any(flat < 0) or np.any(flat >= num_logical_experts):
        raise ValueError("topk_ids contains out-of-range logical expert ids")
    return np.bincount(flat.astype(np.int64), minlength=num_logical_experts).astype(np.int64)


@dataclass
class EplbRecordDump:
    logical_count: np.ndarray  # (num_layers, num_logical_experts) int64


class EplbStatsRecorder:
    """Sliding-window recorder for logical-space expert routing statistics (CPU-side).

    This is intentionally numpy-first so it can be driven by either:
      - CPU-copied topk ids (e.g., captured), or
      - occasional host-side topk computation for measurement.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_logical_experts: int,
        num_experts_per_tok: int,
        window_size: int,
    ):
        if num_layers <= 0:
            raise ValueError(f"Expected {num_layers=} to be > 0.")
        if num_logical_experts <= 0:
            raise ValueError(f"Expected {num_logical_experts=} to be > 0.")
        if num_experts_per_tok <= 0:
            raise ValueError(f"Expected {num_experts_per_tok=} to be > 0.")
        if window_size <= 0:
            raise ValueError(f"Expected {window_size=} to be > 0.")

        self.num_layers = int(num_layers)
        self.num_logical_experts = int(num_logical_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.window_size = int(window_size)

        self._ring = np.zeros(
            (self.window_size, self.num_layers, self.num_logical_experts),
            dtype=np.int64,
        )
        self._pos = 0
        self._filled = 0

    @property
    def filled(self) -> int:
        return int(self._filled)

    def reset(self) -> None:
        self._ring[...] = 0
        self._pos = 0
        self._filled = 0

    def record_step(self, *, topk_ids_by_layer: list[np.ndarray | None]) -> None:
        if len(topk_ids_by_layer) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {len(topk_ids_by_layer)}.")

        # Evict old counts at this slot, then fill new counts.
        self._ring[self._pos, :, :] = 0
        for layer, ids in enumerate(topk_ids_by_layer):
            if ids is None:
                continue
            self._ring[self._pos, layer, :] = counts_from_topk_ids(
                ids,
                num_logical_experts=self.num_logical_experts,
                top_k=self.num_experts_per_tok,
            )

        self._pos = (self._pos + 1) % self.window_size
        self._filled = min(self.window_size, self._filled + 1)

    def dump(self) -> EplbRecordDump:
        logical_count = np.sum(self._ring[: self._filled], axis=0)
        return EplbRecordDump(logical_count=logical_count)
