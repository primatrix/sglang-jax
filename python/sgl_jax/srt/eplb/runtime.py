from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .algorithm import rebalance_experts_greedy
from .metadata import ExpertLocationMetadata
from .recorder import EplbRecordDump, EplbStatsRecorder


@dataclass(frozen=True)
class EplbUpdate:
    metadata: ExpertLocationMetadata
    record_dump: EplbRecordDump


class EplbController:
    """CPU-side EPLB controller: records routing stats and periodically recomputes placement."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_logical_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        window_size: int,
        update_interval: int,
        num_redundant_experts: int,
        max_num_redundant_experts: int = 128,
        seed: int = 0,
    ):
        if ep_size <= 0:
            raise ValueError(f"Expected {ep_size=} to be > 0.")
        if update_interval <= 0:
            raise ValueError(f"Expected {update_interval=} to be > 0.")

        self.ep_size = int(ep_size)
        self.update_interval = int(update_interval)
        self.num_redundant_experts = int(num_redundant_experts)
        self.max_num_redundant_experts = int(max_num_redundant_experts)
        self.seed = int(seed)

        self.recorder = EplbStatsRecorder(
            num_layers=num_layers,
            num_logical_experts=num_logical_experts,
            num_experts_per_tok=num_experts_per_tok,
            window_size=window_size,
        )

        self._steps = 0
        self._last_metadata: ExpertLocationMetadata | None = None

    @property
    def steps(self) -> int:
        return int(self._steps)

    @property
    def last_metadata(self) -> ExpertLocationMetadata | None:
        return self._last_metadata

    def record_step(self, *, topk_ids_by_layer: list[np.ndarray | None]) -> None:
        self.recorder.record_step(topk_ids_by_layer=topk_ids_by_layer)
        self._steps += 1

    def maybe_rebalance(self) -> EplbUpdate | None:
        if self._steps % self.update_interval != 0:
            return None
        if self.recorder.filled < self.recorder.window_size:
            return None

        record_dump = self.recorder.dump()
        tokens_per_logical = record_dump.logical_count.astype(np.float32)

        metadata = rebalance_experts_greedy(
            tokens_per_logical_expert=tokens_per_logical,
            ep_size=self.ep_size,
            num_redundant_experts=self.num_redundant_experts,
            max_num_redundant_experts=self.max_num_redundant_experts,
            seed=self.seed,
        )
        self._last_metadata = metadata
        return EplbUpdate(metadata=metadata, record_dump=record_dump)
