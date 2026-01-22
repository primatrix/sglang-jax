from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def choose_num_physical_experts(
    *,
    num_logical_experts: int,
    ep_size: int,
    requested_num_redundant_experts: int,
    max_num_redundant_experts: int = 128,
) -> tuple[int, int]:
    """Choose (num_physical_experts, num_redundant_experts) under EPLB constraints.

    Constraints:
      - 0 <= num_redundant_experts <= max_num_redundant_experts
      - num_physical_experts = num_logical_experts + num_redundant_experts
      - num_physical_experts % ep_size == 0

    Notes:
      - We adjust the requested redundant count downward until divisibility holds.
      - For stable compilation, callers should keep num_physical_experts fixed across
        rebalances (i.e., choose once at startup).
    """
    if num_logical_experts <= 0:
        raise ValueError(f"Expected {num_logical_experts=} to be > 0.")
    if ep_size <= 0:
        raise ValueError(f"Expected {ep_size=} to be > 0.")
    if max_num_redundant_experts < 0:
        raise ValueError(f"Expected {max_num_redundant_experts=} to be >= 0.")
    if requested_num_redundant_experts < 0:
        raise ValueError(f"Expected {requested_num_redundant_experts=} to be >= 0.")

    r = min(int(requested_num_redundant_experts), int(max_num_redundant_experts))
    while r >= 0:
        e_physical = num_logical_experts + r
        if e_physical % ep_size == 0:
            return e_physical, r
        r -= 1

    raise ValueError(
        "Unable to choose a divisible num_physical_experts. "
        f"Try changing {requested_num_redundant_experts=} or {ep_size=}."
    )


@dataclass(frozen=True)
class ExpertLocationMetadata:
    """EPLB placement metadata (numpy-first, broadcast-friendly).

    Shapes:
      - physical_to_logical_map: (num_layers, num_physical_experts)
      - logical_to_rank_dispatch_physical_map: (num_layers, num_logical_experts, ep_size)
    """

    ep_size: int
    physical_to_logical_map: np.ndarray
    logical_to_rank_dispatch_physical_map: np.ndarray

    def validate(self) -> None:
        if self.ep_size <= 0:
            raise ValueError(f"Expected {self.ep_size=} to be > 0.")

        if not isinstance(self.physical_to_logical_map, np.ndarray):
            raise TypeError("physical_to_logical_map must be a numpy array")
        if not isinstance(self.logical_to_rank_dispatch_physical_map, np.ndarray):
            raise TypeError("logical_to_rank_dispatch_physical_map must be a numpy array")

        if self.physical_to_logical_map.ndim != 2:
            raise ValueError(
                f"Expected physical_to_logical_map to have ndim=2, got {self.physical_to_logical_map.ndim}"
            )
        if self.logical_to_rank_dispatch_physical_map.ndim != 3:
            raise ValueError(
                "Expected logical_to_rank_dispatch_physical_map to have ndim=3, got "
                f"{self.logical_to_rank_dispatch_physical_map.ndim}"
            )

        num_layers, num_physical_experts = self.physical_to_logical_map.shape
        dl, num_logical_experts, ep_size = self.logical_to_rank_dispatch_physical_map.shape
        if dl != num_layers:
            raise ValueError(f"Layer mismatch: {dl=} vs {num_layers=}.")
        if ep_size != self.ep_size:
            raise ValueError(f"EP mismatch: {ep_size=} vs {self.ep_size=}.")
        if num_physical_experts % self.ep_size != 0:
            raise ValueError(
                f"Expected {num_physical_experts=} to be divisible by {self.ep_size=}."
            )

        if self.physical_to_logical_map.dtype.kind not in ("i", "u"):
            raise TypeError("physical_to_logical_map must be an integer array")
        if self.logical_to_rank_dispatch_physical_map.dtype.kind not in ("i", "u"):
            raise TypeError("logical_to_rank_dispatch_physical_map must be an integer array")

        # Bounds checks.
        if np.any(self.physical_to_logical_map < 0) or np.any(
            self.physical_to_logical_map >= num_logical_experts
        ):
            raise ValueError("physical_to_logical_map contains out-of-range logical ids")
        if np.any(self.logical_to_rank_dispatch_physical_map < 0) or np.any(
            self.logical_to_rank_dispatch_physical_map >= num_physical_experts
        ):
            raise ValueError(
                "logical_to_rank_dispatch_physical_map contains out-of-range physical ids"
            )

        # Coverage: every logical expert must appear at least once per layer.
        for layer in range(num_layers):
            counts = np.bincount(self.physical_to_logical_map[layer], minlength=num_logical_experts)
            if np.any(counts == 0):
                missing = np.where(counts == 0)[0].tolist()
                raise ValueError(f"Missing logical experts in layer={layer}: {missing[:8]}...")

        # Dispatch correctness: selected physical id must map back to the intended logical id.
        for layer in range(num_layers):
            p2l = self.physical_to_logical_map[layer]
            selected = self.logical_to_rank_dispatch_physical_map[layer]
            back = p2l[selected]
            expected = np.arange(num_logical_experts, dtype=back.dtype)[:, None]
            expected = np.broadcast_to(expected, back.shape)
            if not np.array_equal(back, expected):
                raise ValueError(f"Dispatch map mismatch in layer={layer}")

    @property
    def num_layers(self) -> int:
        return int(self.physical_to_logical_map.shape[0])

    @property
    def num_physical_experts(self) -> int:
        return int(self.physical_to_logical_map.shape[1])

    @property
    def num_logical_experts(self) -> int:
        return int(self.logical_to_rank_dispatch_physical_map.shape[1])

    @property
    def num_local_physical_experts(self) -> int:
        return self.num_physical_experts // self.ep_size
