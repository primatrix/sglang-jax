"""Kernel-local GLM-5.2 DSA dimensions and microbenchmark cases.

This host-only manifest deliberately excludes model mesh and TP bucket sizes.
Every case is runnable on one TPU device and isolates one attention head.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Glm52DsaContract:
    latent_dim: int = 512
    rope_dim: int = 64
    page_size: int = 128
    packing: int = 2
    index_topk: int = 2048
    index_heads: int = 32
    index_head_dim: int = 128
    attention_scale: float = 256**-0.5

    @property
    def cache_width(self) -> int:
        latent_aligned = ((self.latent_dim + 127) // 128) * 128
        rope_aligned = ((self.rope_dim + 127) // 128) * 128
        return latent_aligned + rope_aligned


GLM52_DSA_CONTRACT = Glm52DsaContract()


@dataclass(frozen=True)
class SparseMlaPerfCase:
    """One single-device sparse-MLA workload presented to the operator."""

    name: str
    mode: str
    query_rows: int
    context_length: int
    top_k: int = GLM52_DSA_CONTRACT.index_topk
    num_heads: int = 1
    latent_dim: int = GLM52_DSA_CONTRACT.latent_dim
    rope_dim: int = GLM52_DSA_CONTRACT.rope_dim
    page_size: int = GLM52_DSA_CONTRACT.page_size
    start_position: int | None = None
    slot_order: str = "unsorted"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("performance case name must not be empty")
        if self.mode not in {"decode", "prefill"}:
            raise ValueError("mode must be 'decode' or 'prefill'")
        if self.query_rows <= 0 or self.context_length <= 0 or self.top_k <= 0:
            raise ValueError("query_rows, context_length, and top_k must be positive")
        if self.top_k > GLM52_DSA_CONTRACT.index_topk:
            raise ValueError("top_k must not exceed the GLM-5.2 K_max")
        if self.num_heads != 1:
            raise ValueError("microbenchmark cases isolate one independent head")
        production_features = (
            GLM52_DSA_CONTRACT.latent_dim,
            GLM52_DSA_CONTRACT.rope_dim,
            GLM52_DSA_CONTRACT.page_size,
        )
        if (self.latent_dim, self.rope_dim, self.page_size) != production_features:
            raise ValueError("performance cases must use production feature dimensions")
        if self.page_size % GLM52_DSA_CONTRACT.packing:
            raise ValueError("page_size must be divisible by packing")
        if self.slot_order not in {"unsorted", "page-sorted"}:
            raise ValueError("slot_order must be 'unsorted' or 'page-sorted'")
        if self.mode == "decode" and self.start_position is not None:
            raise ValueError("decode cases must not set start_position")
        if self.mode == "prefill":
            if self.start_position is None or self.start_position < 0:
                raise ValueError("prefill cases require a nonnegative start_position")
            if self.context_length < self.start_position + self.query_rows:
                raise ValueError("prefill context_length must cover every query row")

    @property
    def shape_tuple(self) -> tuple[str, int, int, int, int]:
        return (
            self.mode,
            self.query_rows,
            self.num_heads,
            self.context_length,
            self.top_k,
        )

    @property
    def valid_count_pattern(self) -> str:
        return "causal" if self.mode == "prefill" else "full"


PERFORMANCE_CASES = (
    SparseMlaPerfCase(
        name="debug-q1-h1-c128-k128",
        mode="decode",
        query_rows=1,
        context_length=128,
        top_k=128,
    ),
    SparseMlaPerfCase(
        name="decode-q1-h1-c8192-k2048",
        mode="decode",
        query_rows=1,
        context_length=8192,
    ),
    SparseMlaPerfCase(
        name="prefill-q128-h1-start2048-k2048",
        mode="prefill",
        query_rows=128,
        context_length=2176,
        start_position=2048,
    ),
)


PERFORMANCE_CASES_BY_NAME = {case.name: case for case in PERFORMANCE_CASES}

if len(PERFORMANCE_CASES_BY_NAME) != len(PERFORMANCE_CASES):
    raise ValueError("GLM-5.2 DSA performance case names must be unique")
