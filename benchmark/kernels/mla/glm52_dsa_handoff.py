"""Frozen GLM-5.2 DSA kernel dimensions and performance scenarios.

This module is intentionally host-only: operator implementations can consume
the manifest without importing JAX or allocating a model checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Glm52DsaContract:
    total_query_heads: int = 64
    tensor_parallel_size: int = 32
    latent_dim: int = 512
    rope_dim: int = 64
    page_size: int = 128
    packing: int = 2
    index_topk: int = 2048
    index_heads: int = 32
    index_head_dim: int = 128
    attention_scale: float = 256**-0.5

    @property
    def local_query_heads(self) -> int:
        return self.total_query_heads // self.tensor_parallel_size

    @property
    def cache_width(self) -> int:
        latent_aligned = ((self.latent_dim + 127) // 128) * 128
        rope_aligned = ((self.rope_dim + 127) // 128) * 128
        return latent_aligned + rope_aligned


GLM52_DSA_CONTRACT = Glm52DsaContract()


@dataclass(frozen=True)
class SparseMlaPerfCase:
    """One precompiled sparse-MLA workload presented to the operator."""

    name: str
    mode: str
    physical_query_rows: int
    active_query_rows: int
    context_length: int
    top_k: int = GLM52_DSA_CONTRACT.index_topk
    num_heads: int = GLM52_DSA_CONTRACT.local_query_heads
    latent_dim: int = GLM52_DSA_CONTRACT.latent_dim
    rope_dim: int = GLM52_DSA_CONTRACT.rope_dim
    page_size: int = GLM52_DSA_CONTRACT.page_size
    start_position: int | None = None
    slot_order: str = "unsorted"
    request_layout: str = "disjoint"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("performance case name must not be empty")
        if self.mode not in {"decode", "prefill"}:
            raise ValueError("mode must be 'decode' or 'prefill'")
        if self.physical_query_rows <= 0:
            raise ValueError("physical_query_rows must be positive")
        if not 0 < self.active_query_rows <= self.physical_query_rows:
            raise ValueError("active_query_rows must be in [1, physical_query_rows]")
        if self.context_length <= 0 or self.top_k <= 1:
            raise ValueError("context_length and top_k must be positive")
        if self.num_heads <= 0 or self.latent_dim <= 0 or self.rope_dim <= 0:
            raise ValueError("head and attention dimensions must be positive")
        production_dimensions = (
            GLM52_DSA_CONTRACT.index_topk,
            GLM52_DSA_CONTRACT.local_query_heads,
            GLM52_DSA_CONTRACT.latent_dim,
            GLM52_DSA_CONTRACT.rope_dim,
            GLM52_DSA_CONTRACT.page_size,
        )
        if (
            self.top_k,
            self.num_heads,
            self.latent_dim,
            self.rope_dim,
            self.page_size,
        ) != production_dimensions:
            raise ValueError(
                "performance cases must use the GLM-5.2 production kernel dimensions"
            )
        if self.page_size <= 0 or self.page_size % GLM52_DSA_CONTRACT.packing:
            raise ValueError("page_size must be divisible by packing")
        if self.slot_order not in {"unsorted", "page-sorted"}:
            raise ValueError("slot_order must be 'unsorted' or 'page-sorted'")
        if self.request_layout not in {"shared", "disjoint"}:
            raise ValueError("request_layout must be 'shared' or 'disjoint'")
        if self.mode == "decode" and self.start_position is not None:
            raise ValueError("decode cases must not set start_position")
        if self.mode == "decode" and self.request_layout != "disjoint":
            raise ValueError(
                "decode performance cases require disjoint request regions"
            )
        if self.mode == "prefill":
            if self.start_position is None or self.start_position < 0:
                raise ValueError("prefill cases require a nonnegative start_position")
            required_context = self.start_position + self.active_query_rows
            if self.context_length < required_context:
                raise ValueError("prefill context_length must cover every active query")
            if self.request_layout != "shared":
                raise ValueError("prefill chunks require one shared request region")

    @property
    def shape_tuple(self) -> tuple[str, int, int, int, int]:
        return (
            self.mode,
            self.physical_query_rows,
            self.active_query_rows,
            self.context_length,
            self.top_k,
        )

    @property
    def valid_count_pattern(self) -> str:
        return "causal" if self.mode == "prefill" else "full"

    @property
    def minimum_cache_capacity(self) -> int:
        multiplier = self.active_query_rows if self.request_layout == "disjoint" else 1
        return self.context_length * multiplier


PERFORMANCE_CASES = (
    SparseMlaPerfCase(
        name="decode-bucket-a1-c512",
        mode="decode",
        physical_query_rows=64,
        active_query_rows=1,
        context_length=512,
    ),
    SparseMlaPerfCase(
        name="decode-bucket-a1-c1024",
        mode="decode",
        physical_query_rows=64,
        active_query_rows=1,
        context_length=1024,
    ),
    SparseMlaPerfCase(
        name="decode-bucket-a1-c2048",
        mode="decode",
        physical_query_rows=64,
        active_query_rows=1,
        context_length=2048,
    ),
    SparseMlaPerfCase(
        name="decode-bucket-a1-c4096",
        mode="decode",
        physical_query_rows=64,
        active_query_rows=1,
        context_length=4096,
    ),
    SparseMlaPerfCase(
        name="decode-bucket-a8-c4096",
        mode="decode",
        physical_query_rows=64,
        active_query_rows=8,
        context_length=4096,
    ),
    SparseMlaPerfCase(
        name="decode-bucket-a32-c4096",
        mode="decode",
        physical_query_rows=64,
        active_query_rows=32,
        context_length=4096,
    ),
    SparseMlaPerfCase(
        name="decode-bucket-a64-c4096",
        mode="decode",
        physical_query_rows=64,
        active_query_rows=64,
        context_length=4096,
    ),
    SparseMlaPerfCase(
        name="decode-long-a1-c160k",
        mode="decode",
        physical_query_rows=1,
        active_query_rows=1,
        context_length=160_000,
    ),
    SparseMlaPerfCase(
        name="decode-throughput-a8-c32k",
        mode="decode",
        physical_query_rows=8,
        active_query_rows=8,
        context_length=32_000,
    ),
    SparseMlaPerfCase(
        name="prefill-t128-start0",
        mode="prefill",
        physical_query_rows=128,
        active_query_rows=128,
        context_length=128,
        start_position=0,
        request_layout="shared",
    ),
    SparseMlaPerfCase(
        name="prefill-t128-start2048",
        mode="prefill",
        physical_query_rows=128,
        active_query_rows=128,
        context_length=2176,
        start_position=2048,
        request_layout="shared",
    ),
)


PERFORMANCE_CASES_BY_NAME = {case.name: case for case in PERFORMANCE_CASES}

if len(PERFORMANCE_CASES_BY_NAME) != len(PERFORMANCE_CASES):
    raise ValueError("GLM-5.2 DSA performance case names must be unique")
