import functools
import json
import logging
import os
import time
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.gmm.megablox_gmm_backend import gmm

logger = logging.getLogger(__name__)


class MegabloxGMMAutoTuner:
    # Class-level tile size definitions to avoid duplication
    TILE_SIZES_M = [128, 256, 512, 1024, 2048]
    TILE_SIZES_K = [128, 256, 512, 1024, 2048]
    TILE_SIZES_N = [
        128,
        256,
        512,
        1024,
        2048,
    ]

    def __init__(self, cache_dir: str = "/tmp/gmm_tuning_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, m: int, k: int, n: int, num_groups: int) -> str:
        return f"m{m}_k{k}_n{n}_g{num_groups}"

    def _get_cache_file(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _load_cached_result(self, cache_key: str) -> Optional[np.ndarray]:
        cache_file = self._get_cache_file(cache_key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    return np.array(data["tuning_result"], dtype=np.int32)
            except Exception:
                pass
        return None

    def _save_cached_result(
        self, cache_key: str, tuning_result: np.ndarray, best_time: float
    ):
        cache_file = self._get_cache_file(cache_key)
        data = {
            "tuning_result": tuning_result.tolist(),
            "best_time_ms": best_time * 1000,
            "timestamp": time.time(),
        }
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def _create_mock_data(
        self,
        m: int,
        k: int,
        n: int,
        num_groups: int,
        dtype: jnp.dtype = jnp.bfloat16,
        seed: int = 42,
    ):
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 2)

        lhs = jax.random.normal(keys[0], (m, k), dtype=dtype)
        rhs = jax.random.normal(keys[1], (num_groups, k, n), dtype=dtype)
        group_sizes = jnp.array([m // num_groups] * num_groups, dtype=jnp.int32)

        return lhs, rhs, group_sizes

    def _benchmark_gmm(
        self,
        lhs,
        rhs,
        group_sizes,
        tiling: Tuple[int, int, int],
        num_warmup: int = 1,
        num_trials: int = 3,
    ) -> float:
        @functools.partial(jax.jit, static_argnames=["tiling"])
        def jitted_gmm(lhs, rhs, group_sizes, tiling):
            return gmm(
                lhs, rhs, group_sizes, preferred_element_type=jnp.float32, tiling=tiling
            )

        for _ in range(num_warmup):
            out = jitted_gmm(lhs, rhs, group_sizes, tiling)
            jax.block_until_ready(out)

        times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            out = jitted_gmm(lhs, rhs, group_sizes, tiling)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - start)

        return np.mean(times)

    def _generate_tiling_candidates(
        self, m: int, k: int, n: int, dtype: jnp.dtype = jnp.bfloat16
    ) -> List[Tuple[int, int, int]]:
        tile_sizes_m = self.TILE_SIZES_M
        tile_sizes_k = self.TILE_SIZES_K
        tile_sizes_n = self.TILE_SIZES_N

        candidates = []

        for tm in tile_sizes_m:
            if tm > m:
                continue
            for tk in tile_sizes_k:
                if tk > k:
                    continue
                for tn in tile_sizes_n:
                    if tn > n:
                        continue

                    # GMM constraint: dimensions must be divisible by tile sizes
                    if m % tm != 0 or k % tk != 0 or n % tn != 0:
                        continue

                    # TPU constraints: check effective dimensions (min of tile_size and actual dimension)
                    effective_tm = min(tm, m)
                    effective_tk = min(tk, k)
                    effective_tn = min(tn, n)

                    # TPU requires: k dimension divisible by 8, n dimension divisible by 128
                    if effective_tk % 8 != 0 or effective_tn % 128 != 0:
                        continue

                    candidates.append((tm, tk, tn))

        default_tm = effective_tm
        default_tk = effective_tk
        default_tn = effective_tn

        # Find the largest tm that divides m (including smaller values for decode)
        for tm in tile_sizes_m:
            if tm <= m and m % tm == 0 and tm <= effective_tm:
                default_tm = tm

        # Find the largest tk that divides k and meets TPU constraints
        for tk in reversed(tile_sizes_k):
            if tk <= k and k % tk == 0 and tk <= effective_tk:
                default_tk = tk
                break

        # Find the largest tn that divides n and meets TPU constraints
        for tn in reversed(tile_sizes_n):
            if tn <= n and n % tn == 0 and tn <= effective_tn:
                default_tn = tn
                break

        default_tiling = (default_tm, default_tk, default_tn)
        if default_tiling not in candidates and all(d > 0 for d in default_tiling):
            candidates.append(default_tiling)

        candidates.sort(key=lambda x: (x[0] * x[1] * x[2], x[0], x[1], x[2]))

        # Filter out candidates that exceed memory limits
        candidates = self._filter_by_memory_limit(candidates, dtype=dtype)

        return candidates

    def _estimate_tile_memory_usage(
        self, tm: int, tk: int, tn: int, dtype: jnp.dtype = jnp.bfloat16
    ) -> int:
        """Estimate memory usage for a tiling configuration in bytes.

        Args:
            tm, tk, tn: Tile dimensions
            dtype: Data type for input/output matrices

        Returns:
            Estimated memory usage in bytes
        """
        # Data type sizes in bytes
        dtype_size = {
            jnp.bfloat16: 2,
            jnp.float16: 2,
            jnp.float32: 4,
        }.get(
            dtype, 4
        )  # Default to 4 bytes if unknown

        # Memory components:
        # 1. lhs tile: (tm, tk)
        lhs_memory = tm * tk * dtype_size

        # 2. rhs tile: (tk, tn)
        rhs_memory = tk * tn * dtype_size

        # 3. accumulator/output tile: (tm, tn) - always fp32 for accumulation
        acc_memory = tm * tn * 4  # float32

        # 4. Additional overhead (temp variables, alignment, etc.) ~20%
        base_memory = lhs_memory + rhs_memory + acc_memory
        overhead = int(base_memory * 0.2)

        total_memory = base_memory + overhead

        return total_memory

    def _filter_by_memory_limit(
        self,
        candidates: List[Tuple[int, int, int]],
        dtype: jnp.dtype = jnp.bfloat16,
        max_memory_mb: int = 64,
    ) -> List[Tuple[int, int, int]]:
        """Filter out tiling candidates that exceed memory limits.

        Args:
            candidates: List of (tm, tk, tn) tuples
            dtype: Data type for computation
            max_memory_mb: Maximum memory limit in MB

        Returns:
            Filtered list of candidates
        """
        max_memory_bytes = max_memory_mb * 1024 * 1024  # Convert MB to bytes
        filtered_candidates = []

        for tm, tk, tn in candidates:
            memory_usage = self._estimate_tile_memory_usage(tm, tk, tn, dtype)

            if memory_usage <= max_memory_bytes:
                filtered_candidates.append((tm, tk, tn))
            else:
                logger.debug(
                    f"Skipping tiling ({tm}, {tk}, {tn}) - "
                    f"memory usage {memory_usage / (1024*1024):.1f}MB > "
                    f"{max_memory_mb}MB limit"
                )

        logger.debug(
            f"Memory filtering: {len(filtered_candidates)}/{len(candidates)} "
            f"candidates within {max_memory_mb}MB limit"
        )

        return filtered_candidates

    def _format_failure_summary(self, failure_reasons: dict) -> str:
        """Format failure reasons into a readable summary."""
        if not failure_reasons:
            return "None"

        summary_parts = []
        for error_type, details in failure_reasons.items():
            count = details["count"]
            examples = details["examples"]
            if count == 1 and examples:
                summary_parts.append(f"{error_type}(1): {examples[0]}")
            else:
                example_str = f" e.g. {examples[0]}" if examples else ""
                summary_parts.append(f"{error_type}({count}){example_str}")

        return "; ".join(summary_parts)

    def _find_best_tiling(
        self, lhs, rhs, group_sizes, candidates: List[Tuple[int, int, int]]
    ) -> Tuple[Optional[np.ndarray], float, dict]:
        """Find the best tiling from candidates and return result with failure stats."""
        best_tiling = None
        best_time = float("inf")
        failure_reasons = {}

        for tiling in candidates:
            try:
                avg_time = self._benchmark_gmm(lhs, rhs, group_sizes, tiling)
                if avg_time < best_time:
                    best_time = avg_time
                    best_tiling = np.array(tiling, dtype=np.int32)
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                if error_type not in failure_reasons:
                    failure_reasons[error_type] = {"count": 0, "examples": []}
                failure_reasons[error_type]["count"] += 1
                if len(failure_reasons[error_type]["examples"]) < 3:
                    failure_reasons[error_type]["examples"].append(
                        f"{tiling}: {error_msg}"
                    )
                logger.debug(f"Tiling {tiling} failed: {error_type}: {error_msg}")

        return best_tiling, best_time, failure_reasons

    def _get_fallback_tiling(self, m: int, k: int, n: int) -> np.ndarray:
        """Get a conservative fallback tiling that should always work."""
        # Find largest tiles that divide dimensions and meet TPU constraints

        # Find largest tm that divides m
        fallback_tm = 8  # Safe minimum for decode
        for tm in reversed(self.TILE_SIZES_M):
            if tm <= m and m % tm == 0:
                fallback_tm = tm
                break

        # Find largest tk that divides k and meets TPU constraints
        fallback_tk = 128  # TPU-safe minimum
        for tk in reversed(self.TILE_SIZES_K):
            if tk <= k and k % tk == 0:
                effective_tk = min(tk, k)
                if effective_tk % 8 == 0:
                    fallback_tk = tk
                    break

        # Find largest tn that divides n and meets TPU constraints
        fallback_tn = 128  # TPU-safe minimum
        for tn in reversed(self.TILE_SIZES_N):
            if tn <= n and n % tn == 0:
                effective_tn = min(tn, n)
                if effective_tn % 128 == 0:
                    fallback_tn = tn
                    break

        return np.array([fallback_tm, fallback_tk, fallback_tn], dtype=np.int32)

    def tune_for_target_size(
        self,
        m: int,
        k: int,
        n: int,
        num_groups: int,
        use_cache: bool = True,
    ) -> np.ndarray[np.int32]:
        """Find the optimal tiling configuration for given problem dimensions.

        Args:
            m: Number of rows in lhs matrix
            k: Shared dimension (lhs cols, rhs rows)
            n: Number of columns in rhs matrix
            num_groups: Number of expert groups
            use_cache: Whether to use cached results

        Returns:
            Optimal tiling configuration as np.ndarray([tm, tk, tn])
        """
        cache_key = self._get_cache_key(m, k, n, num_groups)

        # Check cache first
        if use_cache:
            cached_result = self._load_cached_result(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached tiling for {cache_key}: {cached_result}")
                return cached_result

        logger.debug(f"Tuning tiling for: m={m}, k={k}, n={n}, groups={num_groups}")

        # Create test data and generate candidates
        lhs, rhs, group_sizes = self._create_mock_data(m, k, n, num_groups)
        candidates = self._generate_tiling_candidates(m, k, n, dtype=lhs.dtype)

        if not candidates:
            logger.warning(
                f"No valid candidates for (m={m}, k={k}, n={n}), using fallback"
            )
            return self._get_fallback_tiling(m, k, n)

        # Find best tiling through benchmarking
        best_tiling, best_time, failure_reasons = self._find_best_tiling(
            lhs, rhs, group_sizes, candidates
        )

        # Handle results
        if best_tiling is None:
            # All candidates failed - use fallback
            best_tiling = self._get_fallback_tiling(m, k, n)
            failure_summary = self._format_failure_summary(failure_reasons)
            logger.warning(
                f"[GMM AUTO-TUNE] All {len(candidates)} candidates failed for "
                f"(m={m}, k={k}, n={n}, groups={num_groups}), using fallback {best_tiling}. "
                f"Failures: {failure_summary}"
            )
        else:
            # Some candidates succeeded
            failed_count = sum(details["count"] for details in failure_reasons.values())
            if failed_count > 0:
                failure_summary = self._format_failure_summary(failure_reasons)
                logger.info(
                    f"[GMM AUTO-TUNE] {failed_count}/{len(candidates)} candidates failed for "
                    f"(m={m}, k={k}, n={n}, groups={num_groups}). Failures: {failure_summary}"
                )

            # Cache successful result
            if use_cache:
                self._save_cached_result(cache_key, best_tiling, best_time)
                logger.debug(
                    f"Cached optimal tiling {best_tiling} with time {best_time:.4f}s"
                )

        return best_tiling
