"""Tests for SWA dual-pool memory budget optimisation.

Validates that compute_optimal_swa_ratio() produces correct R values
for different model configurations, and that the resulting pool sizes
make better use of available memory than the fixed R=0.8 baseline.
"""

import unittest

from sgl_jax.srt.utils.memory_budget import compute_optimal_swa_ratio

# ---- Helper: cell-size calculation matching _compute_kv_cell_per_layer ----
# Reproduces the aligned-dim + packing logic without importing heavy deps.


def _cell_per_layer(head_dim: int, v_head_dim: int, kv_heads: int, dtype_bytes: int = 2) -> int:
    """Per-token per-layer KV bytes, matching model_runner._compute_kv_cell_per_layer."""
    hd_aligned = (head_dim + 127) // 128 * 128
    vhd_aligned = (v_head_dim + 127) // 128 * 128
    per_token_dim = (hd_aligned + vhd_aligned) if head_dim != v_head_dim else hd_aligned * 2
    return kv_heads * per_token_dim * dtype_bytes


# MiMo-V2-Flash parameters
MIMO_HEAD_DIM = 192
MIMO_V_HEAD_DIM = 128
MIMO_FA_HEADS = 4
MIMO_SWA_HEADS = 8
MIMO_FA_LAYERS = 9
MIMO_SWA_LAYERS = 39
MIMO_WINDOW = 128
MIMO_CONTEXT = 4096

MIMO_FA_CELL = _cell_per_layer(MIMO_HEAD_DIM, MIMO_V_HEAD_DIM, MIMO_FA_HEADS)
MIMO_SWA_CELL = _cell_per_layer(MIMO_HEAD_DIM, MIMO_V_HEAD_DIM, MIMO_SWA_HEADS)


class TestComputeOptimalSwaRatio(unittest.TestCase):
    """Unit tests for compute_optimal_swa_ratio()."""

    def test_mimo_v2_flash_defaults(self):
        """MiMo-V2-Flash: 9 FA (4 heads) + 39 SWA (8 heads), window=128, ctx=4096."""
        # SWA cell should be ~2x FA cell due to 8 vs 4 heads
        self.assertAlmostEqual(MIMO_SWA_CELL / MIMO_FA_CELL, 2.0, places=1)

        R = compute_optimal_swa_ratio(
            user_R=0.8,  # default
            sliding_window=MIMO_WINDOW,
            context_len=MIMO_CONTEXT,
            fa_cell_per_layer=MIMO_FA_CELL,
            swa_cell_per_layer=MIMO_SWA_CELL,
            fa_layers=MIMO_FA_LAYERS,
            swa_layers=MIMO_SWA_LAYERS,
        )

        # R should be much less than the default 0.8
        self.assertLess(R, 0.5, "Optimal R should be well below the 0.8 default")
        # R should be at least the minimum clamp
        self.assertGreaterEqual(R, 0.01)

    def test_user_override_respected(self):
        """When user explicitly sets a non-default ratio, it should be returned."""
        R = compute_optimal_swa_ratio(
            user_R=0.15,
            sliding_window=MIMO_WINDOW,
            context_len=MIMO_CONTEXT,
            fa_cell_per_layer=MIMO_FA_CELL,
            swa_cell_per_layer=MIMO_SWA_CELL,
            fa_layers=MIMO_FA_LAYERS,
            swa_layers=MIMO_SWA_LAYERS,
        )
        self.assertAlmostEqual(R, 0.15, places=6, msg="User-specified R should be returned as-is")

    def test_large_window_caps_at_095(self):
        """When sliding_window ~ context_len, R should be capped at 0.95."""
        R = compute_optimal_swa_ratio(
            user_R=0.8,
            sliding_window=4000,
            context_len=4096,
            fa_cell_per_layer=MIMO_FA_CELL,
            swa_cell_per_layer=MIMO_SWA_CELL,
            fa_layers=MIMO_FA_LAYERS,
            swa_layers=MIMO_SWA_LAYERS,
        )
        self.assertLessEqual(R, 0.95)

    def test_no_sliding_window_falls_back(self):
        """When sliding_window is 0 or None, fall back to user ratio."""
        for sw in (0, None, -1):
            R = compute_optimal_swa_ratio(
                user_R=0.8,
                sliding_window=sw,
                context_len=MIMO_CONTEXT,
                fa_cell_per_layer=MIMO_FA_CELL,
                swa_cell_per_layer=MIMO_SWA_CELL,
                fa_layers=MIMO_FA_LAYERS,
                swa_layers=MIMO_SWA_LAYERS,
            )
            self.assertAlmostEqual(
                R, 0.8, places=6, msg=f"sliding_window={sw} should fall back to user_R"
            )

    def test_equal_heads_equal_cost(self):
        """When FA and SWA have same head count, cost_ratio adjusts R differently."""
        fa_cell = _cell_per_layer(MIMO_HEAD_DIM, MIMO_V_HEAD_DIM, 4)
        swa_cell = _cell_per_layer(MIMO_HEAD_DIM, MIMO_V_HEAD_DIM, 4)
        self.assertEqual(fa_cell, swa_cell)

        R = compute_optimal_swa_ratio(
            user_R=0.8,
            sliding_window=MIMO_WINDOW,
            context_len=MIMO_CONTEXT,
            fa_cell_per_layer=fa_cell,
            swa_cell_per_layer=swa_cell,
            fa_layers=9,
            swa_layers=39,
        )
        # R_window = 128*1.25/4096 = 0.0390625
        # R_cost = (9/39)*0.039 ~= 0.009 -> clamped to 0.01
        self.assertGreaterEqual(R, 0.01)
        self.assertLess(R, 0.1)

    def test_deterministic(self):
        """Same inputs should always produce same output."""
        kwargs = dict(
            user_R=0.8,
            sliding_window=MIMO_WINDOW,
            context_len=MIMO_CONTEXT,
            fa_cell_per_layer=MIMO_FA_CELL,
            swa_cell_per_layer=MIMO_SWA_CELL,
            fa_layers=MIMO_FA_LAYERS,
            swa_layers=MIMO_SWA_LAYERS,
        )
        R1 = compute_optimal_swa_ratio(**kwargs)
        R2 = compute_optimal_swa_ratio(**kwargs)
        self.assertEqual(R1, R2)


class TestMemoryAllocationImprovement(unittest.TestCase):
    """Verify that the optimised ratio gives strictly more full-pool tokens."""

    def test_more_full_tokens_with_optimal_R(self):
        """Optimal R yields more full_max tokens than R=0.8 for MiMo config."""
        available = 10 * 1024**3  # 10 GB

        full_cell = MIMO_FA_CELL * MIMO_FA_LAYERS
        swa_cell_total = MIMO_SWA_CELL * MIMO_SWA_LAYERS

        # Old approach: R=0.8
        R_old = 0.8
        full_max_old = int(available / (full_cell + R_old * swa_cell_total))

        # New approach
        R_new = compute_optimal_swa_ratio(
            user_R=0.8,
            sliding_window=MIMO_WINDOW,
            context_len=MIMO_CONTEXT,
            fa_cell_per_layer=MIMO_FA_CELL,
            swa_cell_per_layer=MIMO_SWA_CELL,
            fa_layers=MIMO_FA_LAYERS,
            swa_layers=MIMO_SWA_LAYERS,
        )
        full_max_new = int(available / (full_cell + R_new * swa_cell_total))

        self.assertGreater(
            full_max_new,
            full_max_old,
            f"Optimal R={R_new:.4f} should yield more full tokens than R=0.8 "
            f"({full_max_new} vs {full_max_old})",
        )

    def test_swa_pool_still_covers_window(self):
        """The SWA pool must have enough tokens for at least 1 request's window."""
        available = 10 * 1024**3

        full_cell = MIMO_FA_CELL * MIMO_FA_LAYERS
        swa_cell_total = MIMO_SWA_CELL * MIMO_SWA_LAYERS

        R = compute_optimal_swa_ratio(
            user_R=0.8,
            sliding_window=MIMO_WINDOW,
            context_len=MIMO_CONTEXT,
            fa_cell_per_layer=MIMO_FA_CELL,
            swa_cell_per_layer=MIMO_SWA_CELL,
            fa_layers=MIMO_FA_LAYERS,
            swa_layers=MIMO_SWA_LAYERS,
        )
        full_max = int(available / (full_cell + R * swa_cell_total))
        swa_max = int(full_max * R)

        self.assertGreaterEqual(
            swa_max,
            MIMO_WINDOW,
            f"SWA pool ({swa_max}) must hold at least one window ({MIMO_WINDOW})",
        )

    def test_total_memory_not_exceeded(self):
        """The allocated pools must not exceed the available memory budget."""
        available = 10 * 1024**3

        full_cell = MIMO_FA_CELL * MIMO_FA_LAYERS
        swa_cell_total = MIMO_SWA_CELL * MIMO_SWA_LAYERS

        R = compute_optimal_swa_ratio(
            user_R=0.8,
            sliding_window=MIMO_WINDOW,
            context_len=MIMO_CONTEXT,
            fa_cell_per_layer=MIMO_FA_CELL,
            swa_cell_per_layer=MIMO_SWA_CELL,
            fa_layers=MIMO_FA_LAYERS,
            swa_layers=MIMO_SWA_LAYERS,
        )
        full_max = int(available / (full_cell + R * swa_cell_total))
        swa_max = int(full_max * R)

        total_used = full_max * full_cell + swa_max * swa_cell_total
        self.assertLessEqual(total_used, available)


class TestRatioEdgeCases(unittest.TestCase):
    """Edge cases for the ratio computation."""

    def test_very_small_window(self):
        """Window=1 should give a very small R, clamped to 0.01."""
        R = compute_optimal_swa_ratio(
            user_R=0.8,
            sliding_window=1,
            context_len=131072,
            fa_cell_per_layer=MIMO_FA_CELL,
            swa_cell_per_layer=MIMO_SWA_CELL,
            fa_layers=MIMO_FA_LAYERS,
            swa_layers=MIMO_SWA_LAYERS,
        )
        self.assertAlmostEqual(R, 0.01, places=2, msg="Very small window should clamp R to 0.01")

    def test_zero_swa_layers(self):
        """0 SWA layers: function should still return a valid R."""
        R = compute_optimal_swa_ratio(
            user_R=0.8,
            sliding_window=MIMO_WINDOW,
            context_len=MIMO_CONTEXT,
            fa_cell_per_layer=MIMO_FA_CELL,
            swa_cell_per_layer=MIMO_SWA_CELL,
            fa_layers=48,
            swa_layers=0,
        )
        self.assertGreaterEqual(R, 0.01)
        self.assertLessEqual(R, 0.95)

    def test_zero_context_len(self):
        """context_len=0 should fall back to user_R."""
        R = compute_optimal_swa_ratio(
            user_R=0.8,
            sliding_window=128,
            context_len=0,
            fa_cell_per_layer=MIMO_FA_CELL,
            swa_cell_per_layer=MIMO_SWA_CELL,
            fa_layers=9,
            swa_layers=39,
        )
        self.assertAlmostEqual(R, 0.8, places=6)

    def test_custom_headroom(self):
        """Custom headroom factor should affect the result."""
        R_default = compute_optimal_swa_ratio(
            user_R=0.8,
            sliding_window=MIMO_WINDOW,
            context_len=MIMO_CONTEXT,
            fa_cell_per_layer=MIMO_FA_CELL,
            swa_cell_per_layer=MIMO_SWA_CELL,
            fa_layers=MIMO_FA_LAYERS,
            swa_layers=MIMO_SWA_LAYERS,
            headroom=1.25,
        )
        R_larger = compute_optimal_swa_ratio(
            user_R=0.8,
            sliding_window=MIMO_WINDOW,
            context_len=MIMO_CONTEXT,
            fa_cell_per_layer=MIMO_FA_CELL,
            swa_cell_per_layer=MIMO_SWA_CELL,
            fa_layers=MIMO_FA_LAYERS,
            swa_layers=MIMO_SWA_LAYERS,
            headroom=2.0,
        )
        self.assertGreaterEqual(R_larger, R_default, "Larger headroom should not decrease R")


if __name__ == "__main__":
    unittest.main()
