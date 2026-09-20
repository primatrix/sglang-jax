"""V4 pool planning: the SWA/history split reproduces the budgets logged on v7x
(09-19 thrbu / thrc1 servers) and the V4 default ratio is 0.2, not the generic 0.8."""

from sgl_jax.srt.mem_cache.deepseek_v4.capacity import plan_deepseek_v4_pools
from sgl_jax.srt.mem_cache.deepseek_v4.pool import DeepseekV4CacheSpec
from sgl_jax.srt.server_args import ServerArgs

# DeepSeek V4-Flash: 2 dense + 21 ratio-4 (CSA) + 20 ratio-128 (HCA) backbone layers.
_FLASH = DeepseekV4CacheSpec(compress_ratios=(0, 0) + (4, 128) * 20 + (4,))
_AVAILABLE = 41_162_655_334  # bytes per device after weights and mem_fraction 0.8 headroom


def test_flash_spec_bytes():
    assert _FLASH.history_bytes_per_page(128) == 880_640  # 6,880 B/token compressed KV
    assert _FLASH.swa_bytes_per_token == 44_032  # 43 layers x 512 x bf16 window rows


def test_generic_ratio_reproduces_thrbu_budget():
    b = plan_deepseek_v4_pools(_FLASH, _AVAILABLE, 256, 128, swa_full_tokens_ratio=0.8)
    assert (b.history_tokens, b.swa_tokens) == (902_784, 722_304)


def test_v4_ratio_reproduces_thrc1_budget():
    b = plan_deepseek_v4_pools(_FLASH, _AVAILABLE, 256, 128, swa_full_tokens_ratio=0.2)
    assert (b.history_tokens, b.swa_tokens) == (2_423_680, 484_736)
    assert b.allocated_bytes_per_device <= _AVAILABLE


def test_server_args_ratio_default_is_unset():
    # None lets the generic hybrid path keep 0.8 while the V4 planner picks 0.2.
    assert ServerArgs.swa_full_tokens_ratio is None
