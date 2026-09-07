"""VMEM models and backend/tile selection for mHC kernels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MHCPlatform:
    name: str
    device_markers: tuple[str, ...]
    lane_width: int
    # Scoped VMEM the compiler grants without an explicit override.
    vmem_bytes: int
    # Cross-program VMEM available to XLA, less allocator/alignment reserve.
    xla_vmem_bytes: int
    xla_vmem_reserve_bytes: int
    collapse_blocks: tuple[int, ...]
    highest_collapse_blocks: tuple[int, ...]
    gates_blocks: tuple[int, ...]
    post_blocks: tuple[int, ...]


_PLATFORMS = (
    MHCPlatform(
        name="TPU v6e",
        device_markers=("v6e", "v6 lite", "tpu v6"),
        lane_width=128,
        vmem_bytes=32 * 1024 * 1024,
        xla_vmem_bytes=96 * 1024 * 1024,
        xla_vmem_reserve_bytes=64 * 1024,
        collapse_blocks=(8, 16, 32, 64, 128, 256),
        highest_collapse_blocks=(8, 16, 32, 64),
        gates_blocks=(512, 1024, 2048),
        post_blocks=(8, 16, 32, 64, 128, 256),
    ),
    MHCPlatform(
        # ``device_kind`` reports exactly "TPU7x" (not "TPU v7x"), so "tpu7x" is
        # the marker that actually fires; the others are spelling insurance.
        name="TPU v7x",
        device_markers=("tpu7x", "v7x", "tpu v7"),
        lane_width=128,
        # v7x has 64 MiB of VMEM in total, but ``vmem_bytes`` is not the total --
        # it is the *scoped* allocation the compiler grants a Pallas call without
        # an explicit override, and that is 32 MiB, same as v6e. Measured, from
        # the compiler refusing a larger tile:
        #   RESOURCE_EXHAUSTED: E1001: CompileTimeScopedVmemOom: ... Scoped
        #   allocation with size 37.99M and limit 32.00M exceeded scoped vmem
        #   limit by 5.99M     (%mhc-collapse-pre, bf16[256,4096])
        # Setting this to 64 MiB makes the selector choose tiles the compiler
        # then rejects, which is a hard failure at every token count above 128
        # rather than a slowdown.
        vmem_bytes=32 * 1024 * 1024,
        # Cross-program VMEM left for XLA once the scoped reserve is taken:
        # total - scoped. That is the relation the v6e entry already encodes
        # (128 MiB total - 32 MiB scoped = 96 MiB), so v7x is 64 - 32 = 32 MiB.
        # This is the one field where v7x genuinely differs from v6e, and it
        # matters: with a third of v6e's cross-program VMEM, XLA starts spilling
        # the post op above ~681 tokens instead of ~2046, so
        # ``select_post_backend`` hands over to the Pallas post kernel much
        # earlier. Getting it wrong is a perf issue, not a correctness one.
        xla_vmem_bytes=32 * 1024 * 1024,
        xla_vmem_reserve_bytes=64 * 1024,
        # The block lists are candidate sets, not tuned constants -- selection is
        # analytic (``_largest_fitting`` against the closed-form VMEM models
        # below). Since the scoped budget matches v6e's, the candidate sets match
        # too, and the selected tiles come out the same. Against the shipped
        # Flash 0731 geometry (hc_mult=4, hidden=4096, mix_hc=24) at 32 MiB:
        #   collapse bf16: 128 -> 31.52 MiB fits, 256 -> 61.55 MiB does not
        #   post bf16:     128 -> 26.02 MiB fits, 256 -> 52.04 MiB does not
        #   collapse f32:  64  -> 21.51 MiB fits, 128 -> 41.52 MiB does not
        #   gates:         2048 -> 0.81 MiB, never the binding constraint
        collapse_blocks=(8, 16, 32, 64, 128, 256),
        highest_collapse_blocks=(8, 16, 32, 64),
        gates_blocks=(512, 1024, 2048),
        post_blocks=(8, 16, 32, 64, 128, 256),
    ),
)


def _platform_parameters(device_kind: str) -> MHCPlatform:
    normalized = (device_kind or "").lower()
    for platform in _PLATFORMS:
        if any(marker in normalized for marker in platform.device_markers):
            return platform
    supported = ", ".join(platform.name for platform in _PLATFORMS)
    raise ValueError(f"mHC has no schedule for {device_kind!r}; supported: {supported}")


def collapse_vmem_bytes(
    block_tokens: int,
    *,
    hc_mult: int,
    hidden: int,
    rows: int,
    activation_bytes: int,
) -> int:
    if activation_bytes <= 0:
        raise ValueError("activation_bytes must be positive")
    bt, k = block_tokens, hc_mult * hidden
    activation = 2 * bt * k * activation_bytes
    projection_operand = 2 * bt * k * 4  # FP32 projection operand, double buffered
    gated_sum = 2 * bt * hidden * 4  # one stream and the accumulator
    weights = rows * k * 4  # resident, shared by every grid step
    y_out = 2 * bt * hidden * activation_bytes
    mixes_out = 2 * bt * rows * 4
    return activation + projection_operand + gated_sum + weights + y_out + mixes_out


def gates_vmem_bytes(block_tokens: int, *, hc_mult: int, mix_hc: int) -> int:
    bt = block_tokens
    mixes_in = 2 * mix_hc * bt * 4
    state = hc_mult * hc_mult * bt * 4  # the [hc, hc, bt] Sinkhorn state
    outputs = 2 * (hc_mult + hc_mult * hc_mult) * bt * 4
    return mixes_in + state + outputs


def post_vmem_bytes(
    block_tokens: int,
    *,
    hc_mult: int,
    hidden: int,
    x_bytes: int,
    residual_bytes: int,
) -> int:
    if min(x_bytes, residual_bytes) <= 0:
        raise ValueError("x_bytes and residual_bytes must be positive")
    bt = block_tokens
    residual = 2 * bt * hc_mult * hidden * residual_bytes
    block_output = 2 * bt * hidden * x_bytes
    gates = 2 * bt * (hc_mult + hc_mult * hc_mult) * 4
    widened = bt * hc_mult * hidden * 4
    y_out = 2 * bt * hc_mult * hidden * x_bytes
    return residual + block_output + gates + widened + y_out


def post_xla_live_vmem_bytes(
    tokens: int,
    *,
    hc_mult: int,
    hidden: int,
    activation_bytes: int,
) -> int:
    """``B_live = N*D*(sizeof(float32) + HC*sizeof(activation))``."""
    if min(tokens, hc_mult, hidden, activation_bytes) <= 0:
        raise ValueError("tokens, hc_mult, hidden, and activation_bytes must be positive")
    residual = tokens * hc_mult * hidden * activation_bytes
    widened_x = tokens * hidden * 4
    return residual + widened_x


def select_post_backend(
    device_kind: str,
    *,
    tokens: int,
    hc_mult: int,
    hidden: int,
    activation_bytes: int,
    pallas_block_tokens: int = 128,
) -> str:
    """Choose Pallas only after XLA spills and when Pallas needs no padding."""
    if pallas_block_tokens <= 0:
        raise ValueError("pallas_block_tokens must be positive")
    platform = _platform_parameters(device_kind)
    usable_vmem = platform.xla_vmem_bytes - platform.xla_vmem_reserve_bytes
    live_vmem = post_xla_live_vmem_bytes(
        tokens,
        hc_mult=hc_mult,
        hidden=hidden,
        activation_bytes=activation_bytes,
    )
    xla_spills = live_vmem > usable_vmem
    pallas_is_aligned = tokens % pallas_block_tokens == 0
    return "pallas" if xla_spills and pallas_is_aligned else "xla"


def _largest_fitting(blocks, budget: int, cost) -> int:
    """Use the smallest candidate as an explicit compiler failure when none fits."""
    fitting = [block for block in blocks if cost(block) <= budget]
    return max(fitting) if fitting else min(blocks)


def select_collapse_block_tokens(
    device_kind: str,
    *,
    tokens: int,
    hc_mult: int,
    hidden: int,
    activation_bytes: int,
    highest_precision: bool = False,
) -> int:
    """Fit the collapse tile to VMEM and 8-token tile granularity."""
    if min(tokens, hc_mult, hidden, activation_bytes) <= 0:
        raise ValueError("tokens, hc_mult, hidden, and activation_bytes must be positive")
    platform = _platform_parameters(device_kind)
    blocks = platform.highest_collapse_blocks if highest_precision else platform.collapse_blocks
    mix_hc = (2 + hc_mult) * hc_mult
    max_block = _largest_fitting(
        blocks,
        platform.vmem_bytes,
        lambda bt: collapse_vmem_bytes(
            bt,
            hc_mult=hc_mult,
            hidden=hidden,
            rows=mix_hc,
            activation_bytes=activation_bytes,
        ),
    )
    return max(8, min(max_block, -(-tokens // 8) * 8))


def select_gates_block_tokens(
    device_kind: str,
    *,
    tokens: int,
    hc_mult: int,
    block_tokens: int | None = None,
) -> int:
    """Fit the gate tile to platform VMEM and lane alignment."""
    if min(tokens, hc_mult) <= 0:
        raise ValueError("tokens and hc_mult must be positive")
    platform = _platform_parameters(device_kind)
    lane = platform.lane_width

    if block_tokens is None:
        mix_hc = (2 + hc_mult) * hc_mult
        max_block = _largest_fitting(
            platform.gates_blocks,
            platform.vmem_bytes,
            lambda bt: gates_vmem_bytes(bt, hc_mult=hc_mult, mix_hc=mix_hc),
        )
        block_tokens = next((bt for bt in platform.gates_blocks if bt >= tokens), max_block)
        block_tokens = min(block_tokens, max_block, -(-tokens // lane) * lane)

    return max(lane, (int(block_tokens) // lane) * lane)


def select_post_block_tokens(
    device_kind: str,
    *,
    tokens: int,
    hc_mult: int,
    hidden: int,
    x_bytes: int,
    residual_bytes: int,
) -> int:
    """Use BT64 for short inputs, then the largest VMEM-fitting post tile."""
    if min(tokens, hc_mult, hidden, x_bytes, residual_bytes) <= 0:
        raise ValueError("tokens, hc_mult, hidden, x_bytes, and residual_bytes must be positive")
    platform = _platform_parameters(device_kind)
    max_block = _largest_fitting(
        platform.post_blocks,
        platform.vmem_bytes,
        lambda bt: post_vmem_bytes(
            bt,
            hc_mult=hc_mult,
            hidden=hidden,
            x_bytes=x_bytes,
            residual_bytes=residual_bytes,
        ),
    )
    if tokens < 2048:
        max_block = min(max_block, 64)
    return max(8, min(max_block, -(-tokens // 8) * 8))


__all__ = [
    "collapse_vmem_bytes",
    "gates_vmem_bytes",
    "post_xla_live_vmem_bytes",
    "post_vmem_bytes",
    "select_collapse_block_tokens",
    "select_gates_block_tokens",
    "select_post_backend",
    "select_post_block_tokens",
]
