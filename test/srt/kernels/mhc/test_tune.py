"""M1.0b: the mHC platform schedule table.

Pure CPU tests. `tune.py` is all closed-form VMEM arithmetic and candidate
selection -- no Pallas lowering -- so this is exactly the part of the mHC
kernels that can be checked without a TPU. That matters here: the v7x gap this
adds a schedule for was invisible locally, because `test_mhc.py` is
`skipif(default_backend() != "tpu")` and skipped all 55 cases on CPU.
"""

import pytest

from sgl_jax.srt.kernels.mhc.tune import (
    _PLATFORMS,
    _platform_parameters,
    collapse_vmem_bytes,
    post_vmem_bytes,
    select_collapse_block_tokens,
    select_post_block_tokens,
)

MIB = 1024 * 1024

# DeepSeek-V4-Flash-0731 as shipped.
HC, HIDDEN = 4, 4096
MIX_HC = (2 + HC) * HC  # 24

V6E = "TPU v6e"
# The literal string jax reports on v7x. Not "TPU v7x" -- getting this wrong is
# how the schedule silently fails to match.
V7X_DEVICE_KIND = "TPU7x"


def test_v7x_device_kind_resolves():
    """The exact `device_kind` that used to raise."""
    assert _platform_parameters(V7X_DEVICE_KIND).name == "TPU v7x"


@pytest.mark.parametrize(
    "device_kind,expected",
    [
        ("TPU7x", "TPU v7x"),
        ("tpu7x", "TPU v7x"),
        ("TPU v7x", "TPU v7x"),
        ("TPU v6e", "TPU v6e"),
        ("TPU v6 lite", "TPU v6e"),
    ],
)
def test_marker_matching_is_not_ambiguous(device_kind, expected):
    """Adding v7x must not steal v6e's devices, or vice versa."""
    assert _platform_parameters(device_kind).name == expected


def test_unknown_device_still_raises_and_lists_both():
    with pytest.raises(ValueError, match="mHC has no schedule for"):
        _platform_parameters("TPU v5e")
    try:
        _platform_parameters("TPU v5e")
    except ValueError as exc:
        assert "TPU v6e" in str(exc)
        assert "TPU v7x" in str(exc)


def test_v7x_has_twice_the_scoped_budget():
    assert _platform_parameters(V7X_DEVICE_KIND).vmem_bytes == 64 * MIB
    assert _platform_parameters(V6E).vmem_bytes == 32 * MIB


@pytest.mark.parametrize("platform", _PLATFORMS, ids=lambda p: p.name)
def test_selected_collapse_block_fits_its_own_budget(platform):
    """The invariant, not a magic number: whatever gets selected must fit, and
    the next candidate up must not (otherwise the budget is left unused).

    Applies to every entry in the table, so a future platform cannot be added
    with a budget and candidate list that disagree.
    """
    for activation_bytes, blocks in (
        (2, platform.collapse_blocks),
        (4, platform.highest_collapse_blocks),
    ):
        chosen = select_collapse_block_tokens(
            platform.device_markers[0],
            tokens=8192,
            hc_mult=HC,
            hidden=HIDDEN,
            activation_bytes=activation_bytes,
            highest_precision=activation_bytes == 4,
        )
        cost = collapse_vmem_bytes(
            chosen,
            hc_mult=HC,
            hidden=HIDDEN,
            rows=MIX_HC,
            activation_bytes=activation_bytes,
        )
        assert cost <= platform.vmem_bytes, (platform.name, chosen, cost)
        bigger = [b for b in blocks if b > chosen]
        if bigger:
            over = collapse_vmem_bytes(
                min(bigger),
                hc_mult=HC,
                hidden=HIDDEN,
                rows=MIX_HC,
                activation_bytes=activation_bytes,
            )
            assert over > platform.vmem_bytes, (
                f"{platform.name}: {min(bigger)} would also fit, so {chosen} "
                "leaves budget unused"
            )


@pytest.mark.parametrize("platform", _PLATFORMS, ids=lambda p: p.name)
def test_selected_post_block_fits_its_own_budget(platform):
    chosen = select_post_block_tokens(
        platform.device_markers[0],
        tokens=8192,
        hc_mult=HC,
        hidden=HIDDEN,
        x_bytes=2,
        residual_bytes=2,
    )
    cost = post_vmem_bytes(chosen, hc_mult=HC, hidden=HIDDEN, x_bytes=2, residual_bytes=2)
    assert cost <= platform.vmem_bytes, (platform.name, chosen, cost)
    bigger = [b for b in platform.post_blocks if b > chosen]
    if bigger:
        over = post_vmem_bytes(min(bigger), hc_mult=HC, hidden=HIDDEN, x_bytes=2, residual_bytes=2)
        assert over > platform.vmem_bytes


def test_the_bigger_budget_actually_buys_bigger_tiles():
    """Otherwise the v7x entry would be a no-op relative to v6e."""
    kw = dict(tokens=8192, hc_mult=HC, hidden=HIDDEN, activation_bytes=2)
    v6e = select_collapse_block_tokens(V6E, **kw)
    v7x = select_collapse_block_tokens(V7X_DEVICE_KIND, **kw)
    assert v7x > v6e, (v6e, v7x)

    post_kw = dict(tokens=8192, hc_mult=HC, hidden=HIDDEN, x_bytes=2, residual_bytes=2)
    assert select_post_block_tokens(V7X_DEVICE_KIND, **post_kw) > select_post_block_tokens(
        V6E, **post_kw
    )


def test_highest_precision_collapse_gains_a_candidate_on_v7x():
    """The one list that needed a new entry: f32 collapse at 128 costs 41.52 MiB,
    which does not fit v6e's 32 MiB but does fit v7x's 64 MiB."""
    kw = dict(tokens=8192, hc_mult=HC, hidden=HIDDEN, activation_bytes=4, highest_precision=True)
    assert select_collapse_block_tokens(V6E, **kw) == 64
    assert select_collapse_block_tokens(V7X_DEVICE_KIND, **kw) == 128
    cost_128 = collapse_vmem_bytes(128, hc_mult=HC, hidden=HIDDEN, rows=MIX_HC, activation_bytes=4)
    assert 32 * MIB < cost_128 <= 64 * MIB


def test_short_inputs_still_clamp_to_64_on_both_platforms():
    """`select_post_block_tokens` caps at 64 below 2048 tokens regardless of
    budget; the bigger v7x budget must not change that."""
    for kind in (V6E, V7X_DEVICE_KIND):
        chosen = select_post_block_tokens(
            kind, tokens=1024, hc_mult=HC, hidden=HIDDEN, x_bytes=2, residual_bytes=2
        )
        assert chosen == 64, (kind, chosen)


def test_gates_blocks_are_never_the_binding_constraint():
    from sgl_jax.srt.kernels.mhc.tune import gates_vmem_bytes

    for platform in _PLATFORMS:
        worst = max(platform.gates_blocks)
        assert gates_vmem_bytes(worst, hc_mult=HC, mix_hc=MIX_HC) < MIB, platform.name
