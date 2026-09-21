"""Fused vs scatter membership in dsv4 admissible_mask give the same compressed mask (CPU-only)."""

import numpy as np

from sgl_jax.srt.layers.attention.dsv4 import attention as att


def _case(T, E, K, seed):
    rng = np.random.default_rng(seed)
    qpos = np.sort(rng.integers(0, 4096, size=T)).astype(np.int32)
    qreq = np.zeros(T, np.int32)
    valid = np.ones(T, bool)
    wpos = np.arange(128, dtype=np.int32)
    wreq = np.zeros(128, np.int32)
    entry_ids = np.arange(E, dtype=np.int32)
    creq = np.zeros(E, np.int32)
    selected = np.full((T, K), -1, np.int32)
    for t in range(T):
        k = rng.integers(0, K + 1)
        selected[t, :k] = rng.choice(E, size=k, replace=False)
    return dict(
        query_positions=qpos,
        query_request_ids=qreq,
        valid_token_mask=valid,
        window_positions=wpos,
        window_request_ids=wreq,
        compressed_entry_ids=entry_ids,
        compressed_request_ids=creq,
        window_size=128,
        ratio=4,
        selected_entries=selected,
    )


def test_packed_membership_edge_cases():
    # E not a multiple of 32, unused (-1) and out-of-range slots, duplicates, bit 31 and bit 0.
    E = 70
    selected = np.array(
        [[0, 31, 32, 63, 64, 69, -1, -1], [5, 5, 5, 70, 99, -1, -1, -1], [-1] * 8],
        np.int32,
    )
    got = np.asarray(att.packed_membership(selected, E))
    want = np.zeros((3, E), bool)
    want[0, [0, 31, 32, 63, 64, 69]] = True
    want[1, 5] = True
    np.testing.assert_array_equal(got, want)
