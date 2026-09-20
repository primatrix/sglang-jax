"""The one-record-page (production C1 layout) HCA prefill matches a dense reference.

The chunk-prefill kernel gathers one-record pages row by row and keeps the
gathered tile across the query blocks of one request, fetching only the new
records.  A NumPy reference of the HCA semantics (128-token window plus every
completed 128-token record, attention sink) checks that state handling across
requests, prefixes, query blocks, and boundary writes.
"""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.hca.attention import INERT_QUERY_OFFSET, ragged_attention
from sgl_jax.srt.kernels.hca.hca import HCAMetadata
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule
from sgl_jax.srt.layers.attention.hca_backend import _query_schedule

H, D, P = 8, 128, 128


def _build(reqs, cpage, key):
    """Pages are allocated per request in order; page 0 is the dummy page."""
    T = sum(t for _, t in reqs)
    b = len(reqs)
    win, win_cu, comp, comp_cu, seq_lens, comp_lens, pos, seq_ids = [], [0], [], [0], [], [], [], []
    records = {}  # (request, record) -> page 1 / page 8 row locations share one vector
    next_page = 1
    rng = np.random.default_rng(0)
    for r, (prefix, t) in enumerate(reqs):
        L = prefix + t
        pages = -(-L // P)  # ceil: the tail page holds the partial window
        win.extend(range(next_page, next_page + pages))
        win_cu.append(win_cu[-1] + pages * P)
        completed = L // 128
        count = max(1, -(-completed // cpage))
        comp.extend(range(next_page, next_page + count) if completed else [0])
        comp_cu.append(comp_cu[-1] + count * cpage)
        seq_lens.append(L)
        comp_lens.append(completed)
        pos.extend(range(prefix, L))
        seq_ids.extend([r] * t)
        for j in range(prefix // 128):  # history records
            records[(r, j)] = rng.standard_normal(D).astype(np.float32)
        next_page += pages
    cu = np.concatenate(([0], np.cumsum([t for _, t in reqs]))).astype(np.int32)
    blocks, offsets, decode = _query_schedule(cu, 32)
    bcap = T // 32 + b
    pos = np.asarray(pos, np.int32)
    boundary = np.flatnonzero((pos + 1) % 128 == 0).astype(np.int32)
    ints = lambda a: jnp.asarray(np.asarray(a, np.int32))
    md = HCAMetadata(
        state_slots=ints(seq_ids),
        query_seq_ids=ints(seq_ids),
        cu_q_lens=ints(cu),
        valid_token_mask=jnp.ones(T, bool),
        boundary_token_indices=ints(
            np.pad(boundary, (0, T // 128 + b - boundary.size), constant_values=T)
        ),
        window_page_indices=ints(win),
        window_cu_kv_lens=ints(win_cu),
        seq_lens=ints(seq_lens),
        compressed_page_indices=ints(comp),
        compressed_cu_kv_lens=ints(comp_cu),
        compressed_kv_lens=ints(comp_lens),
        query_block_request_ids=ints(np.pad(blocks, (0, bcap - blocks.size))),
        query_block_offsets=ints(
            np.pad(offsets, (0, bcap - offsets.size), constant_values=INERT_QUERY_OFFSET)
        ),
        decode_request_ids=ints(np.pad(decode, (0, b - decode.size), constant_values=-1)),
        max_queries_per_request=max(t for _, t in reqs),
    )
    # Compressed cache: request r's record j lives in physical page
    # comp[comp_cu[r] // cpage + j // cpage] at in-page row j % cpage.
    flat = np.zeros((next_page * cpage, D), np.float32)
    for (r, j), v in records.items():
        page = comp[comp_cu[r] // cpage + j // cpage]
        flat[page * cpage + j % cpage] = v
    if cpage == 1:
        compressed_cache = jnp.asarray(flat, jnp.bfloat16).reshape(next_page, 1, 1, D)
    else:
        compressed_cache = jnp.asarray(flat, jnp.bfloat16).reshape(next_page, cpage // 2, 2, D)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (T, H, D), jnp.bfloat16)
    new_kv = jax.random.normal(k2, (T, D), jnp.bfloat16)
    window_cache = jax.random.normal(k3, (next_page * P, D), jnp.bfloat16)
    write_vals = jax.random.normal(k4, (T, D), jnp.bfloat16)
    write_mask = jnp.asarray((pos + 1) % 128 == 0)
    sink = jnp.zeros((H,), jnp.float32)
    args = (
        q,
        new_kv,
        window_cache,
        compressed_cache,
        jnp.asarray(pos),
        write_vals,
        write_mask,
        sink,
        md,
    )
    aux = dict(win=win, win_cu=win_cu, comp_cu=comp_cu, records=records, cu=cu, pos=pos)
    return args, aux


def _reference(reqs, args, aux):
    """Dense HCA per query: last-128 window (history rows + this chunk) and records < (pos+1)//128."""
    q, new_kv, window_cache, _, _, write_vals, _, sink, _ = args
    f32 = lambda a: np.asarray(jnp.asarray(a).astype(jnp.float32))
    q, new_kv, window_cache, write_vals, sink = map(
        f32, (q, new_kv, window_cache, write_vals, sink)
    )
    out = np.zeros_like(q)
    scale = D**-0.5
    for r, (prefix, t) in enumerate(reqs):
        start = int(aux["cu"][r])
        req_pages = aux["win"][aux["win_cu"][r] // P : aux["win_cu"][r + 1] // P]

        def row(x):  # window row of position x
            if x >= prefix:
                return new_kv[start + x - prefix]
            return window_cache[req_pages[x // P] * P + x % P]

        for i in range(t):
            p = prefix + i
            keys = [row(x) for x in range(max(0, p - 127), p + 1)]
            for j in range((p + 1) // 128):
                if j < prefix // 128:
                    keys.append(aux["records"][(r, j)])
                else:  # written at this chunk's boundary token 128 (j + 1) - 1
                    keys.append(write_vals[start + 128 * (j + 1) - 1 - prefix])
            k = np.stack(keys)  # [K, D]; HCA uses the same row as key and value
            s = q[start + i] @ k.T * scale  # [H, K]
            m = np.maximum(s.max(axis=1), sink)
            e = np.exp(s - m[:, None])
            denom = e.sum(axis=1) + np.exp(sink - m)
            out[start + i] = (e @ k) / denom[:, None]
    return out


def test_one_record_pages_match_dense_reference_across_requests():
    # Two prefixed requests, one fresh request (records appear via boundary
    # writes), one decode row; several 32-query blocks each, one 128-entry tile.
    reqs = [(1536, 300), (0, 200), (640, 130), (100, 1)]
    args, aux = _build(reqs, 1, jax.random.PRNGKey(7))
    sched = get_hca_kernel_schedule(
        "TPU7x", page_size=1, max_compressed_entries=64, local_heads=H, head_dim=D
    )
    assert sched.compressed_tile == 128
    ref = _reference(reqs, args, aux)  # before the call: the caches are donated
    out = ragged_attention(*args, schedule=sched, softmax_scale=D**-0.5, page_size=P)[0]
    out = np.asarray(out.astype(jnp.float32))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, rtol=2e-2, atol=2e-2)


def test_flat_one_record_pool_matches_the_physical_view():
    """The kernels accept the compressed pool as flat [rows, D] with one record per
    page (its native layout; the 4-D view costs a whole-pool copy per layer) and
    produce the same attention and the same cache contents, bit for bit."""
    reqs = [(1536, 300), (0, 200), (640, 130), (100, 1)]
    sched = get_hca_kernel_schedule(
        "TPU7x", page_size=1, max_compressed_entries=64, local_heads=H, head_dim=D
    )
    args4, _ = _build(reqs, 1, jax.random.PRNGKey(7))
    args2, _ = _build(reqs, 1, jax.random.PRNGKey(7))
    flat = args2[3].reshape(-1, D)
    args2 = args2[:3] + (flat,) + args2[4:]
    out4, win4, comp4 = ragged_attention(*args4, schedule=sched, softmax_scale=D**-0.5, page_size=P)
    out2, win2, comp2 = ragged_attention(
        *args2, schedule=sched, softmax_scale=D**-0.5, page_size=P, compressed_page_size=1
    )
    assert comp2.shape == flat.shape and comp4.ndim == 4
    np.testing.assert_array_equal(np.asarray(out2), np.asarray(out4))
    np.testing.assert_array_equal(np.asarray(win2), np.asarray(win4))
    np.testing.assert_array_equal(np.asarray(comp2), np.asarray(comp4).reshape(-1, D))
