"""Focused page-address/score/selection/sink probe, followed separately by the real-weight layer A/B."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from sgl_jax.srt.kernels.csa_decode import paged_csa_decode_scores
from sgl_jax.srt.layers.attention.dsv4.attention import dsv4_attention
from sgl_jax.srt.layers.attention.dsv4.decode import csa_decode_attention
from sgl_jax.srt.layers.attention.dsv4.indexer import (
    csa_indexer_scores,
    csa_indexer_topk,
)

parser = argparse.ArgumentParser()
parser.add_argument("--large", action="store_true")
parser.add_argument("--require-tpu", action="store_true")
parser.add_argument("--out", type=Path)
a = parser.parse_args()
if a.require_tpu:
    assert jax.default_backend() == "tpu"
rng = np.random.default_rng(7129)
lengths = np.array([0, 1, 31, 128, 129, 2056 if a.large else 255], np.int32)
t, h, d, ad = len(lengths), 64, 128, 512
cap, budget = (4096, 2048) if a.large else (256, 64)
page_size = 32
pages = (
    rng.permutation(np.arange(1, 1 + t * cap // page_size))
    .reshape(t, cap // page_size)
    .astype(np.int32)
)
key_cache = jnp.asarray(
    rng.normal(0, 0.1, (1 + t * cap // page_size, page_size, d)), jnp.bfloat16
)
compressed_cache = jnp.asarray(
    rng.normal(0, 0.1, (key_cache.shape[0] * page_size, ad)), jnp.bfloat16
)
window_cache = jnp.asarray(rng.normal(0, 0.1, (1 + t * 128, ad)), jnp.bfloat16)
window_rows = jnp.arange(1, 1 + t * 128, dtype=jnp.int32).reshape(t, 128)
positions = jnp.asarray(lengths * 4 + 2)
valid = jnp.asarray(lengths > 0)
query_ids = jnp.arange(t, dtype=jnp.int32)
q = jnp.asarray(rng.normal(0, 0.1, (t, 8, ad)), jnp.bfloat16)
weights = jnp.asarray(rng.normal(0, 0.1, (t, h)), jnp.float32)
all_slots = (pages[:, :, None] * page_size + np.arange(page_size)).reshape(t, cap)
flat_slots = all_slots.reshape(-1)
flat_keys = key_cache.reshape(-1, d)[flat_slots]
entry_ids = jnp.tile(jnp.arange(cap, dtype=jnp.int32), t)
entry_requests = jnp.repeat(query_ids, cap)
window_positions = (positions[:, None] - 127 + jnp.arange(128)).reshape(-1)
rows = []
for dtype in (jnp.bfloat16, jnp.float32):
    for tie in (False, True):
        iq = jnp.asarray(rng.normal(0, 0.1, (t, h, d)), dtype)
        if tie:
            iq = jnp.zeros_like(iq)
        native = csa_indexer_scores(iq, weights, flat_keys)
        expected = jnp.stack([native[i, i * cap : (i + 1) * cap] for i in range(t)])
        scores = paged_csa_decode_scores(
            iq,
            weights,
            key_cache,
            jnp.asarray(lengths),
            jnp.asarray(pages),
            interpret=jax.default_backend() != "tpu",
        )
        jax.block_until_ready(scores)
        mask = np.arange(cap)[None, :] < lengths[:, None]
        sa, se = np.asarray(scores), np.asarray(expected)
        assert np.all(sa[~mask] == np.finfo(np.float32).min)
        max_abs = float(np.max(np.abs(sa[mask] - se[mask])))
        assert max_abs < 1e-5, max_abs
        reference_selected = csa_indexer_topk(
            iq,
            weights,
            flat_keys,
            positions,
            query_ids,
            entry_requests,
            valid,
            entry_group_ids=entry_ids,
            k=budget,
            ratio=4,
        )
        values, actual_selected = jax.lax.top_k(scores, min(budget, cap))
        actual_selected = jnp.where(
            (actual_selected < jnp.asarray(lengths)[:, None])
            & (values > jnp.finfo(jnp.float32).min),
            actual_selected + query_ids[:, None] * cap,
            -1,
        )
        reference_sorted = np.sort(np.asarray(reference_selected), axis=1)
        actual_sorted = np.sort(np.asarray(actual_selected), axis=1)
        assert np.array_equal(reference_sorted, actual_sorted), (
            str(dtype),
            tie,
            "selection mismatch",
        )
        for sink_value in (0.0, 1000.0, -1000.0):
            sink = jnp.full((8,), sink_value, jnp.float32)
            out = csa_decode_attention(
                q,
                iq,
                weights,
                key_cache.reshape(-1, d),
                compressed_cache,
                window_cache,
                jnp.asarray(pages),
                window_rows,
                query_positions=positions,
                valid_token_mask=valid,
                attention_sink=sink,
                softmax_scale=ad**-0.5,
                compressed_page_size=page_size,
                index_topk=budget,
                ratio=4,
            )
            ref = dsv4_attention(
                q,
                window_cache[window_rows.reshape(-1)],
                compressed_cache[flat_slots],
                query_positions=positions,
                query_request_ids=query_ids,
                valid_token_mask=valid,
                window_positions=window_positions,
                window_request_ids=jnp.where(
                    window_positions >= 0, jnp.repeat(query_ids, 128), -1
                ),
                compressed_entry_ids=entry_ids,
                compressed_request_ids=entry_requests,
                attention_sink=sink,
                softmax_scale=ad**-0.5,
                window_size=128,
                ratio=4,
                selected_entries=reference_selected,
            )
            oa, oe = np.asarray(out), np.asarray(ref)
            assert np.isfinite(oa).all()
            assert np.all(oa[0] == 0)
            rel = float(np.linalg.norm(oa - oe) / max(np.linalg.norm(oe), 1e-30))
            assert rel < 1e-5, rel
            rows.append(
                {
                    "dtype": str(dtype),
                    "ties": tie,
                    "sink": sink_value,
                    "max_score_abs": max_abs,
                    "selection_exact": True,
                    "output_rel_l2": rel,
                }
            )
            print("PROBE", json.dumps(rows[-1]), flush=True)
result = {"backend": jax.default_backend(), "large": a.large, "rows": rows}
if a.out:
    a.out.write_text(json.dumps(result, indent=2))
print("CSA_DECODE_PROBE_COMPLETE", flush=True)
