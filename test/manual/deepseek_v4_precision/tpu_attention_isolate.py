"""Feed GPU's actual Q and decoded cache to TPU attention kernels, without projections."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from common import Capture, Checkpoint, compare
from sgl_jax.srt.layers.attention.dsv4.attention import dsv4_attention
from sgl_jax.srt.kernels.hca.attention import ragged_attention
from sgl_jax.srt.kernels.hca.hca import HCAMetadata
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule
from sgl_jax.srt.layers.attention.hca_backend import _query_schedule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.reference)
    arrays = json.loads((root / "arrays.json").read_text())
    cp = Checkpoint(args.model)
    assert cp.identity == json.loads((root / "run.json").read_text())["checkpoint"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = Capture(out / "actual")
    rows, errors = [], []

    def ref(name):
        item = arrays[name]
        f = root / item["file"]
        assert hashlib.sha256(f.read_bytes()).hexdigest() == item["sha256"]
        return np.load(f, allow_pickle=False)

    def ints(a):
        return jnp.asarray(a, jnp.int32)

    for key in arrays:
        if not key.endswith("/raw_attention"):
            continue
        name = key.rsplit("/", 1)[0]
        layer = int(name.split("/")[1][1:])
        ratio = cp.config["compress_ratios"][layer]
        try:
            q = jnp.asarray(ref(name + "/q"), jnp.bfloat16).reshape(-1, 64, 512)
            positions = ints(ref(name + "/positions"))
            count = q.shape[0]
            end = int(np.asarray(positions)[-1]) + 1
            window = jnp.asarray(ref(name + "/swa_cache"), jnp.bfloat16).reshape(-1, 512)
            compressed = (
                jnp.asarray(ref(name + "/compressed_cache"), jnp.bfloat16).reshape(-1, 512)
                if end // ratio
                else jnp.zeros((0, 512), jnp.bfloat16)
            )
            sink = jnp.asarray(cp.read(f"layers.{layer}.attn.attn_sink"), jnp.float32)
            # <=32 compressed keys: native CSA admits all completed groups (topk=512).
            assert end // ratio <= cp.config["index_topk"]

            def xla(q, w, c, sink):
                return dsv4_attention(
                    q,
                    w,
                    c,
                    query_positions=positions,
                    query_request_ids=ints(np.zeros(count)),
                    valid_token_mask=jnp.ones(count, bool),
                    window_positions=ints(np.arange(end)),
                    window_request_ids=ints(np.zeros(end)),
                    compressed_entry_ids=ints(np.arange(end // ratio)),
                    compressed_request_ids=ints(np.zeros(end // ratio)),
                    attention_sink=sink,
                    softmax_scale=512**-0.5,
                    window_size=128,
                    ratio=ratio,
                )

            result = jax.jit(xla)(q, window, compressed, sink)
            result.block_until_ready()
            expected = ref(key).reshape(result.shape)
            cap.save(name + "/same_qkv_xla", result)
            row = dict(
                case=name, kernel="dsv4_attention_XLA", **compare(np.asarray(result), expected)
            )
            rows.append(row)
            print("ISOLATED_METRIC", json.dumps(row), flush=True)
            if ratio == 128:
                schedule = get_hca_kernel_schedule(
                    jax.devices()[0].device_kind,
                    page_size=128,
                    max_compressed_entries=2,
                    local_heads=64,
                    head_dim=512,
                )
                blocks, offsets, decodes = _query_schedule(
                    np.array([0, count], np.int32), schedule.query_block_size
                )
                md = HCAMetadata(
                    state_slots=ints([0]),
                    query_seq_ids=ints(np.zeros(count)),
                    cu_q_lens=ints([0, count]),
                    valid_token_mask=jnp.ones(count, bool),
                    boundary_token_indices=ints([]),
                    window_page_indices=ints([1, 2]),
                    window_cu_kv_lens=ints([0, 256]),
                    seq_lens=ints([end]),
                    compressed_page_indices=ints([1]),
                    compressed_cu_kv_lens=ints([0, 1]),
                    compressed_kv_lens=ints([end // 128]),
                    query_block_request_ids=ints(blocks),
                    query_block_offsets=ints(offsets),
                    decode_request_ids=ints(decodes),
                    max_queries_per_request=max(count, 1),
                )
                window_cache = (
                    jnp.zeros((3 * 128, 512), jnp.bfloat16)
                    .at[128 : 128 + end]
                    .set(window)
                    .reshape(3, 64, 2, 512)
                )
                compressed_cache = jnp.zeros((2, 1, 1, 512), jnp.bfloat16)
                if end // 128:
                    compressed_cache = compressed_cache.at[1, 0, 0].set(compressed[0])
                new_kv = window[positions]

                def pallas(q, k, w, c, sink, md):
                    return ragged_attention(
                        q,
                        k,
                        w,
                        c,
                        positions,
                        jnp.zeros((count, 512), jnp.bfloat16),
                        jnp.zeros(count, bool),
                        sink,
                        md,
                        schedule=schedule,
                        softmax_scale=512**-0.5,
                    )[0]

                actual = jax.jit(pallas)(q, new_kv, window_cache, compressed_cache, sink, md)
                actual.block_until_ready()
                cap.save(name + "/same_qkv_pallas", actual)
                row = dict(
                    case=name,
                    kernel="HCA_ragged_attention_Pallas",
                    **compare(np.asarray(actual), expected),
                )
                rows.append(row)
                print("ISOLATED_METRIC", json.dumps(row), flush=True)
        except Exception as e:
            traceback.print_exc()
            errors.append(dict(case=name, type=type(e).__name__, message=str(e)))
        finally:
            (out / "metrics.json").write_text(json.dumps(rows, indent=2))
            (out / "errors.json").write_text(json.dumps(errors, indent=2))
    if errors:
        raise RuntimeError("Isolated attention comparisons failed")


if __name__ == "__main__":
    main()
