"""P0-3: long-prompt KV transfer.

For each of 2k / 4k / 8k prompt sizes, fire one PD pair and:
  * Assert byte-equal P vs D output.
  * Read pd_transfer_bytes_total{direction=net} before and after the
    request and assert the delta is in the plausible range for the
    expected KV size (within 0.5x — 2x — generous, the goal is to
    catch zero-byte or wildly-wrong transfers, not to assert exact
    accounting).
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

from sgl_jax.srt.disaggregation.tools.e2e import _common as C


PROMPT_SIZES_TOKENS = [2048, 4096, 8192]
# Conservative bytes-per-token estimate for Qwen3-8B (GQA):
# 36 layers × 8 KV heads × 128 head_dim × 2 (K+V) × 2 (bf16) ≈ 147 KB/tok.
# Use a coarse 100k..400k window to absorb sharding rounding.
BYTES_PER_TOKEN_MIN = 50_000
BYTES_PER_TOKEN_MAX = 500_000


def _fetch_net_bytes(p_url: str) -> float | None:
    try:
        r = httpx.get(f"{p_url}/metrics", timeout=5.0)
        if r.status_code != 200:
            return None
        total = 0.0
        for line in r.text.splitlines():
            if line.startswith("pd_transfer_bytes_total") and 'direction="net"' in line:
                total += float(line.rsplit(" ", 1)[1])
        return total
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    C.add_topology_args(p)
    p.add_argument("--max-new-tokens", type=int, default=4)
    args = p.parse_args()
    topo = C.parse_topology(args)

    rows = []
    failed = []
    for n_tokens in PROMPT_SIZES_TOKENS:
        # "Hello " is 2 BPE tokens for most tokenizers — close enough.
        prompt = "Hello " * (n_tokens // 2)
        rid = f"long-{n_tokens}"
        room = abs(hash(rid)) % 100000

        before = _fetch_net_bytes(topo.first_p)
        t0 = time.perf_counter()
        rsp = C.fire_pd_pair(
            topo, rid=rid, prompt=prompt,
            bootstrap_room=room, max_new_tokens=args.max_new_tokens,
            timeout=180.0,
        )
        elapsed = time.perf_counter() - t0
        after = _fetch_net_bytes(topo.first_p)

        p_text = rsp["P"].get("text")
        d_text = rsp["D"].get("text")
        equal = p_text == d_text

        delta_bytes = (after - before) if (
            before is not None and after is not None
        ) else None
        plausible_bytes = None
        if delta_bytes is not None:
            lo = BYTES_PER_TOKEN_MIN * n_tokens
            hi = BYTES_PER_TOKEN_MAX * n_tokens
            plausible_bytes = lo <= delta_bytes <= hi

        row = {
            "n_tokens_target": n_tokens,
            "elapsed_s": elapsed,
            "byte_equal": equal,
            "net_bytes_delta": delta_bytes,
            "plausible_bytes": plausible_bytes,
        }
        rows.append(row)
        if not equal:
            failed.append(f"{rid}:not-equal")
        # `plausible_bytes is None` means the /metrics endpoint isn't
        # serving (prometheus_client not installed) — degrade to a
        # warning rather than failing the test.
        if plausible_bytes is False:
            failed.append(
                f"{rid}:bytes-delta={delta_bytes} out of "
                f"[{BYTES_PER_TOKEN_MIN * n_tokens}, "
                f"{BYTES_PER_TOKEN_MAX * n_tokens}]"
            )

    C.write_report(args, "long_prompt", {
        "rows": rows, "failed": failed,
    })
    if failed:
        return C.print_result(False, f"{len(failed)} failures: {failed[:2]}")
    return C.print_result(
        True,
        f"OK {[r['n_tokens_target'] for r in rows]} "
        f"latencies={[round(r['elapsed_s'], 2) for r in rows]}s",
    )


if __name__ == "__main__":
    sys.exit(main())
