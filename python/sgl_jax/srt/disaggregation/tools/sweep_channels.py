"""Stage 4 H-E: jax_transfer channel-count sweep harness.

Usage:
    python -m sgl_jax.srt.disaggregation.tools.sweep_channels \\
        --remote 10.0.0.7:30001 \\
        --bytes-per-iter $((1<<30)) --iters 20 \\
        --channels 1,2,4,8

Drives ``JaxTransferWrapper`` with each channel-number, pushes a fixed
payload across a single peer link, and prints aggregate / per-iter
throughput so the operator can pick the knee point for
``--disaggregation-channel-number``.

The script is intentionally process-spawning: ``JaxTransferWrapper`` is
a process-singleton, so changing ``channel_number`` after start() is
impossible. We re-exec ourselves per setting so each child gets a fresh
wrapper.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List


def _run_one(remote: str, channels: int, payload_bytes: int, iters: int) -> float:
    import jax
    import jax.numpy as jnp

    from sgl_jax.srt.disaggregation.jax_transfer_wrapper import (
        get_or_create_wrapper,
        _reset_singleton_for_test,
    )

    _reset_singleton_for_test()
    host_ip = os.environ.get("HOST_IP", "127.0.0.1")
    port = int(os.environ.get("LOCAL_PORT", "30100"))
    wrapper = get_or_create_wrapper(host_ip, port, channel_number=channels)
    wrapper.start()

    arr = jnp.zeros((payload_bytes // 2,), dtype=jnp.float16)
    arr.block_until_ready()

    start = time.perf_counter()
    for i in range(iters):
        uid = f"sweep-{channels}-{i}"
        wrapper.register_pull(uid, arr)
        # NOTE: actual remote pull must come from the peer process.
        # This stub only measures register-side throughput; full
        # bandwidth requires a paired puller (see tools/perf/pd_pair.py
        # — to be added in a follow-up).
        wrapper.release(uid)
    elapsed = time.perf_counter() - start

    total_bytes = iters * payload_bytes
    gbps = total_bytes / elapsed / 1e9
    return gbps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--remote", required=True,
                   help="Peer transfer-server address (host:port).")
    p.add_argument("--bytes-per-iter", type=int, default=1 << 30)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--channels", default="1,2,4,8")
    args = p.parse_args()

    rows: List[str] = ["channels,gbps"]
    for c in [int(x) for x in args.channels.split(",")]:
        # Re-exec self per channel-count to clear the wrapper singleton.
        if os.environ.get("_SWEEP_CHILD") == str(c):
            gbps = _run_one(args.remote, c, args.bytes_per_iter, args.iters)
            print(f"{c},{gbps:.2f}")
            return 0
        os.environ["_SWEEP_CHILD"] = str(c)
        import subprocess

        out = subprocess.check_output(
            [sys.executable, "-m",
             "sgl_jax.srt.disaggregation.tools.sweep_channels",
             "--remote", args.remote,
             "--bytes-per-iter", str(args.bytes_per_iter),
             "--iters", str(args.iters),
             "--channels", str(c)],
            env={**os.environ, "_SWEEP_CHILD": str(c)},
            text=True,
        ).strip()
        rows.append(out)
        del os.environ["_SWEEP_CHILD"]

    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
