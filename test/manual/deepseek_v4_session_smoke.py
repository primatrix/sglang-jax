"""Compare session continuation with cold generation on an already-running V4 server.

Run separately with overlap enabled and disabled. Does not launch a server or
change its configuration. Uses token IDs to avoid chat-template prefix changes.
"""

import argparse
import json
import uuid
from urllib.request import Request, urlopen


def post(base, path, payload):
    req = Request(
        base.rstrip("/") + path,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=1800) as response:
        return json.load(response)


def generate(base, tokens, sid=None):
    payload = {
        "input_ids": tokens,
        "sampling_params": {"temperature": 0, "max_new_tokens": 4, "ignore_eos": True},
    }
    if sid is not None:
        payload["session_params"] = {"id": sid}
    result = post(base, "/generate", payload)
    reason = result["meta_info"]["finish_reason"]
    if reason["type"] == "abort":
        raise RuntimeError(reason)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[127, 128, 129, 4096])
    parser.add_argument("--extend-length", type=int, default=64)
    parser.add_argument("--token-id", type=int, default=42)
    args = parser.parse_args()
    if min(args.prefix_lengths) <= 0 or args.extend_length <= 0:
        parser.error("Lengths must be positive")
    for length in args.prefix_lengths:
        sid = uuid.uuid4().hex
        try:
            prefix = [args.token_id] * length
            first = generate(args.url, prefix, sid)
            full = prefix + first["output_ids"] + [args.token_id] * args.extend_length
            warm = generate(args.url, full, sid)
            cold = generate(args.url, full)
            assert warm["output_ids"] == cold["output_ids"], "Warm/cold output mismatch"
            hit = warm["meta_info"]["cached_tokens"]
            assert hit >= length, f"Expected prefix hit, got {hit}"
            changed = full.copy()
            changed[0] = args.token_id + 1
            miss = generate(args.url, changed, sid)
            assert miss["meta_info"]["cached_tokens"] == 0
            assert miss["output_ids"] == generate(args.url, changed)["output_ids"]
            shorter = generate(args.url, prefix[: max(1, length // 2)], sid)
            assert shorter["meta_info"]["cached_tokens"] == 0
            print(json.dumps({"prefix_length": length, "cached_tokens": hit, "passed": True}))
        finally:
            post(
                args.url,
                "/close_session",
                {"request_id": uuid.uuid4().hex, "session_id": sid},
            )


if __name__ == "__main__":
    main()
