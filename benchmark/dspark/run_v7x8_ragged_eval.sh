#!/usr/bin/env bash
set -euo pipefail

cd /tmp/sglang-jax
export TMPDIR="${TMPDIR:-/tmp/tpu_logs/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/tpu_logs/pip-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/tpu_logs/uv-cache}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export HF_HOME="${HF_HOME:-/tmp/tpu_logs/huggingface}"
export SGLANG_RECORD_STEP_TIME=1
RESULT_ROOT="${DSPARK_RESULT_ROOT:-/tmp/dspark_ragged_deepspec}"
export DSPARK_RESULT_ROOT="$RESULT_ROOT"
export DSPARK_ENABLE_TUNED="${DSPARK_ENABLE_TUNED:-1}"
SERVER_LOG="$RESULT_ROOT/server.log"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$RESULT_ROOT"
ulimit -c 0

test "$(git rev-parse HEAD)" = "${DSPARK_EXPECTED_COMMIT:-aca8d04fe1bad03e93e4630999dcc0234f2b3178}"
uv sync --project python --extra tpu
uv tool install --force evalscope==1.9.0
DEEPSPEC_COMMIT="${DEEPSPEC_COMMIT:-005e03b81cec38b7da6399833d609ee89a2587f2}"
git clone https://github.com/deepseek-ai/DeepSpec.git /tmp/DeepSpec
git -C /tmp/DeepSpec checkout --detach "$DEEPSPEC_COMMIT"

TUNED_ARGS=()
if [ "$DSPARK_ENABLE_TUNED" = "1" ]; then
  TUNED_ARGS+=(--enable-dspark-tuned-config)
fi

python/.venv/bin/python -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-8B \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path deepseek-ai/dspark_qwen3_8b_block7 \
  --speculative-num-steps 1 \
  --speculative-num-draft-tokens 7 \
  --speculative-eagle-topk 1 \
  --tp-size 8 \
  --dp-size 2 \
  --dtype bfloat16 \
  --attention-backend fa \
  --mem-fraction-static 0.65 \
  --page-size 64 \
  --chunked-prefill-size 2048 \
  --max-running-requests 128 \
  --context-length 16384 \
  --precompile-token-paddings 2048 \
  --decode-log-interval 1 \
  --trust-remote-code \
  --disable-radix-cache \
  --grammar-backend none \
  --disable-precompile \
  "${TUNED_ARGS[@]}" \
  --host 0.0.0.0 \
  --port 30000 >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 600); do
  status=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health || true)
  if [ "$status" = "200" ]; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -300 "$SERVER_LOG"
    exit 1
  fi
  sleep 2
done
test "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health || true)" = "200"
curl -s http://127.0.0.1:30000/v1/models >"$RESULT_ROOT/models.json"
ps -o args= -p "$SERVER_PID" >"$RESULT_ROOT/server_args.txt"

cat >"$RESULT_ROOT/run_prompts.py" <<'PY'
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--concurrency", default=64, type=int)
    parser.add_argument("--seed", default=980406, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = Path("/tmp/DeepSpec/eval_datasets") / f"{args.dataset}.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if len(rows) > args.limit:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]

    semaphore = asyncio.Semaphore(args.concurrency)
    outputs = [None] * len(rows)
    timeout = httpx.Timeout(1800.0)
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def run_one(index: int, row: dict) -> None:
            async with semaphore:
                started = time.perf_counter()
                response = await client.post(
                    "http://127.0.0.1:30000/v1/chat/completions",
                    json={
                        "model": "Qwen/Qwen3-8B",
                        "messages": [{"role": "user", "content": row["turns"][0]}],
                        "max_tokens": 2048,
                        "temperature": 0.0,
                        "seed": args.seed + index,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                response.raise_for_status()
                body = response.json()
                outputs[index] = {
                    "index": index,
                    "latency_s": time.perf_counter() - started,
                    "finish_reason": body["choices"][0]["finish_reason"],
                    "usage": body.get("usage"),
                }

        started = time.perf_counter()
        await asyncio.gather(*(run_one(index, row) for index, row in enumerate(rows)))
        elapsed = time.perf_counter() - started

    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in outputs),
        encoding="utf-8",
    )
    print(json.dumps({
        "dataset": args.dataset,
        "samples": len(rows),
        "seed": args.seed,
        "concurrency": args.concurrency,
        "elapsed_s": elapsed,
    }))


asyncio.run(main())
PY

: >"$RESULT_ROOT/ranges.tsv"
for spec in gsm8k:500 math500:500 aime25:30; do
  dataset=${spec%%:*}
  limit=${spec##*:}
  start=$(stat -c %s "$SERVER_LOG")
  python/.venv/bin/python "$RESULT_ROOT/run_prompts.py" \
    --dataset "$dataset" \
    --limit "$limit" \
    --concurrency 64 \
    --seed 980406 \
    --output "$RESULT_ROOT/${dataset}_responses.jsonl" \
    >"$RESULT_ROOT/${dataset}_client.json"
  sleep 3
  end=$(stat -c %s "$SERVER_LOG")
  printf '%s\t%s\t%s\n' "$dataset" "$start" "$end" >>"$RESULT_ROOT/ranges.tsv"
done

cleanup
trap - EXIT
printf '%s\n' "$SERVER_PID" >"$RESULT_ROOT/server_pid.txt"

python/.venv/bin/python - <<'PY'
import json
import re
import subprocess
from importlib.metadata import version
from pathlib import Path

import jax
import sgl_jax

root = Path(__import__("os").environ["DSPARK_RESULT_ROOT"])
server_log = (root / "server.log").read_bytes()
decode_pattern = re.compile(
    rb"#running-req: (\d+).*?accept-len ([0-9.]+), accept-ratio ([0-9.]+)"
)
results = []
for line in (root / "ranges.tsv").read_text().splitlines():
    dataset, start, end = line.split("\t")
    matches = decode_pattern.findall(server_log[int(start):int(end)])
    samples = [(int(n), float(length), float(ratio)) for n, length, ratio in matches]
    proposal_count = sum(n for n, _, _ in samples)
    client = json.loads((root / f"{dataset}_client.json").read_text())
    response_rows = sum(1 for _ in (root / f"{dataset}_responses.jsonl").open())
    results.append({
        **client,
        "validated_response_rows": response_rows,
        "decode_log_rows": len(samples),
        "proposal_count": proposal_count,
        "acceptance_length": (
            sum(n * length for n, length, _ in samples) / proposal_count
            if proposal_count else None
        ),
        "accept_ratio": (
            sum(n * ratio for n, _, ratio in samples) / proposal_count
            if proposal_count else None
        ),
    })

text = server_log.decode(errors="replace")
summary = {
    "schema_version": 1,
    "state": "success" if all(r["samples"] == r["validated_response_rows"] for r in results) else "partial",
    "exp_commit": subprocess.check_output(
        ["git", "-C", "/tmp/sglang-jax", "rev-parse", "HEAD"], text=True
    ).strip(),
    "branch": "epic/dspark-qwen3",
    "mode": "ragged" if __import__("os").environ.get("DSPARK_ENABLE_TUNED", "1") == "1" else "fixed_verify_all",
    "deepspec_commit": subprocess.check_output(
        ["git", "-C", "/tmp/DeepSpec", "rev-parse", "HEAD"], text=True
    ).strip(),
    "model": "Qwen/Qwen3-8B",
    "draft": "deepseek-ai/dspark_qwen3_8b_block7",
    "hardware": {"type": "v7x", "topology": "2x2x1", "chips": 4, "jax_devices": 8},
    "versions": {
        "sglang_jax": sgl_jax.__version__,
        "jax": jax.__version__,
        "libtpu": version("libtpu"),
        "evalscope": "1.9.0",
    },
    "server_pid": int((root / "server_pid.txt").read_text()),
    "tuned_config_loaded": "Using DSPARK tuned config" in text,
    "ragged_plans": sorted(set(re.findall(r"DSPARK ragged verify plan:.*", text))),
    "fixed_verify_fallbacks": len(re.findall(r"keeping fixed verify-all", text)),
    "datasets": results,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

/root/.local/bin/evalscope --version >"$RESULT_ROOT/evalscope_version.txt"
git rev-parse HEAD >"$RESULT_ROOT/git_commit.txt"
touch "$RESULT_ROOT/READY"
echo "DSPARK_RAGGED_DEEPSPEC_READY=$RESULT_ROOT"
while [ ! -f "$RESULT_ROOT/COLLECTED" ]; do sleep 2; done
