# GLM-5.2 DSA Startup Debugging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or execute each task inline with the same RED/GREEN checkpoints.

**Goal:** Make Falcon startup progress observable, remove repeated GCSFuse safetensors-header scans with a validated metadata sidecar, and prove the change with fresh-node GLM-5.2 DSA E2E runs.

**Architecture:** Keep the existing server process and artifact log unchanged, but mirror only rank-0 startup milestones to Falcon stdout through a separately managed `tail -F` process. Add a versioned gzip-JSON sidecar containing tensor shapes, dtypes, byte ranges, and shard fingerprints; process 0 loads it before falling back to the existing header scanner, then broadcasts the same in-memory structure as today. Generate the sidecar once from a writable CPU pod and consume it read-only on TPU workers.

**Tech Stack:** Bash, Python 3.12, pytest, JAX multi-host broadcast, GCSFuse, Falcon v7x32.

---

### Task 1: Stream rank-0 startup milestones

**Files:**
- Modify: `scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh`
- Test: `python/sgl_jax/test/test_glm52_e2e_compare.py`

- [ ] **Step 1: Write the failing runner-contract test**

```python
def test_real_runner_streams_rank_zero_startup_progress_to_falcon_stdout():
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert 'SERVER_LOG_MONITOR_PID=""' in runner
    assert 'start_server_log_monitor()' in runner
    assert 'stop_server_log_monitor()' in runner
    assert 'if [[ "$RANK" != "0" ]]; then' in runner
    assert 'tail -n +1 -F "$SERVER_LOG"' in runner
    assert 'Scanning metadata|Starting parallel weight loading' in runner
    assert 'Precompile finished|Application startup complete' in runner
    assert runner.index('start_server_log_monitor') < runner.index('health_deadline=')
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest -q python/sgl_jax/test/test_glm52_e2e_compare.py::test_real_runner_streams_rank_zero_startup_progress_to_falcon_stdout`

Expected: FAIL because the monitor lifecycle and `tail -F` pipeline are absent.

- [ ] **Step 3: Add the minimal monitor lifecycle**

```bash
SERVER_LOG_MONITOR_PID=""

start_server_log_monitor() {
  if [[ "$RANK" != "0" ]]; then
    return
  fi
  tail -n +1 -F "$SERVER_LOG" 2>/dev/null \
    | grep --line-buffered -E \
      'Scanning metadata|Scanning Metadata:.*100%|Starting parallel weight loading|All weights loaded successfully|Absorbed MLA weights|\[(EXTEND|DECODE)\].*(PRECOMPILE|Precompile finished)|The server is fired up|Application startup complete' &
  SERVER_LOG_MONITOR_PID=$!
}

stop_server_log_monitor() {
  if [[ -n "$SERVER_LOG_MONITOR_PID" ]]; then
    kill "$SERVER_LOG_MONITOR_PID" 2>/dev/null || true
    wait "$SERVER_LOG_MONITOR_PID" 2>/dev/null || true
    SERVER_LOG_MONITOR_PID=""
  fi
}
```

Call `start_server_log_monitor` immediately after launching the server. Call `stop_server_log_monitor` from `finish_server` before copying the complete server log.

- [ ] **Step 4: Verify GREEN and shell syntax**

Run: `python -m pytest -q python/sgl_jax/test/test_glm52_e2e_compare.py`

Run: `bash -n scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh`

Expected: all tests pass and shell syntax exits 0.

- [ ] **Step 5: Commit the observable runner**

```bash
git add scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh python/sgl_jax/test/test_glm52_e2e_compare.py
git commit -m "chore(dsa): stream Falcon startup progress"
```

### Task 2: Load a validated safetensors metadata sidecar

**Files:**
- Modify: `python/sgl_jax/srt/utils/weight_utils.py`
- Create: `scripts/models/build_safetensors_metadata_cache.py`
- Test: `python/sgl_jax/test/test_weight_utils_metadata_cache.py`

- [ ] **Step 1: Write failing round-trip and stale-cache tests**

```python
from pathlib import Path

from sgl_jax.srt.utils.weight_utils import (
    _load_safetensors_metadata_cache,
    _write_safetensors_metadata_cache,
)


def test_safetensors_metadata_cache_round_trip(tmp_path: Path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"checkpoint")
    cache = tmp_path / "sglang_jax.safetensors_metadata.v1.json.gz"
    expected = {
        "weight": [
            {
                "file": str(shard),
                "shape": (2, 3),
                "dtype": "BF16",
                "byte_offset": 128,
                "byte_size": 12,
            }
        ]
    }

    _write_safetensors_metadata_cache(cache, [str(shard)], expected)

    assert _load_safetensors_metadata_cache(cache, [str(shard)]) == expected


def test_safetensors_metadata_cache_rejects_changed_shard(tmp_path: Path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"old")
    cache = tmp_path / "sglang_jax.safetensors_metadata.v1.json.gz"
    _write_safetensors_metadata_cache(
        cache,
        [str(shard)],
        {"weight": [{"file": str(shard), "shape": (1,), "dtype": "BF16"}]},
    )
    shard.write_bytes(b"new-size")

    assert _load_safetensors_metadata_cache(cache, [str(shard)]) is None
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest -q python/sgl_jax/test/test_weight_utils_metadata_cache.py`

Expected: collection FAIL because the sidecar helpers do not exist.

- [ ] **Step 3: Implement versioned gzip-JSON serialization**

```python
_SAFETENSORS_METADATA_CACHE_VERSION = 1
_SAFETENSORS_METADATA_CACHE_BASENAME = "sglang_jax.safetensors_metadata.v1.json.gz"


def _safetensors_shard_fingerprint(weights_files: list[str]) -> list[dict[str, Any]]:
    return [
        {"name": os.path.basename(path), "size": os.path.getsize(path)}
        for path in weights_files
    ]


def _write_safetensors_metadata_cache(path, weights_files, weight_info):
    import gzip

    root = os.path.dirname(weights_files[0])
    serializable = {
        key: [
            {
                **info,
                "file": os.path.relpath(info["file"], root),
                "shape": list(info["shape"]),
            }
            for info in infos
        ]
        for key, infos in weight_info.items()
    }
    payload = {
        "schema_version": _SAFETENSORS_METADATA_CACHE_VERSION,
        "shards": _safetensors_shard_fingerprint(weights_files),
        "weight_info": serializable,
    }
    with gzip.open(path, "wt", encoding="utf-8") as fp:
        json.dump(payload, fp, separators=(",", ":"), sort_keys=True)


def _load_safetensors_metadata_cache(path, weights_files):
    import gzip

    try:
        with gzip.open(path, "rt", encoding="utf-8") as fp:
            payload = json.load(fp)
        if payload.get("schema_version") != _SAFETENSORS_METADATA_CACHE_VERSION:
            return None
        if payload.get("shards") != _safetensors_shard_fingerprint(weights_files):
            return None
        root = os.path.dirname(weights_files[0])
        return {
            key: [
                {
                    **info,
                    "file": os.path.join(root, info["file"]),
                    "shape": tuple(info["shape"]),
                }
                for info in infos
            ]
            for key, infos in payload["weight_info"].items()
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None
```

Process 0 checks `SGLANG_JAX_SAFETENSORS_METADATA_CACHE`, then the default basename under `model_path`. On a valid hit it logs elapsed time and skips `_scan_safetensors_metadata`; on any missing, stale, or malformed cache it logs the fallback and runs the existing scanner.

- [ ] **Step 4: Add the one-shot sidecar builder**

```python
#!/usr/bin/env python3
import argparse
import glob
import os

from sgl_jax.srt.utils.weight_utils import (
    _scan_safetensors_metadata,
    _write_safetensors_metadata_cache,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    files = sorted(glob.glob(os.path.join(args.model_path, "*.safetensors")))
    if not files:
        raise SystemExit(f"no safetensors shards under {args.model_path}")
    output = args.output or os.path.join(
        args.model_path, "sglang_jax.safetensors_metadata.v1.json.gz"
    )
    info = _scan_safetensors_metadata(files, num_threads=args.threads, show_progress=True)
    _write_safetensors_metadata_cache(output, files, info)
    print(f"wrote {output}: shards={len(files)} tensors={len(info)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify the unit and loader suites**

Run: `python -m pytest -q python/sgl_jax/test/test_weight_utils_metadata_cache.py python/sgl_jax/test/test_model_loader_gcsfuse_warmup.py`

Run: `python -m compileall -q python/sgl_jax/srt/utils/weight_utils.py scripts/models/build_safetensors_metadata_cache.py`

Expected: all tests pass and compileall exits 0.

- [ ] **Step 6: Commit the sidecar support**

```bash
git add python/sgl_jax/srt/utils/weight_utils.py scripts/models/build_safetensors_metadata_cache.py python/sgl_jax/test/test_weight_utils_metadata_cache.py
git commit -m "feat(loader): cache safetensors metadata"
```

### Task 3: Populate and validate on Falcon

**Files:**
- Modify: `note/2026-07-18-glm52-dsa-falcon-results.md`
- Create outside git: a Falcon CPU sidecar-generation manifest and a fresh-node v7x32 E2E manifest.

- [ ] **Step 1: Build the sidecar once on a writable CPU pod**

Run the committed builder against `/models/GLM-5.2` with 32 threads and write `sglang_jax.safetensors_metadata.v1.json.gz` beside the checkpoint. Record file size, SHA-256, shard count, tensor count, and generation duration.

- [ ] **Step 2: Submit a fresh-node v7x32 E2E run**

Use the same TP32/EP32 configuration and 3072-token repeated request as `exp-mzbjj4o3f6`. Preserve `--precompile-dsa-context-paddings 512 1024 2048 4096` and `--precompile-top-logprobs 20`. Capture live stdout milestones and the final artifact.

- [ ] **Step 3: Apply the acceptance gates**

Expected evidence:

```text
metadata cache hit: under 30 seconds
EXTEND precompile: 8/8 variants complete
DECODE precompile: 5/5 variants complete
first and repeat output_ids: equal
max output logprob absolute error: 0
max shared top-20 logprob absolute error: 0
first/repeat latency: no first-only compile penalty
```

- [ ] **Step 4: Run a warm-node confirmation only if the fresh run is ambiguous**

Reuse the same node pool and source revision. Change no model or precompile arguments. Compare only metadata/load timing to separate sidecar behavior from GCSFuse data-cache variance.

- [ ] **Step 5: Document and commit the measured result**

```bash
git add note/2026-07-18-glm52-dsa-falcon-results.md
git commit -m "docs(dsa): record startup cache validation"
```

The note must include experiment, job, artifact, source commit, cold/warm phase timing, request outputs, logprob gates, and any remaining `cache_miss_count` interpretation.
