# GLM-5.2 DSA Non-Kernel Profiling Implementation Plan

> **For Codex:** Execute this plan with test-driven changes, Falcon's `falcon-workflow`
> profile lifecycle, and evidence-first optimization. Do not optimize the DSA Pallas kernel
> in this phase.

**Goal:** Produce a reproducible Falcon v7x-32 profile that separates GLM-5.2 prefill,
decode, host/Python overhead, and device execution; then use that evidence to select and
fix the largest non-DSA-kernel bottleneck without weakening correctness gates.

**Architecture:** Extend the existing four-rank real-weight runner with a dedicated
`profile` request mode. A small testable client performs an exact-shape warm-up, starts
stage-based JAX profiling, sends the measured request, waits for both prefill and decode
traces to flush, and records wall-clock events. Each rank writes traces to host-local
`/tmp/tpu_logs` and copies them into its rank-scoped Falcon artifact during teardown. A
Falcon `PROFILING` manifest runs the pinned source revision and the `xprof-summary` plugin
analyzes the collected trace.

**Tech Stack:** Bash, Python standard library, pytest, JAX profiler/XPlane, sglang-jax
HTTP profiling endpoints, Falcon workflow CLI, Falcon `xprof-summary`.

---

## Task 1: Specify the profiling client and runner contract

**Files:**

- Create: `scripts/kernels/profile_glm52_dsa_server.py`
- Modify: `python/sgl_jax/test/test_glm52_e2e_compare.py`

### Step 1: Write failing client orchestration tests

Add tests using a fake HTTP transport that require this sequence:

1. send `profile_warmup.request.json` to `/generate` and persist its response;
2. POST `/start_profile` with `profile_by_stage=true`, stages `prefill/decode`, explicit
   host/Python tracer levels, and a host-local trace directory;
3. send `profile_measured.request.json` and persist its response;
4. poll `/profile_status` until `idle`;
5. persist monotonic event timestamps, request durations, and profile settings in
   `profile_timeline.json`;
6. call `/stop_profile` and fail if the profile deadline expires.

Run:

```bash
../../.venv/bin/python -m pytest -q \
  python/sgl_jax/test/test_glm52_e2e_compare.py -k profile_client
```

Expected: FAIL because the client does not exist.

### Step 2: Implement the minimum client

Implement the orchestration with an injectable transport and monotonic clock. Keep the
CLI dependency-free by using the Python standard library. Write response JSON atomically
enough for a single rank-0 writer and include the final `idle` state in the timeline.

### Step 3: Run the focused tests

Run the same pytest command and require PASS.

## Task 2: Add an exact-shape profiling mode to the real-weight runner

**Files:**

- Modify: `scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh`
- Modify: `python/sgl_jax/test/test_glm52_e2e_compare.py`

### Step 1: Write failing runner contract tests

Require a new `GLM52_DSA_REQUEST_PROFILE=profile` that generates two identical 3072-token
requests named `profile_warmup` and `profile_measured`. Require at least
`profile_steps + 2` decode tokens so the stage trigger has a later decode step on which to
stop and flush. Assert the runner:

- uses `/tmp/tpu_logs` for active trace writes;
- sets `host_tracer_level=2` and `python_tracer_level=1` by default;
- invokes `profile_glm52_dsa_server.py` only for `profile` mode;
- copies the completed trace into each rank's artifact directory;
- retains the existing schema and finite-logprob validation for both responses.

Run:

```bash
../../.venv/bin/python -m pytest -q \
  python/sgl_jax/test/test_glm52_e2e_compare.py -k 'profile_request or profile_runner'
```

Expected: FAIL because the runner does not support the profile contract.

### Step 2: Implement profile mode and teardown-safe trace collection

Add validated environment variables for profile step count, tracer levels, and profile
timeout. Clean the unique host-local trace directory before server launch. In profile mode,
delegate request execution to the client. During the existing EXIT trap, stop the server
and copy any completed local traces into `${OUT}/profile/`; rank-scoped `${OUT}` paths
prevent multi-host artifact collisions.

### Step 3: Run runner tests and shell validation

Run:

```bash
../../.venv/bin/python -m pytest -q python/sgl_jax/test/test_glm52_e2e_compare.py
bash -n scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh
python -m py_compile scripts/kernels/profile_glm52_dsa_server.py
```

Expected: all commands exit 0.

### Step 4: Commit the profiling harness

```bash
git add \
  scripts/kernels/profile_glm52_dsa_server.py \
  scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh \
  python/sgl_jax/test/test_glm52_e2e_compare.py
git commit -m "feat(profile): capture GLM-5.2 prefill and decode traces"
```

## Task 3: Render and submit the Falcon v7x-32 profile

**Files:**

- Create: `scripts/kernels/falcon_glm52_dsa_v7x32_profile.yaml`

### Step 1: Render the reviewed serving template

Base the manifest on Falcon's `references/profiling/sglang-jax-serving.yaml` with:

- `exp_type: PROFILING`, `artifact_type: trace`, and `profile: true`;
- cluster `tpu-training-antgroup`;
- v7x-32 sizing: four replicas, eight visible devices per replica, topology `2x2x4`;
- real checkpoint mounted read-only at `/models/GLM-5.2` with range-read caching disabled;
- TP32 / DP1 / EP32, fused MoE, DSA, 128-token chunked prefill;
- request profile `profile`, 3072 input tokens, 8 output tokens, four profiled steps;
- `TMPDIR`, pip/uv caches, and JAX compilation cache under `/tmp/tpu_logs`;
- an exact source revision from Task 2.

### Step 2: Make the pinned revision remotely available

Push `develop/glm52-dsa-falcon` to the existing `cjx0709` fork, then have the manifest clone
and detach at the exact Task 2 commit. Verify the remote revision before submission:

```bash
git push cjx0709 develop/glm52-dsa-falcon
git ls-remote cjx0709 refs/heads/develop/glm52-dsa-falcon
```

### Step 3: Submit through the Falcon workflow wrapper

Run:

```bash
falcon workflow profile submit \
  -f scripts/kernels/falcon_glm52_dsa_v7x32_profile.yaml \
  --output json
```

Record `ids.exp_id` and validate the stable JSON envelope. The submit command is also the
manifest validation boundary.

### Step 4: Own the run through profile collection

Run:

```bash
falcon workflow profile collect <exp_id> --timeout 2h --output json
```

Require `ids.status == "SUCCEEDED"`; otherwise inspect `data.job`, the latest
`data.conditions[]`, and then `falcon exp logs <exp_id> --output json`.

## Task 4: Analyze and quantify the baseline

**Files:**

- Create locally in `/tmp`: `glm52-dsa-xprof-analysis.yaml`
- Modify: `note/2026-07-20-glm52-dsa-current-status.md`

### Step 1: Enroll xprof-summary

Create one analysis with `plugin_names: [xprof-summary]` and params
`profile_format: auto`, `workload_kind: serving`. Use:

```bash
falcon workflow analysis create -f /tmp/glm52-dsa-xprof-analysis.yaml --output json
falcon workflow analysis wait <analysis_id> --timeout 30m --output json
falcon workflow analysis outputs <analysis_id> --output json
falcon workflow analysis cat <analysis_id> report.md --output json
falcon workflow analysis cat <analysis_id> metrics.json --output json
```

### Step 2: Build a time decomposition

From declared plugin outputs, `profile_timeline.json`, response metadata, and server logs,
record separately for prefill and decode:

- request wall time and tokens/step;
- profiled device step time and device busy percentage;
- host `run_batch`, model dispatch/resolve, result processing, and uncovered gap time;
- top device operations excluding DSA attention, especially fused MoE, matmuls, collectives,
  logits/sampling, and transfer operations;
- evidence of compilation or cache misses inside the measured window.

Do not infer a bottleneck from manifest metadata alone. If `xprof-summary` cannot expose a
required dimension, record the missing analyzer surface and use the captured trace only for
manual XProf inspection; do not read the backing bucket directly.

### Step 3: Update the status note with baseline evidence

Add the Falcon experiment, artifact, analysis IDs, exact source revision, measurement
matrix, device/host decomposition, and one ranked list of non-kernel bottlenecks. Clearly
separate measured facts from hypotheses.

## Task 5: Fix the largest proven non-kernel bottleneck

**Files:**

- Create: `docs/superpowers/plans/2026-07-20-glm52-dsa-profile-followup.md`
- Modify/Test: the exact files named by that evidence-driven follow-up plan

### Step 1: Select one optimization using fixed gates

Select the highest-cost region that is outside the DSA Pallas kernel and satisfies both:

- at least 10% of measured prefill or decode wall time, or at least 100 ms of avoidable host
  gap per step;
- a semantics-preserving change can be isolated and tested.

Do not select a region whose cost is only a profiler artifact, compile event, or warm-up.

### Step 2: Write the exact follow-up plan before changing implementation

Name the concrete root cause, target function(s), failing test, implementation change,
correctness regression suite, and the same-shape Falcon before/after measurement. This step
turns the measured bottleneck into an implementation plan without guessing before the trace.

### Step 3: Implement with TDD and re-profile

Run the focused unit/integration tests first, then the existing DSA kernel and E2E contract
tests. Submit a second Falcon profile with only that optimization changed and compare the
same 3072/8 workload. Accept the optimization only if correctness gates remain unchanged and
the targeted region improves outside run-to-run noise.

## Task 6: Final verification and commits

**Files:**

- Modify: `note/2026-07-20-glm52-dsa-current-status.md`
- Modify: `docs/superpowers/plans/2026-07-20-glm52-dsa-profile-followup.md`

### Step 1: Run local verification

```bash
git diff --check
../../.venv/bin/python -m pytest -q python/sgl_jax/test/test_glm52_e2e_compare.py
bash -n scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh
python -m py_compile scripts/kernels/profile_glm52_dsa_server.py
```

### Step 2: Verify remote evidence

Require a succeeded baseline profile and completed `xprof-summary`; after optimization,
require a succeeded same-shape candidate profile and unchanged response schema/finite-logprob
checks. Record all IDs in the status note.

### Step 3: Commit documentation and the selected optimization separately

Keep profiling evidence/documentation separate from the implementation commit so the
baseline remains auditable. Use commit messages that name the measured bottleneck rather
than claiming a general DSA speedup.
