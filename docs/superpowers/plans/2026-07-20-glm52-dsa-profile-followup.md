# GLM-5.2 MLA Cache Single-Scatter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 GLM-5.2 DSA 主 MLA cache 的 latent/RoPE 两次连续 scatter 合成一次，降低 prefill 的 non-Pallas device time，同时保持全部 selection、mapping、kernel 和 E2E correctness gate 不变。

**Architecture:** `write_mla_kv_cache` 继续负责 slot validation 和 packed cache layout，但先把 `[new_c_kv, alignment padding, new_k_pe]` 组成一个连续 update，再对 `:latent_aligned + rope_dim` 做一次 `.at[].set`。不修改 DSA Pallas kernel、Indexer Top-K、logical-to-physical mapping 或 cache ABI；测试同时锁定数值、invalid-slot drop、padding 边界和单 scatter lowering contract。

**Tech Stack:** Python、JAX、pytest、Falcon v7x-32、XPlane/XProf。

---

### Task 1: 用失败测试锁定 single-scatter contract

**Files:**

- Modify: `python/sgl_jax/test/test_dsa_reference.py`

- [x] **Step 1: 把 sharding test 改成 single-scatter contract**

将 `test_write_mla_kv_cache_preserves_operand_sharding_on_both_scatters` 重命名为
`test_write_mla_kv_cache_packs_latent_and_rope_into_one_sharded_scatter`，保留 `FakeCache`
捕获，并断言：

```python
assert len(captured) == 1
assert captured[0]["kwargs"]["out_sharding"] is expected_sharding
assert captured[0]["values"].shape == (1, LATENT_ALIGNED + ROPE_DIM)
assert captured[0]["index"][-1] == slice(None, LATENT_ALIGNED + ROPE_DIM)
```

- [x] **Step 2: 运行 focused test 并确认 RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  python/sgl_jax/test/test_dsa_reference.py \
  -k packs_latent_and_rope_into_one_sharded_scatter
```

Expected: FAIL，显示当前实现捕获到 2 次 `.set`。

### Task 2: 实现一次连续 scatter

**Files:**

- Modify: `python/sgl_jax/srt/kernels/dsa/reference.py`
- Test: `python/sgl_jax/test/test_dsa_reference.py`

- [x] **Step 1: 构造 packed update 并只调用一次 `.set`**

在 slot index 计算之后，用 cache dtype 构造 alignment padding，再拼接 latent/RoPE：

```python
latent = new_c_kv.astype(cache.dtype)
rope = new_k_pe.astype(cache.dtype)
latent_padding = jnp.zeros(
    (new_c_kv.shape[0], latent_aligned - latent_dim),
    dtype=cache.dtype,
    out_sharding=jax.typeof(new_c_kv).sharding,
)
packed_update = jnp.concatenate((latent, latent_padding, rope), axis=-1)
update_width = latent_aligned + rope_dim
return cache.at[page, row, lane, :update_width].set(
    packed_update,
    mode="drop",
    out_sharding=jax.typeof(cache).sharding,
)
```

这会保留 RoPE 尾部之后的 cache 内容；latent alignment gap 是未被 attention/indexer
读取的 padding，写成 0 不改变语义。

- [x] **Step 2: 运行 focused test 并确认 GREEN**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  python/sgl_jax/test/test_dsa_reference.py \
  -k 'write_mla_kv_cache'
```

Expected: PASS。

- [x] **Step 3: 运行 DSA correctness regression**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  python/sgl_jax/test/test_dsa_reference.py \
  python/sgl_jax/test/test_dsa_backend.py \
  python/sgl_jax/test/test_dsa_glm52.py \
  python/sgl_jax/test/test_dsa_cross_framework.py
```

Expected: PASS；CPU/Torch selection、mapping、sparse MLA 和 backend integration gate
均不变。

- [x] **Step 4: 提交 isolated implementation**

```bash
git add \
  python/sgl_jax/srt/kernels/dsa/reference.py \
  python/sgl_jax/test/test_dsa_reference.py
git commit -m "perf(dsa): combine MLA cache writes"
```

### Task 3: 同 shape Falcon candidate profile

**Files:**

- Modify: `scripts/kernels/falcon_glm52_dsa_v7x32_profile.yaml`
- Modify: `note/2026-07-20-glm52-dsa-current-status.md`

- [x] **Step 1: 固化并 stage candidate source**

将 Task 2 commit 打成 source archive，经 Falcon CPU staging experiment 放到 `/models`
共享路径；记录 archive SHA256 和 staging exp/artifact。profile manifest 必须 pin 到该 commit
和 archive checksum。

- [x] **Step 2: 提交完全相同的 v7x-32 profile**

Run:

```bash
falcon workflow profile submit \
  -f scripts/kernels/falcon_glm52_dsa_v7x32_profile.yaml \
  --output json
falcon workflow profile collect <candidate-exp-id> --timeout 2h --output json
```

保持 checkpoint、TP32/DP1/EP32、fused MoE、3072 input、8 output、chunk size、
precompile variants、tracer levels和 correctness schema gate 全部不变。

- [x] **Step 3: 对 candidate 的 prefill/decode 分别运行 XProf 和 source breakdown**

对 rank-0 `prefill` / `decode` subpath 分别创建 `xprof-summary`，再运行与 baseline
`an-0emu7qinah` 同逻辑的 source breakdown analyzer。记录：

```text
write_mla_kv_cache scatter event count
write_mla_kv_cache device total_ms
jit_jitted_run_model prefill/decode ms
request warmup wall time
output ids/logprobs/top-20 schema gates
```

- [x] **Step 4: 按固定 acceptance gate 决定保留或回滚**

仅在以下条件全部满足时保留：

```text
correctness/schema gate unchanged
two cache-write scatter families -> one
prefill target region improves beyond trace noise
no decode regression beyond trace noise
```

若目标 event 未合并或 prefill 没有可重复改善，使用一个普通 revert commit 撤销
implementation，不改写历史。

### Task 4: 文档、验证与阶段提交

**Files:**

- Modify: `note/2026-07-20-glm52-dsa-current-status.md`
- Modify: `docs/superpowers/plans/2026-07-20-glm52-dsa-profile-followup.md`

- [x] **Step 1: 写入 before/after 和 Falcon IDs**

记录 baseline/candidate exp、artifact、analysis、source commit，以及每项 acceptance gate 的
结果；明确区分 profiler overhead 与 unprofiled E2E wall。

- [x] **Step 2: 最终本地验证**

Run:

```bash
git diff --check
../../.venv/bin/python -m pytest -q \
  python/sgl_jax/test/test_dsa_reference.py \
  python/sgl_jax/test/test_dsa_backend.py \
  python/sgl_jax/test/test_dsa_glm52.py \
  python/sgl_jax/test/test_dsa_cross_framework.py \
  python/sgl_jax/test/test_glm52_e2e_compare.py \
  python/sgl_jax/test/test_scheduler_profiler_mixin.py
bash -n scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh
python -m py_compile scripts/kernels/profile_glm52_dsa_server.py
```

Expected: all commands exit 0。

- [x] **Step 3: 提交 profile evidence**

```bash
git add \
  scripts/kernels/falcon_glm52_dsa_v7x32_profile.yaml \
  note/2026-07-20-glm52-dsa-current-status.md \
  docs/superpowers/plans/2026-07-20-glm52-dsa-profile-followup.md
git commit -m "docs(dsa): record MLA cache-write profile"
```

## 执行结果（2026-07-20）

- 初次 candidate `exp-owybp7zc86` 在 JAX 0.9.0 首次 EXTEND trace 暴露 padding
  sharding 不一致；TDD 修复为 `0324d5a0b`。
- 成功复测：`exp-6sroakc4lh` / `art-q8dine8622`；prefill/decode XProf 为
  `an-u4d5j18uat` / `an-7ncficq29z`。
- 两组 scatter 已变成一组；prefill/decode 单 event cache-write 时长分别下降
  `48.8% / 53.1%`，model envelope 分别下降 `7.1% / 1.4%`。
- correctness/schema gate 不变，DSA custom-kernel aggregate 不变，accept implementation。
