# PD Serial PR Guidance (2026-05-28)

这份文档描述当前推荐的 PD 提交流程：

- **不用 stacked PR**
- **按真正串行方式逐个开 PR、逐个 merge**
- **当前 `epic/pd-disaggregation` 分支只作为 backup / integration branch**

目标是：

- 让 reviewer 每次只看一个清晰边界的增量
- 每次 merge 后给团队一个休整窗口
- 避免 stacked PR 在 GitHub 上带来的 review 负担

---

## 1. 总体策略

### 1.1 分支角色

当前角色分工：

- `epic/pd-disaggregation`
  - **backup / integration branch**
  - 保留完整提交链和所有中间状态
  - 不直接拿来开 PR

- `main`
  - upstream 真实能力基线
  - roadmap `#1196` 只反映这里的能力

- `pr/*`
  - 每个 PR 都从 **当时最新的 `origin/main`** 新拉分支
  - 只 cherry-pick 当前 PR 需要的 commit

### 1.2 提交原则

不走 stacked PR，意味着：

1. 先开 PR1
2. 等 PR1 merge 到 `main`
3. 从新的 `origin/main` 开 PR2
4. 再 cherry-pick PR2 对应 commit
5. 重复直到最后一个 docs-only PR

也就是说：

- **每个 PR 都以最新 `main` 为 base**
- **不让 reviewer 在 GitHub 上看跨多个未合入 PR 的 diff**

---

## 2. 当前 commit 链

当前 `main..epic/pd-disaggregation` 的核心 commit 链：

1. `cdd42424`
   - `feat(disaggregation): Stage 1 — transfer foundation + host pool + side channel`
2. `a8098934`
   - `feat(disaggregation): Stage 2 — bootstrap + scheduler mixin + prefill-only contract`
3. `ecc30004`
   - `feat(disaggregation): Stage 3 — multi-host routing + 内置 mini_lb proxy`
4. `69ee1b05`
   - `feat(disaggregation): production hardening tools + OpenAI /v1 PD passthrough + e2e matrix`
5. `dda61a52`
   - `docs(disaggregation): RFC suite + closeout + ops + research`

注意：

- 当前 commit1 实际覆盖了 roadmap 里的 **Stage 0 + Stage 1**
- 所以 PR1 会同时对应：
  - `#1199`
  - `#1242`

---

## 3. 推荐 PR 拆分

### PR1: Transfer foundation + side channel

**来源 commit**

- `cdd42424`

**建议分支名**

- `pr/pd-foundation-sidechannel`

**建议 PR 标题**

- `feat(disaggregation): transfer foundation and side-channel lifecycle`

**对应 issue**

- closes `#1199`
- closes `#1242`

**内容范围**

- `JaxTransferWrapper`
- `KVManager` / `KVSender` / `KVReceiver` / `KVPoll`
- `JaxTransferKVManager`
- `ZmqPullNotifier`
- `QueueHostKVPool`
- Stage 1 smoke-level `ServerArgs/CLI` coverage

**验证要求**

至少：

```bash
UV_NO_CONFIG=1 /Users/jiongxuan/Library/Python/3.9/bin/uv run \
  --no-project --isolated --python 3.13 \
  --with "sglang-jax[cpu] @ file:///Users/jiongxuan/workspace/sgl-jax/python" \
  --with pytest \
  python -m pytest /Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/test/disaggregation -q
```

建议在 PR 描述里写：

- local result
- 这一层只解决 transfer foundation / side-channel lifecycle
- path-A 仍未接通

### PR2: Bootstrap + scheduler e2e + prefill-only contract

**来源 commit**

- `a8098934`

**建议分支名**

- `pr/pd-scheduler-e2e`

**建议 PR 标题**

- `feat(disaggregation): bootstrap, scheduler integration, and prefill-only contract`

**对应 issue**

- closes `#1241`

**内容范围**

- bootstrap server/client
- prefill/decode mixin
- tokenizer/bootstrap passthrough
- prefill-only response semantics
- PD mode disables overlap

**验证要求**

至少：

- 本地 `python/sgl_jax/test/disaggregation -q`

建议：

- 加一条单 `(P, D)` 正常路径 smoke 说明

### PR3: Multi-host + single-entry proxy

**来源 commit**

- `ecc30004`

**建议分支名**

- `pr/pd-routing-proxy`

**建议 PR 标题**

- `feat(disaggregation): multi-host routing and single-entry proxy`

**对应 issue**

- closes `#1240`

**内容范围**

- `router_args.py`
- `launch_router.py`
- `mini_lb.py`
- `mini_lb_helpers.py`

当前定位应当在 PR 描述里明确写成：

- upstream-shaped launcher + MiniLB
- plus a small local patch layer

**验证要求**

至少：

- 本地 `python/sgl_jax/test/disaggregation -q`

建议补到 PR 描述：

- `/generate` smoke
- `/v1/chat/completions` smoke

### PR4: Benchmark/eval path + operator-side validation tools

**来源 commit**

- `69ee1b05`

**建议分支名**

- `pr/pd-benchmark-eval-tools`

**建议 PR 标题**

- `feat(disaggregation): benchmark/eval entrypoint support and validation tooling`

**不要叫**

- `hardening`
- `stage 4`

因为这不是完整 Stage 4 hardening，只是：

- OpenAI `/v1/*` PD passthrough
- benchmark/eval harness 接入
- operator-side e2e / stress / sweep / chaos 工具

**对应 issue**

- 不直接 closes `#1196`
- 在描述里写：
  - `partial roadmap support for production-path benchmark/eval`

**验证要求**

建议 PR 描述里至少写：

- `bench_serving` smoke
- `run_eval.py gsm8k` smoke
- 当前第一条容量 cliff：
  - `16c / 4k / 128`

### PR5: Docs only

**来源 commit**

- `dda61a52`

**建议分支名**

- `pr/pd-docs`

**建议 PR 标题**

- `docs(disaggregation): RFCs, handoff, and operational guidance`

**内容范围**

- RFC 套件
- master handoff
- closeout
- ops / matrix / benchmark target / support matrix

**对应 issue**

- 可在描述里引用：
  - `#1196`
  - `#1199`
  - `#1242`
  - `#1241`
  - `#1240`

但不建议写 closes。

---

## 4. 推荐执行流程

### Step 0: 备份

不要直接在当前 `epic/pd-disaggregation` 上做危险操作。

建议至少保留：

- 当前 `epic/pd-disaggregation`
- 一个额外 backup branch

例如：

```bash
git checkout epic/pd-disaggregation
git branch backup/pd-disaggregation-pr-split-2026-05-28
```

### Step 1: 开 PR1

```bash
git fetch origin
git checkout -b pr/pd-foundation-sidechannel origin/main
git cherry-pick cdd42424
```

跑验证，push，开 PR1。

### Step 2: 等 PR1 merge

PR1 merge 后：

```bash
git fetch origin
git checkout -b pr/pd-scheduler-e2e origin/main
git cherry-pick a8098934
```

跑验证，开 PR2。

### Step 3: 重复直到 PR5

依次：

```bash
git checkout -b pr/pd-routing-proxy origin/main
git cherry-pick ecc30004

git checkout -b pr/pd-benchmark-eval-tools origin/main
git cherry-pick 69ee1b05

git checkout -b pr/pd-docs origin/main
git cherry-pick dda61a52
```

注意：

- 每个 PR 都从最新 `origin/main` 开新分支
- 不复用前一个 PR 分支
- 不做 stacked base

---

## 5. 每个 PR 描述建议结构

建议统一用：

### Summary

- 这条 PR 做什么

### Scope

- 包含什么
- 不包含什么

### Linked issues

- closes 哪些
- references 哪些

### Validation

- 本地测试命令
- 结果
- 如有远端 smoke，写最小 smoke 结果

### Follow-ups

- 明确后续留给下一个 stage / PR 的内容

---

## 6. 为什么推荐真正串行，而不是 stacked PR

当前这条 PD 链虽然有明确依赖，但你现在更关心：

- reviewer 负担低
- merge 后团队能休整
- roadmap 与 main 分支能力严格一致

那真正串行的优势是：

- reviewer 看的是 **相对 main 的干净 diff**
- 每个 PR merge 后，roadmap 可以立刻准确更新
- 某条 PR 被要求返工时，不会拖累后面整条 stacked 链

代价只是：

- 总周期会慢一些
- 需要反复从最新 `main` 起分支、cherry-pick

对当前这个项目阶段，这是合理取舍。

---

## 7. 一句话建议

当前最稳妥的方案是：

> 把 `epic/pd-disaggregation` 当 backup / integration branch 保留，后续 PR 按 `cdd42424 -> a8098934 -> ecc30004 -> 69ee1b05 -> dda61a52` 的顺序，从最新 `origin/main` 串行 cherry-pick、串行开 PR、串行 merge。
