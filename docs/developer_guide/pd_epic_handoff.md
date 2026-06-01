# PD 分离实现 — Epic 工作上下文 Handoff

> **Primary entrypoint:** This file remains useful for cluster/deployment context, but the recommended first-read handoff is now [pd_master_handoff_2026_05_28.md](/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/pd_master_handoff_2026_05_28.md).

本文件给新会话 cold-start 用：单文件读完即可继续 PD 分离的开发工作。
不进 git，长期本地维护。

**配套文档**：
- `pd_epic_ops_log.md`（同目录）— 每次集群操作、跨 pod 测试命令、踩过
  的坑、各 stage 验收数据。**任何 kubectl / 跑测试 / 复盘场景先翻它**。

---

## 1. 任务总览

为 sgl-jax 实现 Prefill-Decode 分离（PD）。

- **整体计划文档**：5 个 RFC（Stage 0-4），中文写好放 `docs/rfc/`，
  本地 untracked，详见 §3。
- **顶层 GitHub 跟踪**：[Roadmap issue #1196](https://github.com/sgl-project/sglang-jax/issues/1196)
  + sub-issue [#1199 (Stage 0 RFC)](https://github.com/sgl-project/sglang-jax/issues/1199)。
- **本地开发分支**：`epic/pd-disaggregation`（基于 `origin/main`，
  不 push 到上游）。
- **回合策略**：5 个 Stage 在 epic 分支内串行实现并端到端跑通，最终
  从 epic 拆 PR 走正规 review。

---

## 2. 当前进度

### 2.1 PR-1（基础设施，已 ship）

- 分支：`feat/hicache-pd-pr1`（本地）/ `primatrix:feat/hicache-pd-pr1`（远端 fork）
- 上游 PR：[#1195](https://github.com/sgl-project/sglang-jax/pull/1195)
  — 标题 `feat(mem_cache): PR-1 HiCache + PD infra foundation`
- 包含 3 个 commit：
  - `kv_cache_builder.build_kv_cache()` 从 scheduler 抽出
  - `MemoryPools` pytree 加 `host_pool` 槽位（keyword-only，默认 None）
  - `TreeComponent` ABC + `ComponentType` / `CacheTransferPhase` 枚举
- **状态**：等待 review；PD 工作不直接依赖 PR-1 任何代码（Stage 0 RFC
  已说明正交关系），可并行推进。
- **backup 分支**：`backup/feat-hicache-pd-pr1-full` 保留完整 6 commit
  （含 `SingleChipHostPool` prototype + `jx-v6e-4.yaml` + 中文 deployment
  guide），后续 HiCache 阶段或 PD 需要 `SingleChipHostPool` 时
  `git cherry-pick 00f0c9ea`。

### 2.2 PD epic 分支

- `epic/pd-disaggregation` 已在本地创建并 align 到 `origin/main`。
- 还没有任何 PD 代码提交（仅 RFC 草稿在 working tree）。

### 2.3 RFC 完成度

| Stage | RFC 文件 | 状态 |
|---|---|---|
| 0 | `docs/rfc/2026-05-25-pd-transfer-foundation.md` | 中文 ready + 英文版已发 issue #1199 |
| 1 | `docs/rfc/2026-05-25-pd-host-pool-side-channel.md` | 中文 ready，未发 issue |
| 2 | `docs/rfc/2026-05-25-pd-scheduler-e2e.md` | 中文 ready，未发 issue |
| 3 | `docs/rfc/2026-05-25-pd-multihost-routing.md` | 中文 ready，未发 issue |
| 4 | `docs/rfc/2026-05-25-pd-hardening.md` | 中文 ready，未发 issue |

发 sub-issue 的命令在 §6.4。

---

## 3. Epic 工作流

### 3.1 分支结构

```
origin/main
   │
   └── epic/pd-disaggregation     ← 本地长期分支，定期 rebase onto main
         │
         ├── feature/pd-stage0    ← 从 epic 拉，实现完 merge 回 epic
         ├── feature/pd-stage1    ← 同上
         ├── feature/pd-stage2
         ├── feature/pd-stage3
         └── feature/pd-stage4    ← Stage 4 可拆多个子分支
```

### 3.2 工作流约定

1. **每个 stage**：
   - `git checkout -b feature/pd-stageN epic/pd-disaggregation`
   - 按对应 RFC 实现 + 单测 + 集成测
   - 在 feature 分支上自由 commit，最后 squash 成 1-3 个语义干净的
     commit
   - `git checkout epic/pd-disaggregation && git merge --no-ff feature/pd-stageN`
     （保留 stage 边界，方便最后拆 PR）

2. **跨 stage 修问题**：
   - 如果 stage N 实施时发现 stage N-1 代码有问题：在当前 feature 分支
     加 fixup commit，最终 squash 时合并；**不要 amend epic 中间
     commit**（CLAUDE.md 禁 interactive rebase，会破坏 stage 边界）。
   - 如果是更根本的设计偏差：先更新对应 RFC，再写代码。

3. **定期 rebase epic onto upstream/main**：
   - 每周一次：`git fetch origin && git rebase origin/main`
   - 冲突早暴露好处理；6-12 周后再 rebase 很痛。

4. **最终拆 PR**：
   - epic 端到端跑通后，按 stage 边界拆 5 个（或更多）PR：
     `git log --oneline --first-parent main..epic/pd-disaggregation`
   - 每个 PR 走正常 review，按依赖顺序合入 main。

### 3.3 验收节奏

每个 stage 的验收标准都写在对应 RFC 的「测试」章节末尾。建议节奏：

- Stage N 实现完 → 跑 stage 内单测 + 集成测 → 合 feature 到 epic →
  开下一个 stage（不必等当前 stage 上游 review）
- 完整端到端验收（Stage 0-3 串通的"一句话 prompt 跑通"）在 Stage 2 RFC
  里定义，是 epic 第一次 milestone。

---

## 4. TPU 集群与部署

### 4.1 集群清单

| 集群 | kubectl context | 用途 |
|---|---|---|
| v6e | `gke_poc-tpu-partner_us-east5_tpuv6e-256-node` | PD 主要测试集群 |
| v7x | `gke_tpu-service-473302_us-central1_tpu7x-cluster` | 大模型测试，PD 暂不使用 |

切换：
```bash
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
export USE_GKE_GCLOUD_AUTH_PLUGIN=True
kubectl config use-context gke_poc-tpu-partner_us-east5_tpuv6e-256-node
```

### 4.2 Pod 拓扑

| 集群 | Pod 名前缀 | 拓扑 | 代码路径 | 模型路径 | Python |
|---|---|---|---|---|---|
| v6e-4 | `jx-v6e-4-0-*` | 1 host × 4 chip | `/tmp/sgl-jax-test/` | `/models/` | 系统 |
| v6e-16 | `jx-v6e-16-{0..3}-*` | 4 host × 4 chip | `/sglang-jax/` | `/models/` | `/opt/venv/` |

```bash
kubectl get pods | grep jx-v6e-16     # 列当前 pod 名
```

### 4.3 IPC_LOCK 要求

任何 PD pod yaml 必须带 `securityContext.capabilities.add: [IPC_LOCK]`，
否则 `jax.experimental.transfer` 会因 memlock 默认 64KiB 而性能崩坏
（spike 验证：加上 IPC_LOCK 后 `ulimit -l = unlimited`）。

`jx-v6e-4.yaml` 模板在 sgl-jax repo 根目录 untracked（参考已有的
PR-1 backup 分支）。v6e-16 当前部署已加。

### 4.4 代码同步

用 `tar` 管道（**不要 kubectl cp**，大文件会截断）：

```bash
# 单 pod
cd /Users/jiongxuan/workspace/sgl-jax
tar cf - python/sgl_jax/ | kubectl exec -i $POD -- tar xf - -C /sglang-jax/

# 多 host（v6e-16 全 4 pod）
PODS=($(kubectl get pods -o name | grep jx-v6e-16 | sort | sed 's|pod/||'))
for pod in "${PODS[@]}"; do
  tar cf - python/sgl_jax/ | kubectl exec -i "$pod" -- tar xf - -C /sglang-jax/
done
```

### 4.5 单 pod 调试模式（v6e-16 上跑小模型）

```bash
kubectl exec -it $POD -- bash -c '
  rm -f /tmp/libtpu_lockfile
  cd /sglang-jax
  env TPU_HOST_BOUNDS=1,1,1 TPU_TOPOLOGY=2x2 \
      TPU_WORKER_HOSTNAMES=$(hostname).jx-v6e-16-headless-svc \
      TPU_WORKER_ID=0 TPU_TOPOLOGY_WRAP=false,false,false \
      TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 \
      PYTHONPATH=/sglang-jax/python PYTHONUNBUFFERED=1 \
      /opt/venv/bin/python -u -m sgl_jax.launch_server \
      --model-path /models/<model-name> ...
'
```

完整启动参数模板见 backup 分支保留的 `tpu_pod_deployment_guide.md`
英文版（已搬到 wiki：`wiki/docs/projects/sglang-jax/operations/`）。

---

## 5. 关键代码位置

### 5.1 PR-1 已 ship 的基础设施（与 PD 正交，但同 repo）

| 路径 | 用途 |
|---|---|
| `python/sgl_jax/srt/mem_cache/kv_cache_builder.py` | 提供 `build_kv_cache()`，未来 PD 的 D 节点 cache 选择从此处加分支 |
| `python/sgl_jax/srt/mem_cache/memory_pool.py:1360` `MemoryPools` | pytree 已留 `host_pool=None` 槽位（PD 不直接用，HiCache L2 用） |
| `python/sgl_jax/srt/mem_cache/tree_component.py` | `TreeComponent` ABC + 枚举（PD 路径绕过 tree_cache，不直接用） |

### 5.2 PD 即将新增的代码（按 RFC 规划）

```
python/sgl_jax/srt/disaggregation/                            ← Stage 0 起新建
├── __init__.py
├── jax_transfer_wrapper.py             (Stage 0)
├── base/
│   ├── __init__.py
│   └── kv_manager.py                   (Stage 0, 4-tuple ABC)
├── jax_transfer/
│   ├── __init__.py
│   ├── conn.py                         (Stage 0, JaxTransferKVManager)
│   └── zmq_notifier.py                 (Stage 1)
├── bootstrap.py                        (Stage 2)
├── prefill.py                          (Stage 2, PrefillMixin)
├── decode.py                           (Stage 2, DecodeMixin)
├── utils.py                            (Stage 2, enums + factory)
└── metrics.py                          (Stage 4)

python/sgl_jax/srt/mem_cache/host_kv_pool.py                  ← Stage 1 新增
                                         (含 QueueHostKVPool)

python/sgl_jax/srt/managers/scheduler.py                      ← Stage 2 改
                                         (run_scheduler_process dispatch)

python/sgl_jax/srt/managers/tokenizer_manager.py              ← Stage 2 改
                                         (bootstrap_* 字段透传)

python/sgl_jax/srt/server_args.py                             ← Stage 2 / 3 / 4 改
                                         (--disaggregation-* 字段)

python/sgl_jax/test/disaggregation/                           ← Stage 0 起新建
├── test_jax_transfer_wrapper.py        (Stage 0, CPU)
├── test_kv_manager_state.py            (Stage 0, CPU)
├── test_byte_roundtrip.py              (Stage 0, manual TPU)
├── test_queue_host_kv_pool.py          (Stage 1)
├── test_zmq_pull_notifier.py           (Stage 1)
├── test_bootstrap_server.py            (Stage 2)
├── test_tokenizer_bootstrap_passthrough.py (Stage 2)
└── test_pd_e2e_single_host.py          (Stage 2)
```

---

## 6. 参考资料

### 6.1 已 ship 的设计 RFC

- `docs/rfc/2026-05-18-rfc-0-unified-cache-and-kv-infra.md` — RFC-0：
  cache + KV 基础设施（PD 用到 `KVTransferEngine` ABC 占位）
- `docs/rfc/2026-05-18-rfc-2-pd-disaggregation.md` — RFC-2：**PD 整体
  设计文档（最重要的参考）**，含 9 个 ADR、模块依赖图、所有组件的伪
  代码样例。本 epic 的 5 个 Stage RFC 都建立在 RFC-2 之上。

### 6.2 spike 数据（关键决策依据）

- wiki memory：`~/.claude/projects/-Users-jiongxuan-workspace-sgl-jax/memory/hicache_pd_spike_2026_05_23.md`
  — Phase-0 spike 完整结果。关键数据：
  - **跨 pod single channel 8.86 GB/s**（T3 v2，jx-v6e-16，JAX 0.8.1）
  - D2H 2.81 / H2D 45.70 GB/s（T2）
  - `IPC_LOCK` 让 memlock = `RLIM_INFINITY`（T7）
- spike 测试代码：`python/sgl_jax/test/disagg_feasibility/`（untracked）
  - 重点参考 `t3v2_xpod_minimal.py`（跨 pod transfer happy path）

### 6.3 sglang upstream 参考代码

- `sglang/python/sglang/srt/disaggregation/` — 4-tuple ABC、Mixin、
  bootstrap 等的原型，我们在 RFC-2 里 port + 精简。
- `sglang/python/sglang/srt/disaggregation/mini_lb.py` — 本路线 Stage 3
  复用的 router（Python 实现，无需 fork）。

### 6.4 GitHub issue + commands

```bash
# 看 roadmap 进度
gh issue view 1196 --repo sgl-project/sglang-jax

# 看 Stage 0 RFC issue
gh issue view 1199 --repo sgl-project/sglang-jax

# 发新 stage 的 RFC sub-issue (示例 Stage 1)
NEW_URL=$(gh issue create --repo sgl-project/sglang-jax \
  --title "[RFC] PD buffer + side channel: QueueHostKVPool + ZMQ pull-done" \
  --body "$(cat <<'EOF'
... (Stage 1 RFC 英文翻译版)
EOF
)" | tail -1)
NEW_NUM=$(echo "$NEW_URL" | grep -oE '[0-9]+$')
NEW_DB_ID=$(gh api /repos/sgl-project/sglang-jax/issues/$NEW_NUM --jq .id)
gh api -X POST /repos/sgl-project/sglang-jax/issues/1196/sub_issues \
  -F sub_issue_id=$NEW_DB_ID
```

---

## 7. 常用调试命令

### 7.1 跑 CPU 单测

```bash
cd /Users/jiongxuan/workspace/sgl-jax
PYTHONPATH=/Users/jiongxuan/workspace/sgl-jax/python \
  python3 -m pytest python/sgl_jax/test/disaggregation/ -v
```

注意：本机有两个 sgl_jax 安装（`sgl-jax/` 工作目录 + `sglang-jax/.venv/`
里 editable install）。**必须前置 PYTHONPATH** 才会用工作目录版本，
否则 pytest 跑的是另一个 repo 的旧代码。

### 7.2 跑跨 pod 集成（Stage 0 byte round-trip 之类）

```bash
PODS=($(kubectl get pods -o name | grep jx-v6e-16 | sort | sed 's|pod/||'))

# 同步代码
for pod in "${PODS[@]}"; do
  tar cf - python/sgl_jax/ | kubectl exec -i "$pod" -- tar xf - -C /sglang-jax/
done

# Pod 0 跑 P
kubectl exec ${PODS[0]} -- bash -c '
  rm -f /tmp/libtpu_lockfile
  env PYTHONPATH=/sglang-jax/python PYTHONUNBUFFERED=1 \
      <single-pod TPU env override> \
      /opt/venv/bin/python -u <test entry> --role prefill ...
'

# Pod 1 跑 D（端口对接 pod 0）
kubectl exec ${PODS[1]} -- bash -c '
  ... --role decode --remote ${PODS[0]} ...
'
```

### 7.3 看 server log / health

```bash
kubectl exec $POD -- tail -50 /tmp/server.log
kubectl exec $POD -- grep -q 'ready to roll' /tmp/server.log && echo READY
kubectl exec $POD -- bash -c 'ulimit -l'         # 必须显示 unlimited
```

### 7.4 端口转发本机调试

```bash
kubectl port-forward pod/$POD 30271:30271 &
curl http://localhost:30271/v1/chat/completions ...
```

---

## 8. 已知坑 & 注意事项

1. **本机两份 sgl_jax**（`sgl-jax/` 工作目录 vs `sglang-jax/.venv/` editable
   install）—— 跑 pytest **必须前置** `PYTHONPATH=/Users/jiongxuan/workspace/sgl-jax/python`
   才会用工作目录。
2. **kubectl context 会自动漂移**：每次操作前显式 `use-context`。
3. **`test_kv_cache.py` 有 10 个 pre-existing failure**：main 上同样
   失败，与 PD / PR-1 无关；下一 PR 起步前跑一次 baseline 区分。
4. **多 host 启动时不要设置 `TPU_HOST_BOUNDS` 等 env**：那些仅用于
   单 pod 调试场景；多 host 部署 JAX 会自己 init。
5. **`jax.experimental.transfer.await_pull` 是非阻塞**：register 后立
   即返回；P 必须等 D 完成 pull 之后才 release（由 Stage 1 ZMQ
   side channel 解决，Stage 0 用同步占位）。
6. **`jax.ShapeDtypeStruct` 必须带 `sharding=`**：否则 transfer API
   抛 `'NoneType' object has no attribute 'device_set'`。Stage 0 wrapper
   入口强制断言。
7. **TPU runner 不存在**：跨 pod 测试只能手动跑，不进 CI；PR 提交时
   手动跑 + 贴日志到 PR description。

---

## 9. 联系上下文

- Maintainer：`john`（同时也是 cjx0709 GitHub 账号）
- Fork 仓库：`primatrix/sglang-jax` 是 sgl-project/sglang-jax 的 fork
  （PR 提交渠道）；`cjx0709/sglang-jax` 是独立 repo（不是 fork，**不
  能用作 PR 源**，PR-1 时踩过这个坑）。
- 上游仓库：`sgl-project/sglang-jax`
- 个人 wiki：`/Users/jiongxuan/workspace/wiki/`（含已搬过去的 TPU
  deployment guide）

---

## 10. 下一步建议

按依赖顺序：

1. **先确认 #1199 (Stage 0 RFC) 没人反对**（等 1-2 天 review）
2. 起 `feature/pd-stage0` 分支，按 Stage 0 RFC 写代码 + 单测
3. 跑跨 pod byte round-trip 跑通，merge 回 `epic/pd-disaggregation`
4. **同时**用 §6.4 命令把 Stage 1 RFC 发成 #1196 的 sub-issue（英文版
   先翻 / 直接抄中文版翻成英文）
5. 起 `feature/pd-stage1`，循环

每个 stage 都重复这个节奏。最后一次性从 epic 拆 5 个 PR 上去 review。
