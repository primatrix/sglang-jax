# HiCache + PD 路线 PR handoff prompt 模板

把后续 PR (PR-1 → PR-4) 交给独立 agent 实现时用的 prompt 模板。设计目标：
prompt 自包含到能让新 agent 不依赖会话历史就能正确动手。

主对应 roadmap：
`/Users/jiongxuan/workspace/wiki/docs/projects/sglang-jax/design_docs/2026-05-23-hicache-pd-implementation-roadmap.md`

## 模板结构（8 节）

| 节 | 必须放什么 | 不放的代价 |
|---|---|---|
| 1 任务范围 + 验收 | 子任务清单 + 单条验收标准 | 范围蔓延，做多余工作 |
| 2 必读文档 | 路径 + 重点章节 | agent 把当前会话当真理，过时假设当事实 |
| 3 代码现状 | 关键文件 + 行号 | 浪费 token 搜文件 |
| 4 部署/测试环境 | TPU 集群、同步命令、跑测命令 | 跑错集群、用错 venv |
| 5 已知工程坑 | spike 沉淀（F1-F3 + G1-G3）+ memory 索引 | 重踩坑 |
| 6 工作流要求 | task 跟踪、commit 风格、单 PR 边界 | 不符合 project convention |
| 7 边界 | 不要做什么（防 PR 越界） | PR 难 review |
| 8 完成后 | 跑验证、回填 roadmap、开 PR 命令 | 闭环断 |

## PR-1 完整实例（直接复制粘给 agent）

````
# 任务：实现 sgl-jax HiCache + PD 落地 PR-1（基础设施 + spike 沉淀）

你是 senior 工程师，要在 sgl-jax 仓库实现 HiCache + PD 路线的第一个 PR。

## 1. 任务范围与验收

实现 5 个子任务（详见 §6.1 PR-1 任务清单 in roadmap doc）：
- PR-1-1: scheduler.py:516-545 抽出 `kv_cache_builder.build_kv_cache()`
- PR-1-2: `MemoryPools` pytree 扩 `host_pool` slot + 同步 model_runner.py:207 donate
- PR-1-3: `TreeComponent` ABC + `ComponentType`/`CacheTransferPhase` enum stub
- PR-1-4: 所有 v6e/v7x 部署 yaml 加 `securityContext.capabilities.add: [IPC_LOCK]`
  并写到 docs/developer_guide/tpu_pod_deployment_guide.md
- PR-1-5: `SingleChipHostPool` demo class + jit 写入 + ptr 稳定性单测

验收：每个子任务对应 roadmap doc §6.1 表格里的"验收"列那一句话；所有验收过了就开 PR。
**不要**实现 H2-H7（那是 PR-2 的事）；**不要**写 Pallas kernel（PR-3）；**不要**碰 PD（PR-4）。

## 2. 必读文档（按顺序读）

| 文件 | 路径 | 重点 |
|---|---|---|
| 实施路线总报告 | `/Users/jiongxuan/workspace/wiki/docs/projects/sglang-jax/design_docs/2026-05-23-hicache-pd-implementation-roadmap.md` | §3 三大发现+三个坑、§4 RFC 修订、§5 DAG、§6.1 PR-1 任务清单、§7 风险表 |
| HiCache 设计 RFC | `/Users/jiongxuan/workspace/wiki/docs/projects/sglang-jax/design_docs/rfc_1_hicache.md` | §2 总体架构、§2.2.5 mode B（**按 roadmap §4.1 修订理解**） |
| RFC-0 cache builder | `/Users/jiongxuan/workspace/wiki/docs/projects/sglang-jax/design_docs/rfc_0_shared_cache_pd_infra.md` | §1-§2 builder 设计意图 |
| sglang × tpu-inference 调研 | `/Users/jiongxuan/workspace/wiki/docs/projects/sglang-jax/reference/sglang-tpu-inference-multi-level-cache.md` | §1.1.2 UnifiedRadixCache、§4 Builder 模式 |
| spike 测试代码 | `/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/test/disagg_feasibility/` | T5 v2 variant C (`t5v2_shard_map_mode_b.py`) 是 PR-1-5 的 reference implementation |
| TPU 部署指南 | `/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/tpu_pod_deployment_guide.md` | PR-1-4 要更新这份 |

**先把 roadmap §1-§6.1 全读一遍再动手**。如果有信息冲突，roadmap doc 是 single source of truth。

## 3. 代码现状（必读的关键文件 + 行号）

```
sgl-jax 仓库根: /Users/jiongxuan/workspace/sgl-jax/

PR-1-1 抽 builder:
  python/sgl_jax/srt/managers/scheduler.py:516-545  ← 现有 if-elif 在这
  新建: python/sgl_jax/srt/mem_cache/kv_cache_builder.py

PR-1-2 改 pytree:
  python/sgl_jax/srt/mem_cache/memory_pool.py:1360-1393  ← MemoryPools 类
  python/sgl_jax/srt/model_executor/model_runner.py:206-207  ← donate_argnames

PR-1-3 ABC:
  python/sgl_jax/srt/mem_cache/base_prefix_cache.py:12-105  ← 已有 BasePrefixCache、HiCache hook 占位
  新建: python/sgl_jax/srt/mem_cache/tree_component.py

PR-1-4 部署 yaml:
  /tmp/jx-v6e-16-rebuild.yaml  ← 已加 IPC_LOCK，作为模板
  /Users/jiongxuan/workspace/sgl-jax/jx-v6e-4.yaml  ← 待加
  v7x cluster yaml 需要找运维要

PR-1-5 single-chip pool:
  reference: python/sgl_jax/test/disagg_feasibility/t5v2_shard_map_mode_b.py (variant C 部分)
  新建: python/sgl_jax/srt/mem_cache/single_chip_host_pool.py
  单测放: python/sgl_jax/test/mem_cache/test_single_chip_host_pool.py
        (参照已有 test_kv_cache.py / test_radix_cache.py 的风格)
```

**强制要求**：动手前先用 Read 工具读完上面所有 "现有" 文件，确认行号 + 现状没有过时（roadmap 写于 2026-05-23，可能已变）。

## 4. 部署 / 测试环境

### 4.1 单元测试（先在 macbook 跑）

```bash
cd /Users/jiongxuan/workspace/sgl-jax
python3 -m pytest python/sgl_jax/test/mem_cache/test_single_chip_host_pool.py -v
```

**重要**：PR-1-1/PR-1-2/PR-1-3 的单测优先在 CPU 跑通；只有 PR-1-5 必须上 TPU 验证 in-place 写入。

### 4.2 TPU 验证（必须，针对 PR-1-5）

```bash
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
kubectl config use-context gke_poc-tpu-partner_us-east5_tpuv6e-256-node

# 确认 jx-v6e-16 pods Running（如果没有就 kubectl apply -f /tmp/jx-v6e-16-rebuild.yaml 重建）
PODS=($(kubectl get pods -o name | grep jx-v6e-16 | sort | sed 's|pod/||'))

# 同步代码（用 tar 管道，不要 kubectl cp）
for pod in "${PODS[@]}"; do
  tar cf - python/sgl_jax/ | kubectl exec -i "$pod" -- tar xf - -C /sglang-jax/
done

# 跑 PR-1-5 demo（套用现成 /tmp/run_t.sh wrapper 的 TPU env override）
kubectl exec ${PODS[0]} -- bash /tmp/run_t.sh test_single_chip_host_pool

# 验证标准：
# - ptr 稳定 (first_ptr == last_ptr)
# - RSS 增长 < 200 MiB（300+ iter 长稳跑）
```

完整部署细节看 `docs/developer_guide/tpu_pod_deployment_guide.md`。

### 4.3 已有 spike 测试可直接复用

`python/sgl_jax/test/disagg_feasibility/t5v2_shard_map_mode_b.py` 是 PR-1-5 的最小参照——variant C 部分基本就是 `SingleChipHostPool` 的 prototype，可以从那里拷。

## 5. 必须规避的工程坑（spike 已发现）

| ID | 坑 | 规避 |
|---|---|---|
| F1 | 跨 chip 分片 pinned_host + `dynamic_update_slice` 触发 MegaScale collective，单 slice TPU 不支持 | host pool **只能用 single-chip mesh**：`NamedSharding(Mesh([devices[i]]), P('x'), memory_kind='pinned_host')` |
| G3 | `TPU_VISIBLE_CHIPS=0,1` 多 chip 子集触发 jaxlib 校验失败 | PR-1 测试用全 chip view（4 chip）或 1 chip 子集（=0），不要切 2,3 chip |
| H0 | GKE container 默认 memlock=64 KiB → `pinned_host` 静默 fallback unpinned | 任何要测 pinned_host 的 pod yaml 必须加 `securityContext.capabilities.add: [IPC_LOCK]`（PR-1-4 落实） |

完整 spike 沉淀看 memory `~/.claude/projects/-Users-jiongxuan-workspace-sgl-jax/memory/hicache_pd_spike_2026_05_23.md`。

## 6. 工作流要求

1. **用 TaskCreate 跟踪 5 个子任务**，每完成一个 mark completed
2. **commit 不要带 Co-Authored-By**（user 偏好，见 CLAUDE.md memory `git_commit_no_coauthor`）
3. **每个子任务一个 commit**，方便 review 单独看；最后开一个 PR 包含全部
4. **优先用 Read + Edit + Grep；用 Bash 只跑 test / git / kubectl**
5. **不要新建 markdown 文档**，所有结论回填到 roadmap doc §6.1 的"实际工期"列即可
6. **如果 spike 假设不成立**（比如 PR-1-5 长跑 RSS 涨）→ 立即停手，按 roadmap §8 决策树触发条件处理，不要 workaround

## 7. 边界（不要做）

- 不写 H2-H7、H3、H8、P1-P11（不是 PR-1 范围）
- 不改 RFC-0/1/2 文档（任何设计修订意见写到 roadmap doc，不动 RFC）
- 不重构与 PR-1 任务无关的代码（即使顺手能改也忍住）
- 不引入新依赖
- 不在 commit / 代码里写 emoji / 大段注释
- 不假设 spike 结论永久有效——动手前用 Read 重新核行号 + 状态

## 8. 完成后

- 跑完所有单测 + TPU PR-1-5 长稳测，打日志
- 在 roadmap doc §6.1 表格加一列"实际工期 / 偏离原因"，填回数据
- 用 `gh pr create` 开 PR，标题 `feat(mem_cache): PR-1 HiCache + PD infra (builder + ABC + single-chip pool)`
- PR description 引用 roadmap doc + 列出 5 个子任务对应 commits + 跑通的验证步骤

开始之前先用 1 句话告诉我你打算第一步做什么。
````

## PR-2/3/4 适配指南

只需改 4 节，其他 5 节复用：

| 节 | PR-2 | PR-3 | PR-4 |
|---|---|---|---|
| §1 任务范围 | roadmap §6.2 任务清单 8 项 | roadmap §6.3 任务清单 4 项 | roadmap §6.4 任务清单 9 项 |
| §2 必读文档 | 增 `sglang/python/sglang/srt/mem_cache/{unified_radix_cache,unified_cache_components/}*.py` | 增 `tpu-inference/tpu_inference/distributed/kv_transfer.py:416` (Pallas copy_to_host) | 增 `sglang/python/sglang/srt/disaggregation/{base,prefill,decode}.py` + `tpu-inference/tpu_inference/distributed/tpu_connector.py` + roadmap §4.2 PD RFC 不动 |
| §3 代码现状 | PR-1 完成后的新代码 + UnifiedRadixCache 待 port 的文件 | PR-2 完成后的 host pool 接口 + Pallas kernel 模板 | PD 完全是新代码（除 schedule_batch.py:350-361 几个占位字段） |
| §6.1 PR title | `feat(mem_cache): PR-2 HiCache main line (UnifiedRadixCache + HiCacheController)` | `feat(mem_cache): PR-3 Pallas D2H + SWA/Mamba component` | `feat(disagg): PR-4 PD disaggregation MVP` |

§4/5/6/7/8 几乎一字不动。

## 维护惯例

**每个 PR 合并后，handoff prompt 自己也要更新**（行号会漂移）：

1. PR 合并后让 agent 顺手用 grep 重新校 §3 里所有引用过的行号
2. 把新代码现状（新建的类、新增的入口）加进 §3 PR-N+1 那段
3. 把已 fix 的工程坑从 §5 删掉（spike 沉淀只留还在威胁的）
4. roadmap doc §6.X 表格加列"实际工期" + "偏离原因"，作为下一 PR 估时的 calibration 数据

这样 handoff prompt 永远是"对当前代码状态新鲜的"，不会让下一个 agent 摸着过时假设。
