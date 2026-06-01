# RFC-2: PD 分离 — TPU/JAX 上的 Prefill-Decode Disaggregation

**Status**: Draft  
**Author**: john  
**Date**: 2026-05-18  
**Depends on**: RFC-0 (UnifiedRadixCache + KV 缓存与传输基础设施)  
**Related**: RFC-1 (HiCache)  
**Prerequisite reading**: `docs/rfc/2026-05-18-rfc-0-unified-cache-and-kv-infra.md`、`docs/research/2026-05-18-sglang-cache-pd-organization.md`、`docs/research/2026-05-18-tpu-inference-jax-api-survey.md`

---

## 1. 摘要 & 动机

### 1.1 目标

为 sgl-jax 实现「Prefill-Decode 分离」（PD），允许 prefill 和 decode 阶段跑在**不同 host 上的不同 process** 上，提高吞吐和资源利用率。**主要切入点是多 host TP 部署**（例如 P-cluster: 8-chip TP=8 + D-cluster: 8-chip TP=8，各自一个独立 host），这是生产 serving 的典型场景。本 RFC 填充 RFC-0 定义的 `KVTransferEngine` ABC 并搭建配套控制面：

1. `JaxTransferKVEngine`：基于 `jax.experimental.transfer` 的 KV 跨主机/跨进程传输引擎（KVTransferEngine ABC 实现）
2. `QueueHostKVPool`：D2H staging buffer（HostKVPool ABC 的 Queue 实现，与 RFC-1 LRU 实现并列）
3. `BootstrapServer`：HTTP-based P-D 路由握手 + UUID 协商
4. 4-tuple ABC：`KVManager` / `KVSender` / `KVReceiver`（port 自 sglang，但精简）
5. Scheduler 集成：`SchedulerDisaggregationPrefillMixin` + `SchedulerDisaggregationDecodeMixin`
6. ZMQ 侧通道：D 通知 P 传输完成
7. 部署：**多 host TP 为主，单 host 多进程仅作开发测试**

### 1.2 非目标

- ❌ 不引入 mooncake / NIXL / aibrix 等 GPU 专用传输库（参见 [[hicache_pd_rfc_design_principles]]）
- ❌ 不实现 sglang 的 5 个 PD backend（Mooncake/NIXL/MoRI/Ascend/Fake），只做一个 `jax.experimental.transfer` backend
- ❌ 不做 prefix cache 在 D 端的部分命中合并（sglang 已知限制：JAX P2P 不支持 partial pulling）
- ❌ 不做 Pipeline Parallel + PD（sgl-jax 长期不规划支持 PP）
- ❌ **单 host 多进程模式仅作开发/测试用途**（KV pool 切两半每边能装的请求量小，不适合生产 serving）
- ❌ 不做 HiCache 相关功能（属 RFC-1）
- ❌ 不修改现有 prefix cache（ChunkCache / UnifiedRadixCache），PD 是正交维度

### 1.3 范围矩阵

| 子系统 | RFC-0 范围 | RFC-1 范围 | **RFC-2 范围** |
|---|---|---|---|
| KVTransferEngine ABC | ✓ 接口 | — | **JaxTransferKVEngine 实现** |
| HostKVPool ABC | ✓ 接口 | LRU 实现 | **Queue 实现 (D2H staging)** |
| Bootstrap server | — | — | **HTTP-based** |
| 4-tuple PD ABC | — | — | **KVManager/Sender/Receiver port** |
| Scheduler PD Mixin | — | — | **Prefill/Decode mixin** |
| ZMQ 侧通道 | 复用现有 | — | **新增 pull-done 消息** |
| 部署脚本 | — | — | **多 host 示例 (主) + 单 host 开发示例 (附)** |

### 1.4 与 RFC-1 的关系（接口正交 + 第一版 D 节点限制）

PD 与 cache 类型的关系按「接口层」和「部署层」分别理解：

**接口层（永久）**：tree_cache × PD × HiCache 完全正交（参见 RFC-0 §1.3 矩阵）。任何 BasePrefixCache 实现都能搭配 PD 走 KV 传输，不在 ABC 中假设 cache 类型。

**部署层（第一版限制）**：

**P 节点**（与 null 模式一致，遵循 RFC-0 §5 标准 dispatch）：
- ✓ ChunkCache + PD（轻量场景）
- ✓ UnifiedRadixCache + PD（默认，多轮共享 prefix 受益）
- ✓ UnifiedRadixCache + PD + HiCache（生产场景，prefix 多级缓存）

**D 节点**（**ADR-9 第一版只实现 ChunkCache**）：
- ✓ ChunkCache + PD（**第一版唯一实现**）
- ✗ UnifiedRadixCache + PD（第一版不实现；与 sglang 默认一致）
- ✗ ChunkCache + HiCache + PD（ChunkCache 无 HiCache hook）

**未来扩展**：若有 D + RadixCache 需求，按 sglang 模式加 opt-in flag（ADR-9 未来扩展点）。

**PD 主路径绕过 `tree_cache.insert`**，直接读 `req_to_token_pool`（参见 sglang 调研 §3.6.2）。

唯一接触点：PD 与 RFC-1 共享 `HostKVPool` ABC（但实现不同——P 端用 LRU 给 HiCache，D 端用 Queue 给 D2H staging），共享底层 `jax.experimental.transfer` 服务（一个进程一个 server，ADR-2）。

---

## 2. 决策记录（ADR）

### ADR-1: 基于 `jax.experimental.transfer` 实现传输

| | |
|---|---|
| **决策** | RFC-2 的 `JaxTransferKVEngine` 唯一 backend 基于 `jax.experimental.transfer.start_transfer_server` + `await_pull` + `pull`。封装一层 ABC `KVTransferEngine` 隔离 API 风险。 |
| **理由** | (1) tpu-inference 已经验证可用（jax-api 调研 §4.2）；(2) JAX 原生 API，跨平台、与 sgl-jax 整体技术栈一致；(3) 不依赖 GPU 专用库（mooncake/NIXL）；(4) 单机和多主机统一 backend。 |
| **影响** | API 是实验性的，文档稀缺。RFC-2 通过 ABC 隔离风险，并准备 fallback 路径（备选 ZMQ + 显式 `device_get`/`device_put`）。 |
| **替代方案** | (a) ZMQ + manual D2H + H2D：拒绝，效率低 + 字节流序列化开销大；(b) gRPC + numpy：拒绝，同上；(c) mooncake：拒绝，GPU 库无法在 TPU 用。 |

### ADR-2: `transfer_server` 是 per-process singleton（N+M 实例部署）

| | |
|---|---|
| **决策** | 每个 sgl-jax 进程启动**至多一个** `jax.experimental.transfer.start_transfer_server`（per-process singleton）。同进程内多个 `KVTransferEngine` 实例（理论上不会发生，但抽象上要保证）共享同一 server。**全局 server 总数 = N（P 进程数）+ M（D 进程数）**，不是单实例。 |
| **理由** | tpu-inference 的实践（`tpu_connector.py:573`）：一个进程一个 server。每个 JAX 进程是独立的 transfer runtime，server 绑定具体 IP:Port，**不能跨进程共享**。同进程内多 server 会冲突端口 + 浪费资源。 |
| **影响** | `JaxTransferKVEngine` 内部用 `_GLOBAL_TRANSFER_SERVER` 模块级变量做进程内单例（§4.1）。不同部署形态的 server 数：<br>• 单机 1P+1D = 2 server<br>• 单机 1P (4chip TP) + 1D (4chip TP) = 2 server（每 host 一进程）<br>• 多 host 2P + 2D = 4 server（N=M=2）<br>• Bootstrap server 是独立实例（全局 1 个），不计入 N+M |
| **替代方案** | (a) 全局单 server（跨所有进程）——拒绝，跨进程不可能；(b) 每个 KVTransferEngine 实例一个 server——拒绝，端口冲突 + 资源浪费。 |

### ADR-3: 进程级 role（disaggregation_mode）

| | |
|---|---|
| **决策** | 每个 sgl-jax 进程通过 `--disaggregation-mode {null,prefill,decode}` 启动，进程级别确定 role。一个进程不可同时做 P 和 D。 |
| **理由** | (1) 与 sglang 一致（sglang 调研 §3.3）；(2) Scheduler 主循环必须区分 P 路径和 D 路径，进程级 role 让 dispatch 简单；(3) 同进程 P+D 会让资源分配（HBM、host memory）混乱。 |
| **影响** | 单机多进程部署：1 个 P 进程 + 1 个 D 进程 + 1 个 proxy（路由）。 |
| **替代方案** | 单进程动态 role（拒绝，复杂度爆炸）。 |

### ADR-4: HTTP-based Bootstrap Server（不用 ZMQ）

| | |
|---|---|
| **决策** | P 节点暴露 HTTP server 用于 D 节点查询路由信息（host:port、PD rank 映射）。bootstrap 端口与传输端口分开。 |
| **理由** | (1) sglang 也用 HTTP（`CommonKVBootstrapServer.run`）；(2) HTTP 调试方便（curl 可验证）；(3) ZMQ 用于「KV 传输完成通知」（高频、小消息），HTTP 用于「P-D 握手」（低频、有状态），用途分离；(4) sgl-jax 现有 ZMQ 主要服务 scheduler/tokenizer 通信，PD bootstrap 用 HTTP 不污染。 |
| **影响** | 新增 `disaggregation_bootstrap_port`（默认 8998）；P 进程启动时绑定。 |
| **替代方案** | (a) ZMQ REQ/REP：拒绝，理由同上；(b) gRPC：拒绝，引入新依赖；(c) 复用 sgl-jax 现有 ZMQ：拒绝，会与 scheduler 通信冲突。 |

### ADR-5: D2H staging 默认 OFF（path-A 当前未 plumbed）

| | |
|---|---|
| **决策** | `--disaggregation-enable-d2h` 默认 `False`，PD 主路径走 path-B（D 直接从 P HBM pull）。path-A（P HBM → P pinned host → D pull → D HBM）的设计仍保留（参见 [host-pool RFC](./2026-05-25-pd-host-pool-side-channel.md) 「当前实现状态」节），但 scheduler 当前显式向 mixin 传 `host_pool=None`，开启 `--disaggregation-enable-d2h` 会在启动期 raise。 |
| **理由** | (1) `QueueHostKVPool` 当前 buffer shape 仍是旧的 token-major 契约，与 page-bucketed fused payload 不匹配；(2) D 侧没有 host → HBM 写回路径；(3) prefill mixin 的 `producer_handoff()` 路径没有把 `host_pool` 实际传进去。在这三件事修复之前 path-A 不可用；本轮交付优先把 path-B 跑通到 production-like benchmark/eval 入口。 |
| **影响** | ServerArgs `disaggregation_enable_d2h` 默认 False，开启时直接 raise。`QueueHostKVPool` 类与单测保留，但运行时未被 scheduler 创建。host-pool RFC「当前实现状态」节列出接通 path-A 的 3 项 plumbing 工作。 |
| **替代方案** | (a) 强行接 path-A：拒绝，会让本轮交付推迟，且当前主场景（path-B）已经能跑通 production-like benchmark；(b) 删除 `QueueHostKVPool` 与开关：拒绝，path-A 仍是后续 hardening 的目标，类与单测保留可在 plumbing 完成时直接复用。 |

### ADR-6: Mixin 模式分隔 scheduler 事件循环

| | |
|---|---|
| **决策** | 新增 `SchedulerDisaggregationPrefillMixin` + `SchedulerDisaggregationDecodeMixin`（参考 sglang 调研 §3.5）。`Scheduler` 主类不修改 P / D 主路径代码。 |
| **理由** | (1) Mixin 模式让 PD 逻辑不污染 Scheduler 主类（与 sglang 一致）；(2) `event_loop_normal_disagg_prefill` / `event_loop_normal_disagg_decode` 是独立 event loop，与现有 `event_loop_normal` / `event_loop_overlap` 并列；(3) 修改 scheduler 时影响范围小。 |
| **影响** | scheduler.py 中 `run_scheduler_process` dispatch 加 disaggregation_mode 分支。 |
| **替代方案** | 在 Scheduler 主类内 if-else：拒绝，污染。 |

### ADR-7: PD 主路径绕过 tree_cache

| | |
|---|---|
| **决策** | P 侧的 `send_kv_chunk` 直接从 `req_to_token_pool` 取 KV indices 调 `KVSender.send`；D 侧的 `process_decode_queue` 收到 KV 后直接写 paged KV pool，不调 `tree_cache.insert`。tree 插入由 D 完成 decode 后通过标准 `cache_finished_req` 触发。 |
| **理由** | (1) 与 sglang 一致（sglang 调研 §3.6.2）；(2) PD 是"请求级"传输（一次性），与 tree 的"prefix 级"复用语义本质不同；(3) 这样设计保证 PD 与 ChunkCache + UnifiedRadixCache 都兼容（PD 不依赖 tree 存在）。 |
| **影响** | `BasePrefixCache.get_kv_indices_for_send` / `insert_received_kv`（RFC-0 §12.2 已定义）是 PD 的接入点。 |
| **替代方案** | 通过 tree_cache.insert 走 KV：拒绝，会破坏与 ChunkCache 兼容。 |

### ADR-8: 不支持 partial KV pulling（已知 jax.experimental.transfer 限制）

| | |
|---|---|
| **决策** | D 端必须拉取 P 端发送的全部 prefill KV，即使 D 本地已有部分 prefix（命中 tree_cache）。 |
| **理由** | (1) `jax.experimental.transfer` 不支持 partial pull（jax-api 调研 §6.1）；(2) sglang 也有同样限制（sglang 调研：JAX P2P 不支持 RDMA 风格部分拉取）；(3) 处理 partial 会显著增加协议复杂度。 |
| **影响** | D 端 prefix cache 命中无法直接节省传输；通过 P 端减少发送量（只发未命中前缀）来缓解。这是 RFC-2 的「已知限制」，需要文档化。 |
| **替代方案** | 实现部分拉取——拒绝，API 不支持。 |

### ADR-9: 第一版不支持 D 节点 RadixCache（与 sglang 默认行为对齐）

| | |
|---|---|
| **决策** | **第一版** sgl-jax PD 部署中，D 节点（`disaggregation_mode=decode`）**只支持 ChunkCache**，不实现 D + RadixCache/UnifiedRadixCache 路径。不引入 sglang 的 `--disaggregation-decode-enable-radix-cache` opt-in flag——未来如有需求再补。 |
| **理由** | (1) **D 端 prefix cache 收益极低**：受 ADR-8 限制（无 partial pull），D 端命中 prefix 也无法节省 PD 传输；D 端只 decode 不 prefill，prefix 复用价值远小于 P 端。 (2) **与 sglang 默认行为对齐**：sglang 默认 D 节点走 ChunkCache（`disaggregation_decode_enable_radix_cache` 默认 False），且与 SWA/Mamba 模型完全不兼容（`raise ValueError`）。sgl-jax 第一版直接不引入这个 opt-in 简化配置矩阵。 (3) **YAGNI**：D + RadixCache 是少数场景需求，等真正有用户需要时再实现并讨论与 SWA/Mamba 的兼容性。 (4) **PD 与 cache 类型在接口层正交**：tree_cache × PD 接口层正交（RFC-0 §1.3），仅 RFC-2 第一版在部署层加策略限制；接口本身没有耦合。 |
| **影响** | (1) `kv_cache_builder` 在 D mode 下走 ChunkCache 分支，配置 `disable_radix_cache=False` 时输出 warning 提示「第一版未实现 D + RadixCache，后续可加」；(2) `server_args` 不引入 `disaggregation_decode_enable_radix_cache` flag；(3) D 节点不参与 HiCache（ChunkCache 无 hook）——但 P 节点完全可用 HiCache（与 RFC-1 一致）；(4) 文档明确「第一版 PD 部署中 HiCache 仅在 P 节点生效」。 |
| **未来扩展点** | 若后续有 D + RadixCache 需求，按 sglang 模式新增 `disaggregation_decode_enable_radix_cache` opt-in flag + 必要的兼容性检查（D + RadixCache 与 SWA/Mamba 不兼容，参考 sglang `kv_cache_builder.py:187-199`）。当前第一版不阻塞这条扩展路径。 |
| **替代方案** | (a) 第一版就实现 sglang 风格 opt-in flag——拒绝，YAGNI；(b) 永久禁用 D + RadixCache——拒绝，限制过强，应保留未来扩展可能性。 |

---

## 3. 模块全景

### 3.1 模块依赖图

```
                ┌───────────────────────────────────────────────┐
                │  Scheduler (event loop dispatch)              │
                │  run_scheduler_process:                       │
                │    if disagg_mode == NULL: event_loop_normal  │
                │    if PREFILL:  event_loop_normal_disagg_pf   │ ──┐
                │    if DECODE:   event_loop_normal_disagg_de   │ ──┤
                └───────────────────────────────────────────────┘   │
                                                                    │
            ┌───────────────────────────────────────────────────────┘
            │
   ┌────────▼──────────────────────┐    ┌─────────────────────────────┐
   │ SchedulerDisaggregation-       │    │ SchedulerDisaggregation-     │
   │ PrefillMixin §9.1              │    │ DecodeMixin §9.2             │
   │ • event_loop_normal_disagg_pf  │    │ • event_loop_normal_disagg_de│
   │ • process_prefill_chunk        │    │ • process_decode_queue        │
   │ • send_kv_chunk                │    │ • get_new_prebuilt_batch      │
   │ (driving P 侧 KV send)         │    │ (driving D 侧 KV recv)        │
   └────────┬───────────────────────┘    └─────────────────┬───────────┘
            │                                              │
            │              ┌───────────────────────────────┘
            ▼              ▼
   ┌──────────────────────────────────────────────────────────┐
   │ KVManager (4-tuple base, §7)                              │
   │ • 单例 per process                                         │
   │ • 持有 KVArgs (KV pool 指针 + sharding 等)                 │
   │ • 创建 KVSender / KVReceiver instance per req              │
   └──────┬─────────────────────────────────┬─────────────────┘
          │                                 │
   ┌──────▼─────────┐               ┌───────▼──────────────┐
   │ KVSender §7     │               │ KVReceiver §7         │
   │ (P 侧 per req)  │               │ (D 侧 per req)        │
   │ • init          │               │ • init                │
   │ • send (异步)    │               │ • send_metadata       │
   │ • poll → KVPoll │               │ • poll → KVPoll       │
   └──────┬─────────┘               └───────┬──────────────┘
          │                                 │
          │                                 │
          └────────────┬────────────────────┘
                       ▼
           ┌──────────────────────────┐
           │ JaxTransferKVEngine §4   │   (per process 单例)
           │ (KVTransferEngine 实现)    │
           │                          │
           │ • start_transfer_server  │   ← jax.experimental.transfer
           │ • await_pull (P 侧)      │
           │ • connect + pull (D 侧)  │
           │ • notify_pull_done       │   ← 复用 ZMQ
           └────────────┬─────────────┘
                        │
                ┌───────┴────────────────┐
                │                        │
        ┌───────▼──────────┐    ┌────────▼─────────────┐
        │ QueueHostKVPool   │    │ BootstrapServer §6   │
        │ §5 (HostKVPool    │    │ (P-D 握手 + UUID     │
        │  Queue 实现)      │    │  协商, HTTP-based)   │
        │                   │    │                      │
        │ 单机 D2H staging  │    │ P 侧 server          │
        │ (ADR-5)           │    │ D 侧 client          │
        └───────────────────┘    └──────────────────────┘
```

### 3.2 文件路径

| 路径 | 操作 | 行数估计 |
|---|---|---|
| `python/sgl_jax/srt/disaggregation/__init__.py` | RFC-0 提供 placeholder；**填充导出** | +20 |
| `python/sgl_jax/srt/disaggregation/kv_transfer_engine.py` | RFC-0 提供 ABC；**新增 `JaxTransferKVEngine` 实现** | +400 |
| `python/sgl_jax/srt/disaggregation/base/__init__.py` | 新增 | +10 |
| `python/sgl_jax/srt/disaggregation/base/kv_manager.py` | 4-tuple ABC + Common* 中间层 | +400 |
| `python/sgl_jax/srt/disaggregation/jax_transfer/__init__.py` | 新增 backend 子目录 | +10 |
| `python/sgl_jax/srt/disaggregation/jax_transfer/conn.py` | `JaxTransferKVManager` / `Sender` / `Receiver` | +500 |
| `python/sgl_jax/srt/disaggregation/bootstrap.py` | HTTP-based BootstrapServer | +300 |
| `python/sgl_jax/srt/disaggregation/prefill.py` | `SchedulerDisaggregationPrefillMixin` + `PrefillBootstrapQueue` | +400 |
| `python/sgl_jax/srt/disaggregation/decode.py` | `SchedulerDisaggregationDecodeMixin` + `DecodePreallocQueue` + `DecodeTransferQueue` + `DecodeReqToTokenPool` | +600 |
| `python/sgl_jax/srt/disaggregation/utils.py` | enums (`DisaggregationMode`, `TransferBackend`) + factory | +150 |
| `python/sgl_jax/srt/mem_cache/host_kv_pool.py` | RFC-0/RFC-1 已扩展；**新增 `QueueHostKVPool` 实现** | +250 |
| `python/sgl_jax/srt/managers/scheduler.py` | **加 PD event loop dispatch** | +50 |
| `python/sgl_jax/srt/mem_cache/base_prefix_cache.py` | RFC-0 已加 hooks；**实装 `get_kv_indices_for_send` 默认实现** | +40 |
| `python/sgl_jax/srt/server_args.py` | RFC-0 已加配置项；**加默认值 + 验证** | +30 |
| `examples/disagg/run_single_host.sh` | 部署脚本示例 | +60 |
| `examples/disagg/toy_proxy_server.py` | FastAPI proxy | +200 |
| `python/sgl_jax/test/disaggregation/test_jax_transfer_engine.py` | 单元测试 | +300 |
| `python/sgl_jax/test/disaggregation/test_bootstrap.py` | 单元测试 | +200 |
| `python/sgl_jax/test/disaggregation/test_pd_e2e_single_host.py` | 端到端 | +400 |
| `python/sgl_jax/test/disaggregation/test_pd_with_chunkcache.py` | ChunkCache + PD 测试 | +250 |
| `python/sgl_jax/test/disaggregation/test_pd_with_unified.py` | UnifiedRadixCache + PD 测试 | +250 |

总计新增约 4870 行。

---

## 4. 详细设计：JaxTransferKVEngine

### 4.1 接口实现

```python
# python/sgl_jax/srt/disaggregation/kv_transfer_engine.py (RFC-0 ABC 之后 append)

from jax.experimental.transfer import start_transfer_server
import threading

_TRANSFER_SERVER_LOCK = threading.Lock()
_GLOBAL_TRANSFER_SERVER = None   # singleton per process (ADR-2)

class JaxTransferKVEngine(KVTransferEngine):
    """
    基于 jax.experimental.transfer 的 PD KV 传输引擎.

    设计:
    - 每个进程一个 transfer server 单例 (ADR-2)
    - server 在 start() 时创建, 多次 start() 调用幂等
    - 支持 multi-host (每进程独立 server, 通过 IP:Port 路由)
    """

    def __init__(self, config: KVTransferConfig):
        self.config = config
        self.runner = None
        self.server = None
        self._zmq_notifier: Optional["ZmqPullNotifier"] = None
        self._connections: dict[str, Any] = {}  # remote_addr → conn
        self._connections_lock = threading.Lock()

    def register_runner(self, runner: "ModelRunner") -> None:
        """
        注入 ModelRunner, 获取 kv pool / mesh / sharding.

        调用方约定 (sgl-jax 集成):
        - Scheduler 通过 self.tp_worker.model_runner 获取 ModelRunner
        - 在 Scheduler._init_disaggregation_{prefill,decode}() 中调用:
              self.disagg_kv_manager.engine.register_runner(self.tp_worker.model_runner)
        - sgl-jax KVCache 不持有反向引用到 runner; 由 scheduler 显式注入
        - kv_sharding 取自 device_pool.kv_sharding (sgl-jax KVCache 暴露此属性,
          见 memory_pool.py:410 MHATokenToKVPool 等)
        """
        self.runner = runner
        device_pool = runner.kv_caches[0] if hasattr(runner, "kv_caches") else runner.kv_pool
        self.kv_sharding = device_pool.kv_sharding
        # 用于多 host 拼接: 数据 sharding 与 kv pool 一致 (沿 'tensor' 轴分片)

    def start(self) -> None:
        """启动 transfer server 单例 + ZMQ 侧通道"""
        global _GLOBAL_TRANSFER_SERVER
        with _TRANSFER_SERVER_LOCK:
            if _GLOBAL_TRANSFER_SERVER is None:
                _GLOBAL_TRANSFER_SERVER = self._create_transfer_server()
            self.server = _GLOBAL_TRANSFER_SERVER

        # ZMQ 侧通道用于 pull-done 通知
        self._zmq_notifier = ZmqPullNotifier(
            role=self.config.role,
            host=self.config.host_ip,
            port=self.config.side_channel_port,
        )
        self._zmq_notifier.start()

    def _create_transfer_server(self):
        """创建底层 jax.experimental.transfer server"""
        server_addr = f"{self.config.host_ip}:{self.config.port}"
        # transport_addr 0 端口让系统自动选, 避免冲突
        transport_addrs = [f"{self.config.host_ip}:0"] * self.config.channel_number
        return start_transfer_server(
            jax.local_devices()[0].client,
            server_addr,
            transport_addrs,
            max_num_parallel_copies=self.config.max_parallel_copies,
            transfer_size=self.config.transfer_size_bytes,
            use_raw_buffers=False,
        )

    def shutdown(self) -> None:
        if self._zmq_notifier is not None:
            self._zmq_notifier.stop()
        # transfer server 是全局单例, 不主动 shutdown (进程退出时由 JAX 清理)

    # ===== Producer 侧 (P) =====
    def await_pull(self, uuid: bytes, kv_data: jax.Array,
                   timeout_seconds: float = 180) -> TransferStatus:
        """
        P 侧: 注册 kv_data, **阻塞**等待 D 来 pull (或 server 内部超时).

        重要 (与 ABC 文档对齐):
        - 这个方法**阻塞**整个调用线程, 直到 pull 完成或失败
        - 因此调用方 (JaxTransferKVSender._do_send) 必须在 ThreadPoolExecutor
          后台线程里调用本方法
        - kv_data 接受 device array (HBM) 或 pinned_host array
          (tpu-inference `tpu_connector.py:710` 直接 await_pull(device array),
          `tpu_connector.py:760` 同样接受 host_buffer; jax.experimental.transfer
          server 内部根据 array placement 处理 D2H)
        """
        try:
            self.server.await_pull(uuid, kv_data)
            return TransferStatus(uuid=uuid, state="done")
        except Exception as e:
            return TransferStatus(uuid=uuid, state="failed")

    # ===== Consumer 侧 (D) =====
    def connect(self, remote_host: str, remote_port: int) -> "Connection":
        """D 侧建立连接 (复用同一 remote 的 connection)"""
        addr = f"{remote_host}:{remote_port}"
        with self._connections_lock:
            if addr not in self._connections:
                self._connections[addr] = self.server.connect(addr)
            return self._connections[addr]

    def pull(self, conn: "Connection", uuid: bytes,
             kv_spec: list[jax.ShapeDtypeStruct]) -> list[jax.Array]:
        """
        D 侧拉取. kv_spec 必须用 device sharding (jax.experimental.transfer
        会自动落到对应 device HBM, 无需再 device_put).
        """
        result = conn.pull(uuid, kv_spec)
        # 等待 pull 完成 (轮询 is_ready)
        while not all(chunk.is_ready() for chunk in result):
            time.sleep(0.001)
        return result

    def notify_pull_done(self, uuid: bytes, target_host: str,
                         target_port: int) -> None:
        """D 完成 pull 后通知 P 释放 buffer"""
        self._zmq_notifier.send_done(uuid, target_host, target_port)
```

### 4.2 ZMQ pull-done 通知

```python
# 同文件内

class ZmqPullNotifier:
    """
    侧通道: D 通知 P "pull 完成, 可以释放 KV buffer".

    协议:
      D → P:  ZMQ DEALER → ROUTER
      消息体: msgpack({"uuid": bytes})

    端口:
      side_channel_port (默认 9600, 可配置)
    """

    def __init__(self, role: str, host: str, port: int):
        self.role = role
        self.host = host
        self.port = port
        self.ctx = zmq.Context.instance()
        self.socket: Optional[zmq.Socket] = None
        self.listener_thread: Optional[threading.Thread] = None
        # pending_callbacks 在 register (main thread) 和 _listen_loop (bg thread)
        # 都会被 mutation, 需要 lock 保护
        self.pending_callbacks: dict[bytes, Callable] = {}
        self._callbacks_lock = threading.Lock()

    def start(self) -> None:
        if self.role == "prefill":
            # P 侧: ROUTER 监听
            self.socket = self.ctx.socket(zmq.ROUTER)
            self.socket.bind(f"tcp://*:{self.port}")
            self.listener_thread = threading.Thread(
                target=self._listen_loop,
                name="pd_pull_notify_listener",
                daemon=True,
            )
            self.listener_thread.start()
        elif self.role == "decode":
            # D 侧: DEALER 客户端 (per target)
            pass   # 按需建立, 不预连接
        else:
            raise ValueError(f"Unknown role: {self.role}")

    def _listen_loop(self):
        """P 侧后台线程: 收到 D 的 done 通知 → 调 callback"""
        while True:
            try:
                identity, msg = self.socket.recv_multipart()
                payload = msgpack.unpackb(msg)
                uuid = payload["uuid"]
                with self._callbacks_lock:
                    cb = self.pending_callbacks.pop(uuid, None)
                if cb is not None:
                    cb(uuid)
            except Exception:
                if self.socket.closed:
                    break

    def register_callback(self, uuid: bytes, cb: Callable) -> None:
        """P 侧 await_pull 前注册 callback"""
        with self._callbacks_lock:
            self.pending_callbacks[uuid] = cb

    def send_done(self, uuid: bytes, target_host: str, target_port: int) -> None:
        """D 侧: 向 P 发 done 通知"""
        socket = self.ctx.socket(zmq.DEALER)
        socket.connect(f"tcp://{target_host}:{target_port}")
        socket.send(msgpack.packb({"uuid": uuid}))
        socket.close()

    def stop(self) -> None:
        if self.socket is not None:
            self.socket.close()
```

### 4.3 P-D pull 路径的两种模式（ADR-5）

```python
class JaxTransferKVEngine:

    def producer_handoff(self, uuid: bytes,
                         device_kv: list[jax.Array],
                         use_d2h_staging: bool,
                         staging_pool: Optional["QueueHostKVPool"] = None,
                         ) -> TransferStatus:
        """
        P 侧统一入口. 根据 use_d2h_staging 选择两种路径:

        路径 A (use_d2h_staging=True, 默认单机):
            HBM → pinned host (D2H) → await_pull → 网络 → D HBM
            优点: 拷完即可释放 HBM
            实现: 先 copy_to_host 到 staging_pool, 再 await_pull(staging_array)

        路径 B (use_d2h_staging=False, 默认多主机):
            HBM → await_pull → 网络 → D HBM
            优点: 减少一次拷贝
            实现: 直接 await_pull(device_kv) (server 内部处理 D2H)
        """
        if use_d2h_staging:
            assert staging_pool is not None
            staged = staging_pool.copy_from_device(device_kv)
            # await_pull 阻塞, 完成时 D2H staging 数据已传走
            status = self.await_pull(uuid, staged.array)
            # pull 完成后归还 staging buffer (在 ZMQ done callback 里做)
            return status
        else:
            return self.await_pull(uuid, device_kv)
```

---

## 5. 详细设计：QueueHostKVPool（D2H staging）

### 5.1 与 LRUHostKVPool 的对比

| | LRUHostKVPool (RFC-1) | QueueHostKVPool (RFC-2) |
|---|---|---|
| 用途 | HiCache L2 长生命周期缓存 | PD D2H staging 短生命周期 |
| 容量管理 | LRU + lock_ref | 固定大小 queue, 借出/归还 |
| Buffer 引用 | underlying buffer 会被 .at[].set() 重新绑定 | **每个 entry 是独立的 jax.Array, buffer 引用稳定** |
| HostBufferHandle.buffer | None (用 read_indices) | **可填稳定 jax.Array** |
| 接口 | alloc/free/evict/lock_ref/read_indices/write_indices | alloc/free/get_buffer/put_buffer |

### 5.2 实现

```python
# python/sgl_jax/srt/mem_cache/host_kv_pool.py (RFC-0 ABC + RFC-1 LRU 之后 append)

import queue

class QueueHostKVPool(HostKVPool):
    """
    PD D2H staging 用. Queue 管理预分配 buffer.

    设计:
    - 预分配 pool_size 个独立 buffer (每个是独立的 jax.Array, 不共享 underlying memory)
    - get_buffer() 从 queue 借出, put_buffer() 归还
    - 每个 buffer 大小固定 (max_blocks_per_req)
    - alloc/free 操作 token-level (内部 reserve 一个 buffer)
    """

    def __init__(
        self,
        pool_size: int,                    # 预分配 buffer 数量
        max_tokens_per_buffer: int,        # 每 buffer 的 token 容量
        layer_num: int,
        kv_head_per_rank: int,
        head_dim: int,
        dtype: jnp.dtype,
        mesh: jax.sharding.Mesh,
        partition_spec: PartitionSpec,
    ):
        self.pool_size = pool_size
        self.max_tokens_per_buffer = max_tokens_per_buffer
        self.host_sharding = NamedSharding(
            mesh, partition_spec, memory_kind="pinned_host"
        )

        # 预分配独立 buffer
        per_buffer_shape = (max_tokens_per_buffer, layer_num,
                            kv_head_per_rank, 2, head_dim)
        self.buffers: list[jax.Array] = []
        for _ in range(pool_size):
            buf = jax.device_put(
                jnp.zeros(per_buffer_shape, dtype=dtype),
                self.host_sharding,
            )
            jax.block_until_ready(buf)
            self.buffers.append(buf)

        # Queue 管理可用 buffer ID
        self.available_queue: queue.Queue = queue.Queue(maxsize=pool_size)
        for i in range(pool_size):
            self.available_queue.put(i)

        # 跟踪借出状态
        self.in_use: set[int] = set()

    # ===== HostKVPool ABC =====
    def alloc(self, num_tokens: int) -> Optional[HostBufferHandle]:
        """
        借一个 buffer. num_tokens 必须 ≤ max_tokens_per_buffer (一个 req 用一个 buffer).
        失败 (queue 空) 返回 None.
        """
        if num_tokens > self.max_tokens_per_buffer:
            raise ValueError(
                f"Request {num_tokens} > max_tokens_per_buffer {self.max_tokens_per_buffer}"
            )
        try:
            buffer_id = self.available_queue.get_nowait()
        except queue.Empty:
            return None
        self.in_use.add(buffer_id)
        # indices 是 buffer 内的 [0, num_tokens) 子段
        indices = jnp.arange(num_tokens, dtype=jnp.int32)
        return HostBufferHandle(
            indices=indices,
            buffer_id=buffer_id,
            buffer=self.buffers[buffer_id],  # 稳定引用 (Queue 模式可保留)
        )

    def free(self, handle: HostBufferHandle) -> None:
        if handle.buffer_id in self.in_use:
            self.in_use.remove(handle.buffer_id)
            self.available_queue.put(handle.buffer_id)

    def available_size(self) -> int:
        return self.available_queue.qsize() * self.max_tokens_per_buffer

    def total_size(self) -> int:
        return self.pool_size * self.max_tokens_per_buffer

    # ===== 数据读写 =====
    def read_indices(self, indices: jax.Array) -> jax.Array:
        """Queue 模式下不直接用 read_indices, 调用方持 handle.buffer"""
        raise NotImplementedError(
            "QueueHostKVPool: access data via handle.buffer directly"
        )

    def write_indices(self, indices: jax.Array, kv_data: jax.Array) -> None:
        raise NotImplementedError(
            "QueueHostKVPool: write data via copy_from_device"
        )

    # ===== PD 专用 =====
    def get_buffer(self) -> tuple[int, HostBufferHandle]:
        """Alias for alloc with full buffer size"""
        handle = self.alloc(self.max_tokens_per_buffer)
        if handle is None:
            raise RuntimeError("QueueHostKVPool exhausted")
        return handle.buffer_id, handle

    def put_buffer(self, buffer_id: int) -> None:
        """归还 buffer"""
        if buffer_id in self.in_use:
            self.in_use.remove(buffer_id)
            self.available_queue.put(buffer_id)

    def copy_from_device(self, device_kv: jax.Array) -> "StagedData":
        """
        从 device kv (HBM) 拷贝到一个空闲 buffer (pinned host).
        返回 StagedData (含 handle + 实际 token 数).

        Buffer 引用语义:
        - QueueHostKVPool 的"stable buffer"是指 buffer ID 稳定 (借用期内不复用)
        - 但 JAX `.at[].set()` 是 functional update, 返回新 array; 严格意义上没有
          "stable jax.Array 引用". 我们通过及时更新 self.buffers[id] 和 handle.buffer
          来保证「最新引用」总能拿到
        - 调用方约定: 不缓存 handle.buffer 跨多次 copy_from_device 调用;
          每次需要数据时通过 returned StagedData.array (最新引用) 读取
        """
        buffer_id, handle = self.get_buffer()
        num_tokens = int(device_kv.shape[0])
        # 先 D2H 到临时 array
        staged_temp = jax.device_put(device_kv, self.host_sharding)
        jax.block_until_ready(staged_temp)
        # In-place update 现有 buffer 的前 num_tokens 行 (functional update,
        # 返回新 array, 自身预分配的内存被复用)
        new_buffer = self.buffers[buffer_id].at[:num_tokens].set(staged_temp)
        self.buffers[buffer_id] = new_buffer
        # 更新 handle.buffer 指向最新引用
        handle.buffer = new_buffer
        return StagedData(handle=handle, array=new_buffer, num_tokens=num_tokens)


@dataclass
class StagedData:
    """D2H staging 的产物, 用于 await_pull"""
    handle: HostBufferHandle
    array: jax.Array              # 实际 staging array (= handle.buffer)
    num_tokens: int
```

### 5.3 容量配置

`pool_size`（预分配 buffer 数）= 单机模式下的最大并发 P 请求数。默认 64（参考 tpu-inference `TPU_MAX_HOST_KV_BUFFER_SIZE`）。

`max_tokens_per_buffer` = `model_config.max_total_num_tokens / pool_size`（粗略，可调）。

实际值由 `--disaggregation-d2h-pool-size` 和 `--disaggregation-d2h-max-tokens` 控制（§17）。

---

## 6. 详细设计：Bootstrap Server

### 6.1 协议

简洁的 HTTP RESTful API（FastAPI 实现）。

| Endpoint | Method | 说明 |
|---|---|---|
| `/register_prefill` | POST | P 启动时向 bootstrap 注册自己的 `(host, port, tp_rank, tp_size, dp_rank, dp_size)` |
| `/list_prefills` | GET | D 查询当前在线的 P 节点列表 |
| `/get_prefill_info` | GET `?bootstrap_room=<int>` | D 根据 bootstrap_room 查询对应 P 的路由信息 |
| `/health` | GET | 健康检查 |

### 6.2 实现

```python
# python/sgl_jax/srt/disaggregation/bootstrap.py

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import threading

class PrefillInfo(BaseModel):
    host: str
    port: int                          # KV transfer port
    side_channel_port: int              # ZMQ 通知端口
    tp_rank: int
    tp_size: int
    system_dp_rank: int = 0
    # 注: sgl-jax SPMD DP attention 下, 一个 P 进程可能持有多个 dp_rank
    # (mesh 形状 (dp_size, attention_tp_size)), system_dp_rank 不能完全描述,
    # 但 PD 路由级别用 process-level dp_rank=0 标识进程身份即可.
    # KVArgs 中的 dp_rank 字段才是真正用于 KV shard 切分的 rank.

class BootstrapServer:
    """
    P-D 握手 + 路由协商.

    部署模型: 集中式 bootstrap server (单进程), 多个 P/D 进程作为 client.
    生产环境可以与 proxy server (toy_proxy_server.py) 合体.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8998):
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.prefills: dict[str, PrefillInfo] = {}  # key = f"{host}:{port}"
        self._last_seen: dict[str, float] = {}      # 心跳时间戳, 用于 TTL
        self.lock = threading.Lock()
        self._setup_routes()
        self._server_thread: Optional[threading.Thread] = None

    def _setup_routes(self):
        @self.app.post("/register_prefill")
        def register_prefill(info: PrefillInfo):
            with self.lock:
                key = f"{info.host}:{info.port}"
                self.prefills[key] = info
                self._last_seen[key] = time.time()
            return {"status": "ok", "key": key}

        @self.app.post("/heartbeat")
        def heartbeat(key: str):
            """P 周期心跳, 用于 TTL 健康检查"""
            with self.lock:
                if key in self.prefills:
                    self._last_seen[key] = time.time()
                    return {"status": "ok"}
                return {"status": "unknown"}

        @self.app.post("/unregister_prefill")
        def unregister_prefill(key: str):
            with self.lock:
                self.prefills.pop(key, None)
                self._last_seen.pop(key, None)
            return {"status": "ok"}

        @self.app.get("/list_prefills")
        def list_prefills():
            self._evict_stale()
            with self.lock:
                return {"prefills": list(self.prefills.values())}

        @self.app.get("/get_prefill_info")
        def get_prefill_info(bootstrap_room: int):
            self._evict_stale()
            with self.lock:
                prefills = list(self.prefills.values())
            if not prefills:
                return {"error": "No prefill nodes available"}
            chosen = prefills[bootstrap_room % len(prefills)]
            return chosen.dict()

        @self.app.get("/health")
        def health():
            return {"status": "ok", "num_prefills": len(self.prefills)}

    def _evict_stale(self, ttl_seconds: float = 30.0):
        """淘汰超过 ttl_seconds 没心跳的 P (默认 30s)"""
        now = time.time()
        with self.lock:
            stale = [k for k, t in self._last_seen.items()
                     if now - t > ttl_seconds]
            for k in stale:
                self.prefills.pop(k, None)
                self._last_seen.pop(k, None)

    def run(self) -> None:
        """启动 server (后台线程, non-blocking)"""
        def serve():
            uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")
        self._server_thread = threading.Thread(target=serve, daemon=True)
        self._server_thread.start()

    def stop(self) -> None:
        # FastAPI/Uvicorn 不优雅退出, 但 daemon thread 进程退出时会终止
        pass
```

### 6.3 客户端工具

```python
class BootstrapClient:
    """P 和 D 用的 bootstrap 客户端"""

    def __init__(self, bootstrap_url: str):
        self.url = bootstrap_url

    def register_prefill(self, info: PrefillInfo) -> str:
        resp = requests.post(f"{self.url}/register_prefill", json=info.dict())
        return resp.json()["key"]

    def get_prefill_info(self, bootstrap_room: int) -> PrefillInfo:
        resp = requests.get(
            f"{self.url}/get_prefill_info",
            params={"bootstrap_room": bootstrap_room},
        )
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        return PrefillInfo(**data)
```

---

## 7. 详细设计：4-tuple ABC（KVManager / Sender / Receiver）

### 7.1 ABC 设计（精简自 sglang）

```python
# python/sgl_jax/srt/disaggregation/base/kv_manager.py

@dataclass
class KVArgs:
    """KV transfer 元数据 (sglang 同名结构的精简版, 适配 sgl-jax SPMD)"""
    # sgl-jax 用 jax.Array 替代 sglang 的 raw pointers
    kv_pool: KVCache                        # 持有 KV pool 引用
    page_size: int
    layer_num: int
    kv_head_per_rank: int
    head_dim: int
    dtype: jnp.dtype
    # sgl-jax 特有
    mesh: jax.sharding.Mesh
    kv_sharding: NamedSharding
    is_mla: bool = False
    # SPMD DP attention (RFC-0 §3.5): 一个进程内多个 dp_rank, 必须按 rank 区分
    dp_rank: int = 0                        # 当前请求所属的 attention DP rank
    dp_size: int = 1
    # SWA 和 Recurrent-State 暂不支持 PD (后续 RFC 扩展)
    state_type: str = "none"               # "none" / "swa" / "recurrent_state"
    # 注: 不含 pp_rank/pp_size — sgl-jax 长期不规划支持 Pipeline Parallel

class KVPoll(IntEnum):
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4

class KVManager(ABC):
    """单例 per process. 持有 KV transfer engine + 元数据"""

    @abstractmethod
    def __init__(self, args: KVArgs, mode: DisaggregationMode,
                 server_args: ServerArgs): ...

    @abstractmethod
    def register_to_bootstrap(self) -> None: ...

    @abstractmethod
    def create_sender(self, bootstrap_addr: str,
                      bootstrap_room: int) -> "KVSender": ...

    @abstractmethod
    def create_receiver(self, bootstrap_addr: str,
                        bootstrap_room: int) -> "KVReceiver": ...

class KVSender(ABC):
    """Per request, P 侧"""
    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: int) -> None: ...

    @abstractmethod
    def send(self, kv_indices: jax.Array) -> None:
        """异步发起 KV 发送. 不阻塞."""

    @abstractmethod
    def poll(self) -> KVPoll: ...

    @abstractmethod
    def failure_exception(self) -> Exception: ...

class KVReceiver(ABC):
    """Per request, D 侧"""
    @abstractmethod
    def init(self, prefill_dp_rank: int) -> None: ...

    @abstractmethod
    def send_metadata(self, kv_indices: jax.Array, aux_index: int) -> None: ...

    @abstractmethod
    def poll(self) -> KVPoll: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def abort(self) -> None: ...
```

### 7.2 JaxTransfer backend 实现（精简）

```python
# python/sgl_jax/srt/disaggregation/jax_transfer/conn.py

class JaxTransferKVManager(KVManager):
    """JaxTransfer backend 的 KVManager 实现"""

    def __init__(self, args: KVArgs, mode: DisaggregationMode,
                 server_args: ServerArgs):
        self.args = args
        self.mode = mode
        self.server_args = server_args

        # 创建 KVTransferEngine
        engine_config = KVTransferConfig(
            role=mode.value,
            host_ip=get_host_ip(),
            port=server_args.disaggregation_kv_transfer_port,
            side_channel_port=server_args.disaggregation_side_channel_port,
            channel_number=server_args.disaggregation_channel_number,
        )
        self.engine = JaxTransferKVEngine(engine_config)
        # register_runner 在 Scheduler._init_disaggregation_{prefill,decode}() 中显式调用,
        # 因为 Manager.__init__ 时 ModelRunner 尚未就绪 (Scheduler 先创建 Manager).
        # 这里只设占位; 真正调用见 §9.3 scheduler wiring.
        self.engine.start()

        # ===== 后台线程池 (用于异步 send / recv) =====
        # 必须在使用前初始化, Sender._do_send / Receiver._do_pull 依赖这两个
        self._send_executor = ThreadPoolExecutor(
            max_workers=server_args.disaggregation_send_threads or 4,
            thread_name_prefix="disagg_send",
        )
        self._recv_executor = ThreadPoolExecutor(
            max_workers=server_args.disaggregation_recv_threads or 4,
            thread_name_prefix="disagg_recv",
        )

        # D2H staging pool (单机模式默认启用)
        self.staging_pool: Optional[QueueHostKVPool] = None
        if server_args.disaggregation_enable_d2h:
            self.staging_pool = QueueHostKVPool(
                pool_size=server_args.disaggregation_d2h_pool_size,
                max_tokens_per_buffer=server_args.disaggregation_d2h_max_tokens,
                layer_num=args.layer_num,
                kv_head_per_rank=args.kv_head_per_rank,
                head_dim=args.head_dim,
                dtype=args.dtype,
                mesh=args.mesh,
                partition_spec=args.kv_sharding.spec,
            )

        # Bootstrap client (D 侧用) / server (P 侧不主动启 server, 由部署脚本启)
        self.bootstrap_client = BootstrapClient(
            f"http://{server_args.disaggregation_bootstrap_host}:"
            f"{server_args.disaggregation_bootstrap_port}"
        )

    def register_to_bootstrap(self) -> None:
        if self.mode != DisaggregationMode.PREFILL:
            return
        info = PrefillInfo(
            host=self.engine.config.host_ip,
            port=self.engine.config.port,
            side_channel_port=self.engine.config.side_channel_port,
            tp_rank=self.args.kv_pool.tp_rank,
            tp_size=self.args.kv_pool.tp_size,
        )
        self.bootstrap_client.register_prefill(info)

    def create_sender(self, bootstrap_addr: str,
                      bootstrap_room: int) -> "JaxTransferKVSender":
        return JaxTransferKVSender(self, bootstrap_room)

    def create_receiver(self, bootstrap_addr: str,
                        bootstrap_room: int) -> "JaxTransferKVReceiver":
        return JaxTransferKVReceiver(self, bootstrap_room)


class JaxTransferKVSender(KVSender):
    """P 侧 per request"""

    def __init__(self, manager: JaxTransferKVManager, bootstrap_room: int):
        self.manager = manager
        self.bootstrap_room = bootstrap_room
        self.uuid = uuid.uuid4().bytes
        self.state = KVPoll.Bootstrapping
        self.last_error: Optional[Exception] = None

    def init(self, num_kv_indices: int, aux_index: int) -> None:
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index
        self.state = KVPoll.WaitingForInput

    def send(self, kv_indices: jax.Array) -> None:
        """异步发起. 提交到后台线程."""
        # 从 KV pool 按 indices 取 KV (类似 sglang CommonKVManager.get_mha_kv_ptrs)
        device_kv = self.manager.args.kv_pool.get_kv_by_indices(kv_indices)

        # 启动后台 task
        self._send_future = self.manager._send_executor.submit(
            self._do_send, device_kv
        )
        self.state = KVPoll.Transferring

    def _do_send(self, device_kv: jax.Array) -> None:
        """后台线程: D2H staging (可选) → await_pull"""
        try:
            use_staging = self.manager.staging_pool is not None
            status = self.manager.engine.producer_handoff(
                uuid=self.uuid,
                device_kv=device_kv,
                use_d2h_staging=use_staging,
                staging_pool=self.manager.staging_pool,
            )
            if status.state == "done":
                self.state = KVPoll.Success
            else:
                self.state = KVPoll.Failed
        except Exception as e:
            self.last_error = e
            self.state = KVPoll.Failed

    def poll(self) -> KVPoll:
        return self.state

    def failure_exception(self) -> Exception:
        return self.last_error or RuntimeError("Unknown failure")


class JaxTransferKVReceiver(KVReceiver):
    """D 侧 per request"""

    def __init__(self, manager: JaxTransferKVManager, bootstrap_room: int):
        self.manager = manager
        self.bootstrap_room = bootstrap_room
        self.state = KVPoll.Bootstrapping
        self.received_kv: Optional[jax.Array] = None
        self.uuid: Optional[bytes] = None

    def init(self, prefill_dp_rank: int) -> None:
        # 查 bootstrap 获取 P 路由信息
        self.prefill_info = self.manager.bootstrap_client.get_prefill_info(
            bootstrap_room=self.bootstrap_room
        )
        self.state = KVPoll.WaitingForInput

    def send_metadata(self, kv_indices: jax.Array, aux_index: int) -> None:
        """D 侧告知 P 自己的 kv_indices 和 aux_index"""
        # 注: 实际的 metadata 通过 HTTP / 简单消息发给 P
        # 这里简化: bootstrap_room 包含 metadata 路由信息
        self.kv_indices = kv_indices
        self.aux_index = aux_index
        # 启动 pull
        self._pull_future = self.manager._recv_executor.submit(self._do_pull)
        self.state = KVPoll.Transferring

    def _do_pull(self) -> None:
        """后台线程: connect + pull"""
        try:
            conn = self.manager.engine.connect(
                self.prefill_info.host, self.prefill_info.port
            )
            kv_spec = self._build_kv_spec()
            # uuid 由 P 通过 bootstrap_room 间接传递 (简化: D 也用 bootstrap_room 派生 uuid)
            self.uuid = self._derive_uuid(self.bootstrap_room)
            self.received_kv = self.manager.engine.pull(conn, self.uuid, kv_spec)
            # 写到本地 KV pool
            self.manager.args.kv_pool.set_kv_by_indices(
                self.kv_indices, self.received_kv
            )
            # 通知 P pull 完成
            self.manager.engine.notify_pull_done(
                self.uuid,
                self.prefill_info.host,
                self.prefill_info.side_channel_port,
            )
            self.state = KVPoll.Success
        except Exception as e:
            self.state = KVPoll.Failed

    def _build_kv_spec(self) -> list[jax.ShapeDtypeStruct]:
        num_tokens = int(self.kv_indices.shape[0])
        return [
            jax.ShapeDtypeStruct(
                shape=(num_tokens, self.manager.args.kv_head_per_rank,
                       2, self.manager.args.head_dim),
                dtype=self.manager.args.dtype,
                sharding=self.manager.args.kv_sharding,
            )
            for _ in range(self.manager.args.layer_num)
        ]

    def _derive_uuid(self, bootstrap_room: int) -> bytes:
        """从 bootstrap_room 派生 uuid (P/D 必须一致)"""
        return hashlib.sha256(
            bootstrap_room.to_bytes(8, "little") + b"sgl-jax-pd"
        ).digest()[:16]

    def poll(self) -> KVPoll:
        return self.state

    def clear(self) -> None:
        self.received_kv = None

    def abort(self) -> None:
        self.state = KVPoll.Failed
```

> 注：上面的 `_derive_uuid` 是 RFC-2 的简化方案——bootstrap_room 由 router 协调（参见 §6.2 `get_prefill_info`），P 和 D 用同一 bootstrap_room 派生同一 uuid。生产实践（包括 sglang）一般让 D 在 bootstrap 时主动获取 uuid（多一次握手），RFC-2 简化为客户端派生。如未来需要支持多 prefill / 多 receiver 复用同 bootstrap_room，可以改 protocol。

---

## 8. Role Assignment

### 8.1 enums

```python
# python/sgl_jax/srt/disaggregation/utils.py

class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

class TransferBackend(Enum):
    JAX_TRANSFER = "jax_transfer"
    # 未来扩展点: NIXL_TPU = "nixl_tpu" 等

class KVClassType(Enum):
    MANAGER = "manager"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"

def get_kv_class(backend: TransferBackend, class_type: KVClassType):
    """Factory. 与 sglang utils.py:321 类似但精简."""
    if backend == TransferBackend.JAX_TRANSFER:
        from sgl_jax.srt.disaggregation.jax_transfer import conn
        if class_type == KVClassType.MANAGER:
            return conn.JaxTransferKVManager
        elif class_type == KVClassType.SENDER:
            return conn.JaxTransferKVSender
        elif class_type == KVClassType.RECEIVER:
            return conn.JaxTransferKVReceiver
        elif class_type == KVClassType.BOOTSTRAP_SERVER:
            from sgl_jax.srt.disaggregation.bootstrap import BootstrapServer
            return BootstrapServer
    raise ValueError(f"Unknown backend: {backend}")
```

### 8.2 进程级 role 解析（ADR-3）

`server_args.disaggregation_mode` 在启动时解析为 enum：

```python
# scheduler.py
mode_str = self.server_args.disaggregation_mode
self.disaggregation_mode = DisaggregationMode(mode_str)
```

dispatch 事件循环：

```python
# python/sgl_jax/srt/managers/scheduler.py run_scheduler_process
def run_scheduler_process(...):
    scheduler = Scheduler(...)
    if scheduler.disaggregation_mode == DisaggregationMode.NULL:
        if server_args.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
        scheduler.event_loop_normal_disagg_prefill()   # from Mixin
    elif scheduler.disaggregation_mode == DisaggregationMode.DECODE:
        scheduler.event_loop_normal_disagg_decode()    # from Mixin
```

---

## 9. Scheduler 集成（Mixin 模式）

### 9.1 SchedulerDisaggregationPrefillMixin

```python
# python/sgl_jax/srt/disaggregation/prefill.py

@dataclass
class PrefillBootstrapQueue:
    """P 侧: 等待 D 来 receive 的请求队列"""
    reqs: list[Req] = field(default_factory=list)
    # 每个 Req 关联一个 KVSender

class SchedulerDisaggregationPrefillMixin:
    """P 节点的 scheduler mixin. 由 Scheduler 通过多继承获得."""

    def _init_disaggregation_prefill(self):
        """在 Scheduler.__init__ 末尾调用"""
        self.disagg_kv_manager = self._create_kv_manager(DisaggregationMode.PREFILL)
        self.disagg_kv_manager.register_to_bootstrap()
        self.disagg_prefill_inflight: list[Req] = []
        self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue()

    def event_loop_normal_disagg_prefill(self):
        """P 主循环"""
        while True:
            recv_reqs = self.recv_requests()
            for req in recv_reqs:
                self._add_request_to_queue(req)

            # 收到 receiver 来 receive 的请求 (通过 bootstrap_room 匹配)
            self._handle_disagg_prefill_inflight()

            # 调度并跑 prefill
            batch = self.get_next_batch_to_run()
            if batch is not None:
                self.process_prefill_chunk(batch)
                self.send_kv_chunk(batch)   # 完成的 chunk 立即开始 send

    def process_prefill_chunk(self, batch: ScheduleBatch):
        """跑一个 prefill chunk. 与正常 prefill 路径一致."""
        # 调用 run_batch + process_batch_result 等标准路径
        ...

    def send_kv_chunk(self, batch: ScheduleBatch):
        """
        对刚完成 prefill 的 req, 触发 KV send.
        关键: 直接从 req_to_token_pool 读 KV indices, 不走 tree_cache (ADR-7).
        """
        for req in batch.reqs:
            if not req.is_finished_chunk():
                continue
            kv_indices = self.tree_cache.get_kv_indices_for_send(req)
            sender = self.disagg_kv_manager.create_sender(
                bootstrap_addr=None,  # P 侧不需要 addr (D 来 connect)
                bootstrap_room=req.bootstrap_room,
            )
            sender.init(num_kv_indices=len(kv_indices), aux_index=0)
            sender.send(kv_indices)
            self.disagg_prefill_inflight.append((req, sender))

    def _handle_disagg_prefill_inflight(self):
        """检查 inflight reqs 的传输状态. 完成则释放本地 KV pool."""
        still_inflight = []
        for req, sender in self.disagg_prefill_inflight:
            state = sender.poll()
            if state == KVPoll.Success:
                # KV 已被 D pull 走, 可以释放本地 indices
                self.tree_cache.cache_finished_req(req)
                # tree_cache 内部会释放 req 的 token allocator slots
            elif state == KVPoll.Failed:
                self.handle_disagg_failure(req, sender.failure_exception())
            else:
                still_inflight.append((req, sender))
        self.disagg_prefill_inflight = still_inflight
```

### 9.2 SchedulerDisaggregationDecodeMixin

```python
# python/sgl_jax/srt/disaggregation/decode.py

@dataclass
class DecodePreallocQueue:
    """D 侧: 已收到请求但还没接 KV 的队列"""
    reqs: list[Req] = field(default_factory=list)

@dataclass
class DecodeTransferQueue:
    """D 侧: KV 接收中的队列"""
    items: list[tuple[Req, "JaxTransferKVReceiver"]] = field(default_factory=list)

class SchedulerDisaggregationDecodeMixin:
    """D 节点的 scheduler mixin."""

    def _init_disaggregation_decode(self):
        self.disagg_kv_manager = self._create_kv_manager(DisaggregationMode.DECODE)
        self.disagg_decode_prealloc_queue = DecodePreallocQueue()
        self.disagg_decode_transfer_queue = DecodeTransferQueue()

    def event_loop_normal_disagg_decode(self):
        """D 主循环"""
        while True:
            # 收到 client 请求
            recv_reqs = self.recv_requests()
            for req in recv_reqs:
                # 不 tokenize 不 schedule, 直接放 prealloc queue (等待 KV)
                self.disagg_decode_prealloc_queue.reqs.append(req)

            # 处理 prealloc → 启动 KV receive
            self._process_decode_prealloc()

            # 检查 transfer queue → 已完成 KV 接收的 req 入 running batch
            self._process_decode_transfer()

            # 跑 decode batch
            batch = self.get_next_batch_to_run()
            if batch is not None:
                self.process_decode_batch(batch)

    def _process_decode_prealloc(self):
        """从 prealloc queue 取出 req, 分配本地 KV slot, 启动 receive"""
        for req in list(self.disagg_decode_prealloc_queue.reqs):
            # 分配 KV slot
            num_tokens = req.num_prefill_tokens   # P 已 prefill 的 token 数
            kv_indices = self.memory_pools.token_allocator.alloc(num_tokens)
            if kv_indices is None:
                continue   # 无空间, 下轮重试
            req.kv_indices = kv_indices

            # 创建 receiver
            receiver = self.disagg_kv_manager.create_receiver(
                bootstrap_addr=self.server_args.disaggregation_bootstrap_addr,
                bootstrap_room=req.bootstrap_room,
            )
            receiver.init(prefill_dp_rank=req.prefill_dp_rank)
            receiver.send_metadata(kv_indices, aux_index=0)

            self.disagg_decode_transfer_queue.items.append((req, receiver))
            self.disagg_decode_prealloc_queue.reqs.remove(req)

    def _process_decode_transfer(self):
        """检查 transfer 状态, 完成的 req 进 waiting_queue"""
        still_pending = []
        for req, receiver in self.disagg_decode_transfer_queue.items:
            state = receiver.poll()
            if state == KVPoll.Success:
                # KV 已在本地 pool, 可以进入正常 decode 流程
                req.status = ReqStatus.RUNNING
                self.waiting_queue.append(req)
                # 不需要 tree_cache.insert (ADR-7)
            elif state == KVPoll.Failed:
                self.handle_disagg_failure(req, receiver.failure_exception())
            else:
                still_pending.append((req, receiver))
        self.disagg_decode_transfer_queue.items = still_pending
```

### 9.3 Scheduler 主类 wiring

```python
# scheduler.py

class Scheduler(
    # 现有 mixins
    SchedulerMetricsMixin,
    SchedulerOutputProcessorMixin,
    # PD mixins (RFC-2 新增)
    SchedulerDisaggregationPrefillMixin,
    SchedulerDisaggregationDecodeMixin,
):
    def __init__(self, ...):
        ...
        # 在最后初始化 PD
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._init_disaggregation_prefill()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self._init_disaggregation_decode()

    def _create_kv_manager(self, mode: DisaggregationMode) -> KVManager:
        """工厂. 同时 register_runner (因为 Manager.__init__ 完成后需要 runner)."""
        backend = TransferBackend(self.server_args.disaggregation_transfer_backend)
        manager_cls = get_kv_class(backend, KVClassType.MANAGER)

        # 构造 KVArgs (从 self.tp_worker.model_runner 拿 kv_pool 等)
        runner = self.tp_worker.model_runner
        args = KVArgs(
            kv_pool=runner.kv_caches[0],
            page_size=self.page_size,
            layer_num=self.model_config.num_hidden_layers,
            kv_head_per_rank=self.model_config.num_kv_heads // self.mesh.shape["tensor"],
            head_dim=self.model_config.head_dim,
            dtype=runner.kv_caches[0].dtype,
            mesh=self.mesh,
            kv_sharding=runner.kv_caches[0].kv_sharding,
            is_mla=self.model_config.is_mla,
        )
        manager = manager_cls(args, mode, self.server_args)

        # 显式 register_runner (因为 Manager.__init__ 时 runner 已就绪, 但 ABC
        # 设计成两步: 先创建 Manager, 再 register_runner. 这样未来支持 lazy
        # runner injection 时更灵活)
        manager.engine.register_runner(runner)
        return manager
```

---

## 10. P 侧 event loop 详细

### 10.1 时序图

```
[Client/Proxy] ──HTTP req──> [P Scheduler]
                                  │
                                  ├─ recv_requests
                                  ├─ tokenize + schedule
                                  ├─ run_batch (prefill chunk)
                                  ├─ process_batch_result
                                  │      │
                                  │      └─ for each finished chunk:
                                  │           send_kv_chunk()
                                  │              │
                                  │              ├─ tree_cache.get_kv_indices_for_send(req)
                                  │              ├─ create JaxTransferKVSender
                                  │              ├─ sender.init / sender.send (异步)
                                  │              └─ append to disagg_prefill_inflight
                                  │
                                  └─ _handle_disagg_prefill_inflight (每 step 调用)
                                          │
                                          ├─ for each (req, sender):
                                          │     state = sender.poll()
                                          │     if Success:
                                          │         tree_cache.cache_finished_req(req)
                                          │         (释放本地 KV slot)
```

### 10.2 与 chunked prefill 兼容

P 侧的 prefill 仍走标准 chunked prefill 路径。`send_kv_chunk` 在每个 chunk 完成时被调用，每个 chunk 独立发送一份 KV（不等整 req 完）。这与 sglang 的 `process_prefill_chunk` 行为一致。

---

## 11. D 侧 event loop 详细

### 11.1 时序图

```
[Client/Proxy] ──HTTP req with kv_transfer_params──> [D Scheduler]
                                                          │
                                                          ├─ recv_requests
                                                          ├─ 不 schedule, 入 prealloc queue
                                                          │
                                                          ├─ _process_decode_prealloc (每 step):
                                                          │     for each req in queue:
                                                          │         alloc KV slot
                                                          │         create JaxTransferKVReceiver
                                                          │         receiver.init (从 bootstrap 获取 P)
                                                          │         receiver.send_metadata (启动 pull)
                                                          │         入 transfer queue
                                                          │
                                                          ├─ _process_decode_transfer (每 step):
                                                          │     for each (req, receiver):
                                                          │         state = receiver.poll()
                                                          │         if Success:
                                                          │             req.status = RUNNING
                                                          │             加入 waiting_queue
                                                          │             (KV 已在本地 pool, 可直接 decode)
                                                          │
                                                          └─ run_batch (正常 decode)
```

### 11.2 Decode 端的 prefix cache

D 端的 `tree_cache`（ChunkCache 或 UnifiedRadixCache）在 PD 主路径中**不参与 KV 接收**。但在 decode 完成后通过标准 `cache_finished_req` 把整 req 的 KV 注册到 tree（如果是 UnifiedRadixCache）。这意味着：
- 同一 prefix 的多个 D 请求**不能复用** PD 传过来的 KV（因为 tree 不知道）
- 这是已知设计限制；与 sglang 一致

---

## 11.5 PD 端到端工作流详解

本章节回答常见疑问：P 端 prefix cache 与 transfer 的关系、serving 流程、跨进程/跨主机数据路径。

### 11.5.1 P 端 prefix cache 与 transfer 的关系（先后 + 独立）

**先后关系**：tree_cache 命中 → 增量 prefill → transfer，但**两者作用域独立**。

| 阶段 | tree_cache 作用 | transfer 作用 |
|---|---|---|
| 1. P 收到 req | `match_prefix(req.input_ids)` 命中 K tokens（如果用 UnifiedRadixCache）| — |
| 2. P prefill | 只跑剩余 `(prompt_len - K)` tokens，跳过命中部分（节省 P **计算量**）| — |
| 3. P transfer | — | `KVSender.send(req.kv_indices)` 发送 **req 在 P 上的完整 KV**（不管 prefix 是否命中）|
| 4. D pull | — | `KVReceiver.pull(uuid)` 拉取 **完整 KV**（ADR-8: 不支持 partial pull）|

**核心结论**：
- P 的 tree_cache 命中**只优化 P 端计算量**（少跑 K 个 token 的 attention forward）
- transfer 的数据量**与 P 端命中无关**（始终是 req 全长 KV）
- D 端即使本地有 prefix cache 命中也**必须接收完整 KV**（jax.experimental.transfer 限制）

**举例**：req prompt = 1000 tokens, P 端 match_prefix 命中 800 → P 只 prefill 200 tokens → transfer 1000 tokens 的 KV 给 D → D 接收 1000 tokens 全部 KV。

### 11.5.2 Serving 流程（client / proxy / P / D 四方）

**生产 serving 流程**（参考 §16.3 toy_proxy_server）：

```
Client                Proxy                   P                       D
  │                     │                     │                       │
  │── POST completions ─▶│                     │                       │
  │                     │── (1) prefill_req ─▶│                       │
  │                     │   bootstrap_room=X   │                       │
  │                     │   max_tokens=1       │                       │
  │                     │                     │── prefill (一次性) ────│
  │                     │                     │── KVSender.send ─────▶│ (异步 D2H staging + await_pull)
  │                     │                     │                       │
  │                     │◀── first_token, ─────│                       │
  │                     │    KV transfer started                       │
  │                     │                     │                       │
  │                     │── (2) decode_req ──────────────────────────▶│
  │                     │   bootstrap_room=X                            │
  │                     │   prefill_first_token=t0                     │
  │                     │   stream=True                                 │
  │                     │                     │                       │── KVReceiver.pull ◀── (拉 KV)
  │                     │                     │                       │── decode token t1
  │                     │◀──── token t1 (SSE) ───────────────────────│
  │◀── token t1 ─────────│                     │                       │── decode token t2
  │                     │◀──── token t2 (SSE) ───────────────────────│
  │◀── token t2 ─────────│                     │                       │   ...
  │                     │                     │                       │
  │                     │                     │ (与 D pull 同时, P    │
  │                     │                     │  await_pull 阻塞,     │
  │                     │                     │  ZMQ 收到 done 通知后 │
  │                     │                     │  释放本地 KV buffer) │
  │                     │                     │                       │── decode EOS
  │                     │◀──── EOS ──────────────────────────────────│
  │◀── EOS ──────────────│                     │                       │
```

**关键事实**：
- ✓ **D 直接流式回 token 给 proxy 再给 client**——P 不参与后续 token 回传
- ✓ **P 只生成 first_token**（强制 max_tokens=1），用于：(a) 触发 KV 完整生成 (b) 让 D 接力 decode 时有起点
- ✓ **first_token 通过 HTTP 从 P 经 proxy 传给 D**（不是通过 KV transfer 通道）；这是因为 first_token 是 sample 出的结果，不在 KV cache 中
- ✓ **KV transfer 是 P-D 直连**（不经 proxy），proxy 只做 HTTP 路由不接触 KV 数据
- ✗ **D 不回传 token 给 P**——P 收到 first_token 响应后职责就完成了，等 D 通过 ZMQ 通知 pull 完成即可释放 buffer

**为什么这样设计**：proxy 流式转发 D 的输出比走 P 中转少一跳延迟；P 提前释放后可以接下个 req 的 prefill。

### 11.5.3 跨进程 / 跨主机数据路径（D→H→H→D）

`jax.experimental.transfer.start_transfer_server` 的传输数据路径**永远是 4 段**（device → pinned host → network → pinned host → device），不论是同主机跨进程还是跨主机：

```
┌──────────────────────────────────────────────────────────────────┐
│ P process                                                        │
│  ┌──────────┐  ┌──────────────┐                                  │
│  │ TPU HBM   │──┤ Pinned Host  │──┐                              │
│  │ (P chip)  │  │ DRAM (P host)│  │                              │
│  └──────────┘  └──────────────┘  │ TCP socket                    │
│                                  │ (jax.experimental.transfer    │
│                                  │  内部网络栈)                   │
│                                  ▼                              │
├──────────────────────── 跨进程边界 ─────────────────────────────┤
│                                  │                              │
│ D process                        │                              │
│                                  ▼                              │
│  ┌──────────┐  ┌──────────────┐                                  │
│  │ TPU HBM   │◀─┤ Pinned Host  │                                 │
│  │ (D chip)  │  │ DRAM (D host)│                                 │
│  └──────────┘  └──────────────┘                                  │
└──────────────────────────────────────────────────────────────────┘
                ↑
                同主机时: 网络走 localhost (loopback), 物理上不出 NIC
                跨主机时: 网络走 DCN (数据中心以太网)
```

**为什么必须经 host memory**：

| 原因 | 说明 |
|---|---|
| TPU 没有 GPUDirect RDMA 对应物 | 网络接口（NIC）挂在 host CPU 上，TPU 不能直接 push 数据到 NIC，必须先 D2H |
| 跨 JAX process 不共享 device memory | 同主机两个 JAX process 是两个独立 PJRT runtime，HBM allocations 互不可见 |
| `jax.experimental.transfer` 没有 IPC 优化路径 | API 底层基于 PJRT 的 transfer manager，统一走网络抽象层 |

**单主机跨进程 vs 跨主机的差异**（仅在「网络」段）：

| | 单主机跨进程 | 跨主机 |
|---|---|---|
| Device → Host | Pallas DMA 或 JAX device_put | 同左 |
| Host → Host | **localhost TCP (loopback)** — 内核内存拷贝, ~1-2us 延迟 | **DCN TCP** — 经 NIC + 物理网络, ms 级延迟 |
| Host → Device | `jax.device_put` 或 Pallas DMA | 同左 |
| 带宽瓶颈 | host memory bandwidth (~100 GB/s) | DCN 带宽 (~10-100 Gbps = 1.2-12 GB/s) |

**对 RFC-2 设计的影响**：
- ADR-5 启用 D2H staging 在两种场景都有价值（多 host 更受益, 因为网络段慢）
- §4.3 path A（启用 staging）= D2H 由 sgl-jax 显式控制 + 用预分配 buffer 减少 syscall；path B（不启用）= D2H 由 `jax.experimental.transfer` 内部处理。两条路径数据流相同，只是「谁负责 D2H」不同
- D2D 直传（如 GPU 的 NVLink P2P）在 TPU PD 场景下**不可达**——已知硬约束

### 11.5.4 一个完整请求的时序（端到端 latency 估算）

以多 host 部署、prompt=1024 tokens、生成 256 tokens 为例：

| 阶段 | 操作 | 估算延迟 |
|---|---|---|
| 1 | Client → Proxy: HTTP POST | <1ms |
| 2 | Proxy → P: HTTP POST | <1ms |
| 3 | P: prefill 1024 tokens (TPU v6e-8) | 50-200ms (取决于模型) |
| 4 | P: sample first_token + D2H staging | 5-10ms |
| 5 | P → Proxy: HTTP response (first_token) | <1ms |
| 6 | Proxy → D: HTTP POST stream | <1ms |
| 7 | D → P: jax.experimental.transfer pull (DCN) | 10-50ms (取决于 KV 大小 + DCN 带宽) |
| 8 | D: pull 完成 + ZMQ done → P 释放 buffer | <5ms |
| 9 | D: decode 256 tokens (TPU v6e-8) | 1000-3000ms (取决于模型 + batch) |
| 10 | D → Proxy → Client: streaming tokens | 各 step <2ms |

**首 token 延迟 (TTFT)**: 主要由阶段 3 决定，PD 引入的 overhead 约 20-60ms（阶段 4+5+6+7+8）。
**逐 token 延迟 (TPOT)**: 主要由 D 端 decode 决定，与单进程基本一致。
**PD 收益**: P 可以并发处理新 prefill（不被 decode 卡住）；D 可以保持高 decode batch（不被 prefill 中断）。整体吞吐通常比单进程提升 20-50%。

---

## 12. D 节点 cache：第一版仅 ChunkCache

### 12.1 第一版策略（ADR-9）

**第一版 D 节点只实现 ChunkCache 路径**。`kv_cache_builder` 在 `disaggregation_mode == "decode"` 时**自动覆盖** `disable_radix_cache` 配置，无论用户传什么值。理由见 ADR-9。**第一版限制，未来如有需求按 sglang 模式加 opt-in flag**（参见 ADR-9 未来扩展点）。

```python
# kv_cache_builder.build_kv_cache() D 路径 (RFC-0 §5 增强)
if server_args.disaggregation_mode == "decode":
    # 第一版: 强制走 ChunkCache, 与用户配置无关
    if params.is_hybrid_swa:
        tree_cache = SWAChunkCache(memory_pools=memory_pools, ...)
    else:
        tree_cache = ChunkCache(memory_pools=memory_pools, ...)
    if not server_args.disable_radix_cache:
        logger.warning(
            "D + RadixCache not implemented in first release; "
            "forcing ChunkCache. See RFC-2 ADR-9 for future extension."
        )
    if server_args.enable_hierarchical_cache:
        logger.warning(
            "enable_hierarchical_cache is ignored on decode node "
            "(D uses ChunkCache which has no HiCache hooks). "
            "HiCache only takes effect on prefill nodes."
        )
elif server_args.disable_radix_cache:
    # P 节点或 null 模式: 走原有 ChunkCache 路径
    ...
else:
    # P 节点或 null 模式: UnifiedRadixCache (可选 + HiCache)
    ...
```

### 12.2 实现路径（ChunkCache + PD）

`ChunkCache` 继承 `BasePrefixCache.get_kv_indices_for_send`（RFC-0 §12.2 默认实现）：

```python
# RFC-0 §12.2 中的默认实现, ChunkCache 继承不需 override
def get_kv_indices_for_send(self, req) -> jax.Array:
    return self.req_to_token_pool.req_to_token[
        req.req_pool_idx, :req.computed_tokens
    ]
```

D 端收到 KV 后写入 `kv_pool`（标准 write 路径），不调 `tree_cache.insert`（ChunkCache 也没有 insert 概念）。

### 12.3 验证

`test_pd_with_chunkcache.py`：
- 启动 1 P (UnifiedRadixCache 可选 HiCache) + 1 D (ChunkCache，第一版唯一选项)
- 发送 100 req，验证：
  - 每 req 的 prefill 在 P 跑完
  - KV 被 send 到 D
  - D 上完成 decode
  - 输出与单进程 baseline 等价（KL test）
- 反向测试：D 端配 `disable_radix_cache=False` 时，builder 自动覆盖为 ChunkCache + warning 日志（提示"第一版未实现 D + RadixCache"）

---

## 13. P 节点 cache 选择（含 HiCache）

### 13.1 P 节点可选 UnifiedRadixCache 或 ChunkCache

P 节点（`disaggregation_mode == "prefill"`）的 cache 选择遵循 RFC-0 §5 标准 dispatch，**没有 PD-specific 限制**：

- 默认走 `UnifiedRadixCache`（受益于 prefix 复用，多轮对话场景）
- 配 `disable_radix_cache=True` 时走 `ChunkCache`（无 prefix 复用）
- 配 `enable_hierarchical_cache=True` 时叠加 HiCache（仅 UnifiedRadixCache 路径生效）

### 13.2 P 端 HiCache 的价值

P 端多个请求共享 system prompt prefix 时，HiCache 显著降低 prefill 计算量：
- 第一次 prefill `system prompt + Q1` → HiCache 把 `system prompt` 写到 host pool
- 后续 prefill `system prompt + Q2/Q3/...` → match_prefix 命中，从 host pool load_back，节省 prefill

### 13.3 P 端 UnifiedRadixCache + PD 的实现细节

P 端调用 `KVSender.send` 时使用 `tree_cache.get_kv_indices_for_send`：

```python
class UnifiedRadixCache(BasePrefixCache):
    def get_kv_indices_for_send(self, req) -> jax.Array:
        """
        默认行为: 发送完整 prefill KV (与 ChunkCache 一致).
        未来优化点 (受 ADR-8 限制暂未启用): 跳过已在 D 端 prefix tree 内的 tokens.
        """
        return super().get_kv_indices_for_send(req)
```

### 13.4 验证

`test_pd_with_unified.py`：
- 启动 1 P (UnifiedRadixCache + HiCache) + 1 D (ChunkCache 强制)
- 发送多轮对话场景（多 req 共享 system prompt prefix）
- 验证：
  - 输出与单进程 baseline 等价
  - **P 端**第 2 轮起 prefix cache 命中率 > 0（共享 system prompt）
  - **P 端** HiCache 启用时，host pool 命中率随时间上升
  - **D 端**不维护 prefix tree（ChunkCache 无 tree）；每 req decode 独立
  - PD 传输次数 = 总 req 数（无 partial pull 优化）

### 13.5 PD + HiCache 同启时的行为约定

| 组件 | P 节点 | D 节点 |
|---|---|---|
| tree_cache | UnifiedRadixCache（可选 ChunkCache） | **第一版仅 ChunkCache** |
| HiCache | ✓ 可启用（host pool L2 + 可选 L3） | ✗ 静默忽略 + warning（ChunkCache 无 hook） |
| host memory 用途 | HiCache host pool + 可选 D2H staging（如 P 同时收发，理论不会） | 仅 D2H staging（参见 §5 QueueHostKVPool） |
| HBM 用途 | 大（容纳 prefill 大 batch + KV pool + HiCache D2H 暂留 buffer） | 大（容纳 decode 多 req KV pool） |

**资源切分细节** 留待 RFC-1/RFC-2 联合实施时 benchmark 决定。

**未来扩展**：若实现 D + RadixCache（ADR-9 未来扩展点），则 D 节点可叠加 HiCache，与 P 节点对称。

---

## 14. D2H staging buffer 策略

详见 §5 QueueHostKVPool 设计。

### 14.1 何时启用

| 部署模式 | 默认 | 原因 |
|---|---|---|
| 单机多进程（同 host） | **启用** | P 释放 HBM 收益大 |
| 多主机 | **关闭** | 每进程数据量小，staging 收益小 |
| 用户覆盖 | `--disaggregation-enable-d2h {true,false}` | 显式控制 |

### 14.2 容量计算

```python
def compute_d2h_pool_params(server_args, model_config):
    # pool_size = 最大并发 P 请求 (受 transfer server 并发度限制)
    pool_size = server_args.disaggregation_d2h_pool_size or 64

    # max_tokens_per_buffer = 单 req 最大 KV 长度
    max_tokens_per_buffer = (
        server_args.disaggregation_d2h_max_tokens_per_buffer or
        model_config.max_context_len
    )
    return pool_size, max_tokens_per_buffer
```

### 14.3 失败处理

如果 `staging_pool.alloc()` 返回 None（pool 满），P 端 sender 阻塞等待（或快速失败 + 重新调度）。RFC-2 默认快速失败（`Sender.poll()` 返回 `Failed`），scheduler 把 req 退回 waiting_queue 重试。

---

## 15. 通知通道（ZMQ side channel）

详见 §4.2。

### 15.1 协议字段

ZMQ DEALER → ROUTER 消息体（msgpack）：

```python
{
    "uuid": bytes,             # 16 bytes
    "status": str,             # "done" / "failed"
    "error": Optional[str],
}
```

### 15.2 复用 sgl-jax 现有 ZMQ 基础设施

sgl-jax 现有 ZMQ 用于 tokenizer/detokenizer/scheduler/sync 通信，端口在 `server_args` 中已分配。PD 侧通道新增独立端口 `disaggregation_side_channel_port`（默认 9600），不冲突。

---

## 16. 部署

### 16.1 多 host TP 部署（生产主推）

典型生产部署：**P-cluster + D-cluster 各占独立 host(s)，每 cluster 内部跑 TP**。例如 P=1 host (8 chip TP=8), D=1 host (8 chip TP=8), 通过 DCN 网络互连。如果 P 慢于 D，可以加 P-host（N 个 P + M 个 D，bootstrap server round-robin 路由）。

```bash
# examples/disagg/run_multi_host.sh (生产示例)

# === 假设 3 台 host: bootstrap host + P host + D host ===
# bootstrap.example.com:8998
# prefill.example.com:7000  (TPU v6e-8, TP=8)
# decode.example.com:7001   (TPU v6e-8, TP=8)

# 1. [bootstrap host] 启动 bootstrap server (独立小机器即可, 无 TPU)
ssh bootstrap.example.com '
python -m sgl_jax.disaggregation.bootstrap \
    --host 0.0.0.0 \
    --port 8998
' &

# 2. [P host] 启动 prefill server (占满本 host TPU)
ssh prefill.example.com '
python -m sgl_jax.launch_server \
    --model-path $MODEL_PATH \
    --tensor-parallel-size 8 \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-addr bootstrap.example.com:8998 \
    --disaggregation-kv-transfer-port 9100 \
    --disaggregation-side-channel-port 9600 \
    --host 0.0.0.0 --port 7000
' &

# 3. [D host] 启动 decode server (占满本 host TPU)
ssh decode.example.com '
python -m sgl_jax.launch_server \
    --model-path $MODEL_PATH \
    --tensor-parallel-size 8 \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-addr bootstrap.example.com:8998 \
    --disaggregation-kv-transfer-port 9200 \
    --disaggregation-side-channel-port 9700 \
    --host 0.0.0.0 --port 7001
' &

# 4. [任意 host, 通常与 bootstrap 同机] 启动 proxy (路由 client 请求)
ssh bootstrap.example.com '
python examples/disagg/toy_proxy_server.py \
    --prefill-endpoints http://prefill.example.com:7000 \
    --decode-endpoints http://decode.example.com:7001 \
    --port 8000
' &
```

**N P + M D 扩展**：proxy 支持 `--prefill-endpoints` 多 endpoint（逗号分隔），D 同理；bootstrap server 自动 round-robin 路由 bootstrap_room 到不同 P。

**网络要求**：
- P-host 到 D-host 的 DCN 带宽要够：典型 v6e-8 模型 KV 传输每 req 几十 MB - 几百 MB，吞吐 100 req/s 需要 10-100 Gbps DCN
- bootstrap server 流量很小，普通 HTTP 即可
- ZMQ side channel 流量也极小（每 req 几十 bytes done 通知）

### 16.2 K8s / GKE 部署（生产推荐方式）

实际生产建议用 K8s 部署。RFC-2 不写完整 manifest（属运维范围），但要点：
- Bootstrap server: Deployment + Service（ClusterIP）
- P server: StatefulSet（绑定 TPU 节点）+ Headless Service
- D server: StatefulSet + Headless Service
- Proxy: Deployment + Service（LoadBalancer 暴露公网）
- ConfigMap 管理 bootstrap_addr 等共享配置
- 参考现有 `jx-v6e-4.yaml` 风格

### 16.3 toy_proxy_server.py

```python
# examples/disagg/toy_proxy_server.py
# 注: 生产用 nginx / envoy 等成熟 router; 本文件仅作 reference + 开发测试用.

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import random
import itertools

app = FastAPI()
# 支持多 P / 多 D, round-robin 路由
PREFILL_ENDPOINTS = ["http://prefill-0.example.com:7000",
                     "http://prefill-1.example.com:7000"]
DECODE_ENDPOINTS = ["http://decode-0.example.com:7001",
                    "http://decode-1.example.com:7001"]
_prefill_iter = itertools.cycle(PREFILL_ENDPOINTS)
_decode_iter = itertools.cycle(DECODE_ENDPOINTS)

def generate_bootstrap_room() -> int:
    """64-bit 唯一 ID. 见 RFC-2 §7.2 _derive_uuid 宽度约束."""
    return random.getrandbits(64)

@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    bootstrap_room = generate_bootstrap_room()
    p_endpoint = next(_prefill_iter)
    d_endpoint = next(_decode_iter)

    # Step 1: 发给 P, max_tokens=1, 注入 bootstrap_room
    #         P prefill 完成后返回 first token + 已开始 await_pull
    p_body = {**body, "max_tokens": 1, "stream": False,
              "bootstrap_room": bootstrap_room}
    async with httpx.AsyncClient(timeout=60.0) as client:
        p_resp = await client.post(f"{p_endpoint}/v1/completions", json=p_body)
    p_data = p_resp.json()
    first_token = p_data["choices"][0]["text"]   # P 的第一个 token

    # Step 2: 发给 D, 带 bootstrap_room 流式 decode 剩余 token
    #         D 用 bootstrap_room 找 P, pull KV, 用 first_token 接着 decode
    d_body = {**body, "bootstrap_room": bootstrap_room,
              "prefill_first_token": first_token}
    # 流式回传给 client
    async def stream_d():
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream("POST", f"{d_endpoint}/v1/completions",
                                     json=d_body) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
    return StreamingResponse(stream_d(), media_type="text/event-stream")
```

**bootstrap_room 流通路径**（明确化）：
1. **Proxy** 生成 64-bit `bootstrap_room`，注入到 P 和 D 的 HTTP 请求 body
2. **P 端** scheduler 从 HTTP body 解析出 `req.bootstrap_room`（在 tokenizer_manager 或 scheduler.recv_requests 阶段），由 `JaxTransferKVSender.__init__(bootstrap_room=req.bootstrap_room)` 持有
3. **D 端** scheduler 同样从 HTTP body 解析，由 `JaxTransferKVReceiver.__init__(bootstrap_room=req.bootstrap_room)` 持有
4. **P 和 D** 都用同一 `bootstrap_room` 通过 `_derive_uuid` 派生同一 16-byte uuid
5. **D** 用 `bootstrap_room` 查 `BootstrapServer.get_prefill_info` 获取 P 路由信息

⚠️ Req 模型需要新增 `bootstrap_room: int = 0` 字段；tokenizer_manager 解析 HTTP body 时填充。

### 16.4 单 host 多进程部署（仅开发/测试用）

> ⚠️ **不推荐用于生产 serving**：单 host KV pool 切两半，每边能装的请求量很小，吞吐不会比单进程 baseline 高。本节仅作开发/调试用。

```bash
# examples/disagg/run_single_host_dev.sh
# 一台 v6e-8: 4 chip P + 4 chip D, 同 host 内 P-D 通信走 localhost

python -m sgl_jax.disaggregation.bootstrap --host 127.0.0.1 --port 8998 &

TPU_VISIBLE_CHIPS=0,1,2,3 python -m sgl_jax.launch_server \
    --model-path $MODEL_PATH --tensor-parallel-size 4 \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-addr 127.0.0.1:8998 \
    --disaggregation-kv-transfer-port 9100 \
    --disaggregation-side-channel-port 9600 \
    --port 7000 &

TPU_VISIBLE_CHIPS=4,5,6,7 python -m sgl_jax.launch_server \
    --model-path $MODEL_PATH --tensor-parallel-size 4 \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-addr 127.0.0.1:8998 \
    --disaggregation-kv-transfer-port 9200 \
    --disaggregation-side-channel-port 9700 \
    --port 7001 &

python examples/disagg/toy_proxy_server.py \
    --prefill-endpoints http://127.0.0.1:7000 \
    --decode-endpoints http://127.0.0.1:7001 --port 8000 &
```

**与多 host 部署的差异**：仅 `bootstrap_addr` / `kv_transfer_port` 改为 localhost；其余完全相同（同一份代码）。`jax.experimental.transfer` 自动走 localhost TCP（loopback），数据路径仍是 D→H→网络→H→D（不能跨进程共享 device memory，参见 §X 数据路径分析）。

---

## 17. 配置项扩展

| 配置项 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `disaggregation_mode` | str | "null" | "null" / "prefill" / "decode" |
| `disaggregation_transfer_backend` | str | "jax_transfer" | RFC-2 唯一支持 |
| `disaggregation_bootstrap_addr` | str | None | "host:port"，所有 P/D 都连这个 |
| `disaggregation_bootstrap_port` | int | 8998 | 仅 bootstrap server 自己用 |
| `disaggregation_kv_transfer_port` | int | 9100 | 本进程 KV transfer server 端口 |
| `disaggregation_side_channel_port` | int | 9600 | 本进程 ZMQ pull-done 端口 |
| `disaggregation_channel_number` | int | 8 | jax.experimental.transfer 并发通道数 |
| `disaggregation_transfer_size_bytes` | int | 268435456 (256MB) | jax.experimental.transfer chunk size |
| `disaggregation_max_parallel_copies` | int | 8 | server 最大并发拷贝 |
| `disaggregation_enable_d2h` | bool | None (auto: 单机=True, 多主机=False) | 是否启用 D2H staging |
| `disaggregation_d2h_pool_size` | int | 64 | QueueHostKVPool buffer 数 |
| `disaggregation_d2h_max_tokens_per_buffer` | int | None (= max_context_len) | 单 buffer 容量 |
| `disaggregation_pull_timeout_seconds` | float | 180.0 | P 端 await_pull 超时 |
| `disaggregation_send_threads` | int | 4 | P 侧 send_executor 线程数 (§7.2 manager 用) |
| `disaggregation_recv_threads` | int | 4 | D 侧 recv_executor 线程数 |

### 17.1 兼容性检查 + D 节点第一版限制

```python
def _validate_disagg_args(self):
    if self.disaggregation_mode == "null":
        return
    assert self.disaggregation_bootstrap_addr is not None, \
        "PD mode requires --disaggregation-bootstrap-addr"
    assert self.disaggregation_transfer_backend == "jax_transfer", \
        f"RFC-2 only supports jax_transfer backend"

    # ===== D 节点第一版仅支持 ChunkCache (ADR-9) =====
    if self.disaggregation_mode == "decode":
        if not self.disable_radix_cache:
            logger.warning(
                "Decode node currently only supports ChunkCache in this release. "
                "--disable-radix-cache will be force-set to True. "
                "If you need D + RadixCache, please open an issue (see RFC-2 ADR-9 future extension)."
            )
            self.disable_radix_cache = True   # 第一版静默覆盖
        if self.enable_hierarchical_cache:
            logger.warning(
                "enable_hierarchical_cache is ignored on decode node "
                "(ChunkCache has no HiCache hooks). HiCache only takes effect on prefill nodes."
            )
            self.enable_hierarchical_cache = False   # 静默覆盖

    # ===== P 节点 / null 节点保持灵活 =====
    # P 节点 cache 选择走 RFC-0 §5 标准 dispatch (UnifiedRadixCache / ChunkCache / +HiCache 可选)
    # PD 与 chunked prefill 兼容; PD 与 overlap schedule 兼容
```

**为什么用 warning + 静默覆盖而不是 raise**：避免用户在多次启动 P/D 时反复修改 CLI（同一份脚本 P/D 共用配置时友好）；同时 warning 中明确「currently only supports」+「if you need ... open an issue」给出未来扩展信号，不让用户误以为是永久限制。

---

## 18. 测试策略

### 18.1 单元测试

| 测试文件 | 覆盖 |
|---|---|
| `test_jax_transfer_engine.py` | start/await_pull/connect/pull/notify_pull_done；单进程内 mock 双 server 测路由 |
| `test_queue_host_kv_pool.py` | alloc/free/get_buffer/put_buffer；并发借用 |
| `test_bootstrap.py` | register_prefill / get_prefill_info / round-robin 路由 |
| `test_zmq_notifier.py` | DEALER→ROUTER 消息正确传递；超时 / 重连 |
| `test_kv_manager.py` | Manager/Sender/Receiver 4-tuple ABC 行为；状态机转换 (KVPoll) |

### 18.2 集成测试

| 测试文件 | 内容 |
|---|---|
| `test_pd_e2e_single_host.py` | 1 P + 1 D + 1 bootstrap + 1 proxy 全链路；100 req KL 等价 |
| `test_pd_with_chunkcache.py` | ChunkCache + PD（验证正交矩阵） |
| `test_pd_with_unified.py` | UnifiedRadixCache + PD |
| `test_pd_with_hicache.py` | UnifiedRadixCache + HiCache + PD（验证三者共存） |
| `test_pd_failure_modes.py` | P 崩溃 / D 崩溃 / network drop 的恢复行为 |
| `test_pd_d2h_staging.py` | 启用 D2H staging vs 不启用，输出一致性 |

### 18.3 端到端推理测试

| 场景 | 期望 |
|---|---|
| Llama + PD（启用） | KL 等价 + P/D 吞吐独立可调 |
| MLA (DeepSeek) + PD | KL 等价 |
| 长 context (8K+) + PD | 不退化 |
| 多轮对话 + PD + P 端 UnifiedRadixCache + HiCache | **P 端**多轮共享 prefix 命中率高（D 端 ChunkCache 无 tree） |

### 18.4 性能测试

| 指标 | 期望 |
|---|---|
| 单机模式 PD vs single process 吞吐 | PD ≥ single * 1.2（更好的资源利用） |
| TTFT 延迟（P→D 传输开销） | < 50ms（单机 100token） |
| D2H staging 开启 vs 关闭（单机） | 启用更优（HBM 占用降低） |

---

## 19. 风险与未决问题

### 19.1 已识别风险

| Risk | 影响 | 缓解 |
|---|---|---|
| **R1: `jax.experimental.transfer` 实验性 API 变更** | API 签名 / 行为变化导致 RFC-2 失效 | (1) ABC 隔离；(2) 锁 JAX 版本；(3) 准备 fallback (ZMQ + device_get/put) |
| **R2: bootstrap server 单点故障** | bootstrap 挂了 → 新 D 无法 join | 文档化为「短期接受」；后续可加 HA（多实例 + 一致性 hash） |
| **R3: D2H staging pool 容量评估不准** | OOM 或 underutilize | 加 metric + 文档说明；运行时自动调整（后续 PR） |
| **R4: PD + UnifiedRadixCache 的 D 端 tree 一致性** | D 端把 P 传的 KV 不注册到 tree，可能漏命中 | ADR-8 已说明这是已知限制；后续优化加 tree 注册（但需 partial pull 支持） |
| **R5: ZMQ pull-done 通知丢失** | P 端 buffer 永远不释放 | 加超时（disaggregation_pull_timeout_seconds）+ 主动释放 |
| **R6: bootstrap_room 冲突** | 同 room 多 req | proxy 必须保证全局唯一（uuid4 即可） |
| **R7: 多主机 KV transfer 性能** | DCN 带宽限制吞吐 | benchmark 后调整 channel_number / transfer_size |

### 19.2 未决问题

| 问题 | 决策时机 |
|---|---|
| 是否需要 P-D 双向流式（D 主动 push 给 P 用于 KV cache writeback）？ | 后续 PR，目前单向足够 |
| PP（Pipeline Parallel）+ PD 如何组合？ | sgl-jax 暂未支持 PP，搁置 |
| Speculative decoding + PD？ | sgl-jax 暂未支持 spec dec，搁置 |
| SWA / Mamba 模型 + PD？ | RFC-2 仅支持 standard MHA + MLA；SWA / Mamba 后续扩展（state_type 字段已预留） |
| bootstrap server 与 proxy 是否合并为一个进程？ | 部署便利，但 RFC-2 内分离更清晰；用户可自行合并 |

---

## 20. 实施路线

### 20.1 阶段（D0 → D3）

```
┌─────────────────────────────────────────────────────────────────┐
│ D0: 基础设施 + JaxTransferKVEngine + BootstrapServer              │
├─────────────────────────────────────────────────────────────────┤
│ ✓ JaxTransferKVEngine 实现 (start/await_pull/connect/pull)        │
│ ✓ ZmqPullNotifier 实现                                            │
│ ✓ BootstrapServer (FastAPI) + BootstrapClient                    │
│ ✓ QueueHostKVPool 实现                                           │
│ ✓ 单元测试 test_jax_transfer_engine.py / test_bootstrap.py /      │
│   test_queue_host_kv_pool.py / test_zmq_notifier.py               │
│                                                                  │
│ 验收: 单元测试 100% 通过 (无需 model)                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ D1: 4-tuple ABC + Scheduler Mixin                                │
├─────────────────────────────────────────────────────────────────┤
│ ✓ KVManager / KVSender / KVReceiver ABC + JaxTransfer 实现        │
│ ✓ SchedulerDisaggregationPrefillMixin                            │
│ ✓ SchedulerDisaggregationDecodeMixin                             │
│ ✓ scheduler.py dispatch 集成                                      │
│ ✓ 单元测试 test_kv_manager.py                                     │
│                                                                  │
│ 验收: ChunkCache + PD 端到端 (test_pd_with_chunkcache.py KL 等价) │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ D2: 部署 + Proxy + 端到端验证                                     │
├─────────────────────────────────────────────────────────────────┤
│ ✓ examples/disagg/run_single_host.sh                             │
│ ✓ examples/disagg/toy_proxy_server.py                            │
│ ✓ UnifiedRadixCache + PD 端到端 (test_pd_with_unified.py)        │
│ ✓ Llama / MLA 模型 PD KL 等价                                    │
│ ✓ 性能 benchmark                                                  │
│                                                                  │
│ 验收: 单机部署可用, 性能不退化                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ D3: 与 HiCache 兼容性 + 多主机文档                                │
├─────────────────────────────────────────────────────────────────┤
│ ✓ test_pd_with_hicache.py (PD + UnifiedRadixCache + HiCache 三者) │
│ ✓ 多主机部署文档 (Ray / K8s 启动方式)                            │
│ ✓ 失败模式测试 test_pd_failure_modes.py                          │
│                                                                  │
│ 验收: 所有正交组合 (PD + cache_type + HiCache) 端到端通过         │
└─────────────────────────────────────────────────────────────────┘
```

### 20.2 D0/D1 可在 RFC-0 M0 完成后并行起步

D0 不依赖 UnifiedRadixCache，可与 RFC-0 M0 / RFC-1 H0 并行开发。

---

## 21. 附录

### 21.1 sglang 参考源码

| 主题 | sglang 文件 |
|---|---|
| 4-tuple ABC | `python/sglang/srt/disaggregation/base/conn.py` |
| Common* 中间层 | `python/sglang/srt/disaggregation/common/conn.py` |
| Mixin 模式 | `python/sglang/srt/disaggregation/prefill.py` (Prefill Mixin) / `decode.py` (Decode Mixin) |
| utils enums | `python/sglang/srt/disaggregation/utils.py` |
| bootstrap 协议 | `python/sglang/srt/disaggregation/common/conn.py:709` `CommonKVBootstrapServer` |

### 21.2 tpu-inference 参考

| 主题 | tpu-inference 文件 |
|---|---|
| JaxTransferServer 使用 | `tpu_inference/distributed/tpu_connector.py:573` `start_transfer_server` |
| HostKVPool Queue 模式 | `tpu_inference/distributed/host_kv_pool.py` |
| ZMQ pull-notify | `tpu_inference/distributed/tpu_connector.py:551` `_pull_notify_listener` |
| Proxy server | `tpu_inference/examples/disagg/toy_proxy_server.py` |
| 部署脚本 | `tpu_inference/examples/disagg/run_disagg_single_host.sh` |

### 21.3 jax-api 调研引用

| RFC-2 章节 | jax-api 调研对应 |
|---|---|
| ADR-1 (jax.experimental.transfer) | §4 |
| ADR-5 (D2H staging) | §2.2.2 / §6.3 (路径 A vs B) |
| ADR-8 (无 partial pull) | §6.1 |
| §4.1 server 单例 | §4.2 |
| §4.3 producer_handoff | §2.4 |

### 21.4 相关 RFC

- RFC-0: UnifiedRadixCache + KV 缓存与传输基础设施
- RFC-1: HiCache

---

**End of RFC-2**
