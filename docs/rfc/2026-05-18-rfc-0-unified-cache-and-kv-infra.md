# RFC-0: UnifiedRadixCache + KV 缓存与传输基础设施

**Status**: Draft  
**Author**: john  
**Date**: 2026-05-18  
**Related**: RFC-1 (HiCache), RFC-2 (PD 分离)  
**Prerequisite reading**: `docs/research/2026-05-18-sglang-cache-pd-organization.md`, `docs/research/2026-05-18-tpu-inference-jax-api-survey.md`

---

## 1. 摘要 & 动机

### 1.1 目标

sgl-jax 即将实现两个独立特性 —— **HiCache（多级 KV 缓存）** 和 **PD 分离（prefill-decode disaggregation）**。本 RFC 不实现这两个特性，而是为它们建立**共同基础设施**：

1. Port sglang `UnifiedRadixCache`（基于 origin/main 1960 行实现）到 sgl-jax，作为统一前缀缓存
2. 引入 `kv_cache_builder` 模式，替换 scheduler 中可能膨胀的 if-elif 链
3. 定义两个独立 ABC：`HiCacheStorage`（被 RFC-1 用）和 `KVTransferEngine`（被 RFC-2 用）
4. 定义 L2 基础设施 ABC：`HostKVPool` / `KVCacheController` / `HostMemoryAllocator`
5. 扩展 `MemoryPools` 容器（pytree）为 host pool 留位置
6. 完整化 `BasePrefixCache` 的 HiCache hooks 签名
7. 提供 RadixCache / SWARadixCache 删除清单 + 测试迁移路径

### 1.2 非目标

- ❌ 不实现 HiCache 的 D↔H 数据搬运细节（属 RFC-1）
- ❌ 不实现 L3 storage backend（属 RFC-1）
- ❌ 不实现 PD 的 KV transfer / bootstrap server（属 RFC-2）
- ❌ 不实现 layer-wise overlap（TPU/XLA 架构不支持，参见 jax-api 调研 §6.2）
- ❌ 不引入 mooncake / NIXL / aibrix / hf3fs / eic / simm 等 GPU 专用 storage backend（参见 hicache-pd-rfc-design-principles memory）
- ❌ 不为可能的 GPU 支持做硬件抽象层（参见 ADR-2）

### 1.3 关键设计 insight

**tree_cache × PD × HiCache 是正交三维**（接口层），本 RFC 与 RFC-1/2 的边界基于这个矩阵：

| tree_cache | + PD | + HiCache | 备注 |
|---|---|---|---|
| **ChunkCache** | ✓ | ✗ | 保留；无 prefix 复用，HiCache 无 hook 点；适合「PD 单纯传 KV、不要 prefix」场景 |
| **UnifiedRadixCache** | ✓ | ✓ | 默认；PD 与 HiCache 互不感知，可独立或同时启用 |
| ~~RadixCache~~ | — | — | **删除**（功能被 UnifiedRadixCache + FullComponent 取代） |
| ~~SWARadixCache~~ | — | — | **删除**（功能被 UnifiedRadixCache + SWAComponent 取代） |

**重要：RFC-2 第一版对 D 节点叠加策略限制**（不是接口层限制，未来可扩展）：

| PD 节点 | tree_cache 选择 | HiCache |
|---|---|---|
| P 节点 (`disaggregation_mode=prefill`) | 自由（ChunkCache 或 UnifiedRadixCache，默认 UnifiedRadixCache） | 可选（仅 UnifiedRadixCache 路径生效） |
| D 节点 (`disaggregation_mode=decode`) | **第一版仅 ChunkCache**（RFC-2 ADR-9，未来可加 opt-in 与 sglang 对齐） | 静默忽略（ChunkCache 无 hook） |
| null 节点（无 PD） | 自由 | 可选 |

理由参见 RFC-2 ADR-9：D 端 prefix 复用受 ADR-8 (无 partial pull) 限制收益极低；第一版直接不实现 D + RadixCache 路径以减少配置矩阵。**接口层仍然支持** UnifiedRadixCache 在 D 端工作（builder 不会 raise），但第一版会自动覆盖配置 + 输出 warning。未来按需求加 sglang 风格 opt-in flag。

PD 与 HiCache 协议层抽象本质不同（PD = P2P-routed，HiCache = content-addressed），所以对应**两个独立 ABC**（`KVTransferEngine` vs `HiCacheStorage`）。底层传输引擎可在 backend 实现里共享，与 ABC 设计无关。

详细背景：见 `docs/research/2026-05-18-sglang-cache-pd-organization.md` §3.6。

---

## 2. 决策记录（ADR）

### ADR-1: UnifiedRadixCache 默认 + 渐进删除 RadixCache/SWARadixCache

| | |
|---|---|
| **决策** | sgl-jax 默认走 `UnifiedRadixCache`，但 **M1/M2 阶段保留**现有 `RadixCache` / `SWARadixCache` 作为 fallback（env var `SGL_JAX_USE_LEGACY_RADIX_CACHE=true` 切换），稳定 N 个版本后在独立的 **M3 阶段一次性删除**。HiCache 钩子只在 `UnifiedRadixCache` 路径上生效，legacy 路径不支持 HiCache。 |
| **理由** | (1) UnifiedRadixCache 是 port 自 sglang origin/main 的 3300+ 行新代码，一次性切换风险高；(2) sgl-jax 已有大量生产模型依赖现有 RadixCache/SWARadixCache，需要灰度验证；(3) env var fallback 让用户在发现问题时可以秒级回退到 legacy 路径，不阻塞生产；(4) 稳定后再删（M3）避免长期维护两套代码。 |
| **影响** | M1 阶段：UnifiedRadixCache 与 RadixCache 并存，env var 控制 dispatch；M2 阶段：UnifiedRadixCache + SWAComponent 与 SWARadixCache 并存；M3 阶段：删除 legacy，env var 报错提示用户。`kv_cache_builder` 的 dispatch 需要识别 `SGL_JAX_USE_LEGACY_RADIX_CACHE`（见 §5.2 修订）。 |
| **替代方案** | (a) 一次性删除：拒绝，风险过高；(b) 永久保留两套：拒绝，长期维护负担大，且不利于 HiCache/PD 统一集成。 |

### ADR-2: TPU only，接口直接用 JAX 原语

| | |
|---|---|
| **决策** | 所有新引入的 ABC（`HostKVPool` / `KVCacheController` / `HostMemoryAllocator` / `HiCacheStorage` / `KVTransferEngine`）的接口直接使用 `jax.Array`、`NamedSharding(memory_kind=...)`、`jax.device_put`、`jax.experimental.transfer` 等 JAX 原语，不引入 platform-agnostic 抽象。 |
| **理由** | (1) sgl-jax 定位本就是 TPU/JAX；(2) 与 tpu-inference 接口高度一致，未来跨 port 代码友好；(3) 不为想象中的 GPU 支持引入抽象成本，YAGNI；(4) 抽象会掩盖 JAX 的原生能力（如 `donate_argnames` / `optimization_barrier` / `memory_kind`）。 |
| **影响** | 未来若要加 GPU 支持需重构所有 ABC，已知风险。 |
| **替代方案** | Platform-agnostic 抽象层 —— 拒绝。 |

### ADR-3: 双 ABC（HiCacheStorage + KVTransferEngine）独立

| | |
|---|---|
| **决策** | RFC-0 定义两个独立 ABC。HiCache 用 `HiCacheStorage`（content-addressed K/V Store）；PD 用 `KVTransferEngine`（P2P-routed 同步传输）。两 ABC 不共享接口签名。 |
| **理由** | 协议层抽象本质不同。见 sglang 调研 §3.6。强行统一会牺牲语义清晰度。 |
| **影响** | RFC-1 实现一个 `HiCacheStorage` backend（toy file backend）；RFC-2 实现一个 `KVTransferEngine` backend（`jax.experimental.transfer` 基础）。底层引擎复用可在 backend 实现层完成，与 ABC 无关。 |
| **替代方案** | 单一统一 ABC、仅底层 KVBuffer 抽象 —— 均拒绝。 |

### ADR-4: kv_cache_builder 工厂模式

| | |
|---|---|
| **决策** | 引入 `python/sgl_jax/srt/mem_cache/kv_cache_builder.py`，导出 `build_kv_cache(...) -> KVCacheBuildResult`。scheduler 主类只调用此函数，不再做内部 dispatch。 |
| **理由** | (1) sglang origin/main 已采用此模式，便于未来 port；(2) scheduler.py 不污染；(3) builder 比 scheduler 内部 dispatch 更易测试。 |
| **影响** | `python/sgl_jax/srt/managers/scheduler.py` 中 `init_memory_pool_and_cache` 区域代码重写。 |
| **替代方案** | 保留 scheduler 内部 dispatch —— 拒绝。 |

### ADR-5: ChunkCache 是一等公民，非 fallback

| | |
|---|---|
| **决策** | `ChunkCache` 在 RFC-0 中保持现有实现，不做修改。`kv_cache_builder` 的 dispatch 把 ChunkCache 作为与 UnifiedRadixCache 并列的选项（条件是 `disable_radix_cache=True`），不是 fallback。 |
| **理由** | (1) ChunkCache 是 PD 分离场景的合理选择（无 prefix 复用，纯 KV pool 管理）；(2) 现有 chunked prefill + DP + overlap 测试都依赖 ChunkCache；(3) RFC-2 中 PD 必须明确支持 ChunkCache + PD 组合。 |
| **影响** | `chunk_cache.py` 不在删除清单内；现有测试不需迁移。 |
| **替代方案** | 把 ChunkCache 也合并到 UnifiedRadixCache —— 拒绝，因为 ChunkCache 的「无 prefix 复用」是一个明确语义。 |

### ADR-6: 兼容性兜底

| | |
|---|---|
| **决策** | RFC-0 落地不破坏现有以下能力：chunked prefill、overlap schedule (`event_loop_overlap`)、DP attention (SPMD)、mixed chunk、retract、partial rollout、SWA 模型（**MiMo-V2-Flash、MiMo-V2-Pro** 等）、Linear-Recurrent 模型（KDA、Bailing-MoE-Linear）、MLA 模型（DeepSeek）。 |
| **理由** | sgl-jax 已有大量生产模型依赖这些能力；RFC-0 是基础设施重构，不能引入功能回退。 |
| **影响** | UnifiedRadixCache port 必须验证以上场景全部可工作，详见 §14 测试矩阵。 |
| **替代方案** | 无。 |

---

## 3. 模块全景 + 文件路径建议

### 3.1 模块依赖图（ASCII）

```
                    ┌──────────────────────────────────────────┐
                    │       Scheduler (event loop)              │
                    │   complete with current dispatching:      │
                    │   • event_loop_normal / overlap / pp /    │
                    │     pdmux                                 │
                    │   仅修改: init_memory_pool_and_cache       │
                    │   → 改为调 kv_cache_builder.build()       │
                    └──────────────────┬───────────────────────┘
                                       │
                    ┌──────────────────▼───────────────────────┐
                    │  kv_cache_builder.build_kv_cache(args)    │   §5
                    │     ┌─────────────────────────────────┐   │
                    │     │ if disable_radix_cache:          │   │
                    │     │     return ChunkCache(...)       │   │
                    │     │ else:                            │   │
                    │     │     return UnifiedRadixCache(    │   │
                    │     │         components=[FULL,        │   │
                    │     │             +SWA if hybrid_swa,  │   │
                    │     │             +RECURRENT_STATE if hybrid_ssm]│
                    │     │     )                            │   │
                    │     └─────────────────────────────────┘   │
                    └──────────────────┬───────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
   ┌────▼──────┐                ┌──────▼─────────────┐         (PD 由 RFC-2 接管,
   │ChunkCache  │               │UnifiedRadixCache    │         独立 dispatch)
   │ (现存)      │               │ (port from sglang)  │  §4
   │ §3.4 不动   │               │ • match_prefix       │
   └───────────┘                │ • insert / evict     │
                                │ • cache_*_req         │
                                │ • init_load_back      │
                                │ • check_hicache_events│
                                │ • flush_write_through │
                                │   _acks               │
                                │ • ready_to_load_      │
                                │   host_cache          │
                                │   (HiCache 钩子全部齐) │
                                │                       │
                                │  components: list ──┐ │
                                └────────────────────┘ │
                                                       ▼
                              ┌──────────────────────────────────┐
                              │ TreeComponent ABC                  │ §4.3
                              │ • create_match_validator           │
                              │ • redistribute_on_node_split       │
                              │ • evict_component / drive_eviction │
                              │ • acquire/release_component_lock   │
                              │ • build_hicache_transfers          │  ← HiCache hooks
                              │ • commit_hicache_transfer          │
                              │ • drive_host_eviction              │
                              └──┬────────────┬─────────────┬────┘
                                 │            │             │
                       ┌─────────▼────┐ ┌────▼────────┐ ┌──▼────────────┐
                       │ FullComponent │ │ SWAComponent│ │ RecurrentStateComponent│
                       │ §4.4 port     │ │ §4.5 port    │ │ §4.6 port    │
                       └──────────────┘ └─────────────┘ └──────────────┘

   ───── 共享基础设施（被 UnifiedRadixCache、HiCache、PD 不同程度使用） ─────

   ┌────────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
   │ MemoryPools (扩展)  │    │ HostKVPool ABC §7    │    │ HostMemoryAllocator│
   │ §6                  │    │ • alloc / free        │    │ ABC §9             │
   │ pytree + host slot  │    │ • get_buffer / put    │    │ • alloc_group      │
   │                     │    │ • backed by           │    │ • free_group       │
   │ • req_to_token_pool │    │   pinned_host         │    └──────────────────┘
   │ • kv_pool            │    │   NamedSharding       │
   │ • [NEW] host_pool    │    └──────────────────────┘
   └────────────────────┘
                                ┌──────────────────────────┐
                                │ KVCacheController ABC §8 │
                                │ • write_buffer (D→H)      │
                                │ • load_buffer (H→D)       │
                                │ • check_events            │
                                │ (RFC-1 提供 TPU 实现)      │
                                └──────────────────────────┘

   ───── 两个独立 backend ABC（RFC-1 和 RFC-2 各自填实现） ─────

   ┌──────────────────────────────┐    ┌────────────────────────────────┐
   │ HiCacheStorage ABC  §10       │    │ KVTransferEngine ABC  §11      │
   │ (content-addressed K/V Store)  │    │ (P2P-routed sync transport)    │
   │ • batch_get_v2 / set_v2 /     │    │ • register_server              │
   │   exists_v2 (PoolTransfer)    │    │ • await_pull(uuid, data)       │
   │ • get / set / exists 单条     │    │ • connect(remote)              │
   │ • clear / get_stats           │    │ • pull(uuid, spec)             │
   │                                │    │                                │
   │ RFC-1 实现:                    │    │ RFC-2 实现:                     │
   │ • toy file backend             │    │ • jax.experimental.transfer    │
   │   (LocalFileStorage)           │    │   based backend                │
   │ • 未来: GCS / 自研             │    │ • bootstrap (toy HTTP server)  │
   └──────────────────────────────┘    └────────────────────────────────┘
```

### 3.2 文件路径变更清单

| 路径 | 操作 | 内容 |
|---|---|---|
| `python/sgl_jax/srt/mem_cache/base_prefix_cache.py` | **修改** | 完善 HiCache hooks 签名（§12）；保留现有 `MatchResult` / `MatchPrefixParams` 等 dataclass，新增字段（§4.2） |
| `python/sgl_jax/srt/mem_cache/kv_cache_builder.py` | **新增** | `build_kv_cache()` 工厂（§5） |
| `python/sgl_jax/srt/mem_cache/unified_radix_cache.py` | **新增** | port sglang origin/main 1960 行（§4） |
| `python/sgl_jax/srt/mem_cache/unified_cache_components/__init__.py` | **新增** | 导出三个 component |
| `python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py` | **新增** | port sglang origin/main 364 行（§4.3） |
| `python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py` | **新增** | port sglang origin/main 282 行（§4.4） |
| `python/sgl_jax/srt/mem_cache/unified_cache_components/swa_component.py` | **新增** | port sglang origin/main 537 行（§4.5） |
| `python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_state_component.py` | **新增** | port sglang origin/main 448 行（§4.6） |
| `python/sgl_jax/srt/mem_cache/memory_pool.py` | **修改** | `MemoryPools` 容器扩展 host pool slot（§6） |
| `python/sgl_jax/srt/mem_cache/host_kv_pool.py` | **新增** | `HostKVPool` ABC + 必要 dataclass（§7，**仅接口**） |
| `python/sgl_jax/srt/mem_cache/host_memory_allocator.py` | **新增** | `HostMemoryAllocator` ABC（§9，**仅接口**） |
| `python/sgl_jax/srt/mem_cache/cache_controller.py` | **新增** | `KVCacheController` ABC（§8，**仅接口**） |
| `python/sgl_jax/srt/mem_cache/hicache_storage.py` | **新增** | `HiCacheStorage` ABC + `PoolTransfer` 等 dataclass（§10，**仅接口**） |
| `python/sgl_jax/srt/disaggregation/__init__.py` | **新增** | 空 placeholder（RFC-2 填充） |
| `python/sgl_jax/srt/disaggregation/kv_transfer_engine.py` | **新增** | `KVTransferEngine` ABC（§11，**仅接口**） |
| `python/sgl_jax/srt/managers/scheduler.py` | **修改** | `init_memory_pool_and_cache` 改为调 `build_kv_cache`；移除现有 if-elif（§5.3） |
| `python/sgl_jax/srt/server_args.py` | **修改** | 新增配置项（§13） |
| `python/sgl_jax/srt/mem_cache/radix_cache.py` | **保留**（M3 删除） | legacy fallback |
| `python/sgl_jax/srt/mem_cache/swa_radix_cache.py` | **保留**（M3 删除） | legacy fallback |
| `python/sgl_jax/srt/mem_cache/chunk_cache.py` | **保留** | 一等公民，不变 |
| `python/sgl_jax/test/mem_cache/test_radix_cache.py` | **迁移** | 改为 `test_unified_radix_cache_full.py`，验证 FULL 路径等价于原 RadixCache（§14） |
| `python/sgl_jax/test/mem_cache/test_swa_radix_cache.py` | **迁移** | 改为 `test_unified_radix_cache_swa.py`（§14） |
| `python/sgl_jax/test/mem_cache/test_chunk_cache.py` | **不变** | 保留 |

### 3.3 RFC-0 完成后的目录形态

```
python/sgl_jax/srt/mem_cache/
├── base_prefix_cache.py            # 抽象 (含 HiCache hooks 完整签名)
├── chunk_cache.py                  # 保留, 无 prefix 复用
├── unified_radix_cache.py          # NEW, port sglang
├── unified_cache_components/
│   ├── __init__.py
│   ├── tree_component.py           # NEW, ABC
│   ├── full_component.py           # NEW
│   ├── swa_component.py            # NEW
│   └── recurrent_state_component.py          # NEW
├── kv_cache_builder.py             # NEW, 工厂
├── memory_pool.py                  # 修改, host pool slot
├── host_kv_pool.py                 # NEW, ABC only
├── host_memory_allocator.py        # NEW, ABC only
├── cache_controller.py             # NEW, ABC only
├── hicache_storage.py              # NEW, ABC only
├── allocator.py                    # 不变
└── recurrent_state_pool.py         # 不变

python/sgl_jax/srt/disaggregation/   # NEW dir
├── __init__.py
└── kv_transfer_engine.py           # NEW, ABC only
```

### 3.4 ChunkCache 在新架构中的位置

ChunkCache 不被 RFC-0 修改，但要明确它在新架构中的角色：

```
Scheduler
   │
   └─ kv_cache_builder.build_kv_cache()
        │
        ├─ if disable_radix_cache:
        │      → ChunkCache (or SWAChunkCache when hybrid_swa)
        │      • 无 prefix 复用
        │      • 不支持 HiCache (BasePrefixCache HiCache hooks 全部 NotImplementedError)
        │      • 仍支持 PD (PD 是独立维度，由 RFC-2 处理)
        │
        └─ else (默认):
               → UnifiedRadixCache
               • 有 prefix 复用
               • 支持 HiCache (RFC-1)
               • 支持 PD (RFC-2)
```

### 3.5 sgl-jax SPMD DP attention 与 PP 的关系（必读）

sgl-jax 的并行模型**与 sglang GPU 显著不同**，对 HiCache / PD 设计影响重大：

| 维度 | sgl-jax | sglang (GPU) |
|---|---|---|
| **进程模型** | 单 scheduler 进程 + ModelWorker 是 in-process class（thread-based） | 每 TP rank 一个独立进程 |
| **DP attention** | **SPMD**：mesh 形状 `(dp_size, tp_size / dp_size)`，一个进程内多个 DP rank，attention 上按 DP 切分 | 每 DP group 独立 NCCL 进程组，进程之间不共享 mesh |
| **tree_cache 实例** | **单实例 + 内部 per-DP 分区**（`tree_cache.evictable_size(dp_rank=dp)` 这样的 API） | 每 TP rank 一份独立 tree |
| **req_to_token_pool** | 全局单实例（DP 间共享物理 KV 空间） | 每 rank 一份 |
| **DP 路由** | scheduler `select_dp_for_request` round-robin 分给 dp_rank | DataParallelController 分发到独立进程 |
| **PP（Pipeline Parallel）** | **长期不支持**（sgl-jax 当前架构不规划 PP） | 支持 |

**对 HiCache（RFC-1）的影响**：
- `LRUHostKVPool` 必须**支持 per-DP-rank 维度**（host pool 按 DP rank 分区，与 device kv_pool 的 DP 切分对齐），否则跨 DP rank 的 prefix hash key 会冲突
- `TPUKVCacheController` 的 `write/load` 调用必须传 `dp_rank` 参数
- HiCache hash key（RFC-1 §7.3）必须把 `dp_rank` 纳入：`SHA256(prefix_tokens + tp_rank + dp_rank + model_name)`

**对 PD（RFC-2）的影响**：
- `KVArgs` 用 `dp_rank` 而不是 sglang 的 `system_dp_rank`
- bootstrap_room 路由必须考虑 dp_rank（P 的 dp_rank N 的请求路由到 D 的 dp_rank N，确保 KV shard 维度对齐）
- 没有 `pp_rank` / `pp_size` 字段

**SPMD 下的 HiCache 工作模型**：
```
单进程 (dp_size=2, tp_size=8, attention_tp_size=4):

  Scheduler 主线程
       │
       ├─ tree_cache (单实例)
       │    ├─ dp_rank=0 sub-tree
       │    └─ dp_rank=1 sub-tree
       │
       ├─ host_kv_pool (单实例, per-DP 分区)
       │    ├─ dp_rank=0 host indices [0, host_size/2)
       │    └─ dp_rank=1 host indices [host_size/2, host_size)
       │
       └─ cache_controller
            ├─ write(device_indices, node_id, dp_rank=0): host indices 取 dp=0 段
            └─ write(device_indices, node_id, dp_rank=1): host indices 取 dp=1 段
```

> 注：本节是 RFC-0 的设计约束，RFC-1 / RFC-2 实施时必须遵守。

---

## 4. 详细设计：UnifiedRadixCache + 三个 TreeComponent

### 4.1 总体策略：直接 port sglang origin/main

sglang 已经在 origin/main 上落地完整的 D↔H HiCache 集成（参见 sglang 调研 §4）。port 策略：

| 维度 | 策略 |
|---|---|
| **代码量** | 总计约 4000 行 Python（unified_radix_cache 1960 + tree_component 364 + full 282 + swa 537 + mamba 448） |
| **PyTorch → JAX 替换** | `torch.Tensor` → `jax.Array`；`torch.int32` / `torch.bool` 等 dtype 直译；CPU tensor / numpy 用于 indices 数组的部分保持不变 |
| **Stream / Event 替换** | sglang 用 CUDA stream + event 做异步。JAX 默认异步派发，所以**删除所有显式 stream 管理**，只保留 `jax.block_until_ready` 同步点 |
| **Mesh / Sharding** | sgl-jax 现有 KV pool 已用 `NamedSharding(mesh, P("data", None, "tensor", None, None))`，port 时直接复用 |
| **`req_to_token_pool` API** | sgl-jax 用 `ReqToTokenPool` / `HybridReqToTokenPool`（已有），sglang 的 API 名称基本一致，少量字段差异需对齐 |
| **MLA / Mamba 支持** | port 时保留 MLA / Mamba 分支，因为 sgl-jax 已有 MLA / KDA 等 hybrid 模型 |
| **HiCache hooks 现状** | 仅 port 接口签名 + 默认行为；具体 D↔H 实现挂在 `KVCacheController` ABC 上，由 RFC-1 提供 |

### 4.2 `MatchResult` / `MatchPrefixParams` 等 dataclass 扩展

sglang origin/main 的 `MatchResult`（base_prefix_cache.py L145）比 sgl-jax 现状多两个字段：

| 字段 | sgl-jax 现状 | sglang origin/main | RFC-0 操作 |
|---|---|---|---|
| `device_indices` | ✓ | ✓ | 不变 |
| `last_device_node` | ✓ | ✓ | 不变 |
| `last_host_node` | ✓ | ✓ | 不变 |
| `host_hit_length` | ✓ | ✓ | 不变 |
| `best_match_node` | ✗ | ✓ | **新增**，UnifiedRadixCache 的 HiCache load_back 锚点 |
| `mamba_branching_seqlen` | ✗ | ✓ | **新增**，Mamba 分支判定用 |
| `cache_protected_len` | ✗ | ✓ | **新增**，前缀保护长度 |

`MatchPrefixParams`：
| 字段 | sgl-jax 现状 | sglang origin/main | RFC-0 操作 |
|---|---|---|---|
| `key` | ✓ | ✓ | 不变 |
| `req` | ✓ | ✓ | 不变 |
| `cow_mamba` | ✗ | ✓ | **新增**，Mamba CoW 标志 |

### 4.3 `TreeComponent` ABC 核心接口

```python
# python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py
# port from sglang origin/main, ~364 lines

class ComponentType(IntEnum):
    FULL = 0
    SWA = 1
    RECURRENT_STATE = 2   # sgl-jax 重命名 (sglang 历史名 MAMBA), 涵盖所有 linear-recurrent 模型: Mamba/SSM/KDA/RetNet/RWKV 等

class CacheTransferPhase(IntEnum):
    BACKUP_HOST = 0       # D2H
    LOAD_BACK = 1         # H2D
    BACKUP_STORAGE = 2    # H → L3
    PREFETCH = 3          # L3 → H

@dataclass
class ComponentData:
    value: Optional[jax.Array] = None        # GPU/TPU pool indices
    lock_ref: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    host_value: Optional[jax.Array] = None   # CPU pool indices (HiCache)
    host_lock_ref: int = 0

class TreeComponent(ABC):
    # ===== 核心 14 个 hook (port verbatim from sglang) =====
    def node_has_component_data(self, node) -> bool: ...
    def value_len(self, node) -> int: ...
    @abstractmethod
    def create_match_validator(self) -> Callable[["UnifiedTreeNode"], bool]: ...
    def finalize_match_result(self, result, params, value_chunks, best_value_len) -> MatchResult: ...
    def update_component_on_insert_overlap(self, node, prefix_len, total_prefix_len,
                                           value_slice, params) -> int: ...
    def should_skip_leaf_creation(self, total_prefix_len, key_len, params) -> bool: ...
    def commit_insert_component_data(self, node, is_new_leaf, params, result) -> None: ...
    @abstractmethod
    def redistribute_on_node_split(self, new_parent, child) -> None: ...
    @abstractmethod
    def evict_component(self, node, is_leaf) -> int: ...
    def eviction_priority(self, is_leaf) -> int: ...
    @abstractmethod
    def drive_eviction(self, params, tracker) -> None: ...
    @abstractmethod
    def acquire_component_lock(self, node, result) -> IncLockRefResult: ...
    @abstractmethod
    def release_component_lock(self, node, params) -> None: ...
    def prepare_for_caching_req(self, req, insert_params, token_ids_len, is_finished) -> Optional[int]: ...
    def cleanup_after_caching_req(self, ...) -> None: ...

    # ===== HiCache hooks (RFC-0 仅给签名, 默认行为 = 主路径处理) =====
    def build_hicache_transfers(self, node, phase: CacheTransferPhase, **kw
                                ) -> Optional[list["PoolTransfer"]]:
        return None

    def commit_hicache_transfer(self, node, phase: CacheTransferPhase,
                                transfers: Sequence["PoolTransfer"] = ()) -> None:
        return

    def drive_host_eviction(self, num_tokens: int, tracker) -> None:
        return
```

`build_hicache_transfers` 的返回 `PoolTransfer` 定义在 `hicache_storage.py`（§10），是 HiCache D↔H↔L3 的统一传输描述符。

### 4.4 `FullComponent` — 标准 KV (最简)

```python
# python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py
# port from sglang origin/main, ~282 lines

class FullComponent(TreeComponent):
    component_type = ComponentType.FULL

    def create_match_validator(self):
        return lambda node: True   # 所有节点对 FULL 有效

    def redistribute_on_node_split(self, new_parent, child):
        # FULL: lock_ref 从子节点复制到新父节点 (sglang L_TBD)
        ...

    def evict_component(self, node, is_leaf) -> int:
        # FULL: 仅驱逐叶节点
        ...

    def eviction_priority(self, is_leaf) -> int:
        return 2 if not is_leaf else 0  # 内部节点优先级最高 (最后驱逐)

    def drive_eviction(self, params, tracker):
        # 从 evictable leaves 优先级堆驱逐
        ...

    def acquire_component_lock(self, node, result):
        # Path-lock: 从匹配节点到根, 递增所有祖先的 lock_ref
        ...

    # HiCache hooks
    def drive_host_eviction(self, num_tokens, tracker) -> None:
        # L142 in sglang: heap 出 evictable host leaf 调 _evict_host_leaf
        ...

    def build_hicache_transfers(self, node, phase, **kw):
        if phase == CacheTransferPhase.BACKUP_HOST:
            return None   # 主路径处理 (FULL 走 cache_controller 直接调度)
        if phase == CacheTransferPhase.LOAD_BACK:
            # 沿 evicted 链上溯收集 host_value 生成单条 PoolTransfer(name=KV)
            ...

    def commit_hicache_transfer(self, node, phase, transfers=()):
        # device_indices 切片写回各 node cd.value + 更新 evictable leaf 集合
        ...
```

### 4.5 `SWAComponent` — 滑动窗口

要点：
- 双 LRU 链表（full vs swa），匹配验证器追踪累积连续 SWA 覆盖长度直到 `sliding_window_size`
- Window-lock 策略：向上走到填满滑动窗口为止，用 UUID 标记边界
- 节点 split 时切分 value tensor，复制 lock_ref 和 UUID
- HiCache hook：
  - `build_hicache_transfers`：BACKUP 返回 `PoolName.SWA`；LOAD_BACK 只在 `sliding_window_size` 窗口内收 host-only 节点
  - `commit_hicache_transfer`：`_restore_device_value` + `allocator.set_full_to_swa_mapping` 重建 full↔swa 映射
  - `drive_host_eviction`：host LRU 驱动，区分 leaf vs internal (tombstone + cascade)

**HiCache hook 范围**：SWAComponent 的 3 个 HiCache hook 完整 port（不只签名，含 body 实现），属于 M2 阶段范围。这些 hook 的"工作默认"行为已可用——RFC-1 仅需提供底层 `KVCacheController` / `HiCacheStorage` 实现来替换 hook 调用的具体 backend，不需要改 SWAComponent 本身。

### 4.6 `RecurrentStateComponent` — Mamba/SSM 混合

要点：
- 匹配验证器：仅当节点有 mamba 数据时返回 True
- Single-node lock：仅锁匹配节点本身
- copy-on-write 语义
- HiCache hook：
  - `build_hicache_transfers`：BACKUP 返回 `PoolName.RECURRENT_STATE`；LOAD_BACK 支持单节点 restore + per-request CoW（按需 `recurrent_state_pool.alloc(1)`，OOM 触发 `evict(recurrent_state_num=1)`）
  - `commit_hicache_transfer`：写回 cd.value，host LRU → device LRU 迁移

**HiCache hook 范围**：与 SWAComponent 同——3 个 HiCache hook 完整 port（含 body），属于 M2 阶段范围；RFC-1 替换 backend 而非 component 本身。

### 4.7 UnifiedRadixCache 主类接口（与 sglang origin/main 1:1 对齐）

| 方法 | 行号 (sglang) | 签名 |
|---|---|---|
| `__init__` | 202 | `(self, params: CacheInitParams)` |
| `init_hicache` | 298 | `(self, server_args, params) -> None` — 在 builder 中按需调用 |
| `register_sidecar_pool` | 338 | `(self, spec: SidecarPoolSpec) -> None` |
| `match_prefix` | 341 | `(self, params: MatchPrefixParams) -> MatchResult` |
| `insert` | 368 | `(self, params: InsertParams) -> InsertResult` |
| `evict` | 384 | `(self, params: EvictParams) -> EvictResult` |
| `evict_host` | 1035 | host pool 驱逐 |
| `write_backup` | 1160 | `(self, node, write_back: bool = False) -> int` (D→H) |
| `load_back` | 1223 | (H→D) |
| `writing_check` | 1368 | `(self, write_back: bool = False) -> None` |
| `loading_check` | 1415 | `(self) -> None` |
| `init_load_back` | 1432 | `(self, params: InitLoadBackParams) -> tuple[jax.Array, UnifiedTreeNode]` |
| `check_hicache_events` | 1478 | `(self) -> None` |
| `flush_write_through_acks` | 1483 | `(self) -> None` |
| `ready_to_load_host_cache` | 1487 | `(self) -> int` |
| `cache_finished_req` / `cache_unfinished_req` | — | 继承 `BasePrefixCache` 接口 |
| `inc_lock_ref` / `dec_lock_ref` | — | 继承 |

### 4.8 RadixCache / SWARadixCache 渐进迁移清单（ADR-1）

**M1/M2 阶段**：legacy 文件**保留**，作为 `SGL_JAX_USE_LEGACY_RADIX_CACHE=true` env var 时的 fallback：

| 文件 | M1/M2 操作 | M3 操作 |
|---|---|---|
| `python/sgl_jax/srt/mem_cache/radix_cache.py` | **保留**（legacy fallback） | 删除 |
| `python/sgl_jax/srt/mem_cache/swa_radix_cache.py` | **保留**（legacy fallback） | 删除 |
| `python/sgl_jax/srt/managers/scheduler.py` | 接入 builder（builder 内部按 env var dispatch） | 简化（移除 env var 处理） |
| `python/sgl_jax/srt/mem_cache/kv_cache_builder.py` | 新增；内部按 `SGL_JAX_USE_LEGACY_RADIX_CACHE` 二选一 | 简化（移除 legacy 分支） |
| `python/sgl_jax/test/mem_cache/test_radix_cache.py` | **保留 + 新增** `test_unified_radix_cache_full.py` | 删除 legacy 测试 |
| `python/sgl_jax/test/mem_cache/test_swa_radix_cache.py` | **保留 + 新增** `test_unified_radix_cache_swa.py` | 删除 legacy 测试 |
| `python/sgl_jax/server_args.py` | 加 `SGL_JAX_USE_LEGACY_RADIX_CACHE` env var 解析 | 移除 env var |
| `docs/basic_usage/features/radix_cache.md` | 更新：默认 UnifiedRadixCache + 说明 env var fallback | 移除 env var 段落 |
| `docs/design/swa_eviction_and_lru_strategy.md` | 更新引用：SWA 行为已迁到 SWAComponent；legacy 仍可用 | 仅指向 SWAComponent |
| 任何 import RadixCache/SWARadixCache 的代码 | grep 替换为 UnifiedRadixCache（除 builder + legacy fallback 路径） | 清理剩余引用 |

**触发 M3 的条件**（见 §16.1）：
- M2 完成后 N 个 release（建议 ≥ 2）无 production incident
- 端到端 KL test 在所有支持模型上跑通 ≥ 2 个 release
- 没有用户反馈依赖 `SGL_JAX_USE_LEGACY_RADIX_CACHE=true` 的场景

---

## 5. 详细设计：kv_cache_builder

### 5.1 接口

```python
# python/sgl_jax/srt/mem_cache/kv_cache_builder.py

@dataclass
class KVCacheBuildParams:
    server_args: ServerArgs
    model_config: ModelConfig
    mesh: jax.sharding.Mesh
    page_size: int
    max_total_num_tokens: int
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    is_mla: bool
    # ... 其他从 scheduler 抽出的入参

@dataclass
class KVCacheBuildResult:
    tree_cache: BasePrefixCache
    memory_pools: MemoryPools                # §6
    # HiCache 相关 (由 RFC-1 填充, RFC-0 留 None placeholder):
    host_kv_pool: Optional[HostKVPool] = None         # §7
    cache_controller: Optional[KVCacheController] = None  # §8

def build_kv_cache(params: KVCacheBuildParams) -> KVCacheBuildResult:
    ...
```

### 5.2 Dispatch 算法

```python
def build_kv_cache(params: KVCacheBuildParams) -> KVCacheBuildResult:
    sa = params.server_args

    # Step 1: 创建 memory_pools (含 KV pool, req_to_token_pool, 可选 host pool)
    memory_pools = _build_memory_pools(params)

    # HiCache 相关默认 None (ChunkCache 路径不创建)
    host_kv_pool: Optional[HostKVPool] = None
    cache_controller: Optional[KVCacheController] = None

    # Step 1.5: D 节点第一版仅支持 ChunkCache (RFC-2 ADR-9)
    # 在 disagg_mode=decode 时, 静默覆盖 disable_radix_cache + enable_hierarchical_cache.
    # 第一版限制 — 未来可按 sglang 模式加 opt-in flag 支持 D + RadixCache.
    force_chunk_cache_d = (sa.disaggregation_mode == "decode")
    if force_chunk_cache_d:
        if not sa.disable_radix_cache:
            logger.warning(
                "D + RadixCache not implemented in first release (RFC-2 ADR-9); "
                "forcing ChunkCache. See ADR-9 future extension point for opt-in."
            )
        if sa.enable_hierarchical_cache:
            logger.warning(
                "enable_hierarchical_cache ignored on decode node "
                "(ChunkCache has no HiCache hooks)"
            )

    # Step 2: 选择 tree_cache 类型
    if sa.disable_radix_cache or force_chunk_cache_d:
        # ChunkCache 路径 (普通禁用 radix 或 D 节点第一版限制)
        if params.is_hybrid_swa:
            tree_cache = SWAChunkCache(memory_pools=memory_pools, ...)
        else:
            tree_cache = ChunkCache(memory_pools=memory_pools, ...)
        # HiCache 在此路径下不生效 (ChunkCache 无 hook)
    else:
        # UnifiedRadixCache 路径 (P 节点 / null 节点)
        components = [FullComponent(...)]
        if params.is_hybrid_swa:
            components.append(SWAComponent(...))
        if params.is_hybrid_ssm:
            components.append(RecurrentStateComponent(...))

        tree_cache = UnifiedRadixCache(
            CacheInitParams(
                memory_pools=memory_pools,
                tree_components=components,
                page_size=params.page_size,
                ...
            )
        )

        # Step 3: 如果开了 HiCache, 装上 cache_controller (由 RFC-1 提供)
        if sa.enable_hierarchical_cache:
            host_kv_pool, cache_controller = _build_hicache_infra(memory_pools, sa)
            tree_cache.init_hicache(sa, cache_init_params)
            # 把 host_kv_pool 和 cache_controller 挂上去
            ...

    return KVCacheBuildResult(
        tree_cache=tree_cache,
        memory_pools=memory_pools,
        host_kv_pool=host_kv_pool,
        cache_controller=cache_controller,
    )
```

### 5.3 Scheduler 集成

`python/sgl_jax/srt/managers/scheduler.py` 中 `init_memory_pool_and_cache` 改写为：

```python
def init_memory_pool_and_cache(self):
    params = KVCacheBuildParams(
        server_args=self.server_args,
        model_config=self.model_config,
        mesh=self.mesh,
        page_size=self.page_size,
        max_total_num_tokens=self.max_total_num_tokens,
        is_hybrid_swa=self.is_hybrid_swa,
        is_hybrid_ssm=self.is_hybrid_ssm,
        is_mla=self.is_mla,
    )
    result = build_kv_cache(params)
    self.tree_cache = result.tree_cache
    self.memory_pools = result.memory_pools
    self.host_kv_pool = result.host_kv_pool
    self.cache_controller = result.cache_controller
```

---

## 6. 详细设计：MemoryPools 容器扩展

### 6.1 现状

`python/sgl_jax/srt/mem_cache/memory_pool.py:1360` `class MemoryPools`：
- 是 pytree（用 `@jax.tree_util.register_pytree_node`），便于 JIT 边界传递
- 当前字段：`req_to_token_pool`、`kv_pool`（list 或单个）

### 6.2 扩展

新增两个字段：`token_allocator`（被 RFC-1 cache_controller 用）+ `host_pool`（RFC-1 填充）。

```python
@register_pytree_node_class
@dataclass
class MemoryPools:
    req_to_token_pool: ReqToTokenPool
    kv_pool: KVPool
    token_allocator: BaseTokenToKVPoolAllocator   # NEW, 现有 scheduler 单独创建, 改为容器内
    host_pool: Optional["HostKVPool"] = None       # NEW, RFC-1 填充

    def tree_flatten(self):
        children = (self.req_to_token_pool, self.kv_pool,
                    self.token_allocator, self.host_pool)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        req_to_token_pool, kv_pool, token_allocator, host_pool = children
        return cls(req_to_token_pool=req_to_token_pool,
                   kv_pool=kv_pool,
                   token_allocator=token_allocator,
                   host_pool=host_pool)
```

**与 sgl-jax 现状的差异**：当前 `allocator` 在 scheduler 中独立创建（`scheduler.py`），不进 `MemoryPools` 容器。RFC-0 把它纳入容器是为了让 RFC-1 的 `TPUKVCacheController` 通过 `memory_pools.token_allocator` 拿到 allocator 而不需新增依赖注入路径。Scheduler 现有代码改为 `memory_pools.token_allocator = result.memory_pools.token_allocator`。

### 6.3 与 replace_buffer 模式的兼容

sgl-jax 现有 `MemoryPools.replace_buffer` 是 per-jit-step buffer 替换的核心。host_pool 在 JIT 内部一般不参与 forward，所以默认不需要 replace。但 `tree_flatten` / `tree_unflatten` 必须正确处理 None（host_pool 在禁用 HiCache 时是 None）。

### 6.4 Sharding 考虑

host_pool 的内部 buffer 用 `NamedSharding(mesh, P("data", None, "tensor", None, None), memory_kind="pinned_host")`——与 device kv_pool 的 PartitionSpec 一致，仅 `memory_kind` 不同。这允许 D2H/H2D 通过 `jax.device_put` 自动路由（参见 jax-api 调研 §3）。

---

## 7. 详细设计：HostKVPool ABC（仅接口）

### 7.1 设计目标

参考 tpu-inference 的 `HostKVPool`（`tpu_inference/distributed/host_kv_pool.py`）和 sglang 的 `MHATokenToKVPoolHost`，提供一个 host-side KV pool 的统一抽象。**注意**：tpu-inference 的 `HostKVPool` 实际上是 staging buffer pool（PD 用）；sglang 的 `MHATokenToKVPoolHost` 是 LRU storage（HiCache 用）。RFC-0 的 ABC 设计要能同时支持两种语义，但本 RFC 不提供具体实现。

### 7.2 接口

```python
# python/sgl_jax/srt/mem_cache/host_kv_pool.py

@dataclass
class HostBufferHandle:
    """
    对一段 pinned host memory 的 handle.

    设计约定:
    - `indices` 是稳定不变的 token-level indices (调用方可长期持有)
    - `buffer` 是 view-style 字段, 实现可填 None
        - LRU 实现 (RFC-1 LRUHostKVPool): pool 内 underlying buffer 可能因 .at[].set() 重新绑定,
          所以 LRU 实现填 None, 调用方必须用 pool.read_indices(handle.indices) 读数据
        - Queue 实现 (RFC-2 QueueHostKVPool): buffer 是稳定的预分配 chunk, 可直接持有
    - 调用方约束:
        - 如需读 KV 数据: 永远调 pool.read_indices(handle.indices), 不直接读 handle.buffer
        - 如需写 KV 数据: 永远调 pool.write_indices(handle.indices, kv_data)
        - handle.buffer 仅作为「Queue 实现下的 zero-copy 优化」预留字段
    """
    indices: jax.Array              # token-level indices (always valid)
    buffer_id: int                  # 池内 ID, 用于 free 时归还
    buffer: Optional[jax.Array] = None  # 仅在稳定 buffer 实现中非 None

class HostKVPool(ABC):
    """
    Host-side KV pool 抽象。
    用 NamedSharding(memory_kind="pinned_host") 持有 pinned host memory。
    支持两种 use case:
      - HiCache L2 storage (LRU 管理, 长生命周期)
      - PD staging buffer (queue 管理, 短生命周期)
    具体实现 (含 LRU 或 queue 行为) 由 RFC-1/RFC-2 提供。
    """

    @abstractmethod
    def alloc(self, num_tokens: int) -> Optional[HostBufferHandle]:
        """分配 num_tokens 大小的连续 buffer。返回 None 表示分配失败"""

    @abstractmethod
    def free(self, handle: HostBufferHandle) -> None:
        """归还 buffer"""

    @abstractmethod
    def available_size(self) -> int:
        """剩余可分配 token 数"""

    @abstractmethod
    def total_size(self) -> int:
        """总容量 (token 数)"""

    # ===== 数据读写 (所有实现必须支持) =====
    @abstractmethod
    def read_indices(self, indices: jax.Array) -> Any:
        """
        读取 indices 对应的 KV 数据.
        返回类型由实现决定 (np.ndarray 或 jax.Array; per-layer list 或 fused).
        调用方应理解 host pool 数据格式并与之兼容.
        """

    @abstractmethod
    def write_indices(self, indices: jax.Array, kv_data: Any) -> None:
        """
        把 KV 数据写到 indices 对应位置.
        kv_data 类型应与 read_indices 返回一致.
        """

    # HiCache 专用 (RFC-1 LRU 实现): 默认 NotImplementedError
    def evict(self, num_tokens: int) -> int:
        raise NotImplementedError("evict() requires LRU-backed implementation")

    def lock_ref_inc(self, handle: HostBufferHandle) -> None:
        raise NotImplementedError

    def lock_ref_dec(self, handle: HostBufferHandle) -> None:
        raise NotImplementedError

    # PD 专用 (RFC-2 queue 实现): 默认 NotImplementedError
    def get_buffer(self) -> tuple[int, HostBufferHandle]:
        raise NotImplementedError("get_buffer() requires queue-backed implementation")

    def put_buffer(self, buffer_id: int) -> None:
        raise NotImplementedError
```

**关于双 use case 的设计选择**：本 ABC 同时混入了 LRU 风格方法（`evict` / `lock_ref_inc` / `lock_ref_dec`）和 queue 风格方法（`get_buffer` / `put_buffer`），各自默认 `NotImplementedError`。RFC-1 和 RFC-2 实现时**应明确选择一种语义**（不要同时实现两套），常见做法：

- **RFC-1**（HiCache）：实现 `LRUHostKVPool(HostKVPool)`，覆盖 alloc/free/evict/lock_ref_*，不实现 queue 方法
- **RFC-2**（PD）：实现 `QueueHostKVPool(HostKVPool)`，覆盖 alloc/free/get_buffer/put_buffer，不实现 LRU 方法

合并到单 ABC 是为了让 `MemoryPools.host_pool` 字段类型统一（不需要 union 类型）；具体实现的二分由子类完成。

### 7.3 与 BasePrefixCache 的连接

HiCache 用 `MatchResult.last_host_node` 和 `BasePrefixCache.init_load_back` 来触发 H2D；具体执行委托给 `KVCacheController`（§8）。`HostKVPool` 只负责内存分配/释放，不负责 D2H/H2D 的实际搬运。

---

## 8. 详细设计：KVCacheController ABC（仅接口）

### 8.1 设计目标

参考 sglang `HiCacheController`（`managers/cache_controller.py`）。在 sgl-jax 上对应物：调度 D2H/H2D 的异步搬运、协调 BACKUP / LOAD / PREFETCH / BACKUP_STORAGE 四个阶段。

**关键差异 vs sglang**：sgl-jax 没有 CUDA stream，所有传输由 `jax.device_put` + `ThreadPoolExecutor` 完成（参考 jax-api 调研 §5.2 GIL 释放机制）。

### 8.2 接口

```python
# python/sgl_jax/srt/mem_cache/cache_controller.py

@dataclass
class TransferAck:
    """传输完成的标识 (类比 sglang HiCacheAck)"""
    op_id: int
    done: bool
    node_ids: list[int]

class KVCacheController(ABC):
    """
    D2H/H2D 异步搬运调度器。
    具体实现 (TPU 版, 用 jax.device_put + ThreadPoolExecutor) 由 RFC-1 提供。
    """

    # ===== D2H (write_buffer) =====
    @abstractmethod
    def write(self, device_indices: jax.Array, node_id: int
              ) -> Optional[HostBufferHandle]:
        """分配 host buffer + 入队 D2H. 同步调用, 不等待完成"""

    @abstractmethod
    def start_writing(self) -> None:
        """提交 D2H 任务到后台线程"""

    @abstractmethod
    def writing_check(self) -> list[TransferAck]:
        """非阻塞检查已完成的 D2H"""

    # ===== H2D (load_buffer) =====
    @abstractmethod
    def load(self, host_indices: jax.Array, node_id: int
             ) -> Optional[jax.Array]:
        """分配 device indices + 入队 H2D. 同步调用"""

    @abstractmethod
    def start_loading(self) -> int:
        """提交 H2D 任务. 返回 producer_id"""

    @abstractmethod
    def loading_check(self) -> list[TransferAck]:
        """非阻塞检查已完成的 H2D"""

    # ===== L3 prefetch (RFC-1 选做) =====
    def prefetch_from_storage(self, hash_keys: list[bytes],
                              host_indices: jax.Array) -> Optional[int]:
        raise NotImplementedError("L3 prefetch is optional; not in MVP")

    def write_backup_storage(self, hash_keys: list[bytes],
                             host_indices: jax.Array) -> Optional[int]:
        raise NotImplementedError("L3 backup is optional; not in MVP")
```

### 8.3 与 UnifiedRadixCache 的连接

UnifiedRadixCache 内部持有 `KVCacheController` 实例（通过 `init_hicache()` 注入）：

```
UnifiedRadixCache.write_backup(node)
    → self.cache_controller.write(device_indices, node.id)
    → self.cache_controller.start_writing()

UnifiedRadixCache.load_back(node)
    → self.cache_controller.load(host_indices, node.id)
    → self.cache_controller.start_loading()

UnifiedRadixCache.check_hicache_events()
    → self.cache_controller.writing_check() + loading_check()
```

---

## 9. 详细设计：HostMemoryAllocator ABC（仅接口）

```python
# python/sgl_jax/srt/mem_cache/host_memory_allocator.py

class HostMemoryAllocator(ABC):
    """
    Host-side index allocator (类比 device 侧的 TokenToKVPoolAllocator).
    用于在 host_kv_pool 内部管理 token-level indices.
    """

    @abstractmethod
    def alloc(self, n: int) -> Optional[jax.Array]:
        """分配 n 个连续 indices"""

    @abstractmethod
    def free(self, indices: jax.Array) -> None: ...

    @abstractmethod
    def alloc_group(self, sizes: list[int]) -> Optional[list[jax.Array]]:
        """一次性分配多组 indices (减少 fragment)"""

    @abstractmethod
    def free_group(self, groups: list[jax.Array]) -> None: ...

    @abstractmethod
    def available_size(self) -> int: ...

    @abstractmethod
    def total_size(self) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...
```

---

## 10. 详细设计：HiCacheStorage ABC

### 10.1 接口（对齐 sglang origin/main）

```python
# python/sgl_jax/srt/mem_cache/hicache_storage.py

class PoolName(IntEnum):
    KV = 0
    SWA = 1
    RECURRENT_STATE = 2   # sgl-jax 重命名 (sglang 历史名 MAMBA), 涵盖所有 linear-recurrent 模型: Mamba/SSM/KDA/RetNet/RWKV 等
    AUX = 3

class PoolHitPolicy(IntEnum):
    BEST_EFFORT = 0
    WAIT_COMPLETE = 1
    TIMEOUT = 2

@dataclass
class HiCacheStorageConfig:
    backend_name: str
    extra_config: dict[str, Any] = field(default_factory=dict)
    is_mla_model: bool = False
    tp_rank: int = 0
    tp_size: int = 1
    dp_rank: int = 0          # sgl-jax SPMD DP attention: 同进程内多 DP rank, host pool 需 per-DP 隔离
    dp_size: int = 1
    model_name: str = ""
    # 注: PP (pipeline parallel) 字段不在 sgl-jax 第一版 PD/HiCache 范围,
    # sgl-jax 长期不计划支持 PP. 未来如需, 可加 pp_rank/pp_size.

@dataclass
class HiCacheStorageExtraInfo:
    """单次传输的额外上下文 (model_name, layer 范围, etc.)"""
    ...

@dataclass
class PoolTransfer:
    """统一传输描述符 (HiCache D↔H↔L3 通用)"""
    name: PoolName
    hash_keys: list[bytes]                # content-addressed key
    host_indices: jax.Array              # host pool 内的 indices
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class PoolTransferResult:
    success: bool
    completed_tokens: int

class HiCacheStorage(ABC):
    """
    L3 storage 抽象 (content-addressed K/V Store).
    被 HiCache 用 (RFC-1).
    本 RFC 仅定义接口; RFC-1 提供 toy file backend; 未来可扩展 GCS / 自研.

    设计要点 (与 sglang origin/main 对齐):
    - v2 API (batch_*_v2) 接收 PoolTransfer 描述符, 支持多 pool
    - v1 API (batch_*_v1) 是兼容路径, 单 pool. RFC-0 暂不实现 v1.
    """

    def register_mem_pool_host(self, host: HostKVPool) -> None:
        """单 pool 注册 (v1 路径). 默认 no-op."""

    @abstractmethod
    def register_mem_host_pool_v2(self, host_pool: HostKVPool,
                                  host_pool_name: PoolName) -> None:
        """多 pool 注册 (v2 路径). 必须实现."""

    # ===== v2 API (多 pool, PoolTransfer 描述符) =====
    @abstractmethod
    def batch_exists_v2(self, keys: list[bytes], pool_name: PoolName
                        ) -> list[bool]: ...

    @abstractmethod
    def batch_get_v2(self, transfers: list[PoolTransfer]
                     ) -> list[PoolTransferResult]: ...

    @abstractmethod
    def batch_set_v2(self, transfers: list[PoolTransfer]
                     ) -> list[PoolTransferResult]: ...

    # ===== 单条 API =====
    @abstractmethod
    def get(self, key: bytes, pool_name: PoolName
            ) -> Optional[jax.Array]: ...

    @abstractmethod
    def set(self, key: bytes, value: jax.Array,
            pool_name: PoolName) -> bool: ...

    @abstractmethod
    def exists(self, key: bytes, pool_name: PoolName) -> bool: ...

    # ===== 管理 =====
    @abstractmethod
    def clear(self) -> None: ...

    def get_stats(self) -> dict[str, int]:
        """监控用. 默认返回空 dict."""
        return {}
```

### 10.2 不在 RFC-0 范围

- `batch_get_v1` / `batch_set_v1` / `batch_get` / `batch_set` 等兼容性 API：RFC-0 不实现
- 任何具体 backend（file / GCS / mooncake / 等）：RFC-1 实现 file backend

### 10.3 内容寻址 key 设计

`key: bytes` 应使用 SHA256 hash of (prefix tokens + tp_rank + ...)。sglang 的 `compute_node_hash_values` 是参考实现。RFC-1 的 file backend 必须落地这套 key scheme，否则未来无法迁移到分布式 backend。

---

## 11. 详细设计：KVTransferEngine ABC

### 11.1 接口（参考 tpu-inference `TPUConnector`）

```python
# python/sgl_jax/srt/disaggregation/kv_transfer_engine.py

@dataclass
class KVTransferConfig:
    role: str                            # "producer" / "consumer"
    host_ip: str
    port: int
    side_channel_port: int               # ZMQ 通知端口
    channel_number: int = 8              # jax.experimental.transfer 并发通道数
    transfer_size_bytes: int = 256 * 1024 * 1024
    max_parallel_copies: int = 8

@dataclass
class TransferRequest:
    """单次 P→D 传输请求 (P 侧发起)"""
    uuid: bytes                          # 唯一 ID
    block_indices: jax.Array            # P pool 内 indices
    remote_host: str                     # D 节点 host
    remote_port: int                     # D 节点 port

@dataclass
class TransferStatus:
    uuid: bytes
    state: str                           # "pending" / "in_progress" / "done" / "failed"

class KVTransferEngine(ABC):
    """
    PD 分离的 KV 传输抽象 (P2P-routed sync transport).
    被 PD 用 (RFC-2).
    本 RFC 仅定义接口; RFC-2 提供基于 jax.experimental.transfer 的 backend.

    设计要点:
    - P 侧 await_pull(uuid, kv_data); D 侧 connect + pull(uuid, spec)
    - 用 bytes uuid 做路由, 不用内容 hash
    - 同步语义 (P 等 D pull, 超时后释放)
    """

    @abstractmethod
    def register_runner(self, runner: "ModelRunner") -> None:
        """注入 model runner (提供 kv pool, mesh, sharding)"""

    @abstractmethod
    def start(self) -> None:
        """启动传输 server (启动后可接收 await_pull / pull 请求)"""

    @abstractmethod
    def shutdown(self) -> None: ...

    # ===== Producer 侧 =====
    @abstractmethod
    def await_pull(self, uuid: bytes, kv_data: jax.Array,
                   timeout_seconds: float = 180) -> TransferStatus:
        """P 侧: 暴露 kv_data, 等待 D 来 pull"""

    # ===== Consumer 侧 =====
    @abstractmethod
    def connect(self, remote_host: str, remote_port: int) -> "Connection":
        """D 侧: 建立到 P 的连接"""

    @abstractmethod
    def pull(self, conn: "Connection", uuid: bytes,
             kv_spec: list[jax.ShapeDtypeStruct]) -> list[jax.Array]:
        """D 侧: 拉取 P 上 uuid 对应的 kv. 返回的 array 落在 device sharding 上"""

    # ===== 通知 =====
    @abstractmethod
    def notify_pull_done(self, uuid: bytes, target_host: str,
                         target_port: int) -> None:
        """D 拉取完成后, 通过 side channel 通知 P 释放 buffer"""
```

### 11.2 与 ChunkCache / UnifiedRadixCache 都兼容

`KVTransferEngine` 接口不依赖任何 tree_cache 类型——P 侧从 `req_to_token_pool` 直接取出 block_indices 调 `await_pull`；D 侧 pull 后直接写 paged KV cache。这保证了 ChunkCache + PD 和 UnifiedRadixCache + PD 都能跑（兑现 §1.3 矩阵）。

### 11.3 Bootstrap 协议

`KVTransferEngine` 本身不包含 bootstrap server——bootstrap（P-D 路由握手、UUID 协商）属于 RFC-2 的范围。RFC-0 仅定义传输引擎接口。

---

## 12. 详细设计：BasePrefixCache HiCache hooks 完整签名

### 12.1 现状

`python/sgl_jax/srt/mem_cache/base_prefix_cache.py:31` `class BasePrefixCache(abc.ABC)` 已经预留了若干 HiCache 钩子（line 86 `init_load_back`、`ready_to_load_host_cache`、`check_hicache_events`），但都是 `raise NotImplementedError`。

### 12.2 RFC-0 完整化（不实现，只完善签名）

```python
class BasePrefixCache(ABC):
    # ===== 核心 (现有, 不变) =====
    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult: ...
    @abstractmethod
    def insert(self, params: InsertParams) -> InsertResult: ...
    @abstractmethod
    def evict(self, params: EvictParams) -> EvictResult: ...
    @abstractmethod
    def cache_finished_req(self, req, **kwargs) -> None: ...
    @abstractmethod
    def cache_unfinished_req(self, req, chunked, **kwargs) -> None: ...
    @abstractmethod
    def inc_lock_ref(self, node) -> "IncLockRefResult": ...
    @abstractmethod
    def dec_lock_ref(self, node, params) -> None: ...

    # ===== 能力探测 (现有) =====
    def supports_swa(self) -> bool: return False
    def supports_mamba(self) -> bool: return False
    def supports_streaming_session(self) -> bool: return False
    def is_chunk_cache(self) -> bool: return False
    def is_tree_cache(self) -> bool: return False

    # ===== HiCache hooks (RFC-0 完整签名, 默认 NotImplementedError) =====
    def init_load_back(self, params: InitLoadBackParams
                       ) -> Optional[tuple[jax.Array, "TreeNodeBase"]]:
        raise NotImplementedError("HiCache only")

    def ready_to_load_host_cache(self) -> int:
        raise NotImplementedError("HiCache only")

    def check_hicache_events(self) -> None:
        raise NotImplementedError("HiCache only")

    def flush_write_through_acks(self) -> None:
        raise NotImplementedError("HiCache only")

    def take_events(self) -> list:
        return []

    # ===== PD hooks (RFC-2 会用到, RFC-0 留接口 + 工作默认实现) =====
    def get_kv_indices_for_send(self, req) -> jax.Array:
        """
        P 侧用: 给定一个 req, 返回它的 KV indices (给 KVTransferEngine.await_pull).
        ChunkCache 和 UnifiedRadixCache 都继承此默认实现, 无需 override.
        """
        # 默认实现 (适用于所有 BasePrefixCache 子类):
        # 直接从 req_to_token_pool 读 [0, computed_tokens) 的 indices
        # return self.req_to_token_pool.req_to_token[
        #     req.req_pool_idx, :req.computed_tokens
        # ]
        raise NotImplementedError("Implemented in BasePrefixCache concrete subclass init")

    def insert_received_kv(self, req, kv_indices: jax.Array) -> None:
        """
        D 侧用: 收到 P 传来的 KV 后处理.
        ChunkCache: 仅记录 indices, 不做 tree 操作.
        UnifiedRadixCache: 仅写 pool, tree 插入由后续 cache_finished_req 触发.
        默认实现 = 空操作, 因为 KV 已经在 pool 中, 不需要 tree_cache 介入.
        """
        return  # 默认 no-op
```

### 12.3 PD hooks 设计要点

`get_kv_indices_for_send` 和 `insert_received_kv` 在 ChunkCache 和 UnifiedRadixCache 上的实现：

| | ChunkCache | UnifiedRadixCache |
|---|---|---|
| `get_kv_indices_for_send` | 直接读 `req_to_token_pool[req_pool_idx]` | 同左（PD 不走 tree） |
| `insert_received_kv` | 仅记录 indices；后续走 `cache_finished_req` 写 pool | 仅写 pool；tree 插入由 D 完成 decode 后 `cache_finished_req` 触发 |

这与 sglang 的设计一致：PD 主路径绕过 tree_cache（参见 sglang 调研 §3.6.2）。

---

## 13. 配置项扩展（server_args）

### 13.1 新增配置

| 配置项 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `enable_hierarchical_cache` | bool | False | RFC-1 启用 |
| `hicache_ratio` | float | 2.0 | host pool = device pool 的 ratio |
| `hicache_size_gb` | float | None | 显式 host pool 大小（GB），覆盖 ratio |
| `hicache_write_policy` | str | "write_through" | write_through / write_through_selective / write_back |
| `hicache_storage_backend` | str | None | "file" / 未来 "gcs"。None 表示不启用 L3 |
| `hicache_storage_prefetch_policy` | str | "best_effort" | best_effort / wait_complete / timeout |
| `hicache_storage_backend_extra_config` | str (JSON) | "{}" | backend 特定配置 |
| `disaggregation_mode` | str | "null" | null / prefill / decode（RFC-2 启用） |
| `disaggregation_transfer_backend` | str | "jax_transfer" | "jax_transfer" / 未来扩展 |
| `disaggregation_bootstrap_port` | int | 8998 | bootstrap server 端口（RFC-2） |
| `disaggregation_kv_transfer_port` | int | 9100 | KV 传输 server 端口 |

### 13.2 兼容性检查

```python
def _handle_cache_compatibility(self):
    # 已有的检查 (sgl-jax 现有)
    assert not (self.disable_radix_cache and self.enable_hierarchical_cache), \
        "HiCache requires prefix-tree cache (disable_radix_cache must be False)"

    # RFC-0 新增
    if self.disaggregation_mode != "null":
        assert self.disaggregation_transfer_backend in ("jax_transfer",), \
            f"Unknown PD transfer backend: {self.disaggregation_transfer_backend}"
```

---

## 14. 测试策略

### 14.1 现有测试迁移清单

| 现有测试 | 迁移到 | 改动说明 |
|---|---|---|
| `test/mem_cache/test_radix_cache.py` | `test_unified_radix_cache_full.py` | 替换 `RadixCache(...)` 为 `UnifiedRadixCache(CacheInitParams(tree_components=[FullComponent(...)]))`。**断言保持不变**——FullComponent 的语义必须等价于原 RadixCache |
| `test/mem_cache/test_swa_radix_cache.py` | `test_unified_radix_cache_swa.py` | 替换 `SWARadixCache(...)` 为 `UnifiedRadixCache(...components=[Full, SWA]...)`。断言保持不变 |
| `test/mem_cache/test_chunk_cache.py` | 不变 | ChunkCache 保留 |

### 14.2 RFC-0 新增测试矩阵

| 测试文件 | 内容 |
|---|---|
| `test_unified_radix_cache_full.py` | FullComponent 语义验证（match_prefix / insert / evict / lock_ref） |
| `test_unified_radix_cache_swa.py` | SWAComponent 语义验证（双 LRU、tombstone、UUID 锁） |
| `test_unified_radix_cache_mamba.py` | RecurrentStateComponent 语义验证（CoW、single-node lock） |
| `test_unified_radix_cache_hybrid_swa.py` | FULL + SWA 组合验证 |
| `test_unified_radix_cache_hybrid_ssm.py` | FULL + RECURRENT_STATE 组合验证 |
| `test_kv_cache_builder.py` | builder dispatch 矩阵验证（每种 server_args 组合得到正确的 tree_cache 类型） |
| `test_base_prefix_cache_hicache_hooks.py` | HiCache hooks 签名 + ChunkCache/UnifiedRadixCache 默认行为验证 |
| `test_memory_pools_extension.py` | host_pool slot 的 pytree 序列化/反序列化 |
| `test_abc_signatures.py` | HostKVPool / KVCacheController / HostMemoryAllocator / HiCacheStorage / KVTransferEngine 五个 ABC 的接口签名稳定性测试（防止后续 PR 改坏 ABC） |

### 14.3 端到端兼容性测试矩阵（ADR-6 兜底）

| 场景 | 测试用例 | 期望 |
|---|---|---|
| 默认（无 HiCache 无 PD） | 现有 `test/srt/test_*` 全部跑通 | 0 退化 |
| Chunked prefill | 现有测试 | 0 退化（ChunkCache 路径） |
| Overlap schedule | 现有测试 | 0 退化 |
| DP | 现有测试 | 0 退化 |
| Mixed chunk | 现有测试 | 0 退化 |
| Retract | 现有测试 | 0 退化 |
| Partial rollout | 现有测试 | 0 退化（保留 `KVCache.get_cpu_copy/load_cpu_copy`） |
| SWA 模型（**MiMo-V2-Flash / MiMo-V2-Pro**） | 模型推理测试 | 走 UnifiedRadixCache + SWAComponent，输出等价 |
| Recurrent-State 模型（KDA / Bailing-Linear） | 模型推理测试 | 走 UnifiedRadixCache + RecurrentStateComponent，输出等价 |
| MLA 模型（DeepSeek） | 模型推理测试 | 走 UnifiedRadixCache + FullComponent（MLA 在 FullComponent 内部处理），输出等价 |

### 14.4 测试运行环境

- 单元测试：local CPU JAX backend 即可（不需要 TPU）
- 端到端测试：v6e-4 pod 跑全套模型推理（依赖 [[v6e_pod_paths]]）

---

## 15. 风险与未决问题

### 15.1 已识别风险

| Risk | 影响 | 缓解 |
|---|---|---|
| **R1: sglang origin/main port 工作量** | UnifiedRadixCache 1960 行 + 3 component 共 ~1300 行 = ~3300 行 Python 翻译 | 分组件 port，每个 component 单独验证；先 port FullComponent（最简），再 SWA，最后 Mamba |
| **R2: sglang upstream 持续迭代** | 在 RFC-0 落地期间 sglang 可能继续修 HiCache bug | 锁定一个 sglang commit hash 做 port，落地后再 rebase；后续以 PR 形式跟进 upstream fix |
| **R3: 删除 RadixCache 破坏未发现的依赖** | 用户脚本 / 第三方代码可能 import RadixCache | grep 全仓 + 提供 deprecation warning 一个版本后再删 |
| **R4: HiCache 钩子在 ChunkCache 上的默认行为** | ChunkCache 不支持 HiCache，需要明确报错 | BasePrefixCache 默认 hook = NotImplementedError + builder 中 assert |
| **R5: ABC 设计与未来 RFC-1/2 不匹配** | RFC-0 ABC 写完后 RFC-1/2 发现需要返工 | RFC-1/2 brainstorm 时优先验证 ABC；如有缺失允许小幅扩展 ABC（不破坏 RFC-0 主体） |
| **R6: MatchResult 新字段对外暴露** | scheduler / schedule_policy 等代码读 MatchResult，可能漏改 | 加 mypy 类型检查 + grep MatchResult 调用点 |

### 15.2 未决问题（不阻塞 RFC-0 完成，但需要在 RFC-1/2 时解决）

| 问题 | 何时回答 |
|---|---|
| HiCache 的 host pool 总大小如何在 ratio vs absolute size 之间确定？ | RFC-1 §「容量配置」 |
| PD bootstrap server 是 HTTP 还是 ZMQ？ | RFC-2 §「Bootstrap 协议」 |
| PD 多主机部署用 Ray 还是 K8s Service？ | RFC-2 §「部署」 |
| L3 file backend 的 SHA256 key scheme 与未来 backend 的兼容性 | RFC-1 §「Key 设计」 |
| RFC-1 是否需要 layer-wise overlap 的 step-level 替代？ | RFC-1 §「异步策略」 |

---

## 16. 实施路线

### 16.1 阶段拆分（M0 → M3）

```
┌─────────────────────────────────────────────────────────────────┐
│ M0: 基础设施 + ABC 落地 (无功能变化)                              │
├─────────────────────────────────────────────────────────────────┤
│ ✓ 新增 base_prefix_cache.py 中的 HiCache + PD hooks 签名         │
│ ✓ 新增 host_kv_pool.py / host_memory_allocator.py /              │
│   cache_controller.py / hicache_storage.py /                     │
│   disaggregation/kv_transfer_engine.py 全部 ABC                  │
│ ✓ 扩展 MemoryPools 容器 (host_pool slot)                         │
│ ✓ 新增 kv_cache_builder.py 框架 (暂时只 dispatch ChunkCache +     │
│   现有 RadixCache, 不引入 UnifiedRadixCache)                     │
│ ✓ scheduler.py 接入 kv_cache_builder                              │
│ ✓ 新增 test_abc_signatures.py / test_kv_cache_builder.py /        │
│   test_memory_pools_extension.py                                  │
│                                                                  │
│ 验收: 全部现有端到端测试 0 退化 (功能等价, 仅重构入口)            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ M1: UnifiedRadixCache + FullComponent (与 RadixCache 并存)        │
├─────────────────────────────────────────────────────────────────┤
│ ✓ Port unified_radix_cache.py 主体 (~1960 行)                    │
│ ✓ Port tree_component.py ABC                                     │
│ ✓ Port full_component.py                                         │
│ ✓ kv_cache_builder 新增 UnifiedRadixCache dispatch                │
│ ✓ env var SGL_JAX_USE_LEGACY_RADIX_CACHE 控制 dispatch 优先级:    │
│   - 默认 False: 走 UnifiedRadixCache                              │
│   - True: 走 legacy RadixCache                                    │
│ ✓ radix_cache.py 保留 (legacy fallback)                          │
│ ✓ test_radix_cache.py 保留, 新增 test_unified_radix_cache_full.py │
│ ✓ 默认非 SWA / 非 Mamba 模型 (Llama / Qwen 等) 走 UnifiedRadixCache │
│                                                                  │
│ 验收:                                                            │
│  • Llama / Qwen 推理输出等价 (KL test)                            │
│  • UnifiedRadixCache 与 legacy RadixCache 切换功能等价            │
│  • env var True/False 都通过端到端测试                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ M2: SWAComponent + RecurrentStateComponent (与 SWARadixCache 并存)│
├─────────────────────────────────────────────────────────────────┤
│ ✓ Port swa_component.py                                          │
│ ✓ Port recurrent_state_component.py                              │
│ ✓ kv_cache_builder 新增 SWA / RecurrentState dispatch            │
│ ✓ swa_radix_cache.py 保留 (legacy fallback,                       │
│   env var SGL_JAX_USE_LEGACY_RADIX_CACHE 同时控制 SWA 路径)       │
│ ✓ 新增 test_unified_radix_cache_swa.py /                          │
│   test_unified_radix_cache_recurrent.py                           │
│ ✓ 新增 test_unified_radix_cache_hybrid_*                          │
│                                                                  │
│ 验收:                                                            │
│  • MiMo-V2-Flash / MiMo-V2-Pro (SWA) 推理输出等价                │
│  • KDA / Bailing-MoE-Linear (Recurrent-State) 推理输出等价        │
│  • 端到端兼容性矩阵 (§14.3) 全部通过                              │
│  • legacy SWARadixCache 路径仍可用 (env var True)                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ M3: Cleanup — 删除 legacy RadixCache / SWARadixCache              │
├─────────────────────────────────────────────────────────────────┤
│ 触发条件 (M2 稳定 N 个 release 之后):                              │
│  • 0 production incident 报告与 UnifiedRadixCache 相关             │
│  • 端到端 KL test 在所有支持模型上跑通 ≥ 2 个 release             │
│  • 没有用户依赖 SGL_JAX_USE_LEGACY_RADIX_CACHE=true 的反馈        │
│                                                                  │
│ 内容:                                                            │
│  ✓ 删除 radix_cache.py / swa_radix_cache.py                      │
│  ✓ 删除 SGL_JAX_USE_LEGACY_RADIX_CACHE env var                    │
│  ✓ kv_cache_builder 简化 dispatch (移除 legacy 分支)              │
│  ✓ 删除 test_radix_cache.py / test_swa_radix_cache.py             │
│  ✓ 文档清理                                                       │
│                                                                  │
│ 验收: 仍然 0 端到端退化                                          │
└─────────────────────────────────────────────────────────────────┘

M0 → M1 → M2 完成后, RFC-0 主体功能落地. RFC-1 / RFC-2 可以开始.
M3 是独立的清理阶段, 与 RFC-1 / RFC-2 实施无依赖关系.
```

### 16.2 不在 RFC-0 范围内的工作（明确划清）

| 工作 | 归属 RFC |
|---|---|
| HostKVPool 的 TPU 具体实现（用 pinned_host） | RFC-1 |
| KVCacheController 的 TPU 具体实现（用 jax.device_put + ThreadPoolExecutor） | RFC-1 |
| File-based HiCacheStorage 实现 | RFC-1 |
| L3 storage prefetch 流水线 | RFC-1 |
| HiCache 在 SWA / Mamba 上的 backup_storage / prefetch | RFC-1 |
| KVTransferEngine 的 jax.experimental.transfer 实现 | RFC-2 |
| Bootstrap server（HTTP or ZMQ） | RFC-2 |
| Prefill / Decode mixin（SchedulerDisaggregationPrefillMixin 等价物） | RFC-2 |
| PD multi-host 部署 | RFC-2 |

---

## 17. 附录

### 17.1 sglang origin/main 关键源码索引（port 时参考）

| 文件 | 行数 | port 到 sgl-jax |
|---|---|---|
| `python/sglang/srt/mem_cache/unified_radix_cache.py` | 1960 | `python/sgl_jax/srt/mem_cache/unified_radix_cache.py` |
| `python/sglang/srt/mem_cache/unified_cache_components/tree_component.py` | 364 | 同名 |
| `python/sglang/srt/mem_cache/unified_cache_components/full_component.py` | 282 | 同名 |
| `python/sglang/srt/mem_cache/unified_cache_components/swa_component.py` | 537 | 同名 |
| `python/sglang/srt/mem_cache/unified_cache_components/recurrent_state_component.py` | 448 | 同名 |
| `python/sglang/srt/mem_cache/base_prefix_cache.py` | — | 已存在，扩展 |
| `python/sglang/srt/mem_cache/hicache_storage.py` | — | 已存在的接口部分 port 到新文件 |
| `python/sglang/srt/mem_cache/kv_cache_builder.py` | 318 | `python/sgl_jax/srt/mem_cache/kv_cache_builder.py` |

### 17.2 tpu-inference 接口风格参考

| tpu-inference 文件 | 用于参考的部分 |
|---|---|
| `tpu_inference/offload/cpu_backend.py` | `LocalCPUBackend`（OrderedDict + LRU），后续 RFC-1 实现 HiCacheStorage file backend 时参考 |
| `tpu_inference/distributed/host_kv_pool.py` | `HostKVPool` (queue + pinned host)，后续 RFC-2 实现 KVTransferEngine 时参考 |
| `tpu_inference/distributed/tpu_connector.py` | `start_transfer_server` / `await_pull` / `connect` / `pull` 的封装模式 |
| `tpu_inference/offload/tpu_offload_connector.py` | `register_runner` / `host_sharding` 构造逻辑 |

### 17.3 sglang Issue / PR 关注列表（跟进 upstream）

- Issue #20415: Unified Hybrid Radix Cache Refactor Roadmap
- PR #23316: HiCache Framework for UnifiedRadixTree
- PR #23391: SWA HiCache
- PR #24585: Eviction fix
- PR #24691: DeepSeek V4 HiCache (in progress)
- PR #24972: Tombstone lock fix
- PR #25088: Load back start node fix
- PR #25277: Device match semantics fix
- L3 support for UnifiedRadixCache（未排期）
- PD/Spec 兼容性（未排期）

### 17.4 相关文档

- `docs/research/2026-05-18-sglang-cache-pd-organization.md` — sglang 组织结构调研
- `docs/research/2026-05-18-tpu-inference-jax-api-survey.md` — tpu-inference JAX API 调研
- `RFC-1` (待写) — HiCache 详细设计
- `RFC-2` (待写) — PD 分离详细设计

---

**End of RFC-0**
