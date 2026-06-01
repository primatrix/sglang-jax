# RFC-1: HiCache — TPU/JAX 上的多级 KV 缓存

**Status**: Draft  
**Author**: john  
**Date**: 2026-05-18  
**Depends on**: RFC-0 (UnifiedRadixCache + KV 缓存与传输基础设施)  
**Related**: RFC-2 (PD 分离)  
**Prerequisite reading**: `docs/rfc/2026-05-18-rfc-0-unified-cache-and-kv-infra.md`、`docs/research/2026-05-18-sglang-cache-pd-organization.md`、`docs/research/2026-05-18-tpu-inference-jax-api-survey.md`

---

## 1. 摘要 & 动机

### 1.1 目标

为 sgl-jax 实现「多级 KV 缓存」（HiCache），用 TPU/JAX 原语填充 RFC-0 定义的 ABC：

1. `LRUHostKVPool`：HiCache L2 host pool，`HostKVPool` 的 LRU 实现，用 `NamedSharding(memory_kind="pinned_host")` 持有 pinned host memory
2. `HostMemoryAllocatorImpl`：host pool 的 token-level indices 分配器
3. `TPUKVCacheController`：D2H/H2D 调度（**仅 L1↔L2**），用 `jax.device_put` + `ThreadPoolExecutor` 做异步搬运
4. `UnifiedRadixCache.init_hicache()` 在 sgl-jax 上的接入逻辑（已由 RFC-0 §4 port）

**L3（外部存储 / 持久化）只保留 RFC-0 的 `HiCacheStorage` ABC 接口签名，RFC-1 不提供任何具体实现**。这是为了：
- 第一版聚焦在 L1↔L2 这个生产价值最大的路径（命中减少 prefill 计算）
- 避免引入对本地文件系统 / 数据库 / 远端存储的依赖
- 等 L1↔L2 稳定 + 用户实际有 L3 需求后再设计具体 backend

### 1.2 非目标

- ❌ **不实现任何 L3 backend**（文件 / GCS / mooncake / NIXL / aibrix / hf3fs / eic / simm 全部不做）；`HiCacheStorage` ABC 仅作为 RFC-0 接口预留，RFC-1 内 `TPUKVCacheController.storage_backend` 字段恒为 `None`
- ❌ 不实现 layer-wise H2D overlap（XLA 静态编译不支持，已在 jax-api 调研 §6.2 论证）
- ❌ 不实现 LMCache 集成
- ❌ 不在 sgl-jax 的 ChunkCache 上加 HiCache（HiCache 仅在 UnifiedRadixCache 路径生效，参见 RFC-0 §1.3 矩阵）
- ❌ 不做 sglang 的「write_through_selective」策略（YAGNI——只做 write_through + write_back）
- ❌ 不实现 PD 相关功能（属 RFC-2）

### 1.3 范围矩阵

| 子系统 | RFC-0 范围 | **RFC-1 范围** | RFC-2 范围 |
|---|---|---|---|
| HostKVPool ABC | ✓ 接口 | **TPU LRU 实现** | TPU Queue 实现 |
| HostMemoryAllocator ABC | ✓ 接口 | **实现** | — |
| KVCacheController ABC | ✓ 接口 | **TPU 实现（仅 L1↔L2）** | — |
| HiCacheStorage ABC | ✓ 接口 | **不实现 backend**（接口预留） | — |
| UnifiedRadixCache.init_hicache | ✓ port 自 sglang | **JAX 适配 + sgl-jax 集成** | — |
| 写入策略 | — | **write_through + write_back（仅 L1↔L2）** | — |
| 读取/加载路径 | — | **同步阻塞 + step-level async load（仅 L1↔L2）** | — |
| 异步模型 | — | **ThreadPoolExecutor + jax.device_put** | — |
| SPMD / DP attention 一致性 | RFC-0 §3.5 | **per-DP host pool + dp_rank-aware hash key** | — |
| L3 prefetch / backup_storage | — | **不实现**（接口留空，方法 raise NotImplementedError） | — |
| KV transfer (跨进程) | — | — | ✓ |
| Bootstrap server | — | — | ✓ |

---

## 2. 决策记录（ADR）

### ADR-1: 用 `jax.device_put` + `pinned_host` 实现 D2H/H2D，不用 Pallas

| | |
|---|---|
| **决策** | HiCache 的 D2H（HBM→host）和 H2D（host→HBM）统一用 `jax.device_put(arr, sharding_with_memory_kind)`。不引入 Pallas `copy_to_host` / `pltpu.HOST` 等自定义 kernel。 |
| **理由** | (1) `jax.device_put` 跨平台、API 稳定、自动管理 device-host affinity（参见 jax-api 调研 §3）；(2) HiCache 是后台异步操作，对每毫秒延迟不敏感（不像 PD），Pallas 优化收益小；(3) Pallas 适合"预分配 buffer + 系统调用敏感"场景（PD 用），HiCache 用 OrderedDict 长期持有 `jax.Array`，handle 复用本身就避免了系统调用；(4) Pallas 路径需要更多约束（如固定 layout），会限制 HostKVPool 设计。 |
| **影响** | 若未来性能 profile 显示 D2H 是瓶颈，可在 `TPUKVCacheController` 内部替换为 Pallas，不影响 ABC 接口。 |
| **替代方案** | Pallas `copy_to_host` 路径（参考 tpu-inference `kv_transfer.py` `copy_to_host`）—— 拒绝，理由如上。 |

### ADR-2: `ThreadPoolExecutor` 作为异步后端，不用 asyncio

| | |
|---|---|
| **决策** | `TPUKVCacheController` 用 `concurrent.futures.ThreadPoolExecutor` 跑后台 D2H/H2D。Python 主线程提交任务后立即返回，通过 `future.done()` 非阻塞轮询。 |
| **理由** | (1) 与 tpu-inference 已有模式一致（`save_executor` / `pull_executor`，参见 jax-api 调研 §5.2）；(2) JAX 的 `jax.device_put` / `jax.block_until_ready` 在 C++ 层释放 GIL，主线程 forward 与后台 D2H 真正并行；(3) asyncio 需要把 scheduler 主循环改成 coroutine，代价大且无收益（JAX 不是 async-native）；(4) sgl-jax scheduler 是同步事件循环（`event_loop_normal`），ThreadPoolExecutor 模型对齐。 |
| **影响** | 需要新增 `save_executor`（D2H）和 `load_executor`（H2D）两个后台线程池。 |
| **替代方案** | asyncio（拒绝）；单一后台线程 + 任务队列（拒绝，因为 ThreadPoolExecutor 已是其 superset 且性能更好）。 |

### ADR-3: 只实现 `write_through` 和 `write_back`，不做 `write_through_selective`

| | |
|---|---|
| **决策** | `--hicache-write-policy` 只接受 `write_through`（默认）和 `write_back`。不实现 sglang 的 `write_through_selective`（基于 hit_count 的选择性备份）。 |
| **理由** | (1) YAGNI——`write_through_selective` 在 sglang 是 host pool 内存受限场景的优化，sgl-jax 早期工作负载下用不上；(2) 实现需要在 RadixCache hit_count 上加阈值判断，与 UnifiedRadixCache port 范围冲突；(3) `write_through_selective` 的核心价值（只备份热点）可以通过 `write_back` 的「驱逐时才备份」自然实现。 |
| **影响** | RFC-1 只需实现两条 D2H 触发路径：`insert()` 时主动备份（write_through）+ `evict()` 时被动备份（write_back）。 |
| **替代方案** | 全部三种策略 —— 拒绝，YAGNI。 |

### ADR-4: 不做 layer-wise H2D overlap，只做 step-level async load

| | |
|---|---|
| **决策** | H2D（CPU→HBM）的传输与 forward 计算的重叠粒度是「整个请求一次性 load」，不做 sglang 的 per-layer event 重叠。下一步 batch 的 load 与当前 batch 的 forward 并行。 |
| **理由** | (1) XLA 静态编译使 layer-wise event 不可行（jax-api 调研 §6.2 已论证）；(2) Step-level async load 已能隐藏大部分 H2D 延迟（在请求间）；(3) 改造 jit 后的 forward 拆 layer 代价巨大且收益不确定。 |
| **影响** | `BasePrefixCache.ready_to_load_host_cache()` 返回 load 触发数量；具体 load 在主线程 forward 之前发起 + 等待。后续 batch 的 load 可与当前 batch forward 并行。 |
| **替代方案** | Layer-wise overlap（拒绝，XLA 不支持）；完全同步阻塞（拒绝，损失并行机会）。 |

### ADR-5: 第一版不实现任何 L3 backend

| | |
|---|---|
| **决策** | RFC-1 **不实现任何 `HiCacheStorage` 具体 backend**（包括 file backend）。`HiCacheStorage` ABC 由 RFC-0 §10 定义为接口预留，`TPUKVCacheController.storage_backend` 字段在第一版恒为 `None`。`prefetch_from_storage / write_backup_storage` 等 L3 相关方法在 RFC-1 实现中 `raise NotImplementedError("L3 not implemented in first release")`。 |
| **理由** | (1) **第一版聚焦 L1↔L2**：HBM↔Host DRAM 是 KV 多级缓存的核心生产价值（多轮对话/共享 system prompt 命中显著降 TTFT），L3 是锦上添花；(2) **TPU 上没有像 GPUDirect Storage 这种成熟方案**：file 是本地文件系统，依赖单机；mooncake/NIXL 等是 GPU 专用；GCS 等远端存储延迟高、需要异步预取设计才有价值——这些都需要更多设计周期；(3) **避免引入不必要依赖**：file backend 要管文件锁、容量清理、损坏恢复；远端 backend 要管认证、网络重试、一致性；这些都不应该在 L2 还没稳定时引入；(4) **接口预留足够**：`HiCacheStorage` ABC + `compute_node_hash` 工具（含 dp_rank, ADR-8）已经足够支撑未来加任何 backend，第一版后用户提出真实 L3 需求时再设计具体实现。 |
| **影响** | (1) RFC-1 删除原计划的 `LocalFileStorage` 实现 + 相关测试 + 实施路线 H2 阶段；(2) `kv_cache_builder._build_hicache_infra` 不创建 storage_backend；(3) `--hicache-storage-backend` 配置项保留但默认/唯一可选值是 `None`（其他值 raise）；(4) 文档明确「L3 是未来扩展点，需另开 RFC」。 |
| **替代方案** | (a) 实现 file backend 作为 toy 参考：拒绝，引入文件系统依赖且 toy backend 几乎没有生产价值；(b) 实现 GCS backend：拒绝，太重，应等需求明确再做；(c) 实现 in-memory `DictStorage` 作为测试替身：可以，但单元测试用 mock 即可，不需要正式 backend。 |
| **未来扩展** | L3 需求出现时，按 sglang storage backend 风格新增子目录 `python/sgl_jax/srt/mem_cache/storage/<backend>/`，实现 `HiCacheStorage` ABC，注册到 `kv_cache_builder._build_hicache_infra`。RFC-0 §10 接口 + ADR-8 hash scheme 不需改动。 |

### ADR-6: SPMD 一致性用每进程独立 HiCache（不跨进程协调）

| | |
|---|---|
| **决策** | sgl-jax 在多 host 部署下，每个 JAX 进程（`jax.process_count() > 1` 时一进程一 host）独立维护自己的 HiCache：自己的 `LRUHostKVPool`、自己的 `TPUKVCacheController`、自己的 LRU 元数据。**不跨进程协调**。`TPUKVCacheController` 跑在 scheduler 主进程（与 ModelWorker 同进程，因为 sgl-jax 的 ModelWorker 是 thread/class，不是 mp.Process）。 |
| **理由** | (1) sgl-jax 的 ModelWorker 在 scheduler 进程内（`tp_worker.py` `class ModelWorker`，非独立进程），D2H/H2D 的 `jax.device_put` 可以直接在 scheduler 进程内对 `kv_pool` 操作，不需要 IPC；(2) 多 host 部署时每个进程的 device kv_pool shard 是 host-local（"data" / "tensor" 轴沿 device sharding），对应的 pinned_host buffer 也 host-local，跨进程数据共享没意义；(3) sglang 用 all_reduce 是因为它每个 TP rank 一进程，每进程独立持有 RadixCache 实例，需要保证 tree 一致——sgl-jax 是单 scheduler，不存在这个问题；(4) 多 host 下每进程独立做 HiCache 等价于 sglang 的「每 TP rank 一个独立 host pool」（参见 sglang 调研 §2.2「TP 场景下的 L1→L2 数据流」），sglang 也没有跨 rank 共享 host pool。 |
| **影响** | `TPUKVCacheController` 不需要 ZMQ 协调；多 host 部署时每进程的 host pool 容量独立计算（每个 rank 持有 device pool shard 的 ratio×）。 |
| **替代方案** | 跨进程共享 host pool（拒绝，因为 sglang 也没做，pinned host memory 不容易跨进程共享）；leader+broadcast 模式（拒绝，没必要，每进程独立即可）。 |

### ADR-7: Host pool 容量配置：默认 `hicache_ratio=2.0`，可被 `hicache_size_gb` 覆盖

| | |
|---|---|
| **决策** | 与 sglang 一致。默认 host pool = device kv pool × 2。`--hicache-size-gb` 显式指定时覆盖 ratio。 |
| **理由** | sglang 的 2× 比例在大量生产场景验证过；用户调参方便。 |
| **影响** | `_build_hicache_infra` 计算公式：`host_pool_tokens = device_pool_tokens × ratio`。 |
| **替代方案** | 仅支持绝对 size——拒绝，ratio 是用户更直观的表达。 |

### ADR-8: hash key scheme (供未来 L3 backend 用): SHA256(prefix_token_ids + tp_rank + tp_size + dp_rank + model_name)

| | |
|---|---|
| **决策** | RFC-1 实装 `compute_node_hash(prefix_token_ids, tp_rank, tp_size, dp_rank, model_name, is_mla)` 工具函数（在 `python/sgl_jax/srt/mem_cache/storage/utils.py`），但**RFC-1 自身不使用它**（因为不实现 L3 backend）。**专为未来 L3 backend 预留**，提前定型避免后续兼容性问题。**对于 MLA 模型**，key 不含 tp_rank；dp_rank 始终包含（sgl-jax SPMD DP attention 各 rank attention 切分不同）。 |
| **理由** | (1) hash scheme 是跨 backend 必须一致的协议（同 prefix 跨 backend 命中必须 hash 相同），所以即使第一版不用，也要提前定型；(2) sgl-jax 在单进程内有多个 DP rank（参见 RFC-0 §3.5），如果 key 不含 dp_rank，未来不同 DP rank 的 host pool 会用同一 hash 找到对方的数据，导致数据损坏；(3) 函数定型 + 单元测试覆盖（hash 稳定性、MLA 跳过 tp_rank、dp_rank 必含）后，未来加 backend 时直接调用即可。 |
| **影响** | `compute_node_hash` 单元测试在 RFC-1 H2 阶段做（验证 hash 稳定性）；`HiCacheStorageConfig.is_mla_model` / `dp_rank` 字段在 RFC-0 中已预留。 |
| **替代方案** | 不实装 hash 工具，等 L3 backend 加入时再做—— 拒绝，因为 hash 是跨 backend 协议，提前定型可避免未来兼容性 bug。 |

---

## 3. 模块全景

### 3.1 模块依赖图

```
                ┌────────────────────────────────────────────┐
                │  UnifiedRadixCache (port from sglang)       │
                │  (RFC-0 §4 实装)                            │
                │                                              │
                │  HiCache 钩子全部齐:                          │
                │   • write_backup / load_back                 │
                │   • writing_check / loading_check            │
                │   • init_load_back                           │
                │   • check_hicache_events                     │
                │   • flush_write_through_acks                 │
                │   • ready_to_load_host_cache                 │
                │                                              │
                │  init_hicache() 调用 _build_hicache_infra     │
                │  注入 host_kv_pool + cache_controller         │
                └────────────────────┬────────────────────────┘
                                     │
                          ┌──────────┴──────────┐
                          │                     │
              ┌───────────▼─────────┐   ┌──────▼─────────────────┐
              │ LRUHostKVPool §4    │   │ TPUKVCacheController §6 │
              │ (HostKVPool 实现)    │   │ (KVCacheController 实现) │
              │                     │   │                        │
              │ • OrderedDict       │   │ • save_executor        │
              │   [hash, handle]    │   │   (ThreadPoolExecutor) │
              │ • LRU 淘汰          │   │ • load_executor        │
              │ • lock_ref 计数     │   │ • write() / load()     │
              │ • alloc/free        │   │ • writing_check /      │
              │ • evict             │   │   loading_check        │
              │ • per-DP 分区        │   │ • storage_backend=None │
              │   (dp_rank)          │   │   (L3 第一版不实装)     │
              │                     │   └────────────────────────┘
              │ 持有的 jax.Array:    │
              │ NamedSharding(      │
              │   memory_kind=      │
              │   "pinned_host")    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │HostMemoryAllocator   │
              │Impl §5              │
              │ (free-list based)   │
              └─────────────────────┘

   ─── L3 (HiCacheStorage backend) 第一版不实装, 仅 ABC 预留 (§7) ───
   未来加入时: 新增 storage/<backend>/ 子目录 + 注册到 builder
```

### 3.2 文件路径

| 路径 | 操作 | 行数估计 |
|---|---|---|
| `python/sgl_jax/srt/mem_cache/host_kv_pool.py` | RFC-0 提供 ABC；**新增 `LRUHostKVPool` 实现** | +400 |
| `python/sgl_jax/srt/mem_cache/host_memory_allocator.py` | RFC-0 提供 ABC；**新增 `FreeListHostMemoryAllocator` 实现** | +200 |
| `python/sgl_jax/srt/mem_cache/cache_controller.py` | RFC-0 提供 ABC；**新增 `TPUKVCacheController` 实现** | +500 |
| `python/sgl_jax/srt/mem_cache/storage/__init__.py` | 新增 (空 placeholder, 未来 L3 backend 用) | +20 |
| `python/sgl_jax/srt/mem_cache/storage/utils.py` | 新增 `compute_node_hash` (供未来 backend 用) | +80 |
| `python/sgl_jax/srt/mem_cache/kv_cache_builder.py` | RFC-0 提供框架；**完善 `_build_hicache_infra()` (不创建 storage_backend)** | +100 |
| `python/sgl_jax/srt/managers/scheduler.py` | **加 check_hicache_events / ready_to_load_host_cache / flush_write_through_acks 集成点** | +50 |
| `python/sgl_jax/srt/server_args.py` | RFC-0 添加配置项；**加默认值 + 参数解析 + `--hicache-storage-backend` raise if not None** | +30 |
| `python/sgl_jax/test/mem_cache/test_lru_host_kv_pool.py` | 新增 | +300 |
| `python/sgl_jax/test/mem_cache/test_tpu_cache_controller.py` | 新增 | +400 |
| `python/sgl_jax/test/mem_cache/test_hicache_storage_hash.py` | 新增 (hash 工具单元测试, 供未来 backend) | +100 |
| `python/sgl_jax/test/srt/test_hicache_e2e.py` | 新增 (仅 L1↔L2 路径) | +300 |

总计新增约 2480 行（比原计划少 400 行——去掉 L3 backend + 测试）。

---

## 4. 详细设计：LRUHostKVPool

### 4.1 设计目标

`LRUHostKVPool` 是 `HostKVPool` ABC（RFC-0 §7）的 LRU 实现，对应 sglang `MHATokenToKVPoolHost` + tpu-inference `LocalCPUBackend` 的混合：
- 接口形态像 sglang（token-indexed alloc/free + lock_ref + evict）
- 实现风格像 tpu-inference（用 `jax.device_put` 到 `pinned_host`，OrderedDict 管理）

### 4.2 数据结构

```python
# python/sgl_jax/srt/mem_cache/host_kv_pool.py (RFC-0 ABC 之后 append)

from collections import OrderedDict
import jax
from jax.sharding import NamedSharding, PartitionSpec as P

class LRUHostKVPool(HostKVPool):
    """
    HiCache L2 host pool. LRU 淘汰策略.

    内存布局:
    - 单一 jax.Array 作为 underlying buffer (NamedSharding, memory_kind="pinned_host")
    - shape: [host_total_tokens, layer_num, kv_head_per_rank, 2, head_dim]
    - 通过 host_indices (jax.Array of int32) 切片访问
    """

    def __init__(
        self,
        host_total_tokens: int,
        layer_num: int,
        kv_head_per_rank: int,
        head_dim: int,
        dtype: jnp.dtype,
        mesh: jax.sharding.Mesh,
        partition_spec: PartitionSpec,
    ):
        self.host_total_tokens = host_total_tokens
        self.layer_num = layer_num
        self.kv_head_per_rank = kv_head_per_rank
        self.head_dim = head_dim
        self.dtype = dtype
        self.mesh = mesh

        # ===== underlying pinned host buffer =====
        self.host_sharding = NamedSharding(
            mesh,
            partition_spec,
            memory_kind="pinned_host",
        )
        host_shape = (host_total_tokens, layer_num, kv_head_per_rank, 2, head_dim)
        self.buffer: jax.Array = jax.device_put(
            jnp.zeros(host_shape, dtype=dtype),
            self.host_sharding,
        )
        jax.block_until_ready(self.buffer)

        # ===== allocator =====
        self.allocator = FreeListHostMemoryAllocator(host_total_tokens)

        # ===== LRU tracking =====
        # 每个 entry: hash_key (bytes) → host_indices (jax.Array)
        # 同时持有 lock_ref 计数, 防止活跃使用中的 entry 被驱逐
        self.entries: OrderedDict[bytes, "HostEntry"] = OrderedDict()

        # ===== stats =====
        self.num_evictions = 0
        self.num_hits = 0
        self.num_misses = 0

@dataclass
class HostEntry:
    """LRU pool 内的一条记录"""
    hash_key: bytes
    host_indices: jax.Array              # token-level indices into self.buffer
    num_tokens: int
    lock_ref: int = 0                    # >0 表示活跃中, 不可驱逐
    last_access_time: float = 0.0
```

### 4.3 核心方法

```python
class LRUHostKVPool(HostKVPool):

    # ===== HostKVPool ABC 实现 =====
    def alloc(self, num_tokens: int) -> Optional[HostBufferHandle]:
        """
        分配 num_tokens 大小的 host indices.
        如果空间不足, 自动驱逐 (LRU + lock_ref=0).
        失败返回 None.
        """
        indices = self.allocator.alloc(num_tokens)
        if indices is None:
            evicted = self.evict(num_tokens)
            if evicted < num_tokens:
                return None
            indices = self.allocator.alloc(num_tokens)

        # 不立即创建 HostEntry — entry 在 register_with_hash 时创建
        # buffer=None 因为 LRU pool 的 underlying buffer 会被 .at[].set() 重新绑定
        # 调用方必须用 pool.read_indices(handle.indices) 访问数据
        return HostBufferHandle(
            indices=indices,
            buffer_id=-1,                # LRU pool 不用 buffer_id
            buffer=None,                 # LRU 模式无稳定 buffer 引用
        )

    def free(self, handle: HostBufferHandle) -> None:
        self.allocator.free(handle.indices)
        # entry 在 evict() 中清理, free() 只回收 indices

    def available_size(self) -> int:
        return self.allocator.available_size()

    def total_size(self) -> int:
        return self.host_total_tokens

    def evict(self, num_tokens: int) -> int:
        """
        LRU 淘汰. 跳过 lock_ref > 0 的 entry.
        返回实际释放的 tokens 数 (可能 < 要求, 如果 lock_ref 阻塞).
        """
        freed = 0
        keys_to_remove = []
        for hash_key, entry in self.entries.items():  # 已按 LRU 顺序
            if entry.lock_ref > 0:
                continue
            keys_to_remove.append(hash_key)
            freed += entry.num_tokens
            if freed >= num_tokens:
                break

        for k in keys_to_remove:
            entry = self.entries.pop(k)
            self.allocator.free(entry.host_indices)
            self.num_evictions += 1

        return freed

    def lock_ref_inc(self, handle: HostBufferHandle) -> None:
        # 通过 indices 反查 entry (略慢, 但 lock_ref 操作不频繁)
        for entry in self.entries.values():
            if self._indices_match(entry.host_indices, handle.indices):
                entry.lock_ref += 1
                return
        raise ValueError("HostBufferHandle not registered in pool")

    def lock_ref_dec(self, handle: HostBufferHandle) -> None:
        for entry in self.entries.values():
            if self._indices_match(entry.host_indices, handle.indices):
                entry.lock_ref = max(0, entry.lock_ref - 1)
                return

    # ===== LRU 专用 (RFC-1 扩展) =====
    def register_with_hash(self, handle: HostBufferHandle,
                           hash_key: bytes) -> None:
        """alloc 后, 把 indices 关联到 hash_key 并放入 LRU 链表"""
        entry = HostEntry(
            hash_key=hash_key,
            host_indices=handle.indices,
            num_tokens=int(handle.indices.shape[0]),
            lock_ref=0,
            last_access_time=time.monotonic(),
        )
        self.entries[hash_key] = entry
        self.entries.move_to_end(hash_key)  # 加到 LRU 尾 (最新)

    def lookup(self, hash_key: bytes) -> Optional[HostBufferHandle]:
        """查询 hash 是否在 pool 内. 命中则触发 LRU update."""
        entry = self.entries.get(hash_key)
        if entry is None:
            self.num_misses += 1
            return None
        self.num_hits += 1
        self.entries.move_to_end(hash_key)
        entry.last_access_time = time.monotonic()
        return HostBufferHandle(
            indices=entry.host_indices,
            buffer_id=-1,
            buffer=None,
        )
```

### 4.4 PartitionSpec 选择

host pool 的 sharding 必须与 device KV pool 对齐（除了 `memory_kind`）：

```python
# device pool (sgl-jax 现状, memory_pool.py:410)
device_sharding = NamedSharding(
    mesh,
    PartitionSpec("data", None, "tensor", None, None),
    memory_kind="device",
)

# host pool (RFC-1 新增)
host_sharding = NamedSharding(
    mesh,
    PartitionSpec("data", None, "tensor", None, None),  # 同 spec
    memory_kind="pinned_host",                            # 仅此差异
)
```

这保证 `jax.device_put(device_arr, host_sharding)` 是纯本地传输（同一 host 内 HBM → DRAM），零跨节点流量。

### 4.5 容量计算

```python
def compute_host_pool_size(
    device_pool_tokens: int,
    server_args: ServerArgs,
) -> int:
    if server_args.hicache_size_gb is not None:
        # 显式指定绝对 size
        bytes_per_token = compute_bytes_per_token(...)
        return int(server_args.hicache_size_gb * 1024**3 // bytes_per_token)
    else:
        # 默认 ratio
        return int(device_pool_tokens * server_args.hicache_ratio)
```

### 4.6 与 MLA 模型的兼容性

MLA 模型的 KV cache 是「所有 rank 完全相同的 latent representation」。在 host pool 上：
- 每个 rank 仍持有自己的 host pool（不引入跨进程共享，避免复杂度）
- 但所有 rank 的 host pool 内容完全相同 → 内存浪费（4 rank ≈ 4× redundancy）
- 缓解：MLA 模型下 `hicache_ratio` 用户可手动调小（如 0.5）—— RFC-1 **不在代码里自动覆盖默认值**，保持 §13 默认 2.0。文档建议用户根据模型类型手动调整。

> 注：跨进程共享 host pool（sglang 也没做）属于未来优化，不在 RFC-1 范围。自动 ratio 覆盖也是未来工作，避免引入隐式行为。

## 5. 详细设计：HostMemoryAllocator 实现

### 5.1 数据结构

```python
# python/sgl_jax/srt/mem_cache/host_memory_allocator.py (RFC-0 ABC 之后 append)

class FreeListHostMemoryAllocator(HostMemoryAllocator):
    """
    Free-list based token-level index allocator.
    与 sgl-jax 现有 device 侧 TokenToKVPoolAllocator (allocator.py:91) 同构.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # free_slots: jax.Array of int32 (动态收缩的 free list)
        # 初始时 free_slots = [0, 1, 2, ..., capacity-1]
        self.free_slots = jnp.arange(capacity, dtype=jnp.int32)

    def alloc(self, n: int) -> Optional[jax.Array]:
        if int(self.free_slots.shape[0]) < n:
            return None
        # 取前 n 个 (host-side numpy operation, no jit)
        # 注: 这里 jax.Array 用作 numpy 替代, 不进 jit
        free_np = np.asarray(self.free_slots)
        indices_np = free_np[:n]
        remaining_np = free_np[n:]
        self.free_slots = jnp.asarray(remaining_np)
        return jnp.asarray(indices_np)

    def free(self, indices: jax.Array) -> None:
        indices_np = np.asarray(indices)
        current_np = np.asarray(self.free_slots)
        self.free_slots = jnp.asarray(np.concatenate([current_np, indices_np]))

    def alloc_group(self, sizes: list[int]) -> Optional[list[jax.Array]]:
        total = sum(sizes)
        if int(self.free_slots.shape[0]) < total:
            return None
        free_np = np.asarray(self.free_slots)
        result = []
        offset = 0
        for s in sizes:
            result.append(jnp.asarray(free_np[offset:offset+s]))
            offset += s
        self.free_slots = jnp.asarray(free_np[total:])
        return result

    def free_group(self, groups: list[jax.Array]) -> None:
        for g in groups:
            self.free(g)

    def available_size(self) -> int:
        return int(self.free_slots.shape[0])

    def total_size(self) -> int:
        return self.capacity

    def clear(self) -> None:
        self.free_slots = jnp.arange(self.capacity, dtype=jnp.int32)
```

### 5.2 为什么 free_list 不是 jax.Array 计算

free_list 操作（alloc / free）是控制流密集型，不适合 JIT 编译。把 free_slots 当作 numpy buffer 用，仅在最后转 jax.Array 给 D2H/H2D kernel。这与 sgl-jax 现有 `TokenToKVPoolAllocator` 一致。

---

## 6. 详细设计：TPUKVCacheController

### 6.1 数据结构

```python
# python/sgl_jax/srt/mem_cache/cache_controller.py (RFC-0 ABC 之后 append)

from concurrent.futures import ThreadPoolExecutor, Future

@dataclass
class WriteOperation:
    op_id: int
    device_indices: jax.Array
    host_handle: HostBufferHandle
    node_id: int
    future: Optional[Future] = None
    submitted_at: float = 0.0

@dataclass
class LoadOperation:
    op_id: int
    host_indices: jax.Array
    device_indices: jax.Array
    node_id: int
    future: Optional[Future] = None
    submitted_at: float = 0.0

class TPUKVCacheController(KVCacheController):
    """
    HiCache 的 D2H/H2D 调度器 (TPU 实现).
    用 ThreadPoolExecutor 做后台异步, jax.device_put 做实际传输.

    依赖 sgl-jax 现有的 KVCache 接口:
    - device_pool.get_cpu_copy(indices)  -> ndarray (现成 D2H 原语)
    - device_pool.load_cpu_copy(arr, indices) (现成 H2D 原语)
    - device_pool.kv_sharding  (注意: 属性名是 kv_sharding, 不是 sharding)
    - device_pool.get_kv_buffer(layer_id) -> (k, v) per-layer (备选 per-layer 路径)

    Allocator 是单独依赖 (不属于 KVCache):
    - device_allocator: BaseTokenToKVPoolAllocator  (持有 device 侧 alloc/free)
    """

    def __init__(
        self,
        device_pool: KVCache,                              # MHATokenToKVPool / MLATokenToKVPool 等
        device_allocator: BaseTokenToKVPoolAllocator,      # 独立依赖, 由 builder 注入
        host_pool: LRUHostKVPool,
        tree_cache_evict_callback: Callable[[int], int],   # 用于触发 tree_cache.evict (因为 evict 是 tree 的职责)
        storage_backend: Optional[HiCacheStorage] = None,
        save_threads: int = 2,
        load_threads: int = 4,
    ):
        self.device_pool = device_pool
        self.device_allocator = device_allocator
        self.host_pool = host_pool
        self.tree_cache_evict = tree_cache_evict_callback
        self.storage_backend = storage_backend

        self.save_executor = ThreadPoolExecutor(
            max_workers=save_threads,
            thread_name_prefix="hicache_save",
        )
        self.load_executor = ThreadPoolExecutor(
            max_workers=load_threads,
            thread_name_prefix="hicache_load",
        )

        self._next_op_id = 0

        # ===== 并发不变量 (重要) =====
        # pending_writes / pending_loads 字典:
        #   - 只在 scheduler 主线程做 mutation (write/load/writing_check/loading_check)
        #   - 后台线程仅触发 future, 不直接 mutate 字典
        #   - WriteOperation.future 是线程安全的 (concurrent.futures.Future 保证)
        #   - 字典遍历用 list(...) 拷贝 keys, 避免 mutation during iteration
        # 不需要额外的锁.
        self.pending_writes: dict[int, WriteOperation] = {}
        self.pending_loads: dict[int, LoadOperation] = {}

        # 给 host pool 准备好 sharding
        self.host_sharding = host_pool.host_sharding
        self.device_sharding = device_pool.kv_sharding   # 注意属性名
```

### 6.2 D2H (write) 路径

```python
class TPUKVCacheController:

    def write(self, device_indices: jax.Array, node_id: int
              ) -> Optional[HostBufferHandle]:
        """
        分配 host buffer + 排队 D2H. 同步调用 (不等待完成).
        失败返回 None.
        """
        num_tokens = int(device_indices.shape[0])
        host_handle = self.host_pool.alloc(num_tokens)
        if host_handle is None:
            return None

        op = WriteOperation(
            op_id=self._next_op_id,
            device_indices=device_indices,
            host_handle=host_handle,
            node_id=node_id,
        )
        self._next_op_id += 1
        self.pending_writes[op.op_id] = op
        return host_handle

    def start_writing(self) -> None:
        """
        提交所有 pending writes 到后台线程.
        """
        for op_id in list(self.pending_writes.keys()):
            op = self.pending_writes[op_id]
            if op.future is not None:
                continue  # 已提交
            op.future = self.save_executor.submit(
                self._do_write, op
            )
            op.submitted_at = time.monotonic()

    def _do_write(self, op: WriteOperation) -> None:
        """
        后台线程: 执行 D2H.

        使用 sgl-jax 现有的 KVCache.get_cpu_copy 作为 D2H 原语:
            kv_cpu = device_pool.get_cpu_copy(device_indices)  # ndarray (numpy)
        然后把 numpy 数据写到 host pool 的对应 indices.

        关键: get_cpu_copy 内部已经做了 jax.device_put + block_until_ready,
        所以 _do_write 不需要重复.
        """
        # Step 1: D2H via 现有 get_cpu_copy
        # kv_cpu 是 list[np.ndarray] (per layer) 或 np.ndarray (fused)
        kv_cpu = self.device_pool.get_cpu_copy(op.device_indices)

        # Step 2: 写入 host pool 的 underlying buffer 对应 indices
        # host_pool.buffer 是 jax.Array (pinned_host), 用 .at[].set() 更新
        # 注意: .at[].set() 返回新 array, 不修改原 buffer
        self.host_pool.write_indices(op.host_handle.indices, kv_cpu)

    def writing_check(self) -> list[TransferAck]:
        """
        非阻塞检查已完成的 D2H. 只在 scheduler 主线程调用.
        """
        acks = []
        for op_id in list(self.pending_writes.keys()):
            op = self.pending_writes[op_id]
            if op.future is not None and op.future.done():
                # 检查 future 内部异常
                exc = op.future.exception()
                if exc is not None:
                    logger.error(f"D2H op {op_id} failed: {exc}")
                acks.append(TransferAck(
                    op_id=op.op_id,
                    done=True,
                    node_ids=[op.node_id],
                ))
                del self.pending_writes[op_id]
        return acks
```

### 6.3 H2D (load) 路径

```python
class TPUKVCacheController:

    def load(self, host_indices: jax.Array, node_id: int
             ) -> Optional[jax.Array]:
        """
        分配 device indices + 排队 H2D. 同步调用.
        Allocator 是独立依赖 (不属于 device_pool).
        驱逐 device 空间需要回调 tree_cache.evict (因为 evict 是 tree 的职责).
        """
        num_tokens = int(host_indices.shape[0])
        device_indices = self.device_allocator.alloc(num_tokens)
        if device_indices is None:
            # 主动驱逐 tree_cache 后重试
            self.tree_cache_evict(num_tokens)
            device_indices = self.device_allocator.alloc(num_tokens)
            if device_indices is None:
                return None

        op = LoadOperation(
            op_id=self._next_op_id,
            host_indices=host_indices,
            device_indices=device_indices,
            node_id=node_id,
        )
        self._next_op_id += 1
        self.pending_loads[op.op_id] = op
        return device_indices

    def start_loading(self) -> int:
        """提交所有 pending loads."""
        for op_id in list(self.pending_loads.keys()):
            op = self.pending_loads[op_id]
            if op.future is not None:
                continue
            op.future = self.load_executor.submit(
                self._do_load, op
            )
            op.submitted_at = time.monotonic()
        return -1  # producer_id, unused 在 sgl-jax (因为没 layer-wise overlap)

    def _do_load(self, op: LoadOperation) -> None:
        """
        后台线程: 执行 H2D.

        使用 sgl-jax 现有的 KVCache.load_cpu_copy 作为 H2D 原语:
            device_pool.load_cpu_copy(kv_cpu_ndarray, device_indices)
        内部已经做了 jax.device_put + scatter.
        """
        # Step 1: 从 host_pool 读出对应 indices 的 KV (返回 numpy ndarray)
        kv_cpu = self.host_pool.read_indices(op.host_indices)

        # Step 2: H2D + scatter via 现有 load_cpu_copy
        self.device_pool.load_cpu_copy(kv_cpu, op.device_indices)

    def loading_check(self) -> list[TransferAck]:
        acks = []
        for op_id in list(self.pending_loads.keys()):
            op = self.pending_loads[op_id]
            if op.future is not None and op.future.done():
                exc = op.future.exception()
                if exc is not None:
                    logger.error(f"H2D op {op_id} failed: {exc}")
                acks.append(TransferAck(
                    op_id=op.op_id,
                    done=True,
                    node_ids=[op.node_id],
                ))
                del self.pending_loads[op_id]
        return acks
```

### 6.3.1 关于 `get_cpu_copy` / `load_cpu_copy` 的复用

sgl-jax `KVCache` 已经为 partial rollout 实现了 `get_cpu_copy(indices) -> ndarray` 和 `load_cpu_copy(arr, indices)`（memory_pool.py:328、L332、L540、L551、L1205 等），它们是现成的 D2H/H2D 原语。RFC-1 直接复用：

| RFC-1 操作 | 复用的现有接口 |
|---|---|
| D2H (write_backup) | `kv_cpu = device_pool.get_cpu_copy(device_indices)` → 写入 host_pool.buffer 对应 indices |
| H2D (load_back) | `kv_cpu = host_pool.buffer[host_indices]`（numpy 切片）→ `device_pool.load_cpu_copy(kv_cpu, device_indices)` |

**好处**：
- 不引入新的 D2H/H2D kernel
- 与 sgl-jax 已有 retract / partial rollout 共享同一传输路径
- 测试 / debug 时复用现有逻辑

**LRUHostKVPool 需补充两个方法**（§4.3 补充）：

```python
class LRUHostKVPool(HostKVPool):

    def write_indices(self, indices: jax.Array,
                      kv_data: Union[np.ndarray, list[np.ndarray]]) -> None:
        """
        把 numpy KV 数据写到 host pool 的 indices 对应位置.
        kv_data 来自 device_pool.get_cpu_copy() 的返回.

        注意: self.buffer 用 .at[].set() 更新会创建新 jax.Array, 旧 buffer 引用失效.
        为避免 LRU 表里 entries.host_indices 失效, 必须把更新后的 buffer 写回 self.buffer.
        现有持有 HostBufferHandle 的代码不持有 buffer 引用 (只持有 indices), 所以 ok.
        """
        if isinstance(kv_data, np.ndarray):
            self.buffer = self.buffer.at[indices].set(jnp.asarray(kv_data))
        else:  # list of per-layer ndarrays
            for layer_id, layer_kv in enumerate(kv_data):
                self.buffer = self.buffer.at[indices, layer_id].set(
                    jnp.asarray(layer_kv))

    def read_indices(self, indices: jax.Array) -> np.ndarray:
        """读 host pool indices 对应的 KV 数据为 numpy."""
        return np.asarray(self.buffer[indices])
```

**重要约束**：`HostBufferHandle.buffer` 字段在 RFC-0 §7.2 中已明确为 `Optional[jax.Array]` 且 view-style。LRU 实现（`LRUHostKVPool`）填 `None`，因为 `.at[].set()` 会重新绑定 underlying buffer，缓存的 buffer 引用会失效。**所有调用方必须通过 `pool.read_indices(handle.indices)` / `pool.write_indices(...)` 访问数据**，不能直接读 `handle.buffer`。RFC-2 的 Queue 实现可填 stable buffer（因为 queue 模式不重新绑定）。

### 6.4 与 UnifiedRadixCache 的调用关系

```
UnifiedRadixCache.write_backup(node):
    host_handle = self.cache_controller.write(node.value, node.id)
    if host_handle is None:
        ...handle 失败 (host pool 满, 驱逐也无法腾出空间)
    else:
        node.host_value = host_handle.indices
        self.ongoing_write_through[node.id] = node
        self.inc_lock_ref(node)  # D2H 期间锁定
    self.cache_controller.start_writing()

UnifiedRadixCache.load_back(node):
    # node 是 evicted (无 device data, 有 host_value)
    device_indices = self.cache_controller.load(node.host_value, node.id)
    if device_indices is None:
        return None
    node.value = device_indices
    self.ongoing_load_back[node.id] = node
    self.inc_lock_ref(node)
    self.cache_controller.start_loading()
    return device_indices

UnifiedRadixCache.check_hicache_events():
    # 每 scheduler step 调用
    write_acks = self.cache_controller.writing_check()
    for ack in write_acks:
        node = self.ongoing_write_through.pop(ack.node_ids[0])
        node.backuped = True
        self.dec_lock_ref(node)
        # 可选: 触发 L3 backup
        if self.cache_controller.storage_backend is not None:
            self.cache_controller.write_backup_storage(...)

    load_acks = self.cache_controller.loading_check()
    for ack in load_acks:
        node = self.ongoing_load_back.pop(ack.node_ids[0])
        self.dec_lock_ref(node)
```

### 6.5 L3 storage 集成（可选路径）

```python
class TPUKVCacheController:

    def write_backup_storage(self, hash_keys: list[bytes],
                             host_indices: jax.Array) -> Optional[int]:
        """L2 → L3 backup. 完全后台异步."""
        if self.storage_backend is None:
            return None

        op_id = self._next_op_id
        self._next_op_id += 1

        def task():
            transfers = [
                PoolTransfer(
                    name=PoolName.KV,
                    hash_keys=hash_keys,
                    host_indices=host_indices,
                )
            ]
            return self.storage_backend.batch_set_v2(transfers)

        self.save_executor.submit(task)   # fire and forget
        return op_id

    def prefetch_from_storage(self, hash_keys: list[bytes],
                              host_indices: jax.Array) -> Optional[int]:
        """L3 → L2 prefetch. 后台异步."""
        if self.storage_backend is None:
            return None

        # 检查 L3 是否有该 key
        exists = self.storage_backend.batch_exists_v2(hash_keys, PoolName.KV)
        if not any(exists):
            return None

        op_id = self._next_op_id
        self._next_op_id += 1

        def task():
            transfers = [
                PoolTransfer(
                    name=PoolName.KV,
                    hash_keys=[k for k, e in zip(hash_keys, exists) if e],
                    host_indices=host_indices,
                )
            ]
            return self.storage_backend.batch_get_v2(transfers)

        self.load_executor.submit(task)
        return op_id
```

---

## 7. L3 backend：第一版不实现（接口预留）

### 7.1 当前状态

**RFC-1 第一版不提供任何 `HiCacheStorage` 具体 backend**（见 ADR-5 决策与理由）。
- `HiCacheStorage` ABC 由 RFC-0 §10 定义，作为未来扩展点保留
- `TPUKVCacheController.storage_backend` 字段在第一版恒为 `None`
- `TPUKVCacheController.prefetch_from_storage / write_backup_storage` 第一版 `raise NotImplementedError("L3 not implemented in first release")`
- `kv_cache_builder._build_hicache_infra` 不创建 storage_backend
- `--hicache-storage-backend` 配置项在第一版仅接受 `None`（其他值 raise + 明确提示「L3 not yet implemented」）

### 7.2 RFC-1 实装的辅助工具（供未来 backend 用）

虽然不实装具体 backend, 但**实装并测试 hash key 工具函数**（ADR-8 要求），定型未来 backend 必须遵循的内容寻址协议：

```python
# python/sgl_jax/srt/mem_cache/storage/utils.py

import hashlib

def compute_node_hash(
    prefix_token_ids: list[int],
    tp_rank: int,
    tp_size: int,
    dp_rank: int,
    model_name: str,
    is_mla: bool = False,
) -> bytes:
    """
    内容寻址 hash. 供未来 L3 backend 用 (RFC-1 自身不调用).

    MLA 模型: key 不含 tp_rank (所有 rank 数据相同, 同 sglang `backup_skip`)
    非 MLA: key 含 tp_rank (每 rank 是不同的 head shard)
    所有模型: key 始终含 dp_rank (sgl-jax SPMD DP attention)
    """
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(b"|")
    for token_id in prefix_token_ids:
        hasher.update(token_id.to_bytes(4, "little"))
    hasher.update(b"|")
    if not is_mla:
        hasher.update(tp_rank.to_bytes(4, "little"))
        hasher.update(b"|")
        hasher.update(tp_size.to_bytes(4, "little"))
        hasher.update(b"|")
    hasher.update(dp_rank.to_bytes(4, "little"))
    return hasher.digest()
```

单元测试覆盖 (RFC-1 §14.1 新增 `test_hicache_storage_hash.py`)：
- 同输入同输出
- 不同 prefix_token_ids → 不同 hash
- 不同 dp_rank → 不同 hash (新约束)
- MLA: 不同 tp_rank → 同 hash; 非 MLA: 不同 tp_rank → 不同 hash

### 7.3 未来 L3 backend 加入方式

L3 backend 不在 RFC-1, 但为避免实施时混乱, 这里列出未来加入新 backend 的标准流程（参考 sglang `storage/` 子目录组织）：

| 步骤 | 内容 |
|---|---|
| 1 | 新建 `python/sgl_jax/srt/mem_cache/storage/<backend>/` 子目录 |
| 2 | 实现 `<Backend>Storage(HiCacheStorage)` 类, 覆盖 `register_mem_host_pool_v2` / `batch_*_v2` / `get / set / exists / clear / get_stats` |
| 3 | 用 `compute_node_hash` 生成 key, 不要自创 hash scheme |
| 4 | 在 `kv_cache_builder._build_hicache_infra` 加 dispatch 分支 |
| 5 | server_args 加 `--hicache-storage-backend <backend>` 选项 |
| 6 | 加端到端集成测试 (L1→L2→L3→L2→L1 数据完整性) |
| 7 | 提交独立 RFC (RFC-3+) 描述具体 backend 设计与运维 |

候选 backend 优先级（社区讨论后再定）：
- GCS（对 Google Cloud / TPU 部署友好）
- 本地文件（toy, 开发调试用）
- 自研 KV store（针对 TPU 优化）

## 8. UnifiedRadixCache.init_hicache 集成

`UnifiedRadixCache.init_hicache()` 由 RFC-0 §4 port。RFC-1 仅完善它对应的 `_build_hicache_infra()` 实现：

```python
# python/sgl_jax/srt/mem_cache/kv_cache_builder.py (RFC-0 框架内)

def _build_hicache_infra(
    memory_pools: MemoryPools,
    server_args: ServerArgs,
    mesh: jax.sharding.Mesh,
    model_config: ModelConfig,
) -> tuple[LRUHostKVPool, TPUKVCacheController]:

    # Step 1: 计算 host pool 容量
    device_pool_tokens = memory_pools.kv_pool.total_size
    host_pool_tokens = compute_host_pool_size(device_pool_tokens, server_args)

    # Step 2: 创建 LRUHostKVPool
    host_kv_pool = LRUHostKVPool(
        host_total_tokens=host_pool_tokens,
        layer_num=model_config.num_hidden_layers,
        kv_head_per_rank=model_config.num_kv_heads // mesh.shape["tensor"],
        head_dim=model_config.head_dim,
        dtype=memory_pools.kv_pool.dtype,
        mesh=mesh,
        partition_spec=PartitionSpec("data", None, "tensor", None, None),
    )
    memory_pools.host_pool = host_kv_pool   # 装到 pytree 容器

    # Step 3: 创建 L3 storage backend (如配置)
    storage_backend = None
    # Step 3: L3 storage backend - RFC-1 第一版不实装 (ADR-5)
    storage_backend = None
    if server_args.hicache_storage_backend is not None:
        raise ValueError(
            f"--hicache-storage-backend={server_args.hicache_storage_backend!r} "
            f"is not yet implemented in RFC-1 first release. "
            f"L3 backend (file/GCS/etc) requires a separate RFC. "
            f"Please run with --hicache-storage-backend=None (L1↔L2 only)."
        )

    # Step 4: 创建 TPUKVCacheController (第一版 storage_backend=None)
    # 注意: device_allocator 和 tree_cache_evict_callback 是独立依赖,
    # 必须由 builder 显式注入 (KVCache 本身不持有 allocator).
    # tree_cache_evict_callback 由 UnifiedRadixCache 创建后注入 (见下方注释).
    cache_controller = TPUKVCacheController(
        device_pool=memory_pools.kv_pool,
        device_allocator=memory_pools.token_allocator,    # 来自 MemoryPools (RFC-0 §6)
        host_pool=host_kv_pool,
        tree_cache_evict_callback=None,                    # 占位; UnifiedRadixCache 创建后回填
        storage_backend=storage_backend,                   # 第一版恒为 None
        save_threads=server_args.hicache_save_threads or 2,
        load_threads=server_args.hicache_load_threads or 4,
    )

    return host_kv_pool, cache_controller
```

**关于 `tree_cache_evict_callback` 回填**：`TPUKVCacheController` 需要在 device pool 不够空间时回调 `tree_cache.evict(n_tokens)`。但 controller 在 tree_cache 之前创建（因为 tree_cache 在 `init_hicache()` 时需要 controller）。解决方法：

```python
# kv_cache_builder.build_kv_cache() 后续 step
tree_cache = UnifiedRadixCache(CacheInitParams(memory_pools=memory_pools, ...))
if server_args.enable_hierarchical_cache:
    host_kv_pool, cache_controller = _build_hicache_infra(memory_pools, server_args, mesh, model_config)
    # 回填 evict callback
    cache_controller.tree_cache_evict = tree_cache.evict
    # 注入 controller 到 tree_cache
    tree_cache.init_hicache(server_args, cache_init_params)
    tree_cache.cache_controller = cache_controller
```

> 也可改成在 `_build_hicache_infra` 之外创建 controller。RFC-1 暂取「先创建 controller + 后回填 callback」方案，保持 builder 结构清晰。

**关于 `memory_pools.token_allocator`**：sgl-jax 当前 allocator 在 scheduler 中独立创建（参见 `python/sgl_jax/srt/mem_cache/allocator.py` `TokenToKVPoolAllocator`）。RFC-0 §6 扩展 `MemoryPools` 时**必须**把 `token_allocator` 加入 pytree 字段，否则 RFC-1 拿不到。这是 RFC-0 → RFC-1 的隐式依赖，应在 RFC-0 §6.2 字段列表中补充：

```python
# RFC-0 §6.2 应补充 (见 RFC-0 修订记录):
@dataclass
class MemoryPools:
    req_to_token_pool: ReqToTokenPool
    kv_pool: KVPool
    token_allocator: BaseTokenToKVPoolAllocator   # NEW for RFC-1
    host_pool: Optional["HostKVPool"] = None
```

`UnifiedRadixCache` port 时, `init_hicache` 内部把 `cache_controller` 注入 `self.cache_controller`，HiCache 钩子调用它。

---

## 9. 写入策略

### 9.1 write_through（默认）

```
match_prefix → insert → _inc_hit_count → 触发 write_backup
                                         │
                                         ├─ if not parent.backuped:
                                         │   return  (父连续性约束)
                                         │
                                         ├─ alloc host indices (LRU pool)
                                         │   ├─ 成功 → 入队 D2H
                                         │   └─ 失败 → evict_host 重试 → 仍失败放弃
                                         │
                                         └─ start_writing() 提交后台
                                         │
                                         ▼
                              check_hicache_events (next step)
                                         │
                                         ├─ writing_check → ack 处理
                                         ├─ node.backuped = True
                                         └─ dec_lock_ref
```

### 9.2 write_back

```
evict (空间不足) → 选 LRU 节点 → 如 node.backuped:
                                    │ → 直接释放 device indices (降级为 CPU-only)
                                    │
                                  否则 (未备份):
                                    │ → write_backup(write_back=True)
                                    │   阻塞等待 D2H 完成
                                    │ → 然后释放 device indices
```

### 9.3 不实现的策略

- `write_through_selective`：YAGNI（ADR-3）
- per-layer dirty tracking：YAGNI

---

## 10. 读取/加载路径

### 10.1 触发时机

```
match_prefix 发现 host_hit_length > 0
    │
    └─ 调度器: 准备 prefill batch 时
       └─ tree_cache.init_load_back(InitLoadBackParams(...))
          └─ load_back → cache_controller.load + start_loading
              │
              └─ 后台线程: H2D → device_pool 写入完成
```

### 10.2 Step-level async load (ADR-4)

当前 batch 的 forward 在 device 上运行时，下一 batch 的 H2D 可以并行：

```
Step N:
    sched: collect requests, init_load_back for batch_N
        cache_controller.start_loading()  # 后台开始 H2D
    forward(batch_N-1)  # device 上跑 (与 batch_N H2D 并行)
    sched: ready_to_load_host_cache() → wait if needed
    forward(batch_N)

Step N+1:
    init_load_back for batch_N+1
    forward(batch_N)  → 此时已完成
    ...
```

**与 sgl-jax `event_loop_overlap` 的关系**：sgl-jax 的 overlap schedule（`event_loop_overlap`，参见 scheduler.py:807）已经把 CPU 调度逻辑和 TPU forward 计算并行化。HiCache 的 H2D 后台线程是**第三个**并行维度（CPU 后台 vs CPU 主线程 vs TPU），不需要修改 `event_loop_overlap` 本身——`start_loading()` 在调度阶段提交，`ready_to_load_host_cache()` 在 forward 前轮询，与现有 overlap 模式天然兼容。无需新增 scheduler 逻辑。

### 10.3 阻塞等待 (fallback)

如果 H2D 在 forward 开始时还没完成：

```python
def ready_to_load_host_cache(self) -> int:
    """
    返回还在 pending 的 load 数量. >0 表示有 load 未完成.
    Scheduler 应等待 == 0 再开始 forward.
    """
    return len(self.cache_controller.pending_loads)
```

scheduler 主循环里阻塞：

```python
# scheduler.py
while self.tree_cache.ready_to_load_host_cache() > 0:
    self.tree_cache.check_hicache_events()
    time.sleep(0.0005)  # 微秒级 polling
```

---

## 11. 异步模型

### 11.1 线程模型

```
Main Scheduler Thread:
  ├─ event_loop_normal (同步)
  ├─ check_hicache_events (每 step 调用)
  │   ├─ writing_check (非阻塞 future.done())
  │   └─ loading_check (非阻塞)
  └─ ready_to_load_host_cache (read pending count)

Background save_executor (ThreadPoolExecutor, 2 workers):
  └─ _do_write (jax.device_put + block_until_ready, GIL 释放)

Background load_executor (ThreadPoolExecutor, 4 workers):
  └─ _do_load (jax.device_put + block_until_ready, GIL 释放)

Background storage_save (复用 save_executor):
  └─ batch_set_v2 (file I/O, GIL 释放)

Background storage_get (复用 load_executor):
  └─ batch_get_v2 (file I/O, GIL 释放)
```

### 11.2 GIL 行为

参考 jax-api 调研 §5.2，所有 JAX C++ 调用释放 GIL，所以：
- Main thread forward 在 device 上跑（C++）→ GIL 释放
- Background `jax.device_put` 在 C++ 调用 → GIL 释放
- 两者真正并行（无 GIL 竞争）

### 11.3 与 sgl-jax overlap schedule 的兼容

sgl-jax 已有 `event_loop_overlap`（基于 `ModelWorkerClient`）做 CPU/TPU 并行。HiCache 的 D2H/H2D 是**另一个**层次的并行（CPU 主线程 vs CPU 后台线程 vs TPU），不冲突。

---

## 12. SPMD 一致性

### 12.1 sgl-jax vs sglang 的差异

- **sglang**：每个 TP rank 一个独立 scheduler 进程，每个进程有自己的 `HiRadixCache`，通过 `all_reduce(MIN)` 同步队列消费数量。
- **sgl-jax**：单 scheduler 进程（同时持有 ModelWorker 作为内部 class，不是独立 mp.Process），所有 HiCache 元数据决策在此进程；多 host 部署时每个 host 一个 scheduler 进程（JAX 多进程 SPMD），每进程独立维护自己的 HiCache。

### 12.2 RFC-1 同步策略（ADR-6）

1. **`TPUKVCacheController` 跑在 scheduler 主进程**：由于 ModelWorker 是 scheduler 进程内的 class（参见 `python/sgl_jax/srt/managers/tp_worker.py:34`），`jax.device_put` 可以直接 access 同进程的 `device_pool` jax.Array，无 IPC 开销
2. **多 host 部署每进程独立 HiCache**：当 `jax.process_count() > 1` 时（多 host JAX SPMD），每个进程：
   - 持有自己的 `LRUHostKVPool`，容量按本 host 的 device pool shard 计算
   - 持有自己的 `TPUKVCacheController`
   - 独立做 D2H/H2D（数据来自本 host 的 device shard，去本 host 的 pinned host memory）
   - 不跨进程同步 LRU 元数据
3. **与 sglang 等价**：sglang 也是每 TP rank 独立 host pool（[sglang 调研 §2.2 "TP 场景下的 L1→L2 数据流"](../research/2026-05-18-sglang-cache-pd-organization.md) 详述）。区别仅在于 sgl-jax 一进程 = 一 host（含多 TP rank），sglang 一进程 = 一 TP rank。

### 12.3 不需要 ZMQ 协调

sgl-jax 的 `select_dp_for_request` / `sync_pub` ZMQ 通信用于 DP 路由，与 HiCache 无关。HiCache 不引入新的 ZMQ 消息。

### 12.4 元数据一致性

由于 `UnifiedRadixCache` 的 tree 在 scheduler 进程内是唯一的（不像 sglang 每 rank 一份），tree 一致性自然保证。多 host 场景下，每个 host 的 tree 是独立的（每个 host 看到不同的请求子集，因为 DP 已经做了路由），无需跨 host 同步。

---

## 13. 配置项扩展

RFC-0 已定义大部分配置项。RFC-1 补充默认值和验证：

| 配置项 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `enable_hierarchical_cache` | bool | False | 启用 HiCache（要求非 disable_radix_cache） |
| `hicache_ratio` | float | 2.0 | host pool = device pool × ratio |
| `hicache_size_gb` | float | None | 覆盖 ratio |
| `hicache_write_policy` | str | "write_through" | "write_through" / "write_back"。**不支持 selective** |
| `hicache_storage_backend` | str | None | "file" / None。**不支持其他** |
| `hicache_storage_backend_extra_config` | str (JSON) | "{}" | `{"root": "/path/to/dir"}` |
| `hicache_save_threads` | int | 2 | save_executor 线程数 |
| `hicache_load_threads` | int | 4 | load_executor 线程数 |

### 13.1 兼容性检查

```python
def _validate_hicache_args(self):
    if not self.enable_hierarchical_cache:
        return
    assert not self.disable_radix_cache, \
        "HiCache requires prefix-tree cache"
    assert self.hicache_write_policy in ("write_through", "write_back"), \
        f"Unsupported write policy: {self.hicache_write_policy}"
    assert self.hicache_storage_backend in (None, "file"), \
        f"RFC-1 only supports file backend"
    if self.hicache_size_gb is not None:
        assert self.hicache_size_gb > 0
```

---

## 14. 测试策略

### 14.1 单元测试

| 测试文件 | 覆盖范围 |
|---|---|
| `test_lru_host_kv_pool.py` | alloc/free/evict/lock_ref；LRU 顺序；hash key 注册和查找；容量满后的退化行为 |
| `test_free_list_host_memory_allocator.py` | alloc/free/alloc_group/free_group；碎片化场景；clear() |
| `test_tpu_cache_controller.py` | D2H/H2D 正确性（用小张量比对 device 和 host）；writing_check/loading_check 非阻塞；后台线程 GIL 行为；并发提交；`storage_backend=None` 时 prefetch_from_storage/write_backup_storage 调用应 raise NotImplementedError |
| `test_hicache_storage_hash.py` | `compute_node_hash` 稳定性（同输入同输出）；MLA 跳过 tp_rank；dp_rank 必含；供未来 L3 backend 用 |

### 14.2 集成测试

| 测试文件 | 内容 |
|---|---|
| `test_hicache_integration.py` | UnifiedRadixCache + HiCache 端到端 (L1↔L2 only)：插入 → 驱逐 → load_back → 验证数据等价 |
| `test_hicache_write_policy.py` | write_through vs write_back 行为对比 |
| ~~`test_hicache_l3_roundtrip.py`~~ | **不在 RFC-1 范围**（L3 backend 未实装，参见 ADR-5）|

### 14.3 端到端推理测试

| 场景 | 期望 |
|---|---|
| Llama 推理 + HiCache off | 无回退（baseline） |
| Llama 推理 + HiCache on (write_through, no L3) | KL test 等价 + cache 命中率提升 |
| Llama 推理 + HiCache on + L3 file | 同上 + L3 写入文件数 > 0 |
| MiMo-V2-Flash / MiMo-V2-Pro 推理 + HiCache on（SWA + HiCache） | KL 等价（验证 SWAComponent HiCache hook） |
| KDA 推理 + HiCache on（Mamba + HiCache） | KL 等价（验证 RecurrentStateComponent HiCache hook） |
| Cache 命中率 benchmark（多轮对话） | 启用 HiCache 后命中率提升 ≥ 30%（典型 multi-turn） |

### 14.4 性能测试

| 指标 | 期望 |
|---|---|
| TTFT 不退化（HiCache off） | < 1% 抖动 |
| TTFT 降低（HiCache on, 多轮对话） | ≥ 20% 降低 |
| 吞吐不退化（forward 与 D2H 并行） | < 5% 退化 |
| Memory 占用（host_pool） | 符合 `hicache_ratio` × device_pool size |

---

## 15. 风险与未决问题

### 15.1 已识别风险

| Risk | 影响 | 缓解 |
|---|---|---|
| **R1: ThreadPoolExecutor GIL 实际行为偏离预期** | D2H 后台线程实际阻塞主线程 | 微基准测试验证 GIL 释放；如发现问题改用进程池（但代价高） |
| **R2: jax.device_put 在 pinned_host 之间的开销** | D2H 频繁触发，性能退化 | (a) 提高 write_through_threshold 等价物（hit_count）；(b) profile 后改用 Pallas（ADR-1 已留这个口子） |
| **R3: host pool 容量误算导致 OOM** | 用户配置 hicache_size_gb 过大 | 加 sanity check；运行时 `psutil` 监控 |
| **R4: SPMD 跨 worker 协调 ZMQ 消息丢失** | D2H 在某些 worker 没执行 | 添加 ack 机制 + 超时重试 |
| ~~**R5: LocalFileStorage 文件 corruption**~~ | ~~写入崩溃留半文件~~ | **N/A — RFC-1 不实装 file backend (ADR-5)；未来 L3 backend RFC 时再考虑** |
| **R6: L3 file backend 容量无限增长** | 占满磁盘 | 文档警告 + 外部 cron 清理；后续 RFC 加 size limit |

### 15.2 未决问题

| 问题 | 决策时机 |
|---|---|
| Host pool 是否需要支持「reserved region」（如 system prompt 永久驻留）？ | 后续 PR |
| L3 backend 是否需要 size limit / LRU on disk？ | 后续 PR |
| 是否需要 `--hicache-mem-layout`（参考 sglang）来优化 D2H 性能？ | 性能 profile 后决定 |
| SWA / Mamba 模型的 HiCache hook 行为是否需要独立测试覆盖？ | 是，RFC-1 §14.3 已列入端到端测试 |

---

## 16. 实施路线

### 16.1 阶段（H0 → H3）

```
┌─────────────────────────────────────────────────────────────────┐
│ H0: HostKVPool + Allocator                                       │
├─────────────────────────────────────────────────────────────────┤
│ ✓ LRUHostKVPool 实现 (alloc/free/evict/lock_ref + LRU)           │
│ ✓ FreeListHostMemoryAllocator 实现                                │
│ ✓ 单元测试 test_lru_host_kv_pool.py / test_..._allocator.py       │
│                                                                  │
│ 验收: 单元测试 100% 通过, 包括 LRU 顺序 + lock_ref 阻塞 + 碎片场景│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ H1: TPUKVCacheController + D2H/H2D 路径                          │
├─────────────────────────────────────────────────────────────────┤
│ ✓ TPUKVCacheController 实现 (write/load/start_writing/...)        │
│ ✓ save_executor / load_executor 配置                             │
│ ✓ 与 UnifiedRadixCache 的钩子对接                                │
│ ✓ 单元测试 test_tpu_cache_controller.py                          │
│ ✓ 集成测试 test_hicache_integration.py (端到端 D2H + H2D)        │
│                                                                  │
│ 验收:                                                            │
│  • 单元 + 集成测试通过                                            │
│  • Llama 推理 + HiCache 启用, KL 等价                             │
│  • TTFT 多轮对话场景测量, 命中率上升                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ H2: Hash 工具 + SWA/Recurrent HiCache 覆盖 + 性能验证            │
├─────────────────────────────────────────────────────────────────┤
│ ✓ compute_node_hash 工具实装 (供未来 L3 backend 用, ADR-8)        │
│ ✓ 单元测试 test_hicache_storage_hash.py                          │
│ ✓ MiMo-V2-Flash / MiMo-V2-Pro (SWA) HiCache 端到端                │
│ ✓ KDA / Bailing-MoE-Linear (Recurrent-State) HiCache 端到端       │
│ ✓ MLA (DeepSeek) HiCache 端到端                                  │
│ ✓ 性能 benchmark (TTFT 降低 / 吞吐保持 / host pool 命中率)        │
│                                                                  │
│ 验收: 所有 sgl-jax 支持的模型类型 HiCache 启用后 KL 等价          │
└─────────────────────────────────────────────────────────────────┘

   ────── L3 backend 实施 (未来 RFC-3+, 不在 RFC-1 范围) ──────
   触发条件: H2 稳定后, 用户提出真实 L3 需求 (跨实例共享 / 长期持久化等)
   候选 backend: GCS / 本地文件 / 自研 TPU 友好 KV store
```

### 16.2 H0/H1 可以在 RFC-0 M2 完成前并行起步

H0（pool + allocator）的实现不依赖 UnifiedRadixCache，可以在 RFC-0 M0 完成后立即开始。
H1（cache_controller）的最终集成依赖 RFC-0 M1（UnifiedRadixCache port），但可以先开发独立的 `_do_write` / `_do_load` 逻辑。

---

## 17. 附录

### 17.1 sglang origin/main 关键源码索引（port / 参考时用）

| 主题 | sglang 文件 |
|---|---|
| HostKVPool 设计 | `python/sglang/srt/mem_cache/memory_pool_host.py` |
| cache_controller | `python/sglang/srt/managers/cache_controller.py` |
| 写入策略 | `python/sglang/srt/mem_cache/hiradix_cache.py` 中的 `write_backup` |
| Hash 计算 | `python/sglang/srt/mem_cache/utils.py` `compute_node_hash_values` |
| file backend | `python/sglang/srt/mem_cache/hicache_storage.py` `HiCacheFile` |
| MLA backup_skip | `python/sglang/srt/managers/cache_controller.py:443-447` |

### 17.2 tpu-inference 参考

| 主题 | tpu-inference 文件 |
|---|---|
| LocalCPUBackend OrderedDict 模式 | `tpu_inference/offload/cpu_backend.py` |
| ThreadPoolExecutor + jax.device_put 异步 | `tpu_inference/offload/tpu_offload_connector.py` 的 `save_executor` |
| pinned_host sharding 构造 | `tpu_inference/offload/tpu_offload_connector.py:1274-1317` `register_runner` |
| `donate_argnames` + `optimization_barrier` 用法 | `tpu_inference/offload/utils.py:97-119` `stack_kv_cache_cross_layers` |

### 17.3 jax-api 调研引用

| RFC-1 章节 | jax-api 调研对应 |
|---|---|
| ADR-1 (jax.device_put 选 Pallas) | §2.2.1 / §2.2.2 |
| ADR-2 (ThreadPoolExecutor) | §5.2 |
| ADR-4 (无 layer-wise overlap) | §6.2 |
| §4.4 PartitionSpec 设计 | §3 |
| §6.3 H2D 路径 | §2.3 |

### 17.4 相关 RFC

- RFC-0: UnifiedRadixCache + KV 缓存与传输基础设施
- RFC-2: PD 分离（待写）

---

**End of RFC-1**
