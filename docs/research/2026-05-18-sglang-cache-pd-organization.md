# sglang 组织结构调研：HiCache + PD 分离 + UnifiedRadixCache

> **目的**：为 sgl-jax 即将编写的 HiCache / PD 分离 / 共同基础设施 RFC 提供「sglang 视角」的对标参考。本文聚焦**代码组织、模块边界、类继承图、接口签名、扩展点**，不重复 `../../workspace/sglang/hicache_research.md` 已经覆盖的数据流细节、写入策略、storage 后端清单。
>
> **配套**：与 `2026-05-18-tpu-inference-jax-api-survey.md` 一起作为 RFC 输入。
>
> **代码基准**：`/Users/jiongxuan/workspace/sglang` 的 **`origin/main`**（HEAD `f04c52253`），日期 2026-05-18。所有 `git show origin/main:...` 命令读取。所有路径相对该仓库根目录。

---

## 0. TL;DR

| 议题 | 现状 | 对 sgl-jax 的含义 |
|---|---|---|
| HiCache | 完整、稳定（`hiradix_cache.py` + `cache_controller.py` + 7 个 storage backend） | 接口可直接对标，控制面 99% 可复用 |
| PD 分离 | 完整、稳定（4-tuple ABC + 6 个 backend：Mooncake/NIXL/MoRI/Ascend/Fake/Common） | 接口可直接对标，**不要用 vLLM 的 `KVConnectorBase_V1` 风格**——sglang 自有一套 |
| PD ↔ HiCache 耦合 | **主路径独立**（UnifiedRadixCache 1960 行 0 处提及 PD；PD 完全绕过 tree_cache 走自己的 `KVSender.send`）；**底层传输引擎可共享**（mooncake `TransferEngine` 在同进程内可被 PD + HiCache L3 共用）；**唯一融合点**：`DecodeKVCacheOffloadManager`（D 把自己生成的 KV offload 到 L3，**不是** P 写 L3 / D 读 L3）。详见 §3.6 | RFC-1/RFC-2 应保持独立模块；RFC-0 加「传输底座」共享层；"P→D 走 L3 协议" 作为 Alternative 讨论但 upstream 无实践 |
| UnifiedRadixCache | **已合入完整 D↔H HiCache 集成**（`unified_radix_cache.py` 1960 行，三个 component 全部实现 HiCache hook）；仍 env-var gated，与旧实现平行 | **sgl-jax 应作为 port 者，不是先行者**；直接以 origin/main 的 UnifiedRadixCache 为 spec |
| Unified + HiCache L3 集成 | **缺失**：没有 `prefetch_from_storage / attach_storage_backend`——只在 HiRadixCache 里有；UnifiedRadixCache 只覆盖 D↔H（L1↔L2） | sgl-jax 的 L3 工作有窗口期；短期不是关键路径 |
| PR 进展 | Issue #20415 Stage 1 进度 ≈ **3/12 子项**完成；HiCache D↔H、SWA HiCache、Session Cache 已 merge；DeepSeek HiCache (#24691) 等仍在推进 | sgl-jax 跟进节奏：紧盯 #24691、L3 support、PD/Spec 兼容 |
| Scheduler 创建链 | **已改为 builder 模式**：`scheduler.py` 调 `kv_cache_builder.build_kv_cache(...)`，不再是大段 if-elif（旧版调研描述的 if-elif 链在 origin/main 已不存在） | sgl-jax 也应直接采用 builder 模式，避免迁移负担 |
| PD 插件抽象 | sglang 没有运行时注册机制，新 backend 必须改 `disaggregation/utils.py` 加 enum 分支 | sgl-jax 可以做更优雅的注册表 |
| sglang vs vLLM | sglang 的 `BaseKVConnector` 是 **model weight loading** 用的，不是 PD 插件 | sgl-jax 设计 PD 时不要照搬 vLLM v1 的 `KVConnectorBase_V1` 风格 |

---

## 1. 整体目录组织

```
python/sglang/srt/
├── mem_cache/                              # KV/前缀缓存
│   ├── base_prefix_cache.py                # 抽象基类 BasePrefixCache + 7 个 dataclass
│   ├── radix_cache.py                      # RadixCache (GPU-only)
│   ├── hiradix_cache.py                    # HiRadixCache (HiCache 核心) ★
│   ├── swa_radix_cache.py                  # SWARadixCache (滑动窗口)
│   ├── mamba_radix_cache.py / hi_mamba_radix_cache.py  # Mamba 混合
│   ├── chunk_cache.py                      # ChunkCache (无前缀复用)
│   ├── unified_radix_cache.py              # UnifiedRadixCache (新架构, 1024 行) ★
│   ├── unified_cache_components/           # 可插拔组件
│   │   ├── tree_component.py               # TreeComponent ABC (14 个 hook + 3 个 HiCache hook)
│   │   ├── full_component.py / swa_component.py / mamba_component.py
│   ├── memory_pool.py                      # GPU 侧 KV pool (MHA/MLA/NSA/Hybrid)
│   ├── memory_pool_host.py                 # CPU 侧 Host Pool (HiCache L2)
│   ├── hicache_storage.py                  # HiCacheStorage ABC + HiCacheFile
│   ├── hybrid_cache/                       # 多 pool 分级控制器
│   ├── storage/                            # 7 个 L3 backend (file/nixl/mooncake/hf3fs/aibrix/eic/simm) + lmcache 集成
│   ├── radix_cache_cpp.py / cpp_radix_tree/  # C++ 加速版
│   └── allocator.py / evict_policy.py / utils.py
│
├── disaggregation/                         # PD 分离
│   ├── base/conn.py                        # 4-tuple ABC (KVArgs/Manager/Sender/Receiver/BootstrapServer)
│   ├── common/conn.py                      # CommonKVManager/Sender/Receiver/BootstrapServer
│   ├── mooncake/ nixl/ mori/ ascend/ fake/ # 5 个具体 backend
│   ├── prefill.py                          # PrefillBootstrapQueue + SchedulerDisaggregationPrefillMixin
│   ├── decode.py                           # DecodePreallocQueue + DecodeTransferQueue + DecodeReqToTokenPool + Mixin
│   ├── decode_schedule_batch_mixin.py
│   └── utils.py                            # DisaggregationMode/TransferBackend enum + get_kv_class factory
│
├── managers/
│   ├── scheduler.py                        # Scheduler 主类（事件循环 dispatch，tree_cache 创建已下沉到 kv_cache_builder）
│   ├── cache_controller.py                 # HiCacheController (write_stream / load_stream / prefetch_thread / backup_thread)
│   ├── tp_worker.py / tp_worker_overlap_thread.py
│   ├── tokenizer_manager.py / detokenizer_manager.py
│   └── io_struct.py
│
├── mem_cache/kv_cache_builder.py           # ★ tree_cache 工厂 (build_kv_cache, 318 行)
│
└── connector/                              # ⚠️ 不是 PD 插件！是 model weight loading（Redis 等）
    ├── base_connector.py                   # BaseConnector + BaseKVConnector (weight loading 用)
    └── ...
```

**关键边界判读**：
- `mem_cache/` 是「缓存数据结构 + 后端存储」
- `disaggregation/` 是「跨进程 KV 传输 + 调度协议」
- `managers/cache_controller.py` 是 HiCache 的「I/O 调度器」，**不在 `mem_cache/` 里**——它管理 stream 和后台线程，被 `hiradix_cache.py` 依赖
- `connector/` **不是** PD 插件目录，是 model weight loading（容易混淆，sgl-jax 设计 PD 时不要重名）

---

## 2. HiCache 模块

### 2.1 类继承图

```
BasePrefixCache (ABC)  ← base_prefix_cache.py
    │
    └── RadixCache  ← radix_cache.py
            │
            ├── HiRadixCache  ← hiradix_cache.py  ★ HiCache 入口
            ├── HiMambaRadixCache  ← hi_mamba_radix_cache.py
            └── LMCRadixCache  ← storage/lmcache/lmc_radix_cache.py  (替代方案)
```

```
KVCache (abc.ABC)  ← memory_pool.py:668
    │
    ├── MHATokenToKVPool / MHATokenToKVPoolFP4
    ├── MLATokenToKVPool / MLATokenToKVPoolFP4
    │       └── NSATokenToKVPool
    │               └── HiSparseNSATokenToKVPool
    └── HybridLinearKVPool

HostKVCache (ABC)  ← memory_pool_host.py
    │
    ├── MHATokenToKVPoolHost
    ├── MLATokenToKVPoolHost
    ├── MambaPoolHost
    └── NSAIndexerPoolHost

HostPoolGroup / PoolEntry  ← 多 pool 统一管理
```

```
HiCacheStorage (ABC)  ← hicache_storage.py:98
    │
    ├── HiCacheFile          (local disk)
    ├── HiCacheNixl          (NVIDIA NIXL plugin)
    ├── MooncakeStore        (Mooncake)
    ├── HiCacheHF3FS         (3FS)
    ├── AibrixKVCacheStorage
    ├── EICStorage
    └── HiCacheSiMM

StorageBackendFactory  ← storage/backend_factory.py
    └── 懒加载注册表 + dynamic 动态加载
```

### 2.2 HiCacheController (managers/cache_controller.py)

**职责**：HiCache 的 I/O 调度器，管理 CUDA stream 和后台线程。

```python
class HiCacheController:
    # CUDA streams
    write_stream         # 专用 D2H 写 stream
    load_stream          # 专用 H2D 加载 stream

    # 后台线程
    backup_thread        # 异步写入 L3 storage
    prefetch_thread      # 异步从 L3 预取到 L2

    # 关键方法
    write(device_indices, node_id) -> host_indices       # 分配 host 内存 + 入队
    start_writing()                                       # 发起 D2H DMA
    load(host_indices, node_id) -> device_indices        # 分配 device 内存 + 入队
    start_loading() -> producer_id                       # 发起 H2D DMA (per-layer)
```

**SPMD 同步点**（`hiradix_cache.py`）：每个 TP rank 独立维护 host pool 和 radix tree，通过 `all_reduce(op=MIN)` 同步队列消费数量。

### 2.3 关键接口签名

**`BasePrefixCache`**（mem_cache/base_prefix_cache.py:150）：

```python
class BasePrefixCache(ABC, PrefixCacheTrait):
    # 核心 (L178-198, 所有 cache 实现必须 override)
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult: ...
    def cache_finished_req(self, req, **kwargs): ...
    def cache_unfinished_req(self, req, chunked, **kwargs): ...
    def insert(self, params: InsertParams) -> InsertResult: ...
    def evict(self, params: EvictParams) -> EvictResult: ...
    def inc_lock_ref(self, node) -> IncLockRefResult: ...
    def dec_lock_ref(self, node, params): ...

    # HiCache 钩子 (L227-256, 默认 raise NotImplementedError)
    def init_load_back(self, params: InitLoadBackParams): ...
    def ready_to_load_host_cache(self): ...
    def flush_write_through_acks(self): ...
    def check_hicache_events(self): ...
    def take_events(self): ...

    # 能力探测 (L259-286)
    def supports_swa(self) -> bool: ...
    def supports_mamba(self) -> bool: ...
    def supports_streaming_session(self) -> bool: ...
    def is_chunk_cache(self) -> bool: ...
    def is_tree_cache(self) -> bool: ...
```

**关联 dataclass**（base_prefix_cache.py L36-148）：
- `MatchPrefixParams` (L36)
- `InsertParams` (L47) / `InsertResult` (L66)
- `EvictParams` (L74)
- `InitLoadBackParams` (L114)
- `MatchResult` (L123, NamedTuple，已含 `last_host_node` / `host_hit_length` 字段——为 HiCache 预留)

**`HiCacheStorage`**（mem_cache/hicache_storage.py:98）：

```python
class HiCacheStorage(ABC):
    def register_mem_pool_host(self, host): ...               # L105
    def register_mem_host_pool_v2(self, host_pool, pool_name): ...  # L108

    # v2 API (多 pool, 用 PoolTransfer 描述符)
    def batch_exists_v2(self, ...): ...   # L113
    def batch_get_v2(self, transfers: list[PoolTransfer]): ...   # L146
    def batch_set_v2(self, transfers: list[PoolTransfer]): ...   # L157

    # v1 API (单 pool)
    def batch_get_v1(keys, host_indices): ...   # L168
    def batch_set_v1(keys, host_indices): ...   # L180

    # 单条 / 通用
    def get(key) / batch_get(keys) / set / batch_set / exists / batch_exists / clear / get_stats
```

**配套类**（hicache_storage.py L17-95）：
- `HiCacheStorageConfig` (L17)
- `HiCacheStorageExtraInfo` (L34)
- `PoolName(Enum)` (L39)
- `PoolHitPolicy(Enum)` (L50)
- `PoolTransfer` (L62)
- `PoolTransferResult` (L77)

**`KVCache`**（memory_pool.py:668）：

```python
class KVCache(abc.ABC):
    def get_key_buffer(self, layer_id) -> Tensor: ...       # L730
    def get_value_buffer(self, layer_id) -> Tensor: ...     # L734
    def get_kv_buffer(self, layer_id) -> Tuple[Tensor, Tensor]: ...  # L738
    def set_kv_buffer(self, ...): ...                       # L742
    def register_layer_transfer_counter(self, counter): ... # L751  ← HiCache layer-wise overlap 用
```

### 2.4 Scheduler 集成点

`scheduler.py` 中的关键调用（参见 hicache_research.md §6 详表）：
| 时机 | 方法 |
|---|---|
| 初始化 | 创建 `HiRadixCache` / `HiMambaRadixCache`（L834-846） |
| 请求入队 | `tree_cache.prefetch_from_storage()` (L2050) |
| 主循环每 step | `tree_cache.check_hicache_events()` (L2447) |
| Batch 准备 | `tree_cache.ready_to_load_host_cache()` (L2628) |
| 处理 ack | `tree_cache.flush_write_through_acks()` (L2681) |

**HTTP 控制面**：`/clear_hicache_storage_backend`、`HiRadixCache.attach_storage_backend()` / `detach_storage_backend()` 支持运行时挂载后端。

---

## 3. PD 分离模块（disaggregation/）

### 3.1 4-tuple Base Abstraction

**`disaggregation/base/conn.py`**：

```python
class KVArgs:                                  # L15, POD
    kv_data_ptrs: List[int]                   # GPU pool 指针
    kv_data_lens: List[int]
    item_lens: List[int]
    aux_data_ptrs / aux_data_lens / aux_item_lens
    state_data_ptrs / state_data_lens / state_type: "none"|"mamba"|"swa"
    gpu_id: int                                # ← TPU 迁移要替换
    pp_rank: int
    prefill_start_layer: int
    system_dp_rank: int

class KVPoll(IntEnum):                         # L42, 状态机
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4

class BaseKVManager(ABC):                      # L50
    def __init__(self, args: KVArgs, mode: DisaggregationMode,
                 server_args: ServerArgs, is_mla_backend: bool): ...
    def register_to_bootstrap(self): ...

class BaseKVSender(ABC):                       # L68
    def __init__(self, mgr, bootstrap_addr, bootstrap_room,
                 dest_tp_ranks, pp_rank): ...
    def init(self, num_kv_indices, aux_index): ...
    def send(self, kv_indices, state_indices): ...
    def poll(self) -> KVPoll: ...
    def failure_exception(self) -> Exception: ...

class BaseKVReceiver(ABC):                     # L113
    def __init__(self, mgr, bootstrap_addr, bootstrap_room): ...
    def init(self, prefill_dp_rank): ...
    def send_metadata(self, kv_indices, aux_index, state_indices): ...
    def poll(self) -> KVPoll: ...
    def clear(self): ...
    def abort(self): ...

class BaseKVBootstrapServer(ABC):              # L172
    def __init__(self, host, port): ...
```

### 3.2 Connector 继承层级

```
BaseKVManager / BaseKVSender / BaseKVReceiver / BaseKVBootstrapServer  ← base/conn.py
    │
    └── CommonKV*  ← common/conn.py (L88/L425/L514/L709)
            ├── MooncakeKV*  ← mooncake/conn.py (L187/L1639/L1745/L1932)
            │       └── AscendKV*  ← ascend/conn.py (L21/L134/L138/L142)
            ├── NixlKV*      ← nixl/conn.py (L170/L1028/L1094/L1247)
            └── MoRIKV*      ← mori/conn.py (L176/L838/L979/L1112)

FakeKV*  ← fake/conn.py (L21/L35/L79, 直接继承 Base，无 bootstrap)
```

**`CommonKVManager`** 是真正的"基础设施"层（common/conn.py:88）：集中了 bootstrap 注册、TP/PP rank mapping、KV ptr 计算（`get_mha_kv_ptrs_with_pp` L378、`get_mla_kv_ptrs_with_pp` L411）。新 backend 通常继承 `Common*`。

### 3.3 Factory & Role Assignment

**`disaggregation/utils.py`**：

```python
class DisaggregationMode(Enum):                # L33
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

class TransferBackend(Enum):                   # L304
    MOONCAKE / NIXL / MORI / ASCEND / FAKE

class KVClassType(Enum):                       # L312
    MANAGER / SENDER / RECEIVER / BOOTSTRAP_SERVER

def get_kv_class(backend: TransferBackend, class_type: KVClassType):  # L321
    # 5-way dispatch (mooncake/mori/ascend/nixl/fake)
```

**Role 判定**：进程级，由 `server_args.disaggregation_mode` 字符串决定。`scheduler.py:902` 比较 `== "decode"`；L3678/L3685 由 enum 选事件循环。

**关键观察**：**没有运行时注册机制**——新 backend 必须改 `utils.py` 加 enum 分支。sgl-jax 可以做更优雅的注册表（例如 entry_points 或 decorator 注册）。

### 3.4 Bootstrap Server 协议

**HTTP-based**，入口 `common/conn.py:709`：

```python
class CommonKVBootstrapServer:
    def run(self): ...                         # L737
    def _setup_routes(self): ...               # L754

# Prefill 侧调用
CommonKVManager.register_to_bootstrap()        # L328 (注册 PrefillServerInfo L48 / PrefillRankInfo L79)

# Decode 侧调用
CommonKVReceiver._get_bootstrap_info_from_server()  # L624
CommonKVReceiver.query_prefill_dp_ranks()          # L644
```

`bootstrap_room` 由 receiver 在 `__init__` 时 caller-supplied（`BaseKVReceiver.__init__` 第三个参数）。

### 3.5 Scheduler 集成（Mixin 模式）

**Decode 节点**（scheduler.py）：
```python
self.disagg_decode_transfer_queue = DecodeTransferQueue(...)   # L1097
self.disagg_decode_prealloc_queue = DecodePreallocQueue(...)  # L1107
```

**Prefill 节点**：
```python
self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(...)  # L1151
self.disagg_prefill_inflight_queue: List[Req] = []               # L1168
```

**队列实现**：
- `disaggregation/prefill.py:87` `PrefillBootstrapQueue`
- `disaggregation/decode.py:241` `DecodePreallocQueue`
- `disaggregation/decode.py:951` `DecodeTransferQueue`
- `disaggregation/decode.py:96` `DecodeReqToTokenPool`（PD 专用 req-to-token pool）
- `disaggregation/decode.py:171` `HybridMambaDecodeReqToTokenPool`

**Mixin 切分事件循环**：
- `prefill.py:355` `class SchedulerDisaggregationPrefillMixin`
  - `event_loop_normal_disagg_prefill` (L389)
  - `event_loop_overlap_disagg_prefill` (L423)
  - `process_batch_result_disagg_prefill` (L468)
  - `process_disagg_prefill_inflight_queue` (L589)
  - `process_prefill_chunk` (L722)
  - `send_kv_chunk` (L750)
- `decode.py:1176` `class SchedulerDisaggregationDecodeMixin`
  - `event_loop_normal_disagg_decode` (L1179)
  - `event_loop_overlap_disagg_decode` (L1206)
  - `get_next_disagg_decode_batch_to_run` (L1254)
  - `get_new_prebuilt_batch` (L1286)
  - `process_decode_queue` (L1338)

**值得照搬的设计**：Mixin 模式让 PD 逻辑不污染 `Scheduler` 主类——sgl-jax 强烈建议沿用。

### 3.6 PD 与 cache（HiCache / UnifiedRadixCache）的真实耦合分析

直觉上「P→D 的 KV 传输」与「HiCache L3 存储」是同构操作——P 把 KV "存"到某处，D "取"回。但代码层的耦合度只成立 **30%**。下面按维度逐项核实：

#### 3.6.1 共享的（30%）— 底层传输引擎

**Mooncake（同进程可共享 TransferEngine 实例）**：

| 路径 | 文件 | 用的 mooncake API |
|---|---|---|
| PD | `disaggregation/mooncake/conn.py:43` `MooncakeKVManager` | `get_mooncake_transfer_engine()` → `MooncakeTransferEngine` (`distributed/device_communicators/mooncake_transfer_engine.py:93`)，走 `transfer_sync_*`（点对点 RDMA） |
| HiCache L3 | `mem_cache/storage/mooncake_store/mooncake_store.py:252` | `from mooncake.store import MooncakeDistributedStore`，K/V Object Store API (`put/get`) |
| **复用机制** | `mooncake_store.py:362-380` | 显式尝试 `get_mooncake_transfer_engine()` 复用底层 TransferEngine 实例（条件：`device_name` / `P2PHANDSHAKE` / `rdma` 协议匹配） |

**结论**：一个进程里 PD + HiCache-mooncake 可以共享底层传输引擎，但走的是两个不同的上层抽象（P2P RDMA vs K/V Store）。

**NIXL（各自独立 agent）**：
- PD: `disaggregation/nixl/conn.py:226` 创建自己的 `nixl_agent`
- HiCache L3: `mem_cache/storage/nixl/hicache_nixl.py:75-77` 创建独立的 `nixl_agent`，名字带 `hicache_nixl_` 前缀
- 不共享，仅同包不同实例

**CLI 允许同时启用 PD + HiCache**：
- `server_args.py:3424` `_handle_hicache` 只规范化 layout/IO
- `_handle_cache_compatibility` 仅检查 `enable_hierarchical_cache ⊥ disable_radix_cache`
- 没有禁止"PD + HiCache 同启"的检查
- 真正的耦合 flag 是 `disaggregation_decode_enable_offload_kvcache`（`server_args.py:3883`）：强制要求 `disaggregation_mode == "decode"` 且 `hicache_storage_backend is not None`——即"D 节点把自己生成的 KV offload 到 L3"

#### 3.6.2 完全独立的（70%）— 上层协议路径

**UnifiedRadixCache 不知道 PD 存在**：
- `unified_radix_cache.py` 1960 行
- `grep -E "disagg|Disagg|kv_send|kv_recv|KVSender|KVReceiver|bootstrap|PD"` **0 命中**

**PD 完全绕过 tree_cache 做 KV 传输**：
```python
# prefill.py:755-844 (send_kv_chunk)
kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, start:end]
req.disagg_kv_sender.send(page_indices, state_indices)
```
直接读 `req_to_token_pool`、调 `KVSender.send`，**与 tree 完全解耦**。

**D 端接收 KV 后不调 `tree_cache.insert`**：
- `DecodeTransferQueue._commit_transfer_to_req` (`decode.py:1384`) 只写 `metadata_buffers`
- tree 插入由 D 完成 decode 后通过 `cache_finished_req` 走标准路径

**HiCache 的 4 个 phase 全部用于 HiCache 内部**：
- `CacheTransferPhase = {BACKUP_HOST, LOAD_BACK, BACKUP_STORAGE, PREFETCH}` 只在 `unified_radix_cache.py` 内部和 `cache_controller` 中使用
- **PD 完全不使用** 这 4 个 phase，走自己的 `disagg_kv_sender` / `KVManager`

#### 3.6.3 唯一现存的 PD↔HiCache 融合实践

**`disaggregation/decode_kvcache_offload_manager.py`** 是当前唯一把 PD 与 HiCache L3 真正打通的代码：

| 维度 | 内容 |
|---|---|
| 位置 | `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py` |
| 启用 | `disaggregation_decode_enable_offload_kvcache=True`（要求 `disaggregation_mode=="decode"` + `hicache_storage_backend != None`） |
| 行为 | D 侧实例化自己的 `MHATokenToKVPoolHost` + `HiCacheController` + `storage_backend`，把 D 自己生成的增量 KV 写到 L3（`offload_kv_cache` L109-180） |
| **语义** | 是 "**D offload 到 L3**"，不是 "**D 从 L3 拉取 P 写入的 KV**" |
| **不改变** | P→D 的初始 prefix KV 仍然走 `MooncakeKVManager` 的 RDMA 直传，与 L3 无关 |

#### 3.6.4 协议层抽象差异（为什么独立不是缺陷）

| 维度 | PD 协议 | L3 协议 |
|---|---|---|
| 寻址方式 | bootstrap 房间号 + req_id（点对点路由） | 内容哈希（content-addressed） |
| 同步模型 | 同步点对点（P 等 D 来 pull） | 异步 put/get |
| 生命周期 | 请求级（一次性） | prefix 级（长期，为 future hit） |
| 数据形态 | 完整 prefill KV（一次大块发送） | 按 page 切分（细粒度寻址） |
| 触发时机 | prefill 完成立即发起 | 由 write-through/write-back 策略决定 |

**核心结论**：PD 和 L3 解耦不是技术缺陷，是协议层抽象不同。复用底层传输引擎可行，但上层协议必须分开。

#### 3.6.5 对 sgl-jax RFC 边界的含义

| RFC | 影响 |
|---|---|
| **RFC-0（共同基础）** | 应该扩展加入「传输底座」抽象层，把 mooncake `TransferEngine` / NIXL `nixl_agent` / JAX `transfer_server` 显式作为可共享的底层组件，但仍区分 P2P-routed vs content-addressed 两种协议层 |
| **RFC-1（HiCache）** | 保持独立，但应预留「D 接收端可注入 storage backend」扩展点（参考 `DecodeKVCacheOffloadManager`） |
| **RFC-2（PD 分离）** | 保持独立。可以在「Alternative」章节讨论"P 写 mooncake-store / D 从 mooncake-store 拉"的方案，但当前 sglang/tpu-inference 都没有这种实践——需要两侧新增 content-addressed PD 传输模式 |

---

## 4. UnifiedRadixCache 与 HiCache 集成

### 4.1 D↔H HiCache 已完整集成

**最新事实**（基于 origin/main，HEAD `f04c52253`）：

- `unified_radix_cache.py` 是 **1960 行**
- PR 链 `#22924 (ABC hooks) → #23316 (HiCache framework) → #23391 (SWA) → #24585 (eviction fix) → #24691 (DeepSeek V4) → #24972 (tombstone fix) → #25088 (load back) → #25277 (device match fix) → #25348 (CI)` 全部 **Merged**
- 三个 component（FULL/SWA/MAMBA）全部实装了 3 个 HiCache hook（`build_hicache_transfers / commit_hicache_transfer / drive_host_eviction`）

**L3 storage 集成缺失**：UnifiedRadixCache 只有 L1↔L2 (D↔H)。`prefetch_from_storage / attach_storage_backend / detach_storage_backend` 仍只在 `HiRadixCache` / `HiMambaRadixCache` 中存在。Scheduler 通过 `hasattr(self.tree_cache, "attach_storage_backend")` 做能力探测（scheduler.py L3226/L3279）。Issue #20415 中「L3 support」复选框未勾。

**对 sgl-jax 的意义**：
- **sgl-jax 作为 port 者**：直接以 origin/main 的 UnifiedRadixCache 为 spec
- L3 prefetch / PD 兼容 / Spec decoding 兼容是 upstream 真正窗口期，sgl-jax 可在这些方向上贡献
- 紧盯 #24691（DeepSeek HiCache 完成度）、L3 support、PD/Spec 兼容这几条线的进度

### 4.2 UnifiedRadixCache 公共接口（origin/main 实际行号）

`mem_cache/unified_radix_cache.py:201` `class UnifiedRadixCache(BasePrefixCache)` — **平行于** `RadixCache` 和 `HiRadixCache`，通过组合 `tree_components` 列表支持多形态。

| 方法 | 行号 | 签名 |
|---|---|---|
| `__init__` | 202 | `(self, params: CacheInitParams)` |
| `init_hicache` | 298 | `(self, server_args: ServerArgs, params: CacheInitParams) -> None` |
| `register_sidecar_pool` | 338 | `(self, spec: SidecarPoolSpec) -> None` |
| `match_prefix` | 341 | `(self, params: MatchPrefixParams) -> MatchResult` |
| `insert` | 368 | `(self, params: InsertParams) -> InsertResult` |
| `evict` | 384 | `(self, params: EvictParams) -> EvictResult` |
| `evict_host` | 1035 | host pool 驱逐 |
| `write_backup` | 1160 | `(self, node, write_back: bool = False) -> int` (D→H) |
| `load_back` | 1223 | (H→D) |
| `writing_check` | 1368 | `(self, write_back: bool = False) -> None` |
| `loading_check` | 1415 | `(self) -> None` |
| `init_load_back` | 1432 | `(self, params: InitLoadBackParams) -> tuple[torch.Tensor, UnifiedTreeNode]` |
| `check_hicache_events` | 1478 | `(self) -> None` (内部调 writing_check + loading_check) |
| `flush_write_through_acks` | 1483 | `(self) -> None` |
| `ready_to_load_host_cache` | 1487 | `(self) -> int` |

**`MatchResult` 字段**（base_prefix_cache.py L145，NamedTuple）：
- `device_indices, last_device_node, last_host_node, best_match_node, host_hit_length, mamba_branching_seqlen, cache_protected_len`
- L2 load_back 锚点用 `best_match_node`，`last_host_node` 是为 L3 prefetch 预留

### 4.3 TreeComponent ABC + HiCache hooks

`mem_cache/unified_cache_components/tree_component.py:364 行`：

```python
class TreeComponent(ABC):
    # 核心 14 个 hook（结构同 sglang hicache_research.md §12.3 描述）
    ...

    # HiCache hooks（origin/main 行号）
    L342  build_hicache_transfers(node, phase, **kw) -> Optional[list[PoolTransfer]]
    L349  commit_hicache_transfer(node, phase, transfers=()) -> None
    L358  drive_host_eviction(num_tokens, tracker) -> None
```

**三个 component 的 HiCache hook 实装**（全部已实现）：

**`full_component.py` (282 行)** — 最简：FULL KV 走主流程
| Hook | 行号 | 行为 |
|---|---|---|
| `drive_host_eviction` | 142 | heap 出 evictable host leaf 调 `_evict_host_leaf` |
| `build_hicache_transfers` | 213 | BACKUP 阶段 return None（主流程处理），LOAD_BACK 沿 evicted 链上溯收集 `host_value` 生成单条 `PoolTransfer(name=KV)` |
| `commit_hicache_transfer` | 253 | device_indices 切片写回各 node `cd.value` + 更新 evictable leaf 集合 |

**`swa_component.py` (537 行)** — 双池 + 窗口
| Hook | 行号 | 行为 |
|---|---|---|
| `build_hicache_transfers` | 427 | BACKUP 返回 `PoolName.SWA`；LOAD_BACK 在 `sliding_window_size` 窗口内只收 host-only 节点 |
| `commit_hicache_transfer` | 482 | 调 `_restore_device_value` + `allocator.set_full_to_swa_mapping` 重建 full↔swa 映射 |
| `drive_host_eviction` | 517 | host LRU 驱动，区分 leaf（`_evict_host_leaf`）vs internal（tombstone + cascade） |

**`mamba_component.py` (448 行)** — 单节点 + CoW
| Hook | 行号 | 行为 |
|---|---|---|
| `build_hicache_transfers` | 341 | BACKUP 返回 `PoolName.MAMBA`；LOAD_BACK 支持单节点 restore + per-request CoW（按需 `mamba_pool.alloc(1)`，OOM 触发 `evict(mamba_num=1)`） |
| `commit_hicache_transfer` | 397 | 写回 `cd.value`，host LRU → device LRU 迁移 |
| `drive_host_eviction` | 426 | 同 SWA 模式，host LRU 驱动 tombstone + cascade |

**`CacheTransferPhase`** 四态：BACKUP_HOST / LOAD_BACK / BACKUP_STORAGE / PREFETCH。其中 `BACKUP_STORAGE / PREFETCH` 在 UnifiedRadixCache 中**尚未被任何 component 处理**——对应 L3 缺失。

### 4.4 UnifiedTreeNode 结构

```
UnifiedTreeNode:
    component_data: list[ComponentData]   # 每 component 一个 slot
    lru_prev: list                        # 每 component 独立 LRU 指针
    lru_next: list
    children
```

特性：节点可同时在多 LRU 中、tombstone 语义、多 component 交集匹配。

---

## 5. 通用扩展点 / 抽象层

### 5.1 BasePrefixCache（见 §2.3）

### 5.2 sglang **没有** `KVConnectorBase_V1`

**易混淆点**：sglang 的 `python/sglang/srt/connector/base_connector.py`：

```python
class BaseConnector(ABC):     # L13
    def weight_iterator(self): ...
    def pull_files(self): ...

class BaseKVConnector(BaseConnector):    # L75
    def get / set / getstr / setstr / list
```

**这是 model weight loading 用的**（Redis 等存储后端），**不是 vLLM 风格的 PD 插件抽象**。

**sglang 的 PD 插件化方式**：
- ABC：`disaggregation/base/conn.py` 4-tuple
- 注册：硬编码 enum + factory（`disaggregation/utils.py:get_kv_class`）
- **无运行时注册机制**

**对 sgl-jax 的含义**：
- 设计 PD 时 **不要照搬 vLLM v1 的 `KVConnectorBase_V1`** 名字（容易和 sglang 的 BaseKVConnector 冲突）
- 可以借鉴 vLLM v1 的「scheduler-worker 双 connector」风格，但接口签名应对齐 sglang 的 4-tuple

### 5.3 HiCacheStorage（见 §2.3）

### 5.4 KVCache（见 §2.3）

---

## 6. Tree Cache 创建：Builder 模式

`scheduler.py` 通过 builder 创建 tree_cache，不在 scheduler 主类里做 dispatch：

```python
# scheduler.py L417
result = kv_cache_builder.build_kv_cache(...)
self.tree_cache = result.tree_cache        # L444
```

**工厂位置**：`python/sglang/srt/mem_cache/kv_cache_builder.py` (318 行)
**入口**：`build_kv_cache()` 行 132

**Dispatch 优先级**：
```
1. disable_radix_cache + chunked      → ChunkCache / SWAChunkCache       (L230/234)
2. SGLANG_EXPERIMENTAL_CPP_RADIX_TREE → RadixCacheCpp                    (L241)
3. SGLANG_ENABLE_UNIFIED_RADIX_TREE   → UnifiedRadixCache                (L256) ★
   └ 按 hybrid 类型注入 tree_components(FULL / +SWA / +MAMBA)
   └ HiCache 通过 tree_cache.init_hicache(server_args, params) 启用 (L258)
4. enable_hierarchical_cache          → HiMambaRadixCache / HiRadixCache (L268/272)
5. is_hybrid_swa                      → SWARadixCache
6. is_hybrid_ssm                      → MambaRadixCache
7. enable_lmcache                     → LMCRadixCache
8. default                            → RadixCache
```

**重要事实**：
- **UnifiedRadixCache 优先级高于 HiRadixCache**（即如果开了 `SGLANG_ENABLE_UNIFIED_RADIX_TREE`，HiCache 走 Unified 路径，不走 HiRadixCache）
- UnifiedRadixCache 内部通过 `init_hicache()` 启用 HiCache——两者不互斥
- UnifiedRadixCache 仍**默认关闭**（env var gated），与旧实现平行存在
- Issue #20415 的「移除其他实现」目标尚未达成

**事件循环 dispatch**（在 scheduler.py 中）：

```python
if disaggregation_mode == NULL:
    if enable_pdmux:        event_loop_pdmux()
    elif pp_size > 1:       event_loop_pp()
    elif enable_overlap:    event_loop_overlap()
    else:                   event_loop_normal()
elif disaggregation_mode == PREFILL:
    if pp_size > 1:         event_loop_pp_disagg_prefill()
    elif enable_overlap:    event_loop_overlap_disagg_prefill()
    else:                   event_loop_normal_disagg_prefill()
elif disaggregation_mode == DECODE:
    if pp_size > 1:         event_loop_pp_disagg_decode()
    elif enable_overlap:    event_loop_overlap_disagg_decode()
    else:                   event_loop_normal_disagg_decode()
```

**互斥关系总结**：
- HiCache **不与** UnifiedRadixCache 互斥（Unified 内部已实装 HiCache）
- HiCache (HiRadixCache 路径) **互斥** UnifiedRadixCache（dispatch 二选一）
- HiCache **互斥** LMCache
- 所有 tree_cache 类型 **都可以叠加** StreamingSession（装饰器）
- 所有 tree_cache 类型 **都可以叠加** PD（独立维度，详见 §3.6）

---

## 7. sgl-jax 对照位置表

| sglang 组件 | sglang 路径 | sgl-jax 对应路径 | 状态 |
|---|---|---|---|
| `BasePrefixCache` | `mem_cache/base_prefix_cache.py:150` | `python/sgl_jax/srt/mem_cache/base_prefix_cache.py:31` | ✅ 已存在，含 HiCache 钩子签名 |
| `MatchResult` (含 last_host_node) | `base_prefix_cache.py:123` | 同上 (line 12) | ✅ 已有 `host_hit_length` / `last_host_node` 字段 |
| `RadixCache` | `radix_cache.py` | `python/sgl_jax/srt/mem_cache/radix_cache.py:145` | ✅ 已有 |
| `SWARadixCache` | `swa_radix_cache.py` | `python/sgl_jax/srt/mem_cache/swa_radix_cache.py:281` | ✅ 已有（双 LRUList） |
| `ChunkCache` | `chunk_cache.py` | `python/sgl_jax/srt/mem_cache/chunk_cache.py:15` | ✅ 已有 |
| `HiRadixCache` | `hiradix_cache.py` | ❌ 不存在 | **RFC-1 待新建** |
| `HiCacheController` | `managers/cache_controller.py` | ❌ 不存在 | **RFC-1 待新建**（含 JAX 适配） |
| `HostKVCache` / `MHATokenToKVPoolHost` | `memory_pool_host.py` | ❌ 不存在 | **RFC-1 待新建**（用 `pinned_host` sharding） |
| `HiCacheStorage` ABC | `hicache_storage.py:98` | ❌ 不存在 | **RFC-1 待新建** |
| 7 个 L3 backend | `storage/*` | ❌ 不存在 | **RFC-1 可选**（先做 file backend） |
| `KVCache.get_cpu_copy/load_cpu_copy` | `memory_pool.py` | `python/sgl_jax/srt/mem_cache/memory_pool.py:540` / `:551` | ⚠️ Stub（已用于 retract，需扩展为 HiCache 真正的 D2H/H2D） |
| `MemoryPools` 容器 | `memory_pool.py:1360` (HostPoolGroup 等价物) | `python/sgl_jax/srt/mem_cache/memory_pool.py:1360` (`MemoryPools` pytree) | ✅ 已有，新增 host pool 可作为同级条目 |
| `UnifiedRadixCache` | `unified_radix_cache.py` (**1960 行**, 含完整 D↔H HiCache 集成) | ❌ 不存在 | **RFC-0/RFC-1 待 port**（直接以 origin/main 为 spec，含三个 component 的 HiCache hook 实装） |
| `TreeComponent` ABC | `unified_cache_components/tree_component.py:364 行` | ❌ 不存在 | **RFC-0 待 port** |
| 3 个 component | `unified_cache_components/{full,swa,mamba}_component.py` (282/537/448 行，**HiCache hook 已实装**) | ❌ 不存在 | **RFC-0/RFC-1 待 port**（注意 full 是最简版本，可作起点） |
| `kv_cache_builder.build_kv_cache()` | `mem_cache/kv_cache_builder.py:132` (318 行) | ❌ 不存在 | **RFC-0 待 port**（builder 模式，避免 scheduler 大 if-elif） |
| 4-tuple PD ABC | `disaggregation/base/conn.py` | ❌ 不存在 | **RFC-2 待新建** |
| `CommonKV*` | `disaggregation/common/conn.py` | ❌ 不存在 | **RFC-2 待新建** |
| `DisaggregationMode/TransferBackend` enum | `disaggregation/utils.py` | ❌ 不存在 | **RFC-2 待新建** |
| `SchedulerDisaggregation{Prefill,Decode}Mixin` | `disaggregation/{prefill,decode}.py` | ❌ 不存在 | **RFC-2 待新建** |
| `DecodeReqToTokenPool` | `disaggregation/decode.py:96` | ❌ 不存在 | **RFC-2 待新建** |
| Scheduler 主循环 | `managers/scheduler.py` | `python/sgl_jax/srt/managers/scheduler.py:127` | ✅ 已有 |
| Tokenizer/Detokenizer manager (ZMQ) | `managers/{tokenizer,detokenizer}_manager.py` | `python/sgl_jax/srt/managers/{tokenizer,detokenizer}_manager.py` | ✅ 已有，可复用做 PD 控制面 |

**统计**：
- 6 个组件 **已存在**（base prefix cache 抽象 + 3 个 cache 实现 + scheduler/tokenizer 框架）
- 1 个组件 **stub**（KVCache D2H/H2D 已有但不完整）
- 13 个组件 **完全缺失**（HiCache 6 个 + Unified 3 个 + PD 4 个）

---

## 8. 给 RFC 的关键 takeaway

1. **UnifiedRadixCache + HiCache D↔H 集成在 sglang 主分支已完整存在**（1960 行 + 三 component 全部 hook 实装）——sgl-jax 应作为 **port 者**。
2. **L3 storage 集成是 upstream 真正的缺口**：UnifiedRadixCache 没有 `prefetch_from_storage / attach_storage_backend`，只在 HiRadixCache 里有。sgl-jax 的 L3 工作有窗口期，但短期不在关键路径。
3. **sgl-jax 直接采用 builder 模式**（参考 `kv_cache_builder.build_kv_cache()`），避免 scheduler 大 if-elif 链。
4. **sgl-jax 的 day-one 策略**：直接把 UnifiedRadixCache 当默认（甚至唯一）prefix cache。不引入 sglang 多套并存（HiRadixCache / SWARadixCache / MambaRadixCache）的历史包袱。
5. **PD 抽象用 4-tuple，不是 vLLM 风格**：避免与 sgl-jax 未来的 `BaseKVConnector` (model weight) 冲突。
6. **PD 与 cache 在主路径上独立**（详见 §3.6）：sgl-jax 的 PD RFC 与 HiCache RFC 应保持独立模块；共同基础（mooncake TransferEngine / NIXL agent / JAX transfer server）作为底座层共享。
7. **Mixin 模式分隔事件循环**：PD 逻辑不污染 Scheduler 主类。值得照搬。
8. **Backend 注册**：sglang 用硬编码 enum，sgl-jax 可以做 entry_points / decorator 注册表（小优化但有价值）。
9. **HiCacheStorage v2 API + PoolTransfer 描述符**：直接对齐 sglang，避免后续要兼容 v1 的麻烦。
10. **sgl-jax 已有的 hooks 是宝贵起点**：`MatchResult.last_host_node` / `host_hit_length` / `KVCache.get_cpu_copy` 等不需要新增字段——但要扩展 `MatchResult` 加上 `best_match_node`（origin/main 已加）。
11. **`DecodeKVCacheOffloadManager` 是唯一的 PD↔HiCache 融合实践**：D 端把自己生成的 KV offload 到 L3。可以作为 RFC-1 中预留扩展点的参考实现。
12. **紧盯 upstream 节奏**：#24691 (DeepSeek HiCache) → L3 support → PD/Spec 兼容 是接下来 3-6 个月的关键，sgl-jax port 需要持续 rebase。

---

**配套阅读**：
- `../../workspace/sglang/hicache_research.md` — sglang HiCache 数据流细节、调用链、写入策略
- `2026-05-18-tpu-inference-jax-api-survey.md` — tpu-inference 的 JAX/Pallas API 调研（待写）
- `../../workspace/tpu-inference/kv_cache_offload_analysis.md` — TPU offload 实现（含 sglang HiCache Gap 分析 §10）
- `../../workspace/tpu-inference/pd_disaggregation_analysis.md` — TPU PD 分离（含与 GPU NIXL 对比 §8）
- GitHub Issue: https://github.com/sgl-project/sglang/issues/20415 — Unified Hybrid Radix Cache Refactor Roadmap
