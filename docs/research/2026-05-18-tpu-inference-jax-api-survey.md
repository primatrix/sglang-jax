# tpu-inference JAX/Pallas API 调研：KV Offload + PD 分离视角

> **目的**：为 sgl-jax 即将编写的 HiCache / PD 分离 / 共同基础设施 RFC 提供「JAX/Pallas API 选型」的对标参考。本文按**使用场景**整理 tpu-inference 仓库中实际用到的 JAX/Pallas API、签名、约束、与 SGLang CUDA API 的对应关系。
>
> **配套**：与 `2026-05-18-sglang-cache-pd-organization.md` 一起作为 RFC 输入。
>
> **代码基准**：`/Users/jiongxuan/workspace/tpu-inference` 当前 main HEAD，日期 2026-05-18。所有路径相对该仓库根目录。
>
> **省略**：API 详细数据流、Gap 分析、性能数据——这些已在 `kv_cache_offload_analysis.md`（830 行）和 `pd_disaggregation_analysis.md`（1140 行）覆盖。本文只做「API 选型决策表」。

---

## 0. TL;DR

| 决策项 | tpu-inference 用法 | 给 sgl-jax 的建议 |
|---|---|---|
| **D2H/H2D** | `jax.device_put(tensor, NamedSharding(..., memory_kind="pinned_host"))` | **直接采用**——比 Pallas `copy_to_host` 简单且统一；只在性能临界场景考虑 Pallas |
| **D2D scatter (paged)** | Pallas `multi_layer_copy`（`pltpu.make_async_copy` + `SemaphoreType.DMA`） | **必须采用**——`dynamic_update_slice_in_dim` 性能更差，且无法跨 layer 流水线 |
| **跨进程 KV transfer** | `jax.experimental.transfer.start_transfer_server` + `await_pull` / `connect+pull` | **采用**，但要意识到：是**实验性 API**，无公开文档；不支持 partial pulling；考虑封装一层 sgl-jax 自己的 ABC 隔离风险 |
| **ICI 跨 mesh transfer** | `jax.device_put(data, target_sharding_on_other_mesh)` | 仅在**单进程多 mesh**场景可用（架构 1）；多进程下 ICI 不可用 |
| **Host Memory 类型** | 默认 `pinned_host`，可选 `unpinned_host`（`TPU_OFFLOAD_USE_UNPINNED_HOST`） | 默认用 `pinned_host`（PD 强制，offload 默认）；容量优先场景留 `unpinned_host` 开关 |
| **异步并行** | `ThreadPoolExecutor` + `jax.block_until_ready`（GIL 在 JAX C++ 调用中释放） | 直接采用；不需要 CUDA stream 等价物 |
| **JIT 优化点** | `donate_argnames` + `optimization_barrier`（防止 gather 被 XLA 重排到 buffer 重用之后） | **必须知道**；HiCache 的 KV gather 场景一定要加 barrier |
| **Layer-wise Overlap** | **不支持**（XLA 静态编译 + JIT forward） | **不要尝试**；做 step-level / request-level overlap 即可 |
| **GPUDirect RDMA 对应物** | **没有**——所有跨进程 KV 必须经 host 内存 | RFC 中明确这是 TPU 硬约束，不是设计缺陷 |

---

## 1. JAX/Pallas API 索引（按使用场景）

| 场景 | API | 模块 | 关键位置 |
|---|---|---|---|
| **HBM 内 gather** | `jax.Array.at[ids].get()` + `jnp.stack` + `jnp.split` | `jax`, `jax.numpy` | `offload/utils.py:97` `stack_kv_cache_cross_layers` |
| **HBM 内 select (PD)** | 同上 | 同上 | `distributed/tpu_connector.py:913` `select_from_kv_caches` |
| **HBM 内 scatter** | `pltpu.make_async_copy().start()/.wait()` via `shard_map` | `jax.experimental.pallas.tpu`, `jax.experimental.shard_map` | `distributed/kv_transfer.py:296` `multi_layer_copy` |
| **HBM 内 scatter (slice)** | `jax.lax.dynamic_update_slice_in_dim` | `jax.lax` | `offload/utils.py:65` `jitted_insert_kv_cache_slices` |
| **HBM 内 slice (gather)** | `jax.lax.dynamic_slice_in_dim` | `jax.lax` | `runner/kv_cache_manager.py:950` |
| **D2H (offload 用)** | `jax.device_put(tensor, host_sharding)` (memory_kind=`pinned_host`) | `jax` | `offload/tpu_offload_connector.py:1765` |
| **D2H (PD 用)** | Pallas `copy_to_host(src, dest_host_buffer)` | `distributed/kv_transfer.py:416` | 自定义 kernel |
| **H2D** | `jax.device_put(cpu_chunk, device_sharding)` | `jax` | `offload/tpu_offload_connector.py:2107` |
| **ICI 跨 mesh 传输** | `jax.device_put(data, target_sharding)` | `jax` | `runner/kv_cache_manager.py:1106` `transfer_kv_cache` |
| **ICI 跨 mesh 传输 (Pathways)** | `pathwaysutils.experimental.reshard.reshard(...)` | 外部 lib | `runner/kv_cache_manager.py:1100` |
| **跨进程 server 启动** | `jax.experimental.transfer.start_transfer_server` | `jax.experimental.transfer` | `distributed/tpu_connector.py:573` |
| **跨进程 Producer 发布** | `server.await_pull(uuid, data)` | 同上 | `tpu_connector.py:710/760` |
| **跨进程 Consumer 连接** | `server.connect(remote_addr)` | 同上 | `tpu_connector.py:772` |
| **跨进程 Consumer 拉取** | `conn.pull(uuid, spec)` | 同上 | `tpu_connector.py:793` |
| **Sharding 构造** | `jax.sharding.NamedSharding(mesh, P(...))` | `jax.sharding` | `tpu_offload_connector.py:1274` 等 |
| **Mesh 构造** | `jax.sharding.Mesh(devices, axis_names)` | `jax.sharding` | `runner/...` |
| **同步等待** | `jax.block_until_ready(arr_or_list)` | `jax` | 各处 |
| **JIT 优化 (输入 buffer 捐赠)** | `@jax.jit(donate_argnames=(...))` | `jax` | `offload/utils.py:104` |
| **防编译融合** | `jax.lax.optimization_barrier(tensor)` | `jax.lax` | `offload/utils.py:117` |
| **分 shard 执行** | `jax.experimental.shard_map.shard_map(...)` | `jax.experimental.shard_map` | `kv_transfer.py:187` `_async_copy_jit` 内部 |
| **DMA semaphore** | `pltpu.SemaphoreType.DMA` | `jax.experimental.pallas.tpu` | `kv_transfer.py:32-156` |
| **Host 内存空间标记** | `pltpu.HOST`（Pallas memory space） | `jax.experimental.pallas.tpu` | `kv_transfer.py:416` `copy_to_host` 内部 |

---

## 2. KV 数据搬运（按方向）

### 2.1 D2D — Device 内 HBM ↔ HBM

#### 2.1.1 Gather（按索引读）

**场景**：从 paged KV cache pool（共享、非连续）按 block_ids 取出指定 block 的数据。

```python
# offload/utils.py:104-114
@functools.partial(
    jax.jit,
    static_argnames=['num_blocks'],
    donate_argnames=('kv_caches',),       # 关键：允许 XLA 复用输入 buffer
)
def stack_kv_cache_cross_layers(kv_caches, block_ids, num_blocks):
    def _gather_blocks(layer_kv_cache):
        return layer_kv_cache.at[block_ids].get()

    gathered_kv_layers = jax.tree.map(_gather_blocks, kv_caches)
    stacked_blocks = jnp.stack(gathered_kv_layers, axis=1)
    split_blocks = jnp.split(stacked_blocks, num_blocks, axis=0)
    kv_caches = jax.lax.optimization_barrier(kv_caches)   # 关键：防止 XLA 重排
    return kv_caches, split_blocks
```

**坑点 1**：必须用 `optimization_barrier`，否则启用 `donate_argnames` 后 XLA 可能把 gather 重排到 buffer 被覆盖之后。
**坑点 2**：JIT 命中需要 block 数量分桶（如 `[1, 2, 4, 8, 16, 32, 64]`），否则每次新 N 都重新编译。

**SGLang CUDA 对应物**：自定义 CUDA gather kernel；JAX 的 `at[].get()` 是相同语义但由 XLA 编译，性能基本对等。

#### 2.1.2 Scatter（按索引写，paged KV cache 安装）

**核心 API**：`multi_layer_copy`（`distributed/kv_transfer.py:296`）— Pallas async DMA，跨 layer 流水线化。

```python
def multi_layer_copy(
    *,
    src_array: list[jax.Array],
    dest_array: list[jax.Array],
    src_offsets: jax.Array,
    dest_offsets: jax.Array,
    chunk_sizes: jax.Array,
    num_chunks: jax.Array | None = None,
    mesh: Mesh | None = None,
    src_sharding_spec: P | None = None,
    dest_sharding_spec: P | None = None,
    replicated_sharding_spec: P | None = None,
)
```

**实现层级**：
```
multi_layer_copy                  (kv_transfer.py:296)
  └─ _async_copy_jit              (kv_transfer.py:187, JIT 入口)
       └─ jax.shard_map 内部:
            ├─ _start_chunked_copy_kernel      ← Pallas kernel
            │    └─ pltpu.make_async_copy(...).start()
            │
            └─ _wait_for_chunked_copy_kernel   ← Pallas kernel
                 └─ pltpu.make_async_copy(...).wait()
```

**Pallas async DMA 原语**：
```python
# 简化的 Pallas kernel 伪代码
async_copy = pltpu.make_async_copy(
    src_ref.at[src_slice],
    dest_ref.at[dest_slice],
    semaphore_ref,
)
async_copy.start()                # 非阻塞启动 DMA
# ... 其他工作 ...
async_copy.wait()                 # 等待 DMA 完成
```

**`pltpu.SemaphoreType.DMA`** 用于同步。多个 `make_async_copy` 可以**重叠执行**——这是 KV scatter 性能的关键。

**SGLang CUDA 对应物**：自定义 CUDA scatter kernel（`sgl-kernel/csrc/kvcacheio/transfer.cu`）。JAX 用 Pallas 实现，**性能等价但 API 形态不同**——Pallas 编程模型更接近 OpenAI Triton。

#### 2.1.3 Scatter（连续 slice 写入）

**替代方案**：当 dest blocks 在物理上连续时，可以用 `jax.lax.dynamic_update_slice_in_dim`：

```python
# offload/utils.py:65
jitted_insert_kv_cache_slices = jax.jit(
    lambda dest, src, start, axis: jax.lax.dynamic_update_slice_in_dim(
        dest, src, start, axis
    )
)
```

**对比**：
- `dynamic_update_slice_in_dim`：简单、纯 JAX，但**只能更新连续 slice**，无法跨 layer pipeline
- `multi_layer_copy`：Pallas DMA，**可跨 layer 重叠**，但实现复杂

**sgl-jax 建议**：HiCache load 走 `multi_layer_copy`（多 block + 跨 layer），retract / single-block 操作可用 `dynamic_update_slice_in_dim`。

#### 2.1.4 ICI 跨 Mesh 传输（仅单进程多 mesh）

```python
# runner/kv_cache_manager.py:1106 (transfer_kv_cache, 标准 JAX 路径)
sharding = NamedSharding(self.runner.mesh, PartitionSpec(None, ShardingAxisName.ATTN_HEAD))
transferred_kv_cache = jax.device_put(kv_cache_slices, sharding)
jax.block_until_ready(transferred_kv_cache)
```

**机制**：`jax.device_put(data, target_sharding)` 把数据从源 sharding 重新放置到目标 sharding；如果两个 mesh 在同一 JAX runtime 内、且物理芯片之间有 ICI 直连，**JAX/XLA 会自动走 ICI**。

**Pathways 路径**：
```python
# runner/kv_cache_manager.py:1100
from pathwaysutils.experimental import reshard as experimental_reshard
transferred = experimental_reshard.reshard(
    kv_cache_slices, sharding_spec_pytree, donate=False
)
```

**关键约束**：**仅当 P 和 D 在同一 JAX runtime 内**（架构 1 / In-Process PD）才可用。架构 2（多进程 PD）下，每个进程的 `jax.devices()` 只包含本进程的芯片子集，无法做跨 runtime 的 ICI 传输——必须走 `jax.experimental.transfer`。

### 2.2 D2H — HBM → Host DRAM

#### 2.2.1 标准路径：`jax.device_put`（推荐）

```python
# offload/tpu_offload_connector.py:1763-1770
# host_sharding 在 register_runner 时构造（line 1274-1317）：
host_sharding = NamedSharding(
    mesh,
    P(None, None, "model"),        # 只在 model (TP) 轴 partition
    memory_kind="pinned_host",     # ← 关键：告诉 JAX 放到 pinned host memory
)

# D2H 调用
chunks_on_cpu = []
for i in range(total_num_blocks):
    chunks_on_cpu.append(
        jax.device_put(flat_kv_caches_tpu[i], host_sharding)
    )
jax.block_until_ready(chunks_on_cpu)
```

**机制**：JAX runtime 知道每个 device 的 host affinity，自动把 shard 路由到对应 host 的 pinned CPU 内存。**不需要显式多节点协调代码**。

**好处**：纯 JAX API，跨平台（CPU/GPU/TPU 一致）；自动管理多 host 路由。

**坑点**：每次 `jax.device_put` 返回新的 `jax.Array`，引用计数会保活 pinned memory，必须显式管理（LRU 淘汰时丢弃引用才能释放）。

#### 2.2.2 Pallas 路径：`copy_to_host`（PD 专用）

```python
# distributed/kv_transfer.py:416 (函数签名简化)
def copy_to_host(src: jax.Array, dest: jax.Array, ...) -> jax.Array:
    # Pallas kernel 内部：
    # async_copy = pltpu.make_async_copy(src_ref, dest_ref_in_pltpu.HOST, semaphore_ref)
    # async_copy.start(); async_copy.wait()
```

**特点**：
- Pallas kernel 直接操作 DMA 引擎，逐 layer 拷贝
- 目标是**预分配的 pinned host buffer**（`HostKVPool`），避免每次传输的 `MapDmaBuffer` 系统调用开销
- 配合 `is_ready()` 轮询用作传输完成检查（非 `block_until_ready`）

**为什么 PD 用 Pallas、Offload 用 `jax.device_put`**：
- Offload 是后台异步、不在主线程关键路径，`device_put` 简单够用
- PD 在请求关键路径上，对每毫秒延迟敏感；预分配 buffer + Pallas 可避免每次系统调用，且 `is_ready()` 给更细粒度的进度查询

**sgl-jax 建议**：
- HiCache D2H：**先用 `jax.device_put`**，简单稳定；只在 profile 显示瓶颈时再考虑 Pallas
- PD D2H：**直接用 `copy_to_host`** 或类似 Pallas 方案，因为 PD 对延迟敏感

#### 2.2.3 `pinned_host` vs `unpinned_host`

| `memory_kind` | 特点 | 何时用 |
|---|---|---|
| `'device'` | TPU HBM | 默认计算时 |
| `'pinned_host'` | OS 锁页内存（不可 swap），DMA 友好 | **PD 强制用此**（传输延迟敏感）；offload 默认 |
| `'unpinned_host'` | 可 swap 的普通 host memory | offload 容量优先场景（`TPU_OFFLOAD_USE_UNPINNED_HOST=true`） |

**坑点**：`unpinned_host` 在 swap 时性能急剧下降；如果 KV 容量远小于 host RAM 总量，永远用 `pinned_host`。

### 2.3 H2D — Host DRAM → HBM

```python
# offload/tpu_offload_connector.py:2077-2082
raw_chunked_kv_on_tpu = []
for i in range(num_blocks_to_load):
    raw_chunked_kv_on_tpu.append(
        jax.device_put(assembled_kv_on_cpu[i], device_sharding)
    )
jax.block_until_ready(raw_chunked_kv_on_tpu)
```

**对称于 D2H**：同样用 `jax.device_put`，目标 sharding 是 `device_sharding`（`memory_kind="device"`）。

**完整 H2D 流程**：
```
LocalCPUBackend (OrderedDict[chunk_id, jax.Array])
    ↓ Python dict lookup (无数据搬运)
jax.Array (pinned_host)
    ↓ jax.device_put(..., device_sharding)     ← H2D 实际传输
jax.Array (device HBM, "staging")
    ↓ multi_layer_copy (Pallas DMA)            ← Scatter 到 paged KV cache
jax.Array (device HBM, paged pool 内具体 blocks)
```

**注意**：H2D 后还需要一次 D2D scatter（`multi_layer_copy`）把数据写入 paged KV cache 的指定 block 位置，因为 H2D 目标是 staging 区域。

### 2.4 H2H-network — 跨主机网络传输（PD 专用）

#### 2.4.1 实验性 API 警告

**`jax.experimental.transfer` 是内部实验性 API，截至 2026-05 几乎没有公开文档**。所有用法基于 tpu-inference 仓库的代码行为和 TPU 硬件架构推断。

**Risk**：
- API 签名可能在 JAX 版本升级时改变
- 错误处理路径未文档化
- 多 host 并发拉取的行为细节未文档化

**sgl-jax 应对**：封装一层自己的 ABC（如 `KVNetworkBackend`），底层默认走 `jax.experimental.transfer`；未来若需切换到其他传输库（如自研、xla.runtime 公开 API）可只改实现不动接口。

#### 2.4.2 服务器生命周期

```python
# distributed/tpu_connector.py:573
from jax.experimental.transfer import start_transfer_server

self.kv_transfer_server = start_transfer_server(
    jax.local_devices()[0].client,
    server_addr,                                       # 控制地址 "ip:port"
    [transport_addr] * get_transfer_channel_number(), # 数据通道列表（默认 8 个）
    max_num_parallel_copies=8,
    transfer_size=256 * 1024 * 1024,                   # 256 MB
    use_raw_buffers=False,
)
```

**参数说明**：
- `client`：从 `jax.local_devices()[0].client` 拿到的 PJRT 客户端
- `server_addr`：TCP `"host_ip:port"` 控制地址
- `transport_addr` 列表：数据通道地址列表（每个一条 TCP 连接）；通道数由 `TPU_KV_TRANSFER_CHANNEL_NUMBER` 控制
- `transfer_size`：单次传输 chunk 大小，256 MB
- `max_num_parallel_copies`：并发拷贝数，8
- `use_raw_buffers`：是否绕过 JAX 的高层抽象（实验性）

#### 2.4.3 Producer 侧（P 节点）

```python
# tpu_connector.py:710 或 760
self.kv_transfer_server.await_pull(uuid, kv_data)
```

**参数**：
- `uuid`：唯一传输 ID（用 Python `uuid.uuid4()` 生成）
- `kv_data`：要发布的 `jax.Array`（或 list[Array]）；可以在 HBM 上也可以在 pinned host 上
- 阻塞直到 Consumer pull 完成或超时

#### 2.4.4 Consumer 侧（D 节点）

```python
# tpu_connector.py:772 - 建立连接
conn = self.kv_transfer_server.connect(f"{remote_host}:{remote_port}")

# tpu_connector.py:793 - 拉取数据
kv_spec = [
    jax.ShapeDtypeStruct(
        shape=(num_blocks, *cache_inner_shape),
        dtype=dtype,
        sharding=self.sharding,   # ← device sharding → 数据直接落到 HBM
    )
    for _ in range(num_layers)
]
kv = conn.pull(uuid, kv_spec)
```

**关键**：`kv_spec.sharding` 决定数据落在哪儿。如果是 device sharding，数据直接到 Consumer 的 HBM；如果是 host sharding，落到 pinned host memory。

**坑点**：
- **不支持 partial pulling**（不像 RDMA）：即使 D 端有 prefix cache 命中，也必须拉取全部 prefill block
- **不支持 native async pulling**：当前用 ThreadPoolExecutor + `is_ready()` 轮询模拟
- 拉取完成检查：`chunk.is_ready()`（间隔 1ms 轮询），非 `block_until_ready`

#### 2.4.5 数据路径（重要事实）

**架构 2 PD 分离不能做真正的 device-to-device 直传**——即使同一 pod 内芯片有 ICI 物理直连，由于 P 和 D 处于不同 JAX runtime，**必须经过 host 内存**：

```
P TPU Chip (HBM)
       │ PCIe
       ↓
P Host Memory (pinned_host)
       │ DCN / TCP
       ↓
D Host Memory
       │ PCIe
       ↓
D TPU Chip (HBM)
```

**与 GPU 对比**：GPU 有 GPUDirect RDMA（NIC 直接读写 GPU VRAM 绕过 host），TPU 没有对应物。这是**硬件约束**，不是软件设计选择。

---

## 3. Sharding / Mesh / Memory Placement

### 3.1 Mesh 构造

```python
# 单进程
mesh = jax.sharding.Mesh(
    devices=jax.devices(),                      # 全部 host 全部芯片
    axis_names=("data", "attn_dp", "attn_dp_expert", "expert", "model"),
)

# 多进程（PP 场景）
mesh = jax.sharding.Mesh(
    devices=jax.local_devices(),                # 仅本 host
    axis_names=(...,),
)
```

**判断单/多 host**：`len(jax.local_devices()) < len(jax.devices())` 即多 host。

### 3.2 NamedSharding + PartitionSpec

**KV cache 的标准 sharding**（tpu-inference）：

```python
device_sharding = NamedSharding(
    mesh,
    PartitionSpec(None, None, "model"),     # 仅 KV head 维度沿 TP 轴切分
    memory_kind="device",
)
# Tensor shape: [num_blocks, block_size, num_head, 2, head_dim]
# 实际分片：dim[2] 沿 mesh axis "model"
```

**对应的 host sharding**：
```python
host_sharding = NamedSharding(
    mesh,
    P(None, None, "model"),                  # 与 device 同样的 partition
    memory_kind="pinned_host",               # 唯一差别：放 pinned host memory
)
```

**核心公式**：
```
物理放置 = Mesh (设备拓扑) + PartitionSpec (逻辑切分) + memory_kind (存储层级)
```

JAX 通过 device-host affinity 自动路由 D2H/H2D。

**sgl-jax 现状对照**：
- sgl-jax 的 KV pool 已经用 `NamedSharding(mesh, P("data", None, "tensor", None, None))`（memory_pool.py:410）
- axis 命名是 `("data", "tensor")` 而非 tpu-inference 的 `("data", ..., "model")`——sgl-jax 实现 HiCache 时直接 `memory_kind="pinned_host"` + 相同 PartitionSpec 即可

### 3.3 Memory Kind 完整对照

| kind | 物理位置 | DMA 友好 | 何时用 |
|---|---|---|---|
| `"device"` | HBM | ✓ | 默认计算 |
| `"pinned_host"` | 锁页 host DRAM | ✓ | HiCache L2、PD host buffer |
| `"unpinned_host"` | 可 swap host DRAM | ✗ | Offload 容量优先 |

---

## 4. 跨进程传输（详见 §2.4）

略，避免重复。

---

## 5. 异步与并行

### 5.1 JAX 默认异步派发

**关键事实**：JAX 的所有计算（jit 编译后的函数、`jax.device_put`、Pallas kernel）都是**异步派发**的——调用立即返回，实际计算在后台进行。`block_until_ready()` 才是真正的同步点。

**含义**：
```python
result = jit_fn(args)              # 立即返回（kernel 已派发但未执行）
# 此时 Python 继续执行其他代码（与 TPU 并行）
result.block_until_ready()         # 阻塞直到 TPU 完成
```

**与 CUDA stream + event 对比**：
- CUDA：用户必须显式管理 stream/event 才能异步
- JAX：默认全部异步；用户只需要在最后 `block_until_ready` 同步

### 5.2 ThreadPoolExecutor 模式（多线程异步）

```python
# 主线程
from concurrent.futures import ThreadPoolExecutor

self.save_executor = ThreadPoolExecutor(max_workers=4)

# 提交后台任务（不阻塞主线程）
future = self.save_executor.submit(
    self._async_transfer_task, args...
)

# 后台线程
def _async_transfer_task(self, ...):
    # 这里调用 jax.device_put + block_until_ready
    chunks_on_cpu = jax.device_put(..., host_sharding)
    jax.block_until_ready(chunks_on_cpu)        # 仅阻塞后台线程
    self.cpu_backend.add(chunk_id, chunks_on_cpu)
```

**GIL 行为**（重要！）：
| 操作 | 持有 GIL？ |
|---|---|
| `jax.device_put()` 派发 | 短暂持有，C++ 层释放 |
| `jax.block_until_ready()` 等待 | **释放 GIL** |
| Python dict 操作 | 持有 GIL |
| JIT 编译后的 model forward | **释放 GIL**（C++ 层执行） |

**结论**：主线程的 model forward 和后台线程的 D2H 可以**真正并行**，因为两者大部分时间都在 GIL 之外。

**与 CUDA 对比**：CUDA 也有类似行为（PyTorch CUDA op 释放 GIL），但需要额外用 stream/event 协调；JAX 默认就是这种模式。

### 5.3 JIT 优化点

#### 5.3.1 `donate_argnames`（关键！）

```python
@jax.jit(donate_argnames=('kv_caches',))
def stack_kv_cache_cross_layers(kv_caches, block_ids, num_blocks):
    ...
    return kv_caches, split_blocks       # kv_caches 同时作为输出返回
```

**含义**：告诉 XLA「输入 `kv_caches` 的内存 buffer 可以被输出复用」，实现 zero-copy 返回。

**坑点**：必须配合 `optimization_barrier` 使用，否则 XLA 可能把输入 buffer 重用提前到读操作之前。

#### 5.3.2 `optimization_barrier`（关键！）

```python
kv_caches = jax.lax.optimization_barrier(kv_caches)
return kv_caches, split_blocks
```

**含义**：编译器屏障，确保 barrier 之前的所有读操作完成后，buffer 才允许被「donated」（复用/覆盖）。

**何时必须用**：
- 在 `donate_argnames` 标注的输入上做了读操作（如 `at[].get()`）
- 该输入同时作为输出返回（identity 优化目标）
- 外部后续操作会写入该 buffer

**典型场景**：HiCache 的 KV gather——`kv_caches` 既被读（gather 取 block）又被返回（buffer 复用），必须 barrier 防止下一步 forward 写入与 gather 读取竞争。

#### 5.3.3 静态参数和分桶

```python
@functools.partial(
    jax.jit,
    static_argnames=['num_blocks'],   # 不同 num_blocks 触发重编译
)
```

**坑点**：`num_blocks` 是 Python int，不能传 `jnp.array` 进来；动态值要做分桶（如 1/2/4/8/16/32/64）控制编译实例数。

### 5.4 SGLang CUDA 异步原语 → JAX 对应物

| SGLang (CUDA) | JAX 等价 |
|---|---|
| `cudaStream_t` (write_stream, load_stream) | 不需要——JAX 默认异步派发 |
| `cudaEvent_t.record()` | 不需要——`jax.Array` 隐含完成事件 |
| `cudaEventQuery()` | `arr.is_ready()`（Pallas async copy 产物）或 `future.done()`（ThreadPoolExecutor） |
| `cudaEventSynchronize()` | `jax.block_until_ready(arr)` |
| `cudaMemcpyAsync(D2H)` | `jax.device_put(arr, host_sharding)` |
| `cudaMemcpyAsync(H2D)` | `jax.device_put(arr, device_sharding)` |
| `<<<grid, block>>>` kernel launch | `jit_fn(args)` |
| 自定义 CUDA kernel | Pallas kernel (`@pl.kernel` + `pltpu.make_async_copy` 等) |
| Pinned memory (`cudaMallocHost`) | `memory_kind="pinned_host"` |
| `cudaStreamWaitEvent` (跨 stream 同步) | 不需要——JAX 自动管理依赖 |
| Layer-wise overlap (per-layer event) | **无对应物**——XLA 静态编译，无法在 JIT 内插同步点 |
| GPUDirect RDMA | **无对应物**——必须经 host memory |
| NVLink (单 process) | ICI (单 process)——`jax.device_put` 自动用 |
| NVLink (跨 process) | **不可用**——同 GPU 跨进程 NVLink 也有同样问题 |

---

## 6. 已知约束 / 实验性 API 风险

### 6.1 `jax.experimental.transfer` 实验性

- 截至 2026-05，**没有公开文档**
- API 签名可能改变
- 错误处理路径未文档化
- 不支持 partial pulling（影响 PD + prefix cache 集成）
- 不支持原生 async pulling（需 ThreadPoolExecutor + 轮询）
- 多 host 并发拉取行为细节未文档化

**风险缓解**：
- 封装 sgl-jax 自己的 `KVNetworkBackend` ABC
- 短期内只支持单机 PD（避免触发未知多 host 路径）
- 准备 fallback：用 ZMQ + 显式 `jax.device_get`/`device_put` 实现一个备用 backend

### 6.2 Layer-wise Overlap 不可行

SGLang HiCache 的核心性能优化之一是 **layer-wise H2D overlap**：传输完 layer N 立即开始 layer N 的 forward 计算，与 layer N+1 的传输重叠。

**TPU/XLA 下不可行**，原因：
1. **XLA 静态编译**：`jax.jit` 把整个 model forward 编译成一个完整 HLO 图，中间无法插 host 同步点
2. **无 CUDA stream 等价物**：TPU 没有多 stream 概念，DMA 引擎和 MXU 调度由 XLA 统一管理
3. **改造代价极高**：把 jit forward 拆成 per-layer 小函数会破坏算子融合、内存规划、编译时间爆炸

**替代方案**：
- **Step-level overlap**：request A 的 forward 与 request B 的 H2D 并行（用 ThreadPoolExecutor，与现有 D2H 异步对称）
- **Request-level overlap**：先收集足够多的"可加载"请求，批量异步加载，与当前 batch forward 并行

**结论**：sgl-jax HiCache 不应承诺 layer-wise overlap，但应实现 step-level overlap。

### 6.3 SPMD 强约束

JAX 的 SPMD 模型要求**每个 host 运行相同 Python 代码**。这对 HiCache / PD 设计的影响：

- **Tree 结构必须 host 一致**：所有 host 上的 RadixCache（或 UnifiedRadixCache）逻辑状态必须一致；可以通过「只在 leader 上做决策 + 广播」或「所有 host 独立维护 + 隐式一致性」实现
- **Storage backend 注册**：所有 host 必须看到一致的 backend 配置，否则 hash key 不一致会出错
- **PD 控制面**：bootstrap server 只能跑在某个 host 上，其他 host 通过它做协商

**sgl-jax 现状**：scheduler 是单进程的（每个 DP group 一个 scheduler 进程），tree 由 scheduler 维护，worker 只接收 block_id 列表执行 gather/scatter——这与 SPMD 约束兼容，可以照搬。

### 6.4 没有 GPUDirect RDMA 对应物

所有跨进程 KV 传输必须经 host 内存。**这是硬件约束**，在 RFC 中应明确：
- PD 分离的网络延迟比 GPU 高一截
- 多 host 部署时这是主要瓶颈
- 缓解：增大 batch、减少传输次数、用更高带宽 DCN

---

## 7. RFC 选型建议汇总

| 场景 | 推荐 API | 备注 |
|---|---|---|
| HiCache D2H (L1→L2) | `jax.device_put(arr, host_sharding)` + `block_until_ready` | 后台线程异步，主线程 forward 并行 |
| HiCache H2D (L2→L1) | `jax.device_put(arr, device_sharding)` + `multi_layer_copy` scatter | H2D 到 staging，再 D2D scatter 到 paged pool |
| HiCache L2 容器 | `OrderedDict[hash, jax.Array]`（参考 LocalCPUBackend） | jax.Array 持有 pinned host memory 引用 |
| HiCache L3 接口 | 沿用 sglang `HiCacheStorage` ABC | 先实现 file backend |
| KV cache gather (paged) | `at[].get()` + `jnp.stack` + `jnp.split` (JIT + `donate_argnames` + `optimization_barrier`) | 分桶 + barrier 是关键 |
| KV cache scatter (paged) | `multi_layer_copy` (Pallas) | 比 `dynamic_update_slice_in_dim` 性能更优 |
| PD D2H (单机 staging) | Pallas `copy_to_host` | 延迟敏感，预分配 buffer 避免系统调用 |
| PD 跨进程传输 | `jax.experimental.transfer` 封装 ABC | 实验性，做 fallback 计划 |
| PD bootstrap server | HTTP server（参考 sglang `CommonKVBootstrapServer`） | sgl-jax 已有 ZMQ 框架，可复用做轻量协议 |
| Sharding 标注 | `NamedSharding(mesh, P(...), memory_kind=...)` | 三态：`device` / `pinned_host` / `unpinned_host` |
| 异步并行 | `ThreadPoolExecutor` + JAX 默认异步派发 | GIL 在 JAX C++ 调用中释放，主/后台真正并行 |
| 同步等待 | `jax.block_until_ready(arr)` | 仅在必要时；前台轮询用 `is_ready()` |
| Layer-wise overlap | **不实现** | 改做 step/request-level overlap |

---

## 8. 给 RFC 的关键 takeaway

1. **JAX API 比 CUDA 简单**：默认异步派发 + memory_kind 标注 + `device_put` 三件套就能覆盖大部分搬运场景。不需要管理 stream/event。
2. **Pallas 是性能爆炸点**：只在 D2D scatter（必须跨 layer 流水线）和 PD D2H（必须预分配 buffer）这两个场景用 Pallas；其他场景用纯 JAX。
3. **`jax.experimental.transfer` 是 PD 的命门**：实验性 API + 实现细节未文档化 + 关键能力（partial pull / async）缺失。务必封装 ABC + 准备 fallback。
4. **不要尝试 layer-wise overlap**：TPU/XLA 架构使其不可行；做 step/request-level overlap 即可。
5. **没有 GPUDirect RDMA**：所有跨进程 KV 必须经 host 内存。RFC 中明确这是硬约束。
6. **`pinned_host` 是默认**：仅在 offload 容量优先场景考虑 `unpinned_host`。
7. **`donate_argnames` + `optimization_barrier` 是关键**：HiCache gather 一定要会用这两个 JIT 优化。
8. **GIL 不是问题**：JAX C++ 调用释放 GIL，ThreadPoolExecutor 可实现真正并行。
9. **sgl-jax 已有的 sharding 基础设施可直接用**：`MemoryPools` 是 pytree，加 host pool 不需要大改 API；KV pool 已是 `NamedSharding`，加 `memory_kind="pinned_host"` 即可。

---

**配套阅读**：
- `2026-05-18-sglang-cache-pd-organization.md` — sglang 组织结构调研
- `../../workspace/tpu-inference/kv_cache_offload_analysis.md` — TPU offload 完整数据流（§3-§5 详细路径、§10 SGLang HiCache Gap）
- `../../workspace/tpu-inference/pd_disaggregation_analysis.md` — TPU PD 完整数据流（附录 A 是最完整的 API 索引）
- JAX docs: `jax.sharding.NamedSharding`, `jax.experimental.pallas.tpu`, `jax.experimental.shard_map`
- JAX experimental: `jax.experimental.transfer`（无公开文档，看源码 `jax/experimental/transfer.py`）
