# RFC: PD buffer 与带外侧通道 — QueueHostKVPool + ZMQ pull-done

## 摘要

在 [transfer 基础设施](./2026-05-25-pd-transfer-foundation.md) 之上，
本 RFC 引入两件互补的组件：

- `QueueHostKVPool` — pinned-host KV pool 的 Queue 实现，原计划承担
  P 端 D2H staging buffer 的预分配与借/还。
- `ZmqPullNotifier` — D → P 的带外通知通道，告知 P 端某个 `uuid` 已
  完成 pull，可以安全释放 buffer。

本 RFC 同时把 sender 路径切到事件驱动 —— `KVSender` 在
`register_pull()` 后注册 callback，收到 ZMQ ack 才 release。整条链路
不再依赖任何同步等待占位。

## 当前实现状态（2026-05-27）

- ✅ `ZmqPullNotifier` + 事件驱动 sender 已实装（path-B 走通）。
- ✅ `QueueHostKVPool` 类与单测已落（参见 `test_queue_host_kv_pool.py`），
  但 **scheduler 当前显式传 `host_pool=None`**：path-A 没有 plumbing。
- ❌ D2H staging（path A）没接通：开启 `--disaggregation-enable-d2h`
  会在启动期 raise。
- 根因：`QueueHostKVPool` 当前 buffer shape 仍是旧的 token-major 契约，
  即 `buffer_shape=(max_tokens_per_buffer, layer_num, kv_head_per_rank, head_dim)`；
  但 prefill 当前实际产出的 payload 已经是 page-bucketed fused tensor，
  形状是 `(layer_num, padded_pages, page_size, ...)`。两者不只是“没连
  上”，而是 shape contract 本身已经不一致。要接通 path-A，至少需要：
  1) 把 host pool buffer shape 改为 page-bucketed；
  2) 在 prefill mixin 的 `producer_handoff` 路径把 `host_pool` 实际
     传进去；
  3) D 侧补 host->HBM 的写回路径。

D2H staging 默认值因此保持 OFF，hardening RFC 中「默认 ON」一项标注
为未交付。

## 线程安全与生命周期

引入后台 ZMQ listener 线程后，`JaxTransferWrapper._pending` dict 从
只主线程访问变成主线程 + listener 线程并发访问，由 `threading.Lock`
保护 register/release 的 read-then-write 序列。on_done 回调是
sender 状态 → SUCCESS 的触发点；同一回调里把 sender 从 `mgr._senders`
移除，避免无界增长。Receiver 同理在 SUCCESS 后被 pruned。

延后到 hardening RFC 处理的项目：

- **I3 wider uuid hash**：crc32 → blake2s 与/或 int uuids upstream。
- **M4 transfer_size / use_raw_buffers 通过 __init__ 暴露**：当前两
  条路径都走默认值。

## 在 PD 路线中的位置

```
            Transfer wrapper + connection ABC + single backend
              │
              ▼
[本 RFC]    Buffer + 带外侧通道
              │
              ▼
            Bootstrap + scheduler integration (端到端)
              │
              ▼
            Multi-host + routing
              │
              ▼
            Production hardening
```

本 RFC 是接到 scheduler 之前最后一块底层基础设施。完成之后：

- `JaxTransferKVManager` 不再依赖 Stage 0 里的同步等待占位，可以被
  scheduler 在 event loop 里安全驱动。
- `QueueHostKVPool` 给后续 scheduler 集成里的 `producer_handoff()` 提供
  path A 实现入口；多 host routing 阶段把 staging 默认值切到 ON 时无
  需再改 buffer 层。

## 设计

### `QueueHostKVPool`

放在 `python/sgl_jax/srt/mem_cache/host_kv_pool.py`，与 HiCache 的
`LRUHostKVPool`（独立 RFC）共享同一个 `HostKVPool` ABC。

```python
class QueueHostKVPool(HostKVPool):
    def __init__(
        self,
        pool_size: int,                 # 预分配 buffer 数
        max_tokens_per_buffer: int,     # 每个 buffer 的 token 容量
        layer_num: int,
        kv_head_per_rank: int,
        head_dim: int,
        dtype: jnp.dtype,
        mesh: jax.sharding.Mesh,
        partition_spec: PartitionSpec,
    ): ...

    def alloc(self, num_tokens: int) -> Optional[HostBufferHandle]: ...
    def free(self, handle: HostBufferHandle) -> None: ...
    def get_buffer(self) -> tuple[int, HostBufferHandle]: ...
    def put_buffer(self, buffer_id: int) -> None: ...
    def copy_from_device(self, device_kv: jax.Array) -> "StagedData": ...
    def available_size(self) -> int: ...
    def total_size(self) -> int: ...
```

关键约束：

- 预分配 `pool_size` 个独立 `jax.Array`，每个 sharding 为
  `NamedSharding(mesh, partition_spec, memory_kind="pinned_host")`。
- Queue 借出 / 归还，不做 LRU、不做 lock_ref —— 短生命周期。
- `copy_from_device` 内部 `jax.device_put(device_kv, host_sharding)`
  之后 `buffer.at[:num_tokens].set(staged)`，返回 `StagedData` 持新
  buffer 引用。注意 JAX `.at[].set()` 是 functional update，调用方
  不要缓存 `handle.buffer` 跨多次 `copy_from_device` 调用。
- `pool_size` 默认 64（与 tpu-inference `TPU_MAX_HOST_KV_BUFFER_SIZE`
  一致）；`max_tokens_per_buffer` 默认按 `model_config.max_total_num_tokens
  / pool_size` 粗算；两者均由 ServerArgs 覆盖。

### D2H staging vs direct-from-HBM

`JaxTransferKVManager` 暴露统一入口，按 flag dispatch 两条路径：

```
producer_handoff(uuid, device_kv, *, use_d2h_staging: bool):
    if use_d2h_staging:                                          # path A — 当前未接通
        staged = host_pool.copy_from_device(device_kv)
        wrapper.register_pull(uuid, staged.array)
        return TransferStatus(uuid, on_done=lambda: host_pool.put_buffer(...))
    else:                                                         # path B — 当前默认
        wrapper.register_pull(uuid, device_kv)
        return TransferStatus(uuid, on_done=lambda: None)
```

`use_d2h_staging` 来源：`ServerArgs.disaggregation_enable_d2h`，默认
`False`。当前实际只有 path B 走通，path A 的入口存在但未与 scheduler
联通（见上节「当前实现状态」）。

### `ZmqPullNotifier`

放在 Stage 0 的 `python/sgl_jax/srt/disaggregation/jax_transfer/` 子目录，
作为 backend-local 工具（多 host 协调仍由 backend 自己负责）。

```python
class ZmqPullNotifier:
    def __init__(self, role: str, host: str, port: int): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

    # P 侧
    def register_callback(self, uuid: bytes, cb: Callable[[bytes], None]) -> None: ...

    # D 侧
    def send_done(self, uuid: bytes, target_host: str, target_port: int) -> None: ...
```

协议：

- D → P：ZMQ `DEALER → ROUTER`，消息体 `msgpack({"uuid": bytes})`。
- 端口：`ServerArgs.disaggregation_side_channel_port`，默认 9600。
- P 侧 `ROUTER` 在 `start()` 时绑定端口，后台 listener 线程收到 uuid
  后查表回调（`pending_callbacks` 用 lock 保护，主线程 register 与
  后台线程 pop 都会触及）。
- D 侧按需建立 `DEALER` 连接（不预连），发完即关，无连接池。
- ZMQ context 用 `zmq.Context.instance()` 进程级共享。

### 事件驱动 sender 路径

`KVSender.send()` 不阻塞 —— 注册 callback 后立即返回，由 ZMQ listener
推动状态机：

```
KVSender.send():
    transition_to(WAITING_FOR_INPUT)
    status = manager.producer_handoff(
        uuid, kv_at_indices,
        use_d2h_staging=server_args.disaggregation_enable_d2h,
    )
    manager.zmq_notifier.register_callback(uuid, status.on_done)
    transition_to(TRANSFERRING)
    # 状态进 TRANSFERRING 后由外部 loop 通过 poll() 观察;
    # zmq listener 收到 ack 后调 status.on_done(uuid),
    # 同时由 manager 把该 uuid 对应的 sender state transition_to(SUCCESS).
```

`KVReceiver.poll()` 不变（D 端 pull 完成本来就是同步的，pull 返回即
SUCCESS），但 receiver 完成 pull 后**额外**调用
`manager.zmq_notifier.send_done(uuid, p_host, p_port)` 通知对端。

## 测试

### 单元测试（CI，CPU）

- `test_queue_host_kv_pool.py`
  - `alloc/free` 在 pool 满时返回 `None` / 释放后能重新 `alloc`。
  - `pool_size=N` 时 1 次性 `alloc N+1` 第 N+1 次返回 `None`。
  - `copy_from_device` 后 `StagedData.array` 字节相等于输入 device 数据
    的前 `num_tokens` 行。

- `test_zmq_pull_notifier.py`
  - P 启动后 D `send_done(uuid)`，P 侧 callback 在 1s 内被调用。
  - 100 并发 P↔D pair：每个 D 都发一个 uuid，P 收齐 100 个 uuid 且
    callback 调用次数 == 100，无丢失、无重复。
  - 未注册的 uuid 收到通知：listener 不崩、记录 warning。

- `test_kv_sender_event_driven.py`
  - mock `JaxTransferWrapper` + 真 `ZmqPullNotifier`：
    sender 注册 callback → 模拟 D 发 ack → sender state 在 100ms 内
    迁入 `SUCCESS` 且 `on_done` 被调用一次。

### 跨 pod 集成（手动，TPU）

不进 CI。在 Stage 0 byte-equality 脚本基础上扩展：

- 跑 path A（`--disaggregation-enable-d2h=true`）和 path B 两种模式，
  byte-equality 在两种模式下都通过。
- 100 并发 transfer 全部 SUCCESS；P 侧 `QueueHostKVPool.available_size()`
  在 transfer 结束后恢复到初值（无 buffer 泄漏）。
- ZMQ ack 任意丢一次（用 iptables drop 模拟）：sender 在 timeout 配置
  到位前会卡住（这是已知不足，由 hardening RFC 加 full-chain timeout
  解决）—— 本 RFC 仅在文档中记录此局限，不实装恢复。

跑通日志贴 PR description。
