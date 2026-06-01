# RFC: PD 基础设施 — JAX 传输 wrapper 与连接 ABC

## 摘要

为 sgl-jax 的 Prefill-Decode 分离（PD）引入最底层的几个构件：

- `jax.experimental.transfer` 之上的进程级 wrapper。
- 所有传输 backend 都将实现的连接 ABC（`KVManager` / `KVSender` /
  `KVReceiver` / `KVPoll`）。
- 一个建立在 wrapper 之上的具体 backend。
- 一个验证整条传输路径的跨 pod 字节相等 round-trip 测试。

本 RFC 范围是「传输层接口」自身，不接入 scheduler、不做 router 集成。
host pinned pool、ZMQ 带外侧通道、bootstrap、scheduler mixin 各自有
独立 RFC 描述。

## 在 PD 路线中的位置

PD 的整体规划按五段推进：

```
[本 RFC]    传输 wrapper + 连接 ABC + 单 backend
              │
              ▼
            Buffer + 带外侧通道
              │
              ▼
            Bootstrap + scheduler 集成（端到端跑通）
              │
              ▼
            多 host + routing
              │
              ▼
            Production hardening
```

后续每个增量都依赖此处建立的契约：

- Buffer + 侧通道增量把 host pinned pool 接到连接 ABC 上，并把
  sender 切到事件驱动（不再走同步等待占位）。
- Scheduler 增量把 `KVPoll` 状态机迁移转化为 scheduler 事件，加上
  CLI / bootstrap 入口。
- Routing 增量把 (P, D) pair 路由到远端 host 上具体的 `KVManager`
  实例。

## 设计

### `JaxTransferWrapper`

`jax.experimental.transfer` 之上的进程级单例。每个进程最多绑定一个
transfer server（host:port 唯一）；wrapper 用 module 级锁保证这一点。

```python
class JaxTransferWrapper:
    def __init__(self, host_ip: str, port: int, channel_number: int = 1): ...
    def start(self) -> None: ...
    def register_pull(self, uuid: str, data: jax.Array) -> None: ...
    def pull(
        self,
        uuid: str,
        spec: jax.ShapeDtypeStruct,
        remote_addr: str,
    ) -> jax.Array: ...
    def release(self, uuid: str) -> None: ...
```

行为约束：

- `start()` 幂等。重复调用返回同一 server 实例。
- `pull(uuid, spec, remote_addr)` 在 `spec.sharding is None` 时抛
  `ValueError`。`jax.experimental.transfer` 要求每个 `ShapeDtypeStruct`
  显式提供 sharding；wrapper 把这个检查前置，把一个深层的
  `AttributeError` 变成调用点上明确的契约违反。
- `pull(uuid, spec, remote_addr)` 的 `remote_addr` 是必填项 —— 进程级
  wrapper 同时为多个远端 peer 服务（不同 P pod 拉同一个 D pod），
  接口必须接受具体的 peer 地址。wrapper 内部按 `remote_addr` 缓存
  一个 `link`。
- `register_pull` 是非阻塞的：底层 API 把 buffer 注册到 server 之后
  立即返回。调用方必须在远端真正完成 pull 之前保持 buffer 存活。让
  这件事变安全的带外 ack 通道在下一个 RFC 引入。
- `register_pull(uuid, data)` 拒绝重复注册：若 `uuid` 已经在 pending
  队列里，抛 `RuntimeError`。调用方必须先 `release(uuid)` 才能复用
  uuid。Stage 0 没有 retry 场景，这条约束只用来防呆。
- wrapper 内部把 `uuid: str` 用 `zlib.crc32` 映射成 32-bit int 给底
  层 API。32 bit 在 ~65k 并发 uuid 时撞概率不可忽略，Stage 0 单 P-D
  pair 不会触及，Stage 1 起需要换成更宽的 hash 或要求上游直接给 int
  uuid。
- wrapper 不锁定具体的 JAX 版本。当前实现在 JAX 0.8.1 上验证通过；
  如果未来版本改变底层 API，下面的契约测试会失败 —— 这是一条可追溯
  的信号，而不是悄无声息的行为漂移。

### 连接 ABC

ABC 落地在 `python/sgl_jax/srt/disaggregation/base/kv_manager.py`，
`base/` 子目录为未来可能的多 backend 留位置。

`KVPoll` 是每次传输周期经历的状态：

```
BOOTSTRAPPING ──► WAITING_FOR_INPUT ──► TRANSFERRING ──► SUCCESS
       │                  │                   │
       └──────────────────┴────► FAILED ◄─────┘
```

`SUCCESS` 与 `FAILED` 为终态。合法迁移表放在 `base/kv_manager.py`
里，任一 backend 都可以调用共享的 `_transition_to(next_state)` helper
做校验。非法迁移抛 `ValueError`。

```python
class KVPoll(enum.Enum):
    BOOTSTRAPPING = "bootstrapping"
    WAITING_FOR_INPUT = "waiting_for_input"
    TRANSFERRING = "transferring"
    SUCCESS = "success"
    FAILED = "failed"

class KVManager(abc.ABC):
    @abc.abstractmethod
    def create_sender(self, req_id: str) -> "KVSender": ...
    @abc.abstractmethod
    def create_receiver(self, req_id: str) -> "KVReceiver": ...

class KVSender(abc.ABC):
    @abc.abstractmethod
    def init(self, kv_indices) -> None: ...
    @abc.abstractmethod
    def send(self) -> None: ...
    @abc.abstractmethod
    def poll(self) -> KVPoll: ...

class KVReceiver(abc.ABC):
    @abc.abstractmethod
    def init(self, p_metadata) -> None: ...
    @abc.abstractmethod
    def poll(self) -> KVPoll: ...
```

### `JaxTransferKVManager`

本 RFC 范围内唯一的 backend，是后续 host-pool / scheduler RFC 的实现
载体。Sender / Receiver 状态机：

Sender：

```
KVSender.send():
    transition_to(WAITING_FOR_INPUT)
    wrapper.register_pull(uuid, kv_at_indices)
    transition_to(TRANSFERRING)
    # 不同步等待 —— 由 host-pool RFC 引入的 ZmqPullNotifier
    # 收到 D 侧 pull-done ack 后异步推到 SUCCESS
```

Receiver：

```
KVReceiver.poll() 由外部 loop 驱动:
    transition_to(WAITING_FOR_INPUT)  当 P 端 metadata 到达
    transition_to(TRANSFERRING)
    data = wrapper.pull(uuid, spec)
    transition_to(SUCCESS)
    # 然后通过 ZmqPullNotifier.send_done(uuid) 通知 P
```

本 RFC 不把这个 backend 接到 scheduler；scheduler 集成由 scheduler-e2e
RFC 描述。

## 测试

### 单元测试（CI，CPU）

- `test_jax_transfer_wrapper.py`
  - `pull()` 在 `spec.sharding is None` 时抛 `ValueError`。
  - `start()` 幂等：两次调用返回同一 server。
  - `start()` 时将已安装的 JAX 版本写入 log。测试断言这条 log 出现，
    不断言版本号本身 —— 跨 JAX 版本始终通过，但如果未来 API 行为漂移
    导致 wrapper 契约失效，下面那条契约测试会失败给信号。

- `test_kv_manager_state.py`
  - 穷举全部 25 种 `(current, next)` 迁移。合法迁移成功；非法迁移
    抛 `ValueError`。
  - `SUCCESS` 与 `FAILED` 拒绝任何后续迁移。

### 跨 pod 字节相等（手动，TPU）

不进 CI（需要两个 TPU pod + 多进程协调）。提交 PR 前手动跑，把日志贴
到 PR description。

```
# Pod A (prefill 角色)
python -m sgl_jax.test.disaggregation.test_byte_roundtrip \
  --role prefill --remote <pod-B-host> --port 30001

# Pod B (decode 角色)
python -m sgl_jax.test.disaggregation.test_byte_roundtrip \
  --role decode --remote <pod-A-host> --port 30001
```

覆盖 bf16 / fp16 / fp8 e4m3 三种 dtype 与 1 / 16 / 256 三档 page
count，每组 100 次 round-trip，全部字节相等才算通过。在专用 TPU CI
runner 就绪之前保持手动。
