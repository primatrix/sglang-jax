# RFC: PD scheduler 集成 — Bootstrap、Mixin 与 prefill-only 契约

## 摘要

在 [transfer 基础设施](./2026-05-25-pd-transfer-foundation.md) 和
[buffer + 带外侧通道](./2026-05-25-pd-host-pool-side-channel.md) 之上，
本 RFC 把 PD 接进 sgl-jax scheduler，并定下当前阶段的 prefill-only
契约：

- `BootstrapServer`：FastAPI HTTP 服务，承担 P-D 路由握手。
- `ServerArgs` 与 CLI：`--disaggregation-mode {null,prefill,decode}`、
  `--disaggregation-bootstrap-port`、`--disaggregation-bootstrap-url`
  等字段，进程级 role 决定走哪条 event loop。
- `SchedulerDisaggregationPrefillMixin` / `SchedulerDisaggregationDecodeMixin`：
  把 P 的「prefill → 注册 KV → 等 ack 释放」与 D 的「按 bootstrap 拿 KV
  → 进 decode loop」从主 scheduler 类剥离。
- `tokenizer_manager` 透传 `bootstrap_{host,port,room}` 字段从 HTTP
  请求一路到 scheduler。
- **Prefill-only 契约**：P 完成 prefill + KV 传输后**不再本地继续生成**，
  返回空 completion（`finish_reason=length`，`length=0`）；D 负责真实
  生成。
- **PD 模式下 overlap schedule 关闭**：lifecycle 假设 step-by-step
  ownership transfer，overlap 会让 KV 所有权账目错乱。

完成后，两个进程（或两个 pod）一个跑 P、一个跑 D，一句话 prompt 能从
D 端拉出完整 token stream，P 端只产出 KV。

## 在 PD 路线中的位置

```
            Transfer wrapper + connection ABC + single backend
              ↓
            Buffer + 带外侧通道
              ↓
[本 RFC]    Bootstrap + scheduler integration (端到端)
              ↓
            Multi-host + routing
              ↓
            Production hardening
```

本 RFC 是 PD 第一次真正端到端跑通的节点：

- Stage 0 给了传输 ABC + 单 backend；Stage 1 给了 buffer 与 ack 通道。
- 本 RFC 把这些拼成可被 scheduler 驱动的状态机，并把进程角色（P / D）
  从 CLI 一路落到 event loop dispatch。
- 后续 multi-host RFC 在此基础上加 routing 与 per-host server；
  hardening RFC 加 metrics / timeout / auth。

## 设计

### `BootstrapServer`

`python/sgl_jax/srt/disaggregation/bootstrap.py`，FastAPI + uvicorn，
后台线程跑。集中式部署 —— 单进程，所有 P/D 都是 client。

```
POST /register_prefill     P 启动时注册 (host, port, side_channel_port,
                            tp_rank, tp_size, system_dp_rank)
POST /heartbeat            P 周期心跳，配合 TTL 健康检查 (默认 30s 失效)
POST /unregister_prefill   P 退出时主动注销
GET  /list_prefills        D 拿可用 P 列表
GET  /get_prefill_info?bootstrap_room=<int>
                            D 按 room 取一个 P（哈希到 prefills 列表）
GET  /health               健康检查
```

实现细节：

- `prefills: dict[str, PrefillInfo]` 用 `threading.Lock` 保护，
  `_last_seen` 同表跟踪心跳时间戳。
- `_evict_stale()` 在 `list_prefills` 与 `get_prefill_info` 调用前
  懒清理超时项，无独立后台线程。
- 走后台线程跑 uvicorn，daemon 进程退出时一同终止；本 RFC 不实装
  graceful shutdown，留 hardening RFC。

### CLI 与 ServerArgs

新增字段：

| 字段 | 默认 | 说明 |
|---|---|---|
| `disaggregation_mode` | `"null"` | `null` / `prefill` / `decode` |
| `disaggregation_bootstrap_port` | `8998` | bootstrap server 监听端口 (P/D 内部使用) |
| `disaggregation_bootstrap_url` | `None` | 集中式 bootstrap 的 URL（P/D 必填，否则 raise） |
| `disaggregation_transfer_port` | `30001` | `JaxTransferWrapper` 端口（P 注册时上报） |
| `disaggregation_side_channel_port` | `9600` | ZMQ ROUTER 端口（P 监听 D 的 ack） |
| `disaggregation_enable_d2h` | `False` | 是否走 path A（Stage 1 已说明，默认 OFF） |
| `disaggregation_d2h_pool_size` | `64` | `QueueHostKVPool` 预分配 buffer 数 |
| `disaggregation_d2h_max_tokens` | 由 `max_total_num_tokens / pool_size` 推导 | 每 buffer 容量 |

ServerArgs 校验：

- `mode in {"null","prefill","decode"}`，其他值 raise。
- `mode != "null"` 时 `bootstrap_url` 必须提供，否则 raise（D 没 URL
  没法找 P）。
- `mode == "null"` 时所有 PD 字段都被忽略（但 log warning 提示）。

`launch_server` 在 `init_engine` 入口后立即解析 `disaggregation_mode`：
- `null` → 走现有 scheduler 路径，不引入任何 PD 模块。
- `prefill` / `decode` → 实例化 `JaxTransferKVManager`、`BootstrapClient`、
  调用 scheduler 时绑定对应 Mixin。

### Scheduler Mixin

`python/sgl_jax/srt/disaggregation/prefill.py` + `decode.py` 各自定义
一个 Mixin。Scheduler 主类不改主路径代码（与 RFC-2 ADR-6 一致），只
在 `run_scheduler_process` 入口按 mode dispatch event loop：

```
if disaggregation_mode == "null":
    scheduler.event_loop_normal()           # 现状不变
elif disaggregation_mode == "prefill":
    scheduler.event_loop_normal_disagg_prefill()
elif disaggregation_mode == "decode":
    scheduler.event_loop_normal_disagg_decode()
```

**PD 模式下 overlap 关闭**：`Scheduler.__init__` 在
`disaggregation_mode != "null"` 时把 `enable_overlap` 设为
`False`。理由是 lifecycle 假设 step-by-step ownership transfer，开
overlap 会让 KV 所有权账目错乱。

`SchedulerDisaggregationPrefillMixin` 提供：

- `event_loop_normal_disagg_prefill()` — 主循环，与 `event_loop_normal`
  结构类似，但只跑 prefill；不进 decode iter。idle path 在调
  `check_memory()` / `check_tree_cache()` 之前先 `send_kv_chunk()`，
  确保上一拍 side-channel ack 触发的终态请求把 KV 所有权先归还，
  避免被 idle leak check 误报。
- `PrefillBootstrapQueue` — 维持等待 D 拿 KV 的请求队列（按 uuid 索引）。
- `process_prefill_chunk()` — 走标准 prefill，结束后调
  `KVSender.send()`，状态机推进到 `TRANSFERRING`。**对 PD 请求不再调
  `process_batch_result`**（避免本地继续生成 token），只 mark
  next-batch sampling info done。
- `send_kv_chunk()` — 在 `KVPoll.SUCCESS` 后释放 `req_to_token_pool`
  并触发 `_on_prefill_transfer_terminal`。
- 新增三个 prefill-only 终态 hook：
  * `_on_prefill_transfer_terminal(req, sender)` — 终态分发，根据
    `sender.poll()` 走成功或失败路径，最后 `sender.clear()` +
    `_release_prefill_req_resources(req)`。
  * `_finish_prefill_only_success(req)` — 把 req 标成
    `FINISH_LENGTH(length=0)`、`output_ids=[]`，调 `stream_output`
    把空 completion 推回客户端。
  * `_finish_prefill_only_failure(req, sender)` — 把 req 标成
    `FINISH_ABORT` + `PDTransferError`，附带 `sender.failure_exception()`
    的错误细节，stream 出去后清理资源。

`SchedulerDisaggregationDecodeMixin` 提供：

- `event_loop_normal_disagg_decode()` — 主循环，等待 bootstrap 拉回的
  新请求，进入 decode loop。
- `DecodePreallocQueue` / `DecodeTransferQueue` — 预分配 KV 槽位 +
  等 `KVReceiver` 完成 pull。
- `process_decode_queue()` — `KVPoll.SUCCESS` 后把 KV 写到 paged pool
  对应 indices，标准 `event_loop_normal` 接管 decode。
- 此处**绕过 `tree_cache.insert`**（与 RFC-2 ADR-7 一致），decode
  完成后由标准 `cache_finished_req` 触发插入。

### Tokenizer field passthrough

`bootstrap_{host,port,room}` 三字段是 D 在 bootstrap 上找 P 的钥匙：

- `bootstrap_host` / `bootstrap_port`：bootstrap server 地址。
- `bootstrap_room`：请求级 routing key（client 生成的整数，决定哈希到
  哪个 P）。

passthrough 链路：

1. HTTP 请求体顶层字段（与 `prompt` / `sampling_params` 并列）。
2. `TokenizerManager` 解析后塞进 `TokenizedGenerateReqInput` 同名字段。
3. ZMQ 发送给 scheduler 后，`Req` 对象上挂三字段。
4. Mixin 内的 `KVReceiver.init(p_metadata)` 用 `bootstrap_room` 走
   `BootstrapClient.get_prefill_info()` 拿到对应 P 的 `(host, port,
   side_channel_port)`。

校验：mode == `decode` 但请求缺这三字段时返回 4xx，不进 scheduler。

## 测试

### 单元测试（CI，CPU）

- `test_bootstrap_server.py`
  - 注册 / 列表 / 按 room 查询 happy path。
  - 重复注册同一 key：覆盖旧 PrefillInfo + 更新 `_last_seen`。
  - 心跳超时（mock 时钟 +60s 不发心跳）：`list_prefills` 不再返回该项。
  - `get_prefill_info` 在 0 个 P 时返回 `{"error": ...}`。

- `test_server_args_disaggregation.py`
  - 非法 mode raise。
  - `mode != "null"` 缺 `bootstrap_url` raise。
  - `mode == "null"` 时其他 PD 字段被忽略 + log warning。

- `test_tokenizer_bootstrap_passthrough.py`
  - 三字段从 HTTP 请求 → `TokenizedGenerateReqInput` → 序列化往返
    后 `Req` 对象上字段一致。
  - `decode` mode 缺字段返回 4xx，不到达 scheduler。

### 集成测试（CI，CPU mock 或本地双进程）

- `test_pd_e2e_single_host.py`
  - 单机 2 进程：mock TPU pool + 真 bootstrap + 真 ZMQ + 真 transfer
    backend（或 fake backend）。
  - 一句话 prompt 端到端：P 完成 prefill → KV 传输 → D 进 decode →
    token 输出非空。
  - **当前 oracle**：P 端返回空 completion + `finish_reason=length(0)`，
    D 端返回非空 token 序列。原 RFC 草稿里的 P/D byte-equal oracle
    在 prefill-only 契约下不再适用 —— 参考
    `tools/e2e/test_correctness_byte_equal.py` 的实现。

### 端到端（手动，TPU）

- e2e 测试在两个 TPU pod 上跑（一 P 一 D），跑通 + 贴日志。
- 当前只跑 path B（`--disaggregation-enable-d2h=false`）。path A 的
  staging 路径还没有从 scheduler 接通，参考 host-pool RFC 的「当前
  实现状态」节。
