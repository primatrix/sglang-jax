# PD Lifecycle Alignment Notes (2026-05-27)

这份笔记只回答一个问题：

> 当前 `sgl-jax` 的 PD control-plane 生命周期管理，相比 `../sglang`，哪些差异只是实现形态不同，哪些差异会直接放大成我们看到的 `no registered callback` / stuck transfer / 难以定界的 terminal 行为。

---

## 1. 结论先行

目前最重要的静态结论有 3 条：

1. `../sglang` 的生命周期状态是 **manager-owned** 的。
   - 它把 request / room 的 terminal 信息继续留在 manager 里。
   - 典型结构：
     - `request_status`
     - `failure_records`
     - `required_prefill_response_num_table`
     - `prefill_response_tracker`
     - `addr_to_rooms_tracker`

2. 我们当前的生命周期状态是 **object-owned + queue-owned** 的。
   - sender / receiver / queue / notifier 各自持有一部分状态。
   - terminal 后，live callback 会立刻被删掉，但 terminal transfer 的身份不会被保留。

3. 这会直接导致一个观测缺口：
   - 对 prefill notifier 来说，
     - “这是一个已经 terminal 的 transfer 的迟到 ack”
     - 和
     - “这是一个真正未知 / 错序 / 非法的 ack”
   - 现在都会坍缩成同一条 warning：
     - `ZmqPullNotifier received uuid=... with no registered callback; dropping`

所以，**当前最先该补的不是更大的协议重写，而是 terminal transfer 的最小保留状态**，让 late ack 和 true unknown ack 可以被静态地区分。

---

## 2. 参考代码

### `../sglang`

- `../sglang/python/sglang/srt/disaggregation/common/conn.py`
- `../sglang/python/sglang/srt/disaggregation/prefill.py`
- `../sglang/python/sglang/srt/disaggregation/decode.py`
- `../sglang/python/sglang/srt/disaggregation/mori/conn.py`
- `../sglang/python/sglang/srt/disaggregation/mooncake/conn.py`

### `sgl-jax`

- `python/sgl_jax/srt/disaggregation/jax_transfer/zmq_notifier.py`
- `python/sgl_jax/srt/disaggregation/jax_transfer/conn.py`
- `python/sgl_jax/srt/disaggregation/prefill.py`
- `python/sgl_jax/srt/disaggregation/decode.py`
- `python/sgl_jax/srt/managers/scheduler.py`

---

## 3. 生命周期模型对比

### `../sglang` 的模型

`../sglang` 不是只靠 live sender/receiver 对象来表达状态，而是把状态提升到 manager / room 层。

关键特征：

- request/room 有显式状态表
  - `request_status`
  - `failure_records`
- decode 侧会保留“还差多少 prefill 响应”的追踪表
  - `required_prefill_response_num_table`
  - `prefill_response_tracker`
- terminal 之后还有显式 `clear()` / `abort()` 路径
  - success / failure / abort 都不是简单“对象消失”
  - terminal 元信息会先落到 manager，再清理 live object

这意味着：

- 迟到事件到达时，manager 仍然有能力回答：
  - 这个 room / request 以前见过吗？
  - 它现在是成功、失败还是已清理？
  - 这个事件是 benign late arrival 还是 truly unknown?

### 我们当前的模型

`sgl-jax` 的 jax-transfer backend 以 live object 为中心：

- prefill:
  - `JaxTransferKVSender`
  - `PrefillBootstrapQueue`
  - `ZmqPullNotifier._callbacks`
- decode:
  - `JaxTransferKVReceiver`
  - `DecodePreallocQueue`
  - `DecodeTransferQueue`
- manager:
  - `_senders`
  - `_receivers`

关键问题不是“状态太少”，而是“terminal 后状态消失太快”：

- sender terminal:
  - callback 从 notifier 里被 `pop`
  - sender 从 manager `_senders` 里被 prune
- receiver terminal:
  - receiver 从 manager `_receivers` 里被 prune

但是：

- notifier 不保留 retired transfer 的身份
- manager 不保留 transfer-level terminal registry
- 所以 terminal 后只剩“没 callback 了”，没有“它为什么没 callback”

---

## 4. 逐事件对照

### 4.1 Prefill send path

`../sglang`

- sender 初始化后会更新 manager-owned status
- terminal 后可以通过 `clear()` / `failure_exception()` 回到 manager 查状态和原因

`sgl-jax`

- `JaxTransferKVSender.send()` 里：
  - register callback
  - producer handoff
  - transition 到 `TRANSFERRING`
- success / fail 都会直接删除 live callback
- 删除后没有 retired transfer 记录

影响：

- 一旦 ack 晚到，notifier 只能看到 “callback 不在了”

### 4.2 Decode receive path

`../sglang`

- receiver 生命周期和 manager room status 联动
- transfer success / failure 后有 `clear()` 释放 room-tracking

`sgl-jax`

- `JaxTransferKVReceiver.poll()` 自己维护 WAITING/TRANSFERRING/SUCCESS/FAILED
- terminal 后直接 prune receiver object
- decode scheduler queue 释放资源，但不保留 transfer terminal 记录

影响：

- 失败后的追踪依赖当场日志，不依赖 retained status

### 4.3 Unknown / late event classification

`../sglang`

- terminal 之后，manager 仍能回答 request/room 的最近状态

`sgl-jax`

- notifier 只查 live callback dict
- 查不到就直接：
  - `received uuid=... with no registered callback; dropping`

这正是 handoff 里看到的主 warning。

---

## 5. 已确认会导致问题的差异

### 差异 A：live callback 删除后没有 retired transfer registry

这是当前最直接的问题。

现状：

- success / fail / timeout / shutdown 后，callback 会消失
- 迟到 ack 到达时，notifier 无法区分：
  - “这是已经完成的 transfer”
  - “这是未注册的异常 uuid”

结果：

- benign late ack 和真实异常被混成同一个 warning
- 静态分析和现场日志都很难收口

### 差异 B：logical request identity 和 transfer-attempt identity 原来耦合

这个问题已经有最小复现，也已经做了第一步修复：

- 旧实现里 wire uuid 只用 `rid`
- `rid` 复用时，旧 ack 能命中新 sender

当前已改成可选 `disagg_transfer_id`，默认 helper 会生成 per-attempt 唯一值。

这一步修掉的是：

- “错误归属到新 transfer”

但还没修掉的是：

- “旧 transfer 的 late ack 被 benign 地识别”

### 差异 C：abort 协议没有像 `../sglang` 一样形成显式 negative terminal channel

这是一个真实差异，但目前不该先动。

原因：

- prefill sender 提前 fail 是相对简单的
- decode receiver 若中途 abort，目前没有一个对称的 failure/negative-ack 协议立刻通知 prefill
- 这比 notifier terminal registry 更大，属于下一层问题

所以这条先记为 **后续控制面硬化项**，不是本轮第一刀。

---

## 6. 本轮建议的最小静态修复

先做一个最小、可验证、低风险的 control-plane 补丁：

1. 在 notifier / manager 一侧保留一个 **retired transfer registry**
   - key: wire uuid / transfer_id
   - value: terminal state / reason / timestamp

2. sender terminal 时登记 retired transfer
   - success
   - fail
   - timeout
   - shutdown

3. notifier 收到 unknown uuid 时先查 retired registry
   - 如果命中 retired:
     - 记录为 late ack / retired ack
     - 不再用 “no registered callback” 的 generic warning 表达
   - 如果没命中:
     - 继续保留真正的 unknown ack warning

4. 先不改更大的 abort 协议
   - 这一步先让日志语义和 lifecycle 可观测性对齐

---

## 7. 实验策略

静态修复之后，不先跑大矩阵，只做几种定向验证：

1. 本地单测
   - retired transfer 之后收到 late ack，应被分类为 retired/late
   - truly unknown uuid 仍保持 warning

2. 本地 sender/receiver 回归
   - 现有 event-driven tests 必须继续全绿

3. 远端小实验
   - `cache on`
   - P 重启后首个 `1637`
   - 同 base `rid` 重打
   - grep P 日志看是否还出现 generic `no registered callback`

---

## 8. 当前判断

当前最准确的判断是：

> `sgl-jax` 现在的主要差异不是“没有 lifecycle”，而是 terminal lifecycle 没有 retained state；所以 late ack 和 true unknown ack 被压成了同一种日志现象。

这就是本轮最先要对齐到 `../sglang` 思路的地方。

---

## 9. 本轮已落地

本轮已经把第 6 节建议的最小静态修复落地：

1. notifier 增加 retired transfer registry
2. sender 在 `SUCCESS` / `FAILED` terminal 时登记 retired transfer
3. listener 收到 unknown uuid 时先区分：
   - dispatching duplicate ack
   - retired late ack
   - truly unknown ack
4. manager 增加 retained terminal record
5. sender / receiver 补齐最小 `abort()` / `failure_exception()` / `clear()` 合约

对应验证：

- 本地单测：
  - `test_zmq_pull_notifier.py`
  - `test_kv_sender_event_driven.py`
  - `test_stage4_orphan_reaper.py`
- 本地全量：
  - `python/sgl_jax/test/disaggregation`
- 远端小实验：
  - `cache on`
  - 重复 `rid` 的 `1637`
  - `3900`
  - P 日志未再出现 generic `no registered callback`
