# PD Test Coverage Matrix (2026-05-27)

这份矩阵只回答两个问题：

1. 对当前 `sgl-jax` 的 **PD path-B / single-host-per-pod / TP=4** 目标，我们已经覆盖了哪些测试场景？
2. 参考 `../sglang` 的 PD 测试类别后，当前还缺哪些更大范围的场景？

---

## 1. 当前目标范围

当前“完成度”判断只针对下面这个 scope：

- `path B`
- `cache off` / `cache on`
- 单 host 4 chip
- TP=4
- 当前 v6e-16 debug 拓扑

**不包含：**

- `path A` / D2H staging
- 不同 TP 组合
- `dp attention`
- pipeline parallel
- 更大规模 router / LB 压测

---

## 2. 已覆盖矩阵

| 类别 | 场景 | 本地测试 / 验证 | 状态 |
|---|---|---|---|
| 状态机 | `KVPoll` 全状态转移合法性 | `python/sgl_jax/test/disaggregation/test_kv_manager_state.py` | 已覆盖 |
| Notifier 基础 | register / unregister / duplicate / malformed / concurrent ack | `python/sgl_jax/test/disaggregation/test_zmq_pull_notifier.py` | 已覆盖 |
| Notifier 生命周期 | duplicate in-flight ack / retired late ack / truly unknown ack 分类 | `python/sgl_jax/test/disaggregation/test_kv_sender_event_driven.py` | 已覆盖 |
| Sender 生命周期 | send → ack → success、fail cleanup、retry with reused `rid` | `python/sgl_jax/test/disaggregation/test_kv_sender_event_driven.py` | 已覆盖 |
| Manager retained state | terminal record retained across success / timeout / shutdown；new attempt clears stale record | `python/sgl_jax/test/disaggregation/test_kv_sender_event_driven.py`, `python/sgl_jax/test/disaggregation/test_stage4_orphan_reaper.py` | 已覆盖 |
| Sender/receiver terminal contract | `abort()` / `failure_exception()` / `clear()` 语义 | `python/sgl_jax/test/disaggregation/test_kv_sender_event_driven.py`, `python/sgl_jax/test/disaggregation/test_stage4_orphan_reaper.py` | 已覆盖 |
| Receiver 生命周期 | pull timeout、ack send failure path、reaper fail path | `python/sgl_jax/test/disaggregation/test_stage4_orphan_reaper.py`, `python/sgl_jax/test/disaggregation/test_pd_mixin_event_loop.py` | 已覆盖 |
| Bootstrap | register / lookup / protocol version / multi prefill selection | `python/sgl_jax/test/disaggregation/test_bootstrap_server.py`, `test_stage4_protocol_version.py`, `test_multi_prefill_registration.py` | 已覆盖 |
| Auth | bootstrap + notifier shared secret path | `python/sgl_jax/test/disaggregation/test_stage4_auth.py` | 已覆盖 |
| Tokenizer / request plumbing | `bootstrap_*` passthrough、auto derive、`disagg_transfer_id` passthrough | `python/sgl_jax/test/disaggregation/test_tokenizer_bootstrap_passthrough.py` | 已覆盖 |
| Scheduler mixin | PD req extraction, prealloc/transfer queues, failure cleanup, debug hooks | `python/sgl_jax/test/disaggregation/test_pd_mixin_event_loop.py` | 已覆盖 |
| Data correctness | byte roundtrip、single-host fake e2e、decode padded writeback fix | `python/sgl_jax/test/disaggregation/test_byte_roundtrip.py`, `test_pd_e2e_single_host.py`, `test_pd_debug_utils.py`, `test_pd_mixin_event_loop.py` | 已覆盖 |
| Shutdown / orphan handling | reaper, graceful shutdown ordering, timeout force-fail | `python/sgl_jax/test/disaggregation/test_stage4_orphan_reaper.py` | 已覆盖 |
| 远端 focused validation | `cache on`，P 重启后，请求 `1251/1637/2501/3900`；重复 base `rid`；P/D byte-equal | 2026-05-27 手工验证 | 已覆盖 |

当前本地 CPU 回归：

```bash
python -m pytest python/sgl_jax/test/disaggregation -q
```

最新结果：`194 passed`

---

## 3. 对标 `../sglang` 的测试类别

下面这些 `../sglang` 测试类别对我们有参考意义：

- 基础正确性 / 评测：
  - `../sglang/test/registered/disaggregation/test_disaggregation_basic.py`
- 不同 TP 组合：
  - `../sglang/test/registered/distributed/test_disaggregation_different_tp.py`
- DP attention：
  - `../sglang/test/registered/distributed/test_disaggregation_dp_attention.py`
- Pipeline parallel：
  - `../sglang/test/registered/distributed/test_disaggregation_pp.py`
- decode offload / hicache：
  - `../sglang/test/registered/disaggregation/test_disaggregation_decode_offload.py`
  - `../sglang/test/manual/hicache/test_disaggregation_hicache.py`
- tracing / observability：
  - `../sglang/test/registered/observability/test_tracing_disaggregation.py`

这些测试说明 `sglang` 的覆盖面分成两层：

1. **当前能力是否成立**
2. **更大拓扑 / 更多 backend / 更多 scheduler 模式下是否成立**

我们现在已经把第 1 层的大部分 narrow-scope 场景补齐了。  
第 2 层还没有。

---

## 4. 当前仍未覆盖的场景

### 4.1 代码已支持但验证不足

这些不一定都要现在做，但从“矩阵完整性”看是缺口：

- decode-side abort 中途发生时，prefill 侧最终状态是否总能被良性收敛
- shutdown 后 late ack 的远端日志分类
- peer kill mid-transfer 的远端行为
  - 目前已知可能触发 `SocketServer: Connection closed recv() == 0`
- remote cache-on 多轮长时间 soak
  - 目前只有 focused validation，不是长稳回归

### 4.2 当前 scope 外，但要明确标记为未完成

- `path A` / D2H staging
- `prefill TP != decode TP`
- `dp attention`
- pipeline parallel
- router / LB 大并发吞吐
- hicache / decode offload

---

## 5. 推荐的下一批测试

按优先级排序：

1. **Abort / timeout / shutdown focused remote cases**
   - 最贴近当前 control-plane 差距

2. **Late-event remote log classification**
   - 验证 `retired late ack` 和 `truly unknown ack` 在现场是否真的被分开

3. **Longer cache-on soak**
   - 固定 case 循环跑，重点看 lifecycle warning / hang / terminal cleanup

4. **scope 外能力单独开矩阵**
   - `path A`
   - different TP
   - DP attention
   - PP

---

## 5.1 2026-05-27 remote probe findings

这轮 focused remote probe 已经额外验证了 3 类场景：

1. **cache-on repeated-rid / representative lengths**
   - `1251 / 1637 / 1637(repeat) / 2501 / 3900`
   - 结果：全部 P/D byte-equal

2. **abort mid-flight**
   - `3900` prompt，`max_new_tokens=128`
   - 在 P 和 D 上同时调用 `/abort_request`
   - 结果：
     - 早期 abort（~50ms）时：
       - P 侧返回 `finish_reason=abort`
       - D 侧请求最终超时，日志里先出现
         `KVReceiver ... reached failed`
         然后出现 client disconnect abort
     - 较晚 abort（~300ms）时：
       - P / D 都返回 200
       - 但两边 completion 不一致
   - 结论：**decode-side abort lifecycle 还没有收口**

3. **peer-kill mid-transfer**
   - `kill -9` prefill 进程，request 正在进行
   - 结果：
     - 客户端两边都直接断链
     - D 日志仍可复现
       `jax.errors.JaxRuntimeError: INTERNAL: SocketServer: Connection closed recv() == 0`
   - 结论：**runbook 里旧的 FINDING-B 仍然存在**

这些 probe 说明：

- 当前 scope 下，正常 path-B correctness 已经基本稳定
- 但 abort / peer-crash 两类 failure-path 还没有达到“可接受的 control-plane 收敛”

---

## 6. 当前判断

对当前目标范围来说，测试矩阵已经不再是“只有 smoke”了，而是：

- 本地有一整套 control-plane + data correctness CPU 回归
- 远端有 focused lifecycle / cache-on 验证
- 但更大拓扑和额外 backend 模式还没有纳入完成判定

也就是说：

> 当前矩阵对 **path-B 当前 scope** 已经基本完整；对 **完整 PD feature surface** 还不完整。
