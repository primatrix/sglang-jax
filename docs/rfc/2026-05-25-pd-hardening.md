# RFC: PD production hardening — 可观测、错误恢复、auth、性能

## 摘要

把 PD 从「跨 host 跑通」推到「可上生产 SLA」。本 RFC 是 PD 路线最后
一阶段，覆盖 6 个能力面：

1. **可观测** — Prometheus 指标 + Grafana dashboard
2. **错误恢复** — orphan-transfer 清理、host pool 释放、全链路 timeout
3. **鉴权** — transfer / bootstrap / side channel 三条信道
4. **优雅 shutdown 与滚动升级** — drain inflight、跨小版本兼容
5. **多 channel 与 D2H staging ON** — 性能 sweep + 默认值收紧
6. **stress / chaos + runbook**

本 RFC 范围大、子项目互相之间相对独立，是 PD 路线中**唯一可拆多个
PR 提交**的阶段（前面四个 RFC 各对应一个 PR）。

## 当前实现状态（2026-05-27）

本 RFC 当前部分交付，按能力面拆分如下。

**已实装**：

- ✅ `metrics.py` — Prometheus 指标骨架（state transition counter /
  transfer bytes / inflight gauge / failure counter / bootstrap
  registry size）。
- ✅ Orphan reaper — 后台扫线程已落地，但当前是
  `pull_timeout` / `ack_timeout` + `reaper_interval` 的组合，不存在单独
  的 `--disaggregation-orphan-timeout-seconds`。
- ✅ `pd_auth.py` — bootstrap HTTP `Authorization: Bearer` + ZMQ msg
  HMAC，共享 secret 来自 `--disaggregation-shared-secret` /
  `SGL_JAX_PD_SHARED_SECRET`。
- ✅ `protocol_version` 字段 — `PrefillInfo` schema 携带版本号，
  bootstrap / D 端做 minor 兼容性校验。
- ✅ 全链路 timeout 矩阵的 ServerArgs 与 metric label 已落，bootstrap
  query / pull / ack / orphan 四档 timeout 各自可配置。
- ✅ production-like benchmark/eval 入口已经接通：
  - single-entry router 可跑 native `/generate`
  - single-entry router 可跑 OpenAI `/v1/chat/completions`
  - `bench_serving` 与 `run_eval.py gsm8k` 均已通过 smoke 验证

**未交付**：

- ❌ Stress 1h（`tools/stress.py` 文件存在，但 1h 持续报告未跑过）。
- ❌ Chaos（`tools/chaos.sh` 脚本存在，但 kill / iptables drop /
  bootstrap restart 三场景的端到端验证未跑通）。
- ❌ Grafana dashboard JSON（`docs/operations/grafana/pd_dashboard.json`
  未导出）。
- ❌ Multi-channel sweep（`tools/sweep_channels.py` 文件存在，但
  knee-point 实测 + `pd_perf_baseline.md` 数据未落）。
- ❌ D2H staging 默认 ON —— path-A 当前在 scheduler 层未 plumbed（参考
  host-pool RFC「当前实现状态」节），默认值因此保持 OFF，本项不可能
  在不接通 path-A 的前提下交付。
- ❌ 优雅 shutdown / 滚动升级实测（机制部分到位，但端到端切流验证未跑）。

**容量目标尚未达到**：

生产容量目标是 `64 并发 / 16k input / 1k output`，当前第一条容量
cliff 已经在 `16 并发 / 4k input / 128 output` 出现，`16 并发 / 4k input / 256 output`
也同样失败；而 `8 并发 / 4k input / 256 output` 与
`16 并发 / 2k input / 128 output` 已通过。

根因已经收敛到 D 侧 `process_decode_queue() -> _write_kv_to_pool()` 的
scatter 写回路径：不是 router/proxy、不是 req lifecycle、不是 pull
本身，而是 `jit_scatter` 在写回 paged KV pool 时的瞬时 HBM 放大。

因此，在未把这条 cliff 解掉之前，stress / chaos 的 1h 持续目标都不
具备前置条件。

下面各节描述完整设计；具体哪一项已落地以本节为准。

## 在 PD 路线中的位置

```
            Transfer wrapper + connection ABC + single backend
              ↓
            Buffer + 带外侧通道
              ↓
            Bootstrap + scheduler integration (端到端)
              ↓
            Multi-host + routing
              ↓
[本 RFC]    Production hardening
```

完成本 RFC 后，PD 满足以下 SLA 假设（具体数字在 stress 测后定）：

- 滚动 7 天 99.9% 可用率
- 单 channel transfer ≥ 8 GB/s，多 channel sweep ≥ 25 GB/s 聚合
- P / D 任一侧 pod crash 60s 内自动恢复，无请求级丢失
- 跨一个 minor 版本可滚动升级

## 设计

下面 6 节相对独立，按依赖与价值排序：可观测先行（其他改动都要 metric
确认效果），错误恢复 / auth 紧随其后（生产准入），性能与运维收尾。

### 1. 可观测 (observability)

模块：`python/sgl_jax/srt/disaggregation/metrics.py`，复用现有
`sgl_jax/srt/metrics/` 框架（Prometheus client）。

指标 schema：

| 指标 | 类型 | 标签 |
|---|---|---|
| `pd_state_transition_total` | Counter | `from_state`, `to_state`, `role` |
| `pd_transfer_bytes_total` | Counter | `direction` (`d2h`/`h2d`/`net`), `role` |
| `pd_transfer_duration_seconds` | Histogram | `phase` (`bootstrap`/`pull`/`ack`), `role` |
| `pd_transfer_inflight` | Gauge | `role` |
| `pd_host_pool_used_buffers` | Gauge | `pool_name` |
| `pd_transfer_failures_total` | Counter | `reason` (`timeout`/`peer_crash`/`network`/`other`), `role` |
| `pd_bootstrap_registry_size` | Gauge | — |

dashboard 落在 `docs/operations/grafana/pd_dashboard.json`（导出后
checkin），包含：

- per-role 状态机迁移率、每个状态滞留 P99 时长
- transfer 带宽（按 direction）与失败率
- host pool 占用率 + 高水位告警阈值
- bootstrap registry 大小（异常下降 = P 大规模下线）

### 2. 错误恢复

#### Orphan transfer 清理

- 当前实现是 manager 后台 reaper 线程定期扫描 sender / receiver：
  - sender 使用 `ack_timeout_seconds`
  - receiver 使用 `pull_timeout_seconds`
  - 扫描频率由 `reaper_interval_seconds` 控制
- sender 超时后：
  - `wrapper.release(uuid)`
  - 若 path-A 已接通，则执行 `status.on_done()` 归还 host buffer
  - 状态机推到 `FAILED`
  - metric `pd_transfer_failures_total{reason="timeout"}` +1
- receiver 超时后：
  - 状态机推到 `FAILED`
  - metric `pd_transfer_failures_total{reason="timeout"}` +1
- 当前实现**没有单独的 `transfer_aborted` 消息协议**；清理与终态记录都
  在 sender / receiver / manager 本地完成。

#### Pod crash 检测与清理

- bootstrap 心跳 TTL（已在 scheduler RFC 实装）做粗粒度发现。
- D 侧的 `KVReceiver.poll()` 在 `TRANSFERRING` 状态超过 `pull_timeout`
  仍未返回数据：状态机 → `FAILED`，触发请求级 retry（router 层选另一
  个 P）。
- P 侧用 `transfer_aborted` 消息接收 D 主动放弃，立即清理对应 buffer。

#### 全链路 timeout 矩阵

| 阶段 | 默认 timeout | ServerArgs |
|---|---|---|
| Bootstrap query | 5s | `--disaggregation-bootstrap-timeout-seconds` |
| `await_pull` 完成 | 30s | `--disaggregation-pull-timeout-seconds` |
| ZMQ ack 收齐 | 60s | `--disaggregation-ack-timeout-seconds` |
| Reaper 扫描周期 | 5s | `--disaggregation-orphan-reaper-interval-seconds` |

每个 timeout 都有对应的 metric label，超时直接走 `FAILED`，不重试
（重试由上层 router 决策）。

### 3. 鉴权

三条信道各自独立加 auth，共享配置项 `--disaggregation-shared-secret`
（环境变量 `SGL_JAX_PD_SHARED_SECRET` 覆盖）：

#### Bootstrap server (HTTP)

- 所有 endpoint 加 `Authorization: Bearer <secret>` 校验。
- 缺 header 或 secret 不匹配返回 401。

#### Transfer (`jax.experimental.transfer`)

- API 本身无原生 auth，绕路实现：D 在 `connect()` 前先通过 ZMQ
  side channel 发 `request_pull(uuid, hmac(secret, uuid))`，P 校验
  HMAC 通过后才允许该 uuid 被 pull；否则 `release(uuid)` 并记 metric。

#### ZMQ side channel

- `DEALER → ROUTER` 消息体加 `hmac` 字段（msgpack 编码），P 侧
  listener 校验，不通过的消息丢弃 + warning。

可选 mTLS 模式（替换共享 secret）：通过
`--disaggregation-tls-cert/key/ca` 启用；本 RFC 实装 shared-secret，
mTLS 作为 follow-up 提案，不阻塞本 RFC。

### 4. 优雅 shutdown 与滚动升级

#### 优雅 shutdown

P 进程收到 `SIGTERM` 时：

1. 立即取消 bootstrap 注册（`unregister_prefill`）—— 让 router 不再
   把新请求路给本 P。
2. 等当前 `inflight_transfers` 全部 SUCCESS / FAILED，超过 30s（可配置）
   走强制 abort。
3. `JaxTransferWrapper.stop()` + `ZmqPullNotifier.stop()`。
4. 退出。

D 类似：取消 router 上游注册 → 等 decode loop 排空 → 退出。

#### 滚动升级兼容

- bootstrap server 与 P/D 之间使用 versioned 协议：`PrefillInfo`
  schema 加 `protocol_version: int`，D 拒绝小于自己 minor - 1 的 P。
- 同时支持两个 minor 版本并存（N-1 与 N），不支持跨 major。
- 升级顺序：bootstrap → D → P（D 兼容更老的 P）。

### 5. 多 channel + D2H staging ON

#### Multi-channel

`JaxTransferWrapper.__init__(channel_number=N)` 已有参数（Stage 0 默认
1）。本 RFC 把默认值升到根据 host BW 自动选择（环境探测；可显式覆盖），
做以下事：

- 提供 `tools/disaggregation/sweep_channels.py` —— 在 2 host × 2 pod
  layout 上跑 1/2/4/8 channel sweep，输出聚合带宽，结果落
  `docs/operations/pd_perf_baseline.md`（每次大版本升级前重跑）。
- 默认 `channel_number` 取实测 knee-point（首次实测推荐 4，待数据
  覆盖）。

#### D2H staging 默认 ON

把 `--disaggregation-enable-d2h` 默认值从 `False` 切到 `True`（前提是
path-A 已经真正接通；以当前代码状态这一步还不能执行）。配套：

- 加 dashboard panel 监控 `pd_host_pool_used_buffers` 高水位。
- runbook 说明何时手动 OFF（pool 频繁打满时降级）。

### 6. Stress + chaos + runbook

#### Stress

`tools/disaggregation/stress.py`：

- 10k qps 持续 1h，错误率 < 0.1%。
- 输出 P50 / P95 / P99 latency 与 throughput。
- 数据落 `docs/operations/pd_perf_baseline.md`。

#### Chaos

`tools/disaggregation/chaos.sh`：

- 随机 `kill -9` 一个 P pod，验证 60s 内 router 把请求切到其他 P，
  且失败请求 ≤ 5 个。
- 随机 `iptables drop` 跨 host transfer 流量 30s，恢复后 inflight
  transfer 走 timeout → FAILED → retry，最终一致性 OK。
- bootstrap server 重启 60s，期间新请求暂存路由层重试。

#### Runbook

`docs/operations/pd_runbook.md`：

- 部署清单（pod yaml 必须项 / ServerArgs 必填项 / 端口分配表）
- 健康检查命令（curl bootstrap `/health`、grep server log
  `ready to roll`）
- 故障排查决策树（请求 5xx → 看哪个 metric → 哪个 ServerArgs 调整）
- 容量规划公式（`pool_size = max_concurrent_prefill_requests * 1.5`
  之类的经验值）

## 测试

### 单元 / 集成（CI）

每个能力面有对应的 module 级单测，关键集成测试：

- `test_orphan_cleanup.py` — mock 后台扫线程，验证超时 entry 被清理
  + metric +1。
- `test_auth_bootstrap.py` — 缺 token、错 token、对 token 三种 case。
- `test_graceful_shutdown.py` — mock inflight transfer，SIGTERM 后
  在 timeout 内退出。

### 端到端（手动，TPU + 真实拓扑）

- stress 1h 跑通报告（PASS/FAIL + latency 数据）
- chaos 三场景全部恢复
- rolling upgrade 实测（部署 v(N-1) → 升级 D 到 vN → 升级 P 到 vN，
  全程无请求级失败）

数据进 `docs/operations/pd_perf_baseline.md` 与 `pd_runbook.md`。

## 拆 PR 建议

本 RFC 唯一可拆多个 PR 的 stage。建议 PR 边界：

| PR | 能力面 |
|---|---|
| H-A | 可观测（metrics + dashboard） |
| H-B | 错误恢复（orphan / timeout 矩阵） |
| H-C | 鉴权（shared secret 三信道） |
| H-D | 优雅 shutdown + 滚动升级协议 |
| H-E | 多 channel + D2H staging ON 默认（含 sweep tool） |
| H-F | Stress + chaos + runbook |

H-A 优先，其他可并行；H-F 必须最后做（依赖前面全部能力上线）。
