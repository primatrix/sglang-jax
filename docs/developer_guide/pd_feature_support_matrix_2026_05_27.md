# PD Feature Support Matrix (2026-05-27)

这份表只描述 **当前 `sgl-jax` 代码 + 已完成验证** 的实际状态。  
它不回答“理论上能不能做”，而回答：

> 现在代码里有没有路径、有没有测试、能不能作为当前阶段的支持结论。

---

## 1. Soak 的含义

当前阶段里，`soak` 指的不是新增功能测试，而是：

- 在 **正常路径** 下
- 对固定拓扑和固定模型
- 连续、重复、较长时间地跑请求
- 重点观察：
  - P/D 输出是否持续一致
  - `pd_transfer_inflight` 是否回到 0
  - 是否出现新的 generic lifecycle warning
  - 服务是否出现 hang / 进程退出 / socket EOF
  - latency / throughput 是否明显漂移

也就是说，`soak` 更像：

- correctness drift 检查
- resource / state leak 检查
- compile-cache / long-run stability 检查

而不是 kill/abort/chaos 这种 failure-path 测试。

---

## 2. Feature Support Table

| Feature | 代码状态 | 已验证状态 | 当前结论 | 依据 |
|---|---|---|---|---|
| 多 P / 多 D | **有基础支持** | **部分验证** | **支持 normal path，但还没做最新代码下的大矩阵 soak** | bootstrap registry 支持多个 prefill；`bootstrap_room -> pick_for_room()` 是确定性的；有 `test_multi_prefill_registration.py` 和 `test_topology_multi_pd.py`；历史 `pd_e2e_matrix.md` 里 `P0-1` PASS |
| 实例热插拔 / 热重启 | **部分支持** | **部分验证** | **支持注册/重注册/优雅下线；不支持 in-flight seamless failover** | `register_prefill` / `heartbeat` / `unregister_prefill` / `HeartbeatDaemon` / `graceful_shutdown` 已有；`test_multi_prefill_registration.py` 验证 re-register 替换旧 entry；但 hard kill / mid-transfer failover 仍是下一阶段问题 |
| DP attention | **只有部分 plumbing** | **未完成验证** | **当前阶段不算支持** | 保留了 `system_dp_rank`、有 `test_orthogonal_dp.py` 手工入口；但 `pd_e2e_matrix.md` 明确 ORTH-DP 未跑；`pd_vs_sglang_gaps.md` 也把 `/register_dp_rank` + `/query_dp_ranks` 列为 gap |
| overlap | **全局 scheduler 有，PD 路径没有正式接入** | **未验证** | **当前阶段不支持** | scheduler 全局有 `enable_overlap` / `event_loop_overlap()`，但 `disaggregation_mode=prefill|decode` 时入口强制走 `event_loop_normal_disagg_*`，不会走 overlap loop |

---

## 3. Per-Feature Notes

### 3.1 多 P / 多 D

当前多 P / 多 D 的核心机制是：

- P 侧通过 bootstrap 注册多个 `PrefillInfo`
- D 侧按 `bootstrap_room % len(sorted(keys))` 选定 P
- 多个 D 主要依赖外部 router / LB 做 HTTP fan-out

所以当前的“多 P / 多 D 支持”更准确地说是：

- **多 P：有**
- **多 D：有，但更偏 router/LB 层能力**
- **跨多个 P/D 的 normal-path correctness：有基础验证**
- **最新 control-plane 改动后的完整 2P×2D soak：还没做**

### 3.2 热插拔 / 热重启

当前真正支持的是：

- 新 P 注册进 bootstrap
- 旧 P graceful unregister
- 同 key re-register 用新 endpoint 替换旧 endpoint

当前不支持作为“已完成”结论的是：

- in-flight request 在实例 hard kill 时无感迁移
- 某实例 mid-transfer 消失后 request 自动恢复并保持协议级一致

所以这项只能算 **部分支持**。

### 3.3 DP attention

从代码上看，PD 没有完全忽略 DP：

- bootstrap 注册保留 `system_dp_rank`
- tokenizer / request / cache 层也有不少 `dp_rank` plumbing

但就 PD 当前阶段来说，最关键的问题是：

- 没有 current-scope 的自动化验证
- 参考 `../sglang` 的两阶段 DP routing 能力还没完整 port

因此不能把它算作“已支持 feature”。

### 3.4 overlap

这是最容易误判的一项。

全局 scheduler 确实支持 overlap：

- `Scheduler.enable_overlap`
- `Scheduler.event_loop_overlap()`
- overlap 相关 output processor / tp worker 代码

但 PD 当前入口是：

- `mode == "prefill"` → `event_loop_normal_disagg_prefill()`
- `mode == "decode"` → `event_loop_normal_disagg_decode()`

也就是说：

- **PD 模式不会走 overlap loop**
- 所以 overlap 对 PD 当前阶段不能算支持

---

## 4. 当前阶段建议

如果只针对当前 normal-path 目标，建议这样看：

- **可以继续拉大验证矩阵的**
  - 多 P / 多 D normal-path correctness
  - repeated-rid / cache-on soak
  - benchmark / eval

- **先不要纳入当前阶段支持承诺的**
  - 热插拔/热重启的 failure-path 保证
  - DP attention
  - overlap

---

## 5. 一句话总结

当前阶段最准确的结论是：

> `sgl-jax` 的 PD 现在已经能支持 path-B normal-path 下的单 host TP=4 调试拓扑，并有多 P / 多 D 的基础能力；但热插拔/热重启只部分支持，DP attention 和 overlap 还不能算当前阶段已支持 feature。
