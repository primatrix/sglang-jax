# PD 性能夜间交接, 2026-07-05

## 当前状态

夜间已完成事项：

- 扩展了 PD 性能报告，加入 1P1D、non-PD 对比、AIME24 和 2P1D 多 prefill 结果。
- 增加 non-PD baseline 文档：
  `docs/developer_guide/pd_disaggregation/nonpd_mimo_v2_flash_16k4k_baseline_20260705.md`。
- 修复多 prefill routing 正确性：
  - Router 现在会把 `bootstrap_room` 对齐到选中的 prefill index。
  - Decode 现在保留 Raiden endpoint descriptor 中声明的 host，而不是重写成 bootstrap registry host。
- 把热路径上的 per-chunk `RAIDEN-D start_read*` 日志从 warning 降为 debug，减少日志噪声。
- 修复后完成 2P1D C64/C128 16K/4K benchmark。

2026-07-06 补充：

- 增加 pod-count-fair non-PD C64 对比：两个普通 non-PD server，
  每个 Falcon rank 一个，后面接一个轻量 streaming round-robin proxy。
- 在同一个 two-host endpoint 上增加 pod-count-fair non-PD C128 对比。
- 在 two-host non-PD endpoint 上重跑 AIME24。
- 增加 C128 steady-state advantage note：
  `docs/developer_guide/pd_disaggregation/pd_steady_state_advantage_20260706.md`。
- 同步公开 benchmark 口径修订：当前数据应被视为 `request_rate=inf`
  closed-loop burst benchmark 加 serve-log 稳态分析，而不是 DistServe 风格的
  SLO goodput 结果。

## 代码改动

改动文件：

- `python/sgl_jax/srt/disaggregation/mini_lb_helpers.py`
  - 增加 `align_bootstrap_room_to_prefill`。
  - batched requests 现在使用 `room + i * prefill_count`，保证所有 item 都映射到同一个被选中的 prefill。
- `python/sgl_jax/srt/disaggregation/mini_lb.py`
  - `select_pair()` 返回 `prefill_index`。
  - Router 在生成 bootstrap field 时注入 `prefill_index/prefill_count`。
- `python/sgl_jax/srt/disaggregation/decode.py`
  - `_raiden_endpoint_for_dp()` 保留 Raiden 广播 endpoint 中的 host。
  - 这修复了实际观测到的错误连接：decode 曾尝试连接 `10.125.130.4:34189`，
    但该 Raiden endpoint 实际属于 `10.125.132.39:34189`。
- `python/sgl_jax/srt/disaggregation/jax_transfer/conn.py`
  - per-chunk start-read 日志改为 `debug` 级别。
- `python/sgl_jax/test/disaggregation/test_pd_mini_lb_helpers.py`
  - 增加 multi-prefill bootstrap-room alignment 测试。
- `python/sgl_jax/test/test_pd_swa_basic.py`
  - 增加 endpoint-host preservation 测试和热路径日志测试。

## Benchmark 摘要

16K input / 4K output：

| 模式 | C | total tok/s | input tok/s | output tok/s | peak output tok/s | mean TTFT ms | mean ITL ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| non-PD | 128 | 8617 | 6894 | 1723 | 4504 | 83256 | 53.54 |
| non-PD serve-level DP | 64 | 11700 | 9360 | 2340 | 3884 | 13675 | 23.89 |
| non-PD serve-level DP | 128 | 12779 | 10223 | 2556 | 5017 | 46493 | 34.56 |
| PD 1P1D | 128 | 13642 | 10913 | 2728 | 3483 | 57602 | 28.73 |
| PD 2P1D | 128 | 15307 | 12246 | 3061 | 3968 | 20114 | 35.08 |

主要结论：

- 原始 single-host non-PD baseline 证明同设备 prefill/decode contention 是真实存在的，但它不是 pod-count fair。
- 2026-07-06 two-host non-PD C64 run 在 C64 比 PD 更强：`11.70K total tok/s`，
  高于 PD 1P1D 的 `10.55K` 和 PD 2P1D 的 `10.83K`。
- two-host non-PD C128 run 是第一个公平的高压对比，并给了 PD 一个可测量优势：
  PD 1P1D 是 `13.64K`，two-host non-PD serve-level DP 是 `12.78K` total tok/s；
  PD 2P1D 是 `15.31K`。
- 2P1D 主要帮助高并发。C128 total throughput 比 1P1D 高约 `12%`，mean TTFT
  也明显更低。C64 只提升约 `3%`。
- 单个 decode host 仍是长输出阶段的限制因素。更多 prefill 能缓解 queueing 和
  prefill pressure，但不会从根本上改变 decode ITL。
- 2P1D 的 per-request transfer cost 仍然稳定：decode `kv_wait` 约 `2.31s`，
  prefill `transfer` 约 `2.56s`。
- C128 steady-state 视角：
  - PD 1P1D prefill active input 是 `12.79K tok/s`，decode highwater output 是
    `3.18K tok/s`，serve-internal PD handoff total 约 `4.80s`。
  - PD 2P1D prefill active input 是 `15.71K tok/s`，decode highwater output 是
    `3.63K tok/s`，serve-internal PD handoff total 约 `4.37s`。
  - Two-host non-PD C128 prefill active input 是 `11.64K tok/s`；rank-local decode
    highwater 很高，但不够对齐/持续，无法赢下完整 C128 run。
- 按 TensorRT-LLM rate matching 思路折算：
  - PD 1P1D C128 约 `0.780 prefill req/s @16K` 对 `0.776 decode req/s @4K`，
    P/D 基本匹配，是当前最干净的 pod-count-fair C128 结果。
  - PD 2P1D C128 约 `0.959 prefill req/s @16K` 对 `0.886 decode req/s @4K`，
    说明多 prefill 后 decode 开始更紧，下一步不应只继续加 prefill。
- non-PD serve-level DP 的 AIME24 follow-up 得到 `0.8667`（26/30）。之前 PD endpoint
  得到 `0.7667`（23/30）。由于使用 `temperature=1`，这应视为采样波动，而不是精度回退证据。

## 远端状态

主 Falcon exp：

- `exp-5uqgg64144`
- rank0：当前是 port `30010` 上的 non-PD server，IP `10.125.130.4`
- rank1：当前是 port `30010` 上的 non-PD server 加 port `30000` 上的 proxy，IP `10.125.129.4`

额外 prefill Falcon exp：

- `exp-ahgyl3g479`
- rank0：extra prefill，IP `10.125.132.39`

2026-07-06 follow-up 后，主 Falcon exp 不再运行 PD services。它当前运行的是
two-host non-PD follow-up 拓扑：

- rank0：port `30010` 上的 non-PD server。
- rank1：port `30010` 上的 non-PD server。
- rank1：port `30000` 上的 round-robin proxy。
- Run dir: `/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840`。

extra prefill Falcon exp `exp-ahgyl3g479` 没有参与这次 follow-up。旧的 prefill
process 可能还在，但主 rank0 bootstrap 在切到 non-PD 时已经停止。

有用的健康检查：

```bash
falcon exp exec exp-5uqgg64144 --rank 1 -- \
  "curl -sf http://localhost:30000/health"

falcon exp exec exp-5uqgg64144 --rank 0 -- \
  "curl -sf http://localhost:30010/health"
```

## 原始产物

主报告：

- `docs/developer_guide/pd_disaggregation/pd_mimo_v2_flash_final_perf_report_20260705.md`

Non-PD 报告：

- `docs/developer_guide/pd_disaggregation/nonpd_mimo_v2_flash_16k4k_baseline_20260705.md`

C128 steady-state advantage note：

- `docs/developer_guide/pd_disaggregation/pd_steady_state_advantage_20260706.md`

PD 1P1D 原始日志：

- `tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/`

Non-PD 原始日志：

- `tmp/e2e_logs/nonpd_16k_4k_1783265639/`

Non-PD two-host C64/AIME24 原始日志：

- `tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/`
- parsed summary: `tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/nonpd_2host_c64_parsed_summary.json`

Non-PD two-host C128 原始日志：

- `tmp/e2e_logs/nonpd_2host_c128_1783298516/`
- parsed summary: `tmp/e2e_logs/nonpd_2host_c128_1783298516/nonpd_2host_c128_parsed_summary.json`

PD 2P1D 原始日志：

- `tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/`
- parsed summary: `tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/parsed_summary.json`

## 验证

本轮工作期间跑过的本地 targeted tests：

```bash
.venv/bin/python -m pytest \
  python/sgl_jax/test/disaggregation/test_pd_mini_lb_helpers.py \
  python/sgl_jax/test/disaggregation/test_pd_router_admission.py -q

.venv/bin/python -m pytest \
  python/sgl_jax/test/test_pd_swa_basic.py \
  python/sgl_jax/test/disaggregation/test_pd_overlap_schedule.py \
  python/sgl_jax/test/disaggregation/test_pd_time_stats.py \
  python/sgl_jax/test/disaggregation/test_pd_internal_state.py -q
```

endpoint-host regression test 在修复前先失败，decode 改动后通过。

## 后续工作

建议下一步：

1. 继续聚焦 transfer 和 host/device scheduling overlap。Router admission 不是当前最重要的剩余成本。
2. 研究把 transfer discovery/progress 从 decode event-loop tick 中移出来，让 decode forward 和 incoming KV progress 更好地重叠。
3. 对 C128 anchor 增加 open-loop request-rate sweep，并按 goodput/SLO 口径汇报，
   不要只依赖 full-run average。建议 SLO：
   TTFT `<30s` 或 `<60s`，ITL `<40ms` 或 `<60ms`。
4. host 可用时评估 decode-heavy ratio，例如 `1P:2D` 或 `1P:3D`；也可以在
   decode 侧试 MTP。当前 `2P:1D` 已经显示 decode 更紧，公开 PD-ratio 报告也
   提醒 `3P:1D` 不一定是最佳 host 用法。
5. 把 precompile cache 视为启动优化，而不是运行时吞吐问题。当前 precompile 本身在模型加载后大约是几十秒；模型加载和 layout conversion 才是重启主要耗时。
