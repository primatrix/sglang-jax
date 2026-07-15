# PD 稳态优势分析, 2026-07-06

## 目的

这份记录单独抽出一个 PD 分离明显更有优势的 benchmark 视角：
`16K input / 4K output / C128` 高压场景。

核心原则是不要把 client 侧 TTFT 直接当作设备能力。client TTFT 包含
benchmark 一次性 burst 发请求、router/proxy 等待、server 队列以及最后
drain 的影响。衡量运行时能力时，优先看：

- `Prefill batch` 日志中的 prefill active input tok/s。
- `Decode batch` 日志中的 decode highwater output tok/s。
- `PD-TIME-STATS` 中的 PD serve 内部 handoff 时间。

## 数据来源

本地原始产物：

- PD 1P1D: `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/`
- PD 2P1D: `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/`
- non-PD two-host serve-level DP C128: `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c128_1783298516/`

最新本地报告版本以本文档所在 git commit 为准。

原始 PD 1P1D 远端 benchmark 代码版本：

```text
c6105f1cb09119ce40462d9f65776198a312737b
```

## 稳态定义

本记录使用下面的口径：

- **Prefill active window**：benchmark 窗口内第一条到最后一条
  `Prefill batch` 日志。
- **Decode highwater window**：满足
  `running-req >= 0.9 * max_observed_running_req` 的 `Decode batch` 行。

这个口径和 PDF 中 `running-req >= 0.9 * C` 的稳态规则略有差异。对于 C128
没有达到 128 个 active decode request 的运行，按实际观测到的最大 running
request 做 highwater 会更有解释力。

## 最好的 PD C128 数据

`16K/4K C128`，random dataset，request rate `inf`，`num_prompts=384`。

| 模式 | host 数 | client total tok/s | client input tok/s | client output tok/s | client peak output tok/s | client mean TTFT | client mean ITL |
|---|---:|---:|---:|---:|---:|---:|---:|
| non-PD serve-level DP | 2 | 12.78K | 10.22K | 2.56K | 5.02K | 46.49s | 34.56ms |
| PD 1P1D | 2 | 13.64K | 10.91K | 2.73K | 3.48K | 57.60s | 28.73ms |
| PD 2P1D | 3 | 15.31K | 12.25K | 3.06K | 3.97K | 20.11s | 35.08ms |

按 pod 数公平比较：

- PD 1P1D 的 client total tok/s 比 two-host non-PD C128 高约 `6.8%`。
- PD 1P1D 的 client mean TTFT 更差，因为 client TTFT 包含 burst queueing
  和 P/D handoff；但它的 mean ITL 更好。

按最高吞吐比较：

- PD 2P1D 的 client total tok/s 比 two-host non-PD C128 高约 `19.8%`，
  但它多使用了一个 prefill host。
- PD 2P1D 还把 client mean TTFT 从 PD 1P1D 的 `57.6s` 降到 `20.1s`，
  原因是两个 prefill worker 分摊了 burst prefill backlog。

## Serve Log 稳态数据

| 模式 | Prefill active window UTC | Prefill active input tok/s | Decode highwater window UTC | Decode highwater output tok/s | Decode highwater max output tok/s |
|---|---|---:|---|---:|---:|
| non-PD serve-level DP C128 | 两个 rank 都是 `00:42:56-00:51:55` | 11.64K combined | rank 窗口不对齐 | 4.56K rank-local highwater sum | 5.36K sum of rank max |
| PD 1P1D C128 | `13:34:40-13:42:52` | 12.79K | `13:37:41-13:43:02` | 3.18K | 3.41K |
| PD 2P1D C128 | rank0 `16:56:18-17:02:58`, rank2 `16:56:18-17:02:59` | 15.71K combined | `16:57:38-17:04:04` | 3.63K | 3.95K |

解读：

- PD 1P1D 是最干净的 pod-count-fair C128 胜出点。它的 sustained prefill
  active rate 高于 two-host non-PD：`12.79K` vs `11.64K` input tok/s。
- two-host non-PD 的 rank-local decode highwater 很强，但两个 rank 的
  highwater window 不对齐，而且整轮 client 平均 output tok/s 仍只有
  `2.56K`。瓶颈不是原始 decode kernel 能力，而是同设备 prefill/decode
  互相干扰以及 queue/tail 行为。
- PD 2P1D 把 prefill active capacity 提到 `15.71K` input tok/s，并且实际
  达到 `128` 个 observed decode running requests，是本轮测到的最高总吞吐。

## Serve 内部 TTFT 视角

PD request-time stats 给的是 request 进入 PD serving path 之后的 handoff
拆解。这和 client TTFT 不是同一个口径。

`16K/4K C128`，均值：

| 模式 | P queue | P forward | P transfer | P transfer tail | P total | D prealloc wait | D KV wait | D transfer tail | D total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PD 1P1D | 1596ms | 2879ms | 2567ms | 322ms | 4799ms | 2497ms | 2293ms | 45ms | 4790ms |
| PD 2P1D | 1387ms | mixed log schemas | 2564ms | n/a | 4407ms | 2055ms | 2313ms | 51ms | 4369ms |

因此，实际 serve 内部 first-token handoff 成本大约是：

- PD 1P1D C128：均值 `~4.80s`。
- PD 2P1D C128：均值 `~4.37s`。

这个数远小于 client TTFT（`57.6s` / `20.1s`），因为 client TTFT 还包含
benchmark burst 等待，以及 request 到达 active service path 之前的排队。
如果要汇报 device capacity，server-side phase stats 是更合适的诊断口径。

## 为什么这是 PD 有利场景

这个 C128 run 同时具备公开 PD 报告反复提到的三个有利条件：

1. 长 prefill：`16K` input 让 prefill 足够重，同设备 colocate prefill 和
   decode 时会出现可观测的干扰。
2. 长 decode：`4K` output 让 decode 保持足够长的 busy 时间，从而形成
   highwater steady region。
3. burst 压力：`request_rate=inf` 和 `C128` 会暴露 queue/tail 行为。

C64 下 two-host non-PD serve-level DP 更好，因此不能把结论泛化。当前可量化
表述应该是：

```text
在 16K/4K C128 burst pressure 下，PD 1P1D 有 pod-count-fair 的总吞吐优势；
PD 2P1D 是当前测试集合里的最高绝对吞吐配置。
```

## 对齐公开方法后的口径修订

参考公开材料后，这份报告应按下面方式阅读：

- DistServe 的核心口径是 **goodput under SLO**，即在 TTFT/TPOT SLO 约束下仍能
  服务的最大 request rate。因此我们当前的 `request_rate=inf` closed-loop
  benchmark 只能说明 burst 压力下的吞吐和稳态能力，不能直接等价为 SLO goodput。
- NVIDIA NIM / GenAI-Perf 的 metric 口径说明 TTFT 通常包含 request queueing、
  prefill 和 network latency；ITL/TPOT 应主要描述首 token 之后的 decode 部分。
  因此报告需要同时保留 client-visible TTFT 和 serve-internal phase timing，
  不能只用 client TTFT 判断 prefill device 能力。
- TensorRT-LLM 的 disaggregated serving 方法强调先分别测 context/prefill 的
  req/s/GPU、generation/decode 的 tok/s/user 和 tok/s，再做 rate matching。
  这比单纯比较 total tok/s 更能解释 P/D 配比是否合理。
- dstack 的 SGLang PD-ratio benchmark 提醒我们，固定 `2P:1D` 并不一定是最终
  最优配比。当前 2P1D 只说明多一个 prefill 能缓解 C128 burst backlog；还不能
  排除 `1P:2D`、`1P:3D` 在 decode-heavy 或更严格 ITL SLO 下更优。

按 TensorRT-LLM 的 rate matching 思路，把现有 C128 steady 数据折算成
request/s：

| 模式 | prefill active input tok/s | prefill req/s @16K | decode highwater output tok/s | decode req/s @4K | 解释 |
|---|---:|---:|---:|---:|---|
| PD 1P1D C128 | 12.79K | 0.780 | 3.18K | 0.776 | P/D 基本 rate matched，是最干净的 pod-count-fair C128 对比。 |
| PD 2P1D C128 | 15.71K | 0.959 | 3.63K | 0.886 | prefill 端已有余量，decode 端开始更紧；继续加 P 的收益可能下降。 |

因此，当前报告中更合理的表述是：

```text
PD 1P1D 在 16K/4K C128 上已经接近 P/D rate matching；
PD 2P1D 提高了 burst 下的 prefill active capacity 和端到端吞吐，
但从 steady rate matching 看，下一步更应该验证 1P:2D / 1P:3D 或 MTP，
而不是继续单向增加 prefill。
```

## 可参考的公开 PD 报告

有价值的公开参考：

- [DistServe OSDI 2024](https://arxiv.org/html/2401.09670v3)：把 goodput
  定义为满足 TTFT/TPOT SLO 的最大 request rate，并报告严格 SLO 下的大幅收益。
  对我们的启发是要汇报 goodput 和 SLO attainment，而不是只看平均 tok/s。
- [Splitwise](https://arxiv.org/html/2311.18677v2)：生产 trace 说明了
  prefill/decode 分离的动机：prefill 偏 compute-heavy，decode 偏
  memory-bandwidth-heavy。它的收益主要来自按 phase 分别匹配资源。
- [TensorRT-LLM disaggregated serving](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)：
  报告了 `4400/1200`、`8192/256`、`8192/1024` 等长输入场景的收益，并建议
  先测 context req/s/GPU 和 generation tok/s/user，再做 rate matching。
- [vLLM MORI-IO KV connector](https://vllm.ai/blog/2026-04-07-moriio-kv-connector)：
  在 `2000/1000` workload 上展示了 PD-style setup 对 SLO goodput 的提升，
  同时也体现了失败项从 ITL spike 转向 TTFT 的 tradeoff。
- [dstack SGLang PD ratio benchmark](https://dstack.ai/blog/benchmarking-pd-ratios/)：
  对比了 `3P:1D`、`2P:2D`、`1P:3D` 在 C32/C64/C128 下的表现。对我们最重要
  的启发是也要测试 decode-heavy ratio；`3P:1D` 不一定是 host 的最佳用法。
- [NVIDIA Dynamo disaggregated serving docs](https://docs.nvidia.com/dynamo/v-0-7-1/design-docs/disaggregated-serving)：
  明确指出 remote prefill 适合长上下文请求，而短 prompt 或 prefix cache hit
  很高时，本地 prefill 可能更高效。
- [NVIDIA NIM / GenAI-Perf metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)：
  对指标定义很有用。TTFT 通常包含 queueing 和 network time，因此需要同时汇报
  client-visible metrics 和 server-side phase metrics。

## 建议的后续测试矩阵

优先级：

1. 保留 `16K/4K C128` 作为 PD 有利锚点，同时汇报 client throughput 和
   server-side steady input/output tok/s。
2. 增加 C128 等价压力下的 open-loop request-rate sweep，并按 SLO 计算
   goodput。例如 TTFT `<30s` 或 `<60s`，ITL `<40ms` 或 `<60ms`。
3. host 足够时测试配比：
   - `1P:1D` 作为 pod-count-fair baseline。
   - `2P:1D` 作为当前最高绝对吞吐配置。
   - `1P:2D` 和 `1P:3D`，验证长输出/reasoning 是否已经 decode constrained，
     从而让更多 decode host 优于更多 prefill host。
4. 增加长度组合：
   - `16K/512`：长输入短输出，预期更偏向多 prefill。
   - `2K/4K`：decode-heavy，预期更偏向多 decode。
   - `2K/2K`：balanced。
5. 增加混合负载，例如 80% `1K/256` 加 20% `16K/4K`，观察 PD 是否能保护短请求
   的 ITL/TPOT，不被长 prefill 干扰。
6. 增加 prefix-cache hit 变体：0%、50%、80%。公开系统经验提醒我们，高 cache hit
   场景可能让 local prefill 优于 remote prefill。

## 汇报模板

每轮 run 建议汇报：

- Client-visible：total tok/s、input tok/s、output tok/s、peak output tok/s、
  mean/p99 TTFT、mean/p99 ITL、success count。
- Server prefill：active window、active input tok/s、max queue；如果是 PD，再给
  forward/transfer stage means。
- Server decode：highwater window、highwater output tok/s、max running request
  count、max queue。
- PD handoff：prealloc wait、KV wait、transfer tail、total。
- Goodput：满足明确 TTFT 和 ITL SLO 的最大 request rate。
