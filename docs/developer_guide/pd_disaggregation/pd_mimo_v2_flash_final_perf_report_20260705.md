# MiMo-V2-Flash PD 分离性能报告, 2026-07-05

## 摘要

- 本轮小优化是在 decode overlap loop 里，当前 batch `run_batch` launch 后、上一批 `process_batch_result` 前，额外 poll 一次 transfer queue。单测通过，但 C32/C64/C128 回归显示吞吐收益很小，基本在噪声范围内。
- 16K input / 4K output 最好点仍是 C128：client 全程平均 `13.64K total tok/s`，其中 `10.91K input tok/s`、`2.73K output tok/s`；client peak output `3.48K tok/s`。
- 用 server log 看 device 能力：prefill forward-only 估计约 `22.8K input tok/s`（router prefill inflight=4 反推），但包含 transfer/tail 后的实际 active prefill 只有 `12.8K input tok/s`；decode highwater steady 约 `3.18K output tok/s`，max `3.41K output tok/s`。
- 额外做了 `2 prefill : 1 decode` 探索。C64 收益很小（`10.83K total tok/s`，比 1P1D C64 高约 `2.7%`）；C128 提升更明显（`15.31K total tok/s`，比 1P1D C128 高约 `12.2%`，mean TTFT 从 `57.6s` 降到 `20.1s`），但长输出阶段仍由单 decode device 主导。
- 2026-07-06 补了更公平的两机 non-PD serve-level DP。C64 是 `11.70K total tok/s`、`2.34K output tok/s`，高于 PD 1P1D/2P1D 的 C64；但 C128 是 `12.78K total tok/s`、`2.56K output tok/s`，低于 PD 1P1D C128 的 `13.64K` 和 PD 2P1D C128 的 `15.31K`。因此 PD 优势主要出现在更高压的 C128，而不是 C64。
- 2026-07-06 追加了 C128 steady-state 口径分析：
  [pd_steady_state_advantage_20260706.md](/Users/jiongxuan/workspace/sglang-jax/docs/developer_guide/pd_disaggregation/pd_steady_state_advantage_20260706.md)。在该口径下，PD 1P1D C128 的 prefill active input 为 `12.79K tok/s`，PD 2P1D 为 `15.71K tok/s`；serve-internal PD handoff total 分别约 `4.80s` / `4.37s`，明显小于 client TTFT，因为 client TTFT 包含 burst queueing。
- 对齐 DistServe / TensorRT-LLM / dstack / NVIDIA NIM 的公开口径后，本报告应被理解为 `request_rate=inf` closed-loop burst benchmark 加 serve-log 稳态分析，而不是 SLO goodput 结果。现有 C128 steady 数据显示 PD 1P1D 的 prefill/decode 基本 rate matched（`0.780` vs `0.776 req/s`），PD 2P1D 则开始偏 decode-constrained（`0.959` vs `0.886 req/s`）。
- AIME24 两次完整 30 题：PD endpoint 为 `0.7667`（23/30），两机 non-PD serve-level DP endpoint 为 `0.8667`（26/30）。由于使用 `temperature=1` 非贪心采样，这个差异更像采样波动；没有看到精度异常信号。

## 测试代码

远端 benchmark 代码：

- Falcon exp: `exp-5uqgg64144`
- 远端 repo: `/tmp/sglang-jax`
- 远端 git head: `c6105f1cb09119ce40462d9f65776198a312737b`
- 远端 working tree 是 dirty 状态；精确 dirty status 已保存在：
  `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/env.json`
- 本轮在 [decode.py](/Users/jiongxuan/workspace/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py:301) 增加了额外 decode transfer poll。
- Multi-prefill follow-up 修复：
  - Router 会把注入的 `bootstrap_room` 与选中的 prefill index 对齐，使 bootstrap registry selection 与实际 forwarded prefill URL 一致。
  - Decode 保留 Raiden endpoint descriptor 中声明的 host，而不是重写成 bootstrap registry host。这个修复解决了一个真实 `2P1D` 失败：decode 曾尝试连接 `10.125.130.4:34189`，但该 Raiden endpoint 实际属于 `10.125.132.39:34189`。
  - Per-chunk `RAIDEN-D start_read*` 日志从 warning 降为 debug；这样能去掉热路径日志噪声，并且不改变 transfer 行为。

本地验证代码：

- 本地 git head: `df0e812fad11c3fe2bbe30514ee136ef899b5ad6`
- 在 [test_pd_overlap_schedule.py](/Users/jiongxuan/workspace/sglang-jax/python/sgl_jax/test/disaggregation/test_pd_overlap_schedule.py:137) 增加 regression test。
- 验证：`.venv/bin/python -m pytest python/sgl_jax/test/disaggregation/test_pd_overlap_schedule.py python/sgl_jax/test/disaggregation/test_pd_time_stats.py python/sgl_jax/test/disaggregation/test_pd_internal_state.py -q` -> `18 passed in 3.35s`。

## 环境

- Rank 0：PD bootstrap + prefill server
- Rank 1：decode server + PD router + benchmark/eval driver
- Model: `/models/MiMo-V2-Flash`
- Python: `3.12.12`
- Runtime env:

```bash
export TPU_PROCESS_ADDRESSES=localhost:8471
export TPU_WORKER_HOSTNAMES=localhost
export TPU_PROCESS_PORT=8471
export TPU_WORKER_ID=0
export TPU_HOST_BOUNDS=1,1,1
export HOST_BOUNDS=1,1,1
export TPU_TOPOLOGY=2x2x1
export TMPDIR=/tmp/tpu_logs/tmp
export PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache
export JAX_COMPILATION_CACHE_DIR=/tmp/tpu_logs/jit_cache
export LIBTPU_INIT_ARGS=--xla_tpu_dvfs_p_state=7
export PYTHONPATH=/tmp/tpu-raiden-cached/tpu-raiden:/tmp/sglang-jax:${PYTHONPATH:-}
export SGLANG_JAX_USE_RAIDEN=1
```

通过 `JAX_COMPILATION_CACHE_DIR=/tmp/tpu_logs/jit_cache` 启用了 precompile cache。本次重启中，prefill/decode 在模型加载后的 precompile 都约 17s 完成。

## Serve 命令

Bootstrap, rank 0:

```bash
/usr/local/bin/python -m sgl_jax.srt.disaggregation.run_bootstrap \
  --host 0.0.0.0 --port 8998
```

Prefill, rank 0:

```bash
/usr/local/bin/python -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash --trust-remote-code \
  --tp-size 8 --ep-size 8 --moe-backend fused_v2 \
  --nnodes 1 --node-rank 0 --page-size 256 --context-length 262144 \
  --disable-radix-cache --chunked-prefill-size 2048 --max-prefill-tokens 16384 \
  --dtype bfloat16 --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup --log-level info --max-running-requests 256 \
  --dp-size 2 --dp-schedule-policy round_robin \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --disaggregation-enable-d2h --disaggregation-use-raiden \
  --enable-metrics --enable-request-time-stats-logging \
  --host 0.0.0.0 --port 10000 \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-url http://localhost:8998 \
  --disaggregation-max-inflight-transfers 8
```

Decode, rank 1:

```bash
/usr/local/bin/python -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash --trust-remote-code \
  --tp-size 8 --ep-size 8 --moe-backend fused_v2 \
  --nnodes 1 --node-rank 0 --page-size 256 --context-length 262144 \
  --disable-radix-cache --chunked-prefill-size 2048 --max-prefill-tokens 16384 \
  --dtype bfloat16 --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup --log-level info --max-running-requests 256 \
  --dp-size 2 --dp-schedule-policy round_robin \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --disaggregation-enable-d2h --disaggregation-use-raiden \
  --enable-metrics --enable-request-time-stats-logging \
  --host 0.0.0.0 --port 10001 \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-url http://falcon-job-p6bmn75fnu-0.falcon-job-p6bmn75fnu.falcon-jobs.svc.cluster.local:8998 \
  --disaggregation-max-inflight-transfers 32
```

Router, rank 1:

```bash
/usr/local/bin/python -m sgl_jax.srt.disaggregation.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill http://falcon-job-p6bmn75fnu-0.falcon-job-p6bmn75fnu.falcon-jobs.svc.cluster.local:10000 8998 \
  --decode http://localhost:10001 \
  --prefill-bootstrap-host falcon-job-p6bmn75fnu-0.falcon-job-p6bmn75fnu.falcon-jobs.svc.cluster.local \
  --max-concurrent-requests 256 \
  --pd-prefill-max-inflight-requests 4 \
  --pd-decode-prealloc-soft-limit 0 \
  --pd-decode-oldest-prealloc-wait-ms-soft-limit 0 \
  --pd-router-admission-poll-ms 50 \
  --host 0.0.0.0 --port 30000
```

## Benchmark 命令

16K/4K throughput:

```bash
for C in 32 64 128; do
  NUM=$((C * 3))
  /usr/local/bin/python -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --base-url http://localhost:30000 \
    --model /models/MiMo-V2-Flash \
    --tokenizer /models/MiMo-V2-Flash \
    --dataset-name random \
    --random-input-len 16384 \
    --random-output-len 4096 \
    --random-range-ratio 1.0 \
    --num-prompts "${NUM}" \
    --request-rate inf \
    --max-concurrency "${C}" \
    --warmup-requests 0 \
    --seed 12345 \
    --output-details \
    --extra-request-body '{"sampling_params":{"temperature":0.1,"top_p":0.95,"max_new_tokens":4096,"ignore_eos":true}}' \
    --output-file "/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/bench_c${C}.jsonl"
done
```

AIME24:

```bash
/usr/local/bin/python -m pip install -q evalscope==0.17.1
/usr/local/bin/python -m evalscope.run \
  --model /models/MiMo-V2-Flash \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets aime24 \
  --eval-batch-size 16 \
  --timeout 6000000 \
  --work-dir /aime24_workdir \
  --generation-config '{"temperature":1,"top_p":0.95,"max_tokens":30000,"chat_template_kwargs":{"enable_thinking":true}}'
```

## 吞吐结果

Client 全程平均数据，16K/4K：

| Run | C | req/s | input tok/s | output tok/s | client peak output tok/s | total tok/s | mean TTFT ms | p99 TTFT ms | mean ITL ms | p99 ITL ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| extra poll 前 | 32 | 0.39 | 6350 | 1588 | 1984 | 7938 | 10175 | 41913 | 16.68 | 42.24 |
| extra poll 后 | 32 | 0.38 | 6270 | 1568 | 2016 | 7838 | 10266 | 41479 | 16.96 | 41.19 |
| extra poll 前 | 64 | 0.52 | 8589 | 2147 | 2655 | 10736 | 16939 | 83534 | 23.55 | 81.01 |
| extra poll 后 | 64 | 0.51 | 8437 | 2109 | 2653 | 10546 | 16476 | 81917 | 24.13 | 82.07 |
| extra poll 前 | 128 | 0.66 | 10867 | 2717 | 3491 | 13584 | 61510 | 162593 | 27.91 | 88.66 |
| extra poll 后 | 128 | 0.67 | 10913 | 2728 | 3483 | 13642 | 57602 | 161325 | 28.73 | 89.26 |

Server 侧 prefill input capacity：

| Run | C | observed active input tok/s | observed active req/s @16K | forward mean ms | forward-only capacity @4 inflight | prefill total mean ms |
|---|---:|---:|---:|---:|---:|---:|
| extra poll 前 | 32 | 8322 | 0.508 | 2844 | 23044 tok/s | 4056 |
| extra poll 后 | 32 | 8150 | 0.497 | 2860 | 22922 tok/s | 4168 |
| extra poll 前 | 64 | 10736 | 0.655 | 2689 | 24372 tok/s | 3491 |
| extra poll 后 | 64 | 10556 | 0.644 | 2693 | 24343 tok/s | 3501 |
| extra poll 前 | 128 | 12684 | 0.774 | 2899 | 22608 tok/s | 4838 |
| extra poll 后 | 128 | 12788 | 0.780 | 2879 | 22764 tok/s | 4799 |

解读：当 router 保持 4 个 prefill request in flight 时，P 侧纯 forward capacity 大致在 `22K-24K input tok/s` 区间。但实际 active prefill ingress rate 要低很多，C128 约 `12.8K input tok/s`，因为 transfer/prealloc/tail 仍在 critical path 上。

Server 侧 decode output capacity：

| Run | C | max running | all mean output tok/s | all max output tok/s | highwater steady mean | highwater steady max |
|---|---:|---:|---:|---:|---:|---:|
| extra poll 前 | 32 | 32 | 1596 | 1983 | 1847 | 1983 |
| extra poll 后 | 32 | 32 | 1594 | 2016 | 1852 | 2016 |
| extra poll 前 | 64 | 64 | 2069 | 2648 | 2505 | 2648 |
| extra poll 后 | 64 | 64 | 2039 | 2607 | 2462 | 2607 |
| extra poll 前 | 128 | 98 | 2591 | 3374 | 3161 | 3374 |
| extra poll 后 | 128 | 100 | 2616 | 3414 | 3180 | 3414 |

因此 decode peak 大约是 steady/highwater `3.2K output tok/s`，instantaneous serve-log max `3.4K output tok/s`。额外 poll 没有实质改变这个结果。

## 稳态窗口

下面的 steady window 是从 benchmark window 内的 server logs 计算出来的：

- Prefill active window：对应 concurrency case 的第一条到最后一条 `Prefill batch` 日志。
- Decode highwater steady：满足 `running-req >= 0.9 * max_observed_running_req` 的行。C128 中 decode 观测到的 max running 是 100 而不是 128，因此这个 highwater 定义比 PDF 的 `0.9 * bs` 规则更有解释力。

| C | prefill active window UTC | prefill active duration s | prefill active input tok/s | decode highwater threshold | decode steady window UTC | decode steady duration s | highwater mean output tok/s | highwater max output tok/s |
|---:|---|---:|---:|---:|---|---:|---:|---:|
| 32 | 13:22:56-13:26:09 | 193 | 8150 | >= 29/32 | 13:23:35-13:26:43 | 188 | 1852 | 2016 |
| 64 | 13:27:42-13:32:40 | 298 | 10556 | >= 58/64 | 13:28:59-13:32:58 | 239 | 2462 | 2607 |
| 128 | 13:34:40-13:42:52 | 492 | 12788 | >= 90/100 | 13:37:41-13:43:02 | 321 | 3180 | 3414 |

## 公开口径对齐

参考公开 PD benchmark 方法后，当前报告的结论需要按下面口径阅读：

- DistServe 的 goodput 是 **满足 TTFT/TPOT SLO 的最大 request rate**。本报告的
  `request_rate=inf` benchmark 是 closed-loop burst 压力测试，只能说明 burst
  下的吞吐、排队和 steady device capacity，不能直接等价为 SLO goodput。
- NVIDIA NIM / GenAI-Perf 的 TTFT 口径通常包含 request queueing、prefill、
  network latency 等端到端等待；ITL/TPOT 更接近首 token 之后的 decode 行为。
  因此本报告同时保留 client-visible TTFT/ITL 和 serve-internal phase timing。
- TensorRT-LLM 的 disaggregated serving 方法更强调先分别测 prefill/context
  req/s 与 decode/generation tok/s，再做 rate matching。按这个思路看现有
  `16K/4K C128` steady 数据：

| 模式 | prefill active input tok/s | prefill req/s @16K | decode highwater output tok/s | decode req/s @4K | 结论 |
|---|---:|---:|---:|---:|---|
| PD 1P1D C128 | 12.79K | 0.780 | 3.18K | 0.776 | P/D 基本 rate matched，是最干净的 pod-count-fair C128 对比。 |
| PD 2P1D C128 | 15.71K | 0.959 | 3.63K | 0.886 | prefill 端已有余量，decode 端开始更紧；继续加 prefill 的边际收益可能下降。 |

因此更稳妥的阶段性结论是：`PD 1P1D` 在 `16K/4K C128` 上已经接近 P/D
rate matching；`PD 2P1D` 提高了 burst 下的 prefill active capacity 和总吞吐，
但下一轮配比实验应优先覆盖 `1P:2D`、`1P:3D` 或 decode MTP，而不是只继续增加
prefill 节点。

## Transfer / Time Stats

Per-request PD time stats 均值：

| Run | C | P forward ms | P transfer ms | P transfer_tail ms | P total ms | D prealloc_wait ms | D kv_wait ms | D transfer_tail ms | D total ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| extra poll 前 | 32 | 2844 | 2534 | 307 | 4056 | 1710 | 2279 | 50 | 3990 |
| extra poll 后 | 32 | 2860 | 2544 | 308 | 4168 | 1819 | 2284 | 46 | 4103 |
| extra poll 前 | 64 | 2689 | 2395 | 264 | 3491 | 1270 | 2182 | 48 | 3453 |
| extra poll 后 | 64 | 2693 | 2401 | 268 | 3501 | 1258 | 2180 | 46 | 3439 |
| extra poll 前 | 128 | 2899 | 2591 | 329 | 4838 | 2521 | 2320 | 55 | 4842 |
| extra poll 后 | 128 | 2879 | 2567 | 322 | 4799 | 2497 | 2293 | 45 | 4790 |

主要结论：

- Prefill chunk transfer 已经和 forward 部分 overlap，但剩余 transfer/tail 仍然把 C128 的 realized prefill capacity 从 forward-only 的约 `22.8K input tok/s` 降到 active end-to-end 的约 `12.8K input tok/s`。
- Decode receive path 在 request 级别仍基本串行：`prealloc_wait + kv_wait ~= total`。extra poll 后的 C128 是 `2497 + 2293 ~= 4790 ms`。
- launch 后额外 poll 几乎没有可测收益。它风险较低，但不是瓶颈。

Extra poll 后的详细 prefill per-request stage stats：

| C | stage | n | mean ms | p50 ms | p95 ms | p99 ms | max ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | queue | 96 | 997.5 | 1242.8 | 1700.7 | 1915.7 | 2509.7 |
| 32 | forward | 96 | 2859.7 | 2901.4 | 3039.2 | 3051.6 | 3057.5 |
| 32 | chunk_register_span | 96 | 2234.0 | 2257.1 | 2374.9 | 2398.5 | 2403.5 |
| 32 | sender_done_wait | 96 | 308.2 | 323.8 | 340.1 | 346.1 | 359.6 |
| 32 | transfer_tail | 96 | 308.4 | 324.0 | 340.4 | 346.5 | 359.8 |
| 32 | transfer | 96 | 2543.8 | 2578.4 | 2705.2 | 2715.2 | 2724.6 |
| 32 | total | 96 | 4167.5 | 4442.9 | 5019.7 | 5080.9 | 5675.6 |
| 64 | queue | 192 | 537.5 | 0.9 | 1692.0 | 1765.3 | 2509.0 |
| 64 | forward | 192 | 2692.9 | 2774.7 | 3030.5 | 3058.0 | 3070.7 |
| 64 | chunk_register_span | 192 | 2130.8 | 2190.5 | 2360.2 | 2395.1 | 2445.9 |
| 64 | sender_done_wait | 192 | 267.9 | 276.2 | 339.5 | 344.2 | 348.3 |
| 64 | transfer_tail | 192 | 268.4 | 276.7 | 340.0 | 344.7 | 348.8 |
| 64 | transfer | 192 | 2400.5 | 2461.6 | 2695.5 | 2721.5 | 2748.6 |
| 64 | total | 192 | 3500.7 | 3061.5 | 5029.7 | 5056.1 | 5720.4 |
| 128 | queue | 384 | 1595.6 | 1625.4 | 1699.4 | 1743.3 | 2422.0 |
| 128 | forward | 384 | 2879.0 | 2920.1 | 3032.1 | 3047.8 | 3140.1 |
| 128 | chunk_register_span | 384 | 2243.6 | 2273.2 | 2366.8 | 2419.5 | 2510.2 |
| 128 | sender_done_wait | 384 | 320.5 | 326.1 | 342.5 | 346.5 | 361.7 |
| 128 | transfer_tail | 384 | 322.3 | 327.5 | 344.2 | 348.3 | 363.3 |
| 128 | transfer | 384 | 2567.1 | 2597.9 | 2703.1 | 2749.8 | 2835.3 |
| 128 | total | 384 | 4798.8 | 4860.0 | 5031.4 | 5072.2 | 5669.1 |

Extra poll 后的详细 decode per-request stage stats：

| C | stage | n | mean ms | p50 ms | p95 ms | p99 ms | max ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | metadata_wait | 96 | 1818.8 | 2002.3 | 2627.6 | 2772.2 | 3384.4 |
| 32 | prealloc_wait | 96 | 1819.2 | 2002.6 | 2628.0 | 2772.4 | 3384.7 |
| 32 | first_chunk_wait | 96 | 4.6 | 5.0 | 6.6 | 8.0 | 9.3 |
| 32 | chunk_start_span | 96 | 2230.4 | 2251.9 | 2368.6 | 2399.6 | 2405.2 |
| 32 | transfer_tail | 96 | 45.8 | 44.8 | 58.5 | 60.6 | 62.3 |
| 32 | kv_wait | 96 | 2283.6 | 2310.4 | 2422.7 | 2454.6 | 2454.7 |
| 32 | enqueue_decode | 96 | 2.7 | 2.7 | 3.4 | 3.8 | 4.0 |
| 32 | total | 96 | 4103.5 | 4320.9 | 4953.6 | 5079.1 | 5687.2 |
| 64 | metadata_wait | 192 | 1257.9 | 733.2 | 2649.0 | 2719.0 | 3377.2 |
| 64 | prealloc_wait | 192 | 1258.3 | 734.0 | 2649.3 | 2719.3 | 3377.5 |
| 64 | first_chunk_wait | 192 | 3.9 | 3.5 | 6.1 | 7.1 | 7.8 |
| 64 | chunk_start_span | 192 | 2127.1 | 2188.3 | 2355.5 | 2387.4 | 2442.8 |
| 64 | transfer_tail | 192 | 46.3 | 47.2 | 58.3 | 66.2 | 72.1 |
| 64 | kv_wait | 192 | 2180.1 | 2239.2 | 2412.6 | 2443.5 | 2500.7 |
| 64 | enqueue_decode | 192 | 2.8 | 3.0 | 3.8 | 4.3 | 4.4 |
| 64 | total | 192 | 3439.2 | 3007.4 | 5039.4 | 5081.8 | 5682.9 |
| 128 | metadata_wait | 384 | 2496.7 | 2537.9 | 2643.2 | 2705.3 | 3238.2 |
| 128 | prealloc_wait | 384 | 2497.0 | 2538.2 | 2643.5 | 2705.6 | 3238.7 |
| 128 | first_chunk_wait | 384 | 6.2 | 6.1 | 7.5 | 9.1 | 10.7 |
| 128 | chunk_start_span | 384 | 2237.3 | 2271.2 | 2361.4 | 2414.7 | 2501.3 |
| 128 | transfer_tail | 384 | 45.2 | 44.4 | 55.1 | 59.6 | 60.9 |
| 128 | kv_wait | 384 | 2292.5 | 2324.7 | 2418.8 | 2466.6 | 2565.5 |
| 128 | enqueue_decode | 384 | 3.8 | 3.8 | 4.5 | 4.9 | 5.4 |
| 128 | total | 384 | 4790.2 | 4851.1 | 5029.1 | 5088.0 | 5551.3 |

## 2P1D 多 Prefill 探索

这轮使用两个 prefill TPU host 供给一个 decode TPU host：

- 原始 prefill：`exp-5uqgg64144` rank 0, `10.125.130.4`。
- 额外 prefill：`exp-ahgyl3g479` rank 0, `10.125.132.39`。
- Decode/router/driver：`exp-5uqgg64144` rank 1, `10.125.129.4`。
- Router 使用两个 prefill URL 和一个 decode URL，并保持同样的 `--pd-prefill-max-inflight-requests 4`。
- Raw run id: `pd_2p1d_16k_4k_bench_1783270151`。

Extra prefill serve command 与上面的 rank0 prefill command 相同，区别是：

```bash
--disaggregation-bootstrap-url http://falcon-job-p6bmn75fnu-0.falcon-job-p6bmn75fnu.falcon-jobs.svc.cluster.local:8998
```

2P1D router command:

```bash
/usr/local/bin/python -m sgl_jax.srt.disaggregation.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill http://falcon-job-p6bmn75fnu-0.falcon-job-p6bmn75fnu.falcon-jobs.svc.cluster.local:10000 8998 \
  --prefill http://10.125.132.39:10000 8998 \
  --decode http://localhost:10001 \
  --prefill-bootstrap-host falcon-job-p6bmn75fnu-0.falcon-job-p6bmn75fnu.falcon-jobs.svc.cluster.local \
  --max-concurrent-requests 256 \
  --pd-prefill-max-inflight-requests 4 \
  --pd-decode-prealloc-soft-limit 0 \
  --pd-decode-oldest-prealloc-wait-ms-soft-limit 0 \
  --pd-router-admission-poll-ms 50 \
  --host 0.0.0.0 --port 30000
```

Client 结果，16K/4K：

| Mode | C | success | req/s | input tok/s | output tok/s | peak output tok/s | total tok/s | mean TTFT ms | p99 TTFT ms | mean ITL ms | p99 ITL ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PD 1P1D | 64 | 192/192 | 0.51 | 8437 | 2109 | 2653 | 10546 | 16476 | 81917 | 24.13 | 82.07 |
| PD 2P1D | 64 | 192/192 | 0.53 | 8663 | 2166 | 2844 | 10829 | 11488 | 47663 | 25.84 | 79.60 |
| PD 1P1D | 128 | 384/384 | 0.67 | 10913 | 2728 | 3483 | 13642 | 57602 | 161325 | 28.73 | 89.26 |
| PD 2P1D | 128 | 384/384 | 0.75 | 12246 | 3061 | 3968 | 15307 | 20114 | 91477 | 35.08 | 128.74 |

2P1D per-request stage stats：

| C | role | n | forward mean ms | transfer / kv_wait mean ms | total mean ms | p50 total ms | p95 total ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | prefill | 192 | 1649 | 2574 transfer | 4323 | 4875 | 5243 |
| 64 | decode | 192 | n/a | 2321 kv_wait | 4279 | 4853 | 5233 |
| 128 | prefill | 384 | 1512 | 2564 transfer | 4407 | 4844 | 5272 |
| 128 | decode | 384 | n/a | 2313 kv_wait | 4369 | 4838 | 5273 |

重要观察：

- `2P1D` 相比 1P1D 把 C128 total throughput 提升约 `12.2%`（`15.31K / 13.64K`），并把 mean TTFT 降低约 `65%`（`57.6s -> 20.1s`）。它有帮助，是因为 prefill queueing pressure 被两个 P host 分摊。
- C64 提升很小（`10.83K / 10.55K`，约 `2.7%`），说明这个 concurrency 下一个 prefill 已经基本够用。
- Decode 仍是 4K 长输出的限制角色。C128 output throughput 提升到 `3.06K tok/s`，client peak `3.97K tok/s`，但 ITL 变差（`28.73ms -> 35.08ms`），因为单个 decode host 承载了更大的 effective running set。
- 远端 extra prefill 没有让 transfer 本身回退：decode `kv_wait` 仍约 `2.31s`；prefill transfer 仍约 `2.56s`。

## Non-PD Two-Host Serve-Level DP 补充测试

原始 non-PD 对比只使用一个 server host，因此不能和 PD 1P1D 做 pod-count-fair
比较。2026-07-06 使用两个普通 non-PD server 重跑 C64，每个 Falcon rank 一个，
后面接一个轻量 streaming round-robin proxy。

Run ids：

```text
C64 + AIME24: nonpd_2host_c64_aime24_1783295840
C128:         nonpd_2host_c128_1783298516
```

Client 结果，16K input / 4K output：

| Mode | Hosts | C | success | total tok/s | input tok/s | output tok/s | peak output tok/s | mean TTFT ms | mean ITL ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| non-PD single server | 1 | 64 | 192/192 | 7106 | 5685 | 1421 | 2688 | 42904 | 34.55 |
| non-PD serve-level DP | 2 | 64 | 192/192 | 11700 | 9360 | 2340 | 3884 | 13675 | 23.89 |
| PD 1P1D | 2 | 64 | 192/192 | 10546 | 8437 | 2109 | 2653 | 16476 | 24.13 |
| PD 2P1D | 3 | 64 | 192/192 | 10829 | 8663 | 2166 | 2844 | 11488 | 25.84 |
| non-PD serve-level DP | 2 | 128 | 383/384 | 12779 | 10223 | 2556 | 5017 | 46493 | 34.56 |
| PD 1P1D | 2 | 128 | 384/384 | 13642 | 10913 | 2728 | 3483 | 57602 | 28.73 |
| PD 2P1D | 3 | 128 | 384/384 | 15307 | 12246 | 3061 | 3968 | 20114 | 35.08 |

Two-host runs 的 serve-log summary：

| C | Rank | prefill active input tok/s | decode high-load mean tok/s | decode max tok/s | max running |
|---:|---:|---:|---:|---:|---:|
| 64 | 0 | 5958 | 1879 | 1970 | 40 |
| 64 | 1 | 5935 | 1882 | 1963 | 48 |
| 64 | combined | 11893 | 3761 | 3933 | n/a |
| 128 | 0 | 5806 | 2279 | 2492 | 57 |
| 128 | 1 | 5836 | 2393 | 2868 | 75 |
| 128 | combined | 11642 | 4672 | 5361 | n/a |

解读：

- C64 下 serve-level DP 比 PD 更强，因为每个 server 处理约一半 burst，KV local 且没有 transfer path。
- C128 下，每个 non-PD server 实际接近 C64 压力，同设备 prefill/decode contention 重新出现。按 total tok/s，PD 1P1D 比 two-host non-PD 高约 `6.8%`，PD 2P1D 高约 `19.8%`，但 PD 2P1D 多用一个 prefill host。
- 因此结论需要收窄：不能说 PD 1P1D 在 C64 一定比 pod-count-fair non-PD deployment 更快；但在 C128 它确实重新获得可测量优势。

## AIME24

| Dataset | Num | Score | Correct |
|---|---:|---:|---:|
| PD endpoint, AIME24 / `HuggingFaceH4/aime_2024` | 30 | 0.7667 | 23 |
| non-PD serve-level DP endpoint, AIME24 / `HuggingFaceH4/aime_2024` | 30 | 0.8667 | 26 |

PD 结果在 non-greedy reasoning config 下与 PDF 数字（`0.7667`）一致。two-host
non-PD rerun 更高，但 config 使用 `temperature=1`，所以除非增加 deterministic
accuracy protocol，否则 23/30 vs 26/30 的差异应视为采样波动。

## 原始产物

本地产物：

- Parsed summary：`/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/parsed_summary.json`
- 2P1D parsed summary：`/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/parsed_summary.json`
- Rank 1 raw tar：`/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/rank1.tar.gz`
- Rank 0 raw tar：`/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/rank0.tar.gz`
- Benchmark 日志：
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/raw/bench_c32.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/raw/bench_c64.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/raw/bench_c128.log`
- Server 日志：
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/prefill_extra_poll_server.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/decode_extra_poll_server.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/router_extra_poll.log`
- AIME24:
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/raw/aime24_eval.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/aime24_workdir/20260705_135200/reports/MiMo-V2-Flash/aime24.json`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_16k_4k_extra_poll_1783257767/pd_16k_4k_extra_poll_1783257767/aime24_workdir/20260705_135200/predictions/MiMo-V2-Flash/aime24_default.jsonl`
- non-PD two-host C64/AIME24 follow-up:
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/nonpd_2host_c64_aime24_1783295840_rank0.tar.gz`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/nonpd_2host_c64_aime24_1783295840_rank1.tar.gz`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/rank1_extract/nonpd_2host_c64_aime24_1783295840/raw/bench_c64.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/rank1_extract/nonpd_2host_c64_aime24_1783295840/aime24_workdir/20260706_001210/reports/MiMo-V2-Flash/aime24.json`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/nonpd_2host_c64_parsed_summary.json`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c128_1783298516/nonpd_2host_c128_1783298516_rank0_serve.tar.gz`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c128_1783298516/nonpd_2host_c128_1783298516_rank1_serve.tar.gz`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c128_1783298516/rank1_extract/nonpd_2host_c128_1783298516/raw/bench_c128.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c128_1783298516/rank1_extract/nonpd_2host_c128_1783298516/bench_c128.jsonl`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c128_1783298516/nonpd_2host_c128_parsed_summary.json`
- 2P1D:
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/pd_2p1d_16k_4k_bench_1783270151/raw/bench_c64.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/pd_2p1d_16k_4k_bench_1783270151/raw/bench_c128.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/pd_2p1d_16k_4k_fixed_1783269610/raw/decode_server.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/rank0_prefill_logs/prefill_extra_poll_server.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/pd_2p1d_16k_4k_bench_1783270151/extra_prefill_2p1d_1783268093/raw/extra_prefill_server.log`

## 后续优化方向

1. 下一步目标应该是 transfer path，而不是 router admission。真正有用的 gap 在 forward-only `~22.8K input tok/s` 和 realized active prefill `~12.8K input tok/s` 之间。
2. Decode receiver 需要 metadata/prealloc 与 KV transfer wait 之间的真正 overlap。目前 `prealloc_wait + kv_wait` 几乎等于 total，因此仍是串行。
3. Prefill sender tail 仍然可见：C128 `transfer_tail` 约 `322 ms`，transfer span 约 `2.57 s`。减少 bootstrap/register polling 和 Raiden done-sending tail 应该能直接提升 prefill realized bandwidth。
4. 2P1D 值得为高并发 production-like load 保留，但当前最佳 cost/perf point 依赖 workload：C64 收益可以忽略，C128 有明确 TTFT 和吞吐收益；从 rate matching 看，继续加 P 之前应先测 `1P:2D`、`1P:3D` 或 decode MTP。
5. 下一步代码实验应关注更大粒度的 host/device scheduling overlap：让 transfer discovery/progress 独立于 decode event-loop tick，或在当前 decode forward in flight 时 pipeline 下一个 request 的 transfer setup。
6. 下一轮 benchmark 建议增加 open-loop request-rate sweep，并按明确 TTFT/ITL SLO 计算 goodput；这样才能和 DistServe/NIM 风格的公开结果做更公平的对齐。
