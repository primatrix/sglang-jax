# MiMo-V2-Flash Non-PD 16K/4K 基线, 2026-07-05

## 摘要

最初的 2026-07-05 run 使用 1 个 v7x-8 host 作为普通 non-PD server。
这可以作为单机 baseline，但不能和 PD 1P1D 做 pod-count-fair 比较。

2026-07-06 的补充测试增加了一个更公平的 C64 对比：两个普通 non-PD server，
分别运行在一个 Falcon rank 上，并通过一个轻量 streaming round-robin proxy
转发请求。这个拓扑等价于在 serve 层做 data parallelism，而不是做 P/D 分离。

主要结果：

- 单机 non-PD 最好 client throughput 是 C128：`8.62K total tok/s`，其中
  `6.89K input tok/s`、`1.72K output tok/s`。
- PD 1P1D C128 是 `13.64K total tok/s`，其中 `10.91K input tok/s`、
  `2.73K output tok/s`。
- 单机 non-PD C128 的 decode serve-log highwater 很强：均值
  `3.87K output tok/s`，max `4.10K`。端到端损失不是 decode kernel 弱，
  而是同设备 prefill/decode contention 和长 prefill queueing。
- 单机 non-PD C128 有 `383/384` 个 request 成功。失败 request 已保留在原始
  benchmark log 中，因此这个点略有噪声，但不改变整体结论。
- two-host non-PD C64 follow-up 达到 `11.70K total tok/s`、`9.36K input tok/s`
  和 `2.34K output tok/s`。它高于 PD 1P1D C64（`10.55K total tok/s`）和
  PD 2P1D C64（`10.83K total tok/s`）。
- two-host non-PD C128 follow-up 达到 `12.78K total tok/s`、`10.22K input tok/s`
  和 `2.56K output tok/s`，成功数 `383/384`。它低于 PD 1P1D C128
  （`13.64K total tok/s`）和 PD 2P1D C128（`15.31K total tok/s`）。
- 因此公平结论更细：serve-level DP 在 C64 更好，而 PD 在 C128 / high pressure
  下重新获得优势。

## 测试代码和环境

- Falcon exp: `exp-5uqgg64144`, rank 1。
- 远端 repo: `/tmp/sglang-jax`。
- 远端 run dir: `/tmp/e2e_logs/nonpd_16k_4k_1783265639`。
- 本地原始 archive: `tmp/e2e_logs/nonpd_16k_4k_1783265639/nonpd_16k_4k_1783265639.tar.gz`。
- 解析 summary: `tmp/e2e_logs/nonpd_16k_4k_1783265639/parsed_summary.json`。
- Model: `/models/MiMo-V2-Flash`。
- JAX compilation cache: `/tmp/tpu_logs/jit_cache`。

## Serve 命令

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
  --enable-metrics --enable-request-time-stats-logging \
  --host 0.0.0.0 --port 30000
```

## Benchmark 命令

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
    --output-file "/tmp/e2e_logs/nonpd_16k_4k_1783265639/bench_c${C}.jsonl"
done
```

## Client 结果

| C | success | req/s | input tok/s | output tok/s | peak output tok/s | total tok/s | mean TTFT ms | p99 TTFT ms | mean ITL ms | p99 ITL ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 96/96 | 0.29 | 4737 | 1184 | 1952 | 5922 | 22086 | 42060 | 21.63 | 18.15 |
| 64 | 192/192 | 0.35 | 5685 | 1421 | 2688 | 7106 | 42904 | 83541 | 34.55 | 41.88 |
| 128 | 383/384 | 0.42 | 6894 | 1723 | 4504 | 8617 | 83256 | 162879 | 53.54 | 317.23 |

## Serve Log 稳态结果

| C | prefill active window UTC | prefill span s | prefill active input tok/s | prefill max queue | decode highwater window UTC | decode highwater mean tok/s | decode highwater max tok/s |
|---:|---|---:|---:|---:|---|---:|---:|
| 32 | 15:39:16-15:43:39 | 263 | 5980 | 30 | 15:39:59-15:44:48 | 1880 | 1942 |
| 64 | 15:45:26-15:52:58 | 452 | 6960 | 62 | 15:46:51-15:54:39 | 2576 | 2687 |
| 128 | 15:55:26-16:08:17 | 771 | 8160 | 124 | 15:58:14-16:10:31 | 3872 | 4103 |

## PD vs Non-PD

| 模式 | host 数 | C | client total tok/s | client input tok/s | client output tok/s | serve prefill active input tok/s | serve decode highwater output tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| non-PD single server | 1 | 32 | 5922 | 4737 | 1184 | 5980 | 1880 |
| PD 1P1D | 2 | 32 | 7838 | 6270 | 1568 | 8150 | 1852 |
| non-PD single server | 1 | 64 | 7106 | 5685 | 1421 | 6960 | 2576 |
| non-PD serve-level DP | 2 | 64 | 11700 | 9360 | 2340 | 11893 | 3761 |
| PD 1P1D | 2 | 64 | 10546 | 8437 | 2109 | 10556 | 2462 |
| PD 2P1D | 3 | 64 | 10829 | 8663 | 2166 | n/a | n/a |
| non-PD single server | 1 | 128 | 8617 | 6894 | 1723 | 8160 | 3872 |
| non-PD serve-level DP | 2 | 128 | 12779 | 10223 | 2556 | 11642 | 4672 |
| PD 1P1D | 2 | 128 | 13642 | 10913 | 2728 | 12788 | 3180 |
| PD 2P1D | 3 | 128 | 15307 | 12246 | 3061 | n/a | n/a |

解读：

- 原始 single-host non-PD baseline 不能和 PD 做 pod-count-fair 比较。
- C64 two-host 场景下，serve-level DP 的 client total tok/s 比 PD 1P1D 高约
  `10.9%`（`11.70K / 10.55K`），比 PD 2P1D C64 高约 `8.0%`
  （`11.70K / 10.83K`），同时 mean TTFT 也更低（`13.7s` vs PD 1P1D `16.5s`）。
- C128 two-host 场景下，结论反转：PD 1P1D 的 client total tok/s 比 two-host
  non-PD 高约 `6.8%`（`13.64K / 12.78K`）；PD 2P1D 高约 `19.8%`
  （`15.31K / 12.78K`），但 PD 2P1D 多使用了一个 prefill host。
- 这符合预期：C64 时两个完整 non-PD replica 分摊 burst，并且 KV local，
  避免了 PD transfer overhead；C128 时每个 replica 接近 C64 压力，同设备
  prefill/decode contention 和更长 tail latency 又出现了。
- 因此 PD 的 role-isolation 价值主要出现在更高压力区间，而不是中等 C64。

## Two-Host Serve-Level DP 补充测试

Run ids：

```text
C64 + AIME24: nonpd_2host_c64_aime24_1783295840
C128:         nonpd_2host_c128_1783298516
```

拓扑：

- Falcon exp: `exp-5uqgg64144`。
- Rank 0：普通 non-PD server，`http://rank0:30010`。
- Rank 1：普通 non-PD server，`http://localhost:30010`。
- Rank 1 proxy：`http://localhost:30000`，轻量 streaming round-robin proxy。

proxy 除了 round-robin request distribution 外，没有做 admission 或调度优化。
C64 benchmark 后它报告总共转发 `196` 个 request，每个 backend `98` 个。这包含
192 个 benchmark request 以及 metadata/health request。

两个 rank 的 server 命令：

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
  --enable-metrics --enable-request-time-stats-logging \
  --host 0.0.0.0 --port 30010
```

Benchmark 命令：

```bash
/usr/local/bin/python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --base-url http://127.0.0.1:30000 \
  --model /models/MiMo-V2-Flash \
  --tokenizer /models/MiMo-V2-Flash \
  --dataset-name random \
  --random-input-len 16384 \
  --random-output-len 4096 \
  --random-range-ratio 1.0 \
  --num-prompts 192 \
  --request-rate inf \
  --max-concurrency 64 \
  --warmup-requests 0 \
  --seed 12345 \
  --output-details \
  --extra-request-body '{"sampling_params":{"temperature":0.1,"top_p":0.95,"max_new_tokens":4096,"ignore_eos":true}}' \
  --output-file /tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/bench_c64.jsonl
```

Client 结果：

| C | success | duration s | req/s | input tok/s | output tok/s | peak output tok/s | total tok/s | mean TTFT ms | p99 TTFT ms | mean ITL ms | p99 ITL ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 192/192 | 336.08 | 0.57 | 9360 | 2340 | 3884 | 11700 | 13675 | 40857 | 23.89 | 171.86 |
| 128 | 383/384 | 613.79 | 0.62 | 10223 | 2556 | 5017 | 12779 | 46493 | 170968 | 34.56 | 278.65 |

C128 失败 request 保存在 `bench_c128.jsonl`：

```text
idx=354 Bad Gateway: ServerDisconnectedError('Server disconnected')
backend=http://falcon-job-p6bmn75fnu-0.falcon-job-p6bmn75fnu.falcon-jobs.svc.cluster.local:30010
path=/generate
```

Serve-log summary:

| C | Rank | prefill window UTC | prefill span s | prefill input tok/s | prefill max queue | decode high-load mean tok/s | decode max tok/s | max running |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | 0 | 00:06:29-00:10:53 | 264 | 5958 | 30 | 1879 | 1970 | 40 |
| 64 | 1 | 00:06:29-00:10:54 | 265 | 5935 | 30 | 1882 | 1963 | 48 |
| 64 | combined | n/a | n/a | 11893 | n/a | 3761 | 3933 | n/a |
| 128 | 0 | 00:42:56-00:51:55 | 539 | 5806 | 48 | 2279 | 2492 | 57 |
| 128 | 1 | 00:42:56-00:51:55 | 539 | 5836 | 48 | 2393 | 2868 | 75 |
| 128 | combined | n/a | n/a | 11642 | n/a | 4672 | 5361 | n/a |

## AIME24 补充测试

同一个 two-host non-PD serve-level DP endpoint 用来重跑 AIME24：

```bash
/usr/local/bin/python -m evalscope.run \
  --model /models/MiMo-V2-Flash \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets aime24 \
  --eval-batch-size 16 \
  --timeout 6000000 \
  --work-dir /tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/aime24_workdir \
  --generation-config '{"temperature":1,"top_p":0.95,"max_tokens":30000,"chat_template_kwargs":{"enable_thinking":true}}'
```

结果：

| Endpoint | Dataset | Num | Score | Correct |
|---|---|---:|---:|---:|
| non-PD serve-level DP | AIME24 / `HuggingFaceH4/aime_2024` | 30 | 0.8667 | 26 |

更早的 PD run 在相同 non-greedy generation 形态下得到 `0.7667`（23/30）。
由于 config 使用 `temperature=1`，除非增加 deterministic accuracy protocol，
否则这个差异应视为采样波动。两个结果都在合理范围内，没有精度异常信号。

## 原始日志

- Client 日志: `raw/bench_c32.log`, `raw/bench_c64.log`, `raw/bench_c128.log`。
- Server 日志: `raw/nonpd_server.log`。
- JSONL 详情: `bench_c32.jsonl`, `bench_c64.jsonl`, `bench_c128.jsonl`。
- Window markers: `c32.window`, `c64.window`, `c128.window`。
- Two-host C64/AIME24 本地产物：
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/nonpd_2host_c64_aime24_1783295840_rank0.tar.gz`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/nonpd_2host_c64_aime24_1783295840_rank1.tar.gz`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/rank1_extract/nonpd_2host_c64_aime24_1783295840/raw/bench_c64.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/rank1_extract/nonpd_2host_c64_aime24_1783295840/aime24_workdir/20260706_001210/reports/MiMo-V2-Flash/aime24.json`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c64_aime24_1783295840/nonpd_2host_c64_parsed_summary.json`
- Two-host C128 本地产物：
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c128_1783298516/nonpd_2host_c128_1783298516_rank0_serve.tar.gz`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c128_1783298516/nonpd_2host_c128_1783298516_rank1_serve.tar.gz`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c128_1783298516/rank1_extract/nonpd_2host_c128_1783298516/raw/bench_c128.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c128_1783298516/rank1_extract/nonpd_2host_c128_1783298516/bench_c128.jsonl`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c128_1783298516/rank0_extract/nonpd_2host_c128_1783298516/raw/nonpd_server_rank0_c128_slice.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c128_1783298516/rank1_extract/nonpd_2host_c128_1783298516/raw/nonpd_server_rank1_c128_slice.log`
  - `/Users/jiongxuan/workspace/sglang-jax/tmp/e2e_logs/nonpd_2host_c128_1783298516/nonpd_2host_c128_parsed_summary.json`
