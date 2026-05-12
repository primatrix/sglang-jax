# Fused EP MoE Stage4 Wait DMA 实验记录

日期: 2026-05-12

分支: `exp/fused-ep-moe-stage4-wait-dma`

## 背景

Stage4 gather 当前在每个 token block 末尾调用 `wait_a2a_gather_recv_all()`，等待本 device 需要的 gather 结果全部回到本地 HBM，然后再做 `acc_and_store_output()`。

原始实现的 wait 方式是扫描全部 global experts:

```python
for e_id in range(num_experts):
    sz = d2e_count[my_id, e_id]
    if sz != 0:
        wait_dma(a2a_g_hbm[e_id, 0:sz], a2a_gather_sem)
```

直觉上，小 token/decode 场景下实际 routed expert 数远小于 `num_experts`，所以可以尝试只扫描当前 tile 的 `bt * top_k` 条 route。

## 实验方案

| 方案 | 做法 | 结论 |
|---|---|---|
| 单个聚合 wait | 试图把多个 gather recv wait 合成一个 wait | 不安全，未保留 |
| route scan + `expert_offsets` seen marker | 只扫描 routed expert，用 `expert_offsets[..., 1, e_id] = -1` 去重，accumulate 前再修正 offset | correctness 通过，但性能无收益，LLO 变大，已 drop |
| route scan + `d2e_count` seen marker | 复用 `d2e_count[my_id, e_id] = 0` 去重，避免污染 output offset | 不安全，会影响后续 gather-send drain，已 drop |

## 为什么不能做单个聚合 wait

Pallas TPU DMA semaphore 不是按“DMA 次数”计数，而是按 copy size 计数，并且会 clip 到 `SEMAPHORE_MAX_VALUE=32767`。当前每个 `wait_dma(ref, sem)` 的 wait size 来自 `ref` 的元素数。

因此如果把多笔不同 expert 的 DMA 用一个 arbitrary ref 去 wait，很容易出现:

| 风险 | 结果 |
|---|---|
| wait size 小于实际 semaphore credit | semaphore 残留，污染下一轮 DMA |
| wait size 大于实际 semaphore credit | hang |
| 多个 copy size 被 32767 clip 后再求和 | 单个 ref 无法准确表达每笔 copy 的 clipped credit |

所以 Stage4 gather recv wait 不能简单地“按 copy 数量”或“按总 token 数量”合并。

## Correctness

route scan + `expert_offsets` marker 版本跑过以下 smoke，全部通过:

```bash
PYTHONPATH=$PWD/python /tmp/tpu_logs/venv/bin/python -u python/sgl_jax/test/kernels/fused_moe_v1_test.py \
  MoEKernelTest.test_basic0 \
  MoEKernelTest.test_basic1 \
  MoEKernelTest.test_shared_expert \
  MoEKernelTest.test_grouped_topk
```

route scan + `d2e_count` marker 版本在 EP64 / 128 tokens / `stage4_gather_only` 会 hang，已中断并 drop。

最终分支已 revert 两个 route-wait 实验提交，代码回到 `exp/fused-ep-moe-stage4-gather` 的行为；本地 `py_compile` 通过。

## Microbenchmark 结果

Shape: Ling2.6-1T, `num_experts=256`, `top_k=8`, `hidden_size=8192`, `intermediate_size=2048`, shared expert enabled。

对照方式:

| case | 含义 |
|---|---|
| `*_route_wait_off` | 禁用 route scan，等价原始 full-expert wait scan |
| 默认 case | 启用 route scan |

原始日志:

| EP | out dir |
|---|---|
| EP8 | `/tmp/sglang_stage4_wait_dma_ep8_20260512_110831` |
| EP32 | `/tmp/sglang_stage4_wait_dma_ep32_20260512_110831` |
| EP64 | `/tmp/sglang_stage4_wait_dma_ep64_20260512_110831` |

### Full Kernel

表中 `speedup` = `(off - on) / off`。正数表示 route scan 更快。

| EP | tokens | full off (ms) | full on (ms) | speedup |
|---:|---:|---:|---:|---:|
| 8 | 128 | 3.446 | 3.442 | +0.12% |
| 8 | 256 | 3.652 | 3.650 | +0.05% |
| 8 | 512 | 5.102 | 5.106 | -0.08% |
| 8 | 1024 | 6.547 | 6.551 | -0.06% |
| 8 | 8192 | 35.956 | 35.960 | -0.01% |
| 32 | 128 | 0.368 | 0.367 | +0.27% |
| 32 | 256 | 0.380 | 0.380 | +0.00% |
| 32 | 512 | 0.386 | 0.385 | +0.26% |
| 32 | 1024 | 0.474 | 0.473 | +0.21% |
| 32 | 8192 | 9.053 | 9.065 | -0.13% |
| 64 | 128 | 0.579 | 0.579 | +0.00% |
| 64 | 256 | 0.594 | 0.593 | +0.17% |
| 64 | 512 | 0.622 | 0.621 | +0.16% |
| 64 | 1024 | 0.696 | 0.696 | +0.00% |
| 64 | 8192 | 3.678 | 3.674 | +0.11% |

### Stage4 Gather Only

| EP | tokens | gather off (ms) | gather on (ms) | delta (ms) |
|---:|---:|---:|---:|---:|
| 8 | 128 | 0.043 | 0.043 | +0.000 |
| 8 | 256 | 0.067 | 0.067 | +0.000 |
| 8 | 512 | 0.123 | 0.123 | +0.000 |
| 8 | 1024 | 0.236 | 0.236 | +0.000 |
| 8 | 8192 | 1.848 | 1.848 | +0.000 |
| 32 | 128 | 0.046 | 0.046 | +0.000 |
| 32 | 256 | 0.068 | 0.069 | +0.001 |
| 32 | 512 | 0.089 | 0.089 | +0.000 |
| 32 | 1024 | 0.178 | 0.177 | -0.001 |
| 32 | 8192 | 1.444 | 1.443 | -0.001 |
| 64 | 128 | 0.026 | 0.026 | +0.000 |
| 64 | 256 | 0.037 | 0.037 | +0.000 |
| 64 | 512 | 0.061 | 0.061 | +0.000 |
| 64 | 1024 | 0.107 | 0.107 | +0.000 |
| 64 | 8192 | 0.785 | 0.786 | +0.001 |

## LLO 分析

选择最应该受益的 case 做 LLO dump:

| 条件 | 值 |
|---|---|
| EP | 64 |
| tokens | 128 |
| block config | `bt=2`, `top_k=8` |
| 原始 wait scan | 扫 256 个 expert |
| route scan | 扫 16 条 route |

LLO 文件:

| 版本 | 本地文件 |
|---|---|
| route off | `/tmp/stage4_wait_dma_llo_files/off/*fused-moe*` |
| route on | `/tmp/stage4_wait_dma_llo_files/on/*fused-moe*` |

关键计数:

| 指标 | route off | route on |
|---|---:|---:|
| `02-original.txt` 行数 | 7495 | 7741 |
| `68-final_bundles.txt` 行数 | 3137 | 3264 |
| final bundle 中 `sld` 计数 | 279 | 302 |
| final bundle 中 `sst` 计数 | 69 | 77 |
| final bundle 中 `sadd.s32` 计数 | 538 | 575 |

解释:

1. 原始 full-expert wait scan 虽然循环次数是 256，但 LLO 里是一个 compact dynamic loop，body 很小。
2. route scan 把 `top_k=8` 静态展开，并增加 seen marker 的 SMEM load/store 与条件控制。
3. 对 EP64 / 128 tokens 这种最小 case，route scan 仍然没有减少 final bundles，反而增加 scalar 指令。
4. 放到 full kernel 后，这部分 wait scan 本来就是 gather 尾部的一小段控制开销，且大部分 Stage4 成本来自 DMA/同步/overlap drain，route scan 的收益被完全淹没。

## 最终结论

Stage4 wait DMA 这条优化目前不值得保留。

原因可以概括为:

1. DMA semaphore 的计数语义决定了不能安全地把多笔 gather recv wait 粗暴合成一个 wait。
2. route scan 虽然减少了运行时扫描的 expert 数，但需要额外去重状态；LLO 层面没有变小，实测 full kernel 也只有噪声级变化。
3. 尝试复用 `d2e_count` 做去重会破坏后续 `wait_a2a_gather_send()` 对 local/remote size 的计算，存在 hang 风险。

因此当前分支已 drop 相关代码优化，只保留本实验记录。后续如果继续 Stage4，应该优先做真正改变 DMA 粒度或 layout 的方案，而不是只改 wait scan。
