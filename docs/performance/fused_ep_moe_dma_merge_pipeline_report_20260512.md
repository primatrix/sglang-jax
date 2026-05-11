# Fused EP MoE Stage2/Stage4 DMA Merge 与 Pipeline 实验报告

日期: 2026-05-12

实验分支: `exp/fused-ep-moe-stage2-stage4-dma-merge`

对比路径: default path / main-equivalent path。为了做 stage 消融，实验在当前分支上运行，但所有 merge flags 默认关闭；默认路径与 main 的 fused MoE 执行路径等价，新增 flag 只用于实验。

## 结论

本轮尝试了两个 DMA launch reduction 方向:

1. Stage2 scatter: HBM per-expert pack，然后每个 expert 发一笔 DMA。
2. Stage4 gather: per-destination device merge，尝试减少 gather DMA launch。

实测结论:

1. Stage2 HBM per-expert pack **正确但更慢**。isolated scatter 退化 112%-377%，full kernel 退化 8%-39%。
2. Stage4 per-device gather merge **正确但明显更慢**。isolated gather 退化 458%-6070%，full kernel 退化 57%-314%。
3. 当前 main/default pipeline 中，Stage4 gather actual cost 存在，但大部分被 expert FFN / shared expert overlap 掉；优化 Stage4 的可见收益空间很小。
4. 这两个 merge 方向都不建议合入。后续如果继续做，需要先从 layout 层面减少 pack/demux 和 rectangular over-copy，而不是简单把 DMA launch 合并。

## 实现说明

### Stage2: HBM per-expert pack

用户指出的方向是对的: scatter 聚合不应该在 VMEM 开大 buffer。当前实验实现使用已有 `a2a_g_hbm` 作为临时 HBM staging:

```text
default scatter:
  tokens_hbm[token] -> remote a2a_s_hbm[expert, offset]    # token-expert entry 级 remote DMA

HBM per-expert pack:
  tokens_hbm[token] -> a2a_g_hbm[expert, local_offset]     # 本地 HBM pack，多笔本地 DMA
  a2a_g_hbm[expert, 0:n] -> remote a2a_s_hbm[expert, dst]  # 每 expert 一笔 remote DMA
```

这个方案不新增 VMEM 大 buffer，但会增加一次本地 HBM pack 读写，并且需要 drain pack DMA semaphore 后才能发 remote DMA。

### Stage4: per-destination gather merge

当前 gather default 路径是 eager:

```text
for local_expert:
  expert_ffn(local_expert)
  for recv_device:
    send expert output to recv_device
```

实验实现尝试把同一个 source device 上多个 local experts 发往同一个 destination device 的结果合成矩形 buffer:

```text
for recv_device:
  pack local experts' output
  send [local_experts, bt, hidden] rectangular block to recv_device
```

这个方案减少了显式 launch 维度，但代价很高:

1. 发送 rectangular block，包含大量 padding/unused token。
2. 破坏 eager gather 和后续 expert/shared expert 的 overlap。
3. 多维 DMA 在 Mosaic 层面仍需要按行 drain semaphore，实际同步成本不低。

## Correctness

在 EP8 pod 上做了小规模 correctness smoke test:

```text
scatter_expert_merge OK (256, 1024)
gather_device_merge OK (256, 1024)
```

测试方式: 直接调用 `fused_ep_moe`，分别打开 `enable_a2a_scatter_expert_merge=True` 和 `enable_a2a_gather_device_merge=True`，与 `ref_moe` 对比，`atol=2e-1, rtol=2e-1`。

## 原始日志

| 内容 | 路径 |
|---|---|
| DMA merge EP8 | `/tmp/sglang_dma_merge_ep8_20260512_040630` |
| DMA merge EP32 | `/tmp/sglang_dma_merge_ep32_20260512_040630` |
| DMA merge EP64 | `/tmp/sglang_dma_merge_ep64_20260512_040630` |
| pipeline EP8 | `/tmp/sglang_pipeline_main_ep8_20260512_041331` |
| pipeline EP32 | `/tmp/sglang_pipeline_main_ep32_20260512_041331` |
| pipeline EP64 | `/tmp/sglang_pipeline_main_ep64_20260512_041331` |

## Stage2 Scatter Merge 结果

单位: ms。变化列为 merge 相比 default 的变化，正数表示变慢。

| EP | tokens | scatter default | scatter merge | 变化 |
|---:|---:|---:|---:|---:|
| 8 | 512 | 0.108 | 0.515 | +376.9% |
| 8 | 8192 | 1.673 | 7.742 | +362.8% |
| 32 | 512 | 0.069 | 0.185 | +168.1% |
| 32 | 8192 | 0.818 | 2.587 | +216.3% |
| 64 | 512 | 0.049 | 0.104 | +112.2% |
| 64 | 8192 | 0.753 | 1.691 | +124.6% |

full kernel:

| EP | tokens | full default | full + scatter merge | 变化 |
|---:|---:|---:|---:|---:|
| 8 | 512 | 5.114 | 5.547 | +8.5% |
| 8 | 8192 | 35.964 | 42.392 | +17.9% |
| 32 | 512 | 0.383 | 0.531 | +38.6% |
| 32 | 8192 | 9.054 | 10.915 | +20.6% |
| 64 | 512 | 0.621 | 0.690 | +11.1% |
| 64 | 8192 | 3.673 | 4.720 | +28.5% |

结论: HBM pack 的额外本地 DMA 和 semaphore drain 成本明显大于减少 remote DMA launch 的收益。EP 越大，isolated 退化比例下降，但 full path 仍然没有正收益。

## Stage4 Gather Merge 结果

单位: ms。变化列为 merge 相比 default 的变化，正数表示变慢。

| EP | tokens | gather default | gather merge | 变化 |
|---:|---:|---:|---:|---:|
| 8 | 512 | 0.123 | 7.137 | +5702.4% |
| 8 | 8192 | 1.848 | 114.025 | +6070.2% |
| 32 | 512 | 0.089 | 0.928 | +942.7% |
| 32 | 8192 | 1.443 | 13.160 | +812.0% |
| 64 | 512 | 0.061 | 0.397 | +550.8% |
| 64 | 8192 | 0.785 | 4.382 | +458.2% |

full kernel:

| EP | tokens | full default | full + gather merge | 变化 |
|---:|---:|---:|---:|---:|
| 8 | 512 | 5.114 | 12.193 | +138.4% |
| 8 | 8192 | 35.964 | 149.046 | +314.4% |
| 32 | 512 | 0.383 | 1.271 | +231.9% |
| 32 | 8192 | 9.054 | 20.746 | +129.1% |
| 64 | 512 | 0.621 | 0.972 | +56.5% |
| 64 | 8192 | 3.673 | 7.497 | +104.1% |

结论: Stage4 merge 当前不成立。default eager gather 的实际优势不是 launch 数最少，而是能和 expert/shared expert pipeline overlap。merge 后既发送更多无效矩形数据，又把 overlap 变成串行尾部。

## Main/Default Pipeline 拆解

字段说明:

| 字段 | 含义 |
|---|---|
| S2 scatter | `stage2_scatter_only` |
| compute no gather/out | `full_no_gather_no_output - stage2_scatter_only` |
| S4 actual | `stage4_gather_only - stage4_control` |
| S4 hidden | `S4 actual - S4 visible` |
| S4 visible | `full_no_output - full_no_gather_no_output` |
| S6 output | `full - full_no_output` |

单位: ms。

| EP | tokens | S2 scatter | compute no gather/out | S4 actual | S4 hidden | S4 visible | S6 output | full |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 512 | 0.108 | 4.972 | 0.111 | 0.095 | 0.016 | 0.015 | 5.111 |
| 8 | 8192 | 1.672 | 33.395 | 1.698 | 1.089 | 0.609 | 0.291 | 35.967 |
| 32 | 512 | 0.069 | 0.301 | 0.068 | 0.059 | 0.009 | 0.005 | 0.384 |
| 32 | 8192 | 0.818 | 8.002 | 1.366 | 1.234 | 0.132 | 0.087 | 9.039 |
| 64 | 512 | 0.049 | 0.559 | 0.049 | 0.038 | 0.011 | 0.002 | 0.621 |
| 64 | 8192 | 0.753 | 2.810 | 0.742 | 0.667 | 0.075 | 0.036 | 3.674 |

图例:

```text
S = Stage2 scatter
C = Stage3 routed expert + Stage5 shared expert + non-gather compute
g = Stage4 visible tail
O = Stage6 output
S4_hidden_in_C = 被 C 段 overlap 掉的 gather
```

```text
EP8 tokens=512 total=5.111ms
|SCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCgO|
 S2=0.108 C=4.972 S4_visible=0.016 S6=0.015; S4_hidden_in_C=0.095

EP8 tokens=8192 total=35.967ms
|SSCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCgO|
 S2=1.672 C=33.395 S4_visible=0.609 S6=0.291; S4_hidden_in_C=1.089

EP32 tokens=512 total=0.384ms
|SSSSSSSSSCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCgO|
 S2=0.069 C=0.301 S4_visible=0.009 S6=0.005; S4_hidden_in_C=0.059

EP32 tokens=8192 total=9.039ms
|SSSSSCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCgO|
 S2=0.818 C=8.002 S4_visible=0.132 S6=0.087; S4_hidden_in_C=1.234

EP64 tokens=512 total=0.621ms
|SSSSCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCgO|
 S2=0.049 C=0.559 S4_visible=0.011 S6=0.002; S4_hidden_in_C=0.038

EP64 tokens=8192 total=3.674ms
|SSSSSSSSSSCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCgO|
 S2=0.753 C=2.810 S4_visible=0.075 S6=0.036; S4_hidden_in_C=0.667
```

## 对老板汇报建议

可以这样汇报:

```text
我们试了 Stage2 和 Stage4 的 DMA launch 合并。

Stage2 用 HBM staging 做 per-expert pack，不新增 VMEM 大 buffer，功能正确，但 pack 引入额外 HBM 读写和 semaphore drain，isolated scatter 与 full kernel 都退化。

Stage4 做 per-device gather merge 也功能正确，但它破坏了原本 eager gather 和 expert compute 的 overlap，并且矩形合并会发送大量无效数据，所以退化更明显。

从 pipeline 拆解看，main/default 路径里 Stage4 gather actual 存在，但大部分已经被 overlap 掉，EP32/64 下 visible gather 只有 0.075-0.132ms 量级。当前最优策略仍然是保留 eager gather，不合入这两个 merge 优化。
```

## 后续方向

如果还要继续做 DMA merge，必须同时满足两个条件:

1. pack layout 不引入大量额外 HBM 读写。
2. merge 后不破坏 expert compute/gather overlap。

可能方向:

1. 在 metadata 阶段直接生成更适合 DMA 合并的 layout，避免后续重新 pack。
2. 对 Stage2 做 routing-order 重排，让 token-expert entries 本身更连续，减少 pack 成本。
3. 对 Stage4 不做 rectangular per-device copy，而是改变 expert output layout，让同一 destination 的有效 token 原生连续。

这些都属于 layout 级改造，成本明显高于本轮的简单 DMA launch reduction 实验。
