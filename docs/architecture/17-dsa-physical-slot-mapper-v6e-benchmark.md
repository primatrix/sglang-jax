# GLM-5.2 DSA physical slot mapper：v6e-1 实验结论

## 结论

不建议实现原计划中的 Phase 1 代数清理或 Phase 2 显式跨层
`DSASparseSelection` 缓存：前者在 TPU 上没有收益，后者已被当前 XLA
可执行文件等价地完成。当前真正的瓶颈是 `[Q, K]` page-table 间接 gather，
后续优化需要改变该 gather 的执行位置或硬件路径。

推荐的下一步是原型化以下二选一方案，并继续保持现有 XLA 实现作为
fallback：

1. 在 radix top-k 的 SparseCore epilogue 中，仅对最终 K 个 logical index
   做 page-table lookup，直接返回 physical slot；
2. 在 `dsa_sparse_core_gather` 的入口把 logical index 转成 physical slot，
   让映射和后续 KV gather 使用同一条 SparseCore indirect pipeline。

不要把完整 physical values `[Q, score_size]` 传给 radix top-k。生产 shape
下仅这个 int32 张量就有 1.03125 GiB，会引入远大于 page table 的额外输入流量。

## 原 profile 的静态与动态证据

分析对象：

```text
exp-tx6d0cbjkz/profiling/
  glm52-fused-v2-radix-dp16-c32-128k-one-chip-stage/prefill
```

在 TPU:0 的 trace 窗口中，映射对应的事件是：

```text
%gather_fusion.19 = s32[4194304] fusion(
    s32[4224] page_indices,
    s32[4194304] page_ptr), kind=kCustom
duration = 19.068151 ms
```

`4194304 = 2048 * 2048`，与 extend shard 的 `[Q, K]` 完全一致。

`jit_jitted_run_model(1)` 的 optimized HLO 中共有 19 个
`s32[2048,2048] <- gather(s32[4224], ...)`。GLM-5.2 的 78 层布局恰好也有
19 个进入 sparse attention 的 IndexShare group：第 2 层 full indexer 服务
第 3--5 层，之后第 6、10、...、74 层分别开启一个 `full + 3 shared`
分组。如果没有编译器消重，source-level mapper 会出现 75 次；optimized HLO
只有 19 次，说明 shared layers 的相同映射已经被 XLA 合并为每组一次。

因此，显式把 `physical_slots` 从 full layer 传到 shared layers 不会再把
mapper 数量从 4 降到 1；它只会把编译器已完成的复用暴露为模型状态。

## v6e-1 kernel-level A/B

### 环境

- source base：`epic/glm_5_2@c3409a8a7`
- benchmark commit：`665d38356b00b89d57056a3b7e6fc3ce7a1ab1d4`
- GKE：`tpu-service-473302/us-east5-a/perf-v6e-16`
- node pool：`ct6e-standard-1t`，topology `1x1`
- JAX / jaxlib：`0.10.2`
- libtpu：`0.0.42.1`
- shape：`Q=2048, K=2048, page_size=64, page_table=4224`
- ragged sequence lengths：`131584, 132096`
- DVFS p-state：7
- 10 次 warmup，50 次计时

### 结果

| Variant | p50 (ms) | mean (ms) | p90 (ms) | 相对 baseline |
|---|---:|---:|---:|---:|
| 当前 XLA baseline | 25.405520 | 25.413629 | 25.452813 | 1.0000x |
| Phase 1 代数清理 | 25.444855 | 25.446070 | 25.466659 | 0.9985x |
| 预计算 metadata 上限 | 25.422666 | 25.421669 | 25.446052 | 0.9993x |

两种 XLA candidate 都和 baseline 位级一致；0.07%--0.15% 的差异位于噪声
范围内。即使把 `searchsorted`、`seq_lens[seq_ids]` 和 page start 的构造完全
排除在计时之外，耗时也没有下降。这证明 mapper 的主要成本不在 metadata
和整数代数，而在 419 万个间接 lookup 及输出物化。

### TensorCore Pallas candidate

Pallas candidate 在 CPU interpret mode 通过 ragged、invalid top-k、padding
query 和负 page index 的正确性测试；在真实 TPU Mosaic lowering 中，合法的
`BQ=128` 和 `BQ=256` 都失败：

```text
ValueError: Cannot do int indexing on TPU
```

因此，把 4224 项 page table 放进 TensorCore VMEM 再执行向量随机索引，不是
当前 Mosaic Pallas 支持的实现路径。这个 candidate 只应作为负实验保留，
不应接入 production backend。

## 对原优化计划的修订

- Phase 0：保留，尤其是 mapper 的独立 scope、正确性契约与 profile。
- Phase 1：不合入；benchmark 未显示收益。
- Phase 2：不合入；optimized HLO 已经按 IndexShare group 消重。
- Phase 3：不单独做；预计算 metadata 的上限实验没有收益。
- 新 Phase 4：把最终 K 个 page lookup 移到 SparseCore top-k epilogue 或
  SparseCore attention gather 边界，并用完整 indexer/attention A/B 验证。

原始 JSONL 数据见
`benchmark/kernels/dsa/results/physical_slots_v6e_1_665d3835.jsonl`。
