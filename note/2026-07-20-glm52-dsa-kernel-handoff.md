# GLM-5.2 DSA 算子输入交接

> 状态：ABI、CPU PyTorch golden 生成器和性能 shape manifest 已冻结；本文只交接算子正确性与 microbenchmark 输入，不把服务 E2E 性能或 dense attention 等价作为算子验收条件。

## 1. 交付结论

算子同事需要消费两组彼此独立、但共享同一 ABI 的输入：

1. **Correctness bundle**：CPU PyTorch 独立实现产生 `.npz + manifest.json`，覆盖 Indexer selection、logical-to-physical mapping 和 final sparse MLA。候选算子只与该 golden 比较。
2. **Performance matrix**：只在 TPU 上运行候选 sparse MLA；CPU reference 不进入计时。场景同时记录 physical bucket 和 active row，避免把服务中的 `[64, ...]` padding bucket 错测成理想化的 `[1, ...]`。

GLM-5.2 的生产参数来自模型 config 和 TP32 实际部署：

| 参数 | 值 |
| --- | ---: |
| total Q heads | 64 |
| TP | 32 |
| local Q heads | 2 |
| absorbed latent dim | 512 |
| RoPE dim | 64 |
| cache width | `align128(512) + align128(64) = 640` |
| page size / packing | 128 / 2 |
| Indexer heads / head dim | 32 / 128 |
| `K_max` | 2048 |
| attention scale | `256^-0.5` |
| input/cache/output dtype | BF16 |
| score、softmax、reference accumulation | FP32 |

历史 microbenchmark 的 `--num-heads 8` 不是当前 GLM-5.2 TP32 主 shape；交接的默认值已改为 `H_local=2`。

## 2. Final sparse MLA ABI

算子入口只负责读取已经写好的主 MLA cache，并对上游提供的物理 slots 做 sparse attention：

```text
q_latent        [Q_bucket, 2, 512]  BF16
q_rope          [Q_bucket, 2, 64]   BF16
cache           [pages, 64, 2, 640] BF16
physical_slots  [Q_bucket, 2048]    INT32
selected_counts [Q_bucket]          INT32
sm_scale        scalar              FP32, 256^-0.5
output          [Q_bucket, 2, 512]  BF16
```

物理地址语义：

```text
page = slot // 128
offset = slot % 128
row = offset // 2
lane = offset % 2

c_kv = cache[page, row, lane, :512]
k_pe = cache[page, row, lane, 512:576]
```

必须遵守：

- 只有 `physical_slots[q, :selected_counts[q]]` 有语义；padding 内容当前填 0，但算子不能读取 counted prefix 之外的 slot。
- `slot=0` 是合法地址，不能充当 padding sentinel。
- `selected_counts=0` 的 physical padding row 输出全 0。
- slot 是 token-granular 物理 pool 地址，不是 request 内 logical position，也不是 page ID。
- `pages` 由部署时 token pool 容量决定。standalone benchmark 的 `context_length` 表示每个 request 的 visible/address region；`cache_capacity` 默认取当前 request layout 所需的最小值，也可独立设为服务实际 global token-pool capacity。
- final sparse MLA 不包含 Indexer、Top-K、logical-to-physical mapping 或 cache write。这些 stage 的 CPU golden 一并提供，是为了验证完整输入链，不表示必须合并进同一个 Pallas kernel。

数学定义：

```text
score[q,h,j] = 256^-0.5 * (
    dot(q_latent[q,h], c_kv[physical_slots[q,j]])
  + dot(q_rope[q,h],   k_pe[physical_slots[q,j]])
)

output[q,h] = softmax(score[q,h,:selected_counts[q]]) @ selected_c_kv
```

## 3. CPU PyTorch golden

独立 reference 位于：

- `python/sgl_jax/srt/kernels/dsa/torch_reference.py`
  - `torch_glm_dsa_select`
  - `torch_logical_topk_to_physical_slots`
  - `torch_dsa_sparse_mla`
- `benchmark/kernels/mla/export_glm52_dsa_golden.py`
  - 调用以上三个 reference，生成固定输入、预期输出、shape/dtype 描述和 SHA-256。

生成默认 bundle：

```bash
PYTHONPATH=python:. python \
  benchmark/kernels/mla/export_glm52_dsa_golden.py \
  --output-dir /tmp/glm52-dsa-golden
```

本轮以默认 seed 连续生成两次并逐文件 `cmp`，结果 byte-identical；bundle 约 `2.9 MiB`，
`manifest.json` SHA-256 为
`6f0ab647ea2c1f64c18a6e8fd09b24fb08c086fed8d84f09f696a67682fdb3bc`。

默认 Indexer candidate length 覆盖：

```text
1 / 127 / 128 / 129 / 257 / 2047 / 2048 / 2049 / 3072 / 4096
```

这组长度同时覆盖 128 对齐边界和 `K_max=2048` 截断边界。fixture 使用 BF16 量化后的输入、FP32 计算，并主动保证 Top-K score 无 tie；因此 ID 顺序具有确定语义。

boundary fixture 使用结构化 binary score 来保证 `>=1e-3` 的确定性排序间隔；bundle
另外包含 `indexer-selection-realistic-c257`，其两个 query 会使用全部 32 个 Indexer
head、全部 128 个维度、signed head weights 和正负两侧 logits。前者专门钉住 Top-K
边界，后者防止“只实现一个 head / 少量维度 / 忽略 ReLU 或 weighted reduction”的错误
实现误过 gate。

`.npz` 的 BF16 tensor 采用“先由 PyTorch round-trip 到 BF16，再以 FP32 数值存储”的可移植策略。加载端应按 `manifest.json` 的 `semantic_dtype=bfloat16` cast 回 BF16；这不会重新引入量化差异。每个 `.npz` 都有 manifest SHA-256，文件成员顺序和 ZIP timestamp 固定，同 seed 可复现。

golden 刻意不保存 32k/160k 大 cache。正确性 fixture 使用生产 head/dim/page/K 但较小 cache；长地址 footprint 只在 TPU performance matrix 中生成。

## 4. 精度 gate

| Stage | 比较对象 | Gate |
| --- | --- | --- |
| Indexer scores | candidate vs CPU PyTorch FP32 | no-tie fixture，`rtol=1e-6, atol=1e-6` |
| Indexer selection | logical IDs、selected counts | 精确一致 |
| slot mapping | physical slots、selected counts | counted prefix 精确一致 |
| sparse MLA debug/reference | JAX/其他 FP32 reference vs CPU PyTorch | `rtol=2e-5, atol=2e-5` |
| production BF16 Pallas | BF16 output cast FP32 vs CPU PyTorch FP32 | `rtol=2e-2, atol=1e-2` |

额外语义 gate：

- `visible_count <= 2048` 时，sparse 输出应与“同一 visible set 上的 dense attention”一致。
- `visible_count > 2048` 时，full dense attention **不是 golden**。正确链路是 CPU Indexer 选出 Top-K，CPU mapping 得到 physical slots，再由 CPU sparse MLA 计算预期输出。
- padding slot 的任意内容不得影响输出；重复、future、越界 logical ID 和缺失 physical slot 必须在 mapping stage 被清理。
- 生产 Pallas 当前实测最坏 `max_abs=0.0019080639`，在上述 BF16 gate 内；gate 不要求 bitwise 一致，因为归约和 softmax 顺序允许不同。

## 5. 性能 shape matrix

性能测试只测 precompiled candidate kernel，不运行 CPU reference，也不运行 Indexer/mapping。decode 的多请求场景默认使用彼此不相交的 request region，避免所有 row 重读同一组 KV 而产生虚假的 cache reuse。主场景如下：

| Case | Mode | physical / active rows | visible / request | minimum cache capacity | counted slots | 目的 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `decode-bucket-a1-c512` | decode | 64 / 1 | 512 | 512 | `[512, 0 x 63]` | 小 context / selected-count 下界 |
| `decode-bucket-a1-c1024` | decode | 64 / 1 | 1024 | 1024 | `[1024, 0 x 63]` | selected-count scaling |
| `decode-bucket-a1-c2048` | decode | 64 / 1 | 2048 | 2048 | `[2048, 0 x 63]` | Top-K 饱和边界 |
| `decode-bucket-a1-c4096` | decode | 64 / 1 | 4096 | 4096 | `[2048, 0 x 63]` | 当前服务低并发 ITL 主 shape |
| `decode-bucket-a8-c4096` | decode | 64 / 8 | 4096 | 32768 | `[2048 x 8, 0 x 56]` | active-row scaling |
| `decode-bucket-a32-c4096` | decode | 64 / 32 | 4096 | 131072 | `[2048 x 32, 0 x 32]` | active-row scaling |
| `decode-bucket-a64-c4096` | decode | 64 / 64 | 4096 | 262144 | all 2048 | 满 bucket throughput / HBM 压力 |
| `decode-long-a1-c160k` | decode | 1 / 1 | 160000 | 160000 | 2048 | 单请求长地址 footprint |
| `decode-throughput-a8-c32k` | decode | 8 / 8 | 32000 | 256000 | all 2048 | 中等并发长上下文 |
| `prefill-t128-start0` | prefill | 128 / 128 | 128 shared | 128 | `1..128` | 首个 chunk，未到 Top-K 截断 |
| `prefill-t128-start2048` | prefill | 128 / 128 | 2176 shared | 2176 | all 2048 | 截断后的 steady-state chunk |

这里 `physical rows` 是编译 shape，`active rows` 由 `selected_counts>0` 决定。特别是当前 TP32/fused MoE 服务的 decode 预编译 bucket 是 64，即使请求只有 1 条，算子仍应以 `Q_bucket=64, active=1` 作为主 latency 场景；单独的 B1 只用于隔离 kernel 本体。

cache/address footprint 与算术量不是一回事：当 counted slots 固定为 2048 时，32k/160k 主要改变 slot 在 cache 中的跨度和 locality，不改变每个 active query 的理论 selected KV 数。每个 active query 的 BF16 selected-cache payload 约为：

```text
2048 * 640 * 2 bytes = 2.5 MiB
```

unsorted slots 是主结果；`page-sorted` 只作为 locality 上界和优化诊断，不能替代 unsorted correctness/performance。

`--request-layout disjoint` 会为每个 active decode row 分配独立的
`context_length` region；`--cache-capacity` 可在这个 minimum 之上显式设置为服务日志中的
真实 global token-pool capacity。prefill 的 128 行属于同一个 request chunk，因此使用
`--request-layout shared`。

### 运行命令

公共参数：

```bash
COMMON="--top-k 2048 --num-heads 2 --latent-dim 512 --rope-dim 64 \
--page-size 128 --slot-order unsorted --variant sparse \
--warmup-iters 50 --iters 200"
```

Decode：

```bash
for C in 512 1024 2048 4096; do
  PYTHONPATH=python:. python benchmark/kernels/mla/bench_dsa_decode_mla.py \
    --batch-size 64 --active-batch-size 1 --context-length "$C" \
    --request-layout disjoint --valid-count-pattern full $COMMON
done

for A in 8 32 64; do
  PYTHONPATH=python:. python benchmark/kernels/mla/bench_dsa_decode_mla.py \
    --batch-size 64 --active-batch-size "$A" --context-length 4096 \
    --request-layout disjoint --valid-count-pattern full $COMMON
done

PYTHONPATH=python:. python benchmark/kernels/mla/bench_dsa_decode_mla.py \
  --batch-size 1 --active-batch-size 1 --context-length 160000 \
  --request-layout disjoint --valid-count-pattern full $COMMON

PYTHONPATH=python:. python benchmark/kernels/mla/bench_dsa_decode_mla.py \
  --batch-size 8 --active-batch-size 8 --context-length 32000 \
  --request-layout disjoint --valid-count-pattern full $COMMON
```

Prefill：

```bash
PYTHONPATH=python:. python benchmark/kernels/mla/bench_dsa_decode_mla.py \
  --batch-size 128 --active-batch-size 128 --context-length 128 \
  --request-layout shared --valid-count-pattern causal --start-position 0 $COMMON

PYTHONPATH=python:. python benchmark/kernels/mla/bench_dsa_decode_mla.py \
  --batch-size 128 --active-batch-size 128 --context-length 2176 \
  --request-layout shared --valid-count-pattern causal --start-position 2048 $COMMON
```

文件名保留历史 `decode_mla`，但输入 ABI 对 decode/prefill 相同；prefill case 由 Q rows 和 per-row causal `selected_counts` 表达。

## 6. 性能报告格式

每个 case 至少记录：

- TPU SKU、JAX/libtpu 版本、设备数和 mesh；
- 完整 input shape、semantic dtype、slot order、active rows 和 selected-count 分布；
- first compile + execute 显式同步并作为 `compile_ms` 单独记录，不计入 latency；即使 `warmup_iters=0` 也不会污染 timed samples；
- 至少 20 次同步 warmup、至少 100 次同步计时；
- median、p95、p99、min 和 mean；
- 可选 XProf trace；
- 同一 binary/commit 的 correctness bundle SHA-256 和精度 gate 结果。

当前 benchmark 默认 50 次 warmup、200 次 timed call，并在每次调用后 `block_until_ready`；JSON 直接输出 `compile_ms / median / p95 / p99 / mean / min`。比较候选实现时必须保持 shape、seed、slot order、request layout、warmup 和计时次数一致。

## 7. Host 验证

```bash
PYTHONPATH=python:. python -m pytest \
  python/sgl_jax/test/test_dsa_cross_framework.py \
  python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py \
  python/sgl_jax/test/kernels/test_glm52_dsa_handoff.py -q
```

这一步只验证 CPU/JAX reference、fixture schema 和 golden exporter；不触发 Falcon E2E，也不重新评估已解释的 decode/prefill 服务时间差。
