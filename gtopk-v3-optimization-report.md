# Grouped-Top-K v3 Kernel 优化报告(token-in-lane)

**日期**: 2026-07-07
**硬件**: TPU v7x (Ironwood), 8 chips, `tpuv7x-64-node`(Falcon),算子微基准单芯片
**软件**: jax 0.9.0 / libtpu 0.0.34(镜像),counter 采样时 pin libtpu 0.0.38;jax 0.10.2 亦验证可跑
**分支**: `bench/grouped-topk-v2`(benchmark-only,未接入 `gate.py`)
**配置**: E=256, n_group=8, topk_group=4, topk=8(DeepSeek-V3 / Ling 风格路由)

---

## 结论摘要(TL;DR)

把 grouped-topk Pallas kernel 从 v1 的 `[BT, E]`(token 在 sublane、专家在 lane)重写为 **token-in-lane `[E, BT]`**(专家在 sublane、token 在 128 宽 lane),再叠加若干针对性优化,在 v7x 上相对 v1 生产内核加速 **~2.4–3.3×**(decode 相关的 T=16384 点 **3.28×**)。

| T | v1_fused | **v3 (final)** | 加速 |
|---:|---:|---:|---:|
| 256 | 9.31µs | 3.83 | 2.43× |
| 512 | 11.00µs | 4.74 | 2.32× |
| 1024 | 16.17µs | 6.97 | 2.32× |
| 2048 | 28.30µs | 11.02 | 2.57× |
| 4096 | 52.46µs | 19.06 | 2.75× |
| 8192 | 101.78µs | 35.23 | 2.89× |
| 16384 | 221.30µs | **67.44** | **3.28×** |

硬件 perf counter 证实 v3 是 **VPU-bound,Vector ALU 利用率 75.6%,duty cycle 99%,MXU 0%**——已到 Mosaic/VPU 地板;进一步的算子级优化被 Mosaic 不支持 gather/top_k/变参 reduce 挡住。

---

## 1. 背景与问题定位

### 1.1 v1 是 VPU-bound,不是 compute/HBM-bound
对 v1 fused kernel 做 roofline(E=256,bf16 logits):

- 稳态每-token 成本恒定 ≈ **12.4 ns/token**(固定开销摊薄后水平)。
- 有效 HBM 带宽 ≈ **37 GB/s = v7x 峰值(7.38 TB/s)的 ~0.5%** → **不是 HBM-bound**。
- LLO:MXU=0、DMA=0,全是 scalar/vector select + 控制流 → **不是 MXU-compute-bound**。
- 结论:**VPU / 向量单元 compute-bound**(比较/选择/控制流的指令吞吐)。

### 1.2 根因:reduce 沿 lane 维 = 跨-lane
v1 数组是 `[BT, E]`,E 在最内 lane(128 宽)。所有 top-k 选择都是对 E 的归约(`max`/`argmax`/`min`),即**跨-lane 归约**——TPU 上要走 XLU 跨-lane permute 的慢路径,而且 128 条 lane 被浪费在"把专家塌成一个"上,而不是并行处理 128 个独立 token。

---

## 2. 核心思路:token-in-lane `[E, BT]`

把工作布局转置成 `[E, BT]`(专家进 sublane/major,token 进 128 宽 lane)。于是:

- 对专家的归约变成沿 **sublane/major(axis=0)** 的归约,**128 个 token 在 lane 上并行**,无跨-lane permute。
- 算法与 `gate.py:_biased_grouped_topk` 完全一致:post-bias 分数 → 每组 top-2 求和 → 选 `topk_group` 组 → mask 掉落选组 → 选 `topk` 专家 → 权重取 PRE-bias logit。

代码:`python/sgl_jax/srt/kernels/grouped_topk/v2/kernel3.py`(`grouped_topk_pallas_v3`)。

---

## 3. 优化迭代与逐步收益

每步都在 v7x 上跑 A/B(`profile_compare_training`,device 时间取 XLA-Modules median),correctness 每步用 CPU interpret 校验 + 在 TPU 上按 expert-set 交叉校验。

| 步骤 | 关键改动 | T=16384 | vs v1 |
|---|---|---:|---:|
| ① kernel3 初版 | token-in-lane + 循环内 group-mask + 非-padding 输出 | 89.20µs | 2.49× |
| ② mask 外提 | group-mask 提出 pick 循环(loop-invariant,只算一次) | 82.91µs | 2.68× |
| ③ 单 argmax | `final_select` 每 pick 用一次 `jnp.argmax`(替代 max+masked-min 两次归约) | 78.83µs | 2.81× |
| ④ 输出布局 | kernel 直接输出 `[topk, BS]`,wrapper `.T` = **免费 bitcast**,消灭两个 relayout copy | **67.44µs** | **3.28×** |

**④ 是单步最大收益(+15–24%)**,详见 §4.1。

其它已验证的结论性事实:
- **`padded_topk=128` 没必要**:v1/v2 把 topk 补齐到 128 是多余的;v3 直接输出 `topk=8`(在 sublane,仅需 8 对齐)在 jax0.9/0.10 都能编译。
- **非-128 的 T 也正确**:T∈{200,520,1000,1512,3000,5000,7000}(8 的倍数、非 128 倍数)在 TPU 上编译且与 v1 按 set 对齐——lane 维内部 padding 对逐-token 独立的归约无害。

---

## 4. Mosaic 限制墙(哪些优化做不了)

v3 的瓶颈是 `final_select`(§5),但所有能砍它的融合都撞到同一堵墙:**只有专用归约(`max`/`sum`/`argmax`/`min`)能 lower,通用/数据相关原语不能。**

| 想做的优化 | 预期收益 | TPU 上结果 |
|---|---|---|
| pick 循环后一次性 gather 权重 | 每 pick −1 归约 | ❌ `jnp.take_along_axis` → `_gather_lowering_rule` AssertionError |
| 压缩到每-token 128 个存活专家 | 归约宽度减半(~30%) | ❌ 存活集 per-token,需 per-lane gather → 同上 |
| 变参 `jax.lax.reduce`(一次拿 id+权重+精确 tie) | 每 pick −1 归约 **且** 恢复精确 tie-break | ❌ `Unimplemented primitive ... reduce` |
| kernel 内 `jax.lax.top_k` | 换掉整个循环 | ❌ `Unimplemented primitive ... top_k` |
| XLA 级 `jax.lax.top_k` 路径 | — | ✗ 比 v3 慢 **130–153×**(T=16384:10.4ms vs 67µs) |
| bf16 选择 | VPU 吞吐 | ✗ VPU 是 32-bit ALU,无收益,且放宽精度 |

### 4.1 布局 copy 的根因与修法(④)
编译后 HLO 显示,v1/v2 风格输出 `[BS, topk]` 时 topk=8 落在 lane 维(被 padding 到 128),而消费方要的是 BS 在 lane 的密排布局,于是 XLA 插两个 relayout copy(T=16384 各 ~4.2µs,合计 ~11% 端到端):

```
custom-call → f32[16384,8]{1,0:T(8,128)}   (topk 在 lane,padding 到 128)
%copy       → f32[16384,8]{0,1:T(8,128)}   (BS 在 lane,密排)  ← 两次
```

修法:kernel 直接输出 `[topk, BS]`(BS 在 lane,天然密排),wrapper `.T` 得到 `[BS, topk]`。因为 `[topk,BS]{1,0}` 与 `[BS,topk]{0,1}` 是**同一份物理字节**,`.T` 被实现成 **bitcast(免费)**:

```
custom-call → f32[8,16384]{1,0}
%bitcast    → f32[16384,8]{0,1}   ← 免费,copy 消失
```

---

## 5. Profiling:瓶颈与 VPU 利用率

### 5.1 各阶段耗时(region trace,named_scope)
| 阶段 | 占 kernel |
|---|---:|
| **final_select**(迭代 argmax 选 top-k) | **~69%** |
| group_top2 | ~11% |
| bias_add / group_select / expert_mask | ~3% each |

### 5.2 硬件 perf counter(T=16384,`_counters_` track,pin libtpu 0.0.38)
| 单元 | 利用率 |
|---|---:|
| **Vector ALU (VPU)** | **75.6%** ← 瓶颈资源 |
| Vector Load | 55.5% |
| Vector Store | 40.7% |
| XLU(跨-lane) | 6.95%(低 → token-in-lane 生效) |
| Scalar ALU | 0.52% |
| MXU | 0% |
| duty cycle(XLA Ops busy/wall) | **99.2%**(几乎无 stall) |
| Vector Fills / Spills | 25,487 / 12,710 |

**判定:确凿 VPU-bound,VPU 利用率 75.6%,流水线 99% 忙。**

### 5.3 交叉验证:op-count roofline
手数 v3 每-token VPU element-ops ≈ **12,232**(final_select 占 84%)。实测 T=16384/67µs → 达成 **~3.0 Tops/s** 向量吞吐 ≈ 估算 v7x 单核 VPU 峰值(8×128 × ~1GHz × ~4 ALU ≈ 4 Tops/s)的 **~75%**——与硬件 counter 的 75.6% 高度吻合。

### 5.4 spills 的来源
`[E,BT]=[256,2048]` 一个数组 = 32×16 = **512 个 VREG**,远超物理寄存器堆(数十个)→ 数组常驻 VMEM、逐 tile 流式计算。`final_select` **全展开(unroll=8)** 8 个 pick 以隐藏依赖链延迟,软件流水让多个 `[E,BT]` tile 临时量同时活跃 → 超出寄存器堆 → spill/fill(fills≈2×spills 因一处 spill 多处 fill)。这是"有益优化(展开+流水)"的代价,不是 bug(见 §6)。

---

## 6. unroll 调优

`jax.lax.fori_loop` 的 unroll:jax 0.9.0 只支持 `{1, num_steps=8}`;升级 jax 0.10.2 后可测 2/4。

| T | u=1 | u=2 | u=4 | **u=8** |
|---:|---:|---:|---:|---:|
| 4096 | 25.78 | 23.41 | 22.22 | **19.17** |
| 8192 | 48.47 | 43.75 | 41.42 | **35.20** |
| 16384 | 93.57 | 84.16 | 79.37 | **67.09µs** |

**单调:展开越多越快,u=8(全展开)在所有 T 最优。** 即 spill 减少的收益(低 unroll)敌不过展开带来的 overlap/去循环开销。结论:**当前 unroll=8 已最优**,spill 无需修。jax 0.10.2 与 0.9.0 性能一致,kernel 版本无关。

---

## 7. 正确性与 tie-break

- **非精确并列输入(所有真实 router logits):v3 与 `ref_biased_grouped_topk` 完全 id-for-id 一致**(id + 顺序 + 权重),CPU interpret 与 TPU 都验证(random + 人造 partial-tie 都 match)。
- **唯一差异:完全并列(分数逐比特相等)**。因 ③ 用了硬件 `argmax`,Mosaic sublane-argmax 的 tie-break 不是最小-index(flat-tie 时给出 stride-8 图案),与 ref 的最小-index 不同。但此时并列专家等价可换(下游 `EPMoE._permute` 按 id 重排,只需 set + 权重配对),功能正确。
- 若需**逐比特对齐 ref(含病态并列)**:把 `final_select` 回退为 `max` + masked-`min`(最小-index),代价 ~5%(≈72µs@16384,~3.05×)。

---

## 8. v2 vs v3(补充)

同期还实现了 v2(token-in-lane 但 **padding 输出 + 循环内 masked-sum 权重 + max+min tie-break**),并做了 (BT,unroll) 自动调优。v3 相对 tuned v2 在各 T 快 3–26%,16384 处最明显(v2 的 padded `[bt,128]` 输出 write+transpose 随 T 增长);v3 的关键增量是**去输出 padding + 单 argmax**。tuned v2 @16384 ≈ 112.6µs(1.96×),v3 @16384 = 67.4µs(3.28×)。

---

## 9. 复现方式

分支 `bench/grouped-topk-v2`,Falcon job 从 GitHub clone 该分支,`--no-deps -e python` 装进镜像 python(保留镜像 jax)。**结果从 `falcon exp logs` stdout 取**(本地 gcloud 账号无 poc bucket 权限)。

- A/B: `profile-gtopk-v2-ab.yaml` → `python -m benchmark.kernels.grouped_topk.profile_compare_training --T ... --E 256`(输出 v1_fused / v1_training_gather / v2_lane / v3_lane 到 metrics.jsonl)
- 各阶段 region 计时: `benchmark/kernels/grouped_topk/profile_v3_scopes.py`
- 硬件 counter(VPU 利用率/spills): `profile_v3_counters.py` + manifest 里 pin libtpu 0.0.38
- HLO(看 copy/bitcast): `dump_v3_hlo.py`
- unroll 扫: `sweep_v3_unroll.py`(`--unroll` 需 jax≥0.10 才支持 2/4)
- top_k/gather 可行性探针: `probe_topk.py`
- 正确性: `PALLAS_INTERPRET=1 pytest python/sgl_jax/test/kernels/grouped_topk_test.py`

---

## 10. 结论与后续

**v3 已到 Mosaic/VPU 地板**:VPU 75.6% 利用、99% duty,所有能砍 `final_select` 的融合(gather/compaction/变参 reduce/top_k)都被 Mosaic 挡住,unroll 亦已最优。算子级没有明显剩余空间。

剩余选项:
1. **接入 `gate.py`(flag 切换)+ 端到端 decode A/B** ← 建议优先。grouped-topk 占 decode step ~3.2%→~1.2%,算子 ~2× 对端到端是"小而实"的收益。
2. v3 的 **BT 扫**(512/1024/2048,spills vs time)——唯一未测的次要交互,预期收益小。
3. 若要逐比特对齐 ref:回退最小-index tie-break(−~5%)。
