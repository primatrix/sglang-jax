# MiMo V2 Pro 推理乱码 debug 日志

> 记录所有排查路径、修复尝试、验证结果，防止上下文丢失。
> 分支：`feat/mimo-v2-pro`（直接 push 到 prim）

## 现象

Pro 模型 v7x 部署后 generate 输出**乱码 / 单 token 重复**（早期看到 token 3422 重复），temperature=0 也复现。

Flash 同代码路径在 v7x 上**输出连贯英文**，作为对照基线。

## 模型/部署信息

- **模型**：MiMo V2 Pro Private（FP8 block-quant，weight_block_size=[128,128]）
- **集群**：TPU v7x，2 节点 × 4 chip = 8 chip = 16 core
- **拓扑**：tp=16, dp=2, ep=16, fused MoE backend
- **manifest**：`/tmp/v7x-mimo-pro.yaml`（Indexed Job + headless service）
- **Flash 对照**：`/tmp/v7x-mimo-flash.yaml`（v7x-4-pool 单机 Pod，tp=8 dp=2 ep=8）

## Architecture diff: Flash vs Pro

确认 Flash 和 Pro **架构完全相同**，只有数值参数不同：

| 项 | Flash | Pro | 影响 |
|---|---|---|---|
| 层数 | 48 | 70 | scale only |
| num_heads | 64 | 128 | TP 切分粒度 |
| num_kv_heads | 4 | 8 | KV head |
| head_dim | 192 | 192 | 同 |
| v_head_dim | 128 | 128 | 同 |
| hidden_size | 4096 | 6144 | scale only |
| n_routed_experts | 256 | 384 | EP 排布 |
| attention_value_scale | 0.707 | 0.612 | v 反量化系数 |
| n_group / topk_group | 1 / 1 | 1 / 1 | grouped routing 必走 |
| qkv 存储 | 分离 q/k/v | **fused qkv_proj** | Pro 走 `_split_qkv_weight` |
| FP8 V scale 排布 | weight 和 scale 都是 compact 1024 | weight compact 1024，**scale padded 到 1536（12 块）** | Pro 需 gather 重映射 |

## 排查时间线

### 阶段 1：MoE TopK grouping fix（commit d7eeb87a, task #17）

**症状**：Pro 输出乱码。

**根因**：`MiMoV2MoE.__init__` (mimo_v2.py:103) 创建 `TopK` 时未传 `n_group`/`topk_group`，默认 0。Pro config 有 `n_group=1, topk_group=1, scoring_func=sigmoid, topk_method=noaux_tc`。SGLang 参考实现要走 `_biased_grouped_topk`，但我们因 `num_expert_group=0` 走了 `_biased_topk`，路由错误。

**修复**：传入 `n_group / topk_group / routed_scaling_factor / layer_id`。

```python
self.topk = TopK(
    topk=num_experts_per_tok,
    renormalize=getattr(config, "norm_topk_prob", True),
    num_expert_group=getattr(config, "n_group", 0) or 0,
    topk_group=getattr(config, "topk_group", 0) or 0,
    routed_scaling_factor=getattr(config, "routed_scaling_factor", None) or 1.0,
    layer_id=layer_id,
)
```

**回归保护**：Flash config 通常 `n_group=1`，会走 grouped 路径同 Pro。

**结果**：Pro **仍然乱码**。继续排查。

### 阶段 2：Padded-V FP8 scale gather fix（commit c91fdccb, task #18）

**症状**：Pro 仍乱码。Flash 在 v7x 跑 OK。

**调查**：

启动日志看到：
```
Expanding linear block-quant scale model.layers.0.self_attn.v_proj.weight_scale
  from (8, 48) to kernel-ready layout [48, 1, 1024]
```

但 Pro fused QKV 的 weight_scale_inv 形状是 `(216, 48) = (192+12+12, 48)`：
- Q 部分 192 行（全 padded `head_dim=192`）
- K 部分 12 行
- V 部分 **12 行**（V 在量化前被 pad 到 head_dim=192 → 8*192=1536 → 12 个 128 块）

**但 V weight 自己存的是 compact 1024 维**（8 head × 128）！

`_split_qkv_weight` 旧代码取 V scale 的前 8 行（`v_dim_keep = (8*128)//128 = 8`），当作 compact-aligned 用。但 padded 第 0/1 块覆盖 head 0 的 0..127 / 128..255，第 1 块前 64 通道是 head 0 的 pad 区，后 64 通道是 head 1 的 0..63 —— **压根不是 head 0 的数据**。

→ 对 head 0 的 scale 只有第 0 块对，后续每 head 都用错 scale。

**修复方案**（用户指引："不要 padding，直接在切分的原 v proj 运用正确的 scale 反量化"）：

`weight_utils.py::_split_qkv_weight`：
1. V 切片保留全部 12 个 padded 块（不 drop 末尾）
2. 在 post-shard loop 里对 V scale 走特殊路径
3. 用 `expand_block_scale(channel_to_block=...)` 做 per-channel gather：
   - compact channel `c = h*128 + w` → padded scale row `(h*192 + w) // 128`
4. 输出直接是 kernel-ready 3D `[48, 1, 1024]`，跳过标准 `_maybe_expand_linear_block_scale`

```python
n_out_compact = self.num_kv_heads * self.v_head_dim_original  # 1024
head_idx = jnp.arange(n_out_compact) // self.v_head_dim_original
within_idx = jnp.arange(n_out_compact) % self.v_head_dim_original
channel_to_block = (head_idx * self.head_dim_original + within_idx) // block_size
# expand_block_scale 内部：scale_per_channel = scale_2d[channel_to_block]
```

**回归保护**：
- 非 fused QKV 模型：不会进 `_split_qkv_weight` 的 padded 分支
- Flash：V weight 和 scale 都是 compact，`actual_dim == expected_scale_dim_real`，走原路径不变

**已确认对齐项**（非 bug）：
1. partial_rotary_factor：SGLang 和我们都是 `int(192 * 0.334) = 64`
2. attention_value_scale=0.612：mimo_v2.py:265 `v = v * self.v_scale` 已用
3. V 在 attention 前 pad 到 192、attention 后 slice 回 128：mimo_v2.py:264-291 已实现，与 Flash 一致

**结果**：等 Pro 重启验证（c91fdccb）。

### 阶段 3：c91fdccb 验证失败 + 切换 epmoe 排查 fused MoE 嫌疑

**时间**：2026-04-18 20:14 Pro 启动完成。

**Coherence test 结果**（commit c91fdccb，fused backend）：

```
"The capital of France is" → "#叉叉##叉叉叉#叉叉叉叉..." (token 102940 重复主导)
"2+2="                    → "- mountMount:ERSHEY mount mount mount #####  0"
"Hello, my name is"        → "   ##叉叉叉叉#叉#  叉叉叉叉"
```

**特征**：
- 输出**与 prompt 相关**（不同 prompt 出不同 token）→ 模型在算东西，不是完全 dead
- 但 logits 严重失真，token 102940 / 2 / 220 等"垃圾"token 主导
- V scale fix 不足以拯救 → 还有别的 bug

**下一步** Direction A：切 `--moe-backend epmoe`（用户建议）

19:14 删 Job，改 `/tmp/v7x-mimo-pro.yaml` 第 75 行 `fused → epmoe`，re-apply。
20:15 新 Pod `v7x-mimo-pro-0-pwwjk` running，开始 weight load。

**判定**：
- 如果 epmoe 输出连贯英文 → fused MoE backend 在 grouped routing / EP 排布上有 bug
- 如果 epmoe 仍乱码 → MoE 不背锅，bug 在 attention path 或别的层


## 待验证清单

- [x] Flash v7x-4 输出连贯（验证 #17 不破坏 Flash）— **PASS** (commit c91fdccb)
  - "The capital of France is" → " Paris."
  - "2+2=" → "4, 4+2=6, 6+2=8,"
- [ ] Pro v7x-8 输出连贯（验证 #17+#18 修好 Pro）— 重启中（2026-04-18 18:50 起，MoE load ~3%）
- [ ] 如 Pro 仍乱码，下一步备选方向
  - epmoe backend 替换 fused 排除 fused MoE 嫌疑
  - sink_bias TP sharding 数值检查
  - V scale sharding mapping `(None, None)` → `(None, "tensor")` 对齐 model param
  - FP8 dequant 路径 (mimo_v2.py:750-783) 的 V 处理

## 关键文件路径

- 模型实现：`python/sgl_jax/srt/models/mimo_v2.py`
- Weight loader：`python/sgl_jax/srt/utils/weight_utils.py`（`_split_qkv_weight` 在 line 1534）
- TopK gate：`python/sgl_jax/srt/layers/gate.py`（grouped 分流在 line 91）
- Block scale 工具：`python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py`（`expand_block_scale` 在 line 232）
- SGLang 参考：`/Users/ramezes/job/opensource/sglang/python/sglang/srt/models/mimo_v2_flash.py`
- Pro Job manifest：`/tmp/v7x-mimo-pro.yaml`
- Flash Pod manifest：`/tmp/v7x-mimo-flash.yaml`

## Commits（feat/mimo-v2-pro）

- `5568702e` (main) feat: add MiMo V2 Pro model support with fused QKV weight loading
- `dc985028` fix: v_scale path（attention_value_scale 入参）
- `d7eeb87a` fix: pass n_group/topk_group/routed_scaling_factor to MoE TopK
- `c91fdccb` fix: remap padded-V FP8 scale to compact channels in fused QKV split

## 验证日志

### 2026-04-18 19:34 safetensors header 验证

直接读 HF safetensors header 确认 Pro fused QKV 真实 layout（layer 5, layer 0.mtp 都同样）：

```
model.layers.5.self_attn.qkv_proj.weight       [27136, 6144] F8_E4M3
model.layers.5.self_attn.qkv_proj.weight_scale_inv [216,   48] F32
```

- 27136 = q(128*192) + k(8*192) + v(8*128) = 24576 + 1536 + 1024 ✓ **V weight = compact 1024**
- 216 = q_scale(192) + k_scale(12) + v_scale(12) ✓ **V scale = padded 12 行（per-head 0..191 // 128）**

→ 强证 fix 方向对：weight 和 scale 在 V 部分用不同 head 边界，必须 gather 重映射。

### K scale 边界分析（确认 K 走标准路径无 bug）

K 的 head_dim_original=192，per-head 不整除 128。K weight 和 K scale 都按 192/head 对齐：
- block 0: head 0 的 ch 0..127
- block 1: head 0 的 ch 128..191 + head 1 的 ch 0..63（混合）
- block 2: head 1 的 ch 64..191（混合）...

producer 量化时就用这个 padded 边界算的 scale，loader 用同样 `channel // 128` 反量化 → **数学一致**。
V 之所以特殊，是因为 V scale 也用 192/head 算（padded），但 V weight 自己却 compact 存 128/head —— **二者用不同的 head 边界**，必须 gather 修正。

### 2026-04-18 18:55 启动观察

Pro 重启后 weight load 阶段 log 关键证据：

```
Expanding linear block-quant scale model.layers.X.self_attn.q_proj.weight_scale
  from (192, 48) to kernel-ready layout [48, 1, 24576]
Expanding linear block-quant scale model.layers.X.self_attn.k_proj.weight_scale
  from (12, 48) to kernel-ready layout [48, 1, 1536]
```

**没有 v_proj.weight_scale 的 expand log** —— 这正是预期：我的 fix 让 V scale 直接在 `_split_qkv_weight` 走 `expand_block_scale(channel_to_block=...)`，跳过 `_maybe_expand_linear_block_scale`。

K proj 的 `(12, 48) → [48, 1, 1536]` 证明 K weight 是 padded 存储（8 head × 192 = 1536），原代码对 K 没问题。

### 2026-04-18 21:20 epmoe pod 状态 + 网络中断

切 epmoe 后 Pod 重新部署：`v7x-mimo-pro-0-pwwjk`（20:14 起）。
- 20:14 prefetch 启动
- 20:28 layer 22 weight load
- 20:39 layer 12 MoE 17%
- 20:49 layer 21 MoE 30%
- 20:59 layer 36 MoE 52%
- 21:09 layer 52 MoE 75%
- 21:19 layer 65 MoE 94%（最后观察）

预计 21:25 左右 ready。但 21:21 起本机 kubectl proxy (127.0.0.1:7890) 连接被重置，无法访问 GKE API。

### 2026-04-19 epmoe 验证：megablox GMM kernel tile-size bug

恢复 kubectl 后查 epmoe pod 状态：两个 pod 均 `Completed`（启动后约 75 分钟挂掉）。

抓 `kubectl logs v7x-mimo-pro-0-pwwjk` 末尾发现 precompile 阶段 crash：

```
[GMM kernel] using default block sizes for key:
  (1024, 6144, 2048, 384, 24, 'bfloat16', 'float8_e4m3fn', 6144): (128, 2048, 2048)
  (2048, 6144, 2048, 384, 24, 'bfloat16', 'float8_e4m3fn', 6144): (256, 2048, 2048)
  (4096, 6144, 2048, 384, 24, 'bfloat16', 'float8_e4m3fn', 6144): (384, 2048, 2048)

ValueError: 4096 must be divisible by x-dimension tile size (384).
  at megablox_gmm_kernel/gmm.py:80 (_calculate_num_tiles)
```

**根因**：default block-size 启发式按 `tm = m / 16` 之类规则选 tile_m，m=1024→128，m=2048→256，m=4096→384。但 4096 % 384 ≠ 0，`_calculate_num_tiles` 直接抛错。这是 megablox kernel 的默认 block 选择 bug，与我们的 weight loading / 模型代码无关。

**结论**：epmoe 当前不可用于 chunked-prefill-size=4096 + 384 expert 的 Pro。要么换 chunked-prefill-size（4608=384×12 或 2048），要么继续用 fused 排查别的 bug。

**决策**：fused 路径起码能跑通端到端、能产 logits，是唯一能继续验证 backup B/C/D 假设的 channel。改回 fused，重启。



Flash 启动 log（Pro 没 fused QKV 的对照）：
```
Expanding linear block-quant scale model.layers.0.self_attn.v_proj.weight_scale
  from (4, 32) to kernel-ready layout [32, 1, 512]
```

Flash full attention：num_kv_heads=4, v_head_dim=128 → V output = 512，scale = (4, 32) — V 没 padding（HF 量化时直接用 v_head_dim_original=128 算 scale）。

→ 证实 **V padding 是 fused QKV 独有的工件**（HF 为了让 cat 后的 Q/K/V 一起 block-quant 时每 head 在 128 块边界上对齐，把 V 从 128/head pad 到 192/head）。Pro 用 fused QKV 才会触发，Flash 分离 q/k/v 不会。

### Dequantize 路径分析

Pro 是 FP8 → 走 `_dequantize_fp8_to_bf16`（mimo_v2.py:530, 1141）：
- 把 FP8 attention proj + layer-0 MLP 在加载完成后**反量化成 bf16**
- inference 时 attention 用 bf16 matmul（不走 FP8 kernel）
- 我的 V scale fix 在 `_split_qkv_weight` 阶段就把 V scale 修正成对的 `[48, 1, 1024]`
- `_block_dequant` 用这个对的 scale 把 V FP8 weight 反量化成 bf16
- 所以 fix 直接生效在 dequant 路径

→ MoE 层仍是 FP8 量化跑（fused MoE backend 处理）。

## 备选 debug 方向（如 c91fdccb 后 Pro 仍乱码）

按概率/优先级排序：

### A. 用户建议：epmoe backend 替换 fused

排除 fused MoE 嫌疑。`--moe-backend epmoe` 是 default，最简单切换。

如果 epmoe OK / fused 乱码 → fused MoE 内部有 bug（grouped routing 假设、expert 排布、scale 处理）

### B. V scale sharding mapping 不对齐

WeightMapping 写 `sharding=(None, None)` (mimo_v2.py:1108)。但 model param `weight_scale` 期望 `P(None, None, "tensor")` (linear.py:213)。当前两者 mismatch，JAX 自动 reshard，反量化数值应该正确（用户已确认影响小），但可能引入 perf bug。

### C. Sink bias 数值检查

`attention_sink_bias` 已切 `P("tensor")` (mimo_v2.py:244)。需要 attach 进程 dump 实际值，确认每 TP rank 拿到 `num_heads / tp_size` 个独立值，不是全 0 或重复。

### D. partial RoPE 二次确认（已对齐，低优先级）

SGLang 和我们都是 `int(192 * 0.334) = 64`。RoPE 实现应该一致。
