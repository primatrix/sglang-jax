# [RFC] Ling-2.5-1T LinearAttention 模块适配（#153）

## 1. 背景

Ling-2.5-1T 是一个 ~1T 参数的稀疏 MoE 模型，采用混合attention架构（来源：[#151](https://github.com/primatrix/sglang-jax/issues/151)）：

> - **10 层 Multi-Latent Attention (MLA)** — DeepSeek-V2 风格压缩 KV
> - **70 层 Lightning Linear Attention (Simple GLA)** — 常量大小 recurrent state
> - `layer_group_size=8`，1:7 MLA:Linear 比例

本 RFC 覆盖 Linear Attention 层的推理适配（#153），即 `BailingMoeV2_5LinearAttention` 的实现。

Linear Attention 使用固定大小的循环状态替代标准 KV Cache，避免序列长度增长带来的显存开销。
核心计算使用 `tops` 库的两个 kernel：prefill 使用 `simple_gla_fwd`（路由到 `chunk_simple_gla_fwd_varlen`），decode 使用 `fused_recurrent_simple_gla`。

---

## 2. 目标

为 Ling-2.5-1T 实现 `BailingMoeV2_5LinearAttention`（Flax NNX 模块），支持 prefill 和 decode 场景，使用固定大小 recurrent state 替代 KV Cache。

---

## 3. 实现规格

| 参数 | 值 | 来源 |
|------|----|----|
| hidden_size | 8192 | model config |
| num_attention_heads (H) | 64 | model config |
| head_dim (K=V) | 128 | model config |
| rope_dim（= head_dim × partial_rotary_factor） | 64 | 由 partial_rotary_factor=0.5 推导 |
| use_qk_norm（控制是否对 Q/K 做 RMSNorm） | true | model config |
| group_norm_size | 8 | model config |
| chunk_size | 64 | kernel 默认值 |
| rms_norm_eps | 1e-6 | model config |
| use_qkv_bias | false | model config |
| use_bias（dense） | false | model config |
| rope_theta | 6_000_000 | model config |
| max_position_embeddings | 131072 | model config |

---

## 4. 设计

### 4.1 Forward 预处理

```
【模型加载时】
Ling 模型 __init__（待实现，文件名 TBD）
    创建 LinearAttentionBackend，所有 LinearAttention 层共享同一实例
model_runner.load_model()（model_runner.py）
    通过 getattr 读取并存为 model_runner.linear_attn_backend
    # 非 LinearAttention 模型为 None，零影响

【每次 forward 前，JIT 外】
tp_worker.py
    ↓ model_runner.linear_attn_backend.get_forward_metadata(batch)
        按 padded batch size（batch.extend_seq_lens）构建 cu_seqlens，shape [N_padded+1]
        backend.T_packed_bucket  # Σchunk-aligned lengths 对齐到 token_paddings，static
        backend.cu_seqlens_dev   # cu_seqlens device array，dynamic，shape 固定
        backend.scatter_idx      # tight-packed → chunk-aligned 位置映射，dynamic，shape 固定
    ↓ model_runner.forward(forward_batch)                     ← JIT 边界
        model_def 含 T_packed_bucket（static，变化触发重编译）
        model_state 含 cu_seqlens_dev 和 scatter_idx（dynamic，traced）
        ↓ BailingMoeV2_5LinearAttention.__call__(...)         ← 见 4.2
```

### 4.2 Forward 流程

```
hidden_states [T, 8192]
    # T 由上层 schedule.py pad 到 {64,128,...,8192} 之一（JAX JIT 静态 shape）
    ↓ QKV 投影（fused）
[T, 3*H*head_dim] = [T, 24576]
    ↓ split + reshape
q, k, v 各自 [T, H, head_dim] = [T, 64, 128]
    ↓ Q、K 各自 RMSNorm（per-head，128 维）；V 不做
    ↓ Partial RoPE（NeoX style）：前 rope_dim(64) 维加位置信息，后 64 维不动；实现时传 `is_neox_style=True` 给 `RotaryEmbedding`（与 bailing_moe.py MHA 层一致）
    # q, k, v 此时形状 [T, H, K]，tight-packed（各请求 token 紧密拼接，T = outer padding bucket，未做 per-seq chunk-aligned padding）
    ↓ 根据 forward_batch.forward_mode 选择 kernel：
      if forward_batch.forward_mode.is_decode()：
        # decode 阶段每个请求处理 1 个 token（token_paddings == bs_paddings）
        # B=T，每个请求占据独立的 B slot，各请求 state 互不影响
        q, k, v → reshape → [T, 1, H, K]
        output, new_state = fused_recurrent_simple_gla(
            q, k, v,
            g_gamma=slopes,                   # [H] 固定衰减，每层不同
            initial_state=recurrent_state,    # [T, H, K, V]
            output_final_state=True,
            scale=None,  # scale=None → tops 默认 K^-0.5；一致性由第 6 节跨框架测试覆盖
        )
      elif forward_batch.forward_mode == ForwardMode.EXTEND:
        # 精确匹配 EXTEND：非 spec 路径中 MIXED 已在 schedule 阶段转为 EXTEND；DRAFT_EXTEND/TARGET_VERIFY 仅在 spec 路径出现，Ling 不支持
        # q, k, v 当前为 tight-packed [T, H, K]，需 scatter 到 chunk-aligned layout 供 chunk kernel 使用
        # LinearAttentionBackend 在 JIT 外已按 padded batch size 预计算好边界
        T_pb      = self.backend.T_packed_bucket        # chunk-aligned packed buffer 长度，scatter buffer 的静态 shape
        cu_seqlens = self.backend.cu_seqlens_dev.value  # [N_padded+1]，各请求在 packed buffer 中的 chunk-aligned 边界
        # scatter：将 tight-packed [T, H, K] 散布到 chunk-aligned [1, T_pb, H, K]
        scatter_idx = self.backend.scatter_idx.value  # [T]，tight-packed → chunk-aligned 位置映射
        # Pallas/Mosaic kernel 无法被 GSPMD 自动切分（custom_call 对 XLA 分区器不透明）；
        # 用 shard_map 显式切分：每个设备在本地 H 分片上独立执行 scatter + simple_gla_fwd。
        # cu_seqlens 以 P() replicated 传入，各设备持有完整 boundary 信息。
        def _prefill_fn(q_local, k_local, v_local, gamma, h0, scatter_idx_p, cu_seqlens_p):
            q_p = scatter_to_packed(q_local, scatter_idx_p, T_pb)   # [1, T_pb, H_local, K]
            k_p = scatter_to_packed(k_local, scatter_idx_p, T_pb)
            v_p = scatter_to_packed(v_local, scatter_idx_p, T_pb)
            return simple_gla_fwd(
                q_p, k_p, v_p,
                g_gamma=gamma,          # [H_local]
                h0=h0,                  # [N_padded, H_local, K, V]
                cu_seqlens_dev=cu_seqlens_p,
                scale=None, use_ht=True, chunk_size=64,
            )
        output_packed, new_state = shard_map(
            _prefill_fn, mesh=self.mesh,
            in_specs=(
                P(None, "tensor", None),        # q
                P(None, "tensor", None),        # k
                P(None, "tensor", None),        # v
                P("tensor"),                    # slopes [H_local]
                P(None, "tensor", None, None),  # h0
                P(),                            # scatter_idx（replicated）
                P(),                            # cu_seqlens（replicated）
            ),
            out_specs=(
                P(None, None, "tensor", None),  # output_packed
                P(None, "tensor", None, None),  # new_state
            ),
            check_vma=False,
        )(q, k, v, slopes, recurrent_state, scatter_idx, cu_seqlens)
        # output_packed [1, T_pb, H, V]；gather 回 [T, H, V]（含外层 padding slot）
        output = gather_from_packed(output_packed, scatter_idx)  # [T, H, V]
        # new_state [N_padded, H, K, V]（N_padded = cu_seqlens 长度减 1，trailing padding 槽对应零长 seq，state 为零）
      else：
        raise NotImplementedError(forward_batch.forward_mode)
    ↓ reshape → [T, H*head_dim] = [T, 8192]
    ↓ GroupRMSNorm(output) * sigmoid(g_proj(hidden_states))
    ↓ dense proj：Linear(H*head_dim → hidden_size)
返回 (output [T, 8192], new_state [N_padded, H, K, V]（prefill）/ [T, H, K, V]（decode）)
```

### 4.3 关键设计说明

**ALiBi slopes 作为衰减系数**
`g_gamma`（形状 `[H]`）= `-build_slope_tensor(H) * (1 - (layer_idx-1)/(num_hidden_layers-1) + 1e-5)`，layer_idx 0-indexed。公式来源为 HF 原始实现（`modeling_bailing_moe_v2_5.py` line 754-755），以此为 ground truth，勿用 sglang 版 `_build_slope_tensor`（存在 off-by-one 差异）：

```python
# HF ground truth
slope = -BailingMoeV2_5LinearAttention.build_slope_tensor(self.num_heads) * (
    1 - (self.layer_idx - 1) / (self.config.num_hidden_layers - 1) + 1e-5
)
```

**TP 时 slopes 的处理**

两条路径的切分方式不同，原因是 Pallas/Mosaic kernel 对 GSPMD（XLA 自动分区器）不透明（`custom_call` 节点内部无法被分析），遇到分片张量时 GSPMD 会插入隐式 all-gather，而非切分 kernel。

- **Decode**（`fused_recurrent_simple_gla`，纯 JAX lax.scan）：GSPMD 可自动传播 H 维 sharding，`g_gamma=self.slope`（形状 `[H]`）随 q 的 sharding 切分，TP=1 和 TP>1 同一代码路径，无需额外处理。
- **Prefill**（`simple_gla_fwd` → Pallas kernel）：使用 `shard_map` 显式切分。`self.slope` 先 reshard 到 `P("tensor")`，`recurrent_state` 先 reshard 到 `P(None, "tensor", None, None)`，再作为 shard_map 参数传入；scatter 和 kernel call 均在 shard_map 内各设备独立执行，无 all-gather。`cu_seqlens` 以 `P()`（replicated）传入，各设备持有完整 boundary 信息。

这一差异（GSPMD vs shard_map）与项目内其他 Pallas kernel 的处理方式一致（参见 `flashattention_backend.py`）。

**循环状态（Recurrent State）**
- Prefill 返回 `[N_padded, H, K, V]`（N_padded = cu_seqlens 长度减 1，与 padded batch size 一致）；Decode 返回 `[T, H, K, V]`（T = padded batch size）；形状不随序列长度增长
- 两者 padded batch size 来自不同 batch，prefill → 首次 decode 时 state 需由 #156 从 per-request 存储池中取出并组装为 `[T, H, K, V]`
- 作为独立参数传入，state 的存储和传递由 #156 管理
- Prefill 走 `simple_gla_fwd`（路由到 `chunk_simple_gla_fwd_varlen`），decode 走 `fused_recurrent_simple_gla`（无 chunk 对齐约束，避免 decay 过度累积）
- **TP sharding**：recurrent_state 预期已按 H 维 sharding，与 q 一致；本模块不负责 sharding 检查，由 #156 在 gather 时保证正确 sharding

**LinearAttentionBackend**
- 独立的 `LinearAttentionBackend(nnx.Module)` 负责 prefill 元数据预计算，与 `FlashAttentionBackend` 并列；所有 LinearAttention 层共享同一 backend 实例
- `get_forward_metadata(batch)` 在 `tp_worker.py` JIT 调用前执行，decode 和 prefill 均会被调用，按 forward_mode 分发：
  - **DECODE**：no-op，直接 return（`batch.extend_seq_lens` 在 decode 时为 `None`，scatter/gather 元数据仅 prefill 使用）
  - **EXTEND**：使用 numpy `batch.extend_seq_lens` 计算每条请求的 chunk-aligned 长度，构建 `cu_seqlens` 和 `scatter_idx`
- `T_packed_bucket`（Python int）存为 backend 普通属性，进入 NNX graphdef（static），变化时触发重编译；`cu_seqlens_dev` 和 `scatter_idx`（JAX array）存为 `nnx.data`，进入 NNX state（dynamic），N 变化不触发重编译
- `cu_seqlens` shape 使用 padded batch size（`len(batch.seq_lens)`，对齐 bs_paddings），trailing padding 槽对应零长 sequence，kernel 跳过；kernel 行为保证这些 trailing 槽的 state 输出为零，#156 写回 pool 时无需额外 masking

**scatter_to_packed / gather_from_packed 实现**
- `get_forward_metadata`（JIT 外，numpy）预计算 `scatter_idx: [T]`（T = 外层 padding bucket，与 q 第一维对齐）：遍历 N_real 请求，将每个实际 token 映射到其在 chunk-aligned buffer 中的目标位置；尾部 padding slot 映射到位置 `T_pb`（dummy slot，chunk-aligned buffer 末尾多分配的一个无效槽，实际 token 不会到达该位置，不会覆盖真实数据）；`scatter_idx` 存为 `nnx.data`（形状静态，值动态），与 `cu_seqlens_dev` 并列更新
- scatter（JIT 内）：`jnp.zeros([1, T_pb+1, H, K]).at[0, scatter_idx].set(q)[:, :T_pb]`；gather（JIT 内）：`jnp.pad(output_packed, ((0,0),(0,1),(0,0),(0,0)))[0, scatter_idx]`，直接返回 `[T, H, V]`（padding slot 读到 dummy 列的零值，不影响后续计算）；traced 索引由 XLA 编译为 scatter/gather op，无 Python 循环

**Prefill 多请求处理**
- 将所有请求的 q/k/v 通过 scatter 打包到 chunk-aligned 的 `[1, T_packed_bucket, H, K]` buffer，单次调用 `simple_gla_fwd`，`cu_seqlens_dev` 作为 sequence boundary 传入 kernel（路由到 `chunk_simple_gla_fwd_varlen`），在 boundary 处 reset state，各请求 state 天然隔离

**Intra-chunk padding 对 state 的影响**
scatter 后 chunk-aligned buffer 中的 intra-chunk padding 位置填零（q=k=v=0，来自 `jnp.zeros` 初始化）。kernel 处理这些位置时 `h_t = decay * h_{t-1}`，state 额外衰减 `(chunk_size - seq_len % chunk_size) % chunk_size` 步。最坏情况：`seq_len=1` pad 到 64，state 多乘 `decay^63`。实际精度影响取决于各 head 的 |g_gamma| 量级（由 base slope 和 layer-dependent scale factor 共同决定），不在此做不精确的量化估算；具体误差由第 6 节跨框架数值验证覆盖。

接受此精度损失的理由：intra-chunk padding 仅影响 prefill 路径；`seq_len=1` 的 prefill 极少见；decode 路径已通过 `fused_recurrent_simple_gla` 完全规避。

**Decode 多请求处理**
- q/k/v reshape 为 `[T, 1, H, K]`，B 维各 slot 对应独立请求，state 天然隔离，一次 kernel call 处理所有请求

**Padding 层级**
- 上层 `schedule.py` 保证总 T 已 pad 到 `{64, 128, ..., 8192}` 之一（固定 JAX JIT 静态 shape），本模块直接使用
- Prefill：`LinearAttentionBackend.get_forward_metadata` 在 JIT 外计算各请求 chunk-aligned 长度及 `cu_seqlens`，scatter/gather 在 JIT 内以 traced ops 执行；Decode：无需额外 padding

### 4.4 实现决策

| 决策 | 选择 | 原因 |
|------|------|------|
| decode kernel | `fused_recurrent_simple_gla` | decode 每步只有 1 个 token，chunk+padding 会将其 pad 到 64，导致 state 被乘以 `decay^64` 而非 `decay^1` |
| prefill 多请求策略 | scatter → 单次 kernel call（cu_seqlens） | 所有请求一次 `simple_gla_fwd` 调用，cu_seqlens 在 kernel 内做 boundary reset；避免逐请求串行调用带来的 Python 循环和多次 kernel launch 开销 |
| cu_seqlens 构建 | JIT 外 numpy 预计算，存入 LinearAttentionBackend | 与 FlashAttention 的 `get_forward_metadata` 模式一致；ForwardBatch 公共接口不变；cu_seqlens shape 用 padded batch size 固定，防止重编译 |
| scatter_idx 构建 | JIT 外 numpy 预计算，存为 `nnx.data` | 形状 `[T]` 静态（不触发重编译）；JIT 内 `at[].set()` / 高级索引编译为 XLA scatter/gather，无 Python 循环 |
| slopes 存储 | `__init__` 中计算后存为属性 | JAX JIT 将 Python 属性视为常量，等价于 PyTorch `register_buffer` |
| prefill TP 切分方式 | `shard_map` 显式切分 | Pallas kernel 编译为 `custom_call`，GSPMD 无法感知内部语义，遇到分片输入会插入 all-gather；shard_map 让每设备直接在本地 H 分片上运行 scatter + kernel，无通信开销；与项目内 FlashAttention 等 Pallas kernel 的 TP 处理方式一致 |
| GroupRMSNorm 集成 | 直接集成（#884 已合入） | GroupRMSNorm 已在公开仓 PR #884 实现（`layers/attention/fla/group_rmsnorm.py`），本模块直接使用，无需 stub |

---

## 5. 接口

```python
class LinearAttentionBackend(nnx.Module):
    def __init__(self): ...

    def get_forward_metadata(self, batch: ModelWorkerBatch) -> None:
        # 在 tp_worker.py JIT 调用前执行
        # 更新 self.T_packed_bucket（int，static）、self.cu_seqlens_dev 和 self.scatter_idx（nnx.data，dynamic）
        ...

class BailingMoeV2_5LinearAttention(nnx.Module):
    def __init__(self, config, layer_idx, mesh, backend: LinearAttentionBackend,
                 dtype: jnp.dtype = jnp.bfloat16): ...

    def __call__(
        self,
        positions: jax.Array,           # [T]
        hidden_states: jax.Array,       # [T, hidden_size]
        forward_batch: ForwardBatch,    # 含 forward_mode，用于区分 prefill/decode
        recurrent_state: jax.Array,         # prefill: [N_padded, H, K, V]，N_padded = padded batch size；decode: [T, H, K, V]，T = padding 后的 batch size（每请求 1 token）；首次由 #156 传入全零 array
    ) -> tuple[jax.Array, jax.Array]:
        # returns: (output [T, hidden_size], new_state [N_padded, H, K, V] for prefill / [T, H, K, V] for decode)
        ...
```

---

## 6. 验证方式

### 黑盒测试

| 测试点 | 验证方法 |
|--------|----------|
| 输出 shape 正确 | `output.shape == [tokens, 8192]` |
| new_state shape 正确（prefill） | `new_state.shape == [N_padded, H, K, V]`，即 `[N_padded, 64, 128, 128]` |
| new_state shape 正确（decode） | `new_state.shape == [T, H, K, V]`，即 `[T, 64, 128, 128]` |
| Decode 时 state 在更新 | 两次调用的 new_state 数值不同 |
| state 跨步传递有效 | 先跑一步 decode 得到 new_state，再用相同 q/k/v 分别传 recurrent_state=zeros 和 recurrent_state=new_state，两次 output 数值不同 |
| 首次 prefill（zeros state）正常运行 | recurrent_state 为全零 [N_padded, H, K, V]，不报错，输出 shape 正确 |
| Prefill 正常运行 | seq_len=64/128/512（T 须为 chunk_size=64 的倍数，由上层保证），不报错，输出 shape 正确 |
| Decode 正常运行 | 单步 decode（N=1 请求），不报错，输出 shape 正确 |
| 非 chunk 对齐 prefill 后接 N 步 decode（集成，需 #156） | seq_len 非 chunk_size 整数倍，prefill 后接 N 步 decode；state 正确传递，每步 decode 输出数值不同，new_state 持续更新 |

### 白盒测试

| 测试点 | 验证方法 |
|--------|----------|
| QKV 投影 shape | `q.shape == [tokens, 64, 128]` |
| V 不做 RMSNorm | V 在 norm 前后数值不变 |
| RoPE 只作用前 64 维 | 后 64 维数值在 RoPE 前后不变 |
| gating 计算正确 | gate 数值在 0~1 之间 |
| output shape 在 gating 前后不变 | `[tokens, 8192]` 保持不变 |
| dense proj 生效 | dense 前后的 tensor 数值不同 |
| ALiBi slopes 数值正确 | slope 全为负数；layer_idx 越大 magnitude 越小；任意 layer_idx 的数值与按 4.3 节公式手算结果一致 |
| g_gamma 路径数值正确 | 用 g_gamma=[H] 和等价的 g（expand）喂同样输入，两条路径的 output 和 new_state 数值一致；prefill 时 g=[1, T_pb, H]，decode 时 g=[T, 1, H]（T 为 decode 请求数，1 为单步） |
| GLA 封装数值正确（prefill） | 用相同 scatter_idx 对 q、k、v 做 scatter 得到 packed 张量，再直接调 `simple_gla_fwd`（传相同 cu_seqlens_dev），对比模块内部调用的 output 和 new_state 数值是否一致 |
| GLA 封装数值正确（decode） | 用同样的 q、k、v、g_gamma、h0 直接调 `fused_recurrent_simple_gla`，对比模块内部调用的 output 和 new_state 数值是否一致 |
| decode 多请求 state 隔离正确 | 2 个请求分别单独 decode 的 output 和 new_state，与打包在一起（reshape 为 [T,1,H,K]）decode 的结果数值一致 |
| prefill 多请求 state 隔离正确 | 2 个请求分别单独 prefill 的 output 和 new_state，与打包在一起（scatter → 单次 `simple_gla_fwd`，cu_seqlens 分隔）prefill 的结果数值一致 |

### 多卡测试（TP=2）

模块实现中包含 `mesh` 参数用于张量并行，需验证多卡下行为正确：

| 测试点 | 验证方法 |
|--------|----------|
| TP=2 下输出 shape 正确 | 用 2 张卡构造 mesh，输出 `output.shape == [tokens, 8192]` |
| TP=2 与单卡数值一致 | 相同输入，TP=2 与 TP=1 的 output 和 new_state `max abs diff < 1e-5` |

### 跨框架数值一致性

用相同输入（固定随机种子）对比 JAX 实现与 HuggingFace PyTorch `BailingMoeV2_5LinearAttention` 的输出：

| 验证项 | 要求 |
|--------|------|
| 输出 shape 一致 | `output.shape == torch_output.shape` |
| float32 模式数值对齐 | 两侧均以 `dtype=float32` 运行，`max abs diff < 1e-5` |
| bfloat16 模式数值对齐 | 两侧均以 `dtype=bfloat16` 运行，`max abs diff < 0.05`（bfloat16 精度约 1e-2，含 XLA/PyTorch 运算顺序差异） |
| new_state 数值对齐 | 同上，按对应 dtype 分别验证 |

> 跨框架对比按 dtype 分档：float32 用 `1e-5`，bfloat16 用 `0.05`。精度差异来源：XLA 与 PyTorch 浮点运算顺序不同 + dtype 本身精度限制。

---

## 7. 工作拆解

- [ ] 实现 `LinearAttentionBackend`（`linear_attention_backend.py`）：`get_forward_metadata` 计算 `T_packed_bucket`、`cu_seqlens_dev` 和 `scatter_idx`
- [ ] `model_runner.py`：`load_model()` 末尾添加 `self.linear_attn_backend = getattr(self.model, "linear_attn_backend", None)`
- [ ] `tp_worker.py`：`attn_backend.forward_metadata` 赋值后、`model_runner.forward()` 前添加 `linear_attn_backend.get_forward_metadata(batch)` 调用
- [ ] 实现 `__init__`：
  - QKV proj（`scope_name="query_key_value"`）、g_proj（`scope_name="g_proj"`）：列并行，`kernel_axes=(None, "tensor")`
  - dense（`scope_name="dense"`）：行并行，`kernel_axes=("tensor", None)`
  - Q/K RMSNorm（`scope_name="query_layernorm"`/`"key_layernorm"`，注意 `param_dtype=dtype`，默认为 float32）
  - RotaryEmbedding、ALiBi slopes（存为 `self.slope`）
  - g_norm：`GroupRMSNorm(hidden_size=H*head_dim, num_groups=group_norm_size, epsilon=rms_norm_eps, scope_name="g_norm")`，来自 `layers/attention/fla/group_rmsnorm.py`
- [ ] 实现 forward：QKV 投影 → split + reshape → Q/K norm → Partial RoPE → kernel（decode/prefill 分支，prefill 含 scatter/gather）→ gating → dense → 返回 state
- [ ] 接入已就绪的 `GroupRMSNorm`（`layers/attention/fla/group_rmsnorm.py`）
- [ ] 编写单元测试和集成测试

---

## 8. 依赖

| 依赖 | 状态 |
|------|------|
| `simple_gla_fwd` / `chunk_simple_gla_fwd_varlen` cu_seqlens_dev 支持（tops 库，pallas-kernel PR #153 feat/varlen 分支） | **进行中**（参数签名已存在，Pallas TPU kernel 内部支持预计近期完成） |
| `fused_recurrent_simple_gla`（tops 库，用于 decode 正确性） | **已就绪**（[pallas-kernel#89](https://github.com/primatrix/pallas-kernel/issues/89) 通过 [PR #92](https://github.com/primatrix/pallas-kernel/pull/92) 于 2026-03-30 合并） |
| `GroupRMSNorm`（#152 → 公开仓 PR #884） | **已就绪**（`python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py`，已同步至私有仓） |
| DecoderLayer 层级 dispatch（#155） | 后续任务 |
| model runner state 管理（#156） | **前置依赖（阻塞端到端推理）** — 当前 model runner 仅支持 KV Cache；#156 需扩展支持 recurrent state 的存取与请求间传递 |

---

## 9. 参考资料

| 资料 | 路径 |
|------|------|
| 需求 | [#153](https://github.com/primatrix/sglang-jax/issues/153) / [#151](https://github.com/primatrix/sglang-jax/issues/151) |
| HuggingFace 模型 | https://huggingface.co/inclusionAI/Ling-2.5-1T |
| 原始模型 config | `~/.cache/huggingface/hub/models--inclusionAI--Ling-2.5-1T/snapshots/0312aaca89cf97294676319d93dc8bbc7b46284d/config.json` |
| 原始模型实现（PyTorch） | `~/.cache/huggingface/hub/models--inclusionAI--Ling-2.5-1T/snapshots/0312aaca89cf97294676319d93dc8bbc7b46284d/modeling_bailing_moe_v2_5.py`（`BailingMoeV2_5LinearAttention`，forward 流程、slope 计算、gating 逻辑） |
| sglang PyTorch 推理实现 | `sglang/python/sglang/srt/layers/attention/linear/lightning_backend.py`（仅供了解 sglang 推理侧架构；Ling-2.5-1T 实际走 `linear_backend="seg_la"` 路径，即 `_linear_attention_entry` → `seg_la_fwd`，与 tops 使用不同 kernel；`_prefill_and_mix_infer` 仅限 `linear_backend="minimax"` 场景；`_build_slope_tensor` 使用 `layer_id`（0-indexed），slope 公式与 HF 有 off-by-one 差异，不作为公式依据） |
| kernel 实现 | `tops/ops/simple_gla/__init__.py`（`simple_gla_fwd`，dispatch 入口）、`tops/ops/simple_gla/chunk.py`（`chunk_simple_gla_fwd_varlen`，prefill varlen 实现）、`tops/ops/simple_gla/fused_recurrent.py`（`fused_recurrent_simple_gla`，decode 实现）（[pallas-kernel](https://github.com/primatrix/pallas-kernel)） |
| 参考模型实现（已有） | `python/sgl_jax/srt/models/bailing_moe.py`（MHA 层，LinearBase / RMSNorm / RotaryEmbedding 用法参考） |
