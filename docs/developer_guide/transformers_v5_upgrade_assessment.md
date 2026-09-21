# Transformers v5 升级影响评估

日期：2026-09-21

仓库基线：`c8a6175dc9e7b497f885e864c1ffe6036a837f3f`

当前依赖：`transformers~=4.57.1`

上游对照基线：Transformers **v5.0.0**。最终选定的 5.x 小版本仍需单独验证。

## 1. 结论与评估边界

升级影响不仅包括 VLM 和 Omni，还包括纯文本模型的 RoPE、Tokenizer/Processor、HF 参考测试，以及 FLUX/Wan 等多阶段生成模型。

优先处理三个问题：

1. `huggingface-hub` 的现有版本约束与 v5.0.0 不兼容。
2. 仓库有 5 个文件直接依赖已移除的 `transformers.modeling_flax_utils`。
3. 多个模型仍读取旧 RoPE 字段，可能静默使用默认值，造成数值或生成质量偏差。

本文基于仓库静态检查和官方迁移指南、模型源码。**未安装 v5、未执行模型推理或数值回归，也未实施代码迁移。**“确定”表示已有上游变化和仓库调用证据，不表示已运行复现。

优先级：P0 为安装/导入阻断或核心正确性风险；P1 为接口和模型兼容；P2 为旧补丁清理及辅助路径检查。

## 2. 影响总表

| 优先级 | 改动点 | 当前判断 | 主要影响范围 |
|---|---|---|---|
| P0 | Transformers / Hub 依赖联动 | 确定存在约束冲突 | 安装、下载、缓存 |
| P0 | 移除 `modeling_flax_utils.ACT2FN` 依赖 | 确定存在直接导入 | Qwen2.5-VL、FLUX、Wan |
| P0 | RoPE 配置统一解析 | 多处旧字段读取；部分模型已有适配 | 纯文本、VLM、Omni、上下文长度 |
| P1 | HF VLM 子模块访问路径 | 已找到旧路径调用；按模型确认 | HF runner、参考测试 |
| P1 | 视觉输出对象与 DeepStack | 上游已变化；逐调用点适配 | VLM、Omni 参考对齐和阶段边界 |
| P1 | Tokenizer backend / Processor | 已有 `use_fast=True`，仍需行为验证 | 文本、图像、视频、结构化输出 |
| P1 | 嵌套 config 和本地注册 | 部分已处理 | Qwen-VL、Qwen3.5、Omni |
| P1 | 权重 key / shape / 转换 | 待对选定 checkpoint 验证 | 原始权重加载、HF state_dict 对齐 |
| P2 | 旧 tokenizer workaround 清理 | 已找到候选逻辑 | Gemma4、v4 兼容路径 |
| P2 | Remote code 和 HF 内部 API | 待目标版本验证 | 自定义模型、架构解析 |

## 3. 依赖与 Flax 工具移除

### 3.1 依赖必须联动调整

[python/pyproject.toml](../../python/pyproject.toml) 当前同时约束：

```text
transformers~=4.57.1
huggingface-hub~=0.34.3
```

v5.0.0 要求 `huggingface-hub>=1.3.0,<2.0`，现有 Hub 约束无法满足。因此不能只改 Transformers 版本。还需检查解析出的 `tokenizers` 版本、可选 `fastokens` 和多模态依赖，并回归下载、缓存命中及离线加载。

依据：[v5.0.0 setup.py](https://github.com/huggingface/transformers/blob/v5.0.0/setup.py)。

### 3.2 `modeling_flax_utils.ACT2FN`

v5 移除了 Transformers 自带的 TensorFlow/JAX 实现。以下文件直接导入 `modeling_flax_utils`：

| 文件 | 用途 |
|---|---|
| [models/qwen2_5_vl.py](../../python/sgl_jax/srt/models/qwen2_5_vl.py) | 单模型 VLM 视觉激活函数 |
| [qwen2_5_vit.py](../../python/sgl_jax/srt/multimodal/models/qwen2_5VL/qwen2_5_vit.py) | 多阶段 Qwen2.5-VL 视觉激活函数 |
| [visual_embedding.py](../../python/sgl_jax/srt/multimodal/layers/visual_embedding.py) | FLUX/Wan 共用视觉层 |
| [adalayernorm.py](../../python/sgl_jax/srt/multimodal/layers/adalayernorm.py) | FLUX 归一化层 |
| [flux.py](../../python/sgl_jax/srt/multimodal/models/dits/flux.py) | FLUX MLP |

建议维护仓库自己的 JAX 激活函数映射，核对 `gelu`、`gelu_pytorch_tanh`、`silu` 的计算语义。不能直接替换为 PyTorch 的 `transformers.activations.ACT2FN`。仓库自有 Flax NNX 模型并不因此需要重写。

依据：[移除 TensorFlow/JAX](https://github.com/huggingface/transformers/blob/v5.0.0/MIGRATION_GUIDE_V5.md#removal-of-tensorflow-and-jax)。

## 4. RoPE 配置迁移

### 4.1 需要迁移的内容

原始改动方向：

```python
# 旧访问方式
base = int(config.rope_theta)
rope_scaling = config.rope_scaling

# v5 简单、非嵌套配置的访问示意
base = int(config.rope_parameters["rope_theta"])
rope_scaling = config.rope_parameters
```

这不能作为全仓机械替换规则。还需处理 `partial_rotary_factor`、MRoPE、YaRN 等参数，以及按 layer type 嵌套的配置。自定义配置也可能继续使用旧字段。

建议在配置入口统一解析，并明确新旧字段同时存在时的优先级。至少保留：`rope_type`、`rope_theta`、`partial_rotary_factor`、`mrope_section`、`mrope_interleaved`、`factor`、`original_max_position_embeddings` 及模型所需的缩放参数。

依据：[上游 Configuration 变更](https://github.com/huggingface/transformers/blob/v5.0.0/MIGRATION_GUIDE_V5.md#configuration)。

### 4.2 仓库命中范围

| 范围 | 位置 | 风险 |
|---|---|---|
| 纯文本 | [llama.py](../../python/sgl_jax/srt/models/llama.py)、[qwen2.py](../../python/sgl_jax/srt/models/qwen2.py)、[qwen3.py](../../python/sgl_jax/srt/models/qwen3.py)、Qwen2/3-MoE、GLM4-MoE、DeepSeek-V3 | 仍读取旧字段 |
| Omni Thinker | [qwen3_omni_thinker.py](../../python/sgl_jax/srt/multimodal/models/qwen3_omni_moe/qwen3_omni_thinker.py) | attention 参数和 MRoPE section 分别读取旧字段 |
| 多阶段 VLM | [qwen2_5_vl_generation.py](../../python/sgl_jax/srt/multimodal/models/qwen2_5VL/qwen2_5_vl_generation.py) | 旧 MRoPE 字段 |
| 单模型 VLM | [qwen2_5_vl.py](../../python/sgl_jax/srt/models/qwen2_5_vl.py) | 文本 MRoPE 和视觉配置需分别核对 |
| 上下文长度 | [hf_transformers_utils.py](../../python/sgl_jax/srt/hf_transformers_utils.py) 的 `get_context_length()` | 仍只读取 `rope_scaling`，可能遗漏扩展因子 |

尤其注意 `getattr(config, "rope_theta", default)` 和 `getattr(config, "rope_scaling", None)`：旧字段消失后可能不报错，而是静默改变位置编码。

### 4.3 已有适配

- [qwen3_vl.py](../../python/sgl_jax/srt/models/qwen3_vl.py) 已把部分 `rope_parameters` 内容转为旧字段；仍需验证其默认值和保留参数是否覆盖目标模型。
- [configs/qwen3_5.py](../../python/sgl_jax/srt/configs/qwen3_5.py) 和 [configs/qwen4_exp.py](../../python/sgl_jax/srt/configs/qwen4_exp.py) 已实现部分展平逻辑。
- 不能因已有转换就认定全部模型兼容，也不能无条件删除这些兼容层。

## 5. VLM 与 Omni 接口

### 5.1 子模块访问路径

对发生重构的 HF VLM 类，需检查 `model.visual → model.model.visual`，并一起检查 `language_model`、`embed_tokens`。

具体命中：[test/srt/lora/misc/hf_runner.py](../../test/srt/lora/misc/hf_runner.py) 的 `_forward_gme_qwen2_vl()` 直接访问 `self.model.visual`，还把返回结果当 Tensor 调用 `.to()`。

需要区分：

- HF 模型对象的属性路径；
- 仓库自有 NNX 模型的属性路径；
- checkpoint 文件中的权重 key。

三者不能一起替换。仓库自有模型仍定义 `self.visual`。v5.0.0 Qwen3-Omni Thinker 也仍直接定义 `visual`、`audio_tower`，不能照搬普通 VLM 的迁移路径。

依据：[上游 Modeling 变更](https://github.com/huggingface/transformers/blob/v5.0.0/MIGRATION_GUIDE_V5.md#modeling)、[Qwen3-Omni 实现](https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py)。

### 5.2 视觉输出与 DeepStack

HF Qwen3-VL 使用 `BaseModelOutputWithDeepstackFeatures`，继承自 `BaseModelOutputWithPooling`：

| 字段 | Qwen3-VL 语义 |
|---|---|
| `last_hidden_state` | merger 前的视觉特征 |
| `pooler_output` | merger 后、接入语言模型的特征 |
| `deepstack_features` | 注入中间语言层的视觉特征 |

不能统一用 `.last_hidden_state` 替代旧 Tensor。`get_image_features()` 还会把 `pooler_output` 按图片切分，直接调用视觉模型与调用 feature helper 的结果形状不同。

Omni 音频也不能套用视觉规则：v5.0.0 音频 encoder 将结果放在 `last_hidden_state`。仓库 [test_qwen3_omni_moe_encoder.py](../../python/sgl_jax/test/multimodal/test_qwen3_omni_moe_encoder.py) 已按该字段取值。

依据：[Qwen3-VL 实现](https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)、[Qwen3-Omni 实现](https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py)。

### 5.3 Multistage Gen 的验证边界

HF 输出变化不会自动改变仓库自有 JAX encoder 的输出协议。迁移应重点验证参考模型调用和适配边界，再检查阶段间数据是否保持一致：

- Qwen2.5-VL：视觉 embedding、视觉 token 数、MRoPE position IDs。
- Qwen3-Omni：视觉/音频 embedding、DeepStack 特征、mask 和位置编码。
- FLUX/Wan：共享激活函数替换后各阶段数值和最终生成结果。

相关入口包括 [qwen3_omni_thinker_embedding.py](../../python/sgl_jax/srt/multimodal/models/qwen3_omni_moe/qwen3_omni_thinker_embedding.py) 和 [静态 stage 配置](../../python/sgl_jax/srt/multimodal/models/static_configs)。

## 6. Tokenizer、Processor 与配置兼容

### 6.1 `use_fast=True` 不是完整迁移方案

[get_processor()](../../python/sgl_jax/srt/hf_transformers_utils.py) 已默认 `use_fast=True`。v5 调整 tokenizer backend 组织方式，应核对实际加载的实现和行为：

- `tokenizer_mode="slow"` 对目标模型是否仍符合预期。
- `PreTrainedTokenizerFast` 类型判断、`fastokens` monkey patch 是否兼容。
- [llguidance_backend.py](../../python/sgl_jax/srt/constrained/llguidance_backend.py) 的 tokenizer 分支与结构化输出是否正常。
- 特殊 token、BOS/EOS、padding、工具调用模板产生的 token IDs 是否一致。
- 图像/视频的 `pixel_values`、grid、token 数和数值误差是否符合预期。

Tokenizers 和 image processors 的“fast”含义应分别验证。

依据：[v5 Tokenizers 文档](https://huggingface.co/docs/transformers/v5.0.0/en/fast_tokenizers)。

### 6.2 Chat template 返回类型

v5 的 tokenized chat template 返回值需要按 `BatchEncoding` 处理。仓库 [serving_chat.py](../../python/sgl_jax/srt/entrypoints/openai/serving_chat.py) 已通过 `Mapping` 提取 `input_ids`；该项属于已有适配，仍需回归其他调用路径。`tokenize=False` 的字符串路径应单独看待。

### 6.3 嵌套 config 与本地注册

[hf_transformers_utils.py](../../python/sgl_jax/srt/hf_transformers_utils.py) 已有 `get_hf_text_config()` 和部分根配置字段复制逻辑。需要检查直接调用 `AutoConfig` 的路径是否也取得正确的子配置。

该文件已强制注册本地 Qwen3.5 配置，以保留 RoPE 展平和 hybrid/GDN 接口。不能仅因上游提供同名配置就移除本地实现。

### 6.4 旧补丁与保存/重载

- Gemma4 的 `extra_special_tokens={}` workaround 仍需审核是否应按版本限制或移除。
- 其他部分 tokenizer 补丁已有 `<5.0.0` 条件，升级后应验证分支是否如预期跳过。
- 对 processor/tokenizer 的本地缓存、保存后重载进行回归；v5 调整了配置序列化结构。

依据：[官方迁移指南](https://github.com/huggingface/transformers/blob/v5.0.0/MIGRATION_GUIDE_V5.md)。

## 7. 权重与辅助路径：待验证项

以下为需要验证的风险，不代表已经确认失效：

1. **原始 safetensors 与 HF `state_dict()`：** 分别核对 key、shape、前缀和转换，不能从 Python 属性路径变化推导原始权重 key 必须变化。
2. **参考测试内部 API：** Omni 视觉测试使用 `fast_pos_embed_interpolate`、`rot_pos_emb`、`patch_embed` 等内部实现，需针对目标版本验证签名与结果。
3. **Remote code / 架构解析：** [model_loader/arch.py](../../python/sgl_jax/srt/model_loader/arch.py) 使用 HF 动态模块接口；需要测试实际支持的自定义模型。
4. **dtype / generation config：** 审核 HF 参考模型构建和嵌套配置传递；不要把旧参数仍被兼容视为无需回归。

HF v5 引入了动态权重转换机制，因此 HF 加载结果和原始 checkpoint 应分别取证。依据：[Dynamic weight loading](https://github.com/huggingface/transformers/blob/v5.0.0/MIGRATION_GUIDE_V5.md#dynamic-weight-loading)。

## 8. 建议实施顺序与验收

### 8.1 实施顺序

1. 固定目标 Transformers 5.x 小版本和用于回归的 checkpoint revision。
2. 调整依赖约束，替换 `modeling_flax_utils`，完成安装与导入检查。
3. 统一 RoPE 解析，验证嵌套 config 和上下文长度。
4. 适配 HF 子模块访问、输出字段及参考测试。
5. 回归 tokenizer/processor，再执行模型数值和端到端测试。
6. 在覆盖验证后清理 v4 workaround。

### 8.2 验收矩阵

| 层级 | 必测内容 | 验收重点 |
|---|---|---|
| 安装与导入 | 基础环境、多模态环境、关键模型模块 | 无依赖冲突和已删除符号导入 |
| 配置 | 纯文本、Qwen-VL、Omni、Qwen3.5 | 子配置、RoPE 参数、context length 正确 |
| Tokenizer | 普通文本、工具调用、结构化输出 | token IDs、特殊 token 和输出协议符合预期 |
| Processor | 单图、多图、视频、音频 | grid、mask、token 数、dtype 与数值 |
| Encoder | Qwen2.5-VL、Qwen3-VL、Omni 音频/视觉 | 正确字段、shape、精度阈值 |
| 纯文本生成 | Llama/Qwen、至少一个 MoE、长上下文 | logits/生成回归，发现静默 RoPE 偏差 |
| 多阶段生成 | VLM、Omni、FLUX/Wan | 阶段输入输出及最终结果 |
| 权重加载 | 选定 checkpoint、HF 参考权重 | 无意外 missing/unexpected key 或 shape 变化 |

建议复用现有测试，包括 [Omni 音频 encoder](../../python/sgl_jax/test/multimodal/test_qwen3_omni_moe_encoder.py)、[Omni 视觉 encoder](../../python/sgl_jax/test/multimodal/test_qwen3_omni_vision_encoder.py)、[Omni 视觉对齐](../../test/srt/test_qwen3_omni_vision_alignment.py)、[Qwen3.5 配置/模型测试](../../python/sgl_jax/test/models/test_qwen3_5.py)。

数值阈值应沿用或依据现有基线制定。不能只以“不报错”作为升级成功标准，尤其是 RoPE、图像预处理和 DeepStack。
