# EAGLE/MTP Precompile Cache-Miss 修复计划

## 背景

开启 `--speculative-algorithm EAGLE3` 后,即使去掉 `--disable-precompile`,
precompile 结束后首批请求仍触发大量 XLA 重编译。
原因分两类:

1. **Precompile 覆盖不全** —— 多处 jit 边界没在 precompile 阶段被以全部 bucket 组合 trace 一次。
2. **EAGLE runtime padding 不一致** —— 部分张量按 `real_bs`(未 padding)成形,
   或 pad 到非 bucket 值,导致即使输入 batch 命中 bucket,内部张量 shape 仍变化。

调试入口:在 `ModelRunner.__init__` / scheduler 启动处加上
`jax.config.update("jax_explain_cache_misses", True)`(由 env var 控制),
配合现有 `tp_worker.py:418` 处 `count_pjit_cpp_cache_miss` 包到 `forward(...)` 一并使用,
就能落地复现每条 miss 的 (mode, bs, num_tokens, step_id) 指纹。

---

## 根因清单(均带文件:行号定位)

### R1. 开启 spec 后,target 模型 precompile 被完全跳过
`scheduler.py:433-441` 在 `spec_algorithm` 非 None 时只跑
`draft_worker.run_spec_decode_precompile()`,标准 `tp_worker.run_precompile()`
(`compilation_manager._precompile_extend / _precompile_decode`) 不执行。

后果:
- target extend 只在 `bs = max_padded_batch_size, token_bucket=*` 这一条线上被 trace
  (`eagle_worker.py:283-313`,`precompile_spec_extend`);
- target decode (verify) 只在 `(bs, draft_token_num)` 一组上被 trace
  (`precompile_spec_decode`,`eagle_worker.py:315-366`);
- 标准 `_precompile_decode` 的 `bs × bs` decode 矩阵完全没跑。

### R2. Draft 模型 precompile 覆盖不全
`precompile_spec_decode` 对每个 `bs` 只调用一次
`forward_batch_speculative_generation`,该次调用确实在内部完成了:
draft 多步循环 → verify → draft_extend_for_decode 三阶段;
但 **多步循环里每个 step_id 对应一组独立的 attention metadata**
(`flashattention_backend.py:335-379`,`seq_lens += step`、`step_spec_tokens = current_seq_lens + step*topk`)。
循环只跑一次,意味着 step 0..N-1 都只在「该 bs」下被 trace,
其它 bs bucket 完全没跑过完整的 N 步循环。

此外 draft 的底层 `ModelWorker.compilation_manager` 实例存在,
但 `run_precompile()` 在 spec 路径下被绕过——
draft `forward` 自身的 token/bs 桶完全没 trace。

### R3. Extend 阶段 spec 侧无 bs padding
`eagle_worker.py:80-111` 注释
`# FIXME(pc) add padding logic here`。
prefill 真实 bs 直接进入 target extend 和 `draft_extend_for_prefill`,
而 R1 已说明 target extend 只在 `bs = max_padded_batch_size` precompile,
真实 bs 不命中 → 重编译。

### R4. `draft_extend_for_decode` 输入按 real_bs 切片,非 padded_bs
路径:`verify` 在 `eagle_worker.py:159-204` 产出 `next_draft_input`,
其 `verified_id / hidden_states` 形状由 `accept_index` flatten 而来,
长度 `padded_bs*(spec_steps+1)`;
但 `prepare_for_extend_after_verify`(`eagle_util.py:523-560`)和
`draft_extend_for_decode`(`eagle_draft_worker.py:233-278`)中:
- `select_index` 由 `real_bs` + `accept_lens[: real_bs]` 构造
  (`eagle_draft_worker.py:256-261`);
- `rep_logits[select_index]` 得到 `real_bs` 长度的张量,送入 jitted
  `topk_probs_from_logits`;
- `extend_seq_lens = full(bs_real, step_plus_1)` 中的 bs 是 real_bs。

结果:同一 padded_bs 下,real_bs 不同 → shape 不同 → 重编译。

### R5. `padding_for_decode` 之外的 input_ids 重置
`eagle_draft_worker.py:387-388` 在 padding 末尾把
`input_ids / positions` 重置为 `np.empty(bs * topk, np.int32)`,
bs 与 topk 都是参数。topk 在 precompile 中只取 CLI 配置那一个值,
但 padding 后 bs 是命中 `precompile_bs_paddings` 的那一档 —— 这部分 OK。
**真正的问题**:precompile 中 topk 维度的取值集合 = {`speculative_eagle_topk`} 一种,
若运行时存在动态 topk(目前看是静态;此项作为「确认无问题即可」纳入复核)。

### R6. `build_tree_kernel_efficient` `max_context_len` 兜底非 bucket
`eagle_draft_worker.py:451-457` `_pick_context_len`:
当 `max_seq_len` 不在 `precompile_token_paddings` 内,
fallback `1 << (max_seq_len-1).bit_length()`,该值未必在 bucket 集合内。
`build_tree_kernel_efficient_preprocess` 以
`(num_verify_tokens, batch_size, speculative_num_steps)` 为 `static_argnames`,
但 `max_context_len` 进入 traced 形状 → 非 bucket 值 = 新 cache 项。

### R7. `TARGET_PADDING = 16384` 硬编码
`flashattention_backend.py:373`:
`# FIXME Handle padding, this will be move to precompile`。
当 `bs * cdiv(max_seq_len + spec_steps*topk, page_size) > 16384` 时不再 pad,
shape 由真实值决定 → miss。

### R8. cache_loc / page_indices 桶未覆盖 spec 增量
`_compute_cache_loc_buckets`(`common_utils.py:95-97`)按
`bs_bucket * ceil_to_page(max_req_len)` 算,只在标准 decode 场景下成立。
EAGLE verify / draft 多步会消耗额外 page
(`current_seq_lens + step*topk`),
最大可能 cache_loc 应额外加 `(spec_steps+1)*draft_token_num * bs` 的余量。

---

## 修复策略

### F1. 让标准 precompile 在 spec 模式下也跑
- `scheduler.py:433-441`:无论是否启用 spec,都先执行
  `tp_worker.run_precompile()`(target 模型全 bucket 网格),
  再执行 `draft_worker.run_spec_decode_precompile()`(spec 专属形状)。
- 评估:精度无影响,仅延长启动 precompile 时间。

### F2. Spec precompile 覆盖完整 (bs × step) 矩阵
重写 `EAGLEWorker.precompile_spec_decode`:
- 外层遍历 `precompile_bs_paddings`,内层显式 trace 每个 step 的
  draft forward 与 verify(可考虑独立 dummy 走 `draft_model_runner.forward` +
  `verify` 的两条 jit boundary,不走完整 `forward_batch_speculative_generation`,
  以避免一次性 trace 多次重复);
- 对 `precompile_spec_extend`,改为遍历 `(bs_bucket, token_bucket)` 而非
  只 `(max_padded_bs, token_bucket)`,覆盖 prefill 真实 bs。

### F3. Extend 阶段加 spec bs padding
`eagle_worker.py:80-111` 实现 FIXME:
- prefill 进入 spec 路径前对 `model_worker_batch` 的
  `seq_lens / input_ids / positions / cache_loc / extend_seq_lens`
  以 `precompile_bs_paddings` / `precompile_token_paddings` 对齐;
- padding token 不申请 KV cache(避免误占用),仅用作 shape 占位。

### F4. `draft_extend_for_decode` 全程 padded_bs 化
- `prepare_for_extend_after_verify` / `draft_extend_for_decode` 内部
  所有按 `real_bs` 切片的位置都改为 padded_bs:
  - `select_index` 在 padded_bs 维度上构造,padding 行的 `accept_lens` 填 0,
    `select_index` 取一个固定 dummy 行(确保 gather 合法且不影响真实结果);
  - `extend_seq_lens` 长度 = padded_bs,padding 位填 0(对应 padded 行不参与
    attention,padding token 同样不申请 KV)。
- 配套修改 `EagleVerifyOutput` / `EagleDraftInput` 中
  `verified_id / hidden_states` 的尾部 padding 不被下游误读
  (`eagle_worker.py:181-182` 的 `safe_index` 已是 padded 形状,只需消费侧对齐)。

### F5. `build_tree_kernel_efficient` 严格 bucket 化
- `_pick_context_len` 去掉 `bit_length` fallback,
  改为「`bisect_left` 到下一个 `precompile_token_paddings` 桶,
  超出则报错(precompile 配置不足)」,确保只产生 bucket 内的 shape。

### F6. 把 `TARGET_PADDING` 移入 precompile 配置
- 在 `compilation_manager` 计算 `cache_loc_paddings` 时,
  额外加入 spec 增量:
  `extra = (spec_steps + 1) * draft_token_num * bs_bucket`;
- `get_eagle_multi_step_metadata` 中 `TARGET_PADDING` 由
  `model_runner.compilation_manager.cache_loc_paddings[bs_idx]` 提供,
  消除魔法常数 16384。

### F7. cache_loc / page_indices bucket 含 spec 余量
修正 `_compute_cache_loc_buckets`:
```
cache_loc_bucket = bs_bucket * ceil_to_page(max_req_len)
                 + bs_bucket * ceil_to_page((spec_steps + 1) * draft_token_num)
```
确保所有 verify / 多步 draft 的 cache_loc 都落在已 trace 形状内。

### F8. 引入 cache-miss 诊断开关
- env var `SGLANG_JAX_EXPLAIN_CACHE_MISSES=1` 时,
  在 `ModelRunner.__init__` 调用 `jax.config.update("jax_explain_cache_misses", True)`;
- 把 `count_pjit_cpp_cache_miss` 包裹范围从 `sample(...)` 扩展到
  `tp_worker.forward_batch_generation` 全 forward;
- 落实 `compilation_manager.py:329` 的 `register_variant_if_new`
  调用点(forward 前 hook),首次出现的 variant key 打日志,
  便于线上回归侦测。

---

## 实施顺序与验证

1. **F8(诊断先行)** —— 让后续每条修复都能用统一的方法量化效果。
2. **F1 + F2(覆盖)** —— 先把缺失的 trace 补上,
   预期能消除「shape 本身没出现过」类型的 miss。
3. **F3 + F4(padding 对齐)** —— 解决「形状在 padded_bs 与 real_bs 间漂移」类。
4. **F5 + F6 + F7(细节常数)** —— 消除魔法值和 fallback 桶。

每步完成后跑:

```bash
SGLANG_JAX_EXPLAIN_CACHE_MISSES=1 python3 -m sgl_jax \
  --model Qwen/Qwen3-32B --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path AngelSlim/Qwen3-32B_eagle3 \
  --speculative-eagle-topk 1 --speculative-num-steps 2 \
  --speculative-num-draft-tokens 3 ...
```

预期:precompile 完成后,前 N 条请求(N 覆盖多种 bs/prompt_len 组合)
的 forward cache_miss_count 从「数十~上百」降到 0,
日志中无 `jax_explain_cache_misses` 输出。

---

## 涉及文件

- `python/sgl_jax/srt/managers/scheduler.py:433-441`
- `python/sgl_jax/srt/model_executor/compilation_manager.py`(buckets + precompile loops)
- `python/sgl_jax/srt/speculative/eagle_worker.py:80-366`
- `python/sgl_jax/srt/speculative/eagle_draft_worker.py:233-457`
- `python/sgl_jax/srt/speculative/eagle_util.py:523-845`
- `python/sgl_jax/srt/layers/attention/flashattention_backend.py:195-399`
- `python/sgl_jax/srt/utils/common_utils.py:38-97`
- `python/sgl_jax/srt/managers/tp_worker.py:388-429`(cache_miss 包裹范围)

## 待澄清

- `draft_model_runner.forward` 的 jit 边界 / cache key 细节未读;
  F2 实施前需要确认它确实按 `(num_tokens, bs)` 形状变化做 cache key,
  否则需要在 model 层补 `static_argnames`。
- 是否所有调用都已确保「padding token 不申请 KV cache」需要随 F3/F4
  联动复核(用户 memory: padding token 不申请 KV cache,表述要精确)。
