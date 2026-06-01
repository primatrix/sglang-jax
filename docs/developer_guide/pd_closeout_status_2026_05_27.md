# PD Closeout Status (2026-05-27)

> **Primary entrypoint:** This document is now superseded as the first-read handoff by [pd_master_handoff_2026_05_28.md](/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/pd_master_handoff_2026_05_28.md). Keep this file as a detailed closeout/reference note.

这份文档用于收尾当前一轮 PD 分离工作，回答 4 件事：

1. 当前测试是怎么做的
2. 现在已经完成到什么程度
3. 还有哪些工作没有覆盖/没有完成
4. 下一步如何按功能重构当前分支上的 commit

本文档只描述 **截至 2026-05-27 当前工作树和远端验证结果**，不推导未来实现。

---

## 1. 当前代码状态

当前工作分支：

- `epic/pd-disaggregation`

当前有两个重要 checkpoint：

- `7cbb3346`
  - `fix: harden pd transfer lifecycle control plane`
- `a94948fd`
  - `fix: wire pd router and openai benchmark paths`

其中：

- `7cbb3346` 主要是 control-plane / req lifecycle / terminal record / notifier 分类
- `a94948fd` 主要是 production-like 单入口 router/proxy、OpenAI `/v1/*` 路径 PD 字段透传、benchmark/eval 接入

当前工作树仍然是 **dirty** 的。

原因不是代码坏，而是：

- 还有未整理进历史 commit 的 PD 代码
- 还有不少中间 handoff / ops / matrix / research 文档
- 下一步需要相对 `main` 按功能重组 commit，而不是直接沿当前提交历史往上堆

---

## 2. 当前测试方式

### 2.1 本地 CPU 单测

主要跑：

```bash
UV_NO_CONFIG=1 /Users/jiongxuan/Library/Python/3.9/bin/uv run \
  --no-project --isolated --python 3.13 \
  --with "sglang-jax[cpu] @ file:///Users/jiongxuan/workspace/sgl-jax/python" \
  --with pytest \
  python -m pytest /Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/test/disaggregation -q
```

当前结果：

- `214 passed`

本地单测当前覆盖的主要类别：

- sender / receiver / notifier 状态机
- terminal record / retired late ack 分类
- bootstrap register / lookup
- tokenizer / request plumbing
- prefill / decode mixin event loop
- padded writeback correctness
- router / proxy helper
- OpenAI `/v1/*` 的 PD 字段透传

参考：

- [pd_test_coverage_matrix_2026_05_27.md](/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/pd_test_coverage_matrix_2026_05_27.md)

### 2.2 远端语义验证

当前远端验证拓扑：

- 单 host per pod
- TP=4
- `path-B`
- `page_size=128`
- 模型 `Qwen3-8B`

当前主要 remote probe 方式：

- 直接打 P / D `/generate`
- 通过单入口 router 打 `/generate`
- 通过单入口 router 打 `/v1/chat/completions`
- `bench_serving`
- `test/srt/run_eval.py`

### 2.3 production-like benchmark/eval 入口

当前已经有 single-entry router/proxy，可供：

- native `/generate`
- `/v1/completions`
- `/v1/chat/completions`

相关代码：

- [launch_router.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/disaggregation/launch_router.py)
- [mini_lb.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/disaggregation/mini_lb.py)
- [mini_lb_helpers.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/disaggregation/mini_lb_helpers.py)

---

## 3. 当前已完成的工作

### 3.1 正常路径 control-plane 已收敛

当前已经完成的 req lifecycle / control-plane 工作包括：

- `disagg_transfer_id` 独立于 `rid`
- sender / receiver terminal contract
  - `abort()`
  - `failure_exception()`
  - `clear()`
- notifier 对 retired late ack / truly unknown ack 的分类
- manager 侧 terminal record 保留与新 attempt 清理

结论：

- 当前 normal path 下，req lifecycle 管理已经不再是主要 blocker

### 3.2 P prefill-only 语义已生效

当前 phase-1 语义已经改成：

- `P` 只做 prefill
- `P` 成功后返回空 completion
- `D` 负责真实 completion

同时：

- PD 模式下 overlap 已关闭

结论：

- 当前正常路径验证已经符合“P prefill-only，D decode”的阶段目标

### 3.3 单入口 production-like proxy 已可用

当前已确认可用：

- `GET /get_server_info`
- `GET /get_model_info`
- `GET /v1/models`
- `POST /generate`
- `POST /v1/chat/completions`

同时已补齐：

- router 自动注入共享 `rid/disagg_transfer_id`
- router 自动注入 `bootstrap_host/bootstrap_port/bootstrap_room`
- OpenAI `/v1/*` 路径把上述 PD 字段透传到底层 `GenerateReqInput`

### 3.4 benchmark/eval harness 已接通

当前已确认：

- `bench_serving` 可以通过单入口 router 跑通
- `run_eval.py gsm8k` 可以通过 `/v1/chat/completions` 跑通

已跑通的 smoke：

- `bench_serving`
  - `16 prompts`
  - `512 input`
  - `8 output`
  - peak concurrency `16`
- `run_eval.py gsm8k`
  - `10 examples`
  - `10 threads`

注意：

- eval 分数不应当被当成 PD 正确性指标
- 这轮 `gsm8k` 分数偏低，主要是 reasoning 风格长、`max_tokens=512` 容易把最终 `Answer:` 截掉
- 从返回文本看，PD 路径下回复语义本身是正常的

### 3.5 当前 benchmark 边界已经测出第一条容量 cliff

当前已验证的 benchmark 结果：

- 通过：
  - `8 并发 / 4k input / 256 output`
  - `16 并发 / 2k input / 128 output`
- 失败：
  - `16 并发 / 4k input / 128 output`
  - `16 并发 / 4k input / 256 output`

OOM 根因已经收敛：

- 不是 router/proxy
- 不是 req lifecycle
- 不是 transfer pull 本身
- 是 D 侧 `_write_kv_to_pool()` 的 scatter 写回 HBM OOM

关键位置：

- [decode.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/disaggregation/decode.py:338)
- [decode.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/disaggregation/decode.py:470)
- [decode.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/disaggregation/decode.py:577)

---

## 4. 当前未覆盖 / 未完成的工作

### 4.1 path-A 没有接通

当前真正跑的是 `path-B`：

- `P HBM -> D HBM`

`path-A` 也就是：

- `P HBM -> P host -> ... -> D HBM`

还没有接通。

更准确地说：

- scheduler 现在显式传 `host_pool=None`
- 开 `--disaggregation-enable-d2h` 会在启动时直接报错
- 当前 host pool 的 buffer shape 还是旧 token-major 契约，和现在 page-bucketed fused payload 不一致

所以 path-A 不是“没测”，而是“当前代码层面尚未 plumbed”。

### 4.2 `16k input` 当前实现还不成立

当前 gather/scatter page bucket 只有：

- `{1, 2, 4, 8, 16, 32, 64}`

在 `page_size=128` 下：

- `4k input` -> `32 pages`
- `8k input` -> `64 pages`
- `16k input` -> `128 pages`

所以：

- 当前 bucket 设计还没有覆盖 `16k input`
- 要做 `16k input`，至少要补一个 `128-page bucket`

### 4.3 生产容量目标还远未完成

目标规格是：

- `64` 并发
- `16k` input
- `1k` output

而当前第一条容量 cliff 已经在：

- `16 并发 / 4k input / 128 output`

出现。

所以：

- production-path harness 已接通
- production capacity 还远没 close

### 4.4 failure-path 还没有纳回

本轮明确没继续做：

- abort/kill/chaos
- peer hard crash 恢复
- in-flight seamless failover

这些仍应当放在下一阶段。

### 4.5 router/proxy 还是 minimal production-like，不是 final production router

当前 router/proxy 是可用于 production-path benchmark/eval 的最小入口，但还不是最终生产 router：

- 还没有完整多 P / 多 D policy routing
- 还没有 health-aware routing
- 还没有 streaming path
- 还没有完整 failure retry / membership management

---

## 5. 当前 benchmark OOM 的分析结论

当前动态 payload 的核心量纲是 page，不是 token。

当前配置：

- `layer_num = 36`
- global KV heads `= 8`
- TP `= 4`
- per-rank KV heads `= 2`
- `head_dim = 128`
- `dtype = bf16`
- `page_size = 128`

近似量纲：

- 每 token、每层、每 rank：`1024 B`
- 每 page、所有层、每 rank：约 `4.5 MiB`
- 每 page、所有层、全 TP 聚合：约 `18 MiB`

所以：

- `2k input` -> `16 pages` -> 约 `288 MiB aggregate`
- `4k input` -> `32 pages` -> 约 `576 MiB aggregate`

当前 OOM 并不是因为 raw pulled payload 本身太大，而是 D 侧 scatter install 的瞬时 HBM 放大。

日志里实际看到：

- `jit_scatter` 额外想申请约 `673.66 MiB`
- 当时 free HBM 只有 `492 MiB ~ 627 MiB`

因此当前根因可以概括为：

> `4k prompt` 对应的 `32-page` KV writeback 在 `16` 并发下会触发 D 侧 scatter install HBM 不足。

---

## 6. 当前建议的收尾结论

本轮可以认为已经 close 的内容：

- normal-path req lifecycle / control-plane
- P prefill-only 阶段语义
- production-like 单入口 router/proxy
- benchmark/eval harness 接入
- `/v1/*` OpenAI 路径 PD 字段透传

本轮不能认为已经 close 的内容：

- path-A
- 大容量 production benchmark
- `64 并发 / 16k input / 1k output`
- failure-path hardening

---

## 7. 下一步 commit 重构计划

下一步不是继续写代码，而是：

- 先备份当前分支/工作树
- 再相对 `main` 按功能重构 commit

### 7.1 备份建议

建议至少做一个：

- 本地备份分支
- 或者额外 worktree
- 或者 patch / bundle 备份

目的是：

- 重构 commit 期间可以安全地 reset / cherry-pick / 重排
- 避免当前 dirty worktree 上的中间文档和实验脚本丢失

### 7.2 5 个 commit 的目标拆分

按你的要求，下一步建议整理成 **5 个 commit**：

1. `Stage 1`
   - transfer foundation / host-pool-side-channel / lifecycle 基础
   - 主要放：
     - transfer wrapper / sender / receiver / notifier / host-pool 相关基础代码
     - 对应基础单测

2. `Stage 2`
   - scheduler e2e / prefill-only / tokenizer passthrough / bootstrap
   - 主要放：
     - prefill/decode mixin
     - tokenizer bootstrap passthrough
     - prefill-only 语义
     - normal-path e2e 能跑通所需代码

3. `Stage 3`
   - router / proxy / multi-host routing 接口
   - 主要放：
     - `launch_router.py`
     - `mini_lb.py`
     - `mini_lb_helpers.py`
     - router helper tests

4. `Other functional code`
   - 不属于 stage1-3 主线，但属于当前必要功能修复
   - 建议放：
     - OpenAI `/v1/*` 的 PD 字段透传
     - benchmark/eval 路径兼容性修复
     - 其它不适合硬塞进 stage1-3 的功能代码

5. `Docs only`
   - 只放中间记录类文档
   - 包括但不限于：
     - handoff
     - feature matrix
     - test coverage matrix
     - benchmark target / closeout / ops log / path-B status

### 7.3 当前建议的执行顺序

建议下一轮按这个顺序执行：

1. 备份当前分支
2. 以 `main` 为基线列出当前所有功能代码 diff
3. 按上面的 5 类先做文件分组
4. 再决定：
   - 哪些直接 `cherry-pick`
   - 哪些需要 `reset + add -p`
   - 哪些文档留到最后一个 docs commit

### 7.4 与 `pd_stage123_agent_prompt.md` 的关系

当前重构目标和 [pd_stage123_agent_prompt.md](/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/pd_stage123_agent_prompt.md) 是一致的：

- 按 stage 拆
- 不把中间记录文档和功能代码混在一起
- 给后续 review 一个清晰的 stage 边界

但实际重构时要注意：

- 当前分支上不只有 stage1-3 的主线代码
- 还混有 benchmark/eval/router/OpenAI 路径接入等“配套功能修复”

所以才需要额外的第 4 个 `other functional code` commit。

---

## 8. 一句话总结

当前工作已经把 **PD normal-path + production-like benchmark/eval 入口** 打通，并把第一条容量 cliff 收敛到了 **D 侧 `_write_kv_to_pool()` scatter install OOM**；下一步应先备份代码，再把当前分支相对 `main` 按 **Stage1 / Stage2 / Stage3 / other functional / docs** 五个 commit 重构清楚。
