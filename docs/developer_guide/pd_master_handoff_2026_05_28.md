# PD Master Handoff (2026-05-28)

这份文档是当前 PD（Prefill-Decode 分离）工作的 **主入口 handoff**。  
如果后面要继续接手、review、拆 commit、做 benchmark、排查容量问题，**先读这份**。

它的目标是把之前分散在多份 handoff / closeout / matrix / RFC 文档中的高信号内容收敛到一个地方，只保留必要的引用，不再要求接手人先读很多份零散文档。

---

## 1. 一句话结论

当前 `sgl-jax` 的 PD 正常路径已经达到下面这个状态：

- **normal-path req lifecycle / control-plane 已基本收敛**
- **P prefill-only、D 负责真实 completion 的阶段语义已落实**
- **single-entry production-like router/proxy 已接通**
- **`bench_serving` 与 `run_eval.py` 已能走真实单入口路径**
- **第一条容量 cliff 已经明确收敛到 D 侧 `_write_kv_to_pool()` 的 scatter install OOM**

因此：

- 可以认为 **router/proxy 和 production-path harness 这一层已经打通**
- 但不能认为 **production capacity 已达标**

---

## 2. 当前分支与 commit 结构

当前主开发分支：

- `epic/pd-disaggregation`

相对 `main..HEAD` 的当前 commit 链：

1. `cdd42424`
   - `feat(disaggregation): Stage 1 — transfer foundation + host pool + side channel`
2. `a8098934`
   - `feat(disaggregation): Stage 2 — bootstrap + scheduler mixin + prefill-only contract`
3. `ecc30004`
   - `feat(disaggregation): Stage 3 — multi-host routing + 内置 mini_lb proxy`
4. `69ee1b05`
   - `feat(disaggregation): production hardening tools + OpenAI /v1 PD passthrough + e2e matrix`
5. `8446f323`
   - `docs(disaggregation): RFC suite + closeout + ops + research`

额外重要历史 checkpoint：

- `7cbb3346`
  - `fix: harden pd transfer lifecycle control plane`
- `a94948fd`
  - `fix: wire pd router and openai benchmark paths`

这两个 checkpoint 的功能都已经被上面的 5 个 commit 链吸收，只是它们仍然是理解历史演进的关键节点。

---

## 3. 当前代码状态总览

### 3.1 Stage 1：底层 transfer / host pool / side channel

主要内容：

- `JaxTransferWrapper`
- `KVManager` / `KVSender` / `KVReceiver` / `KVPoll`
- `JaxTransferKVManager`
- `ZmqPullNotifier`
- `QueueHostKVPool`
- metrics skeleton
- PD 相关 `ServerArgs` 基础字段

当前结论：

- **path-B（direct from HBM）已走通**
- **path-A（D2H staging）代码壳存在，但没有 scheduler plumbing**

### 3.2 Stage 2：bootstrap + scheduler + prefill-only

主要内容：

- `BootstrapServer`
- `BootstrapClient`
- `SchedulerDisaggregationPrefillMixin`
- `SchedulerDisaggregationDecodeMixin`
- tokenizer/bootstrap 字段 passthrough
- prefill-only response contract
- PD 模式下禁用 overlap

当前结论：

- **P prefill-only / D decode 的阶段语义成立**
- **normal-path e2e 成立**

### 3.3 Stage 3：multi-host routing + single-entry proxy

主要内容：

- `router_args.py`
- `launch_router.py`
- `mini_lb.py`
- `mini_lb_helpers.py`

当前结论：

- 这层已经不是“纯手写的一套最小代理”
- 当前更准确的定位是：
  - **upstream-shaped launcher + MiniLB**
  - **small local patch layer**

当前保留的本地 patch 主要只有：

- 自动注入共享 `rid` / `disagg_transfer_id`
- 自动注入 `bootstrap_host` / `bootstrap_port` / `bootstrap_room`
- 对 backend `/get_server_info` / `/get_model_info` 与 upstream
  `/server_info` / `/model_info` 的 alias 兼容

### 3.4 Stage 4 当前实际落地点

虽然 Stage 4 hardening 没有完整交付，但已经有一部分代码先进入了主线：

- OpenAI `/v1/*` 的 PD 字段透传
- operator-side e2e / stress / sweep / chaos 工具
- benchmark/eval 的实际接入验证

这里面要注意：

- OpenAI `/v1/*` passthrough 现在和 upstream 已经是**同构行为**
- `tools/e2e/`、`pair_stress.py`、`run_pd_e2e_matrix.sh` 这些更多是
  **TPU pod / kubectl / operator-side glue**
- 不适合再为了“像 upstream”去强行改成 upstream fixture 原样

---

## 4. 当前真实数据路径

### 4.1 当前已验证的是真正的 path-B

当前 benchmark / eval / single-entry proxy 走的真实数据路径是：

```text
router/control-plane
  -> P prefill
  -> 从 P 静态 KV pool gather 出单请求 KV tensor
  -> jax transfer 直接从 P HBM pull 到 D HBM
  -> D scatter 回自己的静态 KV pool
  -> normal decode
```

也就是说，当前真正跑通的不是：

```text
P HBM -> P host -> D host -> D HBM
```

而是：

```text
P HBM -> D HBM
```

### 4.2 path-A 当前为什么还不能算“已完成”

当前 `QueueHostKVPool` 还停留在旧 token-major 契约：

- `buffer_shape=(max_tokens_per_buffer, layer_num, kv_head_per_rank, head_dim)`

而 prefill 真实产出的 payload 已经是 page-bucketed fused tensor：

- `(layer_num, padded_pages, page_size, ...)`

所以 path-A 当前不是“只差把 flag 打开”，而是：

- **shape contract 本身就不一致**
- 再加上 scheduler 现在显式传 `host_pool=None`

结论：

- path-A 目前不应当被当作“部分支持”
- 它是**未真正 plumbed**

---

## 5. 当前测试与验证方式

### 5.1 本地 CPU 回归

主命令：

```bash
UV_NO_CONFIG=1 /Users/jiongxuan/Library/Python/3.9/bin/uv run \
  --no-project --isolated --python 3.13 \
  --with "sglang-jax[cpu] @ file:///Users/jiongxuan/workspace/sgl-jax/python" \
  --with pytest \
  python -m pytest /Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/test/disaggregation -q
```

当前结果：

- `220 passed`

这套测试当前已覆盖：

- sender / receiver / notifier 状态机
- bootstrap register / lookup
- tokenizer/bootstrap 字段 passthrough
- prefill/decode mixin
- padded writeback correctness
- router/launcher/helper
- OpenAI `/v1/*` 的 PD passthrough
- operator-side `_common.py` / `pair_stress` helper contract

### 5.2 远端 smoke

当前远端验证拓扑：

- single-host-per-pod
- TP=4
- model=`Qwen3-8B`
- `page_size=128`
- `path-B`

已经确认可用的单入口 endpoint：

- `GET /get_server_info`
- `GET /get_model_info`
- `GET /v1/models`
- `POST /generate`
- `POST /v1/chat/completions`

### 5.3 benchmark / eval harness

已经接通：

- `bench_serving`
- `test/srt/run_eval.py`

smoke 结果：

- `bench_serving`
  - `16 prompts / 512 input / 8 output / peak concurrency 16`
- `run_eval.py gsm8k`
  - `10 examples / 10 threads`

eval 分数目前不能当成 PD 正确性指标，因为：

- 当前 reasoning 风格较长
- `max_tokens=512` 时常把最终 `Answer:` 截断

从文本抽样看，PD 路径下的回复语义本身是正常的。

---

## 6. 当前已完成的内容

可以认为已经 close 的工作：

- normal-path req lifecycle / control-plane
- prefill-only 阶段语义
- single-entry production-like router/proxy
- benchmark/eval harness 接入
- OpenAI `/v1/*` 的 PD 字段透传
- commit 链已经重构成 5 个更可 review 的 commit

其中，commit reviewability 额外修过两点：

- Stage 1 补了自己的 `ServerArgs/CLI` smoke test
- commit4 补了 e2e tool helper smoke test

---

## 7. 当前未完成的内容

### 7.1 生产容量目标未完成

目标规格：

- `64` 并发
- `16k` input
- `1k` output

当前第一条容量 cliff 已经在：

- `16 并发 / 4k input / 128 output`

出现。

### 7.2 `16k input` 当前实现层面也还没成立

当前 page bucket 只有：

- `{1,2,4,8,16,32,64}`

在 `page_size=128` 下：

- `4k` -> `32 pages`
- `8k` -> `64 pages`
- `16k` -> `128 pages`

所以当前 bucket 设计本身还没有覆盖 `16k input`。

### 7.3 failure-path hardening 还没纳回

这一轮没有重新纳回：

- abort / kill / chaos
- in-flight failover
- rolling upgrade 实测
- path-A

---

## 8. 当前容量边界与 OOM 根因

### 8.1 当前已测边界

通过：

- `8 并发 / 4k input / 256 output`
- `16 并发 / 2k input / 128 output`

失败：

- `16 并发 / 4k input / 128 output`
- `16 并发 / 4k input / 256 output`

### 8.2 根因位置

失败不在：

- router/proxy
- req lifecycle
- transfer pull

而在 D 侧：

- `process_decode_queue()`
- `_write_kv_to_pool()`
- `jit_scatter`

关键代码点：

- [decode.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/disaggregation/decode.py:338)
- [decode.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/disaggregation/decode.py:470)
- [decode.py](/Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/srt/disaggregation/decode.py:577)

### 8.3 量纲

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

但真正触发 OOM 的不是 raw payload，而是 D install scatter 的瞬时 HBM 放大：

- `jit_scatter` 想申请约 `673.66 MiB`
- 当时 free HBM 只有 `492 MiB ~ 627 MiB`

所以当前容量 cliff 的一句话总结是：

> `4k prompt` 对应的 `32-page` KV writeback 在 `16` 并发下会触发 D 侧 scatter install HBM 不足。

---

## 9. router/proxy 这一层的最终判断

### 9.1 当前是否还需要大改

判断：**不需要再做大规模重写。**

原因：

- commit3 现在已经明显更接近 upstream `sglang_router`
- 它已经收敛成：
  - upstream-shaped `RouterArgs`
  - upstream-shaped launcher
  - upstream-shaped `MiniLoadBalancer`
  - 少量本地 patch

### 9.2 commit4 是否还要再追求更像 upstream

判断：**没必要强行继续重写。**

原因：

- OpenAI `/v1/*` passthrough 现在已经和 upstream 同构
- 剩下的大头是：
  - `tools/e2e/_common.py`
  - `pair_stress.py`
  - `chaos.sh`
  - `run_pd_e2e_matrix.sh`
- 这些本质上是 TPU pod / `kubectl` / operator-side glue
- 不适合追求 upstream fixture 原样

---

## 10. 当前建议的下一步

如果继续做功能工作，最值得做的是：

1. **decode-side capacity fix**
   - 重点围绕 `_write_kv_to_pool()`
   - 方向包括：
     - chunked scatter / page-chunk install
     - install 节流
     - 更大 page bucket 设计

2. **补 `128-page bucket`**
   - 为 `16k input` 做实现层准备

3. **再谈 path-A**
   - 先统一 host buffer shape 契约
   - 再接 scheduler plumbing

如果继续做文档/交接工作，建议：

1. 后续接手人先读本文档
2. 再按需查：
   - [pd_test_coverage_matrix_2026_05_27.md](/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/pd_test_coverage_matrix_2026_05_27.md)
   - [pd_feature_support_matrix_2026_05_27.md](/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/pd_feature_support_matrix_2026_05_27.md)
   - [pd_production_benchmark_targets_2026_05_27.md](/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/pd_production_benchmark_targets_2026_05_27.md)
   - [pd_closeout_status_2026_05_27.md](/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/pd_closeout_status_2026_05_27.md)
   - [pd_epic_handoff.md](/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/pd_epic_handoff.md)

---

## 11. 文档索引建议

为了减少碎片化，推荐把文档理解成两层：

- **主入口**
  - 本文档：[pd_master_handoff_2026_05_28.md](/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/pd_master_handoff_2026_05_28.md)

- **参考文档**
  - closeout / 状态矩阵 / benchmark 目标
  - RFC 设计文档
  - ops log / deployment / 验证记录

也就是说，后续 handoff 时不要再说“先读很多份”，而是：

> 先读 `pd_master_handoff_2026_05_28.md`，其余文档按需查。

---

## 12. 一句话总结

当前工作已经把 **PD normal-path + production-like benchmark/eval 单入口** 打通，并把当前主要 blocker 收敛到了 **D 侧 `_write_kv_to_pool()` scatter install OOM**；后续如果继续开发，应把重点从 router/proxy 转到 **decode-side capacity fix + 16k input bucket 扩展**。
