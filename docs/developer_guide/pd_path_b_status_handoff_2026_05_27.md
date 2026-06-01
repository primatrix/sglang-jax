# PD Path-B Status Handoff (2026-05-27)

这份文档用于把 **当前 PD path-B 调试现状** 直接交给另一个 agent。  
目标不是重复完整历史，而是给出：

- 当前分支与工作树状态
- 当前测试规格
- 已确认的事实
- 已知的无效/脏实验
- 下一步该怎么继续

如果只看一份文档开始工作，就看这份；需要补充背景再看文末列出的旧文档。

---

## 1. 当前范围

本轮只聚焦：

- **`path B`**
- `cache off` / `cache on` 两种模式
- 目标是把 **PD path-B 稳定跑起来**

明确 **不在本轮范围**：

- `path A` / D2H staging
- 重新设计 KV 传输架构
- 泛化到其它模型或拓扑

---

## 2. 当前分支与工作树

- 分支：`epic/pd-disaggregation`
- 当前工作树里与本轮直接相关的改动：
  - `python/sgl_jax/srt/disaggregation/decode.py`
  - `python/sgl_jax/srt/disaggregation/prefill.py`
  - `python/sgl_jax/srt/disaggregation/debug_utils.py`
  - `python/sgl_jax/test/disaggregation/test_pd_mixin_event_loop.py`
  - `python/sgl_jax/test/disaggregation/test_pd_debug_utils.py`

本地 CPU 回归状态：

```bash
UV_NO_CONFIG=1 /Users/jiongxuan/Library/Python/3.9/bin/uv run \
  --no-project --isolated --python 3.13 \
  --with "sglang-jax[cpu] @ file:///Users/jiongxuan/workspace/sgl-jax/python" \
  --with pytest \
  python -m pytest /Users/jiongxuan/workspace/sgl-jax/python/sgl_jax/test/disaggregation -q
```

最新结果：`186 passed`

---

## 3. 当前测试规格

### 3.1 拓扑

当前一直使用的是 **TP=4**，不是 TP=1。

- 集群：`v6e-16`
- P：`jx-v6e-16-0-hf2dt`
- D：`jx-v6e-16-1-2kdng`
- Bootstrap：独立进程，跑在 P pod 上 `10.31.173.56:8998`

当前 pod IP：

- P pod `jx-v6e-16-0-hf2dt` → `10.31.173.56`
- D pod `jx-v6e-16-1-2kdng` → `10.31.175.54`

### 3.2 单 pod TPU 形态

每个 pod 都强制覆写成 **单 host 4 chip**：

```bash
TPU_HOST_BOUNDS=1,1,1
TPU_TOPOLOGY=2x2
TPU_WORKER_ID=0
TPU_TOPOLOGY_WRAP=false,false,false
TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
TPU_WORKER_HOSTNAMES=$(hostname).jx-v6e-16-headless-svc
```

所以这里测试的是：

- **单 host**
- **TP=4**
- **AxisType.Explicit**

不是多 host 大拓扑。

### 3.3 模型与关键参数

- 模型：`Qwen3-8B`
- `--tp-size 4`
- `page_size=128`
- `mem_fraction_static=0.88`
- mesh：`("data", "tensor") = (1, 4)`，`AxisType.Explicit`
- KV pool sharding：`P("data", None, "tensor", None, None)`

### 3.4 请求方式

请求通过本地 `port-forward` 同时 fan-out 到 P 和 D 的 `/generate`：

- P 本地转发：`127.0.0.1:30100 -> pod0:30100`
- D 本地转发：`127.0.0.1:30200 -> pod1:30200`

每次请求透传：

- `bootstrap_host`
- `bootstrap_port`
- `bootstrap_room`

参考入口：

- `python/sgl_jax/srt/disaggregation/tools/e2e/_common.py`

---

## 4. 已确认的事实

### 4.1 正确性 bug 的根因已经确认

之前的 silent corruption **不是**：

- P gather 本身错误
- transfer 错误

而是 D 侧 `_write_kv_to_pool()` 的 padded writeback 逻辑：

- `num_pages < padded_pages` 时
- 代码重复了最后一个真实 page 的 **page id**
- 但没有重复最后一个真实 page 的 **payload**
- stale padded tail 覆盖了最后一个真实 page

修补位置：

- `python/sgl_jax/srt/disaggregation/decode.py`

关键回归用例：

- `test_write_kv_to_pool_keeps_last_real_page_when_bucket_is_padded`

### 4.2 `path B + cache off` 是健康 baseline

已经拿到过 P/D 输出一致的代表性 case：

- `1251`
- `1637`
- `2501`
- `3900`

### 4.3 `path B + cache on` 也拿到过完整通过样本

已经拿到过 P/D 输出一致的 case：

- `1251`
- `1637`
- `2501`
- `3900`

包括：

- P 使用 `JAX_COMPILATION_CACHE_DIR`
- P 重启后
- 首个 `1637` 请求

也出现过 **PASS** 的样本。

### 4.4 旧文档里的 gather OOM，本轮没有重新稳定复现

目前没有重新建立出一个稳定的：

- path-B gather HBM OOM
- 只在 `bucket=16` 稳定炸

所以当前更像的 blocker 不是 “gather 一定 OOM”，而是下面这个控制面问题。

### 4.5 当前剩余的主要不确定点是控制面/lifecycle

P 日志里反复出现：

```text
ZmqPullNotifier received uuid=... with no registered callback; dropping
```

这条 warning 的特点：

- 有时出现，**请求仍然成功**
- 有时伴随失败/挂住

所以当前最准确的表述是：

- **存在控制面/生命周期时序异常**
- 但它不是“每次必失败”的单一错误

---

## 5. 已知脏实验 / 不要再依赖的结论

下面这些实验结果不要再当作当前事实：

### 5.1 “写回后强制同步”的候选补丁

曾经试过两种候选：

- `jax.block_until_ready(tuple(kv_pool.kv_buffer))`
- 对写入页做 readback 再 `block_until_ready`

这两条都在现场把 baseline 污染了，出现过：

- `SocketServer: Connection closed recv() == 0`

这些候选都已经从本地代码**回退**。  
如果你看到旧日志里有这类错误，不要直接把它们当作当前代码状态的证据。

### 5.2 本地 `port-forward` 失效导致的假失败

至少出现过一次：

- 30100 本地转发已断
- 本地请求侧看到 `ConnectError`

这不是 PD 本体结论，只是环境问题。

### 5.3 “cache on/off 就是根因”这个表述不成立

更准确的是：

- `cache on/off` 会改变 precompile / 首请求时序
- 它更像 **时序放大器**
- 不是数据语义根因本身

---

## 6. 当前静态分析方向

### 6.1 控制面参考：`../sglang`

建议重点看：

- `../sglang/python/sglang/srt/disaggregation/common/conn.py`
- `../sglang/python/sglang/srt/disaggregation/prefill.py`
- `../sglang/python/sglang/srt/disaggregation/decode.py`

目前已经能看出的方向：

- `sglang` 比我们更显式地维护 request lifecycle
- 它有更完整的：
  - `request_status`
  - `failure_records`
  - `prefill_response_tracker`
  - `required_prefill_response_num_table`
  - timeout / stage queue 语义

这说明我们当前的 `no registered callback` 更像生命周期管理薄弱，不像数据面错误。

### 6.2 数据面参考：`../tpu-inference`

建议重点看：

- `../tpu-inference/tpu_inference/offload/tpu_offload_connector.py`
- `../tpu-inference/tpu_inference/offload/utils.py`

目前已经能看出的方向：

- 它的 update path 更强调：
  - bucketed gather / bucketed update
  - cached sharding spec
  - replicated indices
- 目前没有看到直接支持“compile cache 改变数据语义”的证据。

---

## 7. 当前运行中的服务

当前集群内进程：

- P：
  - `/opt/venv/bin/python -u -m sgl_jax.launch_server ... --port 30100 --disaggregation-mode prefill`
- D：
  - `/opt/venv/bin/python -u -m sgl_jax.launch_server ... --port 30200 --disaggregation-mode decode`
- Bootstrap：
  - `/opt/venv/bin/python -u -m sgl_jax.srt.disaggregation.run_bootstrap --host 0.0.0.0 --port 8998`

注意：

- 本地 `port-forward` 是**会话级**状态，不要假设一定还活着
- 接手时应重新确认 `30100/30200` 本地转发是否有效

---

## 8. 接手建议：下一步怎么做

不要先跑大矩阵。  
建议按下面顺序：

1. **先继续静态分析控制面**
   - 对照 `../sglang` 把 lifecycle 差异再收窄一层
   - 目标：把 “no registered callback” 缩成一个窄假设

2. **再看数据面，但只用于排除**
   - 用 `../tpu-inference` 确认这里更像控制面 race，而不是 gather/update 设计本身错误

3. **只为那个窄假设做最小复现**
   - 优先小 probe / 轻量测试
   - 不要直接先跑整套 e2e

4. **拿到窄假设后再改代码**
   - 改动应尽量落在：
     - callback 注册/移除时序
     - ack 生命周期
     - 首请求/重启后 request stage 边界

5. **最后再回归 path-B 代表性矩阵**
   - `1251 / 1637 / 2501 / 3900`

---

## 9. 建议先读的旧文档

如果需要补背景，建议按这个顺序：

1. [pd_path_b_robustness design](/Users/jiongxuan/workspace/sgl-jax/docs/superpowers/specs/2026-05-26-pd-path-b-robustness-design.md)
2. [pd_gather_oom_conclusion.md](/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/pd_gather_oom_conclusion.md)
3. [pd_epic_ops_log.md](/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/pd_epic_ops_log.md)
4. [pd_epic_handoff.md](/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/pd_epic_handoff.md)

---

## 10. 最短总结

当前最准确的一句话：

> path-B 的正确性主 bug 已修；cache-off baseline 健康，cache-on 也拿到过完整通过样本；当前尚未收口的是一个控制面/生命周期时序异常，典型信号是 P 侧 `no registered callback` warning。
