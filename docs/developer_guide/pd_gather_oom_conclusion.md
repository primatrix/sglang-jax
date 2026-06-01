# PD KV Gather OOM 复现与调试结论

> 文档目的：把 PD prefill 端 KV gather 的 OOM / silent corruption 问题完整复现一遍，
> 把"现状 / 问题 / 调试方式 / 调试经验"四件事讲清楚。读完这一篇即可接手后续修复。

## 一、现状

### 1.1 配置基线

生产侧验证场景（v6e-16 单 host 4 chip 模式，pod-0 当 P，pod-1 当 D）：

| 项 | 值 |
|---|---|
| 模型 | Qwen3-8B |
| Mesh | `("data", "tensor")` = (1, 4)，**`AxisType.Explicit`** |
| KV pool sharding | `P("data", None, "tensor", None, None)` |
| `page_size` | 128（PD 模式 `__post_init__` 会强制 ≥128） |
| `max_total_num_tokens` | 689664 |
| `num_pages` | 5388 |
| `mem_fraction_static` | 0.88 |
| Bootstrap | 独立进程 `run_bootstrap --host 0.0.0.0 --port 8998` |
| Page bucket 集合 | `(1, 2, 4, 8, 16, 32, 64)` |

P/D 服务实际启动方式：每个 pod 内手动 export 单 host TPU env（`TPU_HOST_BOUNDS=1,1,1` /
`TPU_CHIPS_PER_HOST_BOUNDS=2,2,1` / `TPU_TOPOLOGY=2x2` / `TPU_WORKER_ID=0` /
`TPU_WORKER_HOSTNAMES=localhost`），再 `python -m sgl_jax.launch_server …`。

### 1.2 已经验证 OK 的部分

- 当前 prefill 路径走 [[python/sgl_jax/srt/disaggregation/prefill.py:108]] 的
  `_jit_gather_all_layers(buffers, page_indices, out_sharding)`：
  - 一个 `@jax.jit` 包整层列表，避免 per-layer cache 爆炸；
  - `page_indices` 用 `jax.device_put(..., NamedSharding(mesh, P(None)))`，
    保证落在 KV pool 同一个 mesh 上；
  - `out_sharding=P(None, *pool_pspec[1:])`：gather 维 replicated（因为 indices
    replicated），其余 KV 维度沿用 pool 自身的 pspec。
- bucket=1 / 2 / 4 / 8 / 32 / 64 这几档下，P/D 输出**逐字一致**，HBM 增长在理论值上下。

### 1.3 仍然有问题的部分

- **bucket=16 这一档独有故障**：首次触发会 80GB scratch alloc OOM；
  之后的同形请求虽然不再报 OOM，但 D 端解码出的内容与 P 端**确定性不一致**
  （cached_tokens 数字看起来正常，输出是另一段毫不相干的文本）。
- **`DISAGG_LAUNCH_BOOTSTRAP=1` 这条 in-process 路径有 health-check bind mismatch bug**，
  目前只能靠独立的 `run_bootstrap` 进程绕过去。
- **KV 抽取失败时 P 侧 logger 只 `logger.exception("failed to extract KV ...")` 后
  `continue`，HTTP 仍 200 OK**；D 侧 fallback 走 self-prefill 就把这个故障"消化掉"了，
  外部观测不到错误，只能看输出对比。

## 二、问题清单

### P-1：`bucket=16` 80GB scratch OOM（首发）

**触发条件**：prompt token 数落在 1010 ～ 2048 之间（即 `ceil(seqlen/128)` ∈ [9, 16]，
经 `_pad_to_page_bucket` 进位到 16）。第一次以 bucket=16 形状跑到
[[python/sgl_jax/srt/disaggregation/prefill.py:418]] `_jit_gather_all_layers(...)` 时，
XLA 报 `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 85899345920 bytes`
（≈80GB scratch）。

**异常之处**：bucket=1 / 2 / 4 / 8 / 32 / 64 都没事，单单 bucket=16 这一档要 80GB。
理论上 bucket=16 的 gather 输出大小是 16 pages × 128 × 8 × 2 × 128 × 2B ≈ 8MB / layer，
36 层 ≈ 288MB，离 80GB 差了 2 个数量级。说明 XLA 给 bucket=16 选了一个 fallback program。

### P-2：`bucket=16` silent KV corruption（再发）

第一次 OOM 之后再发同样长度的 prompt：

- P 端不再 RESOURCE_EXHAUSTED，请求 200 OK；
- D 端 register_pull / pull 都 SUCCESS，cached_tokens 数字与 prompt 长度匹配；
- **D 端解码输出和 P 端跑同一 prompt 的输出完全不同**，且每次都不同（不是稳定的错答案）。

强烈怀疑：XLA 在第一次 OOM 之后切了一个数值上不等价的 fallback 编译产物并缓存下来，
后续 bucket=16 请求都用这个错的 program；或者 D 侧 `_write_kv_to_pool` 在这个 bucket 下
有 stride 错位。两个假设都需要进一步验证。

### P-3：`DISAGG_LAUNCH_BOOTSTRAP=1` health-check bind mismatch

[[python/sgl_jax/srt/managers/scheduler.py:2143]] 把 `BootstrapServer` 的 host 设成
`bootstrap_host`（即本机 LAN IP），但 [[python/sgl_jax/srt/disaggregation/bootstrap.py:322]]
的 `_wait_until_ready` 永远去 ping `http://127.0.0.1:{port}/health`。
uvicorn 如果绑在非 127.0.0.1 上，loopback 这条就拒连，10 秒后 `did not become ready`。

**临时绕过**：始终用独立的 `python -m sgl_jax.srt.disaggregation.run_bootstrap
--host 0.0.0.0 --port 8998`，不要走 `DISAGG_LAUNCH_BOOTSTRAP=1`。

### P-4：KV gather 失败被静默吞掉

[[python/sgl_jax/srt/disaggregation/prefill.py:283-290]] 抓 `_extract_req_kv` 抛的所有异常，
打一行 `logger.exception(...)` 后 `continue`，请求本身仍走完，HTTP 200 OK，
D 侧 fallback 走 self-prefill 得到一个看似合理的回复。

**后果**：用户从 P 服务的 HTTP 接口看不出 KV 抽取已经失败，只能 grep 日志或者比对 P/D 输出。
对线上是个观测大坑。

## 三、复现矩阵

固定 P=`jx-v6e-16-0`、D=`jx-v6e-16-1`、bootstrap=pod-0:8998、`page_size=128`，
唯一变量是 prompt token 数：

| prompt tokens | seq pages | bucket | P/D 一致 | 备注 |
|---:|---:|---:|---|---|
| 5 / 26 | 1 | 1 | OK | 冷启动那次 bucket=1 编译耗时长 |
| 151 | 2 | 2 | OK | |
| 501 | 4 | 4 | OK | |
| 1001 | 8 | 8 | OK | |
| **1251 (首发)** | 10 | **16** | **不一致 + OOM** | 80GB scratch RESOURCE_EXHAUSTED，P-1 |
| **1251 (重试 ×3)** | 10 | **16** | **不一致，无 OOM** | silent corruption，P-2 |
| **1637** | 13 | **16** | **不一致 + OOM** | 同 bucket=16，P-1 |
| 2501 | 20 | 32 | OK | |
| 3501 | 28 | 64 | OK | |

bucket=16 这一档是**唯一**的故障档：上下相邻的 8 / 32 都正常。

## 四、调试方式（这一轮怎么定位的）

### 4.1 先用隔离脚本压 production-scale，把"是不是 PD 链路自己的问题"剥离掉

[[scripts/pd_gather_v5_compare.py]] 直接构造 `(NUM_PAGES=3400, LAYERS=36)` 大小的 KV pool，
用同一个 mesh + sharding 跑三个变体：

- **V5**：`[b.at[idx].get() for b in bufs]`，**不传 out_sharding**（tpu-inference 写法）；
- **V1a**：`out_sharding=P(None, None, None, None, None)`（全 replicated）；
- **V1b**：`out_sharding=P(None, None, "tensor", None, None)`（当前生产写法）。

跑出来：

- **V5 直接 raise**：`Use .at[...].get(out_sharding=) ... could not be resolved unambiguously`。
  在我们 `AxisType.Explicit` 的 mesh 下，JAX 0.8.1 会**强制要求**显式 out_sharding。
- V1a / V1b 都能跑，HBM 增量在理论值上下，scratch 不爆。

> 关键结论：**tpu-inference 之所以能不传 out_sharding 跑 gather，是因为他们用的是
> `AxisType.Auto`（默认）；我们 `mesh_utils` 默认 `use_explicit_sharding=True` →
> `AxisType.Explicit`。在 Explicit mesh 下没有 V5 那条路，必须走 V1b。**

这一步把"是不是写法学习了 tpu-inference 但漏了什么"这个怀疑彻底排掉。

### 4.2 隔离脚本不复现 → 一定是生产 PD 路径上其它东西在叠加

把 launch_server 真起起来按 prompt 长度扫 bucket，在 P 日志里 grep
`RESOURCE_EXHAUSTED` 和 `failed to extract KV`，在 D 日志里 grep
`Prefill batch.*cached_tokens`，再用 curl 对比 P/D 同 prompt 的输出文字。
矩阵就是 §3 那张表。

### 4.3 已经覆盖的隔离脚本

| 脚本 | 用途 |
|---|---|
| `scripts/pd_gather_v1a_test.py` | V1a 单变体压力 |
| `scripts/pd_gather_v1b_test.py` | V1b（=生产）单变体压力 |
| `scripts/pd_gather_v4_test.py` | 用 `addressable_shards` 做 per-chip local gather，绕开 collective |
| `scripts/pd_gather_v5_compare.py` | V5 / V1a / V1b 三选一对比 |
| `scripts/pd_transfer_test.py` | jax.experimental.transfer 端到端 |

bucket=16 在隔离脚本上**不复现**，只在 launch_server 链路上复现 —— 这正是 P-2 假设里
"上游某个 program 把 bucket=16 编译产物污染掉了"还需要进一步验证的点。

### 4.4 推荐排查路径（接手者参考）

1. **关闭 XLA persistent compilation cache** 跑一次 launch_server，看 bucket=16 是否还能在
   重启后稳定复现 OOM。能复现 → P-1 是 XLA program selection 问题；不能复现 → 是缓存污染。
2. 在 [[python/sgl_jax/srt/disaggregation/prefill.py:418]] 调用前后 dump
   `gather_pspec` / `page_indices.sharding` / `layer_buffers[0].sharding` 三件套，
   bucket=16 和 bucket=8 / 32 对比，看是否多出意外的 axis。
3. D 侧加一个 page-level checksum：写入前对每一页 KV 求一个 sum，写入后 readback 再求一次，
   不一致就直接抛 —— 用来把 P-2 的怀疑"D 端 stride 错位"和"P 端 program 错"分开。
4. P-4 那个 `try/except` 改成至少把请求标 fail / 在 metric 上 +1，不要静默 200。

## 五、调试经验（这一轮踩到的坑）

### 5.1 `launch_server` 不会自动覆盖容器多 host TPU env

GKE pod 的 env preset 通常是 `TPU_TOPOLOGY=4x4`、`TPU_HOST_BOUNDS=2,2,1`，进程会去等其他 3 个 pod
握手，表现为"TPU init 卡 60s+ 然后 abort"。**必须**在启动前显式 `export` 5 个 env：

```sh
export TPU_HOST_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_TOPOLOGY=2x2
export TPU_WORKER_ID=0
export TPU_WORKER_HOSTNAMES=localhost
```

隔离脚本里 `os.environ.setdefault(...)` **不够**，因为容器已经 set 过了；要先 `pop` 再 set，
见 [[scripts/pd_gather_v5_compare.py:18]]。

### 5.2 `DISAGG_LAUNCH_BOOTSTRAP=1` 不要用

P-3。改用独立 `run_bootstrap`：

```sh
python -m sgl_jax.srt.disaggregation.run_bootstrap --host 0.0.0.0 --port 8998 \
  > /tmp/bootstrap.log 2>&1 &
```

P / D 启动时不要带 `DISAGG_LAUNCH_BOOTSTRAP=1`；D 用 `--bootstrap-server-url
http://<pod-0-ip>:8998` 指过来。

### 5.3 kubectl exec 的 stdout 是 fully buffered

没 PTY 的时候子进程默认 fully buffered，容易看着像"卡住 20 分钟无输出"，实际上是 buffer 没刷。
两件事都得做：

```sh
PYTHONUNBUFFERED=1 python … 2>&1 | tee /tmp/xxx.log
```

`tee` 到磁盘的好处是另开一个 `kubectl exec -- tail -f /tmp/xxx.log` 就能稳定追踪。

### 5.4 「`cached_tokens` 数字对 ≠ KV 真的对」

P-2 这次最坑的就在这里：cached_tokens 看着完全正确（D 端确实跳过了 prefill），
但 D 解码出来的就是另一段文字。**只有 P/D 输出逐字 diff 才是 KV 真的对**。
在排查 PD 正确性时，永远把"P/D 同 prompt 同采样参数下 token 序列一致"作为唯一可信信号。

### 5.5 隔离脚本必须配 production-scale mesh & sharding

最初一版 `pd_gather_v5_compare.py` 只用 4-page / 4-layer 的小 pool，怎么跑都不爆。
扩到 `NUM_PAGES=3400 / LAYERS=36 / mesh ("data","tensor") AxisType.Explicit` 之后才暴露
"Explicit mesh 下 V5 路根本不存在"这件事。**复现 sharded gather 行为时，pool 形状和 mesh
都必须 production-scale，page 数小了 XLA 走另一个 program**。

### 5.6 `AxisType.Explicit` 是和 tpu-inference 最大的区别

`mesh_utils.py` 默认 `use_explicit_sharding=True`，意味着所有 sharded op 都要显式给
out_sharding；tpu-inference 用 `AxisType.Auto`（默认），可以省略 out_sharding 让 JAX 推。
看人家代码学写法的时候，**先确认 mesh axis_types 是不是一致**，不一致的话省略写法搬过来直接 raise。

---

## 附：当前留在 pod 上的状态

- pod-0：`launch_server`（P 模式，端口 30100）+ `run_bootstrap`（端口 8998）。
- pod-1：`launch_server`（D 模式，端口 30200）。
- 两边日志在 `/tmp/server.log`，bootstrap 日志在 `/tmp/bootstrap.log`。

接手者如果不想保留这套环境，三条 `pkill -f launch_server` / `pkill -f run_bootstrap`
就能清掉。复现矩阵里所有 prompt 在 [[scripts/pd_gather_v5_compare.py]] 同目录下另存了
`prompts/{1251,1637,…}.txt`，可直接 `curl -d @… http://...:30100/generate` 重跑。
