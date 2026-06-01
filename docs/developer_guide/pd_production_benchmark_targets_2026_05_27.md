# PD Production Benchmark Targets (2026-05-27)

这份文档只记录 **production benchmark / eval 入口** 的当前验证状态和下一阶段目标。

---

## 1. 当前已验证

### 1.1 单入口 router/proxy

当前验证通过的入口：

- `GET /get_server_info`
- `GET /get_model_info`
- `GET /v1/models`
- `POST /generate`
- `POST /v1/chat/completions`

其中：

- native `/generate` 通过 router fan-out 到 P/D
- OpenAI `/v1/*` 路径已经补齐 PD 字段透传：
  - `rid`
  - `disagg_transfer_id`
  - `bootstrap_host`
  - `bootstrap_port`
  - `bootstrap_room`

### 1.2 已跑通的 benchmark / eval smoke

已确认可跑通的最小 production-path 验证：

- `bench_serving`
  - `16 prompts`
  - `512 input`
  - `8 output`
  - peak concurrency `16`
- `run_eval.py gsm8k`
  - `10 examples`
  - `10 threads`

说明：

- 这些结果证明 **production benchmark/eval harness 已经接通**
- 它们不代表已经覆盖到目标生产容量

---

## 2. 下一阶段目标

下一阶段 benchmark 目标至少覆盖：

- `64` 并发
- `16k` input tokens
- `1k` output tokens

建议把验证拆成三层：

### 2.1 Capacity smoke

目标：先证明单入口路径在目标规格下能跑完，不追求最终吞吐最优。

建议起点：

- concurrency `8 -> 16 -> 32 -> 64`
- input length `4k -> 8k -> 16k`
- output length `256 -> 512 -> 1k`

### 2.2 Stress benchmark

目标：在接近生产规格时观察：

- request success rate
- transfer failure count
- engine crash / disconnect
- TTFT / E2E 尾延迟
- sustained throughput

### 2.3 Eval under load

目标：确认 eval harness 在有并发时仍然稳定，不只是在单线程 smoke 下可用。

建议：

- `gsm8k` 先做 `10 -> 50 -> 100 examples`
- threads `1 -> 4 -> 10 -> 32`
- 在每个点记录：
  - success/failure
  - total latency
  - backend 是否出现 `KVReceiver failed`
  - backend 是否出现 runtime disconnect

---

## 3. 当前建议

下一轮不要直接从最大点开跑。先按下面顺序扩：

1. `bench_serving`: `16 concurrency, 4k input, 256 output`
2. `bench_serving`: `32 concurrency, 8k input, 512 output`
3. `bench_serving`: `64 concurrency, 16k input, 1k output`
4. `run_eval.py gsm8k`: `50 examples, 10 threads`
5. `run_eval.py gsm8k`: `100 examples, 32 threads`

这样更容易把：

- router/proxy wiring 问题
- PD request lifecycle 问题
- backend runtime capacity 问题

分层定位。
