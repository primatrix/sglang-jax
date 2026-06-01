# RFC: PD multi-host 与 routing — per-host transfer server + 内置 mini_lb

## 摘要

把 PD 从「同一 host 跑 P/D 两个进程」扩展到「跨 host 部署」：

- **per-host transfer server**：多 host 部署下每个 host 启一个独立的
  `JaxTransferWrapper` server（绑定 host 的对外 IP:port）；跨 host
  pull 字节正确，不互相阻塞。
- **内置 mini_lb 单入口 proxy**：sgl-jax 自带的 `launch_router.py` +
  `mini_lb.py` + `mini_lb_helpers.py`，在请求级别同时把请求发给一对
  (P, D) 后端，并自动注入共享的 `rid` / `disagg_transfer_id` /
  `bootstrap_host` / `bootstrap_port` / `bootstrap_room`，让一个
  prompt 走「proxy → (P, D) → token stream」整条请求链路。

> 历史变更：原 RFC 草稿计划直接复用 sglang 仓库里的 `sglang_router`。
> 当前实现不是“完全原样照搬 upstream”，但也不再是纯手写的一套
> launcher/proxy。当前代码已经把 `RouterArgs` / `launch_router` /
> `MiniLoadBalancer` 的整体结构收敛到接近 upstream `sglang_router`
> 的形状，只保留少量 `sgl-jax` 必需的本地 patch：
>
> - 自动注入共享的 `rid` / `disagg_transfer_id`
> - 自动注入 `bootstrap_host` / `bootstrap_port` / `bootstrap_room`
> - 对 backend `/get_server_info` / `/get_model_info` 与 upstream
>   `/server_info` / `/model_info` 的 endpoint alias 兼容
>
> 更准确地说，当前这层是：
>
> - **upstream-shaped launcher + MiniLB**
> - **small local patch layer**
>
> 而不是完全独立的一套自定义 router。它的限制如下：
>
> - 当前支持多个 `--prefill` / `--decode` 后端，但 pair 选择仍是
>   `random`，未做 health-aware policy routing；
> - 没有 health-aware routing；
> - streaming 代码路径已经按 upstream MiniLB 结构接入，但当前生产验证
>   仍然以 non-stream benchmark/eval 为主；
> - 没有 failure retry / membership management。

这是 PD 第一次具备 production-like 部署形态：可以接入
`bench_serving` 与 `test/srt/run_eval.py` 等真实压测/评估入口。完整的
生产路由器（多对多、health-aware、streaming）留到后续阶段。

## 当前验证快照（2026-05-28）

截至当前代码状态，Stage 3 / commit3-4 组合已经完成的 production-like
验证包括：

- 单入口 proxy 已确认可用：
  - `GET /get_server_info`
  - `GET /get_model_info`
  - `GET /v1/models`
  - `POST /generate`
  - `POST /v1/chat/completions`
- `bench_serving` smoke 已接通：
  - `16 prompts / 512 input / 8 output / peak concurrency 16`
- `run_eval.py gsm8k` smoke 已接通：
  - `10 examples / 10 threads`

同时，容量边界也已经测出第一条 cliff：

- 通过：
  - `8 并发 / 4k input / 256 output`
  - `16 并发 / 2k input / 128 output`
- 失败：
  - `16 并发 / 4k input / 128 output`
  - `16 并发 / 4k input / 256 output`

失败根因已确认不在 router/proxy，而在 D 侧
`process_decode_queue() -> _write_kv_to_pool()` 的 scatter 写回 HBM
阶段。也就是说：

> 当前 Stage 3 已经把 production-like 单入口路径打通，但还没有达到
> 最终容量目标。

## 在 PD 路线中的位置

```
            Transfer wrapper + connection ABC + single backend
              ↓
            Buffer + 带外侧通道
              ↓
            Bootstrap + scheduler integration (端到端)
              ↓
[本 RFC]    Multi-host + routing
              ↓
            Production hardening
```

scheduler RFC 已经让单 pod 端到端跑通；本 RFC 把"单 pod"放宽为
"多 host 部署"，并把入口流量从「直接 curl P」改成「走 router」。

后续 hardening RFC 在此基础上加 multi-channel、metrics、运维工具。

## 设计

### per-host transfer server

#### 问题

`JaxTransferWrapper` 是进程级单例（已定，RFC-2 ADR-2）。多 host TP
部署下：

- 同一个 P 角色由多个 host 上的进程共同承担（每个 host 一个进程，按
  ICI 内的 TP rank 切 attention）。
- 每个 host 需要独立的对外 IP:port，跨 host pull 才走 DCN 而不是
  本地 loopback。

#### 设计

`JaxTransferWrapper.__init__(host_ip, port, channel_number)` 的
`host_ip` 在多 host 部署里**必须**是本 host 的对外 IP，不是 `127.0.0.1`
或 `0.0.0.0`。具体做法：

1. ServerArgs 加 `--disaggregation-host-ip`，默认从环境推断
   （TPU pod 内通过 `socket.gethostbyname(socket.gethostname())` 或
   `HOSTNAME` 环境变量得到 in-cluster DNS）。
2. P 启动时把当前 host 的 `(host_ip, port, side_channel_port)` 注册
   到 bootstrap（已经是 scheduler RFC 的字段，本 RFC 只是确保多 host
   下每个 P 进程注册的是各自 host 的 IP）。
3. `BootstrapServer` 不变 —— 它已经能容纳多 P 注册。
4. D 拿到的 `PrefillInfo` 含完整 `(host_ip, port, side_channel_port)`，
   `JaxTransferWrapper.pull()` 直接走 DCN 到对应 host。

#### 端口分配

每个 host 的端口集合：

| 用途 | ServerArgs | 默认 |
|---|---|---|
| transfer server | `--disaggregation-transfer-port` | 30001 |
| ZMQ side channel | `--disaggregation-side-channel-port` | 9600 |
| bootstrap (集中式，只一台机器跑) | `--disaggregation-bootstrap-port` | 8998 |

多 host 部署里同一 host 的两个角色（极少见，仅开发场景）要求端口不
冲突，所以这三个字段都允许通过 CLI 覆盖。

### 内置 mini_lb 集成

#### 组件

放在 `python/sgl_jax/srt/disaggregation/`：

- `router_args.py` — 本地版 `RouterArgs`，字段与 parser 形状尽量贴近
  upstream `sglang_router.router_args.RouterArgs`，只保留当前 `sgl-jax`
  真正用到的子集。
- `launch_router.py` — CLI 入口，走
  `parse_router_args(...) -> launch_router(...) -> MiniLoadBalancer(...)`
  流程，结构对齐 upstream launcher。
- `mini_lb.py` — proxy 主体，整体结构对齐 upstream MiniLB：
  * `POST /generate` — 接收 prompt，把请求**同时**发给一对 (P, D)，
    并在请求体里注入共享的 `rid` / `disagg_transfer_id` /
    `bootstrap_host` / `bootstrap_port` / `bootstrap_room`。P 端走
    prefill-only 契约返回空 completion，D 端流回真实 token。proxy 把
    D 的响应原样返回给 client。
  * `POST /v1/chat/completions` / `POST /v1/completions` — OpenAI 兼容
    入口，先把 PD 字段透传到底层 `GenerateReqInput`，再走相同的「双
    发到 (P, D)」分发路径。
  * `GET /get_server_info` / `GET /get_model_info` / `GET /v1/models` —
    优先打 `sgl-jax` backend 的 `/get_*` endpoint，同时对上游
    `/server_info` / `/model_info` alias 做兼容 fallback。
- `mini_lb_helpers.py` — 字段注入、bootstrap_room 生成（uuid 截断到
  32-bit）、共享 `rid` / `disagg_transfer_id` 注入、请求体 marshalling
  的纯函数 helper。

#### 部署形态

```
                ┌──────────────────┐
                │     mini_lb      │   (单实例; sgl-jax 自带)
                └────────┬─────────┘
                         │ HTTP (双发到 P & D)
            ┌────────────┴────────────┐
            │                         │
         ┌──▼──┐                   ┌──▼──┐
         │  P  │                   │  D  │
         └──┬──┘                   └──┬──┘
            │ KV transfer (DCN)       │
            └─────────────┬───────────┘
                          │
                 ┌────────▼─────────┐
                 │ Bootstrap Server │   (集中式)
                 └──────────────────┘
```

当前 mini_lb 已经接受多个 `--prefill` / `--decode` 后端，并按
`random` policy 选取一对 `(P, D)` 做请求级 fan-out。这已经足以接
`bench_serving` 与 `run_eval.py`，但还不是最终生产路由器。完整的
多对多 policy、health-aware routing、streaming 生产验证、failure
retry / membership management 留到后续 hardening RFC。

#### 选 P 策略

第一版用 bootstrap server 内的 `bootstrap_room % len(prefills)` 哈希
（RFC-2 §6.2 已实装）—— 无负载感知，纯静态分布。

更高级的负载感知策略（如按 P 的 HBM 占用、in-flight transfer 数量）
留到 hardening RFC，与 metrics endpoint 一起做。

### 与现有 sgl-jax launch 脚本的兼容

- `null` 模式（无 PD）：launch 脚本不变。
- `prefill` / `decode` 模式：脚本加 `--disaggregation-mode` 与
  `--disaggregation-bootstrap-url` 即可，其他 TP / multi-host JAX
  init 参数保持原样。
- 多 host TP（P 角色 4 host × 4 chip）启动方式与现状一致 —— 所有 P
  进程在 4 host 上并发 launch，JAX 自动跨 host init；启动后每个进程
  各自注册 bootstrap，bootstrap 看到 4 个 `PrefillInfo`（每 host 一个）。

## 测试

### 单元测试（CI，CPU）

- `test_host_ip_resolution.py`
  - 没传 `--disaggregation-host-ip` 时从 `HOSTNAME` / `socket` 推导
    一致；推导失败 raise。
  - 显式传值时直接使用，不走推导。

- `test_multi_prefill_registration.py`
  - mock bootstrap：4 个 P 用各自 `(host_ip, port)` 注册，
    `list_prefills` 返回 4 项；`get_prefill_info(room)` 对相同 room
    返回固定 P，不同 room 哈希分布合理。

### 集成测试（手动，TPU）

- `test_cross_host_byte_correctness.sh`
  - 2 host × 2 pod 部署：host A 跑 P (4 chip)、host B 跑 D (4 chip)
    （或更复杂拓扑）。
  - 各 dtype / page count 跨 host pull 字节相等（沿用 Stage 0 测试
    矩阵）。
  - 2 个 D 同时各自 pull 不同 P 的 KV：两个 transfer throughput 之和
    不应低于单 transfer 的 1.5×（验证 DCN 没被两个 transfer 互相阻塞）。

- `test_router_e2e.sh`
  - 启动 mini_lb (`launch_router.py`) + 1 P + 1 D + 1 bootstrap。
  - 经 mini_lb 跑 `bench_serving`（16 prompts / 512 input / 8 output /
    并发 16）与 `run_eval.py gsm8k`（10 examples / 10 threads）。
  - 错误率为 0；token 流非空（eval 分数不作为 PD 正确性指标，参考
    closeout § 3.4）。

#### 单元测试（CI，CPU）

- `test_launch_router_args.py`
  - `RouterArgs` 风格 parser
  - `--pd-disaggregation` / `--mini-lb`
  - `--prefill URL [BOOTSTRAP_PORT]`
  - legacy `URL,BOOTSTRAP_PORT` 兼容
- `test_mini_lb_backend_fetch.py`
  - `/get_server_info` / `/get_model_info` 优先路径
  - `/server_info` / `/model_info` fallback
- `test_mini_lb_generate.py`
  - `/generate` 与 `/v1/*` 统一走 backend-forward path
  - 自动注入共享 `rid` / `disagg_transfer_id`
  - batch `text` / `input_ids` 请求支持
- `test_mini_lb_helpers.py`
  - `bootstrap_*` 注入
  - 共享 identity 注入 helper

跑通日志贴 PR description。
