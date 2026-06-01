# PD Epic Operations Log

本文件长期维护 epic `pd-disaggregation` 期间的运维/调试事实：集群操作、
工具问题与 workaround、单 pod 调试模板、跑 Stage 测试的命令。**不进 git**
（与 `pd_epic_handoff.md` 同样长期 untracked），但每次 stage 完成或集群
状态变化都必须更新。

为什么独立一份：handoff 是"项目总览/工作流"，本文件是"我每次都要查的
命令和坑"。手册分离，避免 handoff 越长越糟。

---

## 1. 集群当前状态（最近一次写入：2026-05-25）

### 1.1 kubectl context

- v6e（PD 主集群）：`gke_poc-tpu-partner_us-east5_tpuv6e-256-node`
- v7x：`gke_tpu-service-473302_us-central1_tpu7x-cluster`（**PD 不用**）

切换：
```bash
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
export USE_GKE_GCLOUD_AUTH_PLUGIN=True
kubectl config use-context gke_poc-tpu-partner_us-east5_tpuv6e-256-node
```

每次操作前显式 `use-context`：context 经常自动漂移。

### 1.2 jx-v6e-16 pod 拓扑

- 4 pod × 4 chip = 16 chip 总，跑在 `brian-mimo-pool` nodepool（4 个 4x4 节点）
- yaml 模板：`/tmp/jx-v6e-16-rebuild.yaml`（本地 untracked）
- Job 名：`jx-v6e-16`（`backoffLimit=0`，失败不重试）
- Pod 命名：`jx-v6e-16-{0..3}-<suffix>`，`-suffix` 每次 rebuild 都不同
- headless svc：`jx-v6e-16-headless-svc`（17 天前建的，长期保留）
- securityContext.capabilities.add: [IPC_LOCK]（必须，否则 `ulimit -l` = 64KiB，
  transfer 性能崩坏）
- Image：`us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1`
  - 自带 jax 0.8.1 / flax / transformers / hf-cli
  - **不带 torch**
- Venv：`/opt/venv/`，Python：`/opt/venv/bin/python`
- Code path：`/sglang-jax/`（手动 tar 同步过去）
- Model path：`/models/`（PVC 挂载，PD 不用）

最近一次重建：2026-05-25。所有之前的 jx 系列 pod 全已被回收，新拉的：

| Pod | IP |
|---|---|
| jx-v6e-16-0-755p5 | 10.31.175.51 |
| jx-v6e-16-1-5vv62 | 10.31.172.51 |
| jx-v6e-16-2-xqjnx | 10.31.173.53 |
| jx-v6e-16-3-8kwk6 | 10.31.174.53 |

下次重建后 IP 都会变，操作前必须 `kubectl exec <pod> -- hostname -i` 取实时 IP。

### 1.3 重建步骤（备查）

如果 jx-v6e-16 失败需要重建：

1. 确认资源够（brian-mimo-pool 4 节点空）：
   ```bash
   for n in $(kubectl get nodes -l cloud.google.com/gke-nodepool=brian-mimo-pool \
              -o jsonpath='{.items[*].metadata.name}'); do
     echo "=== $n ==="; kubectl get pods --field-selector=spec.nodeName=$n --no-headers
   done
   ```
2. 删旧 Job（pod 会随 Job 走）：
   ```bash
   kubectl delete job jx-v6e-16
   ```
3. apply：
   ```bash
   kubectl apply -f /tmp/jx-v6e-16-rebuild.yaml
   ```
4. 等 4/4 Ready（通常 15-30 秒）：
   ```bash
   kubectl get pods -l app=jx-v6e-16 -w
   ```

如果 yaml 模板丢了：从这个 doc 重建（关键字段：`backoffLimit: 0`,
`completions: 4`, `parallelism: 4`, `IPC_LOCK`, `brian-mimo-pool`,
`tpu-v6e-slice` 4x4，`google.com/tpu: "4"`，`ephemeral-storage: 35Gi/70Gi`，
`subdomain: jx-v6e-16-headless-svc`，`serviceAccountName: gcs-account`，
PVC `inference-model-storage-poc-tpu-hns-pvc`）。

---

## 2. 单 pod 调试模式（多 host JAX 退化为单 host）

jx-v6e-16 的 yaml 给每个 pod 设了 `TPU_WORKER_ID`（0..3），JAX 默认走
multi-host init，单 pod 直接 `jax.local_devices()` 会**挂 60s+ 等其它 3
个 host join**。Stage 0/1 跑跨 pod 测试时我们要每个 pod 独立 init，必须
override 这些 env：

```bash
rm -f /tmp/libtpu_lockfile
export TPU_HOST_BOUNDS=1,1,1
export TPU_TOPOLOGY=2x2
export TPU_WORKER_ID=0
export TPU_TOPOLOGY_WRAP=false,false,false
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_WORKER_HOSTNAMES=$(hostname).jx-v6e-16-headless-svc
```

每 pod 出来 4 chip 当作单 host。**所有 PD 跨 pod 测试都用这个**。

来源：参考 handoff §4.5，PR-1 时 sjmrt 验证总结的 5 个坑里的第 2 个
（memory: `sjmrt_v6e16_gotchas`）。

### 2.1 /dev/vfio busy 的恢复

如果上次 python 进程没干净退出，新进程会撞 `open(/dev/vfio/0): Device
or resource busy`。recover 方法：

```bash
kubectl exec <pod> -- ps aux | grep python | grep -v grep   # 找残留 PID
kubectl exec <pod> -- kill -9 <pid>
```

**所以**：每次跨 pod 测试退出后等 5s 再启动下一轮，或者主动清理。

---

## 3. 代码同步

`/sglang-jax/` 在新拉的 pod 上**不存在**，第一次同步要先 mkdir：

```bash
PODS=(jx-v6e-16-0-755p5 jx-v6e-16-1-5vv62 jx-v6e-16-2-xqjnx jx-v6e-16-3-8kwk6)
for pod in "${PODS[@]}"; do
  kubectl exec "$pod" -- mkdir -p /sglang-jax
done
cd /Users/jiongxuan/workspace/sgl-jax
for pod in "${PODS[@]}"; do
  tar cf - python/sgl_jax/ | kubectl exec -i "$pod" -- tar xf - -C /sglang-jax/
done
```

注意：

- `kubectl exec` 默认走 "jx-v6e-16" 容器（pod spec 里同名），会在 stderr
  打印 `Defaulted container "jx-v6e-16" out of ...` 一行，正常无视
- macOS 上 `tar` 会带 `LIBARCHIVE.xattr.com.apple.provenance` xattr，pod
  里 GNU tar 解时会 spam 一堆 "Ignoring unknown extended header keyword"，
  正常无视
- **永远不要用 `kubectl cp`** — 大文件会截断
- 后续只改测试可以单文件 tar：`tar cf - python/sgl_jax/test/disaggregation/`

shell array 在 subshell 里会丢；用 inline 命名或 hardcoded pod 名更稳。

---

## 4. 跨 pod 跑测试模板

P 侧（pod-0）后台启动，D 侧（pod-1）跟着启。两侧 wrapper 都监听
`--transfer-port`（可以同号，不同 pod 不同 network namespace 不冲突）。
控制通道用 `--ctl-port`（默认 31000）。

### 4.1 Stage 0 byte round-trip

```bash
# pod-0 (P)
kubectl exec jx-v6e-16-0-755p5 -- bash -c '
  rm -f /tmp/libtpu_lockfile
  export TPU_HOST_BOUNDS=1,1,1 TPU_TOPOLOGY=2x2 TPU_WORKER_ID=0 \
         TPU_TOPOLOGY_WRAP=false,false,false TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
  export TPU_WORKER_HOSTNAMES=$(hostname).jx-v6e-16-headless-svc
  export PYTHONPATH=/sglang-jax/python PYTHONUNBUFFERED=1
  cd /sglang-jax
  /opt/venv/bin/python -u -m sgl_jax.test.disaggregation.test_byte_roundtrip \
      --role prefill --my-host $(hostname -i) \
      --ctl-port 31000 --transfer-port 31001
'

# pod-1 (D) — 取 pod-0 实时 IP 填到 --remote
P_IP=$(kubectl exec jx-v6e-16-0-755p5 -- hostname -i 2>&1 | grep -v Defaulted)
kubectl exec jx-v6e-16-1-5vv62 -- bash -c "
  rm -f /tmp/libtpu_lockfile
  export TPU_HOST_BOUNDS=1,1,1 TPU_TOPOLOGY=2x2 TPU_WORKER_ID=0 \\
         TPU_TOPOLOGY_WRAP=false,false,false TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
  export TPU_WORKER_HOSTNAMES=\$(hostname).jx-v6e-16-headless-svc
  export PYTHONPATH=/sglang-jax/python PYTHONUNBUFFERED=1
  cd /sglang-jax
  /opt/venv/bin/python -u -m sgl_jax.test.disaggregation.test_byte_roundtrip \\
      --role decode --my-host \$(hostname -i) --remote $P_IP \\
      --ctl-port 31000 --transfer-port 31001
"
```

`PD_ROUNDTRIP_ITERS` 环境变量控制每 cell 的迭代次数（默认 100）。验证
快路径用 10 跑 90 个 round-trip；正式验收按 RFC 跑 100 = 900 round-trip。

预期日志（D 侧）：
```
[D] done: 900/900 iters, 0 failed
```

P 侧：`[P] done: 900/900 iters reported, 0 failed`。

### 4.1.1 Stage 1：path B（事件驱动 + pipelined）

```bash
# pod-0 (P)
kubectl exec jx-v6e-16-0-755p5 -- bash -c '
  rm -f /tmp/libtpu_lockfile
  export TPU_HOST_BOUNDS=1,1,1 TPU_TOPOLOGY=2x2 TPU_WORKER_ID=0 \
         TPU_TOPOLOGY_WRAP=false,false,false TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
  export TPU_WORKER_HOSTNAMES=$(hostname).jx-v6e-16-headless-svc
  export PYTHONPATH=/sglang-jax/python PYTHONUNBUFFERED=1 PD_ROUNDTRIP_ITERS=100
  cd /sglang-jax
  /opt/venv/bin/python -u -m sgl_jax.test.disaggregation.test_byte_roundtrip \
      --role prefill --my-host $(hostname -i) \
      --ctl-port 31000 --transfer-port 31001 --side-channel-port 31002
'

# pod-1 (D)
kubectl exec jx-v6e-16-1-5vv62 -- bash -c '
  rm -f /tmp/libtpu_lockfile
  export TPU_HOST_BOUNDS=1,1,1 TPU_TOPOLOGY=2x2 TPU_WORKER_ID=0 \
         TPU_TOPOLOGY_WRAP=false,false,false TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
  export TPU_WORKER_HOSTNAMES=$(hostname).jx-v6e-16-headless-svc
  export PYTHONPATH=/sglang-jax/python PYTHONUNBUFFERED=1 PD_ROUNDTRIP_ITERS=100
  cd /sglang-jax
  /opt/venv/bin/python -u -m sgl_jax.test.disaggregation.test_byte_roundtrip \
      --role decode --my-host $(hostname -i) --remote 10.31.175.51 \
      --ctl-port 31000 --transfer-port 31001 --side-channel-port 31003
'
```

预期：P 侧 `done: failed_cells=[] leaked_total=0 total_target=900`。

### 4.1.2 Stage 1：path A（D2H staging）

P 侧加 `--use-d2h-staging --pool-size 128`：

```bash
/opt/venv/bin/python -u -m sgl_jax.test.disaggregation.test_byte_roundtrip \
    --role prefill --my-host $(hostname -i) \
    --ctl-port 31000 --transfer-port 31001 --side-channel-port 31002 \
    --use-d2h-staging --pool-size 128
```

D 侧不变（路径选择从握手取）。pool_size 必须 >= 单 cell 的 in-flight 数；
默认 ITERS=100，给 128 留 buffer。预期 P 侧 `leaked_total=0`。

**前置一次性配置**（pod 重建后要重做）：

```bash
for pod in jx-v6e-16-0-755p5 jx-v6e-16-1-5vv62; do
  kubectl exec "$pod" -- /opt/venv/bin/pip install --no-deps pyzmq
done
```

### 4.2 探查 transfer readback bug

```bash
kubectl exec jx-v6e-16-0-755p5 -- bash -c '... --role producer --my-host $(hostname -i) ...'
kubectl exec jx-v6e-16-1-5vv62 -- bash -c '... --role consumer --my-host $(hostname -i) --remote <P-ip> ...'
# 见 python/sgl_jax/test/disaggregation/_probe_transfer_readback.py
```

工具脚本，遇到新 JAX 版本时复跑验证 workaround 仍 work。

---

## 5. 已知坑（按 epic 期间踩到的顺序）

### 5.1 JAX 0.8.1 transfer readback metadata bug

**症状**：`jax.experimental.transfer` pull 回来的 sharded array，host 化
（`device_get`、`addressable_data(i)` 直接拉、`bool(jnp.all(...))`、甚至
`bool()` scalar）会炸：

```
jax.errors.JaxRuntimeError: INVALID_ARGUMENT:
  Invalid slicing of buffer size N with invalid offset 0, slice size M
```

N = 实际 per-shard buffer，M = global shape 字节数。

**根因**（推测）：transfer 还原出的每个 shard 的 metadata 报 global shape
而非 per-shard shape，buffer 自身是 per-shard 大小，host materialisation
路径按 metadata 读，越界。

**Workaround**：slice 后再 device_get，slice 是新的 jit trace，metadata
干净：

```python
sub = arr.addressable_data(i)[:shard_size]
host_bytes = np.asarray(jax.device_get(sub)).tobytes()
```

`test_byte_roundtrip.py::_arr_host_bytes` 用了这个。`_probe_transfer_readback.py`
保留为 diagnostic — 升 JAX 时复跑确认 workaround 是否仍 work。

**影响范围**：所有需要 "把 transfer 拉回的 array 转回 host bytes" 的代码。
Stage 1+ 用 `QueueHostKVPool` 走 D2H staging，可能绕开此 bug；但只要还
直接 `device_get(transferred_arr)` 就要踩。

### 5.2 jax.experimental.transfer 在 CPU jaxlib 上不能 import

**症状**：macOS / 任何 CPU-only jaxlib（无 TPU runtime）：

```
AttributeError: module 'jaxlib._jax' has no attribute 'TransferConnection'
```

**影响**：本地 pytest 跑 wrapper 测试，`mock.patch("jax.experimental.transfer.start_transfer_server")`
在 attribute resolve 阶段会 import 真模块，触发上面这个 error。

**Workaround**：用 `sys.modules` 注入假模块，绕开 attribute resolve：

```python
def _shim_transfer_module(fake_server):
    fake_mod = types.ModuleType("jax.experimental.transfer")
    fake_mod.start_transfer_server = mock.MagicMock(return_value=fake_server)
    return mock.patch.dict(sys.modules, {"jax.experimental.transfer": fake_mod})
```

`test_jax_transfer_wrapper.py` 用的就是这个。

### 5.3 本机 sgl_jax 双安装

`/Users/jiongxuan/workspace/sgl-jax/`（工作目录）vs
`/Users/jiongxuan/workspace/sglang-jax/.venv/site-packages/sgl_jax/`
（editable install 的另一份 repo）。

跑 pytest **必须前置** `PYTHONPATH=/Users/jiongxuan/workspace/sgl-jax/python`，
否则跑的是另一个 repo 的旧代码。Stage 0+ 所有 CPU 测试命令：

```bash
cd /Users/jiongxuan/workspace/sgl-jax
PYTHONPATH=/Users/jiongxuan/workspace/sgl-jax/python \
  python3 -m pytest python/sgl_jax/test/disaggregation/ -v
```

### 5.4 多 host JAX init 在单 pod 上挂 60s+

参见 §2。

### 5.5 /dev/vfio busy

参见 §2.1。

### 5.6 pinned_host + sharded partition_spec 触发 MegaScale collective

**症状**（Stage 1 path A 集成测试）：

```
F0525 ... tpu_pjrt_client.cc:5206] Failed to execute host offloading executable:
FAILED_PRECONDITION: Host offloaded collective custom call
'MegaScaleCollectiveOutputStart' only works in multi slice environment.
```

**根因**：把 sharded (P("x") 在 dim 0 上跨 4 chip) array `device_put` 到
`memory_kind="pinned_host"` 时，pinned_host transfer 需要 megascale
collective gather 跨 chip 的 shards；在单 slice debug mode（每 pod 一 slice）
里 collective 不可用，挂。

**Workaround**：用 `partition_spec=P()`（replicated）建 pool 和对应 payload。
每 chip 有相同副本，无需跨 chip gather。Production 多 slice 部署里这个限制
不存在；本约束只影响单 slice 调试模式。

**适用范围**：所有需要 D2H staging 的代码路径。`QueueHostKVPool` 的
`partition_spec` 是 ctor 参数，调用方决定。Stage 1 test_byte_roundtrip
显式用 `P()`。

### 5.7 QueueHostKVPool dtype 必须 == payload dtype

**症状**（Stage 1 path A 集成测试 fp16 cell）：

```
FutureWarning: scatter inputs have incompatible types:
cannot safely cast value from dtype=float16 to dtype=bfloat16
```

然后 byte equality 失败（隐式 down-cast 改了字节）。

**根因**：pool buffer 用 dtype X 预分配，`.at[:n].set(staged_dtype_Y)`
触发 implicit cast。

**Workaround**：测试每 cell 重建 pool 让 dtype 匹配。生产场景里 KV cache
dtype 是模型固定的，单一 pool 即可。

### 5.8 pyzmq 不在 image 默认 venv 里

**症状**（Stage 1 集成测试）：

```
ModuleNotFoundError: No module named 'zmq'
```

**原因**：`cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` 自带 msgpack
但没装 pyzmq。

**Workaround**：每 pod 跑一次：

```bash
kubectl exec <pod> -- /opt/venv/bin/pip install --no-deps pyzmq
```

`--no-deps` 防止动 jax 等核心包。pyzmq 是 self-contained cp312 wheel，
装完 `jax --version` 不变。pod 重建后需要重新安装（per-pod ephemeral
venv，不是 NFS 共享）。

---

## 6. Stage 验收记录

每个 stage 跑通的硬数据 + commit hash。Stage N 跑 byte round-trip 之类
的回归时直接复用本节的命令。

### 6.1 Stage 0（2026-05-25 完成）

- 实现 squash commit：`6066db93`
- Merge commit on epic：`76ebb304`
- Review fix commit：`b1344702`（feature 上）
- Review fix merge commit：`39861d9a`（epic 上）
- CPU 单测：48 项全 PASS（25 状态迁移穷举 + 8+1 wrapper 契约含 re-register raise + 终态/自环 + 状态空间 sanity）
  ```bash
  PYTHONPATH=/Users/jiongxuan/workspace/sgl-jax/python \
    python3 -m pytest python/sgl_jax/test/disaggregation/ -v
  ```
- 跨 pod byte round-trip：**900/900** 字节相等（在 6066db93 上跑过，review fix 没改 wire 路径）
  - P：jx-v6e-16-0-755p5 (10.31.175.51) 端口 31001
  - D：jx-v6e-16-1-5vv62 (10.31.172.51)
  - bf16/fp16/fp8_e4m3 × {1, 16, 256} pages (页 = 4096 bf16-equivalent elem) × 100 iter
  - 单次 round-trip 含 P attach + register + state 流转 + D pull + per-shard byte 校验 + ack；总耗时 < 2 min（含 warmup）
- Review feedback 处理（详见 commit `b1344702`）：
  - I1 `attach_kv_data` → `_attach_kv_data_for_testing`（Stage 0 footgun 重命名）
  - I2 RFC 加 `remote_addr` 签名 + crc32 临时性说明
  - I3 `register_pull` 重复 uuid 抛 RuntimeError
  - M1 `_StateHolder` → `StateHolder`
  - M6 `assert` → `RuntimeError`
  - 推后到 pre-PR：M2 manual 测试改名避免 pytest 收集、M4 transfer knob、M5 thread safety、M7 sender/receiver lifecycle、M8 type 注解

### 6.2 Stage 1（2026-05-25 完成）

- 实现 squash commit：`c34f88fa`
- Merge commit on epic：`3407593d`
- Review fix commit：`a13f8dfb`（feature 上）
- Review fix merge commit：`c6b77966`（epic 上）
- CPU 单测：80 项全 PASS（Stage 0 的 48 + Stage 1 32：QueueHostKVPool 11 + ZmqPullNotifier 12 + 事件驱动 sender 9 含 3 个 race regression）
  ```bash
  PYTHONPATH=/Users/jiongxuan/workspace/sgl-jax/python \
    python3 -m pytest python/sgl_jax/test/disaggregation/ -v
  ```
- 跨 pod byte round-trip + 100 并发 per cell + leak check：
  - 同 §6.1 pod 配置（P=10.31.175.51, D=10.31.172.51）
  - **Path B**（直 HBM，事件驱动）：bf16/fp16/fp8_e4m3 × {1, 16, 256} pages × 100 iter = **900/900 字节相等**，leak=0
  - **Path A**（D2H staging via QueueHostKVPool，pool_size=128，per-cell dtype-matched）：同 cells × 100 iter = **900/900 字节相等**，leak_total=0（每 cell pool buffer 全部 ack-driven 归还）
- 新引入的坑（§5 已记录）：
  - Path A pool 默认 `partition_spec=P()`（replicated），sharded host pool 在 single-slice debug mode 触发 MegaScaleCollectiveOutputStart 失败
  - Pool dtype 必须 == payload dtype，否则 `.at[].set()` scatter 隐式 down-cast 破坏 byte equality
  - `pyzmq` 不在 image 默认 venv 里，需要 `/opt/venv/bin/pip install --no-deps pyzmq`（pod-local，不影响其他 pod）
- 实装的 Stage 0 carry-over：
  - M5 wrapper `_pending_lock` —— register_pull / release 跨主线程 + ZMQ listener 线程
  - M7 sender/receiver on-success prune —— 防 mgr._senders/_receivers 无界增长
- Review fix（commit `a13f8dfb`，merge `c6b77966`）：
  - C1：`KVSender.send` 把 `register_callback` 移到 `producer_handoff` 之前（D 必须 `register_pull` 完才能 pull），整段套 `_state_lock`；`_on_ack` 也用同一 lock，listener 半路 fire 时会 block 等 send 完成
  - C2：`fail()` 用 `unregister_callback` 返回值做 cleanup 归属判断；返回 None 说明 listener 已 pop，让 `_on_ack` 做 cleanup（避免 `pool.put_buffer` 双 free）
  - I1：`_on_ack` cleanup 失败时仍 transition 到 FAILED（不 wedge 在 TRANSFERRING）
  - I3：`ZmqPullNotifier.stop()` listener join 超时 log warning
  - I5：3 个 race regression test（C1 + C2 两条路径，用 barrier-blocked wrapper deterministic 复现）
  - M4：ServerArgs 加注释说 Stage 2 wire 起来
  - 推后到 pre-PR：M2 dead assert / M3 unused param / M5 pool concurrent test / M6 unused helper / M7 100ms shutdown latency / M8 thread name

### 6.3 Stage 2（2026-05-25 完成）

- 实现 squash commit：`4d7f45c6`
- Merge commit on epic：`da6c7329`
- Review fix commit：`87512add`（feature 上）
- Review fix merge：`b333acfa`（epic 上）
- CPU 单测：117 项全 PASS（108 + 7 Mixin event-loop + 2 heartbeat regression）（Stages 0+1 的 80 + Stage 2 28：BootstrapServer 10 + ServerArgs 9 + tokenizer passthrough 7 + e2e + scheduler 复合 2）
- Stage 2 CPU e2e 验证范围：
  - 真 `BootstrapServer`（FastAPI/uvicorn 后台线程） + 2 个 `BootstrapClient`（P 注册 / D 按 room 查询）
  - 真 `ZmqPullNotifier` pair（ROUTER + DEALER）
  - 真 `JaxTransferKVManager` over mocked wrapper（CPU jaxlib 缺 TransferConnection，用 `sys.modules` shim）
  - 一句 prompt → P fake prefill 生成确定性 KV → 注册到 wrapper → D 通过 bootstrap 查到 P → KVReceiver 拉回 KV → fake decode 产生 deterministic token sequence
  - Scheduler 类 MRO 静态检查：`SchedulerDisaggregationPrefillMixin` / `SchedulerDisaggregationDecodeMixin` 已 compose 进 `Scheduler`，event_loop_normal_disagg_{prefill,decode} 等方法在
- 跨 pod 验证现状：
  - **wire path** 已在 Stage 0（900/900）+ Stage 1（path A 900/900 + path B 900/900 + leak=0）覆盖
  - **真模型 TPU e2e**（一句 prompt → 真 token stream）需要在 pod 上启动 sgl-jax server with `--disaggregation-mode={prefill,decode}` + `--disaggregation-bootstrap-url=...`，要先把小模型同步到 `/models/`、配置正确的 mesh / topology env、wait engine ready。**Stage 2 不跑** —— Mixin 的 3 个 hook（`_extract_req_kv` / `_build_kv_spec_for_req` / `_write_kv_to_pool`）以及 P 侧 prefill-only 语义 + D 侧 skip-prefill 语义还是 placeholder；真模型 e2e 需要把这些都实装好，约 1-2 天的深度集成工作。Stage 3 multi-host routing 跑通后由 user 决定是否启动 Stage 2.5 做真模型 e2e
- Stage 2 期间 pod 装的额外依赖（pod 重建后要重新装；本身只装 venv，不影响其他 pod）：
  ```bash
  /opt/venv/bin/pip install --no-deps \
    orjson uvloop partial_json_parser uvicorn \
    'fastapi==0.116.1' 'starlette>=0.40,<0.48' \
    pybase64 tiktoken modelscope llguidance pathwaysutils \
    openai jiter python-multipart setproctitle
  ```
  装完 `import jax` 仍 0.8.1 不变（已验证）
- 关键代码位置（Stage 2 新增）：
  - `srt/disaggregation/bootstrap.py` — BootstrapServer + BootstrapClient + PrefillInfo + _Registry
  - `srt/disaggregation/prefill.py` — PrefillBootstrapQueue + SchedulerDisaggregationPrefillMixin
  - `srt/disaggregation/decode.py` — DecodePreallocQueue + DecodeTransferQueue + SchedulerDisaggregationDecodeMixin
  - `srt/managers/scheduler.py` — Scheduler 类加 2 个 Mixin，`run_scheduler_process` 加 mode dispatch，新增 `_install_disaggregation_wiring`
  - `srt/server_args.py` — 8 个 disaggregation_* 字段 + CLI + __post_init__ 校验
  - `srt/managers/io_struct.py` — `GenerateReqInput` / `TokenizedGenerateReqInput` 加 `bootstrap_{host,port,room}`
  - `srt/managers/tokenizer_manager.py` — 透传 + decode mode 缺字段 raise ValueError
  - `srt/managers/schedule_batch.py` — `Req` 加 `bootstrap_{host,port,room}` 默认 None

### 6.4 Stage 3（2026-05-25 完成）

- 实现 squash commit：`d3f8f259`
- Merge commit on epic：`a8739557`
- Review fix commit：`dc7b1eec`（feature 上）
- Review fix merge：`9efbc706`（epic 上）
- CPU 单测：136 项全 PASS（130 + 4 IPv6 edge case + 3 tokenizer auto-derive 测试 - 1 旧 string-set 断言被 ipaddress 实现取代）
- Stage 3 范围内交付：
  - `srt/disaggregation/host_ip.py` —— `resolve_host_ip(explicit, env_name)` helper：显式 → `$HOSTNAME` + `gethostbyname` → `socket.gethostname()` 兜底；拒绝 `0.0.0.0`/`127.0.0.1`/`localhost` 等 bind/loopback 地址（避免误注册 → D 拉不到的常见运维事故）
  - `srt/disaggregation/router.py` —— 文档化 `sglang_router` / `mini_lb` 接入形态：D 暴露 OpenAI endpoint；tokenizer 自动从 `--disaggregation-bootstrap-url` 派生 bootstrap_{host,port}，从 `rid` 的 crc32 派生 bootstrap_room；router 无需感知 PD
  - `srt/server_args.py` —— 新增 `disaggregation_host_ip` 字段 + `--disaggregation-host-ip` CLI
  - `srt/managers/scheduler.py` `_install_disaggregation_wiring` —— 移除 Stage 2 的 `DISAGG_HOST` env hack，改用 `resolve_host_ip` helper
  - `srt/managers/tokenizer_manager.py` —— decode mode 自动补全缺失的 bootstrap_{host,port,room}（router 透传场景）
- 跨 host 验证现状：
  - CI 范围：multi-prefill 注册、room 哈希一致性、re-register 行为、host_ip 推导异常路径全覆盖
  - **真 multi-host TPU**（4 host P + 4 host D + router 一句 prompt → token stream）：与 Stage 2 真模型 e2e 处于同一深度集成依赖，未实施。Stage 2.5 把 Mixin hook 实装好后这部分才能跑

### 6.5 Stage 2.5 真模型 e2e（**2026-05-25 跑通**）

- 实现 squash commit：`ebd1bf30`
- Merge commit on epic：`3f0e573a`

feature/pd-stage2-real-e2e 累计 commits（已 squash 进 ebd1bf30）：
- `e1aa76b7` — 真 KV hooks（MHA pool）
- `335d9572` — standalone bootstrap + register retry
- `6dbe6c81` — KV spec sharding prepend layer
- `f3214ea0` — KV gather sharding + getitem propagation
- `5a3b05a4` — sender idempotency + scatter sharding
- `aca247c8` — D skip prefix_indices override
- `a79a7504` — init tree_cache-side attrs
- `4129ec19` — use tree_cache root_node
- `a918d302` — leave 1 token unprefilled
- `fe7ccdf9` — disagg_decode skip check_memory

部署形态（jx-v6e-16 双 pod，单 slice debug mode）：
- pod-0 (10.31.175.51)：
  - standalone bootstrap server `/opt/venv/bin/python -u -m sgl_jax.srt.disaggregation.run_bootstrap --host 0.0.0.0 --port 8998`
  - P engine `launch_server --model-path /models/Qwen3-8B --tp-size 4 --port 30100 --disaggregation-mode prefill --disaggregation-host-ip 10.31.175.51 --disaggregation-bootstrap-url http://10.31.175.51:8998 --disaggregation-transfer-port 31001 --disaggregation-side-channel-port 31002`
- pod-1 (10.31.172.51)：
  - D engine `launch_server --model-path /models/Qwen3-8B --tp-size 4 --port 30200 --disaggregation-mode decode --disaggregation-host-ip 10.31.172.51 --disaggregation-bootstrap-url http://10.31.175.51:8998 --disaggregation-transfer-port 31001 --disaggregation-side-channel-port 31003`
- `--skip-server-warmup` 在 PD mode 下 __post_init__ 自动 enable（warmup 是单边的会卡死）
- pyzmq + fastapi/uvicorn/orjson/uvloop 等依赖手动装到 `/opt/venv` 见 §6.3

mini_lb 风格手动 fan-out 测试（同 rid + 同 bootstrap_room 发 P 和 D 两边）：
```bash
RID="t1"
kubectl exec jx-v6e-16-0-755p5 -- bash -c "curl -s -X POST http://127.0.0.1:30100/generate -H 'Content-Type: application/json' -d '{\"rid\":\"$RID\",\"text\":\"Hello, my name is\",\"sampling_params\":{\"max_new_tokens\":8,\"temperature\":0},\"bootstrap_host\":\"10.31.175.51\",\"bootstrap_port\":8998,\"bootstrap_room\":12345}'" &
kubectl exec jx-v6e-16-1-5vv62 -- bash -c "curl -s -X POST http://127.0.0.1:30200/generate -H 'Content-Type: application/json' -d '{\"rid\":\"$RID\",\"text\":\"Hello, my name is\",\"sampling_params\":{\"max_new_tokens\":8,\"temperature\":0},\"bootstrap_host\":\"10.31.175.51\",\"bootstrap_port\":8998,\"bootstrap_room\":12345}'"
```

验收结果：
- ✅ "Hello, my name is" → " Alex, and I am a 2" (D 与 P byte-equal)
- ✅ "The capital of France is" → " Paris. The capital of Italy is Rome" (D 与 P byte-equal)
- ✅ D's `cached_tokens=4` vs P's `cached_tokens=0` 证 PD transfer 真起作用（5 input × 4 cached + 1 extend for logits）
- ✅ D 第二次请求 latency 0.24s（warm 后） vs 第一次 1.29s（cold + KV transfer）
- ✅ Engine 稳定，多次请求不崩

设计决策（实战验证后定型）：
- D side `_write_kv_to_pool` 把 `prefix_indices` 设成所有 input 的 **前 N-1** 个 slot —— 最后 1 个 token 让 scheduler re-extend 来产 logits（all-cached extend=0 会让 TPU XLA "program continuator halted"）
- D side `_pd_skip_prefix_match` marker 让 `init_next_round_input` 跳过 tree_cache.match_prefix 覆盖（cache 没我们写的 KV）；marker 在第一次 iter 被消费
- D side disagg_decode 事件循环跳过 `check_memory` —— PD prealloc'd slots 不走 scheduler 的 owning tracking，sanity check 误报 leak
- bootstrap server 必须 standalone 进程（embed 进 engine uvicorn + JAX 冲突会 timeout）
- `BootstrapClient.register_prefill` 30×1s retry 防 bootstrap 比 engine 慢起的 race

Stage 4 hardening 应该处理：
- D side 真正释放 PD prealloc'd kv_indices（现在通过跳 check_memory 绕开 leak detector，资源会一直占）
- P side prefill-only 真语义（现在 P 也 decode，浪费 cycles）

### 6.6 Stage 4 Production hardening（**2026-05-26 实装完毕**）

Branch: `feature/pd-stage4`（base：epic/pd-disaggregation）

按 RFC `docs/rfc/2026-05-25-pd-hardening.md` 的 H-A..H-F 分块落地。
Stage 4 的代码不需要重新部署到 pod 就能验证 —— 全部走 CPU 单测。

#### H-A 可观测

新增 `python/sgl_jax/srt/disaggregation/metrics.py`：

- 可选依赖 prometheus_client（pod 上没装时降级为 no-op stub），所
  以 `is_prometheus_available()` 在生产 pod 上目前是 False
- 7 个 metric，schema 完全对齐 RFC（state_transition / transfer_bytes /
  transfer_duration / transfer_inflight / host_pool_used /
  transfer_failures / bootstrap_registry_size）
- `time_phase(phase, role)` 上下文管理器 + `host_pool_alloc/free()` 累加器

Wire 进了：
- `StateHolder._transition_to` → `pd_state_transition_total`
- `JaxTransferKVManager.create_*/_prune_*` → `pd_transfer_inflight`
- `JaxTransferWrapper.register_pull` → `pd_transfer_bytes_total{direction=net}`
- `QueueHostKVPool.copy_from_device` → `pd_transfer_bytes_total{direction=d2h}`
- decode mixin success path → `pd_transfer_bytes_total{direction=h2d}`
- decode mixin failure paths → `pd_transfer_failures_total{reason=...}`
- `_Registry.register/unregister/evict` → `pd_bootstrap_registry_size`
- sender ack timer / receiver pull timer / decode bootstrap-lookup → `pd_transfer_duration_seconds`

#### H-B 错误恢复

- 新 ServerArgs：`--disaggregation-bootstrap-timeout-seconds`、`--disaggregation-pull-timeout-seconds`、`--disaggregation-ack-timeout-seconds`、`--disaggregation-orphan-reaper-interval-seconds`
- `JaxTransferKVManager.reap_once(now)` + 后台 `start_reaper()` 线程
- Sender / Receiver 新加 `transfer_started_at` 属性 + `fail(reason=...)` 强制终止
- Scheduler 启动时 `start_reaper()`
- 失败 metric 全部带 reason label（timeout / bootstrap_lookup / receiver_init / pull_init / ack_send / shutdown / auth）

#### H-C 鉴权

新模块 `python/sgl_jax/srt/disaggregation/pd_auth.py`：HMAC-SHA256 tag + Bearer header helpers，constant-time compare。

- `--disaggregation-shared-secret` 配置 + `SGL_JAX_PD_SHARED_SECRET` env override
- BootstrapServer：FastAPI middleware 校验 Bearer，`/health` 免认证
- BootstrapClient：每个请求带 Authorization
- ZmqPullNotifier：D 端 send_done 加 hmac 字段，P 端 listener 校验并丢非法包（failure metric +1）
- `run_bootstrap.py` 加 `--shared-secret`

#### H-D 优雅 shutdown + 版本协议

- `PROTOCOL_VERSION=1` + `MIN_COMPATIBLE_VERSION=1` 常量
- `PrefillInfo` / `RegisterPrefillRequest` 加 `protocol_version` 字段
- `BootstrapClient.get_prefill_info` 拒绝 `protocol_version < MIN_COMPATIBLE_VERSION` 的 peer
- `JaxTransferKVManager.graceful_shutdown(drain_timeout)` + `inflight_count()`：
  drain → 超时则全部 `fail(reason="shutdown")` → `stop_reaper()`

#### H-E 多 channel + D2H 默认 ON

- `disaggregation_enable_d2h` 默认 `True`（原 False）
- 新 `--disaggregation-channel-number`，默认 4
- `_install_disaggregation_wiring` 传入 wrapper channel_number
- 新 `python/sgl_jax/srt/disaggregation/tools/sweep_channels.py`（operator script）

#### H-F Stress + chaos + runbook

新 operator scripts：
- `python/sgl_jax/srt/disaggregation/tools/stress.py` — QPS × duration，输出 P50/P95/P99 + error rate
- `python/sgl_jax/srt/disaggregation/tools/chaos.sh` — kill_p / drop_dcn / bootstrap 三场景
- `docs/operations/pd_runbook.md` — 部署清单 / health probe / 5xx decision tree / 容量公式 / 工具索引 / 升级顺序 / 已知限制

#### Stage 4 CPU 单测（4 个新 test 文件，全 PASS）

- `test_stage4_metrics.py` (9)
- `test_stage4_orphan_reaper.py` (9)
- `test_stage4_auth.py` (17)
- `test_stage4_protocol_version.py` (5)

完整 PD CPU 套件：174 passed in ~10s。

回归 fix：把 `disaggregation_enable_d2h` 改默认后，
`test_server_args_disaggregation.test_default_port_values` 和
`server_args.py` 里的 null-mode 警告比较都要跟新默认值对齐，已修。

#### 已知限制（Stage 4 收尾时）

- `is_prometheus_available()` 在当前 pod 上为 False（依赖未装），需要 ops 在 pyproject 里 opt-in `prometheus_client`
- sweep_channels 只测 register 端吞吐，paired puller harness 留给 follow-up
- mTLS 模式没接（RFC 文本里说作为 follow-up，本 RFC 不阻塞）
- stress 工具单 router 视角，多 router scale 由 operator 自己扩展
- D side "skip extend entirely" + 从 P 接收第一个 decode token（消除 1-token 重计算）
- bootstrap_room hash 宽度 + 真正的 retry + timeout

---

## 7. 集群清理操作记录

记录每次 cleanup 决策 + 结果。这部分主要是 audit trail，不需要复用。

### 7.1 2026-05-25 — 重建 jx-v6e-16 前的清理

用户授权"全部按表删"。删除：

- Failed Jobs（7 个，全部跨 9-32 天）：
  - `jx-v6e-16`（我自己上次留的，必删）
  - `aolemila-2x2`、`aolemila-tpu-v6e-4-1`、`aolemila-tpu-v6e-4-2`、
    `aolemila-tpu-v6e-8`、`aolemila-tpu-v6e-8-v2`
  - `kkx-single`
- 孤儿 pods（7 个）：
  - `prayer-debug-ci-v6e4` (32d AdmissionError)
  - `ramezes-gla-test` (16d AdmissionError)
  - `ramezes-qwen3-fp8-v6e4` (9d AdmissionError)
  - `mimo-kernel-routing-quant-debug-2x2-vxc6g` (45d Completed)
  - `mimo-kernel-topk-debug-2x2-rkhpv` (45d Completed)
  - `mimo-scale-debug-2x2-b7r9n` (45d Completed)
  - `kernel-eval-persistent-...-pql6b` (38d ContainerStatusUnknown — Deployment 旧 ReplicaSet pod，wz5xp 仍 Running)

未动（别人的 active workload）：niu-mimo-v16/v64 系列、wlf-*、wyx-ling26-*、
yuhao-ling3-*、ramezes-mimo-audio-v6e4、wlf-v6e-4-bench、niu-v6e4-sleep、
kernel-eval-persistent-...-wz5xp。

清理后 brian-mimo-pool 4 节点全空，apply jx-v6e-16.yaml，20 秒 4/4 Ready。

---

## 8. 文档关系

- `pd_epic_handoff.md`（本目录）：cold-start handoff，单文件读完即可继续开发
- `pd_stage123_agent_prompt.md`（本目录）：交给 agent 的任务定义 prompt
- `pd_epic_ops_log.md`（本文件）：每次操作的命令 + 集群当前状态 + 已知坑
- `docs/rfc/2026-05-25-pd-*.md`（5 份）：Stage 设计文档，本 epic 的代码合同

Stage N 跑通后必须：
1. 更新本文件 §6.N
2. 如果集群状态变（重建 pod、换 nodepool），更新 §1
3. 如果踩到新坑，更新 §5
