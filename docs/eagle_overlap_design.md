# Eagle Worker Overlap 适配设计方案

## 1. 背景

当前 Eagle worker 和 scheduler overlap 不兼容。本文档描述如何让 Eagle 支持 overlap 调度。

## 2. 现有 Overlap 机制回顾

### 2.1 普通 Overlap 流程

```
时间线:
─────────────────────────────────────────────────────────────────────────────
Scheduler线程:    [准备BatchN] [run_batch] [准备BatchN+1] [run_batch] [处理BatchN结果]
                     │            │              │            │            │
                     │            ↓              │            ↓            ↓
                     │     返回future_ids        │     返回future_ids   resolve真实结果
                     │      (负数占位符)          │      (负数占位符)
                     │                           │
Forward线程:         ────────────[执行BatchN forward]──────────[执行BatchN+1]───
                                    │                              │
                                    ↓                              ↓
                              产生真实token_ids              resolve future_ids
                              写入future_map                 替换为真实值
```

### 2.2 核心机制

1. `forward_batch_generation()` 返回负数 `future_token_ids`（占位符），不阻塞
2. Scheduler 用这些占位符继续组装下一个 batch 的 `input_ids`
3. Forward Thread 执行时：
   - 用 `resolve_future_token_ids()` 将 `input_ids` 中的负数占位符替换为真实值
   - 执行实际 forward
   - 用 `set_future_token_ids()` 将新产生的 `token_ids` 写入 map
4. `resolve_last_batch_result()` 阻塞获取上一批次的真实结果

## 3. Eagle 与普通 Overlap 的区别

| 维度 | 普通 Overlap | Eagle Overlap |
|------|-------------|---------------|
| **Future Token IDs** | 固定 1 个/request | 1~N 个/request，根据 accept_index |
| **KV Cache 申请** | 固定 +1/request | 根据 accept_length 动态决定 |
| **Hidden States** | 不需要 | 需要作为 future 数据传递给下一轮 draft |
| **Spec Info** | 不需要 | 需要传递 EagleDraftInput (topk_p, topk_index 等) |

## 4. Eagle Overlap 需要的 Future 数据

| Future 数据 | 类型 | 说明 |
|-------------|------|------|
| `future_token_ids` | `(bs, max_draft_tokens)` | 被接受的 tokens（变长，1~N） |
| `future_accept_length` | `(bs,)` | 每个 request 接受了多少 token，用于确定 KV cache 申请量 |
| `future_draft_input` | `EagleDraftInput` | 包含 hidden_states, topk_p, topk_index, verified_id 等 |

## 5. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Scheduler 线程                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  [准备BatchN]──→[forward_batch_generation]──→[准备BatchN+1]──→[process结果N] │
│       │                   │                        │                │       │
│       │          返回future占位符                   │         resolve真实值   │
│       │     (token_ids, accept_len, draft_input)  │                │       │
└───────┼───────────────────┼────────────────────────┼────────────────┼───────┘
        │                   │                        │                │
        │              input_queue                   │           output_queue
        │                   ↓                        │                ↑
┌───────┼───────────────────────────────────────────┼────────────────┼───────┐
│       │           Forward 线程                     │                │       │
├───────┼───────────────────────────────────────────┼────────────────┼───────┤
│       │   [resolve future]──→[eagle forward]──→[update future maps]        │
│       │         │              (黑盒)                    │                  │
│       │         ↓                                        ↓                  │
│       │   future_xxx_map ◄─────────────────────── 写入新结果               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 6. 数据结构设计

### 6.1 Future Maps

```python
class EagleWorkerClient:
    def __init__(self, ...):
        max_reqs = self.max_running_requests
        max_draft_tokens = server_args.speculative_num_draft_tokens
        topk = server_args.speculative_eagle_topk
        hidden_size = model_config.hidden_size

        # ============ Future Maps ============

        # 1. Token IDs - 变长，预留 max_draft_tokens 空间
        #    shape: (max_reqs * max_draft_tokens,)
        #    存储方式: [req0_tokens..., req1_tokens..., ...]
        self.future_token_ids_map = jnp.zeros(
            max_reqs * max_draft_tokens, dtype=jnp.int32
        )

        # 2. Accept Length - 每个 request 接受了多少 token
        #    shape: (max_reqs,)
        self.future_accept_length_map = jnp.zeros(max_reqs, dtype=jnp.int32)

        # 3. Draft Input 组件
        #    hidden_states: (max_reqs, hidden_size)
        self.future_hidden_states_map = jnp.zeros(
            (max_reqs, hidden_size), dtype=jnp.bfloat16
        )
        #    topk_p: (max_reqs, topk)
        self.future_topk_p_map = jnp.zeros(
            (max_reqs, topk), dtype=jnp.bfloat16
        )
        #    topk_index: (max_reqs, topk)
        self.future_topk_index_map = jnp.zeros(
            (max_reqs, topk), dtype=jnp.int32
        )
        #    verified_id (最后被接受的 token): (max_reqs,)
        self.future_verified_id_map = jnp.zeros(max_reqs, dtype=jnp.int32)

        # Counter for indexing
        self.future_ct = 0
        self.future_limit = max_reqs * 3
```

### 6.2 FutureEagleDraftInput

```python
@dataclass
class FutureEagleDraftInput:
    """占位符，用于标记需要从 future maps 中 resolve 的 draft input"""
    future_ct: int
    bs: int
```

## 7. 核心流程实现

### 7.1 `forward_batch_generation` (Scheduler 线程调用)

```python
def forward_batch_generation(self, model_worker_batch):
    """准备 batch 并返回 future 占位符"""

    # 1. 复制 sampling_info（避免被下一批修改）
    sampling_info = model_worker_batch.sampling_info
    sampling_info.update_penalties()
    model_worker_batch.sampling_info = dataclasses.replace(
        sampling_info,
        sampling_info_done=threading.Event(),
    )

    # 2. 准备 sampling_metadata
    sampling_metadata = SamplingMetadata.from_model_worker_batch(...)

    # 3. 放入 input_queue
    self.input_queue.put((
        model_worker_batch,
        self.future_ct,
        sampling_metadata,
    ))

    # 4. 生成 future 占位符
    bs = model_worker_batch.real_bs
    stride = self.max_draft_tokens

    # future_token_ids: 负数占位符，stride 个一组
    future_token_ids = np.arange(
        -(self.future_ct * stride + 1),
        -(self.future_ct * stride + 1 + bs * stride),
        -1,
        dtype=np.int32
    ).reshape(bs, stride)

    # future_accept_length: 负数占位符
    future_accept_length = np.arange(
        -(self.future_ct + 1),
        -(self.future_ct + 1 + bs),
        -1,
        dtype=np.int32
    )

    # future_draft_input: 用占位符索引标记
    future_draft_input = FutureEagleDraftInput(
        future_ct=self.future_ct,
        bs=bs,
    )

    self.future_ct = (self.future_ct + bs) % self.future_limit

    return GenerationBatchResult(
        logits_output=None,
        next_token_ids=future_token_ids,
        accept_lens=future_accept_length,
        next_draft_input=future_draft_input,
        ...
    )
```

### 7.2 `forward_thread_func_` (Forward 线程)

```python
def forward_thread_func_(self):
    while True:
        (model_worker_batch, future_ct, sampling_metadata) = self.input_queue.get()
        if not model_worker_batch:
            break

        # 1. Resolve future draft_input (如果 spec_info 是 future 的)
        if isinstance(model_worker_batch.spec_info, FutureEagleDraftInput):
            model_worker_batch.spec_info = self._resolve_future_draft_input(
                model_worker_batch.spec_info
            )

        # 2. Resolve input_ids 中的 future tokens
        if model_worker_batch.forward_batch is not None:
            input_ids = model_worker_batch.forward_batch.input_ids
            model_worker_batch.forward_batch.input_ids = resolve_future_token_ids(
                input_ids, self.future_token_ids_map
            )

        # 3. 执行 Eagle forward (黑盒: draft + verify)
        result = self.eagle_worker.forward_batch_speculative_generation(
            model_worker_batch
        )

        # 4. Update future maps
        self._update_future_maps(future_ct, result)

        # 5. 放入 output_queue
        self.output_queue.put(result)
```

### 7.3 `_resolve_future_draft_input`

```python
def _resolve_future_draft_input(self, future_input: FutureEagleDraftInput):
    """从 future maps 中获取真实的 EagleDraftInput"""
    ct = future_input.future_ct
    bs = future_input.bs

    return EagleDraftInput(
        hidden_states=self.future_hidden_states_map[ct:ct+bs],
        topk_p=self.future_topk_p_map[ct:ct+bs],
        topk_index=self.future_topk_index_map[ct:ct+bs],
        verified_id=self.future_verified_id_map[ct:ct+bs],
        ...
    )
```

### 7.4 `_update_future_maps`

```python
def _update_future_maps(self, future_ct, result: GenerationBatchResult):
    """将结果写入 future maps"""
    draft_input = result.next_draft_input
    accept_lens = result.accept_lens
    token_ids = result.next_token_ids  # shape: (bs, max_draft_tokens)

    bs = len(accept_lens)
    stride = self.max_draft_tokens

    # Update token_ids map
    start = future_ct * stride
    self.future_token_ids_map = self.future_token_ids_map.at[
        start:start + bs * stride
    ].set(token_ids.flatten())

    # Update accept_length map
    self.future_accept_length_map = self.future_accept_length_map.at[
        future_ct:future_ct + bs
    ].set(accept_lens)

    # Update draft input components
    self.future_hidden_states_map = self.future_hidden_states_map.at[
        future_ct:future_ct + bs
    ].set(draft_input.hidden_states)

    self.future_topk_p_map = self.future_topk_p_map.at[
        future_ct:future_ct + bs
    ].set(draft_input.topk_p)

    self.future_topk_index_map = self.future_topk_index_map.at[
        future_ct:future_ct + bs
    ].set(draft_input.topk_index)

    self.future_verified_id_map = self.future_verified_id_map.at[
        future_ct:future_ct + bs
    ].set(draft_input.verified_id)
```

### 7.5 `resolve_last_batch_result` (Scheduler 线程调用)

```python
def resolve_last_batch_result(self, launch_done):
    """获取上一批次的真实结果"""
    result: GenerationBatchResult = self.output_queue.get()

    # Transfer to CPU
    if result.logits_output.next_token_logprobs is not None:
        result.logits_output.next_token_logprobs = jax.device_get(
            result.logits_output.next_token_logprobs
        ).tolist()

    result.next_token_ids = jax.device_get(result.next_token_ids).tolist()
    result.accept_lens = jax.device_get(result.accept_lens).tolist()

    if launch_done is not None:
        launch_done.wait()

    return result
```

## 8. KV Cache 申请策略

### 方案 A: 预分配 (推荐)

在准备 batch 时，始终预留 `max_draft_tokens` 的 KV cache 空间，在 verify 后释放未使用的部分。

```python
def prepare_decode_batch(self, batch):
    for req in batch.reqs:
        # 预留 max_draft_tokens 空间 (已有逻辑)
        alloc_len = ALLOC_LEN_PER_DECODE

    # verify 后在 process_batch_result 中释放多余的
```

优点：
- 与现有逻辑兼容性好
- 实现简单

### 方案 B: 延迟申请

在 resolve 时才根据 `accept_length` 申请 KV cache。需要修改 scheduler 的流程。

## 9. Scheduler 改动

```python
# scheduler.py

def event_loop_overlap(self):
    while True:
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)

        batch = self.get_next_batch_to_run()

        if batch:
            batch.launch_done = threading.Event()
            result = self.run_batch(batch)  # 返回 future 占位符
            self.result_queue.append((batch.copy(), result))

        if self.last_batch:
            tmp_batch, tmp_result = self.result_queue.popleft()

            # Eagle: resolve 时获取真实的 accept_lens
            if self.spec_algorithm and self.spec_algorithm.is_eagle():
                resolved_result = self.tp_worker.resolve_last_batch_result(
                    batch.launch_done if batch else None
                )
                # 用 resolved_result 处理
                self.process_batch_result_eagle(tmp_batch, resolved_result)
            else:
                self.process_batch_result(tmp_batch, tmp_result, ...)
```

## 10. 需要修改的文件

| 文件 | 改动内容 |
|------|----------|
| `tp_worker_overlap_thread.py` | 新增 `EagleWorkerClient` 类 |
| `eagle_util.py` | 新增 `FutureEagleDraftInput` 类 |
| `scheduler.py` | 修改 `event_loop_overlap` 支持 Eagle |
| `scheduler_output_processor_mixin.py` | 修改结果处理逻辑 |
| `server_args.py` | 移除 Eagle + overlap 的限制检查 |

## 11. 实现步骤

1. **阶段一**: 创建 `EagleWorkerClient` 基础框架
   - 初始化 future maps
   - 实现线程和队列

2. **阶段二**: 实现 forward 流程
   - `forward_batch_generation` 返回 future 占位符
   - `forward_thread_func_` 执行 eagle forward 并更新 future maps

3. **阶段三**: 实现 resolve 流程
   - `_resolve_future_draft_input`
   - `resolve_last_batch_result`

4. **阶段四**: 修改 Scheduler
   - `event_loop_overlap` 支持 Eagle
   - 结果处理逻辑适配

5. **阶段五**: 测试和优化
   - 单元测试
   - 端到端测试
   - 性能优化

## 12. 已完成的代码改动摘要

1. **Eagle overlap worker**：新增 `EagleWorkerClient`，独立维护 Eagle 的 future maps（token_ids / accept_length / hidden_states / topk_p / topk_index / verified_id），并在 forward 线程中完成占位符解析与 map 回写（`python/sgl_jax/srt/managers/tp_worker_overlap_thread.py`）。
2. **Future 占位符类型**：新增 `FutureEagleDraftInput`，用于标记需要从 future maps resolve 的 spec_info，同时支持携带 `keep_indices` 以应对批次过滤（`python/sgl_jax/srt/speculative/eagle_util.py`）。
3. **Scheduler overlap 路径**：
   - overlap + eagle 时，scheduler 使用普通 `ModelWorker` 作为 target worker，draft worker 使用 `EagleWorkerClient`；
   - overlap 结果 resolve 由 `draft_worker.resolve_last_batch_result()` 完成；
   - overlap 时不提前用 `accept_lens` 更新 `seq_lens`，由 forward 线程 resolve 后再调整（`python/sgl_jax/srt/managers/scheduler.py`，`python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py`）。
4. **批次过滤兼容**：在 `ScheduleBatch.filter_batch` 中对 `FutureEagleDraftInput` 维护 `keep_indices`，并在 future resolve 时使用映射索引，避免 overlap pipeline 中批次缩减导致的错位（`python/sgl_jax/srt/managers/schedule_batch.py`）。
