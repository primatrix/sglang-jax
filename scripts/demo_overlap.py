"""
对比 sglang-jax 当前双线程 overlap vs 单线程 async dispatch overlap。
严格对齐真实 scheduler 的调用顺序。
包含正确性校验 + 性能对比 + JAX profiling。
"""

import os
import threading
import time
from queue import Queue

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
import numpy as np

# ---------- 模拟 model forward ----------


@jax.jit
@jax.named_scope("model_forward")
def model_forward(input_ids, weight):
    # embedding lookup: 不同 token id → 不同初始向量 → 不同计算结果
    x = weight[input_ids % weight.shape[0]]  # (bs, dim)
    for _ in range(100):
        x = x @ weight + x
    return jnp.argmax(x, axis=-1).astype(jnp.int32)


# ---------- future token ID 操作 ----------


@jax.jit
@jax.named_scope("set_future")
def set_future_token_ids(future_map, start, next_token_ids):
    return jax.lax.dynamic_update_slice(future_map, next_token_ids, (start + 1,))


@jax.jit
@jax.named_scope("resolve_future")
def resolve_future_token_ids(input_ids, future_map):
    return jnp.where(
        input_ids < 0,
        future_map[jnp.clip(-input_ids, min=0)],
        input_ids,
    )


# ---------- 模拟 CPU 调度工作 ----------


def cpu_scheduling(ms: float = 3.0):
    end = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < end:
        pass


# =====================================================================
# 方式 A: 双线程 overlap (对齐真实 scheduler 流程)
#
# 真实 scheduler 每轮循环 (scheduler.py:565-601):
#   1. run_batch(batch_N)
#        → forward_batch_generation(): input_queue.put + 返回 future token ids
#   2. process_batch_result(batch_N-1)
#        → resolve_last_batch_result(): output_queue.get + device_get
#        → CPU 工作: 处理结果, 攒批, 准备下一个 batch
#
# 重叠: step 2 的 CPU 工作 与 forward thread 计算 batch_N 重叠
#
# forward thread (tp_worker_overlap_thread.py:94-128):
#   input_queue.get() → resolve_future → forward → set_future → output_queue.put
#   (全部 async dispatch, 不做 device_get)
# =====================================================================


def run_two_thread(weight, steps, cpu_ms):
    bs = weight.shape[0]
    future_map = jnp.zeros((bs * (steps + 5),), dtype=jnp.int32)
    future_ct = 0

    input_queue = Queue()
    output_queue = Queue()

    fwd_step = [0]  # mutable counter for forward thread

    def forward_thread_func():
        nonlocal future_map
        while True:
            item = input_queue.get()
            if item is None:
                break
            input_ids, ft_ct = item
            with jax.profiler.TraceAnnotation(f"fwd_thread_batch_{fwd_step[0]}"):
                resolved = resolve_future_token_ids(input_ids, future_map)
                tokens = model_forward(resolved, weight)
                future_map = set_future_token_ids(future_map, ft_ct, tokens)
            output_queue.put(tokens)
            fwd_step[0] += 1

    t = threading.Thread(target=forward_thread_func, daemon=True)
    t.start()

    times = []
    results = []
    input_ids = jnp.arange(bs, dtype=jnp.int32) + 1

    for step in range(steps):
        t0 = time.perf_counter()

        # ---- Step 1: run_batch(batch_N) ----
        with jax.profiler.TraceAnnotation(f"send_batch_{step}"):
            input_queue.put((input_ids, future_ct))
            future_indices = -(future_ct + 1 + np.arange(bs))
            future_ct += bs
            input_ids = jnp.array(future_indices, dtype=jnp.int32)

        # ---- Step 2: process_batch_result(batch_N-1) ----
        if step > 0:
            with jax.profiler.TraceAnnotation(f"resolve_batch_{step - 1}"):
                result_device = output_queue.get()
                result_np = np.array(jax.device_get(result_device))
                results.append(result_np)

            with jax.profiler.TraceAnnotation(f"cpu_work_batch_{step - 1}"):
                cpu_scheduling(cpu_ms)

        times.append(time.perf_counter() - t0)

    # drain 最后一个 batch 的结果
    with jax.profiler.TraceAnnotation(f"resolve_batch_{steps - 1}"):
        result_device = output_queue.get()
        results.append(np.array(jax.device_get(result_device)))

    input_queue.put(None)
    t.join(timeout=5)
    return times, results


# =====================================================================
# 方式 B: 单线程 + future token ID (async dispatch 数据依赖链)
#
# 同样的调用顺序, 但去掉 forward thread:
#   1. dispatch resolve + forward + set_future (async, 立即返回)
#   2. device_get 上一步 + CPU 调度 (与 TPU 计算重叠)
#
# TPU 时间线上通过 future_map 数据依赖自动串联:
#   set_future[N] → resolve_future[N+1] → forward[N+1] → set_future[N+1] → ...
# =====================================================================


def run_single_thread(weight, steps, cpu_ms):
    bs = weight.shape[0]
    future_map = jnp.zeros((bs * (steps + 5),), dtype=jnp.int32)
    future_ct = 0
    prev_tokens = None

    input_ids = jnp.arange(bs, dtype=jnp.int32) + 1

    times = []
    results = []
    for step in range(steps):
        t0 = time.perf_counter()

        # ---- Step 1: dispatch batch_N (对应 run_batch) ----
        with jax.profiler.TraceAnnotation(f"dispatch_batch_{step}"):
            resolved = resolve_future_token_ids(input_ids, future_map)
            tokens = model_forward(resolved, weight)
            future_map = set_future_token_ids(future_map, future_ct, tokens)

        future_indices = -(future_ct + 1 + np.arange(bs))
        future_ct += bs
        input_ids = jnp.array(future_indices, dtype=jnp.int32)

        # ---- Step 2: 处理上一步结果 (对应 process_batch_result) ----
        if prev_tokens is not None:
            with jax.profiler.TraceAnnotation(f"resolve_batch_{step - 1}"):
                result_np = np.array(jax.device_get(prev_tokens))
                results.append(result_np)
            with jax.profiler.TraceAnnotation(f"cpu_work_batch_{step - 1}"):
                cpu_scheduling(cpu_ms)

        prev_tokens = tokens
        times.append(time.perf_counter() - t0)

    # drain 最后一步
    if prev_tokens is not None:
        with jax.profiler.TraceAnnotation(f"resolve_batch_{steps - 1}"):
            results.append(np.array(jax.device_get(prev_tokens)))

    return times, results


# ---------- Main ----------


def main():
    print(f"Platform: {jax.default_backend()}, devices: {jax.device_count()}")

    dim = 4096
    steps = 30
    cpu_ms = 3.0
    weight = jax.random.normal(jax.random.PRNGKey(42), (dim, dim), dtype=jnp.bfloat16) * 0.001

    # warmup
    print("Warming up...")
    run_two_thread(weight, 5, cpu_ms)
    run_single_thread(weight, 5, cpu_ms)
    print("Done.\n")

    # === 正确性校验 ===
    print("=" * 50)
    print("正确性校验")
    print("=" * 50)
    _, results_2t = run_two_thread(weight, steps, 3)
    _, results_1t = run_single_thread(weight, steps, 3)

    all_match = True
    for i in range(min(len(results_2t), len(results_1t))):
        if not np.array_equal(results_2t[i], results_1t[i]):
            print(f"  step {i}: MISMATCH! 双线程={results_2t[i]}, 单线程={results_1t[i]}")
            all_match = False

    if all_match:
        print(f"  全部 {len(results_2t)} 步结果完全一致 ✓")
    print()

    # === 性能对比 ===
    print("=" * 50)
    print("性能对比")
    print("=" * 50)
    t_2t, _ = run_two_thread(weight, steps, cpu_ms)
    t_1t, _ = run_single_thread(weight, steps, cpu_ms)

    avg_2t = np.mean(t_2t[3:]) * 1000
    avg_1t = np.mean(t_1t[3:]) * 1000
    print(f"  双线程 overlap: {avg_2t:.2f}ms/step")
    print(f"  单线程 overlap: {avg_1t:.2f}ms/step")
    print(f"  差异:           {(avg_2t - avg_1t) / avg_2t * 100:.1f}%")
    print()

    # === Profiling ===
    profile_dir = "/tmp/jax_profile"
    print(f"Profiling → {profile_dir}")

    with jax.profiler.trace(profile_dir):
        with jax.profiler.TraceAnnotation("two_thread"):
            run_two_thread(weight, 10, cpu_ms)
        with jax.profiler.TraceAnnotation("single_thread"):
            run_single_thread(weight, 10, cpu_ms)

    print(f"Profile 已保存到 {profile_dir}")
    print("用 tensorboard --logdir /tmp/jax_profile 查看")
    print()
    print("在 trace viewer 中应该能看到:")
    print("  TPU: resolve_future → model_forward → set_future → resolve_future → ...")
    print("  CPU: dispatch(快) + device_get(上一步) + cpu_scheduling")


if __name__ == "__main__":
    main()
