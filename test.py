import functools
import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pl_tpu
from scipy import stats

DIM_M = 8
DIM_N = 256
DIM_K = 256


def run_atomic_benchmark(
    op_name, op_func, loop_counts, const_input=None, workload_per_iter=1, acc_dtype=jnp.bfloat16
):
    def probe_kernel(input_ref, *args, loop_count):
        if const_input is not None:
            const_ref, out_ref = args
            const_val = const_ref[...]
        else:
            out_ref = args[0]

        carry = input_ref[...]

        def body(i, acc):
            if const_input is not None:
                return op_func(acc, const_val)
            else:
                return op_func(acc)

        result = jax.lax.fori_loop(0, loop_count, body, carry)
        out_ref[...] = result

    static_idx = 2 if const_input is not None else 1

    @functools.partial(jax.jit, static_argnums=(static_idx,))
    def run_probe(data, *args):
        n = args[-1]
        inputs = [data]
        if const_input is not None:
            inputs.append(args[0])

        bound_kernel = functools.partial(probe_kernel, loop_count=n)

        return pl.pallas_call(
            bound_kernel,
            out_shape=jax.ShapeDtypeStruct(data.shape, data.dtype),
            interpret=False,
            grid=(),
            compiler_params=pl_tpu.CompilerParams(
                vmem_limit_bytes=96 * 1024 * 1024,
                disable_bounds_checks=True,
            ),
        )(*inputs)

    print(f"\n=== Benchmarking: {op_name} ===")

    dummy_data = jnp.zeros((DIM_M, DIM_N), dtype=acc_dtype)

    # Warmup
    print("  -> Warming up...")
    if const_input is not None:
        _ = run_probe(dummy_data, const_input, 100).block_until_ready()
    else:
        _ = run_probe(dummy_data, 100).block_until_ready()

    times_ns = []

    for n in loop_counts:
        jax.block_until_ready(dummy_data)

        t0 = time.perf_counter_ns()

        if const_input is not None:
            out = run_probe(dummy_data, const_input, n)
        else:
            out = run_probe(dummy_data, n)

        out.block_until_ready()
        t1 = time.perf_counter_ns()

        duration = t1 - t0
        times_ns.append(duration)
        print(f"  -> N={n:<6} Time={duration/1e3:.2f} us")

    slope, intercept, r_value, _, _ = stats.linregress(loop_counts, times_ns)

    print(f"\n[Analysis Result for {op_name}]")
    print(f"  R-squared: {r_value**2:.4f}")
    print(f"  1. Base Overhead (Intercept): {intercept/1e3:.2f} us")

    if slope > 0:
        throughput_ops_per_ns = workload_per_iter / slope
        throughput_tflops = throughput_ops_per_ns / 1000.0

        print(f"  2. Per-Iter Latency (Slope):  {slope:.4f} ns")
        print(f"  3. Measured Throughput:       {throughput_ops_per_ns:.2f} Ops/ns")
        print(f"                                ({throughput_tflops:.2f} TFLOPS equivalent)")
    else:
        print("  [Error] Slope is negative or zero.")

    return intercept, slope


# ==========================================
#  Main Execution
# ==========================================

if __name__ == "__main__":

    assert DIM_K == DIM_N, "For recurrent benchmarks, K must equal N to maintain shape invariant."

    print(f"Device: {jax.devices()[0]}")

    # run_atomic_benchmark(
    #    "vadd_bf16",
    #    lambda x: x + 1.0,
    #    [50000, 100000, 200000],
    #    workload_per_iter=128*128,
    #    acc_dtype=jnp.bfloat16
    # )

    # Workload: 2 * 128^3 FLOPs
    b_matrix = jnp.eye(DIM_N, dtype=jnp.bfloat16)
    matmul_flops = 2 * (DIM_M * DIM_N * DIM_K)

    run_atomic_benchmark(
        "vmatmul_bf16_mixed",
        lambda acc, b: jnp.dot(acc, b),
        [500_000, 1_000_000, 2_000_000],
        const_input=b_matrix,
        workload_per_iter=matmul_flops,
        acc_dtype=jnp.float32,
    )
