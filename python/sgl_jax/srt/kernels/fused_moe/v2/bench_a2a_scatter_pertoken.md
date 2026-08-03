# Per-token A2A scatter semaphore benchmark

This benchmark asks whether the semaphore and send-drain schedule used by
fused MoE v2 improves a scatter made of one DMA per token/expert assignment.
The implementation is in `bench_a2a_scatter_pertoken.py`.

## Compared schedules

- `legacy`: one send semaphore and one receive semaphore, drain the send
  semaphore every 128 assignments, then wait for the complete receive buffer.
- `v2`: one send and receive semaphore per local expert, issue all assignment
  DMAs, wait for receives expert-by-expert, then drain sends expert-by-expert.

The payload, route, number of remote copies, and barriers are otherwise the
same. Both variants issue 4,096 one-row `make_async_remote_copy` operations per
device. This isolates the semaphore/drain change from routing metadata, expert
compute, and the return gather.

## Workload

The model dimensions come from the
[GLM-5.2 config](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json).

| Parameter | Value |
|---|---:|
| TPU topology | 16 v7x chips / 32 TPU cores |
| EP size | 32 |
| Tokens per device | 512 |
| Top-k | 8 |
| Assignments per device | 4,096 |
| Routed experts | 256 |
| Local experts per device | 8 |
| Hidden size | 6,144 |
| BF16 message per assignment | 12,288 bytes |
| FP8 message per assignment | 6,144 bytes |

The deterministic route is balanced: every source sends 16 assignments to
every expert, or 128 assignments to every destination device. Self and
same-chip copies are still executed and included in latency. The reported ICI
useful bandwidth excludes those copies and is a per-device value.

## Measurement

Each experiment used three warm-up iterations followed by 100 measured
iterations. The table reports the slowest process's p50, because the four
distributed processes must all keep pace. Two experiments used opposite
variant orders to expose order bias. Positive throughput delta favors `v2`.

| Order | Dtype | Legacy p50 (ms) | v2 p50 (ms) | Legacy ICI (GB/s) | v2 ICI (GB/s) | Throughput delta |
|---|---|---:|---:|---:|---:|---:|
| legacy then v2 | BF16 | 1.586996 | 1.580849 | 29.732844 | 29.848477 | +0.389% |
| legacy then v2 | FP8 | 0.976362 | 0.969395 | 24.164165 | 24.337819 | +0.719% |
| v2 then legacy | BF16 | 1.579808 | 1.575747 | 29.868136 | 29.945112 | +0.258% |
| v2 then legacy | FP8 | 0.967462 | 0.954798 | 24.386446 | 24.709897 | +1.326% |
| two-run average | BF16 | 1.583402 | 1.578298 | 29.800490 | 29.896794 | +0.323% |
| two-run average | FP8 | 0.971912 | 0.962097 | 24.275306 | 24.523858 | +1.024% |

Falcon experiment and analysis IDs:

- normal order: `exp-crovdxnttp`, `an-z63efbz8sv`
- reversed order: `exp-n49yg8nr79`, `an-71q5ftyx16`

The benchmark runs a distributed integer correctness comparison before timing;
the job fails if the two receive buffers differ. Both experiments completed
successfully with the check enabled.

## Conclusion

Per-expert send/receive semaphores and end-of-expert send drains produce a
repeatable but small p50 improvement at hidden size 6,144: about 0.3% for BF16
and 1.0% for FP8. This is not a material bandwidth step-change. BF16 p90 was
1.5% to 1.8% worse in the two 100-iteration runs, and FP8 tail behavior was not
consistent between orders.

The result means the old fixed send drain was not the main throughput limiter
for this shape. The benchmark still pays the enqueue and control overhead of
4,096 small, one-row DMAs, so a larger improvement requires changing that
granularity or overlapping the scatter with expert compute; semaphore layout
alone is insufficient.
