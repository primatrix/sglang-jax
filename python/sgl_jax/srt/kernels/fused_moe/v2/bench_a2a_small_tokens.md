# Small-token fused MoE scatter and gather benchmark

This benchmark measures the communication schedules used by fused MoE v2 at
small decode batch sizes. The implementation is in
`bench_a2a_small_tokens.py`.

## Configuration

- Hardware: 16-chip TPU v7x, 32 visible JAX devices, EP32.
- GLM-5.2 shape: hidden size 6144, 256 routed experts, top-k 8, and 8 local
  experts per device.
- Tokens per device: 2, 4, 8, 16, and 32, corresponding to 64, 128, 256,
  512, and 1024 global tokens.
- Activations: BF16 and FP8.
- Routing: three seeded routes with eight distinct experts per token. Routes
  naturally include empty experts and nonuniform per-expert block sizes.
- Timing: three warmups and 50 timed iterations for every case. For each route,
  the reported latency is the largest process p50 across four processes. The
  table then averages that value across the three route seeds.

Scatter starts one Pallas remote DMA for every token/expert assignment. Gather
matches the fused-v2 return path: for each local expert, it starts one
variable-size DMA for every source device with a nonempty block. It is neither a
single-token gather nor one bulk `lax.all_to_all` collective.

An integer round trip at 2 and 16 tokens per device verified both intermediate
scatter placement and final gather placement with a maximum error of zero.

## Results

All times below are milliseconds. `total` is the sum of the separately measured
scatter and gather p50 values; it does not claim overlap between the two phases.

| tokens/device | global tokens | BF16 scatter | BF16 gather | BF16 total | FP8 scatter | FP8 gather | FP8 total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 64 | 0.404 | 0.406 | 0.809 | 0.396 | 0.401 | 0.797 |
| 4 | 128 | 0.411 | 0.406 | 0.817 | 0.399 | 0.400 | 0.799 |
| 8 | 256 | 0.406 | 0.412 | 0.818 | 0.389 | 0.396 | 0.785 |
| 16 | 512 | 0.426 | 0.434 | 0.860 | 0.407 | 0.398 | 0.805 |
| 32 | 1024 | 0.460 | 0.466 | 0.926 | 0.420 | 0.419 | 0.839 |

The useful payload bandwidth is low because latency is dominated by a roughly
0.4 ms fixed floor rather than payload transfer time:

| tokens/device | BF16 scatter | BF16 gather | FP8 scatter | FP8 gather |
|---:|---:|---:|---:|---:|
| 2 | 0.46 GB/s | 0.46 GB/s | 0.23 GB/s | 0.23 GB/s |
| 4 | 0.90 GB/s | 0.91 GB/s | 0.46 GB/s | 0.46 GB/s |
| 8 | 1.82 GB/s | 1.79 GB/s | 0.95 GB/s | 0.93 GB/s |
| 16 | 3.46 GB/s | 3.40 GB/s | 1.81 GB/s | 1.85 GB/s |
| 32 | 6.41 GB/s | 6.33 GB/s | 3.51 GB/s | 3.52 GB/s |

At 16 tokens per device, scatter issues about 120 one-row remote DMAs per
device. Gather carries the same useful payload in about 96 nonempty remote
blocks per device; its mean block is 1.25 rows, p90 is 2 rows, and the largest
observed block is 5 rows. The gather coalescing is real, but these blocks remain
too small for payload bandwidth to dominate the synchronized kernel latency.

## Roofline implication for 512 global tokens

The bandwidth-only estimate classified this case as HBM-bound when one weight
read was assumed: 0.164 ms for BF16 and 0.0818 ms for FP8. The measured
small-token communication totals are 0.860 ms and 0.805 ms, respectively. They
are 5.2x and 9.8x larger than the corresponding HBM terms.

Therefore the small-token perfect-overlap roofline is communication-bound, not
HBM-bound. The limiting term is the fixed cost of the fine-grained DMA schedule,
semaphore waits, and global synchronization. Halving the activation payload from
BF16 to FP8 only changes the combined latency from 0.860 ms to 0.805 ms, which is
another indication that raw ICI bandwidth is not the active limit.

This benchmark isolates communication. It does not include metadata generation,
expert FFN compute, or production overlap between communication and compute, so
its two phase latencies should not be substituted directly for an end-to-end
fused MoE latency.

## Reproducibility

- Source commit: `66862ca2a373558c7efeed6adfeced41e7c56fc8`
- Falcon experiment: `exp-gp8cf635vv`
- Artifact: `art-blsyemnqaj`
- Operator analysis: `an-t8cdx23977`
- JAX 0.9.1, jaxlib 0.9.1, libtpu 0.0.35
