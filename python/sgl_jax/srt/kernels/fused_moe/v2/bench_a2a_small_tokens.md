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
- Baseline: each communication kernel has a no-DMA counterpart with the same
  arguments, output shape, scratch semaphores, and two global barriers.

Scatter starts one Pallas remote DMA for every token/expert assignment. Gather
matches the fused-v2 return path: for each local expert, it starts one
variable-size DMA for every source device with a nonempty block. It is neither a
single-token gather nor one bulk `lax.all_to_all` collective.

An integer round trip at 2 and 16 tokens per device verified both intermediate
scatter placement and final gather placement with a maximum error of zero.

## Standalone-call results

All times below are milliseconds. `total` is the sum of the separately measured
scatter and gather p50 values; it does not claim overlap between the two phases.
These values include two standalone Pallas custom-call launches and therefore
must not be used directly as the in-kernel communication term of fused MoE v2.

| tokens/device | global tokens | BF16 scatter | BF16 gather | BF16 total | FP8 scatter | FP8 gather | FP8 total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 64 | 0.386 | 0.384 | 0.770 | 0.378 | 0.388 | 0.766 |
| 4 | 128 | 0.390 | 0.394 | 0.784 | 0.384 | 0.386 | 0.770 |
| 8 | 256 | 0.395 | 0.392 | 0.787 | 0.382 | 0.379 | 0.761 |
| 16 | 512 | 0.423 | 0.421 | 0.845 | 0.400 | 0.404 | 0.804 |
| 32 | 1024 | 0.464 | 0.464 | 0.927 | 0.418 | 0.417 | 0.835 |

The matching no-DMA calls take about 0.36--0.39 ms per phase. Subtracting their
p50 gives the following estimate of the DMA-schedule increment. Values below
about 0.02 ms are close to the noise floor of subtracting two host-side p50s.

| tokens/device | global tokens | BF16 scatter delta | BF16 gather delta | BF16 total delta | FP8 scatter delta | FP8 gather delta | FP8 total delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 64 | 0.010 | 0.006 | 0.016 | 0.004 | 0.014 | 0.018 |
| 4 | 128 | 0.015 | 0.020 | 0.035 | 0.006 | 0.008 | 0.014 |
| 8 | 256 | 0.016 | 0.027 | 0.043 | 0.014 | 0.012 | 0.026 |
| 16 | 512 | 0.044 | 0.049 | 0.092 | 0.027 | 0.021 | 0.048 |
| 32 | 1024 | 0.085 | 0.091 | 0.176 | 0.045 | 0.056 | 0.101 |

At 16 tokens per device, scatter issues about 120 one-row remote DMAs per
device. Gather carries the same useful payload in about 96 nonempty remote
blocks per device; its mean block is 1.25 rows, p90 is 2 rows, and the largest
observed block is 5 rows. The gather coalescing is real, but its blocks are
still small enough that gather does not reach the earlier 44 GB/s bulk proxy.

## Roofline implication for 512 global tokens

The bandwidth-only estimate classified this case as HBM-bound when one weight
read was assumed: 0.164 ms for BF16 and 0.0818 ms for FP8. XProf captured 20
calls of each 512-token kernel on each of eight devices in rank 0. Dividing the
reported total device time by 160 occurrences gives:

| dtype | scatter device | scatter no-DMA | gather device | gather no-DMA | device total | baseline-adjusted total | HBM (`N=1`) | roofline bound |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BF16 | 0.0475 | 0.0073 | 0.0463 | 0.0074 | 0.0938 | 0.0790 | 0.164 | HBM |
| FP8 | 0.0293 | 0.0074 | 0.0282 | 0.0075 | 0.0575 | 0.0427 | 0.0818 | HBM |

Using the raw device time and useful remote payload, the corresponding
per-device bandwidth is 31.1 GB/s scatter and 31.9 GB/s gather for BF16, and
25.2 GB/s scatter and 26.2 GB/s gather for FP8. In this small-block shape the
gather path does not retain the 44 GB/s bulk-collective proxy.

The raw device communication total is already smaller than HBM, even before
subtracting the barriers. Therefore the 512-token perfect-overlap roofline
remains HBM-bound for `N=1`; larger weight-read counts only strengthen that
classification.

The standalone wall-clock total is about 0.8 ms because it launches scatter and
gather as two independent Pallas calls. Production fused MoE v2 puts both phases
inside one larger Pallas kernel, so charging those two launch floors to the ICI
term would double-count fixed dispatch overhead. The XProf device time, not the
standalone host total, is the appropriate correction for the communication
resource term.

This benchmark isolates communication. It does not include metadata generation,
expert FFN compute, or production overlap between communication and compute, so
its two phase latencies should not be substituted directly for an end-to-end
fused MoE latency.

## Reproducibility

- Source commit: `f2061d891bd7c49d083957ffa02d8d042e33eb23`
- Falcon experiment: `exp-jwskjiylkq`
- Artifact: `art-n10i1yzsxk`
- Operator analysis: `an-q3dz9jjisa`
- BF16 XProf analysis: `an-ao8xqiwa1i`
- FP8 XProf analysis: `an-i9q7jmxymm`
- JAX 0.9.1, jaxlib 0.9.1, libtpu 0.0.35
