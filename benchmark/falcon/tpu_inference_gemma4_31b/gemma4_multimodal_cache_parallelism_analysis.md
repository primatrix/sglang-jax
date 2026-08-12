# Gemma 4 Multimodal Cache and Parallelism Analysis

This document records the source inspection, Falcon runtime evidence, capacity
calculations, and unresolved questions for the Gemma 4 31B multimodal serving
path in pinned `tpu-inference` and the current SGLang-JAX implementation. It is
intended to prevent three similarly named mechanisms from being conflated:

1. the host-side multimodal processor cache;
2. the device-side encoder-output/embedding cache; and
3. the decoder KV cache.

The pure-text performance investigation is maintained separately in
[`sglang_jax_text_performance_analysis.md`](sglang_jax_text_performance_analysis.md).

Last validated: 2026-08-11.

## Pinned scope

| Component | Value |
| --- | --- |
| Model | `google/gemma-4-31B-it` |
| Hardware | one TPU v7x-8 host, 8 framework-visible devices / 4 chips |
| `tpu-inference` | `a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336` |
| vLLM | `f5bb701fa270f5c801f1572e1478b56f292d8dfc` |
| Workload | 1,000 requests, 1,024 input tokens, one generated 512x512 image, 500 output tokens, request rate `inf` |
| Serving parallelism | DP=4, TP=2 |
| Encoder mode | `mm-encoder-tp-mode=data` |
| Encoder compilation | `cudagraph_mm_encoder=true` |

The exact upstream case, runner, image, and full server options are pinned in
[`README.md`](README.md). The important runtime options for this analysis are:

```text
--max-num-seqs 256
--max-num-batched-tokens 4096
--data-parallel-size 4
--tensor-parallel-size 2
--additional-config {..., "mm-encoder-tp-mode": "data"}
--compilation-config {"cudagraph_mm_encoder": true}
--disable-chunked-mm-input
```

## Conclusions

- On the upstream implementation, multimodal single-process SPMD reaches
  3,534.87 output tok/s versus 10,988.92 tok/s in default multiprocess DP, a
  67.83% loss. The corresponding text-only loss is only 8.42%, so most of this
  effect is specific to the multimodal path interacting with the execution
  topology.
- The final SGLang-JAX multimodal run reaches 2,373.05 output tok/s, 32.87%
  below upstream single-process SPMD. That is not yet a fair parity number:
  SGLang-JAX was given only one fourth of upstream's effective per-DP request
  and prefill budgets, and its client reported a different total input-token
  count. Its very large TTFT and relatively low TPOT are consistent with
  under-admission rather than a uniformly slow decode path.
- In the controlled text-only comparison, the best SGLang-JAX result is
  8,013.90 output tok/s, 43.33% below upstream single-process SPMD, with 2.28x
  median TPOT. An upstream RPA3 ablation is itself 26.26% below Batched RPA,
  directly supporting the attention path as a material part of the residual
  text gap.
- The default upstream multimodal run is **not one global single-process SPMD
  program**. It is four vLLM DP engine processes, each running JAX SPMD TP=2.
- The ViT does not have its own process. Each DP engine contains both the text
  model and a ViT.
- Within each default DP engine, the text model uses TP=2 weight sharding. With
  `mm-encoder-tp-mode=data`, the ViT instead replicates its weights across the
  same two devices and splits the image batch over them.
- The upstream encoder-output cache is enabled by default. For this benchmark,
  its scheduler capacity is 4,096 encoder embeddings per scheduler instance.
- A full Gemma 4 image produces 280 BF16 embeddings of width 5,376. This is
  approximately 2.871 MiB per image and permits 14 full-size images in the
  upstream 4,096-embedding accounting budget.
- The ViT batch input is sharded, but its projected output is explicitly
  replicated before being split into per-image cache entries. A cached
  `jax.Array` is therefore not token- or hidden-sharded.
- The synthetic benchmark generates a new random RGB image for every request.
  Cross-request cache hits should consequently be effectively zero. The cache
  still retains encoder outputs while they are consumed by the request, but
  cross-request reuse cannot explain the measured throughput.
- SGLang-JAX currently defaults `mm_embedding_cache_size_mb` to zero. Neither
  the inspected text launches nor the final 1,000-request multimodal launch
  overrides it, so the device embedding pool was absent. This is irrelevant to
  pure text and should have little effect on this unique-image multimodal
  workload unless a request needs the same item more than once because of
  chunking or retry/recompute.

## Complete measured-data ledger

All values below were read back from Falcon experiment logs or declared
analysis outputs on 2026-08-11. They are kept here instead of relying only on
the shorter conclusions in the text-performance document. Each row is one
successful run, not a repeated-run mean.

### Upstream Batched-RPA baseline

These four rows use the same pinned upstream case, model, v7x-8 hardware,
DP=4, TP=2, FP8 text path, FP8 KV cache, 1,000 requests, 1,024 input tokens,
500 output tokens, and infinite request rate. Multimodal rows additionally use
one generated 512x512 JPEG per request.

| Workload / parallel mode | Request throughput | Output / total token throughput | Median / P99 TTFT | Median / P99 TPOT | Median / P99 ITL | Median / P99 E2EL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text / default multiprocess DP | 30.88 req/s | 15,440.38 / 47,062.28 tok/s | 5,815.62 / 11,142.44 ms | 46.02 / 51.06 ms | 34.63 / 173.09 ms | 28,743.68 / 32,189.85 ms |
| Text / single-process SPMD | 28.28 req/s | 14,140.83 / 43,101.25 tok/s | 7,698.47 / 12,945.25 ms | 46.84 / 51.85 ms | 35.10 / 178.51 ms | 31,069.52 / 35,196.74 ms |
| Multimodal / default multiprocess DP | 21.98 req/s | 10,988.92 / 39,459.83 tok/s | 11,107.87 / 36,983.25 ms | 53.31 / 58.43 ms | 31.50 / 221.23 ms | 37,679.14 / 45,011.83 ms |
| Multimodal / single-process SPMD | 7.07 req/s | 3,534.87 / 12,693.27 tok/s | 84,338.80 / 132,589.53 ms | 98.37 / 107.13 ms | 31.77 / 511.68 ms | 133,440.56 / 140,976.40 ms |

The pure-SPMD multimodal row is 67.83% lower in output throughput than the
default multimodal row; equivalently, the default multiprocess row is 3.11x
faster. Pure-SPMD median TTFT is 7.59x and median TPOT is 1.85x the default
values. By contrast, the text-only pure-SPMD penalty is only 8.42% in output
throughput. This establishes a multimodal/topology interaction, but does not
by itself attribute the loss to one ViT operator.

### SGLang-JAX text measurements

| Configuration | Request throughput | Output / total token throughput | Median / P99 TTFT | Median / P99 TPOT | Median / P99 ITL | Median / P99 E2EL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original global limits `256 / 4096` | 11.395 req/s | 5,697.60 / 17,366.28 tok/s | 22,030.04 / 78,511.78 ms | 33.37 / 39.84 ms | 29.86 / 31.82 ms | 37,486.75 / 87,662.09 ms |
| Corrected DP limits `1024 / 16384` | 15.237 req/s | 7,618.53 / 23,221.26 tok/s | 6,607.19 / 12,916.05 ms | 100.61 / 107.89 ms | 82.51 / 400.39 ms | 55,594.47 / 63,651.44 ms |
| Corrected limits + SWA eviction interval 128 | 16.03 req/s | 8,013.90 / 24,426.36 tok/s | 6,511.46 / 12,627.34 ms | 106.88 / 116.45 ms | 86.66 / 358.50 ms | 59,845.82 / 61,985.11 ms |

The original SGLang-JAX limits were not equivalent to upstream. Upstream
applies `max-num-seqs=256` and `max-num-batched-tokens=4096` independently to
each of four DP engines. SGLang-JAX interpreted the same numbers globally,
giving each DP rank only about 64 requests and 1,024 prefill tokens. Correcting
the global limits to `1024 / 16384` improved output throughput by 33.71% and
reduced median TTFT by 70.01%, while exposing the much slower large-batch
decode path. Shortening the SWA eviction interval added another 5.19% output
throughput, but did not close the residual gap.

The best measured SGLang-JAX text result remains 43.33% below the
topology-matched upstream single-process SPMD baseline in output throughput
(8,013.90 versus 14,140.83 tok/s), with 2.28x its median TPOT (106.88 versus
46.84 ms).

### SGLang-JAX multimodal measurements

| Run | Actual token distribution | Request throughput | Output / total token throughput | Median / P99 TTFT | Median / P99 TPOT | Median / P99 ITL | Median / P99 E2EL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r3, superseded | Randomized lengths; 618,787 text-input and 258,000 vision tokens | 5.980 req/s | 1,493.08 / 6,736.51 tok/s | 91,216.01 / 155,752.78 ms | 136.11 / 276.54 ms | 114.65 / 444.96 ms | 132,096.73 / 163,937.01 ms |
| r4, final exact-length run | 1,206,057 text-input and 258,000 vision tokens; 500,000 generated tokens | 4.746 req/s | 2,373.05 / 9,321.60 tok/s | 95,258.97 / 195,333.62 ms | 49.09 / 94.18 ms | 29.98 / 134.08 ms | 111,654.26 / 210,128.79 ms |

Both rows completed 1,000 requests with one unique random 512x512 JPEG per
request, BF16 vision weights, FP8 W8A8 text weights/activations, FP8 KV cache,
DP=4, TP=2, and `vision-encoder-parallel=dp`. The r3 client actually ran with
a variable-length distribution and generated roughly half the requested
maximum output tokens, so it is retained as historical evidence but is not a
benchmark baseline. The r4 log confirms 500,000 generated tokens and is the
canonical SGLang-JAX multimodal result.

The r4 server still used global `max-running-requests=256` and global
`max-prefill-tokens=4096`. It therefore has the same fourfold DP-budget
mismatch as the original SGLang-JAX text run: approximately 64 requests and
1,024 prefill tokens per rank instead of upstream's 256 and 4,096. Against the
single-process upstream multimodal row, r4 output throughput is 32.87% lower
(2,373.05 versus 3,534.87 tok/s), but median TPOT is 50.10% lower because the
SGLang-JAX decode batch is much smaller. Median TTFT is 12.95% higher and P99
TTFT is 47.32% higher. This is diagnostic evidence of under-admission and a
long prefill tail, not a fair implementation-parity result. A corrected
SGLang-JAX multimodal `1024 / 16384` run has not yet been collected.
The SGLang-JAX client also counted 1,464,057 total input tokens in r4. Derived
from the rounded upstream request/output/total throughput metrics, the upstream
single-process run consumed approximately 1,295,000 input tokens, about 13%
fewer. Client prompt/image-token accounting must therefore be made identical
before treating total-token throughput as a parity metric.

### Upstream RPA3 block-size ablations

Both rows below are text-only, single-process SPMD runs. They disable Batched
RPA, patch the common attention wrapper so the configured RPA3 block sizes
actually reach the kernel, and otherwise reuse the upstream workload.

| RPA3 prefill/mixed block sizes | Request throughput | Output / total token throughput | Median / P99 TTFT | Median / P99 TPOT | Median / P99 ITL | Median / P99 E2EL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `256,256,128,128` | 20.85 req/s | 10,427.42 / 31,782.78 tok/s | 7,868.30 / 13,825.65 ms | 70.77 / 75.46 ms | 60.38 / 202.50 ms | 43,178.44 / 47,808.17 ms |
| `512,512,256,128` | 20.76 req/s | 10,379.81 / 31,637.65 tok/s | 7,902.21 / 13,885.15 ms | 70.86 / 75.57 ms | 60.07 / 203.00 ms | 43,259.02 / 48,033.41 ms |

The first RPA3 setting is 26.26% below the Batched-RPA single-process SPMD
baseline in output throughput, and its median TPOT is 51.09% higher. The two
RPA3 block-size settings differ by only 0.46% in output throughput, so the
tested block-size change does not explain the Batched-RPA advantage. This is
an end-to-end attention-path ablation, not a kernel-only microbenchmark:
disabling Batched RPA also removes the TPU platform's minimum-256 KV block-size
override.

### Correctness smoke and XProf evidence

The successful multimodal smoke experiment returned exactly `TEXT_OK` for the
text request and `Red` for a generated solid-red image, establishing that both
modalities execute semantically before interpreting the performance runs.

The SGLang-JAX XProf summary attached to the shorter-SWA run reports
accumulated TPU operator time as follows:

| Category | Accumulated time | Share |
| --- | ---: | ---: |
| Matmul | 11,396.16 ms | 39.8% |
| Elementwise | 6,632.01 ms | 23.1% |
| Custom kernel | 5,910.46 ms | 20.6% |
| Data formatting | 2,418.45 ms | 8.4% |
| Async | 2,264.10 ms | 7.9% |

All ten leading individual operator entries are sliding-window RPA custom
calls. The profile shows only 14.85 ms of visible AllReduce stall. Its window
primarily covers large prefill, so these shares are not a pure-decode
breakdown; they support, but do not independently quantify, the RPA/matmul
residual bottleneck.

### Falcon record index

| Purpose | Experiment | Artifact / analysis |
| --- | --- | --- |
| Upstream text, multiprocess DP, Batched RPA | `exp-uxr7vn058s` | [`art-1t44htxy4y`](https://falcon.infiscale-infra.org/v1/experiments/exp-uxr7vn058s/artifacts/art-1t44htxy4y/view), `an-26bg18bwur` |
| Upstream text, single-process SPMD, Batched RPA | `exp-prrux3n44r` | [`art-vpbdzhlrjs`](https://falcon.infiscale-infra.org/v1/experiments/exp-prrux3n44r/artifacts/art-vpbdzhlrjs/view), `an-w7jvuy9i1y` |
| Upstream multimodal, multiprocess DP, Batched RPA | `exp-h8lobna2zl` | [`art-wqep29i4kc`](https://falcon.infiscale-infra.org/v1/experiments/exp-h8lobna2zl/artifacts/art-wqep29i4kc/view), `an-j39zkdzm4u` |
| Upstream multimodal, single-process SPMD, Batched RPA | `exp-8dxhtf3vw0` | [`art-ess6cwljgp`](https://falcon.infiscale-infra.org/v1/experiments/exp-8dxhtf3vw0/artifacts/art-ess6cwljgp/view), `an-cby75i4qh6` |
| SGLang-JAX text, original limits | `exp-y0kwk0m30c` | [`art-5e10oqknvs`](https://falcon.infiscale-infra.org/v1/experiments/exp-y0kwk0m30c/artifacts/art-5e10oqknvs/view) |
| SGLang-JAX text, corrected limits | `exp-nkhgv6c82x` | [`art-i2u6p665ok`](https://falcon.infiscale-infra.org/v1/experiments/exp-nkhgv6c82x/artifacts/art-i2u6p665ok/view), `an-p683xivmi5` |
| SGLang-JAX text, shorter SWA interval / profile | `exp-hjqdhqabqj` | [`art-7tsoelfwp6`](https://falcon.infiscale-infra.org/v1/experiments/exp-hjqdhqabqj/artifacts/art-7tsoelfwp6/view), `an-v1gou1et3t`, `an-es1zj03ceq` |
| SGLang-JAX multimodal r3, superseded | `exp-bbu5qzzjtd` | [`art-jit8su7jdz`](https://falcon.infiscale-infra.org/v1/experiments/exp-bbu5qzzjtd/artifacts/art-jit8su7jdz/view) |
| SGLang-JAX multimodal r4, final | `exp-cgnd8pmc8d` | [`art-rwzxorauqs`](https://falcon.infiscale-infra.org/v1/experiments/exp-cgnd8pmc8d/artifacts/art-rwzxorauqs/view) |
| SGLang-JAX text + vision correctness smoke | `exp-uqqejsjpv4` | [`art-n5q13cxxoq`](https://falcon.infiscale-infra.org/v1/experiments/exp-uqqejsjpv4/artifacts/art-n5q13cxxoq/view) |
| Upstream text SPMD, RPA3 `256,256,128,128` | `exp-ippd1153ni` | [`art-24go2w2lnz`](https://falcon.infiscale-infra.org/v1/experiments/exp-ippd1153ni/artifacts/art-24go2w2lnz/view), `an-hgvt7ac8xd` |
| Upstream text SPMD, RPA3 `512,512,256,128` | `exp-ujor5vqwcs` | [`art-qbfee4sqg8`](https://falcon.infiscale-infra.org/v1/experiments/exp-ujor5vqwcs/artifacts/art-qbfee4sqg8/view), `an-wxaa1sxbxi` |

## Process and device topology

### Default upstream mode

The default run leaves `TPU_MULTIPROCESS_DP` unset; at the pinned commit this
resolves to multiprocess DP. The effective topology is:

```text
API/front end
  +-- DP engine 0: text model + ViT + local encoder cache -> TPU devices 0,1
  +-- DP engine 1: text model + ViT + local encoder cache -> TPU devices 2,3
  +-- DP engine 2: text model + ViT + local encoder cache -> TPU devices 4,5
  `-- DP engine 3: text model + ViT + local encoder cache -> TPU devices 6,7
```

Each DP engine is a complete engine process. ViT execution is not delegated to
a separate process or shared encoder service. Within a process:

- text execution uses a two-device model/TP mesh;
- ViT `data` mode folds the local model axis into `VIT_BATCH`, so the two
  devices receive different images while holding replicated ViT weights; and
- the engine process owns an independent Python encoder-cache dictionary.

The relevant pinned sharding implementation defines `VIT_BATCH=data` and
`VIT_MODEL=model`, then changes `VIT_BATCH` to `(data, model)` and clears
`VIT_MODEL` when `mm_encoder_tp_mode=data`:

- [`sharding.py`, vision axes](https://github.com/vllm-project/tpu-inference/blob/a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336/tpu_inference/layers/common/sharding.py#L35-L76)
- [`sharding.py`, data-mode override](https://github.com/vllm-project/tpu-inference/blob/a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336/tpu_inference/layers/common/sharding.py#L396-L403)

### Pure-SPMD control

The pure-SPMD manifest adds `TPU_MULTIPROCESS_DP=0`. One process then controls
one `data=4 x model=2` mesh over all eight visible devices. The same
`mm-encoder-tp-mode=data` rule makes the ViT batch sharding span both mesh
axes. The projected encoder output is nevertheless constrained to
`PartitionSpec(None, None, None)`, so each cached output is replicated across
the full mesh rather than retained on only the device that encoded it.

There is still no explicit `[DP, batch, token, hidden]` cache tensor. The
logical image batch is flat. JAX sharding maps that logical batch onto devices,
and host code later maps individual outputs back to requests. Before text
prefill, `gather_mm_embeddings()` uses `req_ids_dp` and per-rank token offsets
to place each image embedding into the correct DP slot:

- [`gemma4_mm.py`, input sharding and replicated output](https://github.com/vllm-project/tpu-inference/blob/a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336/tpu_inference/models/jax/gemma4_mm.py#L806-L842)
- [`multimodal_manager.py`, DP-aware merge placement](https://github.com/vllm-project/tpu-inference/blob/a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336/tpu_inference/runner/multimodal_manager.py#L177-L267)

An identical image cannot hit another DP process's cache in default
multiprocess mode because the dictionaries are process-local. A single-process
SPMD cache can in principle reuse one key across DP ranks, although its cached
array is replicated across the full mesh.

## The three cache layers

| Cache | Location | Key/value | Default | Capacity in this run | Purpose |
| --- | --- | --- | --- | --- | --- |
| Multimodal processor cache | Host/CPU process memory | media hash -> processed tensors | Enabled | 4 GiB per configured cache instance | Avoid image decode/preprocessing and repeated frontend/engine transfer |
| Encoder-output cache | TPU device memory plus host bookkeeping | `mm_hash -> jax.Array` | Enabled | 4,096 encoder embeddings per scheduler instance | Avoid rerunning ViT/projector and retain outputs until text prefill consumes them |
| Decoder KV cache | TPU HBM | text-token blocks -> K/V tensors | Enabled | Derived from remaining HBM and `gpu-memory-utilization=0.9` | Reuse decoder attention K/V |

### Host processor cache

Pinned vLLM sets `mm_processor_cache_gb=4` and `mm_processor_cache_type=lru`.
This is not the ViT embedding cache and it is not TPU HBM. The source describes
its total configured memory accounting as:

```text
mm_processor_cache_gb * (api_server_count + data_parallel_size)
```

With one API server and DP=4 this is a source-declared upper bound of 20 GiB.
It is an LRU capacity, not evidence that 20 GiB was immediately resident; no
runtime RSS measurement was collected. See pinned
[`MultiModalConfig`](https://github.com/vllm-project/vllm/blob/f5bb701fa270f5c801f1572e1478b56f292d8dfc/vllm/config/multimodal.py#L138-L157).

### Device encoder-output cache

Pinned vLLM initializes both encoder compute budget and encoder-cache size from
`max_num_batched_tokens`, and takes the maximum with the largest single
multimodal item. The benchmark uses 4,096 batched tokens and a Gemma 4 image is
at most 280 embeddings, so:

```text
encoder_cache_size = max(4096, 280) = 4096 embeddings
```

The cache manager retains completed entries after the last request reference
is released; they become LRU-evictable and are physically removed only when
space is needed. See pinned
[`EncoderCacheManager`](https://github.com/vllm-project/vllm/blob/f5bb701fa270f5c801f1572e1478b56f292d8dfc/vllm/v1/core/encoder_cache_manager.py#L18-L69)
and its
[`compute_mm_encoder_budget`](https://github.com/vllm-project/vllm/blob/f5bb701fa270f5c801f1572e1478b56f292d8dfc/vllm/v1/core/encoder_cache_manager.py#L278-L320).

Gemma 4 31B has:

```text
default_output_length = 280 embeddings/image
text hidden_size      = 5376
embedding dtype       = BF16 = 2 bytes
```

The model values are in the public
[`google/gemma-4-31B-it` config](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json).
The resulting logical payload calculations are:

```text
one image       = 280 * 5376 * 2 = 3,010,560 bytes = 2.87109375 MiB
raw cache limit = 4096 * 5376 * 2 = 44,040,192 bytes = 42 MiB
full images     = floor(4096 / 280) = 14
14-image payload                         = 40.1953125 MiB
```

This is a logical capacity, not one contiguous preallocated upstream buffer.
`tpu-inference` stores independently produced arrays in a Python dictionary.

## ViT output splitting and JAX sharding

`tpu-inference` performs two different operations that should not be called
the same kind of "split":

1. **Physical input sharding.** `pixel_values` and `pixel_position_ids` use
   `PartitionSpec(VIT_BATCH, None, None)`, so JAX assigns different batch items
   to different devices.
2. **Logical per-image slicing.** After the encoder returns, Python/JAX indexing
   selects `vt_output[i]`, or the compiled path selects
   `output[local_idx, :true_length]`. This is not `jnp.split` and it does not
   shard one image across devices.

Before logical slicing, Gemma 4 applies:

```python
PartitionSpec(None, None, None)
```

to the projected output. Consequently, a cache entry with logical shape
`[280, 5376]` is replicated on the applicable mesh. The dictionary contains
one global `jax.Array` object per hash, while the JAX array owns replicated
device buffers.

For the default DP=4 x TP=2 multiprocess topology, a full cache of full-size
images is theoretically:

```text
40.1953125 MiB/device * 8 devices = 321.5625 MiB aggregate HBM payload
```

The raw 4,096-row bound is 42 MiB/device, or 336 MiB over eight devices. These
are source-derived payload bounds, not an observed HBM profile; allocator and
executable buffers are additional.

Relevant pinned paths are:

- [`gemma4_mm.py`, eager batch sharding and per-image slicing](https://github.com/vllm-project/tpu-inference/blob/a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336/tpu_inference/models/jax/gemma4_mm.py#L779-L842)
- [`gemma4_mm.py`, compiled output postprocessing](https://github.com/vllm-project/tpu-inference/blob/a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336/tpu_inference/models/jax/gemma4_mm.py#L1038-L1063)
- [`multimodal_manager.py`, `mm_hash` assignment](https://github.com/vllm-project/tpu-inference/blob/a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336/tpu_inference/runner/multimodal_manager.py#L111-L165)

## Why a compiled encoder path can coexist with a dynamic hash cache

The Python hash table is outside the JAX/XLA program. A simplified execution
boundary is:

```text
Host Python:
  collect mm_hashes
  choose an encoder token budget
  pad image tensors to the budget template
        |
        v
JAX/XLA executable:
  inputs:  pixel_values, pixel_position_ids with a fixed shape
  output:  [batch, max_soft_tokens, hidden]
        |
        v
Host Python/JAX dispatch:
  slice the batch into true per-image lengths
  encoder_cache[mm_hash] = per_image_output
```

Neither the string hash nor the Python dictionary is traced by JAX. Only fixed
shape tensors cross the compiled boundary.

The benchmark explicitly enables `cudagraph_mm_encoder=true`. On TPU this name
does not mean that CUDA Graph runs on a TPU. `MMEncoderJITManager` subclasses
vLLM's encoder CUDA-graph manager to reuse its budget derivation and bin
packing, but replaces graph capture/replay with:

- one padded template for each supported encoder token budget;
- one stable `jax.jit` closure;
- startup calls that prime the XLA compilation cache; and
- runtime padding to a known template so XLA reuses the executable for that
  input shape.

Gemma 4 implements the protocol for constructing capture templates, replay
buffers, a fixed-shape `encoder_cudagraph_forward`, output postprocessing, and
an out-of-budget fallback. See:

- [`MMEncoderJITManager`](https://github.com/vllm-project/tpu-inference/blob/a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336/tpu_inference/runner/mm_encoder_jit_manager.py#L1-L47)
- [`Gemma 4 encoder protocol`](https://github.com/vllm-project/tpu-inference/blob/a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336/tpu_inference/models/jax/gemma4_mm.py#L850-L1040)

The source calls the overflow path "eager fallback", but the heavy Gemma 4
encoder function is still decorated with `jax.jit`. The operational distinction
is managed, padded, precompiled budget reuse versus a shape encountered and
compiled lazily outside that manager; it is not XLA versus a Python
interpretation of all ViT layers.

## Expected cache-hit behavior in this benchmark

Pinned vLLM's `RandomMultiModalDataset` creates fresh random `uint8` RGB pixels
for every generated image. The benchmark requests 1,000 such images, one per
request. See pinned
[`RandomMultiModalDataset`](https://github.com/vllm-project/vllm/blob/f5bb701fa270f5c801f1572e1478b56f292d8dfc/vllm/benchmarks/datasets/datasets.py#L906-L981)
and its
[`sample` loop](https://github.com/vllm-project/vllm/blob/f5bb701fa270f5c801f1572e1478b56f292d8dfc/vllm/benchmarks/datasets/datasets.py#L1245-L1325).

Therefore:

- an encoder-cache hit across two ordinary requests is effectively impossible;
- a processor-cache hit across two ordinary requests is likewise effectively
  impossible;
- the encoder cache still holds outputs between encoder execution and their
  use by text prefill; and
- reuse can still matter if the same item is revisited within a request because
  of chunking, preemption, or recomputation, but this case uses
  `--disable-chunked-mm-input` and no such hit count was captured.

The cache being enabled upstream and disabled in SGLang-JAX should not be used
as the primary explanation for the unique-image throughput gap without runtime
hit/miss evidence.

## SGLang-JAX implementation and current configuration

SGLang-JAX exposes:

```text
--mm-embedding-cache-size-mb <integer MiB>
--mm-embedding-page-size <tokens, default 64>
```

The default cache size is zero, which leaves `ModelRunner.embedding_pool` as
`None`. See [`server_args.py`](../../../python/sgl_jax/srt/server_args.py) and
[`model_runner.py`](../../../python/sgl_jax/srt/model_executor/model_runner.py).

The pool is enabled only when all of the following hold:

- the model is multimodal and uses the in-model multimodal architecture;
- this is not a draft worker;
- the legacy/separate `--multimodal` path is not enabled;
- LoRA is disabled; and
- the worker is not a disaggregated decode worker.

This means passing `--multimodal` together with a nonzero pool size still
disables this particular pool. Gemma 4 should use the in-model multimodal path
without that legacy flag.

Unlike upstream's dictionary of independently allocated arrays, SGLang-JAX
allocates one fixed, replicated, device-resident page buffer:

```text
[num_pages, page_size, hidden * (1 + deepstack_dim)]
```

It maintains host-side LRU metadata keyed by the item hash. Misses run the
encoder and scatter the packed output into the pool; hits gather directly from
the pool. The implementation is in:

- [`embedding_pool.py`](../../../python/sgl_jax/srt/multimodal/in_model/embedding_pool.py)
- [`host_orchestration.py`](../../../python/sgl_jax/srt/multimodal/in_model/host_orchestration.py)

The configured embedding-pool bytes are subtracted from the memory available
for KV cache. Enabling the pool therefore trades KV capacity for reusable
multimodal embeddings.

### Capacity matching caveat

With Gemma 4 BF16 embeddings, hidden size 5,376, and the default 64-token page:

```text
one SGLang-JAX page = 64 * 5376 * 2 = 688,128 bytes = 0.65625 MiB
```

Two matching targets are possible:

| Target | SGLang-JAX setting | Result |
| --- | --- | --- |
| Match upstream raw 4,096-row / 42 MiB payload budget | `--mm-embedding-cache-size-mb 42` | 64 pages / 4,096 physical rows, but only 12 full 280-token images because each image consumes five pages |
| Match upstream capacity of 14 full 280-token images | `--mm-embedding-cache-size-mb 46` | 70 pages / 4,480 physical rows, enough for 14 five-page entries |

The second setting is the fairer cache-entry-capacity match for this exact
full-size-image workload. It is not expected to materially improve the
unique-image benchmark, but it would make an explicit repeated-image cache
ablation comparable.

## Evidence status

### Confirmed by runtime record

- Exact upstream server command and workload.
- Default versus pure-SPMD experiment IDs and metrics.
- The default upstream command does not disable either multimodal cache.
- Neither the inspected SGLang-JAX text experiment commands nor the final
  multimodal command passes `--mm-embedding-cache-size-mb`; the default is
  zero. Pure-text execution does not invoke the ViT or embedding pool.
- SGLang-JAX text and vision semantic smoke responses, both successful.
- Full SGLang-JAX multimodal r3 and r4 throughput and latency metrics, including
  the actual token counts that make r3 non-comparable.

### Confirmed by pinned source

- Processor-cache default and configured capacity formula.
- Encoder-cache default, token accounting, hash sharing, and LRU eviction.
- Gemma 4 output length, hidden width, and BF16 type.
- ViT batch sharding, replicated projected output, per-image slicing, and
  host-side hash insertion.
- TPU XLA/JIT implementation behind `cudagraph_mm_encoder=true`.
- Fresh random image generation per benchmark request.
- SGLang-JAX pool default, gating, paging, replication, and KV-budget
  subtraction.

### Not yet measured

- Runtime processor-cache and encoder-cache hit/miss counters for the 1,000
  request run.
- Static or runtime HBM attribution for the upstream encoder cache.
- Actual CPU RSS attributable to each 4 GiB processor-cache instance.
- A corrected-budget SGLang-JAX multimodal run with global request/prefill
  limits `1024 / 16384` and client token accounting identical to upstream.
- An isolated SGLang-JAX multimodal A/B with cache size 0 versus 46 MiB and a
  controlled repeated-image hit rate.
- A topology A/B that independently isolates process fan-out, ViT input
  sharding, replicated encoder output, scheduler behavior, and text-model
  execution. The current default versus pure-SPMD comparison changes the
  complete DP execution architecture and cannot attribute the 3.11x result to
  the ViT alone.

## Recommended follow-up experiments

1. Add cache lookup, hit, miss, eviction, resident-entry, and resident-byte
   counters to both implementations.
2. Run two multimodal datasets under otherwise identical settings:
   - 1,000 independently generated images, expecting approximately zero hits;
   - one identical image repeated 1,000 times, expecting a near-total hit rate
     after warmup.
3. Run SGLang-JAX with cache sizes 0 and 46 MiB on both datasets and record ViT
   invocation count as the correctness oracle.
4. Repeat the SGLang-JAX unique-image benchmark with global limits
   `1024 / 16384` and verify identical text, vision, and generated token counts
   against upstream before comparing implementation performance.
5. Capture an HBM profile to validate the theoretical replication bounds.
6. Profile default multiprocess and pure-SPMD upstream runs over a window that
   includes the ViT, image-to-text merge, prefill, and decode, then separate
   compiler time from steady-state device time.
