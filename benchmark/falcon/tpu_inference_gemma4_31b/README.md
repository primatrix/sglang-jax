# Gemma 4 31B tpu-inference Performance Test

This directory captures the Falcon configuration for the upstream
`tpu-inference` daily Performance Test of `google/gemma-4-31B-it`. It runs the
official text and synthetic multimodal cases on one TPU v7x host, compares the
default multiprocess-DP and pure-SPMD modes, and records both the upstream
Batched-RPA baseline and the default-RPA3 ablation. Raw server/client logs and
parsed metrics are preserved as Falcon artifacts when the server reaches the
benchmark phase.

## Document map

This Markdown is the self-contained source of truth for the run:

1. pinned software, hardware, and workload definitions;
2. Falcon submission and analysis procedure;
3. completed experiment IDs, artifacts, and measured results;
4. operational notes and known limitations; and
5. exact copies of the upstream case and every submitted Falcon manifest.

The adjacent JSON/YAML files duplicate the appendices so they can be validated
and submitted directly without extracting fenced code blocks.

The follow-up comparison against SGLang-JAX, including the corrected DP
scheduling contract, SWA eviction ablation, XProf evidence, and residual
kernel-path analysis, is recorded in
[`sglang_jax_text_performance_analysis.md`](sglang_jax_text_performance_analysis.md).

The complete cross-implementation measured-data ledger, multimodal cache, ViT
compilation, and DP/TP/SPMD source and runtime analysis is recorded in
[`gemma4_multimodal_cache_parallelism_analysis.md`](gemma4_multimodal_cache_parallelism_analysis.md).


## Pinned environment

| Component | Pinned value |
| --- | --- |
| Model | `google/gemma-4-31B-it` |
| TPU shape | v7x-8 visible devices / 4 chips / `2x2x1` |
| Falcon cluster | `tpu-training-antgroup` |
| Container | `vllm/vllm-tpu:nightly-20260810-a5596b2-f5bb701` |
| Container digest | `sha256:56e225be2b4e8466464f4ec938234499a74b2652704d7f561088548ffff3ff6d` |
| tpu-inference | `a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336` |
| vLLM | `f5bb701fa270f5c801f1572e1478b56f292d8dfc` |
| Upstream case | [`Gemma4-31B-dataset_custom-inlen_1024-outlen_500.json`](https://github.com/vllm-project/tpu-inference/blob/a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336/.buildkite/benchmark/cases/daily/Gemma4-31B-dataset_custom-inlen_1024-outlen_500.json) |
| Upstream runner | [`run_bm.sh`](https://github.com/vllm-project/tpu-inference/blob/a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336/.buildkite/benchmark/scripts/run_bm.sh) |

The dated nightly tag is used instead of the moving `nightly` tag. The source
commit is fetched explicitly before execution so the case and runner cannot
drift from the image build.

## Benchmark matrix

| Mode | Upstream case name | Backend / dataset | Workload | Prompts | Request rate |
| --- | --- | --- | --- | ---: | --- |
| Text | `Gemma4-31B-1k/500-GBS256` | `vllm` / `random` | 1024 input tokens, 500 output tokens | 1000 | `inf` |
| Multimodal | `Gemma4-31B-1k/1f512x512/500-GBS256` | `openai-chat` / `random-mm` | 1024 input tokens, one 512×512 image, 500 output tokens | 1000 | `inf` |

Both clients request the `ttft`, `tpot`, `itl`, and `e2el` percentiles, set
temperature to zero, and ignore EOS. `random-mm` uses generated JPEG inputs;
the multimodal result measures the image serving path and vision encoder cost,
not semantic image-understanding accuracy.

## Server configuration

The v7x-8 branch of the upstream case resolves to data parallel size 4 and
tensor parallel size 2. Both modes use:

```text
--model google/gemma-4-31B-it
--max-num-seqs 256
--max-num-batched-tokens 4096
--data-parallel-size 4
--tensor-parallel-size 2
--max-model-len 2048
--kv-cache-dtype fp8
--gpu-memory-utilization 0.9
--async-scheduling
--additional-config {Qwix FP8 weight and activation rules}
```

The upstream Batched-RPA baseline pins `VLLM_USE_V1=1`,
`MODEL_IMPL_TYPE=flax_nnx`, `USE_BATCHED_RPA_KERNEL=1`, and
`VLLM_ENGINE_READY_TIMEOUT_S=3600`. The RPA3 ablation changes only
`USE_BATCHED_RPA_KERNEL` to `0`, plus `TPU_MULTIPROCESS_DP=0` in the pure-SPMD
rows.

The multimodal case additionally sets:

```text
--additional-config {Qwix FP8 rules, mm-encoder-tp-mode=data}
--compilation-config {cudagraph_mm_encoder=true}
--disable-chunked-mm-input
```

## Parallel execution modes

The report now contains an A/B comparison of two DP execution modes. Model,
hardware, workload, TP/DP sizes, and all serving options are identical:

| Label | DP execution | Per-DP-rank execution | Effective topology |
| --- | --- | --- | --- |
| Default | Four vLLM engine processes | JAX SPMD TP=2 | 4-way multiprocess DP × 2-way SPMD TP |
| Pure SPMD | One process with JAX data axis 4 | JAX SPMD TP=2 | Single-process `data=4 × tensor=2` mesh |

The default manifests leave `TPU_MULTIPROCESS_DP` unset. At this pinned
tpu-inference commit, online `vllm serve` with DP > 1 resolves that setting to
`1`. The pure-SPMD manifests add only
`TPU_MULTIPROCESS_DP=0` to `server_command_options.env`; the runtime log
confirms that the variable is present in both actual server commands.

## Attention kernel modes

| Label | Environment | Imported implementation |
| --- | --- | --- |
| Batched RPA | `USE_BATCHED_RPA_KERNEL=1` | `kernels.experimental.batched_rpa.wrapper` |
| RPA3 | `USE_BATCHED_RPA_KERNEL=0` | `kernels.ragged_paged_attention.v3.kernel` |

The pinned implementation selects the branch at module import time. See the
pinned
[`attention_interface.py`](https://github.com/vllm-project/tpu-inference/blob/a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336/tpu_inference/layers/common/attention_interface.py#L43-L58).

This is an end-to-end serving-path comparison, not a kernel-only microbenchmark.
When Batched RPA is enabled, the TPU platform also forces the KV cache block
size to at least 256; setting the flag to `0` removes that override. See the
pinned
[`tpu_platform.py`](https://github.com/vllm-project/tpu-inference/blob/a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336/tpu_inference/platforms/tpu_platform.py#L270-L284).

## Falcon mapping

Falcon's v7x sizing counts framework-visible devices separately from chips:

```yaml
replica: 1
device_count: 8
device_type: v7x
device_topo: 2x2x1
```

The workload redirects temporary, Hugging Face, pip/uv, vLLM XLA, and JAX
compilation caches to `/tmp/tpu_logs` and disables core dumps. Each matrix cell
runs in its own Falcon experiment so a failure or compilation state cannot
contaminate another measurement.

The upstream JSON normally uploads logs to the tpu-inference CI GCS bucket and
reports metrics to its Spanner database. The Falcon runtime copy removes those
four CI routing variables and sets `UPLOAD_DB=false`; all performance parameters
remain unchanged. Falcon artifacts are the sole output destination for this
run.

## Run

Submit the four Batched-RPA baseline manifests:

```bash
falcon workflow exp submit \
  -f benchmark/falcon/tpu_inference_gemma4_31b/text-v7x8.yaml \
  --output json

falcon workflow exp submit \
  -f benchmark/falcon/tpu_inference_gemma4_31b/multimodal-v7x8.yaml \
  --output json

falcon workflow exp submit \
  -f benchmark/falcon/tpu_inference_gemma4_31b/text-spmd-v7x8.yaml \
  --output json

falcon workflow exp submit \
  -f benchmark/falcon/tpu_inference_gemma4_31b/multimodal-spmd-v7x8.yaml \
  --output json
```

Submit the four RPA3 ablation manifests:

```bash
falcon workflow exp submit \
  -f benchmark/falcon/tpu_inference_gemma4_31b/text-rpa3-v7x8.yaml \
  --output json

falcon workflow exp submit \
  -f benchmark/falcon/tpu_inference_gemma4_31b/multimodal-rpa3-v7x8.yaml \
  --output json

falcon workflow exp submit \
  -f benchmark/falcon/tpu_inference_gemma4_31b/text-spmd-rpa3-v7x8.yaml \
  --output json

falcon workflow exp submit \
  -f benchmark/falcon/tpu_inference_gemma4_31b/multimodal-spmd-rpa3-v7x8.yaml \
  --output json
```

The checked-in experiment names contain the original run timestamp. Before a
rerun, change each manifest's `name` to a unique value. After submission,
replace `exp_id` in the matching `analysis-*.yaml` with the returned experiment
ID before creating the analysis.

Wait for terminal state and inspect a failure through Falcon:

```bash
falcon workflow exp wait <exp_id> --timeout 8h --output json
falcon exp get <exp_id> --output json
falcon exp logs <exp_id> --output json
```

## Artifacts

Each successful experiment writes:

| Path | Contents |
| --- | --- |
| `upstream-case.json` | Unmodified upstream daily case snapshot |
| `falcon-runtime-case.json` | Single-case copy with external CI reporting disabled |
| `tpu-inference-commit.txt` | Runner source commit |
| `versions.txt` | Image, digest, and installed package versions |
| `runner.log` | Complete upstream runner output |
| `temp_logs/bm_log.txt` | Raw `vllm bench serve` output |
| `temp_logs/vllm_log.txt` | Raw server output |
| `gemma4-31b-*-v7x8.result` | Metrics parsed by the upstream report script |

## Results

Run date: 2026-08-10.

### Batched-RPA baseline

| Workload / parallel mode | Falcon experiment / analysis | Status | Request throughput | Output / total token throughput | Median / P99 TTFT | Median / P99 TPOT | Median / P99 E2EL |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Text / default | `exp-uxr7vn058s` / `an-26bg18bwur` | `SUCCEEDED` | 30.88 req/s | 15,440.38 / 47,062.28 tok/s | 5,815.62 / 11,142.44 ms | 46.02 / 51.06 ms | 28,743.68 / 32,189.85 ms |
| Text / pure SPMD | `exp-prrux3n44r` / `an-w7jvuy9i1y` | `SUCCEEDED` | 28.28 req/s | 14,140.83 / 43,101.25 tok/s | 7,698.47 / 12,945.25 ms | 46.84 / 51.85 ms | 31,069.52 / 35,196.74 ms |
| Multimodal / default | `exp-h8lobna2zl` / `an-j39zkdzm4u` | `SUCCEEDED` | 21.98 req/s | 10,988.92 / 39,459.83 tok/s | 11,107.87 / 36,983.25 ms | 53.31 / 58.43 ms | 37,679.14 / 45,011.83 ms |
| Multimodal / pure SPMD | `exp-8dxhtf3vw0` / `an-cby75i4qh6` | `SUCCEEDED` | 7.07 req/s | 3,534.87 / 12,693.27 tok/s | 84,338.80 / 132,589.53 ms | 98.37 / 107.13 ms | 133,440.56 / 140,976.40 ms |

#### Pure SPMD relative to the default mode

Positive latency percentages are regressions; negative throughput percentages
mean lower throughput.

| Metric | Text SPMD vs default | Multimodal SPMD vs default |
| --- | ---: | ---: |
| Request throughput | -8.42% | -67.83% |
| Output token throughput | -8.42% | -67.83% |
| Total token throughput | -8.42% | -67.83% |
| Median / P99 TTFT | +32.38% / +16.18% | +659.27% / +258.51% |
| Median / P99 TPOT | +1.78% / +1.55% | +84.52% / +83.35% |
| Median / P99 ITL | +1.36% / +3.13% | +0.86% / +131.29% |
| Median / P99 E2EL | +8.09% / +9.34% | +254.15% / +213.20% |

In this single-run A/B, pure SPMD is slower for both workloads and is
especially unfavorable for the synthetic multimodal case. The multimodal
configuration still uses `mm-encoder-tp-mode=data`; no XProf trace was
captured, so this result does not by itself attribute the regression to a
specific vision encoder, batching, or collective operation.

Within the default mode, adding one synthetic 512×512 image per request:

- reduces request throughput by 28.82% and output-token throughput by 28.83%;
- increases median TTFT by 91.00% and P99 TTFT by 231.91%;
- increases median/P99 TPOT by 15.84%/14.43%; and
- increases median/P99 end-to-end latency by 31.09%/39.83%.

The much larger TTFT change than TPOT change is consistent with image encoding
being concentrated in the request's prefill/first-token path. This is an
inference from the end-to-end metrics, not an operator-level attribution; no
XProf trace was collected in this Performance Test.

These are one successful run per workload and parallel mode. Use repeated runs
before treating small differences as a regression threshold. The Falcon
analyses are the canonical structured records; all four declare `summary.json`
and `report.md` through plugin version `pv-pmbmlfcoz2`.

## End-to-end reproduction procedure

Run all commands from the repository root. These files are direct experiment
submission manifests, not the higher-level orchestration schema accepted by
`falcon workflow validate` (which requires `schema_version`, `workflow`, and
`steps`). Give each experiment a new unique `name`, submit it, record the
returned `exp_id`, and wait for the artifact to become terminal:

```bash
falcon workflow exp submit \
  -f benchmark/falcon/tpu_inference_gemma4_31b/text-v7x8.yaml \
  --output json
falcon workflow exp wait <text_exp_id> --timeout 8h --output json
falcon workflow artifact get <text_exp_id> --output json

falcon workflow exp submit \
  -f benchmark/falcon/tpu_inference_gemma4_31b/multimodal-v7x8.yaml \
  --output json
falcon workflow exp wait <multimodal_exp_id> --timeout 8h --output json
falcon workflow artifact get <multimodal_exp_id> --output json

falcon workflow exp submit \
  -f benchmark/falcon/tpu_inference_gemma4_31b/text-spmd-v7x8.yaml \
  --output json
falcon workflow exp wait <text_spmd_exp_id> --timeout 8h --output json
falcon workflow artifact get <text_spmd_exp_id> --output json

falcon workflow exp submit \
  -f benchmark/falcon/tpu_inference_gemma4_31b/multimodal-spmd-v7x8.yaml \
  --output json
falcon workflow exp wait <multimodal_spmd_exp_id> --timeout 8h --output json
falcon workflow artifact get <multimodal_spmd_exp_id> --output json
```

Update the corresponding analysis manifest's `exp_id`, create each analysis,
and save the returned `analysis_id`:

```bash
falcon workflow analysis create \
  -f benchmark/falcon/tpu_inference_gemma4_31b/analysis-text.yaml \
  --output json
falcon workflow analysis wait <text_analysis_id> --timeout 30m --output json
falcon workflow analysis outputs <text_analysis_id> --output json
falcon workflow analysis cat <text_analysis_id> summary.json --output json
falcon workflow analysis cat <text_analysis_id> report.md --output json

falcon workflow analysis create \
  -f benchmark/falcon/tpu_inference_gemma4_31b/analysis-multimodal.yaml \
  --output json
falcon workflow analysis wait <multimodal_analysis_id> \
  --timeout 30m --output json
falcon workflow analysis outputs <multimodal_analysis_id> --output json
falcon workflow analysis cat <multimodal_analysis_id> summary.json --output json
falcon workflow analysis cat <multimodal_analysis_id> report.md --output json

falcon workflow analysis create \
  -f benchmark/falcon/tpu_inference_gemma4_31b/analysis-text-spmd.yaml \
  --output json
falcon workflow analysis wait <text_spmd_analysis_id> \
  --timeout 30m --output json
falcon workflow analysis outputs <text_spmd_analysis_id> --output json
falcon workflow analysis cat <text_spmd_analysis_id> summary.json --output json

falcon workflow analysis create \
  -f benchmark/falcon/tpu_inference_gemma4_31b/analysis-multimodal-spmd.yaml \
  --output json
falcon workflow analysis wait <multimodal_spmd_analysis_id> \
  --timeout 30m --output json
falcon workflow analysis outputs <multimodal_spmd_analysis_id> --output json
falcon workflow analysis cat <multimodal_spmd_analysis_id> \
  summary.json --output json
```

For a failed or stalled experiment, inspect both the experiment record and its
logs before changing the workload:

```bash
falcon exp get <exp_id> --output json
falcon exp logs <exp_id> --output json
```

## Falcon records

### Successful Batched-RPA records

| Mode | Experiment | Artifact | Analysis | Analysis plugin |
| --- | --- | --- | --- | --- |
| Text / default | `exp-uxr7vn058s` | `art-1t44htxy4y` | `an-26bg18bwur` | `pv-pmbmlfcoz2` |
| Text / pure SPMD | `exp-prrux3n44r` | `art-vpbdzhlrjs` | `an-w7jvuy9i1y` | `pv-pmbmlfcoz2` |
| Multimodal / default | `exp-h8lobna2zl` | `art-wqep29i4kc` | `an-j39zkdzm4u` | `pv-pmbmlfcoz2` |
| Multimodal / pure SPMD | `exp-8dxhtf3vw0` | `art-ess6cwljgp` | `an-cby75i4qh6` | `pv-pmbmlfcoz2` |

All four experiments and analyses reached `SUCCEEDED`. The plugin declares
`summary.json` and `report.md` as its outputs.

## Canonical structured results

Text analysis output from `an-26bg18bwur/summary.json`:

```json
{
  "result_file": "gemma4-31b-text-v7x8.result",
  "request_throughput_req_s": 30.88,
  "output_token_throughput_tok_s": 15440.38,
  "total_token_throughput_tok_s": 47062.28,
  "median_ttft_ms": 5815.62,
  "p99_ttft_ms": 11142.44,
  "median_tpot_ms": 46.02,
  "p99_tpot_ms": 51.06,
  "median_itl_ms": 34.63,
  "p99_itl_ms": 173.09,
  "median_e2el_ms": 28743.68,
  "p99_e2el_ms": 32189.85
}
```

Multimodal analysis output from `an-j39zkdzm4u/summary.json`:

```json
{
  "result_file": "gemma4-31b-mm-v7x8.result",
  "request_throughput_req_s": 21.98,
  "output_token_throughput_tok_s": 10988.92,
  "total_token_throughput_tok_s": 39459.83,
  "median_ttft_ms": 11107.87,
  "p99_ttft_ms": 36983.25,
  "median_tpot_ms": 53.31,
  "p99_tpot_ms": 58.43,
  "median_itl_ms": 31.50,
  "p99_itl_ms": 221.23,
  "median_e2el_ms": 37679.14,
  "p99_e2el_ms": 45011.83
}
```

Text pure-SPMD analysis output from `an-w7jvuy9i1y/summary.json`:

```json
{
  "result_file": "gemma4-31b-text-spmd-v7x8.result",
  "request_throughput_req_s": 28.28,
  "output_token_throughput_tok_s": 14140.83,
  "total_token_throughput_tok_s": 43101.25,
  "median_ttft_ms": 7698.47,
  "p99_ttft_ms": 12945.25,
  "median_tpot_ms": 46.84,
  "p99_tpot_ms": 51.85,
  "median_itl_ms": 35.10,
  "p99_itl_ms": 178.51,
  "median_e2el_ms": 31069.52,
  "p99_e2el_ms": 35196.74
}
```

Multimodal pure-SPMD analysis output from `an-cby75i4qh6/summary.json`:

```json
{
  "result_file": "gemma4-31b-mm-spmd-v7x8.result",
  "request_throughput_req_s": 7.07,
  "output_token_throughput_tok_s": 3534.87,
  "total_token_throughput_tok_s": 12693.27,
  "median_ttft_ms": 84338.80,
  "p99_ttft_ms": 132589.53,
  "median_tpot_ms": 98.37,
  "p99_tpot_ms": 107.13,
  "median_itl_ms": 31.77,
  "p99_itl_ms": 511.68,
  "median_e2el_ms": 133440.56,
  "p99_e2el_ms": 140976.40
}
```

## Operational notes and troubleshooting

- **Outer shell portability:** the first submission failed because Falcon's
  command wrapper is executed by POSIX `/bin/sh`, where
  `set -o pipefail` is not supported. The checked-in manifests therefore use
  `set -eu` in the outer command. The upstream runner remains a Bash script
  and retains its own Bash behavior.
- **Cold start:** model download, server start, and XLA compilation took roughly
  18 minutes before the benchmark client began. The experiment timeout is
  intentionally much longer than a warm serving run.
- **Pure-SPMD elapsed time:** Pod-running to terminal took approximately
  20m13s for text and 22m54s for multimodal, including server initialization,
  compilation, the 1000-prompt client run, reporting, and shutdown.
- **Log visibility:** files mounted through GCSFuse can appear delayed while a
  process still holds the file descriptor open. Treat Falcon experiment state
  and live logs as the progress signal; collect the mounted files after the
  workload exits.
- **Disk headroom:** the worker check observed about 59 GiB used and 414 GiB
  available, sufficient for the pinned image, model/cache data, source checkout,
  and benchmark logs.
- **External CI reporting:** the unmodified upstream case targets the
  tpu-inference CI GCS bucket and Spanner database. The runtime copy removes the
  four routing variables and forces `UPLOAD_DB=false`; Falcon artifacts are
  the only result destination.
- **`/models` GCSFuse mount:** Before the next server launch, inspect the mount
  shallowly and save the output as `models-preflight.txt` in the Falcon
  artifact. Avoid a recursive walk of the bucket. Only replace the Hugging Face
  model ID with a local path after confirming that the candidate directory has
  the model configuration, tokenizer assets, and every weight shard.
- **Interpretation:** each metric row is a single successful daily-case run.
  Repeat the cases and define an aggregation policy before using these numbers
  as a regression gate. No XProf trace was captured, so the TTFT interpretation
  is directional rather than operator-attributed.

Recommended preflight for a newly named rerun manifest, after `OUT` has been
created and before `vllm serve` starts:

```sh
if [ -d /models ]; then
  {
    echo '/models is a GCSFuse mount'
    ls -lah /models
    for candidate in /models/google/gemma-4-31B-it /models/gemma-4-31B-it; do
      if [ -d "$candidate" ]; then
        echo "candidate=$candidate"
        ls -lah "$candidate"
      fi
    done
  } > "$OUT/models-preflight.txt" 2>&1
else
  echo '/models is not mounted' > "$OUT/models-preflight.txt"
fi
```

This preflight is intentionally documented for the next run rather than added
retroactively to the submitted 2026-08-10 manifests below.

## Appendix A: exact upstream daily case

The following JSON is byte-for-byte identical to the file at the pinned
tpu-inference commit. It contains both the text and synthetic multimodal cases.

```json
{
  "global_env": {
    "GCP_PROJECT_ID": "cloud-tpu-inference-test",
    "GCS_BUCKET": "vllm-cb-storage2",
    "GCP_INSTANCE_ID": "vllm-bm-inst",
    "GCP_DATABASE_ID": "vllm-bm-bk-runs",
    "SERVER_WAIT_MINS": 180
  },
  "benchmark_cases": [
    {
      "case_name": "Gemma4-31B-1k/500-GBS256",
      "ci_queue": [
        "tpu_v6e_8_queue",
        "tpu_v7x_8_queue"
      ],
      "env": {
        "MODELTAG": "NEW",
        "EXPECTED_ETEL": 3600000,
        "INPUT_LEN": 1024,
        "OUTPUT_LEN": 500
      },
      "server_command_options": {
        "command_type": "vllm_serve",
        "env": {
          "VLLM_USE_V1": "1",
          "MODEL_IMPL_TYPE": "flax_nnx",
          "USE_BATCHED_RPA_KERNEL": "1",
          "VLLM_ENGINE_READY_TIMEOUT_S": "3600"
        },
        "args": {
          "model": "google/gemma-4-31B-it",
          "max-num-seqs": 256,
          "max-num-batched-tokens": 4096,
          "data-parallel-size": {
              "v6e-8": 2,
              "v7x-8": 4,
              "default": 1
          },
          "tensor-parallel-size": {
            "v6e-8": 4,
            "v7x-8": 2,
            "default": 1
          },
          "max-model-len": 2048,
          "kv-cache-dtype": "fp8",
          "gpu-memory-utilization": 0.9,
          "async-scheduling": true,
          "additional-config": "{\"quantization\": { \"qwix\": { \"rules\": [{ \"module_path\": \".*\", \"weight_qtype\": \"float8_e4m3fn\", \"act_qtype\": \"float8_e4m3fn\"}]}}}"
        }
      },
      "client_command_options": {
        "command_type": "vllm_bench_serve",
        "args": {
          "model": "google/gemma-4-31B-it",
          "backend": "vllm",
          "request-rate": "inf",
          "dataset-name": "random",
          "random-input-len": 1024,
          "random-output-len": 500,
          "num-prompts": 1000,
          "percentile-metrics": "ttft,tpot,itl,e2el",
          "ignore-eos": true,
          "temperature": 0
        }
      }
    },
    {
      "case_name": "Gemma4-31B-1k/1f512x512/500-GBS256",
      "ci_queue": [
        "tpu_v6e_8_queue",
        "tpu_v7x_8_queue"
      ],
      "env": {
        "MODELTAG": "NEW",
        "EXPECTED_ETEL": 3600000,
        "INPUT_LEN": 1024,
        "OUTPUT_LEN": 500
      },
      "server_command_options": {
        "command_type": "vllm_serve",
        "env": {
          "VLLM_USE_V1": "1",
          "MODEL_IMPL_TYPE": "flax_nnx",
          "USE_BATCHED_RPA_KERNEL": "1",
          "VLLM_ENGINE_READY_TIMEOUT_S": "3600"
        },
        "args": {
          "model": "google/gemma-4-31B-it",
          "max-num-seqs": 256,
          "max-num-batched-tokens": 4096,
          "data-parallel-size": {
              "v6e-8": 2,
              "v7x-8": 4,
              "default": 1
          },
          "tensor-parallel-size": {
            "v6e-8": 4,
            "v7x-8": 2,
            "default": 1
          },
          "max-model-len": 2048,
          "kv-cache-dtype": "fp8",
          "gpu-memory-utilization": 0.9,
          "async-scheduling": true,
          "additional-config": "{\"quantization\": { \"qwix\": { \"rules\": [{ \"module_path\": \".*\", \"weight_qtype\": \"float8_e4m3fn\", \"act_qtype\": \"float8_e4m3fn\"}]}}, \"mm-encoder-tp-mode\": \"data\"}",
          "compilation-config": "{\"cudagraph_mm_encoder\": true}",
          "disable-chunked-mm-input": true
        }
      },
      "client_command_options": {
        "command_type": "vllm_bench_serve",
        "args": {
          "model": "google/gemma-4-31B-it",
          "backend": "openai-chat",
          "request-rate": "inf",
          "dataset-name": "random-mm",
          "endpoint": "/v1/chat/completions",
          "random-mm-bucket-config": "{(512, 512, 1): 1.0}",
          "random-mm-limit-mm-per-prompt": "{\"image\": 1}",
          "random-input-len": 1024,
          "random-output-len": 500,
          "num-prompts": 1000,
          "percentile-metrics": "ttft,tpot,itl,e2el",
          "ignore-eos": true,
          "temperature": 0
        }
      }
    }
  ]
}
```

## Appendix B: exact text Falcon experiment manifest

```yaml
name: tpu-inference-gemma4-31b-text-20260810-100723z
exp_type: TRAINING
artifact_type: GCS
cluster_name: tpu-training-antgroup
priority: 0
model_version: google-gemma-4-31b-it
config: '{"kind":"serving-performance","mode":"text","model":"google/gemma-4-31B-it","hardware":"v7x-8","image":"vllm/vllm-tpu:nightly-20260810-a5596b2-f5bb701","tpu_inference_commit":"a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336","vllm_commit":"f5bb701fa270f5c801f1572e1478b56f292d8dfc"}'
tags: [benchmark, performance, tpu-inference, gemma4, text]
role_to_task_spec:
  worker:
    command: |
      set -eu
      export TMPDIR="${TMPDIR:-/tmp/tpu_logs/tmp}"
      export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/tpu_logs/pip-cache}"
      export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/tpu_logs/uv-cache}"
      export HF_HOME="${HF_HOME:-/tmp/tpu_logs/huggingface}"
      export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/tmp/tpu_logs/vllm-xla-cache}"
      export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/tpu_logs/jax-cache}"
      export UPLOAD_DB=false
      export MLCOMPASS_EXPORT_ENABLED=false
      export BUILDKITE=false
      export RUN_TYPE=FALCON
      export DEVICE=v7x-8
      export RECORD_ID=gemma4-31b-text-v7x8
      OUT="${ARTIFACT_LOCAL_DIR:-/tmp/falcon-artifacts}"
      mkdir -p "$OUT" "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
        "$HF_HOME" "$VLLM_XLA_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR"
      ulimit -c 0

      SRC=/tmp/tpu-inference-a5596b2
      git init "$SRC"
      git -C "$SRC" remote add origin https://github.com/vllm-project/tpu-inference.git
      git -C "$SRC" fetch --depth 1 origin a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336
      git -C "$SRC" -c advice.detachedHead=false checkout FETCH_HEAD
      cd "$SRC"

      CASE_SRC=.buildkite/benchmark/cases/daily/Gemma4-31B-dataset_custom-inlen_1024-outlen_500.json
      CASE_RUN=/tmp/gemma4-31b-text-v7x8.json
      python3 - "$CASE_SRC" "$CASE_RUN" <<'PY'
      import json
      import sys

      source, destination = sys.argv[1:]
      target = "Gemma4-31B-1k/500-GBS256"
      with open(source, encoding="utf-8") as handle:
          data = json.load(handle)
      for key in ("GCP_PROJECT_ID", "GCS_BUCKET", "GCP_INSTANCE_ID", "GCP_DATABASE_ID"):
          data["global_env"].pop(key, None)
      data["global_env"]["DEVICE"] = "v7x-8"
      data["benchmark_cases"] = [
          case for case in data["benchmark_cases"] if case["case_name"] == target
      ]
      if len(data["benchmark_cases"]) != 1:
          raise SystemExit(f"expected exactly one case named {target!r}")
      with open(destination, "w", encoding="utf-8") as handle:
          json.dump(data, handle, indent=2)
          handle.write("\n")
      PY

      cp "$CASE_SRC" "$OUT/upstream-case.json"
      cp "$CASE_RUN" "$OUT/falcon-runtime-case.json"
      git rev-parse HEAD > "$OUT/tpu-inference-commit.txt"
      {
        echo 'image=vllm/vllm-tpu:nightly-20260810-a5596b2-f5bb701'
        echo 'image_digest=sha256:56e225be2b4e8466464f4ec938234499a74b2652704d7f561088548ffff3ff6d'
        python3 - <<'PY'
      import importlib.metadata as md
      for package in ("vllm", "tpu-inference", "jax", "jaxlib", "libtpu"):
          try:
              print(f"{package}={md.version(package)}")
          except md.PackageNotFoundError:
              print(f"{package}=MISSING")
      PY
      } > "$OUT/versions.txt"

      if ARTIFACT_FOLDER="$OUT" \
        bash .buildkite/benchmark/scripts/run_bm.sh \
          "$CASE_RUN" 'Gemma4-31B-1k/500-GBS256' \
          > "$OUT/runner.log" 2>&1; then
        cat "$OUT/runner.log"
      else
        status=$?
        cat "$OUT/runner.log"
        exit "$status"
      fi
    replica: 1
    image: vllm/vllm-tpu:nightly-20260810-a5596b2-f5bb701
    device_count: 8
    device_type: v7x
    device_topo: 2x2x1
```

## Appendix C: exact multimodal Falcon experiment manifest

```yaml
name: tpu-inference-gemma4-31b-mm-20260810-100723z
exp_type: TRAINING
artifact_type: GCS
cluster_name: tpu-training-antgroup
priority: 0
model_version: google-gemma-4-31b-it
config: '{"kind":"serving-performance","mode":"multimodal","model":"google/gemma-4-31B-it","hardware":"v7x-8","image":"vllm/vllm-tpu:nightly-20260810-a5596b2-f5bb701","tpu_inference_commit":"a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336","vllm_commit":"f5bb701fa270f5c801f1572e1478b56f292d8dfc"}'
tags: [benchmark, performance, tpu-inference, gemma4, multimodal]
role_to_task_spec:
  worker:
    command: |
      set -eu
      export TMPDIR="${TMPDIR:-/tmp/tpu_logs/tmp}"
      export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/tpu_logs/pip-cache}"
      export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/tpu_logs/uv-cache}"
      export HF_HOME="${HF_HOME:-/tmp/tpu_logs/huggingface}"
      export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/tmp/tpu_logs/vllm-xla-cache}"
      export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/tpu_logs/jax-cache}"
      export UPLOAD_DB=false
      export MLCOMPASS_EXPORT_ENABLED=false
      export BUILDKITE=false
      export RUN_TYPE=FALCON
      export DEVICE=v7x-8
      export RECORD_ID=gemma4-31b-mm-v7x8
      OUT="${ARTIFACT_LOCAL_DIR:-/tmp/falcon-artifacts}"
      mkdir -p "$OUT" "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
        "$HF_HOME" "$VLLM_XLA_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR"
      ulimit -c 0

      SRC=/tmp/tpu-inference-a5596b2
      git init "$SRC"
      git -C "$SRC" remote add origin https://github.com/vllm-project/tpu-inference.git
      git -C "$SRC" fetch --depth 1 origin a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336
      git -C "$SRC" -c advice.detachedHead=false checkout FETCH_HEAD
      cd "$SRC"

      CASE_SRC=.buildkite/benchmark/cases/daily/Gemma4-31B-dataset_custom-inlen_1024-outlen_500.json
      CASE_RUN=/tmp/gemma4-31b-mm-v7x8.json
      python3 - "$CASE_SRC" "$CASE_RUN" <<'PY'
      import json
      import sys

      source, destination = sys.argv[1:]
      target = "Gemma4-31B-1k/1f512x512/500-GBS256"
      with open(source, encoding="utf-8") as handle:
          data = json.load(handle)
      for key in ("GCP_PROJECT_ID", "GCS_BUCKET", "GCP_INSTANCE_ID", "GCP_DATABASE_ID"):
          data["global_env"].pop(key, None)
      data["global_env"]["DEVICE"] = "v7x-8"
      data["benchmark_cases"] = [
          case for case in data["benchmark_cases"] if case["case_name"] == target
      ]
      if len(data["benchmark_cases"]) != 1:
          raise SystemExit(f"expected exactly one case named {target!r}")
      with open(destination, "w", encoding="utf-8") as handle:
          json.dump(data, handle, indent=2)
          handle.write("\n")
      PY

      cp "$CASE_SRC" "$OUT/upstream-case.json"
      cp "$CASE_RUN" "$OUT/falcon-runtime-case.json"
      git rev-parse HEAD > "$OUT/tpu-inference-commit.txt"
      {
        echo 'image=vllm/vllm-tpu:nightly-20260810-a5596b2-f5bb701'
        echo 'image_digest=sha256:56e225be2b4e8466464f4ec938234499a74b2652704d7f561088548ffff3ff6d'
        python3 - <<'PY'
      import importlib.metadata as md
      for package in ("vllm", "tpu-inference", "jax", "jaxlib", "libtpu"):
          try:
              print(f"{package}={md.version(package)}")
          except md.PackageNotFoundError:
              print(f"{package}=MISSING")
      PY
      } > "$OUT/versions.txt"

      if ARTIFACT_FOLDER="$OUT" \
        bash .buildkite/benchmark/scripts/run_bm.sh \
          "$CASE_RUN" 'Gemma4-31B-1k/1f512x512/500-GBS256' \
          > "$OUT/runner.log" 2>&1; then
        cat "$OUT/runner.log"
      else
        status=$?
        cat "$OUT/runner.log"
        exit "$status"
      fi
    replica: 1
    image: vllm/vllm-tpu:nightly-20260810-a5596b2-f5bb701
    device_count: 8
    device_type: v7x
    device_topo: 2x2x1
```

## Appendix D: exact text analysis manifest

```yaml
schema_version: 1
exp_id: exp-uxr7vn058s
spec:
  name: tpu-inference-serving-benchmark-summary
  command: /bin/sh "$FALCON_SCRIPT_PATH"
  script_content: |
    #!/usr/bin/env sh
    set -eu

    result_file=""
    for candidate in "$ARTIFACT_LOCAL_DIR"/*.result; do
      if [ -f "$candidate" ]; then
        result_file="$candidate"
        break
      fi
    done
    if [ -z "$result_file" ]; then
      echo "no top-level .result file found" >&2
      exit 1
    fi

    metric() {
      awk -F= -v key="$1" '$1 == key { print $2; exit }' "$result_file"
    }

    throughput="$(metric Throughput)"
    output_throughput="$(metric OutputTokenThroughput)"
    total_throughput="$(metric TotalTokenThroughput)"
    median_ttft="$(metric MedianTTFT)"
    p99_ttft="$(metric P99TTFT)"
    median_tpot="$(metric MedianTPOT)"
    p99_tpot="$(metric P99TPOT)"
    median_itl="$(metric MedianITL)"
    p99_itl="$(metric P99ITL)"
    median_e2el="$(metric MedianETEL)"
    p99_e2el="$(metric P99ETEL)"

    cat > "$RESULT_LOCAL_DIR/summary.json" <<EOF
    {
      "result_file": "$(basename "$result_file")",
      "request_throughput_req_s": $throughput,
      "output_token_throughput_tok_s": $output_throughput,
      "total_token_throughput_tok_s": $total_throughput,
      "median_ttft_ms": $median_ttft,
      "p99_ttft_ms": $p99_ttft,
      "median_tpot_ms": $median_tpot,
      "p99_tpot_ms": $p99_tpot,
      "median_itl_ms": $median_itl,
      "p99_itl_ms": $p99_itl,
      "median_e2el_ms": $median_e2el,
      "p99_e2el_ms": $p99_e2el
    }
    EOF

    cat > "$RESULT_LOCAL_DIR/report.md" <<EOF
    # tpu-inference serving benchmark

    Source: \`$(basename "$result_file")\`

    | Metric | Value |
    | --- | ---: |
    | Request throughput | $throughput req/s |
    | Output token throughput | $output_throughput tok/s |
    | Total token throughput | $total_throughput tok/s |
    | Median / P99 TTFT | $median_ttft / $p99_ttft ms |
    | Median / P99 TPOT | $median_tpot / $p99_tpot ms |
    | Median / P99 ITL | $median_itl / $p99_itl ms |
    | Median / P99 E2EL | $median_e2el / $p99_e2el ms |
    EOF
  outputs:
    - path: summary.json
      type: json
      required: true
      description: Structured vLLM serving benchmark metrics
    - path: report.md
      type: markdown
      required: true
      description: Human-readable vLLM serving benchmark summary
  params:
    workload: text
    model: google/gemma-4-31B-it
  resources:
    memory_limit: 256Mi
```

## Appendix E: exact multimodal analysis manifest

```yaml
schema_version: 1
exp_id: exp-h8lobna2zl
spec:
  name: tpu-inference-serving-benchmark-summary
  command: /bin/sh "$FALCON_SCRIPT_PATH"
  script_content: |
    #!/usr/bin/env sh
    set -eu

    result_file=""
    for candidate in "$ARTIFACT_LOCAL_DIR"/*.result; do
      if [ -f "$candidate" ]; then
        result_file="$candidate"
        break
      fi
    done
    if [ -z "$result_file" ]; then
      echo "no top-level .result file found" >&2
      exit 1
    fi

    metric() {
      awk -F= -v key="$1" '$1 == key { print $2; exit }' "$result_file"
    }

    throughput="$(metric Throughput)"
    output_throughput="$(metric OutputTokenThroughput)"
    total_throughput="$(metric TotalTokenThroughput)"
    median_ttft="$(metric MedianTTFT)"
    p99_ttft="$(metric P99TTFT)"
    median_tpot="$(metric MedianTPOT)"
    p99_tpot="$(metric P99TPOT)"
    median_itl="$(metric MedianITL)"
    p99_itl="$(metric P99ITL)"
    median_e2el="$(metric MedianETEL)"
    p99_e2el="$(metric P99ETEL)"

    cat > "$RESULT_LOCAL_DIR/summary.json" <<EOF
    {
      "result_file": "$(basename "$result_file")",
      "request_throughput_req_s": $throughput,
      "output_token_throughput_tok_s": $output_throughput,
      "total_token_throughput_tok_s": $total_throughput,
      "median_ttft_ms": $median_ttft,
      "p99_ttft_ms": $p99_ttft,
      "median_tpot_ms": $median_tpot,
      "p99_tpot_ms": $p99_tpot,
      "median_itl_ms": $median_itl,
      "p99_itl_ms": $p99_itl,
      "median_e2el_ms": $median_e2el,
      "p99_e2el_ms": $p99_e2el
    }
    EOF

    cat > "$RESULT_LOCAL_DIR/report.md" <<EOF
    # tpu-inference serving benchmark

    Source: \`$(basename "$result_file")\`

    | Metric | Value |
    | --- | ---: |
    | Request throughput | $throughput req/s |
    | Output token throughput | $output_throughput tok/s |
    | Total token throughput | $total_throughput tok/s |
    | Median / P99 TTFT | $median_ttft / $p99_ttft ms |
    | Median / P99 TPOT | $median_tpot / $p99_tpot ms |
    | Median / P99 ITL | $median_itl / $p99_itl ms |
    | Median / P99 E2EL | $median_e2el / $p99_e2el ms |
    EOF
  outputs:
    - path: summary.json
      type: json
      required: true
      description: Structured vLLM serving benchmark metrics
    - path: report.md
      type: markdown
      required: true
      description: Human-readable vLLM serving benchmark summary
  params:
    workload: multimodal
    model: google/gemma-4-31B-it
  resources:
    memory_limit: 256Mi
```

## Appendix F: exact text pure-SPMD Falcon experiment manifest

```yaml
name: tpu-inference-gemma4-31b-text-spmd-20260810-135317z
exp_type: TRAINING
artifact_type: GCS
cluster_name: tpu-training-antgroup
priority: 0
model_version: google-gemma-4-31b-it
config: '{"kind":"serving-performance","mode":"text","parallel_mode":"single-process-spmd","model":"google/gemma-4-31B-it","hardware":"v7x-8","image":"vllm/vllm-tpu:nightly-20260810-a5596b2-f5bb701","tpu_inference_commit":"a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336","vllm_commit":"f5bb701fa270f5c801f1572e1478b56f292d8dfc"}'
tags: [benchmark, performance, tpu-inference, gemma4, text, spmd]
role_to_task_spec:
  worker:
    command: |
      set -eu
      export TMPDIR="${TMPDIR:-/tmp/tpu_logs/tmp}"
      export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/tpu_logs/pip-cache}"
      export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/tpu_logs/uv-cache}"
      export HF_HOME="${HF_HOME:-/tmp/tpu_logs/huggingface}"
      export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/tmp/tpu_logs/vllm-xla-cache}"
      export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/tpu_logs/jax-cache}"
      export UPLOAD_DB=false
      export MLCOMPASS_EXPORT_ENABLED=false
      export BUILDKITE=false
      export RUN_TYPE=FALCON
      export DEVICE=v7x-8
      export RECORD_ID=gemma4-31b-text-spmd-v7x8
      OUT="${ARTIFACT_LOCAL_DIR:-/tmp/falcon-artifacts}"
      mkdir -p "$OUT" "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
        "$HF_HOME" "$VLLM_XLA_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR"
      ulimit -c 0

      SRC=/tmp/tpu-inference-a5596b2
      git init "$SRC"
      git -C "$SRC" remote add origin https://github.com/vllm-project/tpu-inference.git
      git -C "$SRC" fetch --depth 1 origin a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336
      git -C "$SRC" -c advice.detachedHead=false checkout FETCH_HEAD
      cd "$SRC"

      CASE_SRC=.buildkite/benchmark/cases/daily/Gemma4-31B-dataset_custom-inlen_1024-outlen_500.json
      CASE_RUN=/tmp/gemma4-31b-text-spmd-v7x8.json
      python3 - "$CASE_SRC" "$CASE_RUN" <<'PY'
      import json
      import sys

      source, destination = sys.argv[1:]
      target = "Gemma4-31B-1k/500-GBS256"
      with open(source, encoding="utf-8") as handle:
          data = json.load(handle)
      for key in ("GCP_PROJECT_ID", "GCS_BUCKET", "GCP_INSTANCE_ID", "GCP_DATABASE_ID"):
          data["global_env"].pop(key, None)
      data["global_env"]["DEVICE"] = "v7x-8"
      data["benchmark_cases"] = [
          case for case in data["benchmark_cases"] if case["case_name"] == target
      ]
      if len(data["benchmark_cases"]) != 1:
          raise SystemExit(f"expected exactly one case named {target!r}")
      selected = data["benchmark_cases"][0]
      selected["server_command_options"]["env"]["TPU_MULTIPROCESS_DP"] = "0"
      with open(destination, "w", encoding="utf-8") as handle:
          json.dump(data, handle, indent=2)
          handle.write("\n")
      PY

      cp "$CASE_SRC" "$OUT/upstream-case.json"
      cp "$CASE_RUN" "$OUT/falcon-runtime-case.json"
      git rev-parse HEAD > "$OUT/tpu-inference-commit.txt"
      {
        echo 'image=vllm/vllm-tpu:nightly-20260810-a5596b2-f5bb701'
        echo 'image_digest=sha256:56e225be2b4e8466464f4ec938234499a74b2652704d7f561088548ffff3ff6d'
        echo 'parallel_mode=single-process-spmd'
        echo 'TPU_MULTIPROCESS_DP=0'
        python3 - <<'PY'
      import importlib.metadata as md
      for package in ("vllm", "tpu-inference", "jax", "jaxlib", "libtpu"):
          try:
              print(f"{package}={md.version(package)}")
          except md.PackageNotFoundError:
              print(f"{package}=MISSING")
      PY
      } > "$OUT/versions.txt"

      if ARTIFACT_FOLDER="$OUT" \
        bash .buildkite/benchmark/scripts/run_bm.sh \
          "$CASE_RUN" 'Gemma4-31B-1k/500-GBS256' \
          > "$OUT/runner.log" 2>&1; then
        cat "$OUT/runner.log"
      else
        status=$?
        cat "$OUT/runner.log"
        exit "$status"
      fi
    replica: 1
    image: vllm/vllm-tpu:nightly-20260810-a5596b2-f5bb701
    device_count: 8
    device_type: v7x
    device_topo: 2x2x1
```

## Appendix G: exact multimodal pure-SPMD Falcon experiment manifest

```yaml
name: tpu-inference-gemma4-31b-mm-spmd-20260810-135317z
exp_type: TRAINING
artifact_type: GCS
cluster_name: tpu-training-antgroup
priority: 0
model_version: google-gemma-4-31b-it
config: '{"kind":"serving-performance","mode":"multimodal","parallel_mode":"single-process-spmd","model":"google/gemma-4-31B-it","hardware":"v7x-8","image":"vllm/vllm-tpu:nightly-20260810-a5596b2-f5bb701","tpu_inference_commit":"a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336","vllm_commit":"f5bb701fa270f5c801f1572e1478b56f292d8dfc"}'
tags: [benchmark, performance, tpu-inference, gemma4, multimodal, spmd]
role_to_task_spec:
  worker:
    command: |
      set -eu
      export TMPDIR="${TMPDIR:-/tmp/tpu_logs/tmp}"
      export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/tpu_logs/pip-cache}"
      export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/tpu_logs/uv-cache}"
      export HF_HOME="${HF_HOME:-/tmp/tpu_logs/huggingface}"
      export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/tmp/tpu_logs/vllm-xla-cache}"
      export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/tpu_logs/jax-cache}"
      export UPLOAD_DB=false
      export MLCOMPASS_EXPORT_ENABLED=false
      export BUILDKITE=false
      export RUN_TYPE=FALCON
      export DEVICE=v7x-8
      export RECORD_ID=gemma4-31b-mm-spmd-v7x8
      OUT="${ARTIFACT_LOCAL_DIR:-/tmp/falcon-artifacts}"
      mkdir -p "$OUT" "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
        "$HF_HOME" "$VLLM_XLA_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR"
      ulimit -c 0

      SRC=/tmp/tpu-inference-a5596b2
      git init "$SRC"
      git -C "$SRC" remote add origin https://github.com/vllm-project/tpu-inference.git
      git -C "$SRC" fetch --depth 1 origin a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336
      git -C "$SRC" -c advice.detachedHead=false checkout FETCH_HEAD
      cd "$SRC"

      CASE_SRC=.buildkite/benchmark/cases/daily/Gemma4-31B-dataset_custom-inlen_1024-outlen_500.json
      CASE_RUN=/tmp/gemma4-31b-mm-spmd-v7x8.json
      python3 - "$CASE_SRC" "$CASE_RUN" <<'PY'
      import json
      import sys

      source, destination = sys.argv[1:]
      target = "Gemma4-31B-1k/1f512x512/500-GBS256"
      with open(source, encoding="utf-8") as handle:
          data = json.load(handle)
      for key in ("GCP_PROJECT_ID", "GCS_BUCKET", "GCP_INSTANCE_ID", "GCP_DATABASE_ID"):
          data["global_env"].pop(key, None)
      data["global_env"]["DEVICE"] = "v7x-8"
      data["benchmark_cases"] = [
          case for case in data["benchmark_cases"] if case["case_name"] == target
      ]
      if len(data["benchmark_cases"]) != 1:
          raise SystemExit(f"expected exactly one case named {target!r}")
      selected = data["benchmark_cases"][0]
      selected["server_command_options"]["env"]["TPU_MULTIPROCESS_DP"] = "0"
      with open(destination, "w", encoding="utf-8") as handle:
          json.dump(data, handle, indent=2)
          handle.write("\n")
      PY

      cp "$CASE_SRC" "$OUT/upstream-case.json"
      cp "$CASE_RUN" "$OUT/falcon-runtime-case.json"
      git rev-parse HEAD > "$OUT/tpu-inference-commit.txt"
      {
        echo 'image=vllm/vllm-tpu:nightly-20260810-a5596b2-f5bb701'
        echo 'image_digest=sha256:56e225be2b4e8466464f4ec938234499a74b2652704d7f561088548ffff3ff6d'
        echo 'parallel_mode=single-process-spmd'
        echo 'TPU_MULTIPROCESS_DP=0'
        python3 - <<'PY'
      import importlib.metadata as md
      for package in ("vllm", "tpu-inference", "jax", "jaxlib", "libtpu"):
          try:
              print(f"{package}={md.version(package)}")
          except md.PackageNotFoundError:
              print(f"{package}=MISSING")
      PY
      } > "$OUT/versions.txt"

      if ARTIFACT_FOLDER="$OUT" \
        bash .buildkite/benchmark/scripts/run_bm.sh \
          "$CASE_RUN" 'Gemma4-31B-1k/1f512x512/500-GBS256' \
          > "$OUT/runner.log" 2>&1; then
        cat "$OUT/runner.log"
      else
        status=$?
        cat "$OUT/runner.log"
        exit "$status"
      fi
    replica: 1
    image: vllm/vllm-tpu:nightly-20260810-a5596b2-f5bb701
    device_count: 8
    device_type: v7x
    device_topo: 2x2x1
```

## Appendix H: exact text pure-SPMD analysis manifest

```yaml
schema_version: 1
exp_id: exp-prrux3n44r
spec:
  name: tpu-inference-serving-benchmark-summary
  command: /bin/sh "$FALCON_SCRIPT_PATH"
  script_content: |
    #!/usr/bin/env sh
    set -eu

    result_file=""
    for candidate in "$ARTIFACT_LOCAL_DIR"/*.result; do
      if [ -f "$candidate" ]; then
        result_file="$candidate"
        break
      fi
    done
    if [ -z "$result_file" ]; then
      echo "no top-level .result file found" >&2
      exit 1
    fi

    metric() {
      awk -F= -v key="$1" '$1 == key { print $2; exit }' "$result_file"
    }

    throughput="$(metric Throughput)"
    output_throughput="$(metric OutputTokenThroughput)"
    total_throughput="$(metric TotalTokenThroughput)"
    median_ttft="$(metric MedianTTFT)"
    p99_ttft="$(metric P99TTFT)"
    median_tpot="$(metric MedianTPOT)"
    p99_tpot="$(metric P99TPOT)"
    median_itl="$(metric MedianITL)"
    p99_itl="$(metric P99ITL)"
    median_e2el="$(metric MedianETEL)"
    p99_e2el="$(metric P99ETEL)"

    cat > "$RESULT_LOCAL_DIR/summary.json" <<EOF
    {
      "result_file": "$(basename "$result_file")",
      "request_throughput_req_s": $throughput,
      "output_token_throughput_tok_s": $output_throughput,
      "total_token_throughput_tok_s": $total_throughput,
      "median_ttft_ms": $median_ttft,
      "p99_ttft_ms": $p99_ttft,
      "median_tpot_ms": $median_tpot,
      "p99_tpot_ms": $p99_tpot,
      "median_itl_ms": $median_itl,
      "p99_itl_ms": $p99_itl,
      "median_e2el_ms": $median_e2el,
      "p99_e2el_ms": $p99_e2el
    }
    EOF

    cat > "$RESULT_LOCAL_DIR/report.md" <<EOF
    # tpu-inference serving benchmark

    Source: \`$(basename "$result_file")\`

    | Metric | Value |
    | --- | ---: |
    | Request throughput | $throughput req/s |
    | Output token throughput | $output_throughput tok/s |
    | Total token throughput | $total_throughput tok/s |
    | Median / P99 TTFT | $median_ttft / $p99_ttft ms |
    | Median / P99 TPOT | $median_tpot / $p99_tpot ms |
    | Median / P99 ITL | $median_itl / $p99_itl ms |
    | Median / P99 E2EL | $median_e2el / $p99_e2el ms |
    EOF
  outputs:
    - path: summary.json
      type: json
      required: true
      description: Structured vLLM serving benchmark metrics
    - path: report.md
      type: markdown
      required: true
      description: Human-readable vLLM serving benchmark summary
  params:
    workload: text-spmd
    model: google/gemma-4-31B-it
  resources:
    memory_limit: 256Mi
```

## Appendix I: exact multimodal pure-SPMD analysis manifest

```yaml
schema_version: 1
exp_id: exp-8dxhtf3vw0
spec:
  name: tpu-inference-serving-benchmark-summary
  command: /bin/sh "$FALCON_SCRIPT_PATH"
  script_content: |
    #!/usr/bin/env sh
    set -eu

    result_file=""
    for candidate in "$ARTIFACT_LOCAL_DIR"/*.result; do
      if [ -f "$candidate" ]; then
        result_file="$candidate"
        break
      fi
    done
    if [ -z "$result_file" ]; then
      echo "no top-level .result file found" >&2
      exit 1
    fi

    metric() {
      awk -F= -v key="$1" '$1 == key { print $2; exit }' "$result_file"
    }

    throughput="$(metric Throughput)"
    output_throughput="$(metric OutputTokenThroughput)"
    total_throughput="$(metric TotalTokenThroughput)"
    median_ttft="$(metric MedianTTFT)"
    p99_ttft="$(metric P99TTFT)"
    median_tpot="$(metric MedianTPOT)"
    p99_tpot="$(metric P99TPOT)"
    median_itl="$(metric MedianITL)"
    p99_itl="$(metric P99ITL)"
    median_e2el="$(metric MedianETEL)"
    p99_e2el="$(metric P99ETEL)"

    cat > "$RESULT_LOCAL_DIR/summary.json" <<EOF
    {
      "result_file": "$(basename "$result_file")",
      "request_throughput_req_s": $throughput,
      "output_token_throughput_tok_s": $output_throughput,
      "total_token_throughput_tok_s": $total_throughput,
      "median_ttft_ms": $median_ttft,
      "p99_ttft_ms": $p99_ttft,
      "median_tpot_ms": $median_tpot,
      "p99_tpot_ms": $p99_tpot,
      "median_itl_ms": $median_itl,
      "p99_itl_ms": $p99_itl,
      "median_e2el_ms": $median_e2el,
      "p99_e2el_ms": $p99_e2el
    }
    EOF

    cat > "$RESULT_LOCAL_DIR/report.md" <<EOF
    # tpu-inference serving benchmark

    Source: \`$(basename "$result_file")\`

    | Metric | Value |
    | --- | ---: |
    | Request throughput | $throughput req/s |
    | Output token throughput | $output_throughput tok/s |
    | Total token throughput | $total_throughput tok/s |
    | Median / P99 TTFT | $median_ttft / $p99_ttft ms |
    | Median / P99 TPOT | $median_tpot / $p99_tpot ms |
    | Median / P99 ITL | $median_itl / $p99_itl ms |
    | Median / P99 E2EL | $median_e2el / $p99_e2el ms |
    EOF
  outputs:
    - path: summary.json
      type: json
      required: true
      description: Structured vLLM serving benchmark metrics
    - path: report.md
      type: markdown
      required: true
      description: Human-readable vLLM serving benchmark summary
  params:
    workload: multimodal-spmd
    model: google/gemma-4-31B-it
  resources:
    memory_limit: 256Mi
```
