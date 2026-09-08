# Real-weight module precision

## First batch: MLP, MoE and mHC

Operator-approved scope: one Falcon H100 SGLang experiment followed by one TPU
JAX experiment, then stop and review results before attention experiments.

GPU source: sgl-project/sglang `8ef646a5c65bd2f8922483057dddc02e2b0de18c`.
JAX base: PR362 `09500f3b3e30c3cc18ddff691bbba79de84ed220`.
Checkpoint: Flash 0731, 43-layer config, 256 experts, H4096/I2048; load only
embedding, layer 0 and first learned-router layer's FFNs, and layer 0/root mHC.
Inputs: 128 random IDs, seed 17, through the real SGLang embedding. Export the
BF16 values as portable FP32 NumPy arrays with content hashes. Both executions
read the same hidden/residual/branch inputs; mHC post also shares gate/comb inputs.

This batch compares source-exact dequantized BF16 weights, BF16 activations, TP1
and EP1. GPU calls SGLang DeepseekV2MLP, DeepseekV2MoE (V4 settings, unquantized
Triton experts) and SGLang's own mHC PyTorch reference functions on CUDA. TPU
calls DeepseekV4SharedMLP, DeepseekV4MoE/EPMoE, Pallas mHC pre/post and the model's
reference head collapse. A separate forced-route EPMoE/FusedMoE output isolates
expert zero. Shared experts are tested separately from routed experts.

This is not native GPU MXFP4 vs TPU dynamic-FP8 acceptance, multi-card reduction
acceptance, or full-model serving acceptance. No precision threshold is relaxed:
we report max absolute error, relative L2, worst-token metrics, cosine and routing
ID mismatches for operator interpretation. Source weight payload hashes are also
matched, and TPU expert promotion is checked against independent source decoding.

GPU entry: `gpu_capture.py --model /models/deepseek-v4 --out /tmp/capture`.
TPU entry: `tpu_compare.py --model /models/deepseek-v4 --reference /tmp/capture --out /tmp/comparison`.
The GPU source-dequantization decoder is a format adapter; it is not a reference
implementation of MoE/MLP/mHC. Those computations remain in SGLang.


## Second batch: native attention

After reviewing the first batch, the operator approved one paired H100/TPU
attention batch. `gpu_attention_capture.py` calls the real SGLang MQALayer and
DeepseekV4AttnBackend for layer 2 (CSA/4) and layer 3 (HCA/128). Schedules use one
seed-17 embedding sequence: whole 128, whole 129, and continuous 63+65+1 decode.
Each schedule starts with fresh native pools. Projection weights use the same
source-dequantized BF16 baseline; H100 retains its native FP8-nope/BF16-rope cache.
The RoPE cache allocation is bounded to 256 positions, preserving original YaRN
parameters. Set `GPU_CAPTURE_SCRIPT=gpu_attention_capture.py` for `run_gpu.sh`.

`tpu_attention_compare.py` calls the production JAX modules and resource bridge;
HCA uses Pallas and CSA uses the existing XLA path. `--schedule` and `--layer`
select a diagnostic subset. Raw cache dumps retain physical expired rows for
auditing; SWA metrics compare only the final live window. Sources and source
weight hashes must match before interpreting metrics.

`tpu_attention_isolate.py` feeds GPU Q and native decoded cache to the existing
XLA attention function and, for HCA, native Pallas ragged attention. This isolates
attention arithmetic from projection, compressor and cache-quantization changes.
At <=129 tokens the CSA indexer has at most 32 entries for topk=512, so these cases
do not validate pruning or long-context selection.

`--diagnostic-csa-token-rope` is an explicitly labeled intervention: multiply the
CSA compressor's group coordinate by four before looking up RoPE. It is disabled
by default, restores the original function on exit, and does not edit production
code. The paired 2026-09-08 experiment isolated a group-id versus original-token
coordinate discrepancy; the intervention is evidence for a proposed fix, not a
claim that the model implementation has been fixed.

Falcon runs: GPU `exp-5fsnkttdcv`, TPU `exp-l8arc30kyy`. Supplemental isolation,
cache and coordinate-intervention runs reuse the same TPU allocation and retain
their own status/logs. Final metrics and limitations belong in the experiment
report; no model-quality or performance acceptance follows from this harness.
