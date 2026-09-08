# First batch: real-weight module precision

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
