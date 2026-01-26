# Debugging `fused_ep_moe` TPU Mosaic BoundsCheck

This repo currently has a TPU driver crash (Mosaic DMA BoundsCheck) during `fused_ep_moe` execution/compilation when using sparse routing (`topk_ids/topk_weights`), and sometimes only for larger token counts (e.g. `num_tokens >= 512`) and/or shared expert enabled.

## Minimal repro

Run the TPU-only unit test:

```bash
PYTHONPATH=python python -m sgl_jax.test.kernels.fused_moe_boundscheck_repro_test
```

Default parameters aim to be close to the failing kernel signature:
- `top_k=8`
- `hidden_size=1024`
- `intermediate_size=512`
- `num_tokens=512` (auto-aligned to `ep_size`)
- shared expert enabled

## Control-variable sweep

Run a small matrix to bisect the failing path:

```bash
SGLANG_FUSED_MOE_BOUNDSCHECK_SWEEP=1 \
PYTHONPATH=python python -m sgl_jax.test.kernels.fused_moe_boundscheck_repro_test
```

This toggles:
- `has_shared_expert`: `False/True`
- `prepad_topk`: `False/True` (pad `(num_tokens, top_k)` → `(num_tokens, 128)` on the host before calling the kernel)
- `num_tokens`: `256 / base / 1024`

Interpretation:
- Fails only when `has_shared_expert=True` → focus on shared-expert HBM→VMEM DMAs (`w*_shared` / shared-expert token staging).
- Fails only when `prepad_topk=False` → focus on the `pad` materialization / DMA alignment of top-k buffers.
- Fails only at larger `num_tokens` → focus on token tiling (`bt/bts/btc`) and DMA stride/step math.

## Helpful XLA dumps

If you can reproduce in the unit test, dumping HLO can help map the failing DMA to the exact operand:

```bash
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"
SGLANG_FUSED_MOE_BOUNDSCHECK_SWEEP=0 \
PYTHONPATH=python python -m sgl_jax.test.kernels.fused_moe_boundscheck_repro_test
```

Then inspect `/tmp/xla_dump` for the corresponding fused computation name (the crash log usually includes an HLO name similar to `fused-moe-k_8-...`).
