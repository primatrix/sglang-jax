# DeepSeek V4 static expert FP8 checkpoint

The V4 loader can consume either the original mixed MXFP4 checkpoint or a static
expert FP8 export. Both paths produce the same expert parameters. The static path
skips MXFP4 conversion; it still reads and places the weights on devices.

## Export

Use the same source revision that was validated with load-time conversion:

```sh
python -m sgl_jax.srt.utils.quantization.deepseek_v4_static_fp8 \
  /models/deepseek-v4 /output/checkpoint \
  --source-revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
  --converter-revision "$(git rev-parse HEAD)"
```

`/output` may be a writable Falcon GCS mount. Use a new output directory. Add
`--resume` to continue the same export; source shard hashes, config/index hashes,
converter identity, and completed output shard checksums must match. Corruption
fails explicitly. There is no lossy export mode.

Conversion is bounded to one expert matrix at a time; unchanged tensor bytes are
streamed in 8 MiB blocks. Every shard is closed and read back before its receipt is
written. `static-fp8-complete.json` is published last. Do not use an incomplete
output as a model or publish it to a model registry.

## Format

`config.json` adds only
`sglang_jax_expert_format=sglang-jax-deepseek-v4-expert-fp8-per-channel-v1`.
The existing global blockwise FP8 configuration is preserved.

- Trunk routed `layers.L.ffn.experts.E.w{1,2,3}.weight`: E4M3FN, logical `[N,K]`.
- Its `.scale`: FP32 power-of-two scale `[N]`, broadcasting along K.
- Other tensors retain their source dtype and exact bytes, including non-expert
  blockwise FP8 scales and unused MTP tensors. MTP serving is not supported here.
- Source model assets, license, and original model card are retained. The output
  model card describes this specialized format. This export is not a generic
  upstream GPU/blockwise-FP8 checkpoint.

`conversion-source.json` records source revision, converter revision, and hashes
of every input shard. The completion file records every output shard checksum,
per-tensor payload checksums, and exact tensor/index coverage. Header and payload
sizes are checked on reading. Runtime validates completion, metadata, and shard
sizes; full shard SHA256 verification can be run separately with
`validate_static_checkpoint(path, verify_files=True)`.

## Serve and validate

Point the normal V4 serving command at the exported directory. The loader logs
`static FP8 load` for each expert projection. No conversion flag is needed.
Runtime parameter identity is tested on single-device and DP2/TP2 CPU meshes with
the conversion function forbidden during static loading.

A full artifact still needs a real TPU load and request check. Compare the same
prompts and generated token IDs against the original load-time-conversion run.
Use the checkpoint's `encoding/encoding_dsv4.py` for chat prompts; this source
checkpoint has no Hugging Face chat template, and the generic fallback is not an
equivalent prompt. Record read, load, compilation and request times separately.
