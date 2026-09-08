"""Real SGLang MQALayer and native FP8 cache on H100, paired attention capture."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

# The non-quantized weight baseline uses the real BF16 projection path.
os.environ["SGLANG_OPT_FP8_WO_A_GEMM"] = "0"
os.environ["SGLANG_OPT_FUSE_WQA_WKV"] = "0"
os.environ["SGLANG_OPT_USE_ONLINE_COMPRESS"] = "0"

import numpy as np
import torch
from common import Capture, Checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from sglang.srt.models.deepseek_v4 import MQALayer
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged

    server = ServerArgs(
        model_path=args.model,
        device="cuda",
        tp_size=1,
        ep_size=1,
        disable_shared_experts_fusion=True,
        moe_runner_backend="triton",
    )
    set_global_server_args_for_scheduler(server)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29572",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    initialize_moe_config()
    cp = Checkpoint(args.model)
    names = {f.name for f in dataclasses.fields(DeepSeekV4Config)}
    cfg = DeepSeekV4Config(**{k: v for k, v in cp.config.items() if k in names})
    for k, v in cp.config.items():
        setattr(cfg, k, v)
    cfg.quantization_config = None
    cfg.max_position_embeddings = 256
    cap = Capture(args.out)
    meta = dict(
        source=os.environ.get("SGLANG_SHA"),
        checkpoint=cp.identity,
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        seed=17,
        tp=1,
        ep=1,
        weight_path="source-dequantized-BF16",
        kv_cache="native-fp8",
        rope_cache_length=256,
        cases=[],
        errors=[],
    )

    def tensor(a, dtype=torch.bfloat16):
        return torch.from_numpy(np.asarray(a, dtype=np.float32).copy()).to("cuda", dtype)

    def copy(param, a):
        value = tensor(a, param.dtype)
        assert param.shape == value.shape, (param.shape, value.shape)
        param.copy_(value)

    ids_np = np.random.default_rng(17).integers(0, cfg.vocab_size, size=129, dtype=np.int32)
    ids = torch.from_numpy(ids_np.astype(np.int64)).cuda()
    with torch.device("cuda"):
        embed = VocabParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size, params_dtype=torch.bfloat16, enable_tp=False
        )
    copy(embed.weight, cp.read("embed.weight"))
    hidden = embed(ids).contiguous()
    cap.save("input/token_ids", ids)
    cap.save("input/hidden", hidden)
    del embed
    gc.collect()
    torch.cuda.empty_cache()

    def load_compressor(module, stem):
        copy(
            module.wkv_gate.weight,
            np.concatenate(
                [cp.read(stem + ".wkv.weight"), cp.read(stem + ".wgate.weight")], axis=0
            ),
        )
        module.load_ape_weight(module.ape, tensor(cp.read(stem + ".ape"), module.ape.dtype))
        copy(module.norm.weight, cp.read(stem + ".norm.weight"))

    for layer in (2, 3):
        ratio = cfg.compress_ratios[layer]
        with torch.device("cuda"):
            module = MQALayer(cfg, layer, quant_config=None)
        stem = f"layers.{layer}.attn"
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            copy(getattr(module, name).weight, cp.block_fp8(stem + "." + name))
        for name in ("q_norm", "kv_norm"):
            copy(getattr(module, name).weight, cp.read(stem + "." + name + ".weight"))
        copy(module.attn_sink, cp.read(stem + ".attn_sink"))
        load_compressor(module.compressor, stem + ".compressor")
        if module.indexer is not None:
            copy(module.indexer.wq_b.weight, cp.block_fp8(stem + ".indexer.wq_b"))
            copy(module.indexer.weights_proj.weight, cp.read(stem + ".indexer.weights_proj.weight"))
            load_compressor(module.indexer.compressor, stem + ".indexer.compressor")
        for schedule, lengths in (("whole128", [128]), ("whole129", [129]), ("split", [63, 65, 1])):
            prefix = f"attention/l{layer}/{schedule}"
            try:
                pool = DeepSeekV4TokenToKVPool(
                    max_num_reqs=1,
                    swa_size=1024,
                    c4_size=256,
                    c128_size=8,
                    c4_state_pool_size=128,
                    c128_state_pool_size=256,
                    page_size=256,
                    swa_page_size=128,
                    dtype=torch.float8_e4m3fn,
                    c4_state_dtype=torch.float32,
                    c128_state_dtype=torch.float32,
                    qk_nope_head_dim=448,
                    qk_rope_head_dim=64,
                    indexer_head_dim=128,
                    layer_num=cfg.num_hidden_layers,
                    device="cuda",
                    enable_memory_saver=False,
                    compression_ratios=cfg.compress_ratios,
                    start_layer=layer,
                    end_layer=layer + 1,
                )
                pool.register_mapping(torch.arange(2048, device="cuda", dtype=torch.int64))
                table = torch.zeros((2, 256), device="cuda", dtype=torch.int32)
                table[0] = torch.arange(256, 512, device="cuda", dtype=torch.int32)
                req_pool = SimpleNamespace(req_to_token=table, size=1)
                runner = SimpleNamespace(
                    device="cuda",
                    model_config=SimpleNamespace(
                        context_len=256,
                        head_dim=512,
                        v_head_dim=512,
                        hf_text_config=cfg,
                        hf_config=cfg,
                    ),
                    page_size=256,
                    req_to_token_pool=req_pool,
                    token_to_kv_pool=pool,
                    hisparse_coordinator=None,
                    server_args=server,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    is_draft_worker=False,
                )
                backend = DeepseekV4AttnBackend(runner)
                current = {"name": ""}
                original = backend.forward

                def forward(*a, **kw):
                    q = a[0] if a else kw["q"]
                    cap.save(current["name"] + "/q", q)
                    value = original(*a, **kw)
                    cap.save(current["name"] + "/raw_attention", value.clone())
                    return value

                backend.forward = forward
                handles = []
                for name in ("wq_a", "q_norm", "wq_b", "wkv", "wo_b"):

                    def hook(m, a, value, name=name):
                        cap.save(
                            current["name"] + "/" + name,
                            value[0] if isinstance(value, tuple) else value,
                        )

                    handles.append(getattr(module, name).register_forward_hook(hook))
                handles.append(
                    module.wo_b.register_forward_pre_hook(
                        lambda m, a: cap.save(current["name"] + "/wo_a", a[0])
                    )
                )
                start = 0
                for step, count in enumerate(lengths):
                    end = start + count
                    current["name"] = f"{prefix}/step{step}"
                    positions = torch.arange(start, end, device="cuda", dtype=torch.int64)

                    def ints(v):
                        return torch.tensor(v, device="cuda", dtype=torch.int32)

                    mode = ForwardMode.DECODE if count == 1 and start else ForwardMode.EXTEND
                    fb = ForwardBatch(
                        forward_mode=mode,
                        batch_size=1,
                        input_ids=ids[start:end],
                        req_pool_indices=ints([0]),
                        seq_lens=ints([end]),
                        out_cache_loc=positions + 256,
                        seq_lens_sum=end,
                    )
                    fb.positions = positions
                    fb.seq_lens_cpu = torch.tensor([end], dtype=torch.int32, device="cpu")
                    fb.extend_num_tokens = count
                    fb.extend_seq_lens = ints([count])
                    fb.extend_prefix_lens = ints([start])
                    fb.extend_start_loc = ints([0])
                    fb.extend_seq_lens_cpu = [count]
                    fb.extend_prefix_lens_cpu = [start]
                    fb.global_num_token_non_padded_cpu = count
                    with forward_context(ForwardContext(backend)):
                        backend.init_forward_metadata(fb)
                        value = module(hidden[start:end].contiguous(), positions, fb)
                    cap.save(current["name"] + "/output", value)
                    cap.save(current["name"] + "/positions", positions)
                    # Actual native cache values, decoded by the SGLang cache kernel.
                    sw = dequantize_k_cache_paged(
                        pool.get_swa_key_buffer_radix(layer),
                        torch.arange(256, 256 + end, device="cuda", dtype=torch.int32),
                        128,
                    )
                    cap.save(current["name"] + "/swa_cache", sw)
                    if end // ratio:
                        ck = dequantize_k_cache_paged(
                            pool.get_extra_key_buffer(layer),
                            torch.arange(
                                256 // ratio,
                                256 // ratio + end // ratio,
                                device="cuda",
                                dtype=torch.int32,
                            ),
                            256 // ratio,
                        )
                        cap.save(current["name"] + "/compressed_cache", ck)
                    md = backend.forward_metadata
                    for name, value in vars(md).items():
                        if hasattr(value, "swa_page_indices"):
                            for field in (
                                "swa_page_indices",
                                "swa_topk_lengths",
                                "c4_sparse_page_indices",
                                "c4_topk_lengths",
                                "c128_page_indices",
                                "c128_topk_lengths",
                            ):
                                array = getattr(value, field, None)
                                if isinstance(array, torch.Tensor):
                                    cap.save(current["name"] + "/" + field, array)
                    torch.cuda.synchronize()
                    print("ATTENTION_STEP_CAPTURED", current["name"], flush=True)
                    start = end
                meta["cases"].append(dict(name=prefix, status="captured", chunks=lengths))
            except Exception as e:
                traceback.print_exc()
                meta["errors"].append(dict(name=prefix, type=type(e).__name__, message=str(e)))
            finally:
                for handle in locals().get("handles", []):
                    handle.remove()
                Path(args.out, "run.json").write_text(json.dumps(meta, indent=2))
                Path(args.out, "weights.json").write_text(json.dumps(cp.digests, indent=2))
        del module
        gc.collect()
        torch.cuda.empty_cache()
    print("GPU_ATTENTION_COMPLETE", len(meta["cases"]), "errors", len(meta["errors"]), flush=True)
    if meta["errors"]:
        raise RuntimeError("Attention reference cases failed; inspect run.json")


if __name__ == "__main__":
    main()
