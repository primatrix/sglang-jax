"""Real-weight SGLang module reference on one H100; no serving process."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import time
from pathlib import Path

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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_grad_enabled(False)
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE
    from sglang.srt.models.deepseek_v4 import hc_head_torch
    from sglang.kernels.ops.layernorm.mhc import _mhc_pre_torch, _mhc_post_torch

    set_global_server_args_for_scheduler(
        ServerArgs(
            model_path=args.model,
            device="cuda",
            tp_size=1,
            ep_size=1,
            disable_shared_experts_fusion=True,
            moe_runner_backend="triton",
        )
    )
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29571",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    initialize_moe_config()
    checkpoint = Checkpoint(args.model)
    raw = checkpoint.config
    names = {f.name for f in dataclasses.fields(DeepSeekV4Config)}
    cfg = DeepSeekV4Config(**{k: v for k, v in raw.items() if k in names})
    for k, v in raw.items():
        setattr(cfg, k, v)
    cfg.quantization_config = None
    cfg.router_fp32 = True
    cfg.n_shared_experts = 0  # Compare routed and shared branches independently.
    cap = Capture(args.out)
    metadata = {
        "source": os.environ.get("SGLANG_SHA"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "checkpoint": checkpoint.identity,
        "tokens": 128,
        "seed": 17,
        "tp": 1,
        "ep": 1,
        "weight_path": "source-dequantized-BF16",
        "tf32": False,
        "activation_quantization": False,
        "mhc_path": "SGLang _mhc_pre_torch/_mhc_post_torch/hc_head_torch on CUDA",
        "cases": [],
        "errors": [],
    }
    out = Path(args.out)

    def tensor(a, dtype=torch.bfloat16):
        return torch.from_numpy(np.asarray(a, dtype=np.float32).copy()).to(
            device="cuda", dtype=dtype
        )

    def run(name, fn):
        start = time.monotonic()
        try:
            fn()
            torch.cuda.synchronize()
            metadata["cases"].append(
                {"name": name, "status": "captured", "seconds": time.monotonic() - start}
            )
            print("GPU_CASE_CAPTURED", name, flush=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            metadata["errors"].append({"name": name, "type": type(e).__name__, "message": str(e)})
        finally:
            (out / "run.json").write_text(json.dumps(metadata, indent=2))

    ids_np = np.random.default_rng(17).integers(0, cfg.vocab_size, size=128, dtype=np.int32)
    ids = torch.from_numpy(ids_np.astype(np.int64)).cuda()
    with torch.device("cuda"):
        embed = VocabParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size, params_dtype=torch.bfloat16, enable_tp=False
        )
    ew = tensor(checkpoint.read("embed.weight"))
    assert embed.weight.shape == ew.shape
    embed.weight.copy_(ew)
    hidden = embed(ids).contiguous()
    cap.save("input/token_ids", ids)
    cap.save("input/hidden", hidden)
    residual = hidden[:, None, :].expand(-1, cfg.hc_mult, -1).contiguous()
    cap.save("input/residual", residual)
    branch = torch.roll(hidden, shifts=1, dims=0).contiguous()
    cap.save("input/branch", branch)
    del ew, embed
    gc.collect()
    torch.cuda.empty_cache()

    def mlp_case(layer_id=0, expert=None):
        name = f"mlp/l{layer_id}" if expert is None else f"expert/l{layer_id}/e{expert}"
        stem = (
            f"layers.{layer_id}.ffn.shared_experts"
            if expert is None
            else f"layers.{layer_id}.ffn.experts.{expert}"
        )
        reader = checkpoint.block_fp8 if expert is None else checkpoint.expert_reference
        with torch.device("cuda"):
            module = DeepseekV2MLP(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                "silu",
                quant_config=None,
                tp_rank=0,
                tp_size=1,
                reduce_results=False,
                swiglu_limit=cfg.swiglu_limit,
            )
        w1, w3, w2 = [tensor(reader(stem + "." + key)) for key in ("w1", "w3", "w2")]
        module.gate_up_proj.weight.copy_(torch.cat([w1, w3]))
        module.down_proj.weight.copy_(w2)
        cap.save(name + "/output", module(hidden))
        cap.save(name + "/gate_up", module.gate_up_proj(hidden)[0])
        del module, w1, w2, w3
        gc.collect()
        torch.cuda.empty_cache()

    run("mlp/l0", lambda: mlp_case())
    run("expert/l0/e0", lambda: mlp_case(expert=0))

    def mhc_case():
        prefix = "layers.0."
        for family in ("attn", "ffn"):
            stem = prefix + "hc_" + family
            fn, scale, base = [
                tensor(checkpoint.read(stem + "_" + key), torch.float32)
                for key in ("fn", "scale", "base")
            ]
            post, comb, x = _mhc_pre_torch(
                residual,
                fn,
                scale,
                base,
                cfg.rms_norm_eps,
                cfg.hc_eps,
                cfg.hc_eps,
                2.0,
                cfg.hc_sinkhorn_iters,
            )
            cap.save(f"mhc/{family}/pre", x)
            cap.save(f"mhc/{family}/post_gate", post.squeeze(-1))
            cap.save(f"mhc/{family}/comb", comb)
            cap.save(f"mhc/{family}/post", _mhc_post_torch(branch, residual, post, comb))
        fn, scale, base = [
            tensor(checkpoint.read("hc_head_" + key), torch.float32)
            for key in ("fn", "scale", "base")
        ]
        cap.save(
            "mhc/head",
            hc_head_torch(residual, fn, scale, base, norm_eps=cfg.rms_norm_eps, hc_eps=cfg.hc_eps),
        )

    run("mhc", mhc_case)

    def moe_case(layer_id):
        with torch.device("cuda"):
            module = DeepseekV2MoE(cfg, layer_id=layer_id, quant_config=None, is_deepseek_v4=True)
        module.gate.weight.copy_(
            tensor(checkpoint.read(f"layers.{layer_id}.ffn.gate.weight"), torch.float32)
        )
        if module.is_hash:
            module.topk.tid2eid.copy_(
                torch.from_numpy(checkpoint.read(f"layers.{layer_id}.ffn.gate.tid2eid")).cuda()
            )
        else:
            module.gate.e_score_correction_bias.copy_(
                tensor(checkpoint.read(f"layers.{layer_id}.ffn.gate.bias"), torch.float32)
            )
        for expert in range(cfg.n_routed_experts):
            stem = f"layers.{layer_id}.ffn.experts.{expert}"
            w1, w3, w2 = [
                tensor(checkpoint.expert_reference(stem + "." + key)) for key in ("w1", "w3", "w2")
            ]
            module.experts.w13_weight[expert].copy_(torch.cat([w1, w3]))
            module.experts.w2_weight[expert].copy_(w2)
            if expert % 32 == 0:
                print("GPU_EXPERT_LOAD", layer_id, expert, flush=True)
        name = f"moe/l{layer_id}"
        logits = module.gate(hidden)
        route = module.topk(hidden, logits, **({"input_ids": ids} if module.is_hash else {}))
        cap.save(name + "/logits", logits)
        cap.save(name + "/ids", route.topk_ids)
        cap.save(name + "/weights", route.topk_weights)
        forced = (
            torch.arange(cfg.num_experts_per_tok, dtype=torch.int32, device="cuda")
            .expand(hidden.shape[0], -1)
            .contiguous()
        )
        forced_weights = torch.zeros_like(route.topk_weights)
        forced_weights[:, 0] = 1
        isolated_route = type(route)(
            topk_weights=forced_weights, topk_ids=forced, router_logits=logits
        )
        cap.save(name + "/expert0", module.experts(hidden.clone(), isolated_route))
        cap.save(
            name + "/output",
            module.forward_normal(
                hidden.clone(), input_ids=ids, input_ids_global=ids, skip_shared_experts=True
            ),
        )
        metadata["moe_implementation"] = (
            type(module.experts).__module__ + "." + type(module.experts).__name__
        )
        del module, w1, w2, w3, route
        gc.collect()
        torch.cuda.empty_cache()

    run("moe/l0", lambda: moe_case(0))
    run(f"moe/l{cfg.num_hash_layers}", lambda: moe_case(cfg.num_hash_layers))
    (out / "weights.json").write_text(json.dumps(checkpoint.digests, indent=2))
    (out / "run.json").write_text(json.dumps(metadata, indent=2))
    print(
        "GPU_REFERENCE_COMPLETE",
        len(metadata["cases"]),
        "errors",
        len(metadata["errors"]),
        flush=True,
    )
    if metadata["errors"]:
        raise RuntimeError("Some GPU reference cases failed; inspect run.json")


if __name__ == "__main__":
    main()
