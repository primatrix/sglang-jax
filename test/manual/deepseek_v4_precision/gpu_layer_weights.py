"""Load decoder-only weights and observe native SGLang layer boundaries."""

import torch


def load_layer_weights(block, cp, cfg, layer, copy, tensor):
    stem = f"layers.{layer}"
    for family in ("attn", "ffn"):
        for part in ("fn", "base", "scale"):
            name = f"hc_{family}_{part}"
            copy(getattr(block, name), cp.read(stem + "." + name))
    for source, target in (
        ("attn_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
    ):
        copy(getattr(block, target).weight, cp.read(stem + "." + source + ".weight"))
    moe = block.mlp
    copy(moe.gate.weight, cp.read(stem + ".ffn.gate.weight"))
    if moe.is_hash:
        moe.topk.tid2eid.copy_(torch.from_numpy(cp.read(stem + ".ffn.gate.tid2eid")).cuda())
    else:
        copy(moe.gate.e_score_correction_bias, cp.read(stem + ".ffn.gate.bias"))
    shared = moe.shared_experts
    w1, w3 = [tensor(cp.block_fp8(stem + ".ffn.shared_experts." + k)) for k in ("w1", "w3")]
    shared.gate_up_proj.weight.copy_(torch.cat([w1, w3]))
    copy(shared.down_proj.weight, cp.block_fp8(stem + ".ffn.shared_experts.w2"))
    for expert in range(cfg.n_routed_experts):
        prefix = stem + f".ffn.experts.{expert}"
        w1, w3 = [tensor(cp.expert_reference(prefix + "." + k)) for k in ("w1", "w3")]
        moe.experts.w13_weight[expert].copy_(torch.cat([w1, w3]))
        copy(moe.experts.w2_weight[expert], cp.expert_reference(prefix + ".w2"))
        if expert % 32 == 0:
            print("GPU_LAYER_EXPERT_LOAD", layer, expert, flush=True)


def attach_layer_capture(block, cap, current):
    handles = []
    for name, module in (
        ("attn_norm", block.input_layernorm),
        ("ffn_norm", block.post_attention_layernorm),
        ("attn_output", block.self_attn),
        ("ffn_output", block.mlp),
    ):

        def hook(m, args, value, name=name):
            cap.save(current["name"] + "/" + name, value.clone())

        handles.append(module.register_forward_hook(hook))

    def route_hook(m, args, value):
        cap.save(current["name"] + "/route_ids", value.topk_ids)
        cap.save(current["name"] + "/route_weights_raw", value.topk_weights)

    handles.append(block.mlp.topk.register_forward_hook(route_hook))
    # Restore methods after each schedule, just like the tensor hooks.
    original_pre, original_post = block.hc_pre, block.hc_post

    def pre(*args, **kwargs):
        result = original_pre(*args, **kwargs)
        kind = "attn" if current["pre_count"] == 0 else "ffn"
        current["pre_count"] += 1
        value, post, comb, norm_fused = result
        cap.save(current["name"] + f"/{kind}_pre_or_norm", value)
        cap.save(current["name"] + f"/{kind}_post_gate", post)
        cap.save(current["name"] + f"/{kind}_comb", comb)
        # Native TileLang pre can fuse RMSNorm; capture the actual module input too.
        return result

    def post(*args, **kwargs):
        value = original_post(*args, **kwargs)
        kind = "attn" if current["post_count"] == 0 else "ffn"
        current["post_count"] += 1
        cap.save(current["name"] + f"/{kind}_post", value)
        return value

    block.hc_pre, block.hc_post = pre, post

    class Restore:
        def remove(self):
            block.hc_pre, block.hc_post = original_pre, original_post

    handles.append(Restore())
    for name, module in (("attn_input", block.self_attn), ("ffn_input", block.mlp)):

        def before(m, args, kwargs, name=name):
            value = args[0] if args else kwargs["x"]
            cap.save(current["name"] + "/" + name, value.clone())

        handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
    return handles
