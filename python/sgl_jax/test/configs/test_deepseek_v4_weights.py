"""M1.1b -- checkpoint key mapping and disposition.

The point of these tests is **totality**: expanded against the real Flash 0731
checkpoint's tensor inventory, every one of its 72317 keys must be classified, and
the classification must agree with the layer types. A key that falls through is a
weight the loader silently leaves at its initial value.

The inventory is committed as a family summary
(`deepseek_v4_flash_0731_weight_families.json`, generated from the real
`model.safetensors.index.json` at the pinned revision) rather than the 5.6 MB index.
"""

import json
import pathlib
import re

import pytest

from sgl_jax.srt.configs.deepseek_v4 import DeepseekV4Config, DeepseekV4LayerType, classify_layers
from sgl_jax.srt.models.deepseek_v4 import (
    Disposition,
    build_weight_mappings,
    classify_checkpoint,
    classify_key,
    expected_trunk_keys,
)

_FAMILIES = pathlib.Path(__file__).with_name("deepseek_v4_flash_0731_weight_families.json")


@pytest.fixture(scope="module")
def inventory():
    if not _FAMILIES.exists():  # pragma: no cover
        pytest.skip(f"weight family inventory missing at {_FAMILIES}")
    return json.loads(_FAMILIES.read_text())


@pytest.fixture(scope="module")
def cfg():
    return DeepseekV4Config()


def _expand(inventory, cfg):
    """Turn the family summary back into the concrete key set.

    The summary records, per family, the exact layer (or draft-block) indices it
    appears on -- not just a count -- so a family that exists on 41 of 43 layers
    expands onto those 41 and no others. Recording only counts would force the
    expansion to consult the code under test, which would make this circular.
    """
    keys = set()
    for family, record in inventory["families"].items():
        if "experts.E." in family:
            variants = [
                family.replace("experts.E.", f"experts.{e}.") for e in range(cfg.n_routed_experts)
            ]
        else:
            variants = [family]
        indices = record.get("layers")
        if indices is not None:
            keys |= {v.replace("layers.N.", f"layers.{i}.") for i in indices for v in variants}
            continue
        blocks = record.get("blocks")
        if blocks is not None:
            keys |= {v.replace("mtp.M.", f"mtp.{b}.") for b in blocks for v in variants}
            continue
        keys |= set(variants)
    return keys


# --------------------------------------------------------------------------
# the inventory itself
# --------------------------------------------------------------------------


def test_inventory_matches_the_pinned_revision(inventory):
    assert inventory["revision"] == "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    assert inventory["total_tensors"] == 72317
    assert inventory["shards"] == 48
    assert sum(r["count"] for r in inventory["families"].values()) == 72317


def test_layer_family_counts_encode_the_classification(inventory, cfg):
    """The checkpoint's own counts agree with `classify_layers`, independently."""
    f = {k: v["count"] for k, v in inventory["families"].items()}
    types = classify_layers(cfg)
    swa_only = sum(t is DeepseekV4LayerType.SWA_ONLY for t in types)
    csa = sum(t is DeepseekV4LayerType.C4A for t in types)
    assert f["layers.N.attn.compressor.ape"] == cfg.num_hidden_layers - swa_only == 41
    assert f["layers.N.attn.indexer.weights_proj.weight"] == csa == 21
    assert f["layers.N.ffn.gate.tid2eid"] == cfg.num_hash_layers == 3
    assert f["layers.N.ffn.gate.bias"] == cfg.num_hidden_layers - cfg.num_hash_layers == 40
    # Every trunk layer is MoE: no dense prefix layers.
    assert f["layers.N.ffn.experts.E.w1.weight"] == cfg.num_hidden_layers * cfg.n_routed_experts


def test_the_mtp_blocks_are_the_bulk_of_what_gets_dropped(inventory):
    mtp = {k: v["count"] for k, v in inventory["families"].items() if k.startswith("mtp.")}
    assert sum(mtp.values()) == 4705
    # mtp.2 carries the DSpark heads, which is why there are three blocks not one.
    assert any("markov_head" in k for k in mtp)
    assert any("confidence_head" in k for k in mtp)


# --------------------------------------------------------------------------
# totality -- the reason this module exists
# --------------------------------------------------------------------------


def test_every_checkpoint_key_is_classified(inventory, cfg):
    keys = _expand(inventory, cfg)
    assert len(keys) == inventory["total_tensors"]
    facts = classify_checkpoint(cfg, keys)  # raises on anything unrecognised
    assert len(facts) == len(keys)
    counts = {d: 0 for d in Disposition}
    for f in facts.values():
        counts[f.disposition] += 1
    experts = cfg.num_hidden_layers * cfg.n_routed_experts * 3 * 2  # w1/w2/w3, weight+scale
    assert counts[Disposition.EXPERT_CONVERTED] == experts == 66048
    assert counts[Disposition.DROPPED] == 4705
    assert counts[Disposition.CLAIMED] == 72317 - experts - 4705
    assert sum(counts.values()) == 72317


def test_the_mapping_claims_exactly_the_claimed_keys(inventory, cfg):
    """Coverage in both directions: no claimed key without a mapping, and no mapping
    for a key the checkpoint does not contain."""
    keys = _expand(inventory, cfg)
    facts = classify_checkpoint(cfg, keys)
    claimed = {k for k, f in facts.items() if f.disposition is Disposition.CLAIMED}
    mapped = set(build_weight_mappings(cfg))
    assert mapped == claimed, {
        "mapped_but_absent_from_checkpoint": sorted(mapped - claimed)[:8],
        "claimed_but_unmapped": sorted(claimed - mapped)[:8],
    }


def test_expected_trunk_keys_matches_the_real_inventory(inventory, cfg):
    """Derive the key universe from the config alone and require it to equal the
    checkpoint's trunk. This is the check that would catch a wrong layer-family gate."""
    keys = _expand(inventory, cfg)
    trunk = {k for k in keys if not k.startswith("mtp.")}
    assert expected_trunk_keys(cfg) == trunk


def test_no_mapping_targets_the_same_parameter_twice(cfg):
    """Two source keys landing on one target silently makes load order significant."""
    targets = [m.target_path for m in build_weight_mappings(cfg).values()]
    duplicates = {t for t in targets if targets.count(t) > 1}
    assert not duplicates


# --------------------------------------------------------------------------
# the layer-type gates, in both directions
# --------------------------------------------------------------------------


def test_compressor_only_exists_where_there_is_compressed_history(cfg):
    types = classify_layers(cfg)
    swa = next(i for i, t in enumerate(types) if t is DeepseekV4LayerType.SWA_ONLY)
    hca = next(i for i, t in enumerate(types) if t is DeepseekV4LayerType.C128A)
    assert classify_key(cfg, f"layers.{hca}.attn.compressor.ape").disposition is Disposition.CLAIMED
    with pytest.raises(ValueError, match="SWA-only layer, which has no compressor"):
        classify_key(cfg, f"layers.{swa}.attn.compressor.ape")


def test_indexer_only_exists_on_csa_layers(cfg):
    types = classify_layers(cfg)
    csa = next(i for i, t in enumerate(types) if t is DeepseekV4LayerType.C4A)
    hca = next(i for i, t in enumerate(types) if t is DeepseekV4LayerType.C128A)
    assert classify_key(cfg, f"layers.{csa}.attn.indexer.wq_b.scale").disposition is (
        Disposition.CLAIMED
    )
    with pytest.raises(ValueError, match="not CSA"):
        classify_key(cfg, f"layers.{hca}.attn.indexer.compressor.ape")


def test_gate_bias_and_tid2eid_are_mutually_exclusive(cfg):
    assert classify_key(cfg, "layers.0.ffn.gate.tid2eid").disposition is Disposition.CLAIMED
    assert classify_key(cfg, "layers.10.ffn.gate.bias").disposition is Disposition.CLAIMED
    with pytest.raises(ValueError, match="mutually exclusive"):
        classify_key(cfg, "layers.0.ffn.gate.bias")
    with pytest.raises(ValueError, match="does not route by token id"):
        classify_key(cfg, "layers.10.ffn.gate.tid2eid")


def test_a_layer_beyond_the_trunk_is_rejected(cfg):
    """The 46-vs-43 trap: `compress_ratios` is longer than the trunk."""
    with pytest.raises(ValueError, match="outside the 43-layer trunk"):
        classify_key(cfg, "layers.43.attn_norm.weight")


def test_an_unrecognised_key_raises_rather_than_being_skipped(cfg):
    with pytest.raises(ValueError, match="unrecognised"):
        classify_key(cfg, "layers.0.attn.something_new.weight")
    with pytest.raises(ValueError, match="unrecognised"):
        classify_key(cfg, "totally_unexpected")


def test_an_expert_beyond_the_configured_count_is_rejected(cfg):
    with pytest.raises(ValueError, match="beyond n_routed_experts"):
        classify_key(cfg, f"layers.0.ffn.experts.{cfg.n_routed_experts}.w1.weight")


# --------------------------------------------------------------------------
# quantisation families
# --------------------------------------------------------------------------


def test_routed_experts_are_pending_and_shared_experts_are_not(cfg):
    """The mixed-format split: routed experts are FP4 and wait on M0.2's layout;
    shared experts are ordinary FP8 block-scale linears."""
    routed = classify_key(cfg, "layers.0.ffn.experts.0.w1.scale")
    assert routed.disposition is Disposition.EXPERT_CONVERTED
    assert "M0.2" in routed.reason
    shared = classify_key(cfg, "layers.0.ffn.shared_experts.w1.scale")
    assert shared.disposition is Disposition.CLAIMED
    assert "block scale" in shared.reason


def test_every_fp8_linear_maps_both_weight_and_scale(cfg):
    mappings = build_weight_mappings(cfg)
    for stem in ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b"):
        w = mappings[f"layers.0.{stem}.weight"]
        s = mappings[f"layers.0.{stem}.scale"]
        assert w.target_path.endswith(".weight_q")
        assert s.target_path.endswith(".weight_scale")
        assert not w.transpose  # checkpoint is already [out, in] for the quantised path


def test_unquantised_tensors_have_no_scale_sibling(cfg, inventory):
    """Pushing these through a quantised path would look for a scale the checkpoint
    does not contain."""
    families = set(inventory["families"])
    unquantised = (
        "layers.N.attn.compressor.wkv.weight",
        "layers.N.attn.compressor.wgate.weight",
        "layers.N.attn.compressor.norm.weight",
        "layers.N.attn.compressor.ape",
        "layers.N.attn.indexer.compressor.wkv.weight",
        "layers.N.attn.indexer.weights_proj.weight",
        "layers.N.attn.attn_sink",
        "layers.N.ffn.gate.weight",
    )
    for family in unquantised:
        assert family in families
        stem = family[: -len(".weight")] if family.endswith(".weight") else family
        assert f"{stem}.scale" not in families, family
    mappings = build_weight_mappings(cfg)
    for family in unquantised:
        stem = family[: -len(".weight")] if family.endswith(".weight") else family
        assert f"{stem.replace('layers.N.', 'layers.0.')}.scale" not in mappings


def test_quantised_tensors_do_have_a_scale_sibling(inventory):
    """The other half of the same check, so the list above cannot just be wrong."""
    families = set(inventory["families"])
    for stem in (
        "layers.N.attn.wq_a",
        "layers.N.attn.wq_b",
        "layers.N.attn.wkv",
        "layers.N.attn.wo_a",
        "layers.N.attn.wo_b",
        "layers.N.attn.indexer.wq_b",
        "layers.N.ffn.shared_experts.w1",
        "layers.N.ffn.experts.E.w1",
    ):
        assert f"{stem}.weight" in families and f"{stem}.scale" in families, stem


def test_mhc_gates_are_replicated_and_untransposed(cfg):
    """They are float32 Sinkhorn coefficients, not [out, in] projections."""
    mappings = build_weight_mappings(cfg)
    for key in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        assert not mappings[key].transpose
        assert mappings[key].sharding == (None,)
    for key in ("hc_attn_fn", "hc_ffn_base", "hc_attn_scale"):
        m = mappings[f"layers.0.{key}"]
        assert not m.transpose
        assert m.sharding == (None,)


def test_wo_b_is_row_parallel_and_wo_a_is_column_parallel(cfg):
    """wo_a widens per group to G*R; wo_b reduces G*R back to hidden."""
    mappings = build_weight_mappings(cfg)
    a = mappings["layers.0.attn.wo_a.weight"].sharding
    b = mappings["layers.0.attn.wo_b.weight"].sharding
    assert a != b
    assert "tensor" in a and "tensor" in b


# --------------------------------------------------------------------------
# the dropped set
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "mtp.0.attn.wkv.weight",
        "mtp.1.ffn.experts.5.w2.scale",
        "mtp.2.markov_head.markov_w1.weight",
        "mtp.2.confidence_head.proj.weight",
        "mtp.2.hc_head_fn",
    ],
)
def test_mtp_keys_are_dropped_with_a_reason(cfg, key):
    facts = classify_key(cfg, key)
    assert facts.disposition is Disposition.DROPPED
    assert "MTP" in facts.reason
    assert key not in build_weight_mappings(cfg)


def test_dropping_is_explicit_not_loader_tolerance(cfg, inventory):
    """ "Unclaimed" and "deliberately skipped" have to be distinguishable, so every
    mtp key is classified rather than merely absent from the mapping."""
    keys = _expand(inventory, cfg)
    mtp = {k for k in keys if k.startswith("mtp.")}
    facts = classify_checkpoint(cfg, mtp)
    assert {f.disposition for f in facts.values()} == {Disposition.DROPPED}
    assert re.match(r"^mtp\.\d+\.", next(iter(mtp)))
