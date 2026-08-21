from types import SimpleNamespace

import pytest

from sgl_jax.srt.speculative.dflash_util import DFlashDraftConfig, parse_dflash_draft_config


def _parse(monkeypatch, config):
    monkeypatch.setattr(
        "sgl_jax.srt.hf_transformers_utils.get_config",
        lambda *args, **kwargs: config,
    )
    return parse_dflash_draft_config("draft")


def test_parse_legacy_dflash_widths(monkeypatch):
    config = SimpleNamespace(
        architectures=["DFlashDraftModel"],
        block_size=16,
        target_layer_ids=[1, 9, 17, 25, 33],
        mask_token_id=42,
    )

    parsed = _parse(monkeypatch, config)

    assert parsed == DFlashDraftConfig(
        block_size=16,
        draft_width=16,
        verify_width=16,
        proposal_hidden_start=1,
        dialect="legacy",
        target_layer_ids=[1, 9, 17, 25, 33],
        mask_token="<|MASK|>",
        mask_token_id=42,
    )


def test_parse_deepspec_dflash_widths(monkeypatch):
    config = SimpleNamespace(
        architectures=["Qwen3DSparkModel"],
        block_size=7,
        markov_rank=0,
        target_layer_ids=[1, 9, 17, 25, 33],
        mask_token_id=42,
    )

    parsed = _parse(monkeypatch, config)

    assert parsed.dialect == "deepspec"
    assert parsed.draft_width == 7
    assert parsed.verify_width == 8
    assert parsed.proposal_hidden_start == 0


def test_parse_deepspec_dflash_rejects_markov_checkpoint(monkeypatch):
    config = SimpleNamespace(
        architectures=["Qwen3DSparkModel"],
        block_size=7,
        markov_rank=8,
        target_layer_ids=[1, 9, 17, 25, 33],
        mask_token_id=42,
    )

    with pytest.raises(ValueError, match="Use DSPARK"):
        _parse(monkeypatch, config)
