from types import SimpleNamespace

import pytest

from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.speculative import dflash_util


def _dflash_args(**overrides):
    kwargs = dict(
        model_path="target",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path="draft",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        disable_overlap_schedule=True,
        grammar_backend="none",
    )
    kwargs.update(overrides)
    return ServerArgs(**kwargs)


def test_dflash_server_args_infers_default_block_size(monkeypatch):
    calls = []

    def fake_parse(model_path, revision=None, trust_remote_code=True):
        calls.append((model_path, revision, trust_remote_code))
        return SimpleNamespace(block_size=16)

    monkeypatch.setattr(dflash_util, "parse_dflash_draft_config", fake_parse)

    args = _dflash_args(speculative_draft_model_revision="abc", trust_remote_code=True)
    args.check_server_args()

    assert calls == [("draft", "abc", True)]
    assert args.speculative_num_draft_tokens == 16


def test_dflash_server_args_preserves_nondefault_block_size(monkeypatch):
    def fail_parse(*args, **kwargs):
        raise AssertionError("non-default DFlash draft token count should not be inferred")

    monkeypatch.setattr(dflash_util, "parse_dflash_draft_config", fail_parse)

    args = _dflash_args(speculative_num_draft_tokens=8)
    args.check_server_args()

    assert args.speculative_num_draft_tokens == 8


def test_dflash_server_args_preserves_explicit_default_block_size(monkeypatch):
    def fail_parse(*args, **kwargs):
        raise AssertionError("explicit DFlash draft token count should not be inferred")

    monkeypatch.setattr(dflash_util, "parse_dflash_draft_config", fail_parse)

    args = ServerArgs.from_cli(
        [
            "--model-path",
            "target",
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            "draft",
            "--speculative-num-draft-tokens",
            "4",
            "--speculative-num-steps",
            "1",
            "--speculative-eagle-topk",
            "1",
            "--disable-overlap-schedule",
            "--grammar-backend",
            "none",
        ]
    )
    args.check_server_args()

    assert args.speculative_num_draft_tokens == 4


def test_dflash_server_args_allows_tensor_parallel(monkeypatch):
    def fail_parse(*args, **kwargs):
        raise AssertionError("non-default DFlash draft token count should not be inferred")

    monkeypatch.setattr(dflash_util, "parse_dflash_draft_config", fail_parse)

    args = _dflash_args(speculative_num_draft_tokens=16, tp_size=4, dp_size=1)
    args.check_server_args()

    assert args.tp_size == 4


def test_dflash_server_args_allows_data_parallel_attention(monkeypatch):
    def fail_parse(*args, **kwargs):
        raise AssertionError("non-default DFlash draft token count should not be inferred")

    monkeypatch.setattr(dflash_util, "parse_dflash_draft_config", fail_parse)

    args = _dflash_args(speculative_num_draft_tokens=16, tp_size=4, dp_size=2)
    args.check_server_args()

    assert args.dp_size == 2
    assert args.tp_size // args.dp_size == 2


def test_dflash_server_args_parses_flashback_controls(monkeypatch):
    monkeypatch.setattr(
        dflash_util,
        "parse_dflash_draft_config",
        lambda *args, **kwargs: SimpleNamespace(block_size=8),
    )
    args = ServerArgs.from_cli(
        [
            "--model-path",
            "target",
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            "draft",
            "--speculative-num-steps",
            "1",
            "--speculative-eagle-topk",
            "1",
            "--enable-dflash-flashback",
            "--dflash-flashback-bonus",
            "1.25",
            "--dflash-flashback-target-margin-weight",
            "0.75",
            "--dflash-flashback-position-decay",
            "0.6",
            "--grammar-backend",
            "none",
        ]
    )
    args.check_server_args()

    assert args.enable_dflash_flashback
    assert args.dflash_flashback_bonus == 1.25
    assert args.dflash_flashback_target_margin_weight == 0.75
    assert args.dflash_flashback_position_decay == 0.6


def test_dflash_server_args_parses_anchor_layout(monkeypatch):
    monkeypatch.setattr(
        dflash_util,
        "parse_dflash_draft_config",
        lambda *args, **kwargs: SimpleNamespace(block_size=7),
    )
    args = ServerArgs.from_cli(
        [
            "--model-path",
            "target",
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            "draft",
            "--speculative-num-steps",
            "1",
            "--speculative-eagle-topk",
            "1",
            "--enable-dflash-anchor",
            "--grammar-backend",
            "none",
        ]
    )
    args.check_server_args()

    assert args.enable_dflash_anchor
    assert args.speculative_num_draft_tokens == 7


def test_dflash_server_args_parses_redenoise_controls(monkeypatch):
    monkeypatch.setattr(
        dflash_util,
        "parse_dflash_draft_config",
        lambda *args, **kwargs: SimpleNamespace(block_size=7),
    )
    args = ServerArgs.from_cli(
        [
            "--model-path",
            "target",
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            "draft",
            "--speculative-num-steps",
            "1",
            "--speculative-eagle-topk",
            "1",
            "--enable-dflash-anchor",
            "--enable-dflash-redenoise",
            "--dflash-redenoise-margin-threshold",
            "1.5",
            "--dflash-redenoise-prefix-len",
            "2",
            "--grammar-backend",
            "none",
        ]
    )
    args.check_server_args()

    assert args.enable_dflash_redenoise
    assert args.dflash_redenoise_margin_threshold == 1.5
    assert args.dflash_redenoise_prefix_len == 2


def test_dflash_redenoise_requires_anchor():
    args = _dflash_args(
        speculative_num_draft_tokens=7,
        enable_dflash_redenoise=True,
    )
    with pytest.raises(ValueError, match="requires --enable-dflash-anchor"):
        args.check_server_args()


def test_dflash_server_args_parses_ngram_controls(monkeypatch):
    monkeypatch.setattr(
        dflash_util,
        "parse_dflash_draft_config",
        lambda *args, **kwargs: SimpleNamespace(block_size=7),
    )
    args = ServerArgs.from_cli(
        [
            "--model-path",
            "target",
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            "draft",
            "--speculative-num-steps",
            "1",
            "--speculative-eagle-topk",
            "1",
            "--enable-dflash-anchor",
            "--enable-dflash-ngram",
            "--disable-overlap-schedule",
            "--dflash-ngram-min-match",
            "3",
            "--dflash-ngram-max-match",
            "8",
            "--dflash-ngram-bonus",
            "1.5",
            "--dflash-ngram-position-decay",
            "0.75",
            "--dflash-ngram-max-rerank-positions",
            "1",
            "--grammar-backend",
            "none",
        ]
    )
    args.check_server_args()

    assert args.enable_dflash_ngram
    assert args.dflash_ngram_bonus == 1.5
    assert args.dflash_ngram_position_decay == 0.75
    assert args.dflash_ngram_max_rerank_positions == 1


def test_dflash_ngram_requires_non_overlap():
    args = _dflash_args(
        speculative_num_draft_tokens=7,
        enable_dflash_ngram=True,
        disable_overlap_schedule=False,
    )
    with pytest.raises(ValueError, match="requires --disable-overlap-schedule"):
        args.check_server_args()


def test_dflash_feedback_shadow_parses_and_requires_non_overlap():
    args = _dflash_args(
        speculative_num_draft_tokens=7,
        enable_dflash_feedback_shadow=True,
    )
    args.check_server_args()
    assert args.enable_dflash_feedback_shadow

    args.disable_overlap_schedule = False
    with pytest.raises(ValueError, match="requires --disable-overlap-schedule"):
        args.check_server_args()


def test_dflash_feedback_shadow_rejects_active_reranking():
    args = _dflash_args(
        speculative_num_draft_tokens=7,
        enable_dflash_feedback_shadow=True,
        enable_dflash_ngram=True,
    )
    with pytest.raises(ValueError, match="reranking to remain disabled"):
        args.check_server_args()


def test_flashback_requires_dflash():
    args = ServerArgs(model_path="target", enable_dflash_flashback=True)
    with pytest.raises(ValueError, match="requires --speculative-algorithm DFLASH"):
        args.check_server_args()


def test_anchor_requires_dflash():
    args = ServerArgs(model_path="target", enable_dflash_anchor=True)
    with pytest.raises(ValueError, match="requires --speculative-algorithm DFLASH"):
        args.check_server_args()
