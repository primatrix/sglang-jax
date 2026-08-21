from types import SimpleNamespace

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


def _legacy_config(block_size):
    return SimpleNamespace(
        block_size=block_size,
        draft_width=block_size,
        verify_width=block_size,
        dialect="legacy",
    )


def test_dflash_server_args_infers_default_block_size(monkeypatch):
    calls = []

    def fake_parse(model_path, revision=None, trust_remote_code=True):
        calls.append((model_path, revision, trust_remote_code))
        return SimpleNamespace(
            block_size=16,
            draft_width=16,
            verify_width=16,
            dialect="legacy",
        )

    monkeypatch.setattr(dflash_util, "parse_dflash_draft_config", fake_parse)

    args = _dflash_args(speculative_draft_model_revision="abc", trust_remote_code=True)
    args.check_server_args()

    assert calls == [("draft", "abc", True)]
    assert args.speculative_num_draft_tokens == 16


def test_dflash_server_args_normalizes_deepspec_block_to_verify_width(monkeypatch):
    monkeypatch.setattr(
        dflash_util,
        "parse_dflash_draft_config",
        lambda *a, **k: SimpleNamespace(
            block_size=7,
            draft_width=7,
            verify_width=8,
            dialect="deepspec",
        ),
    )

    args = _dflash_args()
    args.check_server_args()

    assert args.speculative_num_draft_tokens == 8
    args.check_server_args()
    assert args.speculative_num_draft_tokens == 8


def test_dflash_server_args_accepts_explicit_deepspec_proposal_count(monkeypatch):
    monkeypatch.setattr(
        dflash_util,
        "parse_dflash_draft_config",
        lambda *a, **k: SimpleNamespace(
            block_size=7,
            draft_width=7,
            verify_width=8,
            dialect="deepspec",
        ),
    )
    args = _dflash_args(speculative_num_draft_tokens=7)
    args._explicit_cli_args = {"--speculative-num-draft-tokens"}

    args.check_server_args()

    assert args.speculative_num_draft_tokens == 8


def test_dflash_server_args_preserves_nondefault_block_size(monkeypatch):
    monkeypatch.setattr(dflash_util, "parse_dflash_draft_config", lambda *a, **k: _legacy_config(8))

    args = _dflash_args(speculative_num_draft_tokens=8)
    args.check_server_args()

    assert args.speculative_num_draft_tokens == 8


def test_dflash_server_args_preserves_explicit_default_block_size(monkeypatch):
    monkeypatch.setattr(dflash_util, "parse_dflash_draft_config", lambda *a, **k: _legacy_config(4))

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
    monkeypatch.setattr(
        dflash_util, "parse_dflash_draft_config", lambda *a, **k: _legacy_config(16)
    )

    args = _dflash_args(speculative_num_draft_tokens=16, tp_size=4, dp_size=1)
    args.check_server_args()

    assert args.tp_size == 4


def test_dflash_server_args_allows_data_parallel_attention(monkeypatch):
    monkeypatch.setattr(
        dflash_util, "parse_dflash_draft_config", lambda *a, **k: _legacy_config(16)
    )

    args = _dflash_args(speculative_num_draft_tokens=16, tp_size=4, dp_size=2)
    args.check_server_args()

    assert args.dp_size == 2
    assert args.tp_size // args.dp_size == 2
