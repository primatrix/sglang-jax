from types import SimpleNamespace

import jax
import numpy as np

from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.speculative import dspark_util


def _config():
    return SimpleNamespace(gamma=7, draft_width=7, verify_width=8)


def _args(**overrides):
    kwargs = dict(
        model_path="target",
        speculative_algorithm="DSPARK",
        speculative_draft_model_path="draft",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        disable_overlap_schedule=True,
        grammar_backend="none",
    )
    kwargs.update(overrides)
    return ServerArgs(**kwargs)


def test_dspark_server_args_infers_checkpoint_gamma_plus_anchor(monkeypatch):
    monkeypatch.setattr(dspark_util, "parse_dspark_draft_config", lambda *a, **k: _config())
    args = _args()
    args.check_server_args()
    assert args.speculative_num_draft_tokens == 8


def test_dspark_explicit_gamma_is_normalized_to_verify_width(monkeypatch):
    monkeypatch.setattr(dspark_util, "parse_dspark_draft_config", lambda *a, **k: _config())
    args = _args(speculative_num_draft_tokens=7)
    args._explicit_cli_args = {"--speculative-num-draft-tokens"}
    args.check_server_args()
    assert args.speculative_num_draft_tokens == 8
    args.check_server_args()
    assert args.speculative_num_draft_tokens == 8


def test_dspark_rejects_explicit_gamma_mismatch(monkeypatch):
    monkeypatch.setattr(dspark_util, "parse_dspark_draft_config", lambda *a, **k: _config())
    args = _args(speculative_num_draft_tokens=6)
    args._explicit_cli_args = {"--speculative-num-draft-tokens"}
    try:
        args.check_server_args()
    except ValueError as exc:
        assert "must match checkpoint block_size=7" in str(exc)
    else:
        raise AssertionError("DSPARK must reject a gamma/checkpoint mismatch")


def test_dspark_tuned_config_cli_is_opt_in():
    args = ServerArgs.from_cli(
        [
            "--model-path",
            "target",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            "draft",
            "--enable-dspark-tuned-config",
        ]
    )
    assert args.enable_dspark_tuned_config is True


def test_dspark_tuned_config_rejects_other_algorithms():
    args = ServerArgs(model_path="target", enable_dspark_tuned_config=True)
    try:
        args.check_server_args()
    except ValueError as exc:
        assert "requires --speculative-algorithm DSPARK" in str(exc)
    else:
        raise AssertionError("DSpark tuned config must not apply to other algorithms")


def test_dspark_worker_uses_separate_draft_and_verify_metadata_widths():
    from sgl_jax.srt.speculative.dspark_worker import DSparkWorker

    worker = object.__new__(DSparkWorker)
    worker.__dict__.update(
        block_size=8,
        draft_width=7,
        verify_width=8,
        mesh=jax.sharding.Mesh(
            np.asarray(jax.devices()).reshape(1, 1),
            ("data", "tensor"),
        ),
        _verify_bucket_templates={},
    )
    mwb = SimpleNamespace(
        dp_size=1,
        per_dp_bs_size=2,
        real_bs=2,
        logits_indices_selector=np.asarray([0, 1], dtype=np.int32),
    )
    draft = worker._get_verify_bucket_template(mwb, bs=2, width=worker.draft_width)
    verify = worker._get_verify_bucket_template(mwb, bs=2, width=worker.verify_width)
    assert draft.extend_seq_lens.tolist() == [7, 7]
    assert verify.extend_seq_lens.tolist() == [8, 8]
    assert np.asarray(draft.cu_q_lens).tolist() == [0, 7, 14]
    assert np.asarray(verify.cu_q_lens).tolist() == [0, 8, 16]
