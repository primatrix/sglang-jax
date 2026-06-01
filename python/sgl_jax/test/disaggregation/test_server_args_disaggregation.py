"""Unit tests for the PD-related fields on ``ServerArgs``."""

from __future__ import annotations

import argparse
import dataclasses
import logging

import pytest

from sgl_jax.srt.server_args import ServerArgs


def _make_args(**overrides) -> ServerArgs:
    """Build a minimal ServerArgs for the PD validation tests.

    ``ServerArgs`` requires ``model_path`` and runs ``__post_init__``
    which expects a few fields to default sensibly; we pass the
    minimum to keep the validation focused on disaggregation fields.
    """

    defaults = dict(
        model_path="dummy/model",
        device="cpu",
        random_seed=42,
        mem_fraction_static=0.5,
    )
    defaults.update(overrides)
    return ServerArgs(**defaults)


def test_default_mode_is_null():
    args = _make_args()
    assert args.disaggregation_mode == "null"
    assert args.disaggregation_bootstrap_url is None


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="disaggregation-mode"):
        _make_args(disaggregation_mode="proxy")


def test_prefill_mode_requires_bootstrap_url():
    with pytest.raises(ValueError, match="bootstrap-url"):
        _make_args(disaggregation_mode="prefill")
    with pytest.raises(ValueError, match="bootstrap-url"):
        _make_args(disaggregation_mode="decode")


def test_prefill_mode_with_url_accepts():
    args = _make_args(
        disaggregation_mode="prefill",
        disaggregation_bootstrap_url="http://127.0.0.1:8998",
        page_size=128,
    )
    assert args.disaggregation_mode == "prefill"


def test_decode_mode_with_url_accepts():
    args = _make_args(
        disaggregation_mode="decode",
        disaggregation_bootstrap_url="http://127.0.0.1:8998",
        page_size=128,
    )
    assert args.disaggregation_mode == "decode"


def test_pd_mode_rejects_small_page_size():
    """Stage 4 e2e FINDING-D: PD requires page_size >= 128. With
    smaller pages, the per-request sharded KV gather OOMs the XLA
    collective planner on TPU. Catch this at boot, not at first
    request."""

    with pytest.raises(ValueError, match=r"page-size=1 is below the PD minimum"):
        _make_args(
            disaggregation_mode="prefill",
            disaggregation_bootstrap_url="http://127.0.0.1:8998",
            # page_size defaults to 1
        )
    with pytest.raises(ValueError, match=r"page-size=64 is below the PD minimum"):
        _make_args(
            disaggregation_mode="decode",
            disaggregation_bootstrap_url="http://127.0.0.1:8998",
            page_size=64,
        )


def test_null_mode_warns_on_override(caplog):
    with caplog.at_level(logging.WARNING):
        _make_args(
            disaggregation_mode="null",
            disaggregation_bootstrap_url="http://127.0.0.1:8998",
        )
    assert any(
        "ignores PD options" in rec.getMessage()
        for rec in caplog.records
    )


def test_null_mode_no_warning_at_defaults(caplog):
    with caplog.at_level(logging.WARNING):
        _make_args()
    assert not any(
        "ignores PD options" in rec.getMessage()
        for rec in caplog.records
    )


def test_default_port_values():
    args = _make_args()
    assert args.disaggregation_bootstrap_port == 8998
    assert args.disaggregation_transfer_port == 30001
    assert args.disaggregation_side_channel_port == 9600
    assert args.disaggregation_enable_d2h is False
    assert args.disaggregation_d2h_pool_size == 64
    assert args.disaggregation_d2h_max_tokens is None


def test_cli_args_parse_disaggregation_flags():
    """Stitch ``add_cli_args`` + ``from_cli_args`` and check the
    new flags survive the round trip.
    """

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    argv = [
        "--model-path", "dummy/model",
        "--device", "cpu",
        "--mem-fraction-static", "0.5",
        "--page-size", "128",
        "--disaggregation-mode", "prefill",
        "--disaggregation-bootstrap-url", "http://127.0.0.1:8998",
        "--disaggregation-transfer-port", "31001",
        "--disaggregation-side-channel-port", "31002",
        "--disaggregation-enable-d2h",
        "--disaggregation-d2h-pool-size", "128",
    ]
    ns = parser.parse_args(argv)
    # ``from_cli_args`` expects tp_size / dp_size shorthand renames.
    ns.tensor_parallel_size = getattr(ns, "tensor_parallel_size", 1) or 1
    ns.data_parallel_size = getattr(ns, "data_parallel_size", 1) or 1
    args = ServerArgs.from_cli_args(ns)
    assert args.disaggregation_mode == "prefill"
    assert args.disaggregation_bootstrap_url == "http://127.0.0.1:8998"
    assert args.disaggregation_transfer_port == 31001
    assert args.disaggregation_side_channel_port == 31002
    assert args.disaggregation_enable_d2h is True
    assert args.disaggregation_d2h_pool_size == 128
