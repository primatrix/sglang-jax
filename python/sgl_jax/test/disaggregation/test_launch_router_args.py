from __future__ import annotations

from sgl_jax.srt.disaggregation.launch_router import parse_router_args
from sgl_jax.srt.disaggregation.router_args import RouterArgs


def test_parse_router_args_pd_mode_builds_router_args():
    args = parse_router_args(
        [
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill",
            "http://127.0.0.1:30100",
            "8998",
            "--decode",
            "http://127.0.0.1:30200",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    )
    assert isinstance(args, RouterArgs)
    assert args.pd_disaggregation is True
    assert args.mini_lb is True
    assert args.prefill_urls == [("http://127.0.0.1:30100", 8998)]
    assert args.decode_urls == ["http://127.0.0.1:30200"]
    assert args.host == "0.0.0.0"
    assert args.port == 8000


def test_parse_router_args_accepts_prefill_bootstrap_host_override():
    args = parse_router_args(
        [
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill",
            "http://127.0.0.1:30100",
            "8998",
            "--decode",
            "http://127.0.0.1:30200",
            "--prefill-bootstrap-host",
            "10.31.173.56",
        ]
    )
    assert args.prefill_bootstrap_host == "10.31.173.56"


def test_parse_router_args_supports_legacy_prefill_comma_syntax():
    args = parse_router_args(
        [
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill",
            "http://127.0.0.1:30100,8998",
            "--decode",
            "http://127.0.0.1:30200",
        ]
    )
    assert args.prefill_urls == [("http://127.0.0.1:30100", 8998)]


def test_parse_router_args_supports_none_bootstrap_port():
    args = parse_router_args(
        [
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill",
            "http://127.0.0.1:30100",
            "none",
            "--decode",
            "http://127.0.0.1:30200",
        ]
    )
    assert args.prefill_urls == [("http://127.0.0.1:30100", None)]
