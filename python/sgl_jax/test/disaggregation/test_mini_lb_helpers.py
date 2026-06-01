from __future__ import annotations

from sgl_jax.srt.disaggregation import mini_lb_helpers as H


def test_maybe_wrap_ipv6_address_wraps_ipv6_only():
    assert H.maybe_wrap_ipv6_address("fe80::1") == "[fe80::1]"
    assert H.maybe_wrap_ipv6_address("127.0.0.1") == "127.0.0.1"
    assert H.maybe_wrap_ipv6_address("example.com") == "example.com"


def test_get_request_batch_size_handles_scalar_and_batch_shapes():
    assert H.get_request_batch_size({"text": "hi"}) is None
    assert H.get_request_batch_size({"text": ["a", "b"]}) == 2
    assert H.get_request_batch_size({"input_ids": [1, 2, 3]}) is None
    assert H.get_request_batch_size({"input_ids": [[1], [2], [3]]}) == 3


def test_inject_bootstrap_fields_scalar_request():
    payload = {"text": "hi"}
    out = H.inject_bootstrap_fields(
        payload,
        prefill_server="http://127.0.0.1:30100",
        bootstrap_port=8998,
    )
    assert out["bootstrap_host"] == "127.0.0.1"
    assert out["bootstrap_port"] == 8998
    assert isinstance(out["bootstrap_room"], int)


def test_inject_bootstrap_fields_batch_request():
    payload = {"text": ["a", "b", "c"]}
    out = H.inject_bootstrap_fields(
        payload,
        prefill_server="http://[fe80::1]:30100",
        bootstrap_port=8998,
    )
    assert out["bootstrap_host"] == ["[fe80::1]"] * 3
    assert out["bootstrap_port"] == [8998] * 3
    assert len(out["bootstrap_room"]) == 3
    assert out["bootstrap_room"][1] == out["bootstrap_room"][0] + 1


def test_inject_bootstrap_fields_with_host_override():
    payload = {"text": "hi"}
    out = H.inject_bootstrap_fields(
        payload,
        prefill_server="http://127.0.0.1:30100",
        bootstrap_port=8998,
        bootstrap_host_override="10.31.173.56",
    )
    assert out["bootstrap_host"] == "10.31.173.56"
