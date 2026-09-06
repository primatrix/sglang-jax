import asyncio
import json
from types import SimpleNamespace

import pytest

from sgl_jax.srt.request_time_stats import (
    RequestTimeStatsMiddleware,
    mark_request_time_stats,
    merge_part_time_stats,
    should_sample_request,
)


def test_sampling_boundaries_and_lazy_first_timestamp(monkeypatch):
    assert not should_sample_request("request", 0)
    assert should_sample_request("request", 1)
    clock = iter([7])
    monkeypatch.setattr("sgl_jax.srt.request_time_stats.time.time_ns", lambda: next(clock))
    stats = {}
    mark_request_time_stats([None], "done_ns")
    mark_request_time_stats([None, stats], "done_ns")
    mark_request_time_stats([stats], "done_ns")
    assert stats == {"done_ns": 7}
    assert merge_part_time_stats(
        [
            {"encode_start_ns": 3, "encode_done_ns": 4},
            {"encode_start_ns": 1, "encode_done_ns": 9},
        ]
    ) == {"encode_start_ns": 1, "encode_done_ns": 9}


@pytest.mark.parametrize("enabled, fail_send", [(False, False), (True, False), (True, True)])
def test_trace_emitted_once_only_after_content_is_sent(monkeypatch, caplog, enabled, fail_send):
    if not enabled:
        monkeypatch.setattr("time.time_ns", lambda: pytest.fail("disabled tracing read the clock"))

    async def receive():
        return {"type": "http.request", "more_body": False}

    async def send(message):
        if message.get("body") == b"content":
            assert not caplog.records
            if fail_send:
                raise ConnectionError("disconnected")

    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start"})
        await send({"type": "http.response.body", "body": b"role"})
        if enabled:
            scope["state"]["request_time_stats"]["openai_first_content_ready_ns"] = 1
        await send({"type": "http.response.body", "body": b"content"})
        await send({"type": "http.response.body", "body": b"done"})

    scope = {
        "type": "http",
        "path": "/v1/chat/completions",
        "app": SimpleNamespace(
            server_args=SimpleNamespace(enable_request_time_stats_logging=enabled)
        ),
    }
    caplog.set_level("INFO", logger="sgl_jax.srt.request_time_stats")
    call = RequestTimeStatsMiddleware(app)(scope, receive, send)
    if fail_send:
        with pytest.raises(ConnectionError):
            asyncio.run(call)
    else:
        asyncio.run(call)
    assert len(caplog.records) == int(enabled and not fail_send)
    if caplog.records:
        trace = json.loads(caplog.messages[0].removeprefix("REQUEST-TIME-TRACE "))
        times = trace["timestamps_ns"]
        assert times["server_first_content_send_done_ns"] >= times["openai_first_content_ready_ns"]
