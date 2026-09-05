import asyncio
import json
import logging
import time
from types import SimpleNamespace

from sgl_jax.srt.request_time_stats import (
    REQUEST_TIME_STATS_RID_STATE_KEY,
    REQUEST_TIME_STATS_SCHEMA_VERSION,
    REQUEST_TIME_STATS_STATE_KEY,
    RequestTimeStatsMiddleware,
    mark_batch_time_stats,
    mark_request_time_stats,
    should_sample_request,
)


def test_should_sample_request_is_stable_and_bounded():
    assert not should_sample_request("request-a", 0.0)
    assert should_sample_request("request-a", 1.0)
    assert should_sample_request("request-a", 0.125) == should_sample_request(
        "request-a", 0.125
    )

    sampled = sum(
        should_sample_request(f"request-{index}", 0.125) for index in range(1024)
    )
    assert 80 <= sampled <= 180


def test_mark_batch_time_stats_only_marks_sampled_entries():
    sampled = {"existing_ns": 10}
    batch = SimpleNamespace(request_time_stats=[sampled, None])

    mark_batch_time_stats(batch, "stage_ns", 20)
    mark_batch_time_stats(batch, "stage_ns", 30)

    assert sampled == {"existing_ns": 10, "stage_ns": 20}


def test_mark_request_time_stats_is_lazy_and_preserves_first_mark(monkeypatch):
    calls = 0

    def fake_time_ns():
        nonlocal calls
        calls += 1
        return 20

    monkeypatch.setattr("sgl_jax.srt.request_time_stats.time.time_ns", fake_time_ns)
    mark_request_time_stats([], "stage_ns")
    mark_request_time_stats([None], "stage_ns")
    assert calls == 0

    first = {"stage_ns": 10}
    second = {}
    mark_request_time_stats([first, None, second], "stage_ns")
    assert calls == 1
    assert first["stage_ns"] == 10
    assert second["stage_ns"] == 20


def test_middleware_logs_once_after_first_content_send(caplog):
    sent_messages = []

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        sent_messages.append(message)

    async def inner_app(scope, receive_call, send_call):
        await receive_call()
        state = scope["state"]
        stats = state[REQUEST_TIME_STATS_STATE_KEY]
        state[REQUEST_TIME_STATS_RID_STATE_KEY] = "bench-request-a"
        await send_call({"type": "http.response.start", "status": 200, "headers": []})
        await send_call(
            {"type": "http.response.body", "body": b"role", "more_body": True}
        )
        assert not any(message.startswith("REQUEST-TIME-TRACE ") for message in caplog.messages)
        stats["openai_first_content_ready_ns"] = time.time_ns()
        await send_call(
            {"type": "http.response.body", "body": b"content", "more_body": True}
        )
        await send_call({"type": "http.response.body", "body": b"done"})

    scope = {
        "type": "http",
        "path": "/v1/chat/completions",
        "app": SimpleNamespace(
            server_args=SimpleNamespace(enable_request_time_stats_logging=True)
        ),
    }
    caplog.set_level(logging.INFO, logger="sgl_jax.srt.request_time_stats")

    asyncio.run(RequestTimeStatsMiddleware(inner_app)(scope, receive, send))

    traces = [
        record.message.removeprefix("REQUEST-TIME-TRACE ")
        for record in caplog.records
        if record.message.startswith("REQUEST-TIME-TRACE ")
    ]
    assert len(traces) == 1
    trace = json.loads(traces[0])
    assert trace["schema_version"] == REQUEST_TIME_STATS_SCHEMA_VERSION
    timestamps = trace["timestamps_ns"]
    assert trace["request_id"] == "bench-request-a"
    assert timestamps["server_asgi_enter_ns"] <= timestamps["server_body_receive_start_ns"]
    assert timestamps["server_body_receive_start_ns"] <= timestamps["server_body_receive_done_ns"]
    assert timestamps["server_first_body_send_done_ns"] <= timestamps[
        "server_first_content_send_start_ns"
    ]
    assert timestamps["server_first_content_send_start_ns"] <= timestamps[
        "server_first_content_send_done_ns"
    ]
    assert len(sent_messages) == 4


def test_middleware_is_passthrough_when_disabled():
    called = False

    async def inner_app(scope, receive, send):
        nonlocal called
        called = True

    async def receive():
        raise AssertionError("receive should not be called")

    async def send(message):
        raise AssertionError("send should not be called")

    scope = {
        "type": "http",
        "path": "/v1/chat/completions",
        "app": SimpleNamespace(
            server_args=SimpleNamespace(enable_request_time_stats_logging=False)
        ),
    }

    asyncio.run(RequestTimeStatsMiddleware(inner_app)(scope, receive, send))
    assert called
    assert "state" not in scope
