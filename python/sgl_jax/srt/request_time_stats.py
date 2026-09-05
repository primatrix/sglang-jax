"""Low-overhead, request-correlated timing helpers for serving requests.

All cross-process boundaries use ``time.time_ns()`` so timestamps from the
HTTP, scheduler, encoder, and detokenizer processes share one clock domain on
the serving host. Monotonic clocks remain appropriate only for local durations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)

REQUEST_TIME_STATS_STATE_KEY = "request_time_stats"
REQUEST_TIME_STATS_RID_STATE_KEY = "request_time_stats_rid"
REQUEST_TIME_STATS_SCHEMA_VERSION = 2


def should_sample_request(request_id: str, sample_rate: float) -> bool:
    """Return a stable sampling decision for a request ID.

    A stable hash lets the benchmark and server independently select the same
    requests without another wire-level control field.
    """

    if sample_rate <= 0:
        return False
    if sample_rate >= 1:
        return True
    digest = hashlib.blake2b(
        request_id.encode("utf-8"), digest_size=8, person=b"sgljax-ts"
    ).digest()
    bucket = int.from_bytes(digest, byteorder="big", signed=False)
    return bucket < int(sample_rate * (1 << 64))


def mark_request_time_stats(
    stats_iterable: Iterable[dict[str, int] | None] | None,
    field: str,
    timestamp_ns: int | None = None,
) -> None:
    """Set one shared timestamp on each sampled request in an iterable.

    The clock is read lazily, so an empty list or a batch containing only
    unsampled requests adds no ``time.time_ns()`` call to the hot path.
    """

    if stats_iterable is None:
        return
    value = timestamp_ns
    for stats in stats_iterable:
        if stats is None:
            continue
        if value is None:
            value = time.time_ns()
        stats.setdefault(field, value)


def mark_batch_time_stats(batch: Any, field: str, timestamp_ns: int | None = None) -> None:
    """Set a timestamp on every sampled request carried by a batch output."""

    mark_request_time_stats(
        getattr(batch, "request_time_stats", None), field, timestamp_ns
    )


class RequestTimeStatsMiddleware:
    """Capture HTTP boundaries without buffering or decoding the request body.

    The middleware is a pure ASGI wrapper. When timing is disabled it only does
    a path/flag check and forwards the original call. For sampled requests, the
    OpenAI handler keeps the state dictionary and the streaming generator adds
    ``openai_first_content_ready_ns`` immediately before yielding real content.
    The final trace is logged only after ASGI has accepted that content body.
    """

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        fastapi_app = scope.get("app")
        server_args = getattr(fastapi_app, "server_args", None)
        enabled = bool(
            getattr(server_args, "enable_request_time_stats_logging", False)
        )
        if (
            scope.get("type") != "http"
            or scope.get("path") != "/v1/chat/completions"
            or not enabled
        ):
            await self.app(scope, receive, send)
            return

        state = scope.setdefault("state", {})
        stats: dict[str, int] = {"server_asgi_enter_ns": time.time_ns()}
        state[REQUEST_TIME_STATS_STATE_KEY] = stats
        body_started = False
        body_done = False
        first_body_sent = False
        content_trace_logged = False

        async def receive_with_timing() -> dict[str, Any]:
            nonlocal body_started, body_done
            if not body_started:
                body_started = True
                stats["server_body_receive_start_ns"] = time.time_ns()
            message = await receive()
            if (
                not body_done
                and message.get("type") == "http.request"
                and not message.get("more_body", False)
            ):
                body_done = True
                stats["server_body_receive_done_ns"] = time.time_ns()
            return message

        async def send_with_timing(message: dict[str, Any]) -> None:
            nonlocal first_body_sent, content_trace_logged
            current_stats = state.get(REQUEST_TIME_STATS_STATE_KEY)
            is_response_start = message.get("type") == "http.response.start"
            if is_response_start and current_stats is not None:
                current_stats.setdefault("server_response_start_ns", time.time_ns())

            is_body = message.get("type") == "http.response.body"
            if is_body and current_stats is not None and not first_body_sent:
                first_body_sent = True
                current_stats.setdefault("server_first_body_send_start_ns", time.time_ns())

            is_first_content = bool(
                is_body
                and current_stats is not None
                and not content_trace_logged
                and "openai_first_content_ready_ns" in current_stats
            )
            if is_first_content:
                current_stats.setdefault("server_first_content_send_start_ns", time.time_ns())

            await send(message)

            if is_response_start and current_stats is not None:
                current_stats.setdefault("server_response_start_done_ns", time.time_ns())
            if is_body and current_stats is not None and first_body_sent:
                current_stats.setdefault("server_first_body_send_done_ns", time.time_ns())
            if is_first_content:
                content_trace_logged = True
                current_stats["server_first_content_send_done_ns"] = time.time_ns()
                request_id = state.get(REQUEST_TIME_STATS_RID_STATE_KEY)
                payload = {
                    "schema_version": REQUEST_TIME_STATS_SCHEMA_VERSION,
                    "request_id": request_id,
                    "timestamps_ns": current_stats,
                }
                logger.info(
                    "REQUEST-TIME-TRACE %s",
                    json.dumps(payload, separators=(",", ":"), sort_keys=True),
                )

        await self.app(scope, receive_with_timing, send_with_timing)
