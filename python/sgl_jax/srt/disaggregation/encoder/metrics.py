from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_PIPELINE_SPANS = {
    "queue_ms": ("enqueue_ns", "dequeue_ns"),
    "encode_stage_wait_ms": ("dequeue_ns", "preprocess_start_ns"),
    "preprocess_ms": ("preprocess_start_ns", "preprocess_done_ns"),
    "encode_wait_ms": ("preprocess_done_ns", "transfer_reserve_start_ns"),
    "transfer_reserve_ms": ("transfer_reserve_start_ns", "transfer_reserve_done_ns"),
    "encode_launch_gap_ms": ("transfer_reserve_done_ns", "encode_start_ns"),
    # encode_done_ns records asynchronous dispatch returning, not device completion.
    "encode_host_dispatch_ms": ("encode_start_ns", "encode_done_ns"),
    "encode_ms": ("dequeue_ns", "encode_done_ns"),
    "post_vit_to_copy_ms": ("encode_done_ns", "transfer_copy_start_ns"),
    "runtime_return_gap_ms": ("encode_done_ns", "runtime_encode_return_ns"),
    "runtime_to_copy_gap_ms": ("runtime_encode_return_ns", "transfer_copy_start_ns"),
    "publish_ms": ("encode_done_ns", "publish_done_ns"),
    "transfer_handoff_ms": ("encode_done_ns", "transfer_enqueue_ns"),
    "transfer_queue_ms": ("transfer_enqueue_ns", "transfer_start_ns"),
    "transfer_pool_setup_ms": ("transfer_copy_start_ns", "transfer_pool_ready_ns"),
    "transfer_copy_submit_ms": ("transfer_pool_ready_ns", "transfer_copy_submit_ns"),
    "transfer_copy_wait_ms": ("transfer_copy_submit_ns", "transfer_copy_done_ns"),
    "transfer_worker_wait_ms": ("transfer_start_ns", "transfer_copy_done_ns"),
    "transfer_post_copy_queue_ms": ("transfer_copy_done_ns", "transfer_register_start_ns"),
    "transfer_register_ms": ("transfer_register_start_ns", "transfer_register_done_ns"),
    "transfer_publish_finalize_ms": ("transfer_publish_ready_ns", "publish_done_ns"),
    "transfer_total_ms": ("transfer_copy_start_ns", "publish_done_ns"),
    "receive_ms": ("publish_done_ns", "receive_done_ns"),
    "mm_prepare_ms": ("receive_done_ns", "language_ready_ns"),
    "receive_metadata_wait_ms": ("publish_done_ns", "receive_metadata_ns"),
    "receive_setup_ms": ("receive_metadata_ns", "receive_setup_done_ns"),
    "receive_transfer_wait_ms": ("receive_setup_done_ns", "receive_transfer_done_ns"),
    "receive_completion_to_materialize_ms": (
        "receive_transfer_done_ns",
        "receive_materialize_start_ns",
    ),
    "receive_materialize_wait_ms": ("receive_materialize_start_ns", "receive_materialize_done_ns"),
    "receive_poll_delay_ms": ("receive_materialize_done_ns", "receive_embedding_ns"),
    "receive_finalize_ms": ("receive_embedding_ns", "receive_done_ns"),
    "receive_concat_ms": ("receive_concat_start_ns", "receive_concat_done_ns"),
    "receive_extra_meta_ms": ("receive_extra_meta_start_ns", "receive_extra_meta_done_ns"),
    "receive_result_pack_ms": ("receive_done_ns", "receive_result_ready_ns"),
    "language_prepare_submit_ms": ("receive_result_ready_ns", "language_prepare_submit_ns"),
    "language_prepare_queue_ms": ("language_prepare_submit_ns", "language_prepare_start_ns"),
    "language_prepare_ms": ("language_prepare_start_ns", "language_ready_ns"),
    "language_prepare_wait_ms": ("receive_result_ready_ns", "language_apply_start_ns"),
    "language_get_mm_data_ms": ("language_apply_start_ns", "language_get_mm_data_done_ns"),
    "language_radix_finalize_ms": ("language_get_mm_data_done_ns", "language_ready_ns"),
    "receive_mm_ms": ("publish_done_ns", "language_ready_ns"),
    "language_admission_wait_ms": ("language_ready_ns", "language_scheduler_pickup_ns"),
    "language_queue_after_pickup_ms": ("language_scheduler_pickup_ns", "language_prefill_start_ns"),
    "language_queue_ms": ("language_ready_ns", "language_prefill_start_ns"),
    # Scheduler-observed time includes dispatch and result processing.
    "prefill_observed_ms": ("language_prefill_start_ns", "language_prefill_done_ns"),
    "total_to_prefill_ms": ("enqueue_ns", "language_prefill_start_ns"),
    "total_to_prefill_done_ms": ("enqueue_ns", "language_prefill_done_ns"),
}

_PIPELINE_DURATIONS = {
    "server_postprocess_ms": "encode_server_postprocess_duration_ns",
    "server_token_count_ms": "encode_token_count_duration_ns",
    "server_embedding_slice_ms": "encode_embedding_slice_duration_ns",
    "server_split_compile_wait_ms": "encode_split_compile_wait_duration_ns",
    "server_split_dispatch_ms": "encode_split_dispatch_duration_ns",
    "server_metadata_ms": "encode_metadata_duration_ns",
    "server_result_pack_ms": "encode_result_pack_duration_ns",
    "server_postprocess_residual_ms": "encode_server_postprocess_residual_ns",
    "runtime_postprocess_ms": "runtime_postprocess_duration_ns",
    "runtime_metadata_prepare_ms": "runtime_metadata_prepare_duration_ns",
    "runtime_embedding_data_ms": "runtime_embedding_data_duration_ns",
    "runtime_result_pack_ms": "runtime_result_pack_duration_ns",
    "runtime_postprocess_residual_ms": "runtime_postprocess_residual_ns",
    "runtime_timing_attach_ms": "runtime_timing_attach_duration_ns",
}

_PREPROCESS_SPANS = {
    "dispatch_ms": ("dispatch_start_ns", "enqueue_ns"),
    "admission_ms": ("preprocess_start_ns", "preprocess_request_start_ns"),
    "image_load_ms": ("image_load_start_ns", "image_load_done_ns"),
    "processor_queue_ms": ("processor_submit_ns", "processor_start_ns"),
    "processor_ms": ("processor_start_ns", "processor_done_ns"),
    "finalize_ms": ("processor_done_ns", "preprocess_request_done_ns"),
    "request_total_ms": ("preprocess_request_start_ns", "preprocess_request_done_ns"),
    "batch_tail_ms": ("preprocess_request_done_ns", "preprocess_done_ns"),
}

_PIPELINE_FIELDS = frozenset(
    field for span in _PIPELINE_SPANS.values() for field in span
) | frozenset(_PIPELINE_DURATIONS.values())
_PREPROCESS_FIELDS = frozenset(field for span in _PREPROCESS_SPANS.values() for field in span)


def _span_ms(timing: dict[str, int], spans: dict[str, tuple[str, str]]) -> dict[str, float]:
    return {
        name: max(0, timing[end] - timing[start]) / 1_000_000
        for name, (start, end) in spans.items()
    }


def log_encoder_pipeline_timing(req_id: str, timing: dict[str, int]) -> None:
    if timing.get("_encoder_pipeline_logged") or not _PIPELINE_FIELDS.issubset(timing):
        return
    metrics = _span_ms(timing, _PIPELINE_SPANS)
    metrics.update(
        (name, max(0, timing[field]) / 1_000_000) for name, field in _PIPELINE_DURATIONS.items()
    )
    overlap_start = max(timing["transfer_copy_submit_ns"], timing["transfer_register_start_ns"])
    overlap_end = min(timing["transfer_copy_done_ns"], timing["transfer_register_done_ns"])
    metrics["transfer_copy_register_overlap_ms"] = max(0, overlap_end - overlap_start) / 1_000_000
    timestamps = " ".join(f"{key}={value}" for key, value in timing.items() if key.endswith("_ns"))
    logger.info(
        "ENCODER-PIPELINE-TIME req_id=%s %s %s",
        req_id,
        timestamps,
        " ".join(f"{name}={value:.3f}" for name, value in metrics.items()),
    )
    timing["_encoder_pipeline_logged"] = 1
    if _PREPROCESS_FIELDS.issubset(timing):
        logger.info(
            "ENCODER-PREPROCESS-TIME req_id=%s %s",
            req_id,
            " ".join(
                f"{name}={value:.3f}" for name, value in _span_ms(timing, _PREPROCESS_SPANS).items()
            ),
        )


_TRANSFER_INFLIGHT_RE = re.compile(r"ENCODER-RAIDEN-INFLIGHT (?P<body>[^\n]+)")
_KEY_VALUE_RE = re.compile(r"([a-z_]+)=([^\s]+)")
_TERMINAL_EVENTS = frozenset({"sent", "failed", "release", "close"})


def _summarize_window(
    events: list[dict],
    encoders: list[str],
    *,
    start_ns: int,
    end_ns: int,
) -> dict:
    states = {encoder: (0, 0) for encoder in encoders}
    for event in events:
        if event["time_ns"] >= start_ns:
            break
        states[event["encoder"]] = (event["groups"], event["requests"])

    current_groups = sum(state[0] for state in states.values())
    current_requests = sum(state[1] for state in states.values())
    peak_groups = current_groups
    peak_requests = current_requests
    group_area_ns = 0
    request_area_ns = 0
    time_ns_by_groups: dict[int, int] = {}
    cursor_ns = start_ns
    window_events = []

    for event in events:
        event_ns = event["time_ns"]
        if event_ns < start_ns:
            continue
        if event_ns > end_ns:
            break
        delta_ns = max(0, event_ns - cursor_ns)
        group_area_ns += current_groups * delta_ns
        request_area_ns += current_requests * delta_ns
        time_ns_by_groups[current_groups] = time_ns_by_groups.get(current_groups, 0) + delta_ns

        states[event["encoder"]] = (event["groups"], event["requests"])
        current_groups = sum(state[0] for state in states.values())
        current_requests = sum(state[1] for state in states.values())
        peak_groups = max(peak_groups, current_groups)
        peak_requests = max(peak_requests, current_requests)
        cursor_ns = max(cursor_ns, event_ns)
        window_events.append(event)

    delta_ns = end_ns - cursor_ns
    group_area_ns += current_groups * delta_ns
    request_area_ns += current_requests * delta_ns
    time_ns_by_groups[current_groups] = time_ns_by_groups.get(current_groups, 0) + delta_ns
    duration_ns = end_ns - start_ns
    busy_ns = duration_ns - time_ns_by_groups.get(0, 0)

    return {
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_s": duration_ns / 1e9,
        "n_events": len(window_events),
        "starts": sum(event["group_size"] for event in window_events if event["event"] == "start"),
        "completions": sum(
            event["group_size"] for event in window_events if event["event"] == "sent"
        ),
        "failures": sum(
            event["group_size"] for event in window_events if event["event"] == "failed"
        ),
        "mean_groups": group_area_ns / duration_ns,
        "mean_requests": request_area_ns / duration_ns,
        "peak_groups": peak_groups,
        "peak_requests": peak_requests,
        "busy_fraction": busy_ns / duration_ns,
        "busy_mean_groups": group_area_ns / busy_ns if busy_ns else 0.0,
        "busy_mean_requests": request_area_ns / busy_ns if busy_ns else 0.0,
        "time_fraction_by_groups": {
            str(groups): elapsed_ns / duration_ns
            for groups, elapsed_ns in sorted(time_ns_by_groups.items())
            if elapsed_ns
        },
    }


def summarize_raiden_transfer_inflight(
    log_paths: list[Path],
    *,
    start_ns: int,
    end_ns: int,
) -> dict:
    """Summarize time-weighted Raiden sender occupancy in a wall-clock window."""
    if end_ns <= start_ns:
        raise ValueError("transfer inflight window must have positive duration")

    events = []
    event_index = 0
    for path in log_paths:
        for match in _TRANSFER_INFLIGHT_RE.finditer(path.read_text()):
            row = dict(_KEY_VALUE_RE.findall(match.group("body")))
            try:
                event_ns = int(row["time_ns"])
                groups = int(row["inflight_groups"])
                requests = int(row["inflight_requests"])
                group_size = int(row.get("group_size", 1))
            except (KeyError, ValueError):
                continue
            events.append(
                {
                    "time_ns": event_ns,
                    "encoder": path.name,
                    "event": row.get("event", "unknown"),
                    "group_size": group_size,
                    "groups": groups,
                    "requests": requests,
                    "index": event_index,
                }
            )
            event_index += 1
    events.sort(key=lambda item: (item["time_ns"], item["index"]))

    summary = _summarize_window(
        events,
        [path.name for path in log_paths],
        start_ns=start_ns,
        end_ns=end_ns,
    )
    window_events = [event for event in events if start_ns <= event["time_ns"] <= end_ns]
    starts = [event for event in window_events if event["event"] == "start"]
    terminals = [event for event in window_events if event["event"] in _TERMINAL_EVENTS]
    active_start_ns = starts[0]["time_ns"] if starts else None
    active_end_ns = terminals[-1]["time_ns"] if terminals else None
    active_window = None
    if (
        active_start_ns is not None
        and active_end_ns is not None
        and active_end_ns > active_start_ns
    ):
        active_window = _summarize_window(
            events,
            [path.name for path in log_paths],
            start_ns=active_start_ns,
            end_ns=active_end_ns,
        )

    return {
        "available": bool(events),
        **summary,
        "active_window": active_window,
    }
