from __future__ import annotations

import re
from pathlib import Path

_TRANSFER_INFLIGHT_RE = re.compile(r"ENCODER-RAIDEN-INFLIGHT (?P<body>[^\n]+)")
_KEY_VALUE_RE = re.compile(r"([a-z_]+)=([^\s]+)")


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
            except (KeyError, ValueError):
                continue
            events.append(
                {
                    "time_ns": event_ns,
                    "encoder": path.name,
                    "event": row.get("event", "unknown"),
                    "groups": groups,
                    "requests": requests,
                    "index": event_index,
                }
            )
            event_index += 1
    events.sort(key=lambda item: (item["time_ns"], item["index"]))

    states = {path.name: (0, 0) for path in log_paths}
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

    return {
        "available": bool(events),
        "n_events": len(window_events),
        "starts": sum(event["event"] == "start" for event in window_events),
        "completions": sum(event["event"] == "sent" for event in window_events),
        "mean_groups": group_area_ns / duration_ns,
        "mean_requests": request_area_ns / duration_ns,
        "peak_groups": peak_groups,
        "peak_requests": peak_requests,
        "time_fraction_by_groups": {
            str(groups): elapsed_ns / duration_ns
            for groups, elapsed_ns in sorted(time_ns_by_groups.items())
        },
    }
