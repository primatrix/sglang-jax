#!/usr/bin/env python3
"""Warm up and profile one exact-shape GLM-5.2 serving request."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request
from typing import Any, Callable


class HttpTransport:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout_seconds: float,
    ) -> Any:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content = response.read().decode("utf-8")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content

    def post(self, path: str, payload: dict[str, Any], timeout_seconds: float) -> Any:
        return self._request("POST", path, payload, timeout_seconds)

    def get(self, path: str, timeout_seconds: float) -> Any:
        return self._request("GET", path, None, timeout_seconds)


def _write_json(path: pathlib.Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _send_generation(
    *,
    name: str,
    request_payload: dict[str, Any],
    output_dir: pathlib.Path,
    transport: Any,
    timeout_seconds: float,
    monotonic: Callable[[], float],
) -> dict[str, float]:
    started_at = monotonic()
    response = transport.post("/generate", request_payload, timeout_seconds)
    finished_at = monotonic()
    _write_json(output_dir / f"{name}.json", response)
    return {
        "started_at_monotonic": started_at,
        "finished_at_monotonic": finished_at,
        "duration_seconds": finished_at - started_at,
    }


def _remaining_seconds(
    deadline: float, monotonic: Callable[[], float], timeout_seconds: float
) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError(
            f"profile did not become idle within {timeout_seconds} seconds"
        )
    return remaining


def run_profile_session(
    *,
    output_dir: str | pathlib.Path,
    profile_output_dir: str,
    warmup_request: dict[str, Any],
    measured_request: dict[str, Any],
    num_steps: int,
    host_tracer_level: int,
    python_tracer_level: int,
    timeout_seconds: float,
    poll_interval_seconds: float = 1.0,
    transport: Any,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timeline: dict[str, Any] = {
        "settings": {
            "profile_output_dir": profile_output_dir,
            "num_steps": num_steps,
            "host_tracer_level": host_tracer_level,
            "python_tracer_level": python_tracer_level,
            "profile_by_stage": True,
            "profile_stages": ["prefill", "decode"],
        },
        "session_started_at_monotonic": monotonic(),
    }

    timeline["warmup"] = _send_generation(
        name="profile_warmup",
        request_payload=warmup_request,
        output_dir=output_path,
        transport=transport,
        timeout_seconds=timeout_seconds,
        monotonic=monotonic,
    )

    profile_payload = {
        "output_dir": profile_output_dir,
        "num_steps": num_steps,
        "host_tracer_level": host_tracer_level,
        "python_tracer_level": python_tracer_level,
        "profile_by_stage": True,
        "profile_stages": ["prefill", "decode"],
    }
    transport.post("/start_profile", profile_payload, timeout_seconds)
    profile_started_at = monotonic()
    timeline["profile_started_at_monotonic"] = profile_started_at
    deadline = profile_started_at + timeout_seconds
    profile_active = True
    try:
        timeline["measured"] = _send_generation(
            name="profile_measured",
            request_payload=measured_request,
            output_dir=output_path,
            transport=transport,
            timeout_seconds=_remaining_seconds(deadline, monotonic, timeout_seconds),
            monotonic=monotonic,
        )

        poll_count = 0
        while True:
            status_payload = transport.get(
                "/profile_status",
                _remaining_seconds(deadline, monotonic, timeout_seconds),
            )
            poll_count += 1
            status = (
                status_payload.get("status")
                if isinstance(status_payload, dict)
                else None
            )
            if status == "idle":
                profile_active = False
                timeline["final_profile_status"] = status
                timeline["profile_idle_at_monotonic"] = monotonic()
                timeline["profile_status_poll_count"] = poll_count
                _write_json(output_path / "profile_timeline.json", timeline)
                return timeline
            sleep(poll_interval_seconds)
    finally:
        if profile_active:
            try:
                transport.post(
                    "/stop_profile", {}, min(float(timeout_seconds), 30.0)
                )
            except Exception:
                pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-output-dir", required=True)
    parser.add_argument("--warmup-request", required=True)
    parser.add_argument("--measured-request", required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--host-tracer-level", type=int, required=True)
    parser.add_argument("--python-tracer-level", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    warmup_request = json.loads(pathlib.Path(args.warmup_request).read_text(encoding="utf-8"))
    measured_request = json.loads(
        pathlib.Path(args.measured_request).read_text(encoding="utf-8")
    )
    timeline = run_profile_session(
        output_dir=args.output_dir,
        profile_output_dir=args.profile_output_dir,
        warmup_request=warmup_request,
        measured_request=measured_request,
        num_steps=args.num_steps,
        host_tracer_level=args.host_tracer_level,
        python_tracer_level=args.python_tracer_level,
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        transport=HttpTransport(args.base_url),
    )
    print(
        "GLM52_DSA_PROFILE_CAPTURED "
        f"warmup_seconds={timeline['warmup']['duration_seconds']:.6f} "
        f"measured_seconds={timeline['measured']['duration_seconds']:.6f}"
    )


if __name__ == "__main__":
    main()
