"""Per-request time_stats for PD disaggregation.

Records wall-clock marks at a request's lifecycle points and derives a
phase-by-phase latency breakdown. Each role (prefill / decode) records its
own marks in its own process; the breakdown is logged when
``--enable-request-time-stats-logging`` is set.

The structure is deliberately dependency-free (no jax / numpy) so the
hot-path cost is a dict insert and it is trivially unit-testable.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


# Ordered (start_mark, end_mark, phase_label) per role. A phase is emitted
# only when both endpoint marks are present, so partial requests degrade
# gracefully instead of reporting bogus durations.
_PHASE_SPECS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "prefill": (
        ("queue_entry", "forward_start", "queue"),
        ("forward_start", "forward_done", "forward"),
        ("forward_done", "transfer_start", "stage"),
        ("transfer_start", "first_chunk_registered", "first_chunk_register_wait"),
        ("first_chunk_registered", "last_chunk_registered", "chunk_register_span"),
        ("last_chunk_registered", "sender_done", "sender_done_wait"),
        ("sender_done", "transfer_done", "prefill_reap_gap"),
        ("last_chunk_registered", "transfer_done", "transfer_tail"),
        ("transfer_start", "transfer_done", "transfer"),
        ("queue_entry", "transfer_done", "total"),
    ),
    "decode": (
        ("bootstrap_start", "bootstrap_done", "bootstrap"),
        ("prealloc_entry", "metadata_ready", "metadata_wait"),
        ("metadata_ready", "kv_alloc_done", "kv_alloc"),
        ("kv_alloc_done", "receiver_init_done", "receiver_init"),
        ("receiver_init_done", "transfer_entry", "transfer_setup"),
        ("prealloc_entry", "transfer_entry", "prealloc_wait"),
        ("transfer_entry", "first_chunk_start_read", "first_chunk_wait"),
        ("first_chunk_start_read", "last_chunk_start_read", "chunk_start_span"),
        ("last_chunk_start_read", "done_recving", "transfer_tail"),
        ("done_recving", "enqueue_decode", "enqueue_decode"),
        ("transfer_entry", "first_token", "kv_wait"),
        ("bootstrap_start", "first_token", "total"),
    ),
}


class TimeStats:
    """Lifecycle marks + derived phase durations for one request."""

    __slots__ = ("role", "marks", "duration_totals", "counts", "_clock")

    def __init__(self, role: str, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self.role = role
        self.marks: dict[str, float] = {}
        self.duration_totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._clock = clock

    def mark(self, name: str, *, overwrite: bool = False) -> None:
        """Record the current time for ``name`` (first write wins)."""
        if overwrite or name not in self.marks:
            self.marks[name] = self._clock()

    def add_duration(self, name: str, seconds: float) -> None:
        """Accumulate a repeatable duration, e.g. one entry per chunk."""
        self.duration_totals[name] = self.duration_totals.get(name, 0.0) + max(
            0.0,
            float(seconds),
        )
        self.increment(name)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + int(amount)

    def duration(self, start: str, end: str) -> float | None:
        a = self.marks.get(start)
        b = self.marks.get(end)
        if a is None or b is None:
            return None
        if b < a:
            return None
        return b - a

    def phases(self) -> dict[str, float]:
        """Role-specific phase durations, skipping any with unset endpoints."""
        out: dict[str, float] = {}
        for start, end, label in _PHASE_SPECS.get(self.role, ()):
            d = self.duration(start, end)
            if d is not None:
                out[label] = d
        return out


def format_time_stats(ts: TimeStats, *, req_id: str) -> str:
    phases = ts.phases()
    fields = [f"{label}={dur * 1000:.1f}ms" for label, dur in phases.items()]
    for label, total in ts.duration_totals.items():
        count = max(1, ts.counts.get(label, 0))
        fields.append(f"{label}_sum={total * 1000:.1f}ms")
        fields.append(f"{label}_count={count}")
        fields.append(f"{label}_avg={(total / count) * 1000:.1f}ms")
    for label, count in ts.counts.items():
        if label not in ts.duration_totals:
            fields.append(f"{label}_count={count}")

    if fields:
        body = " ".join(fields)
    else:
        # Unknown role or no derivable phases: fall back to raw marks so the
        # line is still informative.
        body = " ".join(sorted(ts.marks)) or "(no marks)"
    return f"PD-TIME-STATS role={ts.role} req_id={req_id} {body}"


def maybe_log_time_stats(ts: TimeStats | None, *, req_id: str, enabled: bool) -> None:
    """Log the phase breakdown when ``enabled`` and ``ts`` is present."""
    if not enabled or ts is None:
        return
    logger.info("%s", format_time_stats(ts, req_id=req_id))
