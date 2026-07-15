from __future__ import annotations

import logging
import time
from collections import deque
from collections import defaultdict
from typing import TYPE_CHECKING

from sgl_jax.srt.managers.schedule_policy import PrefillAdder
from sgl_jax.srt.managers.scheduler import Req, ScheduleBatch
from sgl_jax.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sgl_jax.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")


class ScheduleActivityTracker:
    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = float(window_seconds)
        self.events = deque()
        self.last_forward_end: float | None = None
        self.last_mode: str | None = None
        self.last_batch_size = 0
        self.last_new_tokens = 0
        self.last_forward_s = 0.0
        self.last_idle_gap_s: float | None = None

    def record_forward(
        self,
        *,
        mode: str,
        batch_size: int,
        new_tokens: int,
        start: float,
        end: float,
    ) -> None:
        duration_s = max(0.0, end - start)
        idle_gap_s = (
            None
            if self.last_forward_end is None
            else max(0.0, start - self.last_forward_end)
        )
        event = {
            "mode": mode,
            "batch_size": int(batch_size),
            "new_tokens": int(new_tokens),
            "start": float(start),
            "end": float(end),
            "duration_s": duration_s,
            "idle_gap_s": idle_gap_s,
        }
        self.events.append(event)
        self.last_forward_end = float(end)
        self.last_mode = mode
        self.last_batch_size = int(batch_size)
        self.last_new_tokens = int(new_tokens)
        self.last_forward_s = duration_s
        self.last_idle_gap_s = idle_gap_s
        self._prune(float(end))

    def snapshot(self, now: float | None = None) -> dict:
        now = time.perf_counter() if now is None else float(now)
        self._prune(now)
        ret = {
            "last_mode": self.last_mode,
            "last_batch_size": self.last_batch_size,
            "last_new_tokens": self.last_new_tokens,
            "last_forward_ms": _round_ms(self.last_forward_s),
            "last_idle_gap_ms": (
                None if self.last_idle_gap_s is None else _round_ms(self.last_idle_gap_s)
            ),
        }
        for window_s in (1.0, 5.0, 60.0):
            ret.update(self._snapshot_window(now, window_s))
        return ret

    def _snapshot_window(self, now: float, window_s: float) -> dict:
        start_cutoff = now - window_s
        count = 0
        intervals: list[tuple[float, float]] = []
        for event in self.events:
            overlap_start = max(event["start"], start_cutoff)
            overlap_end = min(event["end"], now)
            if overlap_end <= overlap_start:
                continue
            count += 1
            intervals.append((overlap_start, overlap_end))

        busy_s = _merged_interval_seconds(intervals)
        idle_s = max(0.0, window_s - busy_s)

        suffix = str(int(window_s))
        return {
            f"forward_count_{suffix}s": count,
            f"busy_ms_{suffix}s": _round_ms(busy_s),
            f"busy_fraction_{suffix}s": round(min(1.0, busy_s / window_s), 4),
            f"idle_gap_ms_{suffix}s": _round_ms(idle_s),
        }

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.events and self.events[0]["end"] < cutoff:
            self.events.popleft()


def _round_ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 3)


def _merged_interval_seconds(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0

    intervals = sorted(intervals)
    total_s = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total_s += current_end - current_start
        current_start, current_end = start, end

    total_s += current_end - current_start
    return total_s


class SchedulerMetricsMixin:
    def init_metrics(self: Scheduler):
        self.last_gen_throughput: float = 0.0
        self.last_input_throughput: float = 0.0
        self.step_time_dict = defaultdict(list)  # Dict[batch size -> step time]
        self.spec_num_total_accepted_tokens = 0
        self.spec_num_total_forward_ct = 0
        self.cum_spec_accept_length = 0
        self.cum_spec_accept_count = 0
        self.total_retracted_reqs = 0
        self.schedule_activity = ScheduleActivityTracker()

    def record_schedule_activity(
        self: Scheduler,
        batch: ScheduleBatch,
        start: float,
        end: float,
    ) -> None:
        tracker = getattr(self, "schedule_activity", None)
        if tracker is None:
            return
        tracker.record_forward(
            mode=_forward_mode_label(batch),
            batch_size=batch.batch_size(),
            new_tokens=_forward_new_tokens(batch),
            start=start,
            end=end,
        )

    def get_schedule_activity_state(self: Scheduler) -> dict:
        tracker = getattr(self, "schedule_activity", None)
        return tracker.snapshot() if tracker is not None else {}

    def log_prefill_stats(
        self: Scheduler,
        adder: PrefillAdder,
        can_run_list: list[Req],
        running_bs: int,
    ):
        gap_latency = time.perf_counter() - self.last_prefill_stats_tic
        self.last_prefill_stats_tic = time.perf_counter()
        self.last_input_throughput = self.last_prefill_tokens / gap_latency
        self.last_prefill_tokens = adder.log_input_tokens

        if self.is_hybrid:
            (
                full_num_used,
                swa_num_used,
                full_token_usage,
                swa_token_usage,
                _,
                _,
                _,
                _,
            ) = self._get_swa_token_info()
            num_used = max(full_num_used, swa_num_used)
            token_usage = max(full_token_usage, swa_token_usage)
            token_msg = (
                f"full token usage: {full_token_usage:.2f}, "
                f"swa token usage: {swa_token_usage:.2f}, "
            )
        else:
            num_used, token_usage, _, _ = self._get_token_info()
            token_msg = f"token usage: {token_usage:.2f}, "

        num_new_seq = sum(len(v) for v in adder.can_run_list.values())
        f = (
            f"Prefill batch. "
            f"#new-seq: {num_new_seq}, "
            f"#new-token: {adder.log_input_tokens}, "
            f"#cached-token: {adder.log_hit_tokens}, "
            f"{token_msg}"
        )

        f += f"#running-req: {running_bs}, "
        if self.dp_size > 1:
            per_dp_prefill = [len(adder.can_run_list[i]) for i in range(self.dp_size)]
            per_dp_running = [
                (
                    len(self.running_batch.reqs_info[i].reqs)
                    if self.running_batch.reqs_info[i].reqs
                    else 0
                )
                for i in range(self.dp_size)
            ]
            f += f"#prefill per DP: {per_dp_prefill}, #running per DP: {per_dp_running}, "

        f += f"#queue-req: {len(self.waiting_queue)}, "

        logger.info(f)

    def log_decode_stats(self: Scheduler, running_batch: ScheduleBatch = None):
        batch = running_batch or self.running_batch

        gap_latency = time.perf_counter() - self.last_decode_stats_tic
        self.last_decode_stats_tic = time.perf_counter()
        self.last_gen_throughput = self.num_generated_tokens / gap_latency
        self.num_generated_tokens = 0
        num_running_reqs = batch.batch_size()
        if self.is_hybrid:
            (
                full_num_used,
                swa_num_used,
                full_token_usage,
                swa_token_usage,
                _,
                _,
                _,
                _,
            ) = self._get_swa_token_info()
            num_used = max(full_num_used, swa_num_used)
            token_usage = max(full_token_usage, swa_token_usage)
            token_msg = (
                f"#full token: {full_num_used}, "
                f"full token usage: {full_token_usage:.2f}, "
                f"#swa token: {swa_num_used}, "
                f"swa token usage: {swa_token_usage:.2f}, "
            )
        else:
            num_used, token_usage, _, _ = self._get_token_info()
            token_msg = f"#token: {num_used}, " f"token usage: {token_usage:.2f}, "

        if RECORD_STEP_TIME:
            self.step_time_dict[num_running_reqs].append(
                gap_latency / self.server_args.decode_log_interval
            )

        msg = f"Decode batch. #running-req: {num_running_reqs}, {token_msg}"

        if batch.dp_size > 1:
            per_dp_running = [len(info.reqs) if info.reqs else 0 for info in batch.reqs_info]
            msg += f"#running-req per DP: {per_dp_running}, "

        if (
            self.spec_algorithm is not None
            and not self.spec_algorithm.is_none()
            and self.draft_token > 0
            and self.spec_num_forward_ct > 0
        ):
            accept_ratio = self.accept_token / self.draft_token
            accept_len = self.accept_token / self.spec_num_forward_ct
            self.accept_token = 0
            self.draft_token = 0
            self.spec_num_forward_ct = 0
            msg += f"accept-len {accept_len:.2f}, accept-ratio {accept_ratio:.2f}, "

        msg += (
            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}, "
        )

        if batch.cache_miss_count > 0:
            msg += f"#cache_miss: {batch.cache_miss_count}"

        logger.info(msg)


def _forward_mode_label(batch: ScheduleBatch) -> str:
    forward_mode = getattr(batch, "forward_mode", None)
    if forward_mode is None:
        return "unknown"
    if forward_mode.is_extend():
        return "prefill"
    if forward_mode.is_decode():
        return "decode"
    if forward_mode.is_idle():
        return "idle"
    if getattr(forward_mode, "is_dummy_first", lambda: False)():
        return "dummy_first"
    return str(forward_mode)


def _forward_new_tokens(batch: ScheduleBatch) -> int:
    forward_mode = getattr(batch, "forward_mode", None)
    if forward_mode is not None and forward_mode.is_decode():
        return batch.batch_size()

    total = 0
    for info in getattr(batch, "reqs_info", ()) or ():
        for req in getattr(info, "reqs", ()) or ():
            total += int(getattr(req, "extend_input_len", 0) or 0)
    return total
