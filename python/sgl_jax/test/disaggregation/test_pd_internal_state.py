from types import SimpleNamespace


class FakeQueue:
    def __init__(self, entries):
        self._entries = list(entries)

    def __len__(self):
        return len(self._entries)

    def items_fifo(self):
        return list(self._entries)


class FakeAllocator:
    def available_size(self, dp_rank=0):
        return 1000 - dp_rank

    def get_kvcache(self):
        return SimpleNamespace(mem_usage=0.25)


class FakeBatch:
    def __init__(self, reqs_by_dp):
        self.reqs_info = [SimpleNamespace(reqs=reqs) for reqs in reqs_by_dp]

    def is_empty(self):
        return not any(info.reqs for info in self.reqs_info)

    def batch_size(self):
        return sum(len(info.reqs) for info in self.reqs_info)


def test_pd_decode_admission_state_empty_when_no_queue():
    from sgl_jax.srt.managers.scheduler import Scheduler

    scheduler = SimpleNamespace(disagg_prealloc_queue=None)

    assert Scheduler._get_pd_decode_admission_state(scheduler) == {}


def test_internal_state_reports_enable_overlap():
    from sgl_jax.srt.managers.scheduler import Scheduler

    scheduler = SimpleNamespace(
        last_gen_throughput=0.0,
        token_to_kv_pool_allocator=FakeAllocator(),
        max_total_num_tokens=1024,
        _engine_paused=False,
        waiting_queue=[],
        running_batch=FakeBatch([[]]),
        cur_batch=None,
        last_batch=None,
        chunked_reqs=[None],
        tree_cache=None,
        req_to_token_pool=None,
        num_generated_tokens=0,
        forward_ct_decode=0,
        new_token_ratio=0.0,
        init_new_token_ratio=0.0,
        disagg_prefill_queue=None,
        disagg_prealloc_queue=None,
        disagg_transfer_queue=None,
        enable_overlap=True,
        _get_pd_decode_admission_state=lambda: {},
        get_schedule_activity_state=lambda: {},
    )

    output = Scheduler.get_internal_state(scheduler, SimpleNamespace())

    assert output.internal_state["enable_overlap"] is True


def test_pd_decode_admission_state_counts_queue_tokens_and_dp_capacity():
    from sgl_jax.srt.managers.scheduler import Scheduler

    req = SimpleNamespace(
        origin_input_ids=[1, 2, 3],
        pd_time_stats=SimpleNamespace(marks={}),
    )
    scheduler = SimpleNamespace(
        disagg_prealloc_queue=FakeQueue([SimpleNamespace(req=req)]),
        disagg_transfer_queue=FakeQueue([object(), object()]),
        token_to_kv_pool_allocator=FakeAllocator(),
        dp_size=2,
        running_batch=FakeBatch([[object()], [object(), object()]]),
        server_args=SimpleNamespace(
            disaggregation_max_inflight_transfers=8,
            disaggregation_num_reserved_decode_tokens=512,
        ),
    )

    state = Scheduler._get_pd_decode_admission_state(scheduler)

    assert state["prealloc_queue_size"] == 1
    assert state["transfer_queue_size"] == 2
    assert state["running_reqs"] == 3
    assert state["max_inflight_transfers"] == 8
    assert state["reserved_decode_tokens"] == 512
    assert state["kv_available_by_dp"] == [1000, 999]
    assert state["oldest_prealloc_wait_ms"] is None
    assert state["pending_prealloc_prompt_tokens"] == 3


def test_pd_decode_admission_state_reports_oldest_prealloc_wait(monkeypatch):
    from sgl_jax.srt.managers import scheduler as scheduler_module
    from sgl_jax.srt.managers.scheduler import Scheduler

    monkeypatch.setattr(scheduler_module.time, "perf_counter", lambda: 12.5)
    req = SimpleNamespace(
        origin_input_ids=[1],
        pd_time_stats=SimpleNamespace(marks={"prealloc_entry": 10.0}),
    )
    scheduler = SimpleNamespace(
        disagg_prealloc_queue=FakeQueue([SimpleNamespace(req=req)]),
        disagg_transfer_queue=FakeQueue([]),
        token_to_kv_pool_allocator=FakeAllocator(),
        dp_size=1,
        running_batch=None,
        server_args=SimpleNamespace(
            disaggregation_max_inflight_transfers=8,
            disaggregation_num_reserved_decode_tokens=512,
        ),
    )

    state = Scheduler._get_pd_decode_admission_state(scheduler)

    assert state["oldest_prealloc_wait_ms"] == 2500.0


def test_schedule_activity_reports_recent_forward_busy_and_idle_windows():
    from sgl_jax.srt.managers.scheduler_metrics_mixin import ScheduleActivityTracker

    tracker = ScheduleActivityTracker(window_seconds=60.0)

    tracker.record_forward(
        mode="prefill",
        batch_size=2,
        new_tokens=4096,
        start=10.0,
        end=10.25,
    )
    tracker.record_forward(
        mode="decode",
        batch_size=8,
        new_tokens=8,
        start=10.75,
        end=10.85,
    )

    state = tracker.snapshot(now=11.0)

    assert state["last_mode"] == "decode"
    assert state["last_batch_size"] == 8
    assert state["last_new_tokens"] == 8
    assert state["last_forward_ms"] == 100.0
    assert state["last_idle_gap_ms"] == 500.0
    assert state["forward_count_1s"] == 2
    assert state["busy_ms_1s"] == 350.0
    assert state["busy_fraction_1s"] == 0.35
    assert state["idle_gap_ms_1s"] == 650.0
    assert state["forward_count_5s"] == 2


def test_schedule_activity_prunes_old_forward_events():
    from sgl_jax.srt.managers.scheduler_metrics_mixin import ScheduleActivityTracker

    tracker = ScheduleActivityTracker(window_seconds=1.0)

    tracker.record_forward(
        mode="prefill",
        batch_size=1,
        new_tokens=2048,
        start=1.0,
        end=1.1,
    )
    tracker.record_forward(
        mode="decode",
        batch_size=1,
        new_tokens=1,
        start=3.0,
        end=3.2,
    )

    state = tracker.snapshot(now=3.2)

    assert state["forward_count_1s"] == 1
    assert state["busy_ms_1s"] == 200.0
    assert state["idle_gap_ms_1s"] == 800.0
    assert state["last_idle_gap_ms"] == 1900.0


def test_schedule_activity_idle_gap_is_bounded_to_window():
    from sgl_jax.srt.managers.scheduler_metrics_mixin import ScheduleActivityTracker

    tracker = ScheduleActivityTracker(window_seconds=60.0)

    tracker.record_forward(
        mode="decode",
        batch_size=1,
        new_tokens=1,
        start=10.0,
        end=10.1,
    )

    state = tracker.snapshot(now=20.0)

    assert state["forward_count_1s"] == 0
    assert state["busy_ms_1s"] == 0.0
    assert state["idle_gap_ms_1s"] == 1000.0
