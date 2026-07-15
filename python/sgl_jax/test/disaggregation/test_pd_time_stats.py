import pytest


def test_time_stats_mark_can_overwrite_when_tracking_last_event():
    from sgl_jax.srt.disaggregation.req_time_stats import TimeStats

    times = iter([1.0, 2.0])
    stats = TimeStats("prefill", clock=lambda: next(times))

    stats.mark("forward_done")
    stats.mark("forward_done")
    stats.mark("forward_done", overwrite=True)

    assert stats.marks["forward_done"] == 2.0


def test_time_stats_formats_duration_totals_and_counts():
    from sgl_jax.srt.disaggregation.req_time_stats import (
        TimeStats,
        format_time_stats,
    )

    stats = TimeStats("prefill")
    stats.add_duration("forward_chunk", 0.1)
    stats.add_duration("forward_chunk", 0.3)
    stats.increment("chunks")

    text = format_time_stats(stats, req_id="rid")

    assert "forward_chunk_sum=400.0ms" in text
    assert "forward_chunk_count=2" in text
    assert "forward_chunk_avg=200.0ms" in text
    assert "chunks_count=1" in text


def test_time_stats_reports_new_decode_transfer_phases():
    from sgl_jax.srt.disaggregation.req_time_stats import TimeStats

    times = iter([0.0, 1.0, 1.2, 1.3, 1.5, 2.0, 2.4, 2.6, 2.7])
    stats = TimeStats("decode", clock=lambda: next(times))

    for name in (
        "prealloc_entry",
        "metadata_ready",
        "kv_alloc_done",
        "receiver_init_done",
        "transfer_entry",
        "first_chunk_start_read",
        "last_chunk_start_read",
        "done_recving",
        "enqueue_decode",
    ):
        stats.mark(name)

    phases = stats.phases()

    assert phases["metadata_wait"] == pytest.approx(1.0)
    assert phases["kv_alloc"] == pytest.approx(0.2)
    assert phases["receiver_init"] == pytest.approx(0.1)
    assert phases["first_chunk_wait"] == pytest.approx(0.5)
    assert phases["chunk_start_span"] == pytest.approx(0.4)
    assert phases["transfer_tail"] == pytest.approx(0.2)
    assert phases["enqueue_decode"] == pytest.approx(0.1)


def test_time_stats_reports_prefill_sender_done_and_reap_gap():
    from sgl_jax.srt.disaggregation.req_time_stats import TimeStats

    times = iter([0.0, 2.0, 2.3, 2.5])
    stats = TimeStats("prefill", clock=lambda: next(times))

    for name in (
        "transfer_start",
        "last_chunk_registered",
        "sender_done",
        "transfer_done",
    ):
        stats.mark(name)

    phases = stats.phases()

    assert phases["sender_done_wait"] == pytest.approx(0.3)
    assert phases["prefill_reap_gap"] == pytest.approx(0.2)
    assert phases["transfer_tail"] == pytest.approx(0.5)


def test_prefill_queue_marks_sender_done_when_terminal_success():
    from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
    from sgl_jax.srt.disaggregation.prefill import PrefillBootstrapQueue
    from sgl_jax.srt.disaggregation.req_time_stats import TimeStats

    class Sender:
        def poll(self):
            return KVPoll.SUCCESS

    stats = TimeStats("prefill", clock=lambda: 42.0)
    queue = PrefillBootstrapQueue()
    queue.add("rid", Sender(), time_stats=stats)

    terminal = queue.drain_terminal()

    assert len(terminal) == 1
    assert terminal[0].req_id == "rid"
    assert stats.marks["sender_done"] == pytest.approx(42.0)


def test_time_stats_skips_negative_duration_for_overlapped_phases():
    from sgl_jax.srt.disaggregation.req_time_stats import TimeStats

    stats = TimeStats("prefill")
    stats.marks["forward_done"] = 2.0
    stats.marks["transfer_start"] = 1.0

    assert stats.duration("forward_done", "transfer_start") is None
    assert "stage" not in stats.phases()
