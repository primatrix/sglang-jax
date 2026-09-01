from types import SimpleNamespace

from sgl_jax.srt.managers.scheduler import Scheduler


def test_language_logs_encoder_pipeline_timing(caplog):
    timing = {
        "enqueue_ns": 1_000_000,
        "dequeue_ns": 2_000_000,
        "encode_done_ns": 4_000_000,
        "publish_done_ns": 7_000_000,
        "receive_done_ns": 9_000_000,
        "language_ready_ns": 11_000_000,
    }
    req = SimpleNamespace(rid="request-0", encoder_timing=timing)
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        reqs_info=[SimpleNamespace(reqs=[req])],
    )
    scheduler = SimpleNamespace(server_args=SimpleNamespace(enable_request_time_stats_logging=True))

    caplog.set_level("INFO")
    Scheduler._mark_encoder_prefill_start(scheduler, batch)
    Scheduler._log_encoder_pipeline_timing(scheduler, batch)

    assert "ENCODER-PIPELINE-TIME req_id=request-0" in caplog.text
    assert "queue_ms=1.000" in caplog.text
    assert "encode_ms=2.000" in caplog.text
    assert "publish_ms=3.000" in caplog.text
    assert "receive_ms=2.000" in caplog.text
    assert "mm_prepare_ms=2.000" in caplog.text
    assert "receive_mm_ms=4.000" in caplog.text
    assert timing["language_prefill_start_ns"] >= timing["language_ready_ns"]
    assert timing["language_prefill_done_ns"] >= timing["language_prefill_start_ns"]
