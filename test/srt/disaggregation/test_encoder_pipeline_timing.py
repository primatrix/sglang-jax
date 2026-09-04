from types import SimpleNamespace

from sgl_jax.srt.managers.scheduler import Scheduler


def test_language_logs_encoder_poll_time(caplog):
    pending = SimpleNamespace(
        recv_req=SimpleNamespace(rid="request-0"),
        started_at=0.0,
        poll=lambda: None,
    )
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            encoder_request_timeout_seconds=0,
            enable_request_time_stats_logging=True,
        ),
        encoder_waiting={"request-0": pending},
    )

    caplog.set_level("INFO")
    assert Scheduler.process_encoder_requests(scheduler, []) == []

    assert "ENCODER-POLL-TIME req_id=request-0" in caplog.text
    assert "duration_ns=" in caplog.text
    assert "status=pending" in caplog.text


def test_language_logs_encoder_pipeline_timing(caplog):
    timing = {
        "enqueue_ns": 1_000_000,
        "dequeue_ns": 2_000_000,
        "preprocess_start_ns": 3_000_000,
        "preprocess_done_ns": 4_000_000,
        "transfer_reserve_start_ns": 4_200_000,
        "transfer_reserve_done_ns": 4_500_000,
        "encode_start_ns": 5_000_000,
        "encode_done_ns": 7_000_000,
        "transfer_copy_start_ns": 7_100_000,
        "transfer_pool_ready_ns": 7_200_000,
        "transfer_copy_submit_ns": 7_400_000,
        "transfer_enqueue_ns": 8_000_000,
        "transfer_start_ns": 10_000_000,
        "transfer_copy_done_ns": 14_000_000,
        "transfer_stage_done_ns": 14_000_000,
        "publish_done_ns": 15_000_000,
        "receive_metadata_ns": 16_000_000,
        "receive_setup_done_ns": 17_000_000,
        "receive_transfer_done_ns": 18_000_000,
        "receive_materialize_start_ns": 19_000_000,
        "receive_materialize_done_ns": 20_000_000,
        "receive_embedding_ns": 21_000_000,
        "receive_done_ns": 22_000_000,
        "receive_concat_start_ns": 22_100_000,
        "receive_concat_done_ns": 22_300_000,
        "receive_extra_meta_start_ns": 22_400_000,
        "receive_extra_meta_done_ns": 22_700_000,
        "receive_result_ready_ns": 22_800_000,
        "language_apply_start_ns": 23_000_000,
        "language_get_mm_data_done_ns": 23_500_000,
        "language_radix_done_ns": 23_800_000,
        "language_ready_ns": 24_000_000,
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
    assert "encode_stage_wait_ms=1.000" in caplog.text
    assert "preprocess_ms=1.000" in caplog.text
    assert "encode_wait_ms=0.200" in caplog.text
    assert "transfer_reserve_ms=0.300" in caplog.text
    assert "encode_dispatch_ms=0.500" in caplog.text
    assert "encode_compute_ms=2.000" in caplog.text
    assert "encode_ms=5.000" in caplog.text
    assert "publish_ms=8.000" in caplog.text
    assert "transfer_handoff_ms=1.000" in caplog.text
    assert "transfer_queue_ms=2.000" in caplog.text
    assert "transfer_pool_setup_ms=0.100" in caplog.text
    assert "transfer_copy_submit_ms=0.200" in caplog.text
    assert "transfer_copy_wait_ms=6.600" in caplog.text
    assert "transfer_worker_wait_ms=4.000" in caplog.text
    assert "transfer_register_ms=1.000" in caplog.text
    assert "transfer_total_ms=7.900" in caplog.text
    assert "receive_ms=7.000" in caplog.text
    assert "mm_prepare_ms=2.000" in caplog.text
    assert "receive_metadata_wait_ms=1.000" in caplog.text
    assert "receive_setup_ms=1.000" in caplog.text
    assert "receive_transfer_wait_ms=1.000" in caplog.text
    assert "receive_completion_to_materialize_ms=1.000" in caplog.text
    assert "receive_materialize_wait_ms=1.000" in caplog.text
    assert "receive_poll_delay_ms=1.000" in caplog.text
    assert "receive_finalize_ms=1.000" in caplog.text
    assert "receive_concat_ms=0.200" in caplog.text
    assert "receive_extra_meta_ms=0.300" in caplog.text
    assert "receive_result_pack_ms=0.800" in caplog.text
    assert "language_pickup_wait_ms=0.200" in caplog.text
    assert "language_get_mm_data_ms=0.500" in caplog.text
    assert "language_radix_finalize_ms=0.500" in caplog.text
    assert "receive_mm_ms=9.000" in caplog.text
    assert timing["language_prefill_start_ns"] >= timing["language_ready_ns"]
    assert timing["language_prefill_done_ns"] >= timing["language_prefill_start_ns"]


def test_language_logs_encoder_preprocess_timing(caplog):
    timing = {
        "dispatch_start_ns": 1_000_000,
        "enqueue_ns": 2_000_000,
        "dequeue_ns": 3_000_000,
        "preprocess_start_ns": 4_000_000,
        "preprocess_request_start_ns": 5_000_000,
        "image_load_start_ns": 6_000_000,
        "image_load_done_ns": 8_000_000,
        "processor_submit_ns": 9_000_000,
        "processor_start_ns": 12_000_000,
        "processor_done_ns": 17_000_000,
        "preprocess_request_done_ns": 18_000_000,
        "preprocess_done_ns": 19_000_000,
        "transfer_reserve_start_ns": 19_100_000,
        "transfer_reserve_done_ns": 19_200_000,
        "encode_start_ns": 20_000_000,
        "encode_done_ns": 21_000_000,
        "transfer_copy_start_ns": 21_020_000,
        "transfer_pool_ready_ns": 21_030_000,
        "transfer_copy_submit_ns": 21_050_000,
        "transfer_enqueue_ns": 21_100_000,
        "transfer_start_ns": 21_200_000,
        "transfer_copy_done_ns": 21_800_000,
        "transfer_stage_done_ns": 21_800_000,
        "publish_done_ns": 22_000_000,
        "receive_metadata_ns": 22_100_000,
        "receive_setup_done_ns": 22_200_000,
        "receive_transfer_done_ns": 22_300_000,
        "receive_materialize_start_ns": 22_300_000,
        "receive_materialize_done_ns": 22_400_000,
        "receive_embedding_ns": 22_500_000,
        "receive_done_ns": 23_000_000,
        "receive_concat_start_ns": 23_050_000,
        "receive_concat_done_ns": 23_100_000,
        "receive_extra_meta_start_ns": 23_150_000,
        "receive_extra_meta_done_ns": 23_200_000,
        "receive_result_ready_ns": 23_250_000,
        "language_apply_start_ns": 23_300_000,
        "language_get_mm_data_done_ns": 23_500_000,
        "language_radix_done_ns": 23_800_000,
        "language_ready_ns": 24_000_000,
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

    assert "ENCODER-PREPROCESS-TIME req_id=request-0" in caplog.text
    assert "dispatch_ms=1.000" in caplog.text
    assert "image_load_ms=2.000" in caplog.text
    assert "processor_queue_ms=3.000" in caplog.text
    assert "processor_ms=5.000" in caplog.text
    assert "request_total_ms=13.000" in caplog.text
