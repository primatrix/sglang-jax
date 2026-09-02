from sgl_jax.srt.disaggregation.encoder.metrics import (
    summarize_raiden_transfer_inflight,
)


def test_summarize_encoder_transfer_inflight_uses_formal_window(tmp_path):
    log_path = tmp_path / "encoder_0.log"
    log_path.write_text(
        "\n".join(
            [
                "ENCODER-RAIDEN-INFLIGHT time_ns=100 event=start transfer_id=a "
                "group_size=4 inflight_groups=1 inflight_requests=4",
                "ENCODER-RAIDEN-INFLIGHT time_ns=200 event=start transfer_id=b "
                "group_size=4 inflight_groups=2 inflight_requests=8",
                "ENCODER-RAIDEN-INFLIGHT time_ns=400 event=sent transfer_id=a "
                "group_size=4 inflight_groups=1 inflight_requests=4",
                "ENCODER-RAIDEN-INFLIGHT time_ns=600 event=sent transfer_id=b "
                "group_size=4 inflight_groups=0 inflight_requests=0",
            ]
        )
        + "\n"
    )

    summary = summarize_raiden_transfer_inflight(
        [log_path],
        start_ns=150,
        end_ns=650,
    )

    assert summary["available"] is True
    assert summary["n_events"] == 3
    assert summary["starts"] == 1
    assert summary["completions"] == 2
    assert summary["failures"] == 0
    assert summary["mean_groups"] == 1.3
    assert summary["mean_requests"] == 5.2
    assert summary["peak_groups"] == 2
    assert summary["peak_requests"] == 8
    assert summary["busy_fraction"] == 0.9
    assert summary["time_fraction_by_groups"] == {"0": 0.1, "1": 0.5, "2": 0.4}
    assert summary["active_window"] == {
        "start_ns": 200,
        "end_ns": 600,
        "duration_s": 4e-7,
        "n_events": 3,
        "starts": 1,
        "completions": 2,
        "failures": 0,
        "mean_groups": 1.5,
        "mean_requests": 6.0,
        "peak_groups": 2,
        "peak_requests": 8,
        "busy_fraction": 1.0,
        "busy_mean_groups": 1.5,
        "busy_mean_requests": 6.0,
        "time_fraction_by_groups": {"1": 0.5, "2": 0.5},
    }
