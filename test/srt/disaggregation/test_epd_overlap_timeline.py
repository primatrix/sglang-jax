import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).parents[3] / "scripts" / "disaggregation" / "trace_to_overlap_html.py"
_SPEC = importlib.util.spec_from_file_location("trace_to_overlap_html", _SCRIPT)
overlap = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = overlap
_SPEC.loader.exec_module(overlap)


def test_overlap_analysis_uses_union_time_without_double_counting():
    request_spans = [
        overlap.Span(10, 20, "encoder", "encoder", "R1"),
        overlap.Span(12, 18, "encoder", "encoder", "R2"),
    ]
    transfers = [overlap.Span(20, 30, "transfer", "transfer", "encoder_0")]
    devices = [
        overlap.Span(15, 25, "prefill", "prefill", "language device"),
        overlap.Span(30, 40, "decode", "decode", "language device"),
    ]

    result = overlap._analyse(request_spans, transfers, devices)

    assert result["stage_time_ms"] == {
        "encoder": 1e-05,
        "transfer": 1e-05,
        "language_prefill": 1e-05,
        "language_decode": 1e-05,
        "language_total": 2e-05,
    }
    assert result["overlap_ms"] == {
        "encoder_language": 5e-06,
        "transfer_language": 5e-06,
        "upstream_language": 1e-05,
        "any_cross_stage": 1e-05,
    }
    assert result["coverage_pct"] == {
        "encoder_hidden_by_language": 50.0,
        "transfer_hidden_by_language": 50.0,
        "language_overlapped_by_upstream": 50.0,
    }


def test_overlap_parser_keeps_actual_transfer_separate_from_pickup_wait(tmp_path):
    encoder_log = tmp_path / "encoder_0.log"
    encoder_log.write_text(
        "ENCODER-RAIDEN-INFLIGHT time_ns=100 event=start transfer_id=req-0 "
        "group_size=1 inflight_groups=1 inflight_requests=1\n"
        "ENCODER-RAIDEN-INFLIGHT time_ns=140 event=sent transfer_id=req-0 "
        "group_size=1 inflight_groups=0 inflight_requests=0\n"
    )

    spans = overlap._parse_transfers([encoder_log], 0, 1_000)

    assert spans == [
        overlap.Span(
            100,
            140,
            "transfer",
            "transfer",
            "encoder_0 ch1",
            "req-0 · group=1",
        )
    ]
