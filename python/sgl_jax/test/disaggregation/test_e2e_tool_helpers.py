from __future__ import annotations

import json
from pathlib import Path

from sgl_jax.srt.disaggregation.tools import pair_stress as pair_stress
from sgl_jax.srt.disaggregation.tools.e2e import _common as C


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_fire_pd_request_sends_pd_payload(monkeypatch):
    captured = {}

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(C.httpx, "post", _fake_post)

    out = C.fire_pd_request(
        "http://p",
        rid="req-1",
        disagg_transfer_id="xfer-1",
        prompt="hello",
        bootstrap_host="10.0.0.1",
        bootstrap_port=8998,
        bootstrap_room=7,
        max_new_tokens=16,
        temperature=0.3,
        timeout=12.0,
    )

    assert out == {"ok": True}
    assert captured["url"] == "http://p/generate"
    assert captured["timeout"] == 12.0
    assert captured["json"] == {
        "rid": "req-1",
        "text": "hello",
        "sampling_params": {
            "max_new_tokens": 16,
            "temperature": 0.3,
        },
        "bootstrap_host": "10.0.0.1",
        "bootstrap_port": 8998,
        "bootstrap_room": 7,
        "disagg_transfer_id": "xfer-1",
    }


def test_fire_pd_pair_reuses_one_transfer_id(monkeypatch):
    calls = []

    def _fake_fire_pd_request(url, **kwargs):
        calls.append((url, kwargs))
        return {"url": url}

    monkeypatch.setattr(C, "fire_pd_request", _fake_fire_pd_request)

    topo = C.Topology(
        p_urls=["http://p0"],
        d_urls=["http://d0"],
        bootstrap_url="http://bootstrap:8998",
    )
    monkeypatch.setattr(topo, "pick_p_for_room", lambda room: "http://p-picked")

    out = C.fire_pd_pair(
        topo,
        rid="req-2",
        prompt="world",
        bootstrap_room=11,
        max_new_tokens=4,
        timeout=9.0,
    )

    assert out == {
        "P": {"url": "http://p-picked"},
        "D": {"url": "http://d0"},
    }
    assert len(calls) == 2
    transfer_ids = {kwargs["disagg_transfer_id"] for _, kwargs in calls}
    assert len(transfer_ids) == 1
    assert [url for url, _ in calls] == ["http://p-picked", "http://d0"]


def test_pair_stress_fire_one_accepts_prefill_only_contract(monkeypatch):
    def _fake_fire_pd_pair(*args, **kwargs):
        return {
            "P": {
                "text": "",
                "output_ids": [],
                "meta_info": {
                    "completion_tokens": 0,
                    "finish_reason": {"type": "length", "length": 0},
                },
            },
            "D": {
                "text": "hello",
                "output_ids": [1, 2],
                "meta_info": {
                    "completion_tokens": 2,
                    "finish_reason": {"type": "length", "length": 2},
                },
            },
        }

    monkeypatch.setattr(pair_stress.C, "fire_pd_pair", _fake_fire_pd_pair)
    topo = C.Topology(
        p_urls=["http://p0"],
        d_urls=["http://d0"],
        bootstrap_url="http://bootstrap:8998",
    )

    row = pair_stress._fire_one(
        topo,
        room=3,
        rid="stress-1",
        prompt="hello",
        max_new=2,
        use_explicit_endpoints=False,
    )

    assert row["ok"] is True
    assert row["error"] is None
    assert row["elapsed"] >= 0.0


def test_write_report_writes_json_and_print_result_codes(tmp_path):
    args = type("Args", (), {"out": str(tmp_path / "report.json")})()
    summary = {"ok": True, "count": 3}
    path = C.write_report(args, "ignored", summary)
    assert path == str(tmp_path / "report.json")
    assert json.loads(Path(path).read_text()) == summary
    assert C.print_result(True, "good") == 0
    assert C.print_result(False, "bad") == 1
