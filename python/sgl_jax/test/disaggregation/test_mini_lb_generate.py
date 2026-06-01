from __future__ import annotations

from fastapi.responses import ORJSONResponse
from fastapi.testclient import TestClient

from sgl_jax.srt.disaggregation import mini_lb


class _DummyLB:
    def __init__(self):
        self.prefill_urls = ["http://127.0.0.1:30100"]
        self.decode_urls = ["http://127.0.0.1:30200"]
        self.prefill_bootstrap_host = None
        self.calls = []

    def select_pair(self):
        return (
            "http://127.0.0.1:30100",
            8998,
            "http://127.0.0.1:30200",
        )

    async def generate(
        self,
        modified_request: dict,
        prefill_server: str,
        decode_server: str,
        endpoint: str,
    ):
        self.calls.append(
            (
                modified_request,
                prefill_server,
                decode_server,
                endpoint,
            )
        )
        return ORJSONResponse(content={"ok": True}, status_code=200)

    async def generate_stream(
        self,
        modified_request: dict,
        prefill_server: str,
        decode_server: str,
        endpoint: str,
    ):
        self.calls.append(
            (
                modified_request,
                prefill_server,
                decode_server,
                endpoint,
                "stream",
            )
        )
        return ORJSONResponse(content={"stream": True}, status_code=200)


def test_generate_accepts_batch_text_requests(monkeypatch):
    dummy = _DummyLB()
    monkeypatch.setattr(mini_lb, "lb", dummy)

    with TestClient(mini_lb.app) as client:
        response = client.post(
            "/generate",
            json={
                "text": ["hello", "world"],
                "sampling_params": {"max_new_tokens": 4},
            },
        )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    modified_request, prefill_server, decode_server, endpoint = dummy.calls[0]
    assert prefill_server == "http://127.0.0.1:30100"
    assert decode_server == "http://127.0.0.1:30200"
    assert endpoint == "generate"
    assert modified_request["text"] == ["hello", "world"]
    assert modified_request["bootstrap_host"] == ["127.0.0.1", "127.0.0.1"]
    assert modified_request["bootstrap_port"] == [8998, 8998]
    assert len(modified_request["bootstrap_room"]) == 2


def test_generate_accepts_input_ids_requests(monkeypatch):
    dummy = _DummyLB()
    monkeypatch.setattr(mini_lb, "lb", dummy)

    with TestClient(mini_lb.app) as client:
        response = client.post(
            "/generate",
            json={
                "input_ids": [1, 2, 3],
                "sampling_params": {"max_new_tokens": 2},
            },
        )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    modified_request, _, _, endpoint = dummy.calls[0]
    assert endpoint == "generate"
    assert modified_request["input_ids"] == [1, 2, 3]
    assert modified_request["bootstrap_host"] == "127.0.0.1"
    assert modified_request["bootstrap_port"] == 8998
    assert isinstance(modified_request["bootstrap_room"], int)


def test_generate_injects_shared_identity_when_client_omits_rid(monkeypatch):
    dummy = _DummyLB()
    monkeypatch.setattr(mini_lb, "lb", dummy)

    with TestClient(mini_lb.app) as client:
        response = client.post(
            "/generate",
            json={
                "text": "hello world",
                "sampling_params": {"max_new_tokens": 2},
            },
        )

    assert response.status_code == 200
    modified_request, _, _, _ = dummy.calls[0]
    assert isinstance(modified_request["rid"], str)
    assert modified_request["rid"]
    assert modified_request["disagg_transfer_id"] == modified_request["rid"]
