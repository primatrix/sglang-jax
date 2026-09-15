"""HTTP response contract tests for the standalone session smoke client."""

import importlib.util
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest


@pytest.fixture
def client():
    path = Path(__file__).parents[1] / "manual/deepseek_v4_session_smoke.py"
    spec = importlib.util.spec_from_file_location("session_smoke_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "path,body,expected",
    [
        ("/close_session", b"", None),
        ("/close_session", b'{"closed": true}', {"closed": True}),
        ("/generate", b'{"output_ids": [1, 2]}', {"output_ids": [1, 2]}),
    ],
)
def test_successful_responses(client, monkeypatch, path, body, expected):
    def urlopen(request, timeout):
        assert request.full_url == "http://localhost:30000" + path
        assert request.get_method() == "POST"
        assert json.loads(request.data) == {"session_id": "test"}
        assert timeout == 1800
        return io.BytesIO(body)

    monkeypatch.setattr(client, "urlopen", urlopen)
    assert client.post("http://localhost:30000/", path, {"session_id": "test"}) == expected


@pytest.mark.parametrize(
    "path,body",
    [("/generate", b""), ("/generate", b"invalid"), ("/close_session", b"invalid")],
)
def test_invalid_json_is_not_silently_accepted(client, monkeypatch, path, body):
    monkeypatch.setattr(client, "urlopen", lambda *args, **kwargs: io.BytesIO(body))
    with pytest.raises(json.JSONDecodeError):
        client.post("http://localhost:30000", path, {})


def test_http_error_still_propagates(client, monkeypatch):
    error = HTTPError("http://localhost:30000/close_session", 400, "Bad Request", {}, None)

    def urlopen(*args, **kwargs):
        raise error

    monkeypatch.setattr(client, "urlopen", urlopen)
    with pytest.raises(HTTPError) as caught:
        client.post("http://localhost:30000", "/close_session", {})
    assert caught.value is error


def test_empty_close_does_not_stop_remaining_prefix_cases(client, monkeypatch, capsys):
    import sys

    sessions = {}
    closed = []

    def urlopen(request, timeout):
        payload = json.loads(request.data)
        if request.full_url.endswith("/close_session"):
            closed.append(payload["session_id"])
            return io.BytesIO(b"")
        sid = payload.get("session_params", {}).get("id")
        cached = 0
        if sid:
            sessions[sid] = sessions.get(sid, 0) + 1
            if sessions[sid] == 2:
                cached = len(payload["input_ids"]) - 64
        result = {
            "output_ids": [7] * 4,
            "meta_info": {"finish_reason": {"type": "length"}, "cached_tokens": cached},
        }
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(client, "urlopen", urlopen)
    monkeypatch.setattr(sys, "argv", ["smoke", "--prefix-lengths", "127", "128"])
    client.main()
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["prefix_length"] for row in rows] == [127, 128]
    assert all(row["passed"] for row in rows)
    assert len(closed) == 2 and set(closed) == set(sessions)
