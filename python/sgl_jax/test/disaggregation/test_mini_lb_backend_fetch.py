from __future__ import annotations

import asyncio

from sgl_jax.srt.disaggregation import mini_lb


class _FakeResponse:
    def __init__(self, status: int, payload=None, text: str = ""):
        self.status = status
        self._payload = payload
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


class _FakeSession:
    def __init__(self, responses: dict[str, _FakeResponse]):
        self.responses = responses
        self.urls: list[str] = []

    def get(self, url: str):
        self.urls.append(url)
        return self.responses[url]


def test_fetch_backend_json_prefers_get_alias():
    session = _FakeSession(
        {
            "http://p/get_server_info": _FakeResponse(
                200,
                payload={"internal_states": [{"last_gen_throughput": 1.0}]},
            ),
        }
    )

    result = asyncio.run(
        mini_lb.fetch_backend_json(
            session,
            "http://p",
            ("get_server_info", "server_info"),
        )
    )

    assert result == {"internal_states": [{"last_gen_throughput": 1.0}]}
    assert session.urls == ["http://p/get_server_info"]


def test_fetch_backend_json_falls_back_to_legacy_alias():
    session = _FakeSession(
        {
            "http://p/get_model_info": _FakeResponse(404, text="Not Found"),
            "http://p/model_info": _FakeResponse(
                200,
                payload={"model_path": "Qwen3-8B"},
            ),
        }
    )

    result = asyncio.run(
        mini_lb.fetch_backend_json(
            session,
            "http://p",
            ("get_model_info", "model_info"),
        )
    )

    assert result == {"model_path": "Qwen3-8B"}
    assert session.urls == [
        "http://p/get_model_info",
        "http://p/model_info",
    ]
