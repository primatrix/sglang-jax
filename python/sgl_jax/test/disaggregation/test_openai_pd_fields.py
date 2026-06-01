from __future__ import annotations

from types import SimpleNamespace

from sgl_jax.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from sgl_jax.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sgl_jax.srt.entrypoints.openai.serving_completions import (
    OpenAIServingCompletion,
)


class _DummyTemplateManager:
    completion_template_name = None


def test_completion_request_keeps_pd_fields():
    serving = OpenAIServingCompletion(
        tokenizer_manager=object(),
        template_manager=_DummyTemplateManager(),
    )
    request = CompletionRequest(
        model="dummy",
        prompt="hello",
        bootstrap_host="10.0.0.1",
        bootstrap_port=8998,
        bootstrap_room=123,
        rid="req-1",
        disagg_transfer_id="xfer-1",
    )

    adapted_request, _ = serving._convert_to_internal_request(request)

    assert adapted_request.rid == "req-1"
    assert adapted_request.disagg_transfer_id == "xfer-1"
    assert adapted_request.bootstrap_host == "10.0.0.1"
    assert adapted_request.bootstrap_port == 8998
    assert adapted_request.bootstrap_room == 123


def test_chat_request_keeps_pd_fields(monkeypatch):
    tokenizer_manager = SimpleNamespace(
        model_config=None,
        server_args=SimpleNamespace(multimodal=False),
    )
    serving = OpenAIServingChat(
        tokenizer_manager=tokenizer_manager,
        template_manager=object(),
    )
    monkeypatch.setattr(
        serving,
        "_process_messages",
        lambda request, is_multimodal: SimpleNamespace(
            prompt="hello",
            prompt_ids=[1, 2, 3],
            image_data=None,
            video_data=None,
            audio_data=None,
            stop=[],
            tool_call_constraint=None,
        ),
    )
    request = ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "hello"}],
        bootstrap_host="10.0.0.1",
        bootstrap_port=8998,
        bootstrap_room=123,
        rid="req-2",
        disagg_transfer_id="xfer-2",
    )

    adapted_request, _ = serving._convert_to_internal_request(request)

    assert adapted_request.rid == "req-2"
    assert adapted_request.disagg_transfer_id == "xfer-2"
    assert adapted_request.bootstrap_host == "10.0.0.1"
    assert adapted_request.bootstrap_port == 8998
    assert adapted_request.bootstrap_room == 123
