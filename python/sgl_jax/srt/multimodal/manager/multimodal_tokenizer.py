import asyncio
import base64
import dataclasses
import logging
import os
import signal
import time
import uuid
from http import HTTPStatus
from typing import Any

import fastapi
import numpy as np
import psutil
import setproctitle
from transformers import AutoImageProcessor

from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager
from sgl_jax.srt.multimodal.manager.io_struct import (
    AudioDecodeRequest,
    AudioDecodeResponse,
    AudioEncodeRequest,
    AudioEncodeResponse,
    AudioGenerationRequest,
    AudioGenerationResponse,
    DataType,
    GenerateMMReqInput,
    TokenizedGenerateMMReqInput,
)
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import (
    configure_logger,
    dataclass_to_string_truncated,
    kill_itself_when_parent_died,
)
from sgl_jax.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MMReqState:
    """Store the state of a request."""

    rid: str
    out_list: list[dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: GenerateMMReqInput
    created_time: float


class MultimodalTokenizer(TokenizerManager):
    """Tokenization manager for multimodal requests.

    `MultimodalTokenizer` accepts high-level multimodal generation requests
    (`GenerateMMReqInput`), tokenizes text inputs (and prepares image
    references when supported), forwards tokenized requests to the
    scheduler pipeline, and waits for/streams back results. It tracks the
    state of outstanding requests via `MMReqState` and uses a
    `TypeBasedDispatcher` to handle results arriving from the pipeline.
    """

    def __init__(self, server_args, port_args):
        """Initialize tokenizer, processor and result dispatcher.

        Loads an image processor (best-effort), initializes an in-memory
        map `rid_to_state` to track request state objects, and prepares a
        result dispatcher that routes batches of outputs back to
        `_handle_batch_output`.
        """
        super().__init__(server_args, port_args)
        self.wait_timeout = int(os.environ.get("SGLANG_WAIT_TIMEOUT", "600"))
        # Use slow image processor to avoid torchvision dependency warning
        try:
            self.mm_processor = AutoImageProcessor.from_pretrained(
                server_args.model_path, use_fast=False
            )
        except Exception:
            logger.warning("Failed to load image processor from %s", server_args.model_path)

        # Initialize audio processor (MelSpectrumExtractor) for audio models
        self.audio_processor = None
        self._init_audio_processor(server_args.model_path)

        self.rid_to_state: dict[str, MMReqState] = {}
        self._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    (list),
                    self._handle_batch_output,
                ),
                (
                    AbortReq,
                    self._handle_abort_req,
                ),
            ]
        )

    def _init_audio_processor(self, model_path: str):
        """Initialize audio processor if the model is an audio model."""
        import json
        import os
        try:
            import huggingface_hub
            if os.path.isdir(model_path):
                config_path = os.path.join(model_path, "config.json")
            else:
                config_path = huggingface_hub.hf_hub_download(
                    model_path,
                    "config.json",
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                )
            with open(config_path, "r") as f:
                config = json.load(f)

            # Check if this is an audio model by looking for audio-specific config keys
            if "n_mels" in config and "sampling_rate" in config:
                from sgl_jax.srt.multimodal.models.mimo_audio.mimo_audio_tokenizer import MelSpectrumExtractor
                self.audio_processor = MelSpectrumExtractor(
                    sample_rate=config.get("sampling_rate", 24000),
                    n_fft=config.get("nfft", 960),
                    hop_length=config.get("hop_length", 240),
                    win_length=config.get("window_size", config.get("nfft", 960)),
                    n_mels=config.get("n_mels", 128),
                    f_min=config.get("fmin", 0),
                    f_max=config.get("fmax") or (config.get("sampling_rate", 24000) // 2),
                )
                logger.info("Initialized audio processor for model %s", model_path)
        except Exception as e:
            logger.warning("Failed to initialize audio processor from %s: %s", model_path, e)

    def _preprocess_audio_to_mel(self, audio_array: np.ndarray) -> tuple:
        """Convert raw audio waveform to mel spectrogram.

        Args:
            audio_array: Raw audio waveform as numpy array.

        Returns:
            Tuple of (mel_spectrogram, input_lengths).
        """
        import jax.numpy as jnp

        if self.audio_processor is None:
            raise ValueError("Audio processor not initialized. Cannot preprocess audio.")

        # Convert to JAX array for mel extraction
        audio = jnp.array(audio_array)
        if audio.ndim == 1:
            audio = audio[None, :]

        mels = self.audio_processor(audio)
        if mels.ndim == 2:
            mels = mels[None, :]
        # Transpose to [batch, time, n_mels]
        mels = jnp.transpose(mels, (0, 2, 1))
        input_lens = jnp.array([mels.shape[1]])

        return mels, input_lens

    def _handle_batch_output(self, reqs: list):
        """Handle a batch of outputs returned from the pipeline.

        Marks the corresponding `MMReqState` as finished, sets its event to
        wake any waiters, and stores a simple success meta record. If a
        result arrives for an unknown `rid` it logs a warning.
        """
        if len(reqs) > 0 and self.server_args.log_requests:
            logger.info("handle_batch_output %s, self.rid_to_state %s", reqs, self.rid_to_state)
        for req in reqs:
            if req.rid in self.rid_to_state:
                self.rid_to_state[req.rid].finished = True
                self.rid_to_state[req.rid].event.set()

                out_data = {"success": True, "meta_info": {}}
                if hasattr(req, "audio_mode") and req.audio_mode is not None:
                    if req.audio_mode == "encode" and req.output is not None:
                        out_data["codes"] = req.output.tolist() if hasattr(req.output, "tolist") else req.output
                    elif req.audio_mode == "decode" and req.output is not None:
                        out_data["audio_data"] = req.output

                self.rid_to_state[req.rid].out_list = [out_data]
            else:
                logger.warning(
                    "Received result for unknown request rid=%s. Known rids: %s",
                    req.rid,
                    list(self.rid_to_state.keys()),
                )

    def _handle_abort_req(self, recv_obj: AbortReq):
        """Handle an AbortReq returned from the scheduler.

        When a request is aborted (e.g., removed from the scheduler's queue
        before processing started), the scheduler sends an AbortReq back to
        notify the tokenizer. This method marks the request as finished with
        an abort status and wakes any waiting coroutines.
        """
        if recv_obj.rid not in self.rid_to_state:
            logger.warning(
                "Received abort for unknown request rid=%s. Known rids: %s",
                recv_obj.rid,
                list(self.rid_to_state.keys()),
            )
            return

        state = self.rid_to_state[recv_obj.rid]
        state.finished = True
        state.out_list.append(
            {
                "success": False,
                "meta_info": {
                    "id": recv_obj.rid,
                    "finish_reason": {
                        "type": "abort",
                        "message": recv_obj.aborted_message or "Request aborted",
                        "status_code": HTTPStatus.BAD_REQUEST,
                    },
                },
            }
        )
        state.event.set()
        logger.info("Abort completed for rid=%s", recv_obj.rid)

    async def generate_request(
        self,
        obj: GenerateMMReqInput,
        request: fastapi.Request | None = None,
    ):
        """High level API: accept a generation request and stream responses.

        This coroutine tokenizes the input (text and optional image refs),
        sends the tokenized request to the scheduler pipeline, and then
        asynchronously yields results as they arrive (supporting streaming
        if `obj.stream` is True). It respects client disconnects and a
        configured wait timeout.
        """
        created_time = time.time()
        async with self._cond:
            await self._cond.wait_for(lambda: not self._updating)

        self.auto_create_handle_loop()

        if self.log_requests:
            max_length, skip_names, _ = self.log_request_metadata
            logger.info(
                "Receive: obj=%s",
                dataclass_to_string_truncated(obj, max_length, skip_names=skip_names),
            )

        tokenized_obj = await self._tokenize_one_request(obj)
        state = self._send_one_request(obj, tokenized_obj, created_time)
        async for response in self._wait_one_response(obj, state, request):
            yield response

    async def _tokenize_one_request(self, obj: GenerateMMReqInput):
        """
        Converts text fields to token ids using the configured tokenizer.
        Image preprocessing / references are noted as TODO; when provided
        `input_ids` are passed through unchanged.
        """
        # Support both 'prompt' (multimodal) and 'text' (text-only) fields
        input_text = getattr(obj, "prompt", None) or getattr(obj, "text", None)
        neg_input_text = getattr(obj, "neg_prompt", None) or getattr(obj, "text", None)
        input_ids = getattr(obj, "input_ids", None)
        neg_input_ids = getattr(obj, "neg_input_ids", None)
        if input_ids is None and input_text is not None:
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer is not initialized but input_text requires tokenization"
                )
            encoded = self.tokenizer(input_text)
            input_ids = encoded["input_ids"]
        if neg_input_ids is None and neg_input_text is not None:
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer is not initialized but neg_input_text requires tokenization"
                )
            encoded = self.tokenizer(neg_input_text)
            neg_input_ids = encoded["input_ids"]
        if getattr(obj, "input_reference", None) is not None:
            # TODO: Handle image preprocessing for multimodal inputs
            pass

        return self._create_tokenized_object(
            obj, input_text, input_ids, neg_input_text, neg_input_ids
        )

    def _create_tokenized_object(
        self, obj: GenerateMMReqInput, input_text, input_ids, neg_input_text, neg_input_ids
    ):
        """Build `TokenizedGenerateMMReqInput` from the original request.

        Ensures a request id (`rid`) exists, and copies over relevant
        properties such as size, num_frames, data type and save_output flag.
        """
        rid = getattr(obj, "rid", None)
        if rid is None:
            rid = uuid.uuid4().hex

        tokenized_obj = TokenizedGenerateMMReqInput(
            rid=rid,
            prompt=input_text,
            negative_prompt=neg_input_text,
            input_ids=input_ids,
            negative_input_ids=neg_input_ids,
            size=getattr(obj, "size", None),
            num_frames=getattr(obj, "num_frames", None),
            num_inference_steps=getattr(obj, "num_inference_steps", 50),
            data_type=getattr(obj, "data_type", None),
            save_output=getattr(obj, "save_output", True),
        )
        return tokenized_obj

    def _send_one_request(
        self,
        obj: GenerateMMReqInput,
        tokenized_obj: TokenizedGenerateMMReqInput,
        created_time: float | None = None,
    ):
        """Send a tokenized request into the scheduling pipeline and track it.

        Constructs an `MMReqState` to wait for results and stores it in
        `rid_to_state` keyed by the request id.
        """
        self.send_to_scheduler.send_pyobj(tokenized_obj)
        state = MMReqState(
            rid=tokenized_obj.rid,
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            created_time=created_time,
        )
        self.rid_to_state[tokenized_obj.rid] = state
        return state

    async def _wait_one_response(
        self,
        obj: GenerateMMReqInput,
        state: MMReqState,
        request: fastapi.Request | None = None,
    ):
        """Wait for results for a single request, yielding responses.

        This method waits on `state.event` with a timeout (`self.wait_timeout`),
        handles client disconnects (aborting the request), and yields
        intermediate/final outputs according to `obj.stream`.
        """
        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=self.wait_timeout)
            except TimeoutError:
                if request is not None and await request.is_disconnected():
                    self.abort_request(state.rid)
                    raise ValueError(
                        f"Request is disconnected from the client side. Abort request rid={state.rid}"
                    ) from None
                continue

            out = state.out_list[-1]

            state.out_list = []
            if state.finished:
                if self.log_requests:
                    max_length, skip_names, out_skip_names = self.log_request_metadata
                    msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}, out={dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)}"
                    logger.info(msg)

                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    finish_reason = out["meta_info"]["finish_reason"]
                    if (
                        finish_reason.get("type") == "abort"
                        and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST
                    ):
                        raise ValueError(finish_reason["message"])

                yield out
                break

            state.event.clear()

            if obj.stream:
                yield out
            else:
                if request is not None and await request.is_disconnected():
                    self.abort_request(state.rid)
                    raise ValueError(
                        f"Request is disconnected from the client side. Abort request rid={state.rid}"
                    )

    async def encode_audio(
        self,
        obj: AudioEncodeRequest,
        request: fastapi.Request | None = None,
    ):
        """Encode audio data to codes using the audio tokenizer.

        Args:
            obj: AudioEncodeRequest containing base64 encoded audio data.
            request: FastAPI request object for disconnect handling.

        Returns:
            AudioEncodeResponse containing the encoded codes.
        """
        created_time = time.time()
        async with self._cond:
            await self._cond.wait_for(lambda: not self._updating)

        self.auto_create_handle_loop()

        rid = uuid.uuid4().hex

        if obj.audio_data:
            audio_bytes = base64.b64decode(obj.audio_data)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        else:
            audio_array = np.zeros((24000,), dtype=np.float32)

        # Preprocess audio to mel spectrogram in tokenizer
        mel_input, mel_input_lens = self._preprocess_audio_to_mel(audio_array)

        from sgl_jax.srt.multimodal.manager.schedule_batch import Req

        audio_req = Req(
            rid=rid,
            mel_input=mel_input,
            mel_input_lens=mel_input_lens,
            audio_mode="encode",
            use_quantizer=obj.use_quantizer,
            n_q=obj.n_q,
            sample_rate=obj.sample_rate,
            data_type=DataType.AUDIO,
        )

        state = MMReqState(
            rid=rid,
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            created_time=created_time,
        )
        self.rid_to_state[rid] = state

        self.send_to_scheduler.send_pyobj(audio_req)

        try:
            await asyncio.wait_for(state.event.wait(), timeout=self.wait_timeout)
        except TimeoutError:
            raise ValueError(f"Audio encode request timed out for rid={rid}") from None

        del self.rid_to_state[rid]

        out = state.out_list[-1] if state.out_list else {"success": True, "meta_info": {}}

        return AudioEncodeResponse(
            id=rid,
            codes=out.get("codes"),
            hidden_states_shape=out.get("hidden_states_shape"),
        )

    async def decode_audio(
        self,
        obj: AudioDecodeRequest,
        request: fastapi.Request | None = None,
    ):
        """Decode codes to audio data using the audio tokenizer.

        Args:
            obj: AudioDecodeRequest containing the codes to decode.
            request: FastAPI request object for disconnect handling.

        Returns:
            AudioDecodeResponse containing base64 encoded audio data.
        """
        created_time = time.time()
        async with self._cond:
            await self._cond.wait_for(lambda: not self._updating)

        self.auto_create_handle_loop()

        rid = uuid.uuid4().hex

        codes_array = np.array(obj.codes, dtype=np.int32)

        from sgl_jax.srt.multimodal.manager.schedule_batch import Req

        audio_req = Req(
            rid=rid,
            codes=codes_array,
            audio_mode="decode",
            data_type=DataType.AUDIO,
        )

        state = MMReqState(
            rid=rid,
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            created_time=created_time,
        )
        self.rid_to_state[rid] = state

        self.send_to_scheduler.send_pyobj(audio_req)

        try:
            await asyncio.wait_for(state.event.wait(), timeout=self.wait_timeout)
        except TimeoutError:
            raise ValueError(f"Audio decode request timed out for rid={rid}") from None

        del self.rid_to_state[rid]

        out = state.out_list[-1] if state.out_list else {"success": True, "meta_info": {}}

        audio_data_b64 = None
        if out.get("audio_data") is not None:
            audio_bytes = out["audio_data"].astype(np.float32).tobytes()
            audio_data_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return AudioDecodeResponse(
            id=rid,
            audio_data=audio_data_b64,
            sample_rate=24000,
        )

    async def generate_audio(
        self,
        obj: AudioGenerationRequest,
        request: fastapi.Request | None = None,
    ):
        created_time = time.time()
        async with self._cond:
            await self._cond.wait_for(lambda: not self._updating)

        self.auto_create_handle_loop()

        rid = uuid.uuid4().hex

        if obj.audio_data:
            audio_bytes = base64.b64decode(obj.audio_data)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        else:
            audio_array = np.zeros((24000,), dtype=np.float32)

        # Preprocess audio to mel spectrogram in tokenizer
        mel_input, mel_input_lens = self._preprocess_audio_to_mel(audio_array)

        input_ids = None
        if obj.prompt and self.tokenizer is not None:
            encoded = self.tokenizer(obj.prompt)
            input_ids = encoded["input_ids"]

        from sgl_jax.srt.multimodal.manager.schedule_batch import Req

        audio_req = Req(
            rid=rid,
            mel_input=mel_input,
            mel_input_lens=mel_input_lens,
            audio_mode="generation",
            sample_rate=obj.sample_rate,
            data_type=DataType.AUDIO,
            save_output=obj.save_output,
            prompt=obj.prompt,
            input_ids=input_ids,
            n_q=8,
        )

        state = MMReqState(
            rid=rid,
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            created_time=created_time,
        )
        self.rid_to_state[rid] = state

        self.send_to_scheduler.send_pyobj(audio_req)

        try:
            await asyncio.wait_for(state.event.wait(), timeout=self.wait_timeout)
        except TimeoutError:
            raise ValueError(f"Audio generation request timed out for rid={rid}") from None

        del self.rid_to_state[rid]

        out = state.out_list[-1] if state.out_list else {"success": True, "meta_info": {}}

        audio_data_b64 = None
        if out.get("audio_data") is not None:
            audio_bytes = out["audio_data"].astype(np.float32).tobytes()
            audio_data_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return AudioGenerationResponse(
            id=rid,
            audio_data=audio_data_b64,
            sample_rate=obj.sample_rate,
        )


def run_multimodal_tokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang-jax::multimodal_tokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        tokenizer = MultimodalTokenizer(server_args, port_args)
        tokenizer.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("MultimodalTokenizerManager hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGQUIT)

    return tokenizer
