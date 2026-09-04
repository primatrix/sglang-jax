from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.profiler
import numpy as np
import uvicorn
import zmq.asyncio
from fastapi import FastAPI
from fastapi.responses import Response
from zmq.constants import LINGER, PUSH

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.disaggregation.encoder.bootstrap import EncoderBootstrapClient
from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.raiden_transfer import (
    RaidenEncoderServerTransfer,
)
from sgl_jax.srt.disaggregation.encoder.runtime import EncoderRuntime
from sgl_jax.srt.disaggregation.encoder.scheduler import DisaggEncoderScheduler
from sgl_jax.srt.disaggregation.encoder.sim_transfer import SimEncoderServerTransfer
from sgl_jax.srt.disaggregation.encoder.transfer_layout import PackedEmbeddingSlice
from sgl_jax.srt.disaggregation.host_ip import resolve_host_ip
from sgl_jax.srt.hf_transformers_utils import (
    get_processor,
    get_tokenizer_from_processor,
)
from sgl_jax.srt.model_loader import get_model
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalInputs
from sgl_jax.srt.multimodal.manager.multimodal_processor import (
    get_mm_processor,
    import_processors,
)
from sgl_jax.srt.multimodal.tokenizer_utils import resolve_tokenizer_subdir
from sgl_jax.srt.server_args import ServerArgs, apply_multimodal_model_defaults
from sgl_jax.srt.utils import configure_logger, set_uvicorn_logging_configs
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

# Adapted for JAX from SGLang's encoder server:
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/encoder/server.py


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreparedEncoderBatch:
    modality: Modality
    inputs: list[MultimodalInputs]
    request_timings: list[dict[str, int]]
    done_ns: int


class MMEncoder:
    """Run the model's native multimodal encoder for EPD requests."""

    _GRID_KEYS = {
        Modality.IMAGE: "image_grid_thw",
        Modality.VIDEO: "video_grid_thw",
    }
    _TOKEN_ID_KEYS = {
        Modality.IMAGE: "image_token_id",
        Modality.VIDEO: "video_token_id",
    }

    def __init__(self, server_args: ServerArgs) -> None:
        self.model_config = ModelConfig.from_server_args(server_args)
        apply_multimodal_model_defaults(server_args, self.model_config)
        if not self.model_config.is_multimodal:
            raise ValueError("--encoder-only requires an in-model multimodal architecture")

        # CPU simulation: skip the real vision encoder forward in ``encode``
        # and emit a modeled sleep + zero embedding of the correct shape/dtype.
        self._simulate_compute = server_args.simulate_compute
        self._sim_encoder_base_ms = server_args.simulate_compute_encoder_base_ms
        self._sim_encoder_ms_per_token = server_args.simulate_compute_encoder_ms_per_token
        self._log_timing = server_args.enable_request_time_stats_logging
        self._max_batch_size = max(1, int(server_args.encoder_max_batch_size))
        self._packed_capacities: tuple[int, ...] = ()

        config = self.model_config.hf_config
        config.vision_encoder_parallel = server_args.vision_encoder_parallel
        config.precompile_vision_patch_paddings = server_args.precompile_vision_patch_paddings
        if self._simulate_compute:
            # ``encode()`` replaces get_feature() with a sleep + zeros, so the
            # vision tower is never executed — don't build it or allocate weights.
            self.model = None
        else:
            mesh = create_device_mesh(
                ici_parallelism=[
                    server_args.dp_size,
                    server_args.tp_size // server_args.dp_size,
                ],
                dcn_parallelism=[1, 1],
                device_indexes=server_args.device_indexes,
            )
            self.model = get_model(
                model_config=self.model_config,
                load_config=LoadConfig(
                    load_format=server_args.load_format,
                    download_dir=server_args.download_dir,
                ),
                mesh=mesh,
            )
            if not server_args.disable_precompile:
                logger.info("Precompiling multimodal encoder")
                self.model.precompile_multimodal()
            get_capacities = getattr(
                self.model,
                "get_multimodal_embedding_packed_capacities",
                None,
            )
            if get_capacities is not None:
                self._packed_capacities = tuple(map(int, get_capacities()))

        tokenizer_path = server_args.tokenizer_path
        tokenizer_subdir = resolve_tokenizer_subdir(server_args.model_path, tokenizer_path)
        if tokenizer_subdir:
            tokenizer_path = os.path.join(tokenizer_path, tokenizer_subdir)
        processor = get_processor(
            tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            use_fast=True,
        )
        import_processors("sgl_jax.srt.multimodal.processors")
        self.mm_processor = get_mm_processor(config, server_args, processor)
        self.tokenizer = get_tokenizer_from_processor(processor)

    async def preprocess(self, requests: list[dict[str, Any]]) -> PreparedEncoderBatch:
        if not requests:
            raise ValueError("MMEncoder batches must not be empty")
        modality = Modality.from_str(requests[0]["modality"])
        if any(Modality.from_str(request["modality"]) != modality for request in requests):
            raise ValueError("MMEncoder batches must contain one modality")

        prepared = await asyncio.gather(
            *(self._process_request(request, modality) for request in requests)
        )
        processed, timings = map(list, zip(*prepared))
        return PreparedEncoderBatch(modality, processed, timings, time.time_ns())

    def encode(self, batch: PreparedEncoderBatch) -> list[tuple[jax.Array, dict[str, Any]]]:
        modality = batch.modality
        processed = batch.inputs
        encode_start_ns = time.time_ns()
        simulate = getattr(self, "_simulate_compute", False)
        if simulate:
            token_count_total = sum(
                end - start
                for mm_inputs in processed
                for item in mm_inputs.mm_items
                for start, end in item.placeholder_ranges or ()
            )
            with jax.profiler.TraceAnnotation(f"mm_encode:{modality.name}:{len(processed)}"):
                sleep_ms = (
                    self._sim_encoder_base_ms + self._sim_encoder_ms_per_token * token_count_total
                )
                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000.0)
            packed = np.zeros(
                (token_count_total, self.model_config.hidden_size),
                dtype=self.model_config.dtype,
            )
        else:
            with jax.profiler.TraceAnnotation(f"mm_encode:{modality.name}:{len(processed)}"):
                items = [item for mm_inputs in processed for item in mm_inputs.mm_items]
                target = self.model.thinker if hasattr(self.model, "thinker") else self.model
                get_feature = getattr(target, f"get_{modality.name.lower()}_feature", None)
                if get_feature is None:
                    raise ValueError(f"model has no {modality.name} encoder")
                packed = get_feature(items)
        encode_done_ns = time.time_ns()
        postprocess_start_ns = time.perf_counter_ns()

        encoder_timing = {
            "preprocess_done_ns": batch.done_ns,
            "encode_start_ns": encode_start_ns,
            "encode_done_ns": encode_done_ns,
        }

        results = []
        token_count_duration_ns = 0
        embedding_slice_duration_ns = 0
        metadata_duration_ns = 0
        result_pack_duration_ns = 0

        phase_start_ns = time.perf_counter_ns()
        token_counts = tuple(
            sum(
                end - start
                for item in mm_inputs.mm_items
                for start, end in item.placeholder_ranges or ()
            )
            for mm_inputs in processed
        )
        token_count_duration_ns += time.perf_counter_ns() - phase_start_ns

        phase_start_ns = time.perf_counter_ns()
        if sum(token_counts) > packed.shape[0]:
            raise ValueError(f"incomplete {modality.name} encoder output")
        if simulate:
            offset = 0
            embeddings = []
            for token_count in token_counts:
                embeddings.append(jax.device_put(packed[offset : offset + token_count]))
                offset += token_count
        else:
            offset = 0
            max_batch_size = max(
                len(token_counts),
                getattr(self, "_max_batch_size", len(token_counts)),
            )
            packed_capacities = getattr(self, "_packed_capacities", ())
            embeddings = []
            for token_count in token_counts:
                embeddings.append(
                    PackedEmbeddingSlice(
                        packed=packed,
                        offset=offset,
                        rows=token_count,
                        max_batch_size=max_batch_size,
                        packed_capacities=packed_capacities,
                    )
                )
                offset += token_count
        if any(
            embedding.shape[0] != token_count
            for embedding, token_count in zip(embeddings, token_counts)
        ):
            raise ValueError(f"incomplete {modality.name} encoder output")
        embedding_slice_duration_ns += time.perf_counter_ns() - phase_start_ns

        for embedding, mm_inputs, request_timing in zip(
            embeddings,
            processed,
            batch.request_timings,
        ):
            phase_start_ns = time.perf_counter_ns()
            metadata = self._metadata(mm_inputs, modality)
            metadata_duration_ns += time.perf_counter_ns() - phase_start_ns

            phase_start_ns = time.perf_counter_ns()
            metadata["_encoder_timing"] = {**encoder_timing, **request_timing}
            results.append((embedding, metadata))
            result_pack_duration_ns += time.perf_counter_ns() - phase_start_ns

        postprocess_done_ns = time.time_ns()
        postprocess_duration_ns = time.perf_counter_ns() - postprocess_start_ns
        postprocess_residual_ns = max(
            0,
            postprocess_duration_ns
            - token_count_duration_ns
            - embedding_slice_duration_ns
            - metadata_duration_ns
            - result_pack_duration_ns,
        )
        postprocess_timing = {
            "encode_server_postprocess_done_ns": postprocess_done_ns,
            "encode_server_postprocess_duration_ns": postprocess_duration_ns,
            "encode_token_count_duration_ns": token_count_duration_ns,
            "encode_embedding_slice_duration_ns": embedding_slice_duration_ns,
            "encode_metadata_duration_ns": metadata_duration_ns,
            "encode_result_pack_duration_ns": result_pack_duration_ns,
            "encode_server_postprocess_residual_ns": postprocess_residual_ns,
        }
        for _, metadata in results:
            metadata["_encoder_timing"].update(postprocess_timing)
        # JAX keeps bucket padding in the encoder output to preserve static shapes.
        # Transfer only the placeholder-backed prefix, as upstream SGLang does.
        return results

    async def _process_request(
        self, request: dict[str, Any], modality: Modality
    ) -> tuple[MultimodalInputs, dict[str, int]]:
        timing = {}
        if getattr(self, "_log_timing", False):
            timing["preprocess_request_start_ns"] = time.time_ns()
        mm_items = request.get("mm_items") or []
        if not mm_items:
            raise ValueError("encoder request contains no multimodal items")
        request_obj = SimpleNamespace(
            image_data=mm_items if modality == Modality.IMAGE else None,
            video_data=mm_items if modality == Modality.VIDEO else None,
            audio_data=mm_items if modality == Modality.AUDIO else None,
            fps=request.get("fps"),
            num_frames=request.get("num_frames"),
        )
        mm_inputs = await self.mm_processor.process_encoder_mm_data_async(
            image_data=request_obj.image_data,
            input_text=self._placeholder(modality) * len(mm_items),
            request_obj=request_obj,
            encoder_timing=timing if timing else None,
        )
        items = [item for item in mm_inputs.mm_items if item.modality == modality]
        if len(items) != len(mm_items):
            raise ValueError(
                f"processor produced {len(items)} {modality.name} items for {len(mm_items)} inputs"
            )
        mm_inputs.mm_items = items
        if timing:
            timing["preprocess_request_done_ns"] = time.time_ns()
        return mm_inputs, timing

    def _placeholder(self, modality: Modality) -> str:
        config = self.mm_processor.hf_config
        token_id = getattr(config, self._TOKEN_ID_KEYS.get(modality, ""), None)
        if token_id is None:
            raise ValueError(f"model has no {modality.name} placeholder token")
        return "".join(
            self.tokenizer.convert_ids_to_tokens(
                [config.vision_start_token_id, token_id, config.vision_end_token_id]
            )
        )

    def _metadata(self, mm_inputs: MultimodalInputs, modality: Modality) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        grid_key = self._GRID_KEYS.get(modality)
        if grid_key is not None:
            metadata["grid_dim"] = np.concatenate(
                [np.asarray(item.get(grid_key)) for item in mm_inputs.mm_items], axis=0
            )
        if modality == Modality.VIDEO:
            timing = [item.get("second_per_grid_ts") for item in mm_inputs.mm_items]
            if all(value is not None for value in timing):
                metadata["second_per_grid_ts"] = timing
        return metadata

    def shutdown(self) -> None:
        self.mm_processor.shutdown()


class EncoderServer:
    def __init__(
        self,
        encoder: MMEncoder,
        transfer: Any,
        receiver_timeout: float | None = 300.0,
        encoder_register_urls: list[str] | None = None,
        advertise_url: str | None = None,
        bootstrap_timeout: float = 5.0,
        max_batch_size: int = 8,
        batch_coalesce_ms: float = 0.0,
        max_inflight_batches: int = 1,
        request_timeout: float | None = 300.0,
        network_rtt_ms: float = 0.0,
        log_queue_timing: bool = False,
    ) -> None:
        encoder_register_urls = list(encoder_register_urls or ())
        if bool(encoder_register_urls) != bool(advertise_url):
            raise ValueError("encoder_register_urls and advertise_url must be configured together")

        self._network_rtt_s = max(0.0, float(network_rtt_ms)) / 1000.0
        self.runtime = EncoderRuntime(
            encoder,
            transfer,
            pipeline_depth=max_inflight_batches,
        )
        self.scheduler = DisaggEncoderScheduler(
            self.runtime,
            max_batch_size=max_batch_size,
            batch_coalesce_ms=batch_coalesce_ms,
            max_inflight_batches=max_inflight_batches,
            request_timeout=request_timeout,
            log_queue_timing=log_queue_timing,
        )
        self._zmq = zmq.asyncio.Context.instance()
        self._receiver_timeout = receiver_timeout
        self._receiver_addresses: dict[str, str] = {}
        self._receiver_events: dict[str, asyncio.Event] = {}
        self._receiver_sockets: dict[str, zmq.asyncio.Socket] = {}
        self._notify_lock = asyncio.Lock()

        @asynccontextmanager
        async def lifespan(_: FastAPI):
            bootstrap_clients = [
                (
                    url,
                    EncoderBootstrapClient(url, timeout=bootstrap_timeout),
                )
                for url in encoder_register_urls
            ]
            registration_task = None
            if advertise_url is not None:
                registration_task = asyncio.create_task(
                    self._register_with_bootstraps(
                        bootstrap_clients,
                        advertise_url.rstrip("/"),
                    )
                )
            self.start()
            try:
                yield
            finally:
                try:
                    await self.stop()
                finally:
                    if registration_task is not None:
                        registration_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await registration_task
                    if advertise_url is not None:
                        await self._unregister_from_bootstraps(
                            bootstrap_clients,
                            advertise_url.rstrip("/"),
                        )

        self.app = FastAPI(openapi_url=None, lifespan=lifespan)
        self._trace_active = False
        self._trace_dir: str | None = None
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/encode", self.encode, methods=["POST"])
        self.app.add_api_route(
            "/scheduler_receive_url",
            self.register_scheduler_receiver,
            methods=["POST"],
        )
        self.app.add_api_route("/start_profile", self.start_profile, methods=["POST"])
        self.app.add_api_route("/stop_profile", self.stop_profile, methods=["POST"])
        self.app.add_api_route("/profile_status", self.profile_status, methods=["GET"])

    def start(self) -> None:
        self.runtime.start()
        self.scheduler.start()

    async def stop(self) -> None:
        try:
            await self.scheduler.stop()
        finally:
            try:
                await self.runtime.stop()
            finally:
                for socket in self._receiver_sockets.values():
                    socket.close()
                self._receiver_sockets.clear()
                self._receiver_events.clear()
                self._receiver_addresses.clear()

    @staticmethod
    async def _register_with_bootstraps(
        clients: list[tuple[str, EncoderBootstrapClient]],
        encoder_url: str,
    ) -> None:
        pending = list(clients)
        for attempt in range(30):
            results = await asyncio.gather(
                *(client.register(encoder_url) for _, client in pending),
                return_exceptions=True,
            )
            pending = [
                pair for pair, result in zip(pending, results) if isinstance(result, Exception)
            ]
            if not pending:
                return
            if attempt < 29:
                await asyncio.sleep(5)

        logger.error(
            "Encoder registration failed after 30 attempts: %s",
            [url for url, _ in pending],
        )

    @staticmethod
    async def _unregister_from_bootstraps(
        clients: list[tuple[str, EncoderBootstrapClient]],
        encoder_url: str,
    ) -> None:
        results = await asyncio.gather(
            *(client.unregister(encoder_url) for _, client in clients),
            return_exceptions=True,
        )
        for (url, _), result in zip(clients, results):
            if isinstance(result, Exception):
                logger.warning("Encoder unregister from %s failed: %s", url, result)
        await asyncio.gather(*(client.close() for _, client in clients))

    async def health(self) -> Response:
        return Response("OK")

    async def register_scheduler_receiver(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        req_id = request["req_id"]
        self._receiver_addresses[req_id] = request["receive_url"]
        self._receiver_events.setdefault(req_id, asyncio.Event()).set()
        return {"req_id": req_id}

    async def encode(self, request: dict[str, Any]) -> dict[str, Any]:
        # Model the language->encoder network hop (loopback has none).
        if self._network_rtt_s:
            await asyncio.sleep(self._network_rtt_s)
        try:
            data = await self.scheduler.submit(request)
        except Exception as exc:
            try:
                req_id = request["req_id"]
                await self.send_to_scheduler(
                    req_id,
                    EmbeddingData(
                        req_id=req_id,
                        num_parts=request.get("num_parts", 1),
                        part_idx=request.get("part_idx", 0),
                        grid_dim=None,
                        modality=Modality.from_str(request["modality"]),
                        error_msg=str(exc),
                    ),
                )
            except Exception:
                logger.exception(
                    "Encoder error delivery failed. req_id=%s",
                    request.get("req_id"),
                )
            raise

        try:
            await self.send_to_scheduler(data.req_id, data)
        except Exception:
            self.runtime.release(data.transfer_id)
            raise
        # The response is only an ACK. Metadata travels over ZMQ and the
        # embedding itself travels over the configured transfer backend.
        return {"req_id": request["req_id"]}

    async def send_to_scheduler(self, req_id: str, data: EmbeddingData) -> None:
        try:
            event = self._receiver_events.setdefault(req_id, asyncio.Event())
            if self._receiver_timeout is None or self._receiver_timeout <= 0:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), self._receiver_timeout)
            address = self._receiver_addresses[req_id]
            async with self._notify_lock:
                socket = self._receiver_sockets.get(address)
                if socket is None:
                    socket = self._zmq.socket(PUSH)
                    socket.setsockopt(LINGER, 1000)
                    socket.connect(f"tcp://{address}")
                    self._receiver_sockets[address] = socket
                await socket.send_pyobj(data)
        finally:
            self._receiver_events.pop(req_id, None)
            self._receiver_addresses.pop(req_id, None)

    async def start_profile(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        """Arm a jax.profiler trace on the encoder process.

        The encoder batch scheduler has no SchedulerProfilerMixin; this minimal
        endpoint lets the EPD driver capture the encoder tier alongside the
        language server's prefill/decode traces.
        """
        request = request or {}
        if self._trace_active:
            return {"status": "in_progress", "output_dir": self._trace_dir}
        base = request.get("output_dir") or os.path.join(
            os.getenv("SGLANG_JAX_PROFILER_DIR", "/tmp"), "encoder"
        )
        Path(base).mkdir(parents=True, exist_ok=True)
        options = jax.profiler.ProfileOptions()
        host_tracer_level = request.get("host_tracer_level")
        python_tracer_level = request.get("python_tracer_level")
        if host_tracer_level is not None:
            options.host_tracer_level = int(host_tracer_level)
        if python_tracer_level is not None:
            options.python_tracer_level = int(python_tracer_level)
        jax.profiler.start_trace(base, profiler_options=options)
        self._trace_active = True
        self._trace_dir = base
        logger.info("Encoder profiling started -> %s", base)
        return {"status": "in_progress", "output_dir": base}

    async def stop_profile(self) -> dict[str, Any]:
        if not self._trace_active:
            return {"status": "idle"}
        jax.profiler.stop_trace()
        self._trace_active = False
        logger.info("Encoder profiling stopped -> %s", self._trace_dir)
        return {"status": "idle", "output_dir": self._trace_dir}

    async def profile_status(self) -> dict[str, Any]:
        return {"status": "in_progress" if self._trace_active else "idle"}

    def run(self, host: str, port: int) -> None:
        uvicorn.run(self.app, host=host, port=port)


def launch(server_args: ServerArgs) -> None:
    configure_logger(server_args)
    set_uvicorn_logging_configs()
    encoder = MMEncoder(server_args)
    try:
        if server_args.simulate_compute:
            # Sim transfer needs no routable peer IP; bind/advertise on loopback.
            host_ip = server_args.disaggregation_host_ip or "127.0.0.1"
            transfer = SimEncoderServerTransfer(
                setup_ms=server_args.simulate_transfer_setup_ms,
                parallelism=server_args.disaggregation_channel_number,
                pool_size=server_args.encoder_transfer_pool_size,
                timeout_s=server_args.encoder_request_timeout_seconds,
                ms_per_mb=server_args.simulate_transfer_ms_per_mb,
                rtt_ms=server_args.simulate_network_rtt_ms,
                log_inflight=server_args.enable_request_time_stats_logging,
            )
        else:
            host_ip = resolve_host_ip(server_args.disaggregation_host_ip)
            transfer = RaidenEncoderServerTransfer(
                host_ip,
                parallelism=server_args.disaggregation_channel_number,
                pool_size=server_args.encoder_transfer_pool_size,
                timeout_s=server_args.encoder_request_timeout_seconds,
                log_inflight=server_args.enable_request_time_stats_logging,
            )
        advertise_host = f"[{host_ip}]" if ":" in host_ip else host_ip
        advertise_url = (
            f"http://{advertise_host}:{server_args.port}"
            if server_args.encoder_register_urls
            else None
        )
        control_timeout = server_args.encoder_control_timeout_seconds
        server = EncoderServer(
            encoder,
            transfer,
            receiver_timeout=server_args.encoder_request_timeout_seconds,
            encoder_register_urls=server_args.encoder_register_urls,
            advertise_url=advertise_url,
            bootstrap_timeout=control_timeout if control_timeout > 0 else 5.0,
            max_batch_size=server_args.encoder_max_batch_size,
            batch_coalesce_ms=server_args.encoder_batch_coalesce_ms,
            max_inflight_batches=server_args.encoder_max_inflight_batches,
            request_timeout=server_args.encoder_request_timeout_seconds,
            network_rtt_ms=(
                server_args.simulate_network_rtt_ms if server_args.simulate_compute else 0.0
            ),
            log_queue_timing=server_args.enable_request_time_stats_logging,
        )
        server.run(server_args.host, server_args.port)
    finally:
        encoder.shutdown()
