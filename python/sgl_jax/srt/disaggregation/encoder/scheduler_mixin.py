from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from sgl_jax.srt.disaggregation.encoder.client import (
    EncoderClient,
    PendingEncoderRequest,
    create_encoder_client,
)
from sgl_jax.srt.managers.io_struct import AbortReq, TokenizedGenerateReqInput
from sgl_jax.srt.multimodal.common.modality_enum import build_radix_input_ids

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
    from sgl_jax.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def _elapsed_ms(timing: dict[str, int], start: str, end: str) -> float:
    return max(0, timing[end] - timing[start]) / 1_000_000


def _duration_ms(timing: dict[str, int], field: str) -> float:
    return max(0, timing[field]) / 1_000_000


class SchedulerDisaggregationEncoderMixin:
    """Encoder-disaggregation request handling for the language scheduler."""

    encoder_client: EncoderClient | None
    encoder_waiting: dict[str, PendingEncoderRequest]

    def init_encoder_disaggregation(self: Scheduler) -> None:
        self.encoder_client = None
        self.encoder_waiting = {}
        if not self.server_args.language_only:
            return
        if self.nnodes > 1:
            raise RuntimeError("encoder disaggregation does not support multi-host schedulers yet")
        if self._mm_processor is None:
            raise ValueError("encoder disaggregation requires a multimodal processor")
        self.encoder_client = create_encoder_client(self.server_args, self.mesh)
        if self.encoder_client.background_progress:
            self.encoder_client.set_result_preparer(self._apply_encoder_result)

    def process_encoder_requests(
        self: Scheduler,
        recv_reqs: list,
    ) -> list:
        ready = []
        now = time.monotonic()

        for recv_req in recv_reqs:
            if not self._needs_encoder(recv_req):
                ready.append(recv_req)
                continue
            if recv_req.rid in self.encoder_waiting:
                continue

            try:
                pending = self.encoder_client.receive(recv_req)
            except Exception as exc:
                self._abort_encoder_request(recv_req, str(exc))
                continue

            self.encoder_waiting[recv_req.rid] = pending

        timeout = self.server_args.encoder_request_timeout_seconds
        log_poll_time = getattr(
            self.server_args,
            "enable_request_time_stats_logging",
            False,
        )
        encoder_client = getattr(self, "encoder_client", None)
        background_progress = bool(
            encoder_client is not None and encoder_client.background_progress
        )
        if background_progress:
            pending_items = [
                (pending.recv_req.rid, pending) for pending in encoder_client.drain_completed()
            ]
        else:
            pending_items = list(self.encoder_waiting.items())

        for rid, pending in pending_items:
            if self.encoder_waiting.get(rid) is not pending:
                continue
            recv_req = pending.recv_req

            result = None
            poll_error = None
            poll_start_ns = time.monotonic_ns()
            try:
                result = pending.poll()
            except Exception as exc:
                poll_error = exc
            poll_duration_ns = time.monotonic_ns() - poll_start_ns

            if log_poll_time:
                if poll_error is not None:
                    poll_status = "error"
                elif result is not None:
                    poll_status = "ready"
                else:
                    poll_status = "pending"
                logger.info(
                    "ENCODER-POLL-TIME req_id=%s duration_ns=%d duration_ms=%.3f status=%s",
                    rid,
                    poll_duration_ns,
                    poll_duration_ns / 1_000_000,
                    poll_status,
                )

            if poll_error is not None:
                self._abort_encoder_request(recv_req, str(poll_error))
                self._remove_encoder_waiting(rid)
                continue

            if result is not None:
                try:
                    pickup_ns = time.time_ns()
                    if getattr(pending, "prepared_in_background", False):
                        encoder_timing = getattr(recv_req, "encoder_timing", None)
                        if encoder_timing is not None:
                            encoder_timing["language_scheduler_pickup_ns"] = pickup_ns
                    else:
                        encoder_timing = result.get("encoder_timing")
                        if encoder_timing is not None:
                            encoder_timing["language_scheduler_pickup_ns"] = pickup_ns
                        self._apply_encoder_result(recv_req, result)
                except Exception as exc:
                    self._abort_encoder_request(recv_req, str(exc))
                else:
                    ready.append(recv_req)
                self._remove_encoder_waiting(rid)
                continue

            if timeout > 0 and now - pending.started_at >= timeout:
                self._abort_encoder_request(
                    recv_req,
                    f"encoder timed out after {timeout}s",
                )
                self._remove_encoder_waiting(rid)

        if background_progress and timeout > 0:
            for rid, pending in list(self.encoder_waiting.items()):
                if now - pending.started_at < timeout:
                    continue
                self._abort_encoder_request(
                    pending.recv_req,
                    f"encoder timed out after {timeout}s",
                )
                self._remove_encoder_waiting(rid)

        return ready

    def _apply_encoder_result(self: Scheduler, recv_req, result: dict) -> None:
        embeddings = result.get("embeddings")
        if embeddings is None:
            raise ValueError("encoder result contains no embeddings")
        encoder_timing = result.get("encoder_timing")
        if encoder_timing is not None:
            encoder_timing.setdefault("language_apply_start_ns", time.time_ns())
        prompt = recv_req.input_ids if recv_req.input_ids is not None else recv_req.text
        mm_inputs = self._mm_processor.get_mm_data(
            prompt,
            embeddings,
            **{
                key: value
                for key, value in result.items()
                if key not in ("embeddings", "encoder_timing")
            },
        )
        if encoder_timing is not None:
            encoder_timing["language_get_mm_data_done_ns"] = time.time_ns()
        recv_req.mm_inputs = mm_inputs
        recv_req.input_ids = mm_inputs.input_ids
        recv_req.radix_input_ids = mm_inputs.radix_input_ids or build_radix_input_ids(
            recv_req.input_ids, mm_inputs
        )
        if encoder_timing is not None:
            encoder_timing["language_radix_done_ns"] = time.time_ns()
        recv_req.need_wait_for_mm_inputs = False
        if encoder_timing:
            recv_req.encoder_timing = {
                **encoder_timing,
                "language_ready_ns": time.time_ns(),
            }

    @staticmethod
    def _needs_encoder(recv_req) -> bool:
        return isinstance(recv_req, TokenizedGenerateReqInput) and bool(
            recv_req.need_wait_for_mm_inputs
        )

    def _abort_encoder_request(self: Scheduler, recv_req, error_msg: str) -> None:
        logger.error("Encoder request failed. rid=%s error=%s", recv_req.rid, error_msg)
        output = AbortReq(rid=recv_req.rid, aborted_message=error_msg)
        if self._comm_backend is not None:
            self._comm_backend.send_pyobj(output)
        else:
            self.send_to_tokenizer.send_pyobj(output)

    def _remove_encoder_waiting(self: Scheduler, rid: str) -> None:
        pending = self.encoder_waiting.pop(rid)
        pending.close()

    def _cancel_encoder_requests(self: Scheduler, recv_req: AbortReq) -> None:
        for rid in list(self.encoder_waiting):
            if recv_req.abort_all or rid.startswith(recv_req.rid):
                self._remove_encoder_waiting(rid)
                output = AbortReq(rid=rid)
                if self._comm_backend is not None:
                    self._comm_backend.send_pyobj(output)
                else:
                    self.send_to_tokenizer.send_pyobj(output)

    def _mark_encoder_prefill_start(self: Scheduler, batch: ScheduleBatch) -> None:
        if (
            not getattr(self.server_args, "enable_request_time_stats_logging", False)
            or not batch.forward_mode.is_extend()
        ):
            return

        prefill_start_ns = time.time_ns()
        for info in batch.reqs_info:
            for req in info.reqs or ():
                timing = getattr(req, "encoder_timing", None)
                if not timing or "language_prefill_start_ns" in timing:
                    continue
                timing["language_prefill_start_ns"] = prefill_start_ns

    def _log_encoder_pipeline_timing(self: Scheduler, batch: ScheduleBatch) -> None:
        if (
            not getattr(self.server_args, "enable_request_time_stats_logging", False)
            or not batch.forward_mode.is_extend()
        ):
            return

        prefill_done_ns = time.time_ns()
        for info in batch.reqs_info:
            for req in info.reqs or ():
                timing = getattr(req, "encoder_timing", None)
                if not timing or "language_prefill_done_ns" in timing:
                    continue
                timing["language_prefill_done_ns"] = prefill_done_ns
                required = (
                    "enqueue_ns",
                    "dequeue_ns",
                    "preprocess_start_ns",
                    "preprocess_done_ns",
                    "encode_start_ns",
                    "encode_done_ns",
                    "encode_server_postprocess_done_ns",
                    "encode_server_postprocess_duration_ns",
                    "encode_token_count_duration_ns",
                    "encode_embedding_slice_duration_ns",
                    "encode_split_compile_wait_duration_ns",
                    "encode_split_dispatch_duration_ns",
                    "encode_metadata_duration_ns",
                    "encode_result_pack_duration_ns",
                    "encode_server_postprocess_residual_ns",
                    "runtime_encode_return_ns",
                    "runtime_postprocess_done_ns",
                    "runtime_postprocess_duration_ns",
                    "runtime_metadata_prepare_duration_ns",
                    "runtime_embedding_data_duration_ns",
                    "runtime_result_pack_duration_ns",
                    "runtime_postprocess_residual_ns",
                    "runtime_timing_attach_duration_ns",
                    "transfer_enqueue_ns",
                    "transfer_start_ns",
                    "transfer_reserve_start_ns",
                    "transfer_pool_ready_ns",
                    "transfer_reserve_done_ns",
                    "transfer_copy_start_ns",
                    "transfer_copy_submit_ns",
                    "transfer_copy_done_ns",
                    "transfer_register_start_ns",
                    "transfer_register_done_ns",
                    "transfer_stage_done_ns",
                    "publish_done_ns",
                    "receive_metadata_ns",
                    "receive_setup_done_ns",
                    "receive_transfer_done_ns",
                    "receive_materialize_start_ns",
                    "receive_materialize_done_ns",
                    "receive_embedding_ns",
                    "receive_done_ns",
                    "receive_concat_start_ns",
                    "receive_concat_done_ns",
                    "receive_extra_meta_start_ns",
                    "receive_extra_meta_done_ns",
                    "receive_result_ready_ns",
                    "language_apply_start_ns",
                    "language_get_mm_data_done_ns",
                    "language_radix_done_ns",
                    "language_ready_ns",
                    "language_scheduler_pickup_ns",
                    "language_prefill_start_ns",
                    "language_prefill_done_ns",
                )
                if not all(name in timing for name in required):
                    continue

                logger.info(
                    "ENCODER-PIPELINE-TIME req_id=%s enqueue_ns=%d dequeue_ns=%d "
                    "preprocess_start_ns=%d preprocess_done_ns=%d encode_start_ns=%d "
                    "encode_done_ns=%d transfer_enqueue_ns=%d transfer_start_ns=%d "
                    "transfer_reserve_start_ns=%d transfer_pool_ready_ns=%d "
                    "transfer_reserve_done_ns=%d transfer_copy_start_ns=%d "
                    "transfer_copy_submit_ns=%d "
                    "transfer_copy_done_ns=%d transfer_register_start_ns=%d "
                    "transfer_register_done_ns=%d transfer_stage_done_ns=%d "
                    "publish_done_ns=%d receive_done_ns=%d "
                    "receive_metadata_ns=%d receive_setup_done_ns=%d "
                    "receive_transfer_done_ns=%d receive_materialize_start_ns=%d "
                    "receive_materialize_done_ns=%d receive_embedding_ns=%d "
                    "receive_concat_start_ns=%d receive_concat_done_ns=%d "
                    "receive_extra_meta_start_ns=%d receive_extra_meta_done_ns=%d "
                    "receive_result_ready_ns=%d language_apply_start_ns=%d "
                    "language_get_mm_data_done_ns=%d language_radix_done_ns=%d "
                    "language_ready_ns=%d language_scheduler_pickup_ns=%d "
                    "language_prefill_start_ns=%d "
                    "language_prefill_done_ns=%d queue_ms=%.3f "
                    "encode_stage_wait_ms=%.3f preprocess_ms=%.3f "
                    "encode_wait_ms=%.3f transfer_reserve_ms=%.3f "
                    "encode_dispatch_ms=%.3f encode_compute_ms=%.3f encode_ms=%.3f "
                    "post_vit_to_copy_ms=%.3f server_postprocess_ms=%.3f "
                    "server_token_count_ms=%.3f server_embedding_slice_ms=%.3f "
                    "server_split_compile_wait_ms=%.3f server_split_dispatch_ms=%.3f "
                    "server_metadata_ms=%.3f server_result_pack_ms=%.3f "
                    "server_postprocess_residual_ms=%.3f runtime_return_gap_ms=%.3f "
                    "runtime_postprocess_ms=%.3f runtime_metadata_prepare_ms=%.3f "
                    "runtime_embedding_data_ms=%.3f runtime_result_pack_ms=%.3f "
                    "runtime_postprocess_residual_ms=%.3f "
                    "runtime_timing_attach_ms=%.3f runtime_to_copy_gap_ms=%.3f "
                    "publish_ms=%.3f transfer_handoff_ms=%.3f "
                    "transfer_queue_ms=%.3f transfer_pool_setup_ms=%.3f "
                    "transfer_copy_submit_ms=%.3f transfer_copy_wait_ms=%.3f "
                    "transfer_worker_wait_ms=%.3f transfer_post_copy_queue_ms=%.3f "
                    "transfer_register_ms=%.3f transfer_publish_finalize_ms=%.3f "
                    "transfer_total_ms=%.3f receive_ms=%.3f mm_prepare_ms=%.3f "
                    "receive_metadata_wait_ms=%.3f receive_setup_ms=%.3f "
                    "receive_transfer_wait_ms=%.3f "
                    "receive_completion_to_materialize_ms=%.3f "
                    "receive_materialize_wait_ms=%.3f "
                    "receive_poll_delay_ms=%.3f receive_finalize_ms=%.3f "
                    "receive_concat_ms=%.3f receive_extra_meta_ms=%.3f "
                    "receive_result_pack_ms=%.3f language_pickup_wait_ms=%.3f "
                    "language_get_mm_data_ms=%.3f language_radix_finalize_ms=%.3f "
                    "receive_mm_ms=%.3f language_admission_wait_ms=%.3f "
                    "language_queue_after_pickup_ms=%.3f language_queue_ms=%.3f "
                    "prefill_ms=%.3f "
                    "total_to_prefill_ms=%.3f total_to_prefill_done_ms=%.3f",
                    req.rid,
                    timing["enqueue_ns"],
                    timing["dequeue_ns"],
                    timing["preprocess_start_ns"],
                    timing["preprocess_done_ns"],
                    timing["encode_start_ns"],
                    timing["encode_done_ns"],
                    timing["transfer_enqueue_ns"],
                    timing["transfer_start_ns"],
                    timing["transfer_reserve_start_ns"],
                    timing["transfer_pool_ready_ns"],
                    timing["transfer_reserve_done_ns"],
                    timing["transfer_copy_start_ns"],
                    timing["transfer_copy_submit_ns"],
                    timing["transfer_copy_done_ns"],
                    timing["transfer_register_start_ns"],
                    timing["transfer_register_done_ns"],
                    timing["transfer_stage_done_ns"],
                    timing["publish_done_ns"],
                    timing["receive_done_ns"],
                    timing["receive_metadata_ns"],
                    timing["receive_setup_done_ns"],
                    timing["receive_transfer_done_ns"],
                    timing["receive_materialize_start_ns"],
                    timing["receive_materialize_done_ns"],
                    timing["receive_embedding_ns"],
                    timing["receive_concat_start_ns"],
                    timing["receive_concat_done_ns"],
                    timing["receive_extra_meta_start_ns"],
                    timing["receive_extra_meta_done_ns"],
                    timing["receive_result_ready_ns"],
                    timing["language_apply_start_ns"],
                    timing["language_get_mm_data_done_ns"],
                    timing["language_radix_done_ns"],
                    timing["language_ready_ns"],
                    timing["language_scheduler_pickup_ns"],
                    timing["language_prefill_start_ns"],
                    timing["language_prefill_done_ns"],
                    _elapsed_ms(timing, "enqueue_ns", "dequeue_ns"),
                    _elapsed_ms(timing, "dequeue_ns", "preprocess_start_ns"),
                    _elapsed_ms(timing, "preprocess_start_ns", "preprocess_done_ns"),
                    _elapsed_ms(timing, "preprocess_done_ns", "transfer_reserve_start_ns"),
                    _elapsed_ms(
                        timing,
                        "transfer_reserve_start_ns",
                        "transfer_reserve_done_ns",
                    ),
                    _elapsed_ms(timing, "transfer_reserve_done_ns", "encode_start_ns"),
                    _elapsed_ms(timing, "encode_start_ns", "encode_done_ns"),
                    _elapsed_ms(timing, "dequeue_ns", "encode_done_ns"),
                    _elapsed_ms(timing, "encode_done_ns", "transfer_copy_start_ns"),
                    _duration_ms(timing, "encode_server_postprocess_duration_ns"),
                    _duration_ms(timing, "encode_token_count_duration_ns"),
                    _duration_ms(timing, "encode_embedding_slice_duration_ns"),
                    _duration_ms(timing, "encode_split_compile_wait_duration_ns"),
                    _duration_ms(timing, "encode_split_dispatch_duration_ns"),
                    _duration_ms(timing, "encode_metadata_duration_ns"),
                    _duration_ms(timing, "encode_result_pack_duration_ns"),
                    _duration_ms(timing, "encode_server_postprocess_residual_ns"),
                    _elapsed_ms(
                        timing,
                        "encode_server_postprocess_done_ns",
                        "runtime_encode_return_ns",
                    ),
                    _duration_ms(timing, "runtime_postprocess_duration_ns"),
                    _duration_ms(timing, "runtime_metadata_prepare_duration_ns"),
                    _duration_ms(timing, "runtime_embedding_data_duration_ns"),
                    _duration_ms(timing, "runtime_result_pack_duration_ns"),
                    _duration_ms(timing, "runtime_postprocess_residual_ns"),
                    _duration_ms(timing, "runtime_timing_attach_duration_ns"),
                    _elapsed_ms(
                        timing,
                        "runtime_postprocess_done_ns",
                        "transfer_copy_start_ns",
                    ),
                    _elapsed_ms(timing, "encode_done_ns", "publish_done_ns"),
                    _elapsed_ms(timing, "encode_done_ns", "transfer_enqueue_ns"),
                    _elapsed_ms(timing, "transfer_enqueue_ns", "transfer_start_ns"),
                    _elapsed_ms(timing, "transfer_copy_start_ns", "transfer_pool_ready_ns"),
                    _elapsed_ms(timing, "transfer_pool_ready_ns", "transfer_copy_submit_ns"),
                    _elapsed_ms(timing, "transfer_copy_submit_ns", "transfer_copy_done_ns"),
                    _elapsed_ms(timing, "transfer_start_ns", "transfer_copy_done_ns"),
                    _elapsed_ms(
                        timing,
                        "transfer_copy_done_ns",
                        "transfer_register_start_ns",
                    ),
                    _elapsed_ms(
                        timing,
                        "transfer_register_start_ns",
                        "transfer_register_done_ns",
                    ),
                    _elapsed_ms(timing, "transfer_register_done_ns", "publish_done_ns"),
                    _elapsed_ms(timing, "transfer_copy_start_ns", "publish_done_ns"),
                    _elapsed_ms(timing, "publish_done_ns", "receive_done_ns"),
                    _elapsed_ms(timing, "receive_done_ns", "language_ready_ns"),
                    _elapsed_ms(timing, "publish_done_ns", "receive_metadata_ns"),
                    _elapsed_ms(timing, "receive_metadata_ns", "receive_setup_done_ns"),
                    _elapsed_ms(timing, "receive_setup_done_ns", "receive_transfer_done_ns"),
                    _elapsed_ms(
                        timing,
                        "receive_transfer_done_ns",
                        "receive_materialize_start_ns",
                    ),
                    _elapsed_ms(
                        timing,
                        "receive_materialize_start_ns",
                        "receive_materialize_done_ns",
                    ),
                    _elapsed_ms(
                        timing,
                        "receive_materialize_done_ns",
                        "receive_embedding_ns",
                    ),
                    _elapsed_ms(timing, "receive_embedding_ns", "receive_done_ns"),
                    _elapsed_ms(
                        timing,
                        "receive_concat_start_ns",
                        "receive_concat_done_ns",
                    ),
                    _elapsed_ms(
                        timing,
                        "receive_extra_meta_start_ns",
                        "receive_extra_meta_done_ns",
                    ),
                    _elapsed_ms(timing, "receive_done_ns", "receive_result_ready_ns"),
                    _elapsed_ms(
                        timing,
                        "receive_result_ready_ns",
                        "language_apply_start_ns",
                    ),
                    _elapsed_ms(
                        timing,
                        "language_apply_start_ns",
                        "language_get_mm_data_done_ns",
                    ),
                    _elapsed_ms(
                        timing,
                        "language_get_mm_data_done_ns",
                        "language_ready_ns",
                    ),
                    _elapsed_ms(timing, "publish_done_ns", "language_ready_ns"),
                    _elapsed_ms(
                        timing,
                        "language_ready_ns",
                        "language_scheduler_pickup_ns",
                    ),
                    _elapsed_ms(
                        timing,
                        "language_scheduler_pickup_ns",
                        "language_prefill_start_ns",
                    ),
                    _elapsed_ms(
                        timing,
                        "language_ready_ns",
                        "language_prefill_start_ns",
                    ),
                    _elapsed_ms(
                        timing,
                        "language_prefill_start_ns",
                        "language_prefill_done_ns",
                    ),
                    _elapsed_ms(
                        timing,
                        "enqueue_ns",
                        "language_prefill_start_ns",
                    ),
                    _elapsed_ms(
                        timing,
                        "enqueue_ns",
                        "language_prefill_done_ns",
                    ),
                )

                preprocess_fields = (
                    "dispatch_start_ns",
                    "preprocess_request_start_ns",
                    "image_load_start_ns",
                    "image_load_done_ns",
                    "processor_submit_ns",
                    "processor_start_ns",
                    "processor_done_ns",
                    "preprocess_request_done_ns",
                )
                if all(name in timing for name in preprocess_fields):
                    logger.info(
                        "ENCODER-PREPROCESS-TIME req_id=%s dispatch_ms=%.3f "
                        "admission_ms=%.3f image_load_ms=%.3f "
                        "processor_queue_ms=%.3f processor_ms=%.3f "
                        "finalize_ms=%.3f request_total_ms=%.3f batch_tail_ms=%.3f",
                        req.rid,
                        _elapsed_ms(timing, "dispatch_start_ns", "enqueue_ns"),
                        _elapsed_ms(
                            timing,
                            "preprocess_start_ns",
                            "preprocess_request_start_ns",
                        ),
                        _elapsed_ms(
                            timing,
                            "image_load_start_ns",
                            "image_load_done_ns",
                        ),
                        _elapsed_ms(
                            timing,
                            "processor_submit_ns",
                            "processor_start_ns",
                        ),
                        _elapsed_ms(
                            timing,
                            "processor_start_ns",
                            "processor_done_ns",
                        ),
                        _elapsed_ms(
                            timing,
                            "processor_done_ns",
                            "preprocess_request_done_ns",
                        ),
                        _elapsed_ms(
                            timing,
                            "preprocess_request_start_ns",
                            "preprocess_request_done_ns",
                        ),
                        _elapsed_ms(
                            timing,
                            "preprocess_request_done_ns",
                            "preprocess_done_ns",
                        ),
                    )
