from __future__ import annotations

import json
import logging
import os

import jax
import numpy as np

from sgl_jax.srt.speculative.dflash_worker import DFlashWorker
from sgl_jax.srt.speculative.dspark_util import parse_dspark_draft_config

logger = logging.getLogger(__name__)


class DSparkWorker(DFlashWorker):
    """DSpark stage1 worker: Markov draft plus fixed verify-all."""

    def __init__(self, server_args, target_worker):
        self._sts_capture_path = os.getenv("SGL_JAX_DSPARK_STS_CAPTURE_PATH")
        self.dspark_tuned_config = None
        self._dspark_sts_temperatures = None
        super().__init__(server_args, target_worker)

    @staticmethod
    def _draft_model_class():
        from sgl_jax.srt.models.dspark import DSparkDraftModel

        return DSparkDraftModel

    @staticmethod
    def _parse_block_config(server_args):
        return parse_dspark_draft_config(
            server_args.speculative_draft_model_path,
            revision=server_args.speculative_draft_model_revision,
            trust_remote_code=server_args.trust_remote_code,
        )

    def _configure_widths(self, block_config) -> None:
        config = block_config
        if self.verify_width != config.verify_width:
            raise ValueError(
                "DSPARK internal verify width must be checkpoint block_size + 1: "
                f"got {self.verify_width}, expected {config.verify_width}."
            )
        self.draft_width = config.draft_width
        self.verify_width = config.verify_width
        self.block_size = self.verify_width
        self._proposal_hidden_start = 0
        if getattr(self.server_args, "enable_dspark_tuned_config", False):
            from sgl_jax.srt.speculative.dspark_tuned_config import (
                get_tuned_dspark_config,
                make_dspark_tuned_key,
            )
            from sgl_jax.srt.utils.jax_utils import get_device_name

            key = make_dspark_tuned_key(
                target_model=self.server_args.model_path,
                draft_model=self.server_args.speculative_draft_model_path,
                target_revision=self.server_args.revision,
                draft_revision=self.server_args.speculative_draft_model_revision,
                device_name=get_device_name(),
                device_count=self.mesh.size,
                dtype=self.server_args.dtype,
                quantization=self.server_args.quantization,
                tp_size=self.server_args.tp_size,
                dp_size=self.server_args.dp_size,
                gamma=config.gamma,
                page_size=self.server_args.page_size,
                attention_backend=self.server_args.attention_backend,
                overlap=not self.server_args.disable_overlap_schedule,
            )
            self.dspark_tuned_config = get_tuned_dspark_config(key)
            if self.dspark_tuned_config is None:
                logger.warning(
                    "DSPARK tuned-config lookup miss for key=%s; keeping fixed verify-all.", key
                )
            else:
                self._dspark_sts_temperatures = self.dspark_tuned_config.sts_temperatures
                logger.info(
                    "Using DSPARK tuned config: %s", self.dspark_tuned_config.provenance
                )

    def verify(self, model_worker_batch, cur_allocate_lens=None, *, update_relay=False):
        plan = getattr(model_worker_batch, "_dflash_target_verify_plan", None)
        result = super().verify(
            model_worker_batch,
            cur_allocate_lens,
            update_relay=update_relay,
        )
        if self._sts_capture_path and plan is not None and plan.draft_confidence is not None:
            confidence, accept_lens, active_mask = jax.device_get(
                (plan.draft_confidence, result.accept_lens, plan.active_mask)
            )
            confidence = np.asarray(confidence, dtype=np.float32)
            accepted_draft = np.asarray(accept_lens, dtype=np.int32) - 1
            active_mask = np.asarray(active_mask, dtype=np.bool_)
            with open(self._sts_capture_path, "a", encoding="utf-8") as capture:
                for row, accepted in zip(
                    confidence[active_mask], accepted_draft[active_mask], strict=True
                ):
                    capture.write(
                        json.dumps(
                            {
                                "confidence": row.tolist(),
                                "accepted_draft": int(accepted),
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        return result
