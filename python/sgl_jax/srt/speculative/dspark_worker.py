from __future__ import annotations

import json
import os

import jax
import numpy as np

from sgl_jax.srt.speculative.dflash_worker import DFlashWorker
from sgl_jax.srt.speculative.dspark_util import parse_dspark_draft_config


class DSparkWorker(DFlashWorker):
    """DSpark stage1 worker: Markov draft plus fixed verify-all."""

    def __init__(self, server_args, target_worker):
        self._sts_capture_path = os.getenv("SGL_JAX_DSPARK_STS_CAPTURE_PATH")
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
