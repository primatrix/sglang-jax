from __future__ import annotations

from sgl_jax.srt.speculative.dflash_worker import DFlashWorker
from sgl_jax.srt.speculative.dspark_util import parse_dspark_draft_config


class DSparkWorker(DFlashWorker):
    """DSpark stage1 worker: Markov draft plus fixed verify-all."""

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
