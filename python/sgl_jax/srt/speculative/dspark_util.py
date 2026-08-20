from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DSparkDraftConfig:
    """Stage-1 DSpark checkpoint contract.

    ``gamma`` and ``draft_width`` are both the checkpoint ``block_size``.
    Target verification includes the anchor and is therefore one token wider.
    """

    gamma: int
    draft_width: int
    verify_width: int
    target_layer_ids: list[int]
    mask_token_id: int
    markov_rank: int
    markov_head_type: str
    confidence_head_with_markov: bool


def dspark_config_from_hf(hf_config) -> DSparkDraftConfig:
    architectures = list(getattr(hf_config, "architectures", []) or [])
    if "Qwen3DSparkModel" not in architectures:
        raise ValueError(
            f"DSPARK stage1 requires architectures=['Qwen3DSparkModel'], got {architectures!r}."
        )

    block_size = int(getattr(hf_config, "block_size", 0))
    if block_size <= 0:
        raise ValueError(f"DSPARK block_size must be > 0, got {block_size}.")

    target_layer_ids = getattr(hf_config, "target_layer_ids", None)
    if target_layer_ids is None:
        raise ValueError("DSPARK requires explicit target_layer_ids.")
    target_layer_ids = [int(x) for x in target_layer_ids]
    if len(target_layer_ids) != int(getattr(hf_config, "num_hidden_layers", 0)):
        raise ValueError(
            "DSPARK target_layer_ids must contain one target feature per draft layer: "
            f"got {len(target_layer_ids)} ids for "
            f"num_hidden_layers={getattr(hf_config, 'num_hidden_layers', None)}."
        )

    mask_token_id = getattr(hf_config, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError("DSPARK requires mask_token_id in the draft config.")
    mask_token_id = int(mask_token_id)
    vocab_size = int(getattr(hf_config, "vocab_size", 0))
    if mask_token_id < 0 or mask_token_id >= vocab_size:
        raise ValueError(
            f"DSPARK mask_token_id={mask_token_id} is outside vocab_size={vocab_size}."
        )

    markov_rank = int(getattr(hf_config, "markov_rank", 0))
    if markov_rank <= 0:
        raise ValueError(f"DSPARK markov_rank must be > 0, got {markov_rank}.")
    markov_head_type = str(getattr(hf_config, "markov_head_type", "")).lower()
    if markov_head_type != "vanilla":
        raise ValueError(
            f"DSPARK stage1 only supports markov_head_type='vanilla', got {markov_head_type!r}."
        )
    confidence_head_with_markov = bool(getattr(hf_config, "confidence_head_with_markov", False))
    if not bool(getattr(hf_config, "enable_confidence_head", True)):
        raise ValueError("DSPARK stage1 requires enable_confidence_head=true.")
    if not confidence_head_with_markov:
        raise ValueError("DSPARK stage1 requires confidence_head_with_markov=true.")

    return DSparkDraftConfig(
        gamma=block_size,
        draft_width=block_size,
        verify_width=block_size + 1,
        target_layer_ids=target_layer_ids,
        mask_token_id=mask_token_id,
        markov_rank=markov_rank,
        markov_head_type=markov_head_type,
        confidence_head_with_markov=confidence_head_with_markov,
    )


def parse_dspark_draft_config(
    model_path: str,
    revision: str | None = None,
    trust_remote_code: bool = True,
) -> DSparkDraftConfig:
    from sgl_jax.srt.hf_transformers_utils import get_config

    hf_config = get_config(
        model_path,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    return dspark_config_from_hf(hf_config)


__all__ = [
    "DSparkDraftConfig",
    "dspark_config_from_hf",
    "parse_dspark_draft_config",
]
