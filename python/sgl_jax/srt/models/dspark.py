from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.models.dflash import DFlashDraftModel
from sgl_jax.srt.speculative.dspark_util import dspark_config_from_hf
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class VanillaMarkovHead(nnx.Module):
    """Low-rank token-to-token correction used by DSpark stage1."""

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        if int(markov_rank) <= 0:
            raise ValueError(f"VanillaMarkovHead requires markov_rank > 0, got {markov_rank}.")
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.mesh = mesh
        self.markov_w1 = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (self.vocab_size, self.markov_rank),
                dtype=dtype,
                out_sharding=P(None, None),
            )
        )
        self.markov_w2 = LinearBase(
            input_size=self.markov_rank,
            output_size=self.vocab_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="markov_w2",
        )

    def get_prev_embeddings(self, token_ids: jax.Array) -> jax.Array:
        sharding = NamedSharding(self.mesh, P("data", None))
        return self.markov_w1.value.at[token_ids.astype(jnp.int32)].get(out_sharding=sharding)

    def apply_step_logits(
        self,
        base_logits: jax.Array,
        token_ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        markov_embedding = self.get_prev_embeddings(token_ids)
        bias, _ = self.markov_w2(markov_embedding)
        return base_logits + bias, markov_embedding


class DSparkConfidenceHead(nnx.Module):
    """Confidence projection over ``[draft_hidden, markov_embedding]``."""

    def __init__(
        self,
        *,
        hidden_size: int,
        markov_rank: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.input_size = int(hidden_size) + int(markov_rank)
        self.proj = LinearBase(
            input_size=self.input_size,
            output_size=1,
            use_bias=True,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="proj",
        )

    def raw_confidence(
        self,
        hidden_states: jax.Array,
        markov_embeddings: jax.Array,
    ) -> jax.Array:
        if hidden_states.shape[:-1] != markov_embeddings.shape[:-1]:
            raise ValueError(
                "DSPARK confidence hidden/Markov leading shapes differ: "
                f"{hidden_states.shape} vs {markov_embeddings.shape}."
            )
        # Explicit-sharding JAX requires concatenate operands to carry the
        # same layout. Draft hidden features are TP-sharded on their last
        # dimension while the replicated W1 lookup naturally produces a
        # data-sharded Markov embedding, so align the embedding to the hidden
        # feature layout before concatenating them.
        markov_embeddings = markov_embeddings.astype(hidden_states.dtype)
        hidden_sharding = jax.typeof(hidden_states).sharding
        hidden_mesh = getattr(hidden_sharding, "mesh", None)
        if hidden_mesh is not None and hidden_mesh.axis_names:
            markov_embeddings = jax.sharding.reshard(markov_embeddings, hidden_sharding)
        features = jnp.concatenate([hidden_states, markov_embeddings], axis=-1)
        raw, _ = self.proj(features)
        return raw[..., 0]

    def __call__(
        self,
        hidden_states: jax.Array,
        markov_embeddings: jax.Array,
    ) -> jax.Array:
        """Return uncalibrated confidence logits.

        STS owns the only sigmoid so calibration can apply ``sigmoid(logit / T)``
        directly without losing saturated-logit information through a
        probability round trip.
        """
        return self.raw_confidence(hidden_states, markov_embeddings)


class DSparkDraftModel(DFlashDraftModel):
    """DSpark stage1 draft backbone with vanilla Markov and confidence heads."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        dspark_config = dspark_config_from_hf(config)
        super().__init__(config=config, mesh=mesh, dtype=dtype)
        self.gamma = dspark_config.gamma
        self.markov_head = VanillaMarkovHead(
            vocab_size=int(config.vocab_size),
            markov_rank=dspark_config.markov_rank,
            mesh=mesh,
            dtype=dtype,
        )
        self.confidence_head = DSparkConfidenceHead(
            hidden_size=int(config.hidden_size),
            markov_rank=dspark_config.markov_rank,
            mesh=mesh,
            dtype=dtype,
        )

    def generate_markov_block(
        self,
        base_logits: jax.Array,
        hidden_states: jax.Array,
        first_prev_tokens: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Generate ``gamma`` proposals serially and return confidence logits.

        Stage1 always verifies every generated proposal. Raw logits are still
        returned so checkpoint/head parity can be tested and later stages can
        apply STS without a sigmoid/logit round trip.
        """

        if base_logits.ndim != 3 or base_logits.shape[1] != self.gamma:
            raise ValueError(
                f"DSPARK base_logits must have shape [batch,{self.gamma},vocab], "
                f"got {base_logits.shape}."
            )
        if hidden_states.shape[:2] != base_logits.shape[:2]:
            raise ValueError(
                "DSPARK hidden/base-logit block shapes differ: "
                f"{hidden_states.shape} vs {base_logits.shape}."
            )

        prev_tokens = first_prev_tokens.astype(jnp.int32)
        tokens = []
        confidence_logits = []
        for step in range(self.gamma):
            corrected_logits, markov_embedding = self.markov_head.apply_step_logits(
                base_logits[:, step, :],
                prev_tokens,
            )
            next_tokens = jnp.argmax(corrected_logits, axis=-1).astype(jnp.int32)
            tokens.append(next_tokens)
            confidence_logits.append(
                self.confidence_head(hidden_states[:, step, :], markov_embedding)
            )
            prev_tokens = next_tokens
        return jnp.stack(tokens, axis=1), jnp.stack(confidence_logits, axis=1)

    def _create_weight_mappings(self) -> dict[str, WeightMapping]:
        # Call the base implementation explicitly so mapping-only coverage tests
        # can use a lightweight object carrying just ``config``.
        mappings = DFlashDraftModel._create_weight_mappings(self)
        mappings.update(
            {
                "markov_head.markov_w1.weight": WeightMapping(
                    target_path="markov_head.markov_w1",
                    sharding=(None, None),
                    transpose=False,
                ),
                "markov_head.markov_w2.weight": WeightMapping(
                    target_path="markov_head.markov_w2.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                "confidence_head.proj.weight": WeightMapping(
                    target_path="confidence_head.proj.weight",
                    sharding=(None, None),
                    transpose=True,
                ),
                "confidence_head.proj.bias": WeightMapping(
                    target_path="confidence_head.proj.bias",
                    sharding=(None,),
                    transpose=False,
                ),
            }
        )
        return mappings

    def load_weights(self, model_config) -> None:
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        mappings = self._create_weight_mappings()
        if not loader.dummy_mode:
            checkpoint_keys = set(loader._scan_weight_info())
            expected = set(mappings)
            missing = expected - checkpoint_keys
            allowed_shared = {"embed_tokens.weight", "lm_head.weight"}
            unexpected = checkpoint_keys - expected - allowed_shared
            if missing:
                raise ValueError(
                    f"DSPARK checkpoint is missing required stage1 weights: {sorted(missing)}."
                )
            if unexpected:
                raise ValueError(
                    f"DSPARK stage1 checkpoint contains unsupported weights: {sorted(unexpected)}."
                )
        loader.load_weights_from_safetensors(mappings)
        logger.info(
            "DSpark stage1 weights loaded; checkpoint embed_tokens/lm_head are intentionally "
            "skipped in favor of the live target modules."
        )


class Qwen3DSparkModel(DSparkDraftModel):
    pass


EntryClass = [Qwen3DSparkModel]
