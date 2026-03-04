"""Shared utilities for Qwen VL models (Qwen2.5-VL and Qwen3-VL)."""

import logging
from typing import Literal, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.layers.embeddings import Embed, apply_rotary_emb
from sgl_jax.srt.utils.weight_utils import WeightLoader

logger = logging.getLogger(__name__)

_FLASH_MHA = None


def _get_flash_mha():
    """Lazy import of flash_mha to avoid import errors when not available."""
    global _FLASH_MHA
    if _FLASH_MHA is None:
        from flash_attn_jax import flash_mha as _FLASH_MHA
    return _FLASH_MHA


# TypedDict definitions for image inputs
class QwenVLImagePixelInputs(TypedDict):
    """Image inputs with pixel values."""

    type: Literal["pixel_values"]
    pixel_values: jax.Array
    image_grid_thw: tuple[tuple[int, int, int], ...]


class QwenVLImageEmbeddingInputs(TypedDict):
    """Image inputs with pre-computed embeddings."""

    type: Literal["image_embeds"]
    image_embeds: jax.Array
    image_grid_thw: jax.Array


QwenVLImageInputs = QwenVLImagePixelInputs | QwenVLImageEmbeddingInputs


class QwenVLVisionRotaryEmbedding(nnx.Module):
    """Rotary position embedding for vision encoder.

    Used by both Qwen2.5-VL and Qwen3-VL vision transformers.
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, seq_len: int) -> jax.Array:
        inv_freq = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        seq = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(seq, inv_freq)
        return freqs


class BaseQwenVLVisionModel(nnx.Module):
    """Base class for Qwen VL vision models.

    Provides shared multimodal input processing, weight loading pattern,
    and vision encoding interface. Subclasses must:
    - Set self.config, self.dtype, self.mesh, self.visual in __init__
    - Implement _create_vision_weight_mappings()
    - Override get_single_image_embedding() if visual returns a tuple
    """

    def load_weights(self, model_config) -> None:
        if not hasattr(self, "text_embed"):
            self.text_embed = Embed(
                num_embeddings=model_config.vocab_size,
                features=model_config.text_hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=(None, None),
                mesh=self.mesh,
            )

        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_vision_weight_mappings()

        if self.mesh is not None:
            with self.mesh:
                loader.load_weights_from_safetensors(weight_mappings)
        else:
            loader.load_weights_from_safetensors(weight_mappings)

    def _create_vision_weight_mappings(self) -> dict:
        raise NotImplementedError

    def _validate_and_reshape_mm_tensor(self, mm_input: object, name: str) -> jax.Array:
        if isinstance(mm_input, list):
            arrays_to_concat = [jnp.asarray(item) for item in mm_input]
            return jnp.concatenate(arrays_to_concat, axis=0)

        if hasattr(mm_input, "ndim"):
            array_input = jnp.asarray(mm_input)
            if array_input.ndim == 2:
                return array_input
            if array_input.ndim == 3:
                return array_input.reshape(-1, array_input.shape[-1])

        raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")

    def _parse_and_validate_image_input(
        self, image_grid_thw: tuple[tuple[int, int, int], ...], **kwargs: object
    ) -> QwenVLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(pixel_values, "image pixel values")
            return QwenVLImagePixelInputs(
                type="pixel_values", pixel_values=pixel_values, image_grid_thw=image_grid_thw
            )

        return None

    def _parse_and_validate_multimodal_inputs(
        self, image_grid_thw: tuple[tuple[int, int, int], ...], **kwargs: object
    ) -> dict:
        mm_input_by_modality = {}

        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    image_grid_thw, **kwargs
                )
        return mm_input_by_modality

    def get_single_image_embedding(self, image_pixel_values, image_grid_thw):
        return self.visual(image_pixel_values, (image_grid_thw,))

    def _process_image_input(self, image_input: QwenVLImageInputs) -> tuple[jax.Array, ...]:
        grid_thw = image_input["image_grid_thw"]

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].astype(self.dtype)
        else:
            pixel_values = image_input["pixel_values"]
            image_embeds = []
            current_idx = 0
            for image_thw in grid_thw:
                t, h, w = image_thw
                image_size = t * h * w
                end_idx = current_idx + image_size
                image_pixel_values = pixel_values[current_idx:end_idx, :]
                image_embeds.append(self.get_single_image_embedding(image_pixel_values, image_thw))
                current_idx = end_idx
            image_embeds = jnp.concatenate(image_embeds, axis=0)

        merge_size = self.visual.config.spatial_merge_size
        sizes = np.prod(np.array(grid_thw, dtype=np.int64), axis=-1) // merge_size // merge_size

        if sizes.size == 0:
            return ()
        if sizes.size == 1:
            return (image_embeds,)

        split_indices = np.cumsum(sizes)[:-1]
        return tuple(jnp.split(image_embeds, split_indices))

    def get_multimodal_embeddings(
        self, image_grid_thw: tuple[tuple[int, int, int], ...], **kwargs: object
    ) -> list[jax.Array]:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(image_grid_thw, **kwargs)
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: tuple[jax.Array, ...] = ()

        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings

        return list(multimodal_embeddings)

    def __call__(
        self,
        pixel_values: jax.Array,
        image_grid_thw: tuple[tuple[int, int, int], ...] = None,
        video_grid_thw: tuple[tuple[int, int, int], ...] = None,
    ) -> jax.Array:
        combined_grid_thw = []
        if image_grid_thw:
            combined_grid_thw.extend(image_grid_thw)
        if video_grid_thw:
            combined_grid_thw.extend(video_grid_thw)

        if not combined_grid_thw:
            return jnp.zeros((0, self.config.hidden_size), dtype=pixel_values.dtype)

        combined_grid_thw = tuple(combined_grid_thw)
        vision_embeds_list = self.get_multimodal_embeddings(
            image_grid_thw=combined_grid_thw,
            pixel_values=pixel_values,
        )
        return jnp.concatenate(vision_embeds_list, axis=0)


def _apply_interleaved_rope(x: jax.Array, mrope_section: list[int]) -> jax.Array:
    """Apply interleaved MRoPE layout.

    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THTHWHTHW...TT] pattern.
    """
    x_t = x[0]
    x_t = x_t.at[..., 1 : mrope_section[1] * 3 : 3].set(x[1, ..., 1 : mrope_section[1] * 3 : 3])
    x_t = x_t.at[..., 2 : mrope_section[2] * 3 : 3].set(x[2, ..., 2 : mrope_section[2] * 3 : 3])
    return x_t


class MRotaryEmbedding:
    """Rotary Embedding with Multimodal Sections for Qwen VL models.

    Partitions head dimension into sections for temporal (T), height (H),
    and width (W) positions.

    Used by both Qwen2.5-VL and Qwen3-VL.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        mrope_section: list[int],
        mrope_interleaved: bool = False,
    ) -> None:
        del max_position_embeddings
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.mrope_section = list(mrope_section)
        self.mrope_interleaved = mrope_interleaved

        inv_freq_np = 1.0 / (base ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
        self._inv_freq_np = inv_freq_np

        # Validate and adjust section sizes
        expected_sum = rotary_dim // 2
        actual_sum = sum(self.mrope_section)
        if actual_sum != expected_sum:
            logger.warning(
                "MRoPE section sum mismatch: expected %s, got %s. Adjusting.",
                expected_sum,
                actual_sum,
            )
            if actual_sum > 0:
                scale_factor = expected_sum / actual_sum
                self.mrope_section = [
                    max(1, int(section * scale_factor)) for section in self.mrope_section
                ]
                current_sum = sum(self.mrope_section)
                if current_sum != expected_sum:
                    self.mrope_section[-1] += expected_sum - current_sum
            else:
                self.mrope_section = [expected_sum // len(self.mrope_section)] * len(
                    self.mrope_section
                )
                remainder = expected_sum % len(self.mrope_section)
                for i in range(remainder):
                    self.mrope_section[i] += 1

    def __call__(
        self,
        positions: jax.Array,
        query: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        inv_freq = jnp.asarray(self._inv_freq_np, dtype=self.dtype)

        if positions.ndim == 1:
            freqs = jnp.einsum("n,d->nd", positions.astype(jnp.float32), inv_freq)
            cos = jnp.cos(freqs).astype(self.dtype)
            sin = jnp.sin(freqs).astype(self.dtype)
        else:
            freqs = jnp.einsum("tn,d->tnd", positions.astype(jnp.float32), inv_freq)
            cos = jnp.cos(freqs).astype(self.dtype)
            sin = jnp.sin(freqs).astype(self.dtype)
            if self.mrope_interleaved:
                cos = _apply_interleaved_rope(cos, self.mrope_section)
                sin = _apply_interleaved_rope(sin, self.mrope_section)
            else:
                cos_slices = []
                sin_slices = []
                offset = 0
                for i, section in enumerate(self.mrope_section):
                    cos_slices.append(cos[i, :, offset : offset + section])
                    sin_slices.append(sin[i, :, offset : offset + section])
                    offset += section
                cos = jnp.concatenate(cos_slices, axis=-1)
                sin = jnp.concatenate(sin_slices, axis=-1)

        num_tokens = positions.shape[-1]
        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = jnp.concatenate((query_rot, query_pass), axis=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = jnp.concatenate((key_rot, key_pass), axis=-1).reshape(key_shape)

        return query, key
