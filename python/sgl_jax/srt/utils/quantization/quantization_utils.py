# Quantization utilities for sglang-jax

import itertools
import logging
import re

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.configs.quantization_config import DTYPE_MAP

logger = logging.getLogger(__name__)


class Quantizer:
    """
    Handles model quantization logic, including rule compilation and layer replacement.
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.quant_config = model_config.quantization_config

        if self.quant_config is None:
            logger.debug("No quantization config found. Quantizer will be no-op.")
            self.linear_rules = []
        else:
            self.linear_rules = self._compile_linear_rules()

    def _compile_linear_rules(self) -> list[dict]:
        """Compile regex rules from config into usable patterns."""
        raw_rules = self.quant_config.get_linear_rules()
        compiled_rules = []

        for rule in raw_rules:
            try:
                pattern = re.compile(rule["module_path"])
            except re.error as e:
                raise ValueError(
                    f"Invalid regex pattern in quantization rules: {rule['module_path']}"
                ) from e

            weight_dtype_str = rule.get("weight_dtype")
            activation_dtype_str = rule.get("activation_dtype")

            # Convert string dtypes to jnp dtypes
            weight_dtype = DTYPE_MAP.get(weight_dtype_str)
            activation_dtype = DTYPE_MAP.get(activation_dtype_str)

            if weight_dtype is None:
                raise ValueError(f"weight_dtype is required in rule but got: {weight_dtype_str}")

            compiled_rules.append(
                {
                    "pattern": pattern,
                    "weight_dtype": weight_dtype,
                    "activation_dtype": activation_dtype,
                    # Preserve other potential config options like scale_policy
                    "scale_policy": rule.get("scale_policy", "per_channel"),
                }
            )
        return compiled_rules

    def _find_matching_rule(self, path: str) -> dict | None:
        """Find the first rule that matches the given module path."""
        for rule in self.linear_rules:
            if rule["pattern"].match(path):
                return rule
        return None

    def apply_linear_quantization(self, model: nnx.Module) -> nnx.Module:
        """
        [Scheme A] Online Quantization.

        Replaces LinearBase with QuantizedLinear AND calculates scales from current weights.
        Used when loading BF16 weights and quantizing them on the fly.
        """
        if not self.linear_rules:
            return model

        # Import locally to avoid circular imports
        from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear

        def _replace_recursive(obj, path: str = "", visited: set[int] | None = None):
            if visited is None:
                visited = set()

            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            if hasattr(obj, "__dict__"):
                for attr_name, attr_value in list(obj.__dict__.items()):
                    child_path = f"{path}/{attr_name}" if path else attr_name

                    if isinstance(attr_value, LinearBase):
                        rule = self._find_matching_rule(child_path)
                        if rule is not None:
                            logger.info(
                                "Quantizing (Online) %s: w_dtype=%s, a_dtype=%s",
                                child_path,
                                rule["weight_dtype"],
                                rule["activation_dtype"],
                            )
                            # Use from_linear to calculate scales from existing weights
                            quantized_linear = QuantizedLinear.from_linear(
                                attr_value,
                                weight_dtype=rule["weight_dtype"],
                                activation_dtype=rule["activation_dtype"],
                            )
                            setattr(obj, attr_name, quantized_linear)
                            del attr_value
                        else:
                            logger.debug("Skipping %s - no matching rule", child_path)

                    elif isinstance(attr_value, nnx.Module):
                        _replace_recursive(attr_value, child_path, visited)

                    elif isinstance(attr_value, list):
                        for idx, item in enumerate(attr_value):
                            if isinstance(item, nnx.Module):
                                item_path = f"{child_path}[{idx}]"
                                _replace_recursive(item, item_path, visited)

        logger.info("Applying online linear quantization...")
        _replace_recursive(model)
        return model

    def apply_linear_quantization_structure(self, model: nnx.Module) -> nnx.Module:
        """
        [Scheme B] Structure Replacement Only.

        Replaces LinearBase with empty QuantizedLinear layers suitable for direct loading.
        Does NOT calculate scales (assumes they will be loaded).
        """
        if not self.linear_rules:
            return model

        from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear

        def _replace_recursive_struct(obj, path: str = "", visited: set[int] | None = None):
            if visited is None:
                visited = set()

            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            if hasattr(obj, "__dict__"):
                for attr_name, attr_value in list(obj.__dict__.items()):
                    child_path = f"{path}/{attr_name}" if path else attr_name

                    if isinstance(attr_value, LinearBase):
                        rule = self._find_matching_rule(child_path)
                        if rule is not None:
                            logger.info(
                                "Replacing Structure %s: w_dtype=%s (Waiting for load)",
                                child_path,
                                rule["weight_dtype"],
                            )

                            # For structure replacement, we need to create placeholder weights
                            # Extract dimensions from weight shape: (input_size, output_size)
                            input_size, output_size = attr_value.weight.value.shape

                            # Create ShapeDtypeStruct placeholders for quantized weights
                            # QuantizedLinear expects weight_q: [output_size, input_size] (transposed)
                            weight_q_struct = jax.ShapeDtypeStruct(
                                (output_size, input_size),
                                rule["weight_dtype"],
                            )
                            # Per-channel scale: [output_size]
                            weight_scale_struct = jax.ShapeDtypeStruct(
                                (output_size,),
                                jnp.float32,
                            )
                            # Bias: [output_size] if exists
                            bias_struct = None
                            if attr_value.bias is not None:
                                bias_struct = jax.ShapeDtypeStruct(
                                    (output_size,),
                                    attr_value.params_dtype,
                                )

                            new_layer = QuantizedLinear(
                                weight_q=weight_q_struct,
                                weight_scale=weight_scale_struct,
                                bias=bias_struct,
                                activation_dtype=rule["activation_dtype"],
                                mesh=attr_value.mesh,
                                kernel_axes=attr_value.kernel_axes,
                                skip_bias_add=attr_value.skip_bias_add,
                                params_dtype=attr_value.params_dtype,
                                scope_name=getattr(attr_value, "name", "quantized_linear"),
                            )

                            setattr(obj, attr_name, new_layer)
                            del attr_value
                        else:
                            pass  # No match

                    elif isinstance(attr_value, nnx.Module):
                        _replace_recursive_struct(attr_value, child_path, visited)

                    elif isinstance(attr_value, list):
                        for idx, item in enumerate(attr_value):
                            if isinstance(item, nnx.Module):
                                item_path = f"{child_path}[{idx}]"
                                _replace_recursive_struct(item, item_path, visited)

        logger.info("Applying linear quantization structure replacement...")
        _replace_recursive_struct(model)
        return model

    def apply_moe_quantization(self, model: nnx.Module) -> nnx.Module:
        """
        Apply quantization to MoE weights in-place.
        """
        if self.quant_config is None or not self.quant_config.has_moe_quantization():
            return model

        from sgl_jax.srt.layers.moe import EPMoE

        def _quantize_moe_recursive(obj, visited=None):
            if visited is None:
                visited = set()

            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            if isinstance(obj, EPMoE):
                # EPMoE usually handles its own "online" quantization via this method.
                # If doing direct FP8 load, EPMoE might need a similar "structure only"
                # path if it creates new parameters. For now, we assume standard behavior.
                obj.quantize_weights()
                return

            if hasattr(obj, "__dict__"):
                for attr_name, attr_value in obj.__dict__.items():
                    if isinstance(attr_value, nnx.Module):
                        _quantize_moe_recursive(attr_value, visited)
                    elif isinstance(attr_value, list):
                        for item in attr_value:
                            if isinstance(item, nnx.Module):
                                _quantize_moe_recursive(item, visited)

        logger.info("Applying MoE quantization...")
        _quantize_moe_recursive(model)
        return model

    def apply_moe_quantization_structure(self, model: nnx.Module) -> nnx.Module:
        """
        [Scheme B] MoE Structure Replacement Only.
        Replicates the logic of EPMoE.quantize_weights but only initializes
        the ShapeDtypeStructs for weights and scales.
        """
        if self.quant_config is None or not self.quant_config.has_moe_quantization():
            return model

        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.layers.moe import EPMoE

        weight_dtype = self.quant_config.get_moe_weight_dtype()

        def _init_moe_structure_recursive(obj, visited=None):
            if visited is None:
                visited = set()
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            if isinstance(obj, EPMoE):
                with jax.sharding.use_abstract_mesh(obj.updated_mesh):

                    # --- 1. wi_0 (Expert, Interim, Hidden) ---
                    # Sharding: ("expert", "tensor", None)
                    # Quantized axis: 2 (Hidden)
                    # Scale logic in moe.py:
                    #   w0_scale comes from axis 2 -> Shape (E, I)
                    #   Reshaped to (E, 1, 1, I)
                    #   Scale Sharding: ("expert", None, None, "tensor")

                    obj.wi_0 = nnx.Param(
                        jax.ShapeDtypeStruct(obj.wi_0.value.shape, weight_dtype),
                        out_sharding=P("expert", "tensor", None),
                    )

                    # wi_0_scale
                    if hasattr(obj, "wi_0_scale"):
                        del obj.wi_0_scale
                    scale_shape_0 = (obj.wi_0.value.shape[0], 1, 1, obj.wi_0.value.shape[1])
                    obj.wi_0_scale = nnx.Param(
                        jax.ShapeDtypeStruct(scale_shape_0, jnp.float32),
                        out_sharding=P("expert", None, None, "tensor"),
                    )

                    # --- 2. wi_1 (Expert, Interim, Hidden) ---
                    # Logic same as wi_0

                    obj.wi_1 = nnx.Param(
                        jax.ShapeDtypeStruct(obj.wi_1.value.shape, weight_dtype),
                        out_sharding=P("expert", "tensor", None),
                    )

                    if hasattr(obj, "wi_1_scale"):
                        del obj.wi_1_scale
                    scale_shape_1 = (obj.wi_1.value.shape[0], 1, 1, obj.wi_1.value.shape[1])
                    obj.wi_1_scale = nnx.Param(
                        jax.ShapeDtypeStruct(scale_shape_1, jnp.float32),
                        out_sharding=P("expert", None, None, "tensor"),
                    )

                    obj.wo = nnx.Param(
                        jax.ShapeDtypeStruct(obj.wo.value.shape, weight_dtype),
                        out_sharding=P("expert", None, "tensor"),
                    )

                    if hasattr(obj, "wo_scale"):
                        del obj.wo_scale
                    scale_shape_o = (obj.wo.value.shape[0], 1, 1, obj.wo.value.shape[1])
                    obj.wo_scale = nnx.Param(
                        jax.ShapeDtypeStruct(scale_shape_o, jnp.float32),
                        out_sharding=P("expert", None, None, None),
                    )

                    logger.info(
                        "Initialized MoE FP8 Structure for Layer %s",
                        getattr(obj, "layer_id", "?"),
                    )

                return

            if hasattr(obj, "__dict__"):
                for attr_name, attr_value in obj.__dict__.items():
                    if isinstance(attr_value, nnx.Module):
                        _init_moe_structure_recursive(attr_value, visited)
                    elif isinstance(attr_value, list):
                        for item in attr_value:
                            if isinstance(item, nnx.Module):
                                _init_moe_structure_recursive(item, visited)

        logger.info("Applying MoE quantization structure replacement...")
        _init_moe_structure_recursive(model)
        return model


def apply_linear_quantization(model_config: ModelConfig, model: nnx.Module) -> nnx.Module:
    return Quantizer(model_config).apply_linear_quantization(model)


def apply_moe_quantization(model_config: ModelConfig, model: nnx.Module) -> nnx.Module:
    return Quantizer(model_config).apply_moe_quantization(model)


def quantize_tensor_simple(
    x: jax.Array, dtype: jnp.dtype, dim: int = -1, out_dtype: jnp.dtype = jnp.float32
):
    """Simple per-token quantization for activations."""
    if jnp.issubdtype(dtype, jnp.integer):
        dtype_info = jnp.iinfo(dtype)
        max_val = int(dtype_info.max)
        min_val = int(dtype_info.min)
    else:
        dtype_info = jnp.finfo(dtype)
        max_val = float(dtype_info.max)
        min_val = float(dtype_info.min)

    x_abs_max = jnp.max(jnp.abs(x), axis=dim, keepdims=True)
    scale = x_abs_max / max_val
    x_q = jnp.clip(x / scale, min_val, max_val).astype(dtype)
    return x_q, scale.astype(out_dtype)


def quantize_tensor(
    dtype: jnp.dtype,
    tensor: jax.Array,
    axis: int | tuple | None = -1,
    block_size: int | None = None,
    pad_tensor: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Quantize tensor."""
    if axis is None:
        axis = [i for i in range(tensor.ndim)]
    if isinstance(axis, int):
        axis = [axis]

    orig_shape = tensor.shape
    mask = jnp.ones_like(tensor, jnp.int32)

    if block_size is not None:
        if isinstance(block_size, int):
            block_size = [block_size] * len(axis)

        blocked_shape = [[i] for i in orig_shape]
        pad_width = [[0, 0] for _ in range(tensor.ndim)]
        for i, block in zip(axis, block_size):
            num_blocks = (tensor.shape[i] + block - 1) // block
            padding_size = num_blocks * block - tensor.shape[i]
            if padding_size and not pad_tensor:
                raise ValueError(
                    f"Unable to perform block quantization. axis={i} of "
                    f"{tensor.shape=} is not divisible by {block=}"
                )
            pad_width[i][1] = padding_size
            blocked_shape[i] = (num_blocks, block)

        tensor = jnp.pad(tensor, pad_width, "edge")
        mask = jnp.pad(mask, pad_width)
        orig_shape = tensor.shape
        axis = sorted([i % tensor.ndim for i in axis])
        axis = [1 + n + i for n, i in enumerate(axis)]
        blocked_shape = list(itertools.chain(*blocked_shape))
        tensor = tensor.reshape(blocked_shape)

    dtype_info = jnp.iinfo(dtype) if jnp.issubdtype(dtype, jnp.integer) else jnp.finfo(dtype)
    dtype_max = float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    abs_max = jnp.max(jnp.abs(tensor), axis=axis, keepdims=True)
    scale = abs_max / dtype_max

    tensor_q = jnp.clip(tensor / scale, dtype_min, dtype_max)
    tensor_q = tensor_q.reshape(orig_shape)
    tensor_q = tensor_q.astype(dtype)
    tensor_q = jnp.where(mask, tensor_q, 0)
    scale = jnp.squeeze(scale, axis).astype(jnp.float32)

    return tensor_q, scale


def dequantize_tensor(
    tensor_q: jax.Array,
    scale: jax.Array,
    axis: int | None | tuple = -1,
    out_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Dequantize a quantized tensor."""
    if axis is None:
        axis = [i for i in range(tensor_q.ndim)]
    if isinstance(axis, int):
        axis = [axis]

    orig_shape = tensor_q.shape
    if tensor_q.ndim == scale.ndim:
        blocked_shape = [[i] for i in orig_shape]
        for i in axis:
            num_blocks = scale.shape[i]
            if tensor_q.shape[i] % num_blocks:
                raise ValueError(
                    f"Unable to perform block dequantization. axis={i} of "
                    f"{tensor_q.shape=} is not divisible by {num_blocks=}",
                )
            block_size = tensor_q.shape[i] // num_blocks
            blocked_shape[i] = (num_blocks, block_size)

        axis = sorted([(i + tensor_q.ndim) % tensor_q.ndim for i in axis])
        axis = [1 + n + i for n, i in enumerate(axis)]
        blocked_shape = list(itertools.chain(*blocked_shape))
        tensor_q = tensor_q.reshape(blocked_shape)

    scale = jnp.expand_dims(scale, axis)
    tensor = (tensor_q.astype(jnp.float32) * scale).astype(out_dtype)
    return tensor.reshape(orig_shape)
