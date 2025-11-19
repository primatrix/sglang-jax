import glob
import logging
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from safetensors import safe_open
from tqdm import tqdm

from sgl_jax.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


class LazyWeightLoader:
    """Helper class for loading weight slices on-demand from safetensors files."""

    def __init__(self, st_files: list[str], hf_key: str, model_path: str):
        self.st_files = st_files
        self.hf_key = hf_key
        self.model_path = model_path
        self._weight_location = None  # (file_idx, tensor_name)
        self._weight_shape = None
        self._weight_dtype = None
        self._find_weight()

    def _find_weight(self):
        """Locate which safetensors file contains this weight."""
        for file_idx, st_file in enumerate(self.st_files):
            with safe_open(st_file, framework="flax") as f:
                keys = list(f.keys())
                if self.hf_key in keys:
                    self._weight_location = (file_idx, self.hf_key)
                    tensor_slice = f.get_slice(self.hf_key)
                    self._weight_shape = tensor_slice.get_shape()
                    self._weight_dtype = tensor_slice.get_dtype()
                    logger.debug(
                        "Found %s in %s, shape=%s, dtype=%s",
                        self.hf_key,
                        os.path.basename(st_file),
                        self._weight_shape,
                        self._weight_dtype,
                    )
                    return
        raise ValueError(f"Weight {self.hf_key} not found in any safetensors file")

    def load_slice(self, slice_spec: tuple) -> np.ndarray:
        """Load a specific slice of the weight.

        Args:
            slice_spec: Tuple of slices for each dimension

        Returns:
            The requested slice as a numpy array
        """
        if self._weight_location is None:
            raise RuntimeError("Weight location not found")

        file_idx, tensor_name = self._weight_location
        st_file = self.st_files[file_idx]

        with (
            jax.default_device(jax.local_devices(backend="cpu")[0]),
            safe_open(st_file, framework="flax") as f,
        ):
            # Load the full tensor first (safetensors doesn't support arbitrary slicing)
            # This is still per-process, so better than all-processes loading
            full_tensor = f.get_tensor(tensor_name)
            # Apply the slice
            sliced = full_tensor[slice_spec]
            return np.array(sliced)

    @property
    def shape(self):
        return self._weight_shape

    @property
    def dtype(self):
        return self._weight_dtype


class ShardingHelper:
    """Helper class for computing source slices from target sharding indices."""

    @staticmethod
    def compute_source_slice(
        target_indices: tuple,
        target_shape: tuple,
        source_shape: tuple,
        transpose: bool = False,
        split_dim: int | None = None,
        split_offset: int = 0,
    ) -> tuple:
        """Compute which slice of source data is needed for target indices.

        Args:
            target_indices: Indices in the target array (from make_array_from_callback)
            target_shape: Shape of target array
            source_shape: Shape of source array (before transformations)
            transpose: Whether source needs to be transposed
            split_dim: Dimension along which source is split (for QKV)
            split_offset: Offset in the split dimension (for K and V in QKV)

        Returns:
            Tuple of slices for the source array
        """
        # Target is transposed version of source
        # target shape (M, N) = transpose(source shape (N, M))
        # target_indices (i:j, k:l) -> source_indices (k:l, i:j)
        source_indices = tuple(reversed(target_indices)) if transpose else target_indices

        if split_dim is not None:
            # Adjust the split dimension to account for offset
            # E.g., for K in QKV, we need source[:, q_dim:q_dim+kv_dim]
            source_indices = list(source_indices)
            dim_slice = source_indices[split_dim]
            if isinstance(dim_slice, slice):
                start = (dim_slice.start or 0) + split_offset
                stop = (dim_slice.stop or target_shape[split_dim]) + split_offset
                source_indices[split_dim] = slice(start, stop, dim_slice.step)
            else:
                # Single index
                source_indices[split_dim] = dim_slice + split_offset
            source_indices = tuple(source_indices)

        return source_indices

    @staticmethod
    def compute_kv_head_replica_mapping(
        target_indices: tuple,
        target_shape: tuple,
        source_shape: tuple,
        num_original_heads: int,
        num_target_heads: int,
        head_dim: int,
        transpose: bool = False,
    ) -> tuple:
        """Compute source slice for KV head replication.

        When tp_size > num_kv_heads, we replicate KV heads.
        E.g., 4 original heads -> 16 target heads means each original head is replicated 4 times.

        Args:
            target_indices: Indices in target (replicated) array
            target_shape: Shape of target array
            source_shape: Shape of source array (before replication)
            num_original_heads: Number of KV heads in checkpoint
            num_target_heads: Number of KV heads after replication (usually tp_size)
            head_dim: Dimension of each head
            transpose: Whether the weight is transposed

        Returns:
            Tuple of slices for source array
        """
        num_replicas = num_target_heads // num_original_heads

        # Determine which dimension contains the heads
        # For k_proj/v_proj: shape is (hidden, num_heads * head_dim)
        # After transpose: (num_heads * head_dim, hidden)
        head_dim_axis = 1 if not transpose else 0

        # Get the slice in the head dimension
        target_slice = target_indices[head_dim_axis]

        if isinstance(target_slice, slice):
            # Convert target head indices to source head indices
            target_start = target_slice.start or 0
            target_stop = target_slice.stop or target_shape[head_dim_axis]

            # Map target head range to source head range
            source_head_start = target_start // head_dim // num_replicas
            source_head_stop = (target_stop + head_dim - 1) // head_dim // num_replicas

            source_start = source_head_start * head_dim
            source_stop = (source_head_stop + 1) * head_dim

            # Build source indices
            source_indices = list(target_indices)
            source_indices[head_dim_axis] = slice(source_start, source_stop, target_slice.step)
            return tuple(source_indices)
        else:
            # Single index case
            target_head_idx = target_slice // head_dim
            source_head_idx = target_head_idx // num_replicas
            offset_in_head = target_slice % head_dim
            source_idx = source_head_idx * head_dim + offset_in_head

            source_indices = list(target_indices)
            source_indices[head_dim_axis] = source_idx
            return tuple(source_indices)


@dataclass
class WeightMapping:
    target_path: str | list[str]
    sharding: tuple | None = None
    transpose: bool = False
    reshape: tuple | None = None
    head_dim_padding: bool = False
    kv_head_padding: bool = False
    concat_axis: int | None = None
    is_eagle3: bool = False

    def __post_init__(self):
        if self.sharding is None:
            self.sharding = self._infer_default_sharding()

    def _infer_default_sharding(self) -> tuple:
        path = self.target_path[0] if isinstance(self.target_path, list) else self.target_path

        if any(pattern in path for pattern in ["embedding", "lm_head"]):
            return (None, None)
        elif any(
            pattern in path
            for pattern in [
                "q_proj",
                "k_proj",
                "v_proj",
                "w1",
                "w2",
                "gate_proj",
                "up_proj",
            ]
        ):
            return (None, "tensor")
        elif any(pattern in path for pattern in ["c_proj", "o_proj", "down_proj"]):
            return ("tensor", None)
        elif "bias" in path or "weight" in path:
            return (None,)
        else:
            return (None,)


class WeightLoader:
    def __init__(
        self,
        model: nnx.Module,
        model_config: ModelConfig,
        mesh: Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.model = model
        self.model_config = model_config
        self.mesh = mesh
        self.dtype = dtype
        self.dummy_mode = getattr(model_config, "_dummy_mode", False)

        self.num_heads = model_config.num_attention_heads
        self.num_kv_heads = (
            model_config.get_total_num_kv_heads()
        )  # Use original count for replication logic
        self.hidden_size = model_config.hidden_size
        self.head_dim_original = getattr(
            model_config, "head_dim", self.hidden_size // self.num_heads
        )

        self.head_dim_pad = (self.head_dim_original + 127) // 128 * 128 - self.head_dim_original
        self.head_dim = self.head_dim_original
        if hasattr(self.mesh, "shape") and "tensor" in self.mesh.shape:
            self.sharding_size = self.mesh.shape["tensor"]
        else:
            self.sharding_size = 1

        if hasattr(model_config, "ep_size") and model_config.ep_size > 1:
            world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get("tensor", 1)
            tp_size = world_size // model_config.ep_size
            ep_size = model_config.ep_size
            abstract_mesh = self.mesh.abstract_mesh
            self.moe_abstract_mesh = abstract_mesh.update(
                axis_sizes=(ep_size, tp_size), axis_names=("expert", "tensor")
            )
        else:
            self.moe_abstract_mesh = None

        # Cache for safetensors files (will be populated during loading)
        self._st_files = None

    def _shard_weight_lazy(
        self,
        lazy_loader: LazyWeightLoader,
        target_shape: tuple,
        sharding_spec: tuple,
        mapping: WeightMapping,
        mesh: jax.sharding.Mesh = None,
    ) -> jax.Array:
        """Create a sharded array using lazy loading from safetensors.

        Args:
            lazy_loader: LazyWeightLoader instance
            target_shape: Shape of the target (transformed) array
            sharding_spec: Sharding specification
            mapping: WeightMapping with transformation info
            mesh: Mesh to use (defaults to self.mesh)

        Returns:
            Sharded jax.Array
        """
        if mesh is None:
            mesh = self.mesh
        # Ensure sharding_spec and target_shape are tuples
        if isinstance(sharding_spec, list):
            sharding_spec = tuple(sharding_spec)
        if isinstance(target_shape, list):
            target_shape = tuple(target_shape)
        target_sharding = jax.sharding.NamedSharding(mesh, P(*sharding_spec))

        if jax.process_count() == 1:
            # Single process: load full weight and shard
            full_weight = lazy_loader.load_slice(tuple(slice(None) for _ in lazy_loader.shape))
            weight_jax = jnp.array(full_weight)
            return jax.device_put(weight_jax, target_sharding)

        # Multi-process: use lazy loading
        def make_shard(indices):
            # Compute which slice of source data we need
            source_slice = self._compute_source_slice_for_mapping(
                indices, target_shape, lazy_loader.shape, mapping
            )

            # Load only the needed slice
            source_data = lazy_loader.load_slice(source_slice)

            # Apply transformations
            transformed = self._apply_transformations(
                source_data, indices, target_shape, lazy_loader.shape, mapping
            )

            return transformed

        return jax.make_array_from_callback(
            shape=target_shape, sharding=target_sharding, data_callback=make_shard
        )

    def _compute_source_slice_for_mapping(
        self,
        target_indices: tuple,
        target_shape: tuple,
        source_shape: tuple,
        mapping: WeightMapping,
    ) -> tuple:
        """Compute source slice needed for target indices based on mapping."""
        # Handle KV head padding first (needs special logic)
        if mapping.kv_head_padding and self.model_config.needs_kv_head_replication(
            self.sharding_size
        ):
            return ShardingHelper.compute_kv_head_replica_mapping(
                target_indices=target_indices,
                target_shape=target_shape,
                source_shape=source_shape,
                num_original_heads=self.num_kv_heads,
                num_target_heads=self.sharding_size,
                head_dim=self.head_dim,
                transpose=mapping.transpose,
            )

        # Handle regular transformations
        return ShardingHelper.compute_source_slice(
            target_indices=target_indices,
            target_shape=target_shape,
            source_shape=source_shape,
            transpose=mapping.transpose,
            split_dim=None,  # Will handle split separately
            split_offset=0,
        )

    def _apply_transformations(
        self,
        source_data: np.ndarray,
        target_indices: tuple,
        target_shape: tuple,
        source_shape: tuple,
        mapping: WeightMapping,
    ) -> jax.Array:
        """Apply necessary transformations to source data."""
        data = jnp.array(source_data)

        # Apply transpose if needed
        if mapping.transpose:
            data = jnp.transpose(data, (1, 0))

        # Apply KV head replication if needed
        if mapping.kv_head_padding and self.model_config.needs_kv_head_replication(
            self.sharding_size
        ):
            data = self._apply_kv_head_replication_to_slice(
                data, target_indices, target_shape, mapping
            )

        # Apply reshape if needed
        if mapping.reshape is not None:
            # Need to be careful with reshape and sharding
            data = jnp.reshape(data, self._compute_shard_reshape(target_indices, mapping.reshape))

        return data

    def _apply_kv_head_replication_to_slice(
        self,
        data: jax.Array,
        target_indices: tuple,
        target_shape: tuple,
        mapping: WeightMapping,
    ) -> jax.Array:
        """Replicate KV heads in a slice of data."""
        num_replicas = self.sharding_size // self.num_kv_heads
        padding_strategy = self.model_config.get_kv_padding_strategy()

        if padding_strategy != "replicate":
            # Zero padding - just pad with zeros
            # This is simpler but less correct
            return data

        # For replication, we need to replicate each source head
        # The source data already contains only the heads we need (computed by compute_kv_head_replica_mapping)
        # We need to replicate them to fill the target slice

        # Determine head dimension axis
        head_dim_axis = 1 if not mapping.transpose else 0

        # Extract the number of heads in this slice
        source_head_dim_size = data.shape[head_dim_axis]
        num_source_heads = source_head_dim_size // self.head_dim

        # Replicate each head
        replicated_parts = []
        for i in range(num_source_heads):
            start_idx = i * self.head_dim
            end_idx = (i + 1) * self.head_dim
            if head_dim_axis == 1:
                head_data = data[:, start_idx:end_idx]
                for _ in range(num_replicas):
                    replicated_parts.append(head_data)
            else:
                head_data = data[start_idx:end_idx, :]
                for _ in range(num_replicas):
                    replicated_parts.append(head_data)

        return jnp.concatenate(replicated_parts, axis=head_dim_axis)

    def _compute_shard_reshape(self, target_indices: tuple, full_reshape: tuple) -> tuple:
        """Compute the reshape for a shard given target indices."""
        # This is complex - for now, just return the portion of reshape that matches the shard
        # In practice, reshape with sharding needs careful handling
        return full_reshape

    def _load_weights_lazy(
        self,
        params: nnx.State,
        weight_mappings: dict[str, str | list[str] | WeightMapping],
        safetensors_partition: int,
    ):
        """Load weights using lazy loading (Option 3 from JAX guide).

        Each process only loads the weight slices it needs for its local devices.
        """
        # Get list of safetensors files
        model_path = self.model_config.model_path
        self._st_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

        if len(self._st_files) == 0:
            raise RuntimeError(f"Cannot find any *.safetensors files in {model_path}")

        logger.info("Found %s safetensors files", len(self._st_files))

        # Separate regular and MOE mappings
        regular_mappings = {}
        moe_mappings = {}

        for key, mapping in weight_mappings.items():
            if key.startswith("__MOE_EXPERTS__"):
                moe_mappings[key] = mapping
            else:
                regular_mappings[key] = mapping

        # Process regular weights with lazy loading
        for hf_key, mapping in tqdm(regular_mappings.items(), desc="[LAZY LOADING] Weights"):
            if isinstance(mapping, (str, list)):
                mapping = WeightMapping(target_path=mapping)

            try:
                # Create lazy loader
                lazy_loader = LazyWeightLoader(
                    st_files=self._st_files,
                    hf_key=hf_key,
                    model_path=model_path,
                )

                # Handle split vs single weight
                if isinstance(mapping.target_path, list):
                    self._handle_split_weight_lazy(params, hf_key, lazy_loader, mapping)
                else:
                    self._handle_single_weight_lazy(params, hf_key, lazy_loader, mapping)

            except ValueError as e:
                if not self._is_excluded_layer_weight(hf_key):
                    logger.warning("Could not load %s: %s", hf_key, e)
                continue

        # Process MOE weights with lazy loading
        if moe_mappings:
            logger.info("Loading MoE expert weights with lazy loading...")
            self._load_moe_weights_lazy(params, moe_mappings, safetensors_partition)

        nnx.update(self.model, params)
        logger.info("Lazy weight loading complete!")

    def _handle_single_weight_lazy(
        self,
        params: nnx.State,
        hf_key: str,
        lazy_loader: LazyWeightLoader,
        mapping: WeightMapping,
    ):
        """Handle loading a single weight with lazy loading."""
        jax_path = mapping.target_path
        model_param = self._get_param(params, jax_path)

        # Compute target shape after transformations
        source_shape = lazy_loader.shape
        target_shape = self._compute_target_shape(source_shape, mapping, is_single=True)

        # Apply output_multiplier_scale if needed (for lm_head)
        # Note: This is tricky with lazy loading, might need special handling
        if "lm_head" in hf_key and hasattr(self.model_config.hf_config, "output_multiplier_scale"):
            logger.warning(
                "output_multiplier_scale for %s not yet supported in lazy loading, weight will not be scaled",
                hf_key,
            )

        # Create sharded array using lazy loading
        sharded_weight = self._shard_weight_lazy(
            lazy_loader=lazy_loader,
            target_shape=target_shape,
            sharding_spec=mapping.sharding,
            mapping=mapping,
        )

        model_param.value = sharded_weight.astype(model_param.value.dtype)
        logger.debug("Lazy loaded %s -> %s, target_shape: %s", hf_key, jax_path, target_shape)

    def _handle_split_weight_lazy(
        self,
        params: nnx.State,
        hf_key: str,
        lazy_loader: LazyWeightLoader,
        mapping: WeightMapping,
    ):
        """Handle loading split QKV weights with lazy loading."""
        jax_paths = mapping.target_path
        source_shape = lazy_loader.shape

        # Compute split offsets for Q, K, V
        if hf_key.endswith(".bias"):
            q_dim = self.num_heads * self.head_dim_original
            kv_dim = self.num_kv_heads * self.head_dim_original
            split_offsets = [0, q_dim, q_dim + kv_dim]
            target_shapes = [(q_dim,), (kv_dim,), (kv_dim,)]
        else:
            q_dim = self.num_heads * self.head_dim_original
            kv_dim = self.num_kv_heads * self.head_dim_original

            if mapping.transpose:
                # Source shape: (hidden, q_dim + 2*kv_dim)
                # After transpose: (q_dim + 2*kv_dim, hidden)
                split_dim = 0  # Split along first dimension after transpose
                other_dim = source_shape[0]  # hidden size
                split_offsets = [0, q_dim, q_dim + kv_dim]
                target_shapes = [
                    (q_dim, other_dim),
                    (kv_dim, other_dim),
                    (kv_dim, other_dim),
                ]
            else:
                # Source shape: (q_dim + 2*kv_dim, hidden)
                split_dim = 0
                other_dim = source_shape[1]
                split_offsets = [0, q_dim, q_dim + kv_dim]
                target_shapes = [
                    (q_dim, other_dim),
                    (kv_dim, other_dim),
                    (kv_dim, other_dim),
                ]

        # Create lazy loaders for each split (Q, K, V)
        for jax_path, split_offset, target_shape in zip(jax_paths, split_offsets, target_shapes):
            # Create a modified mapping for this split
            split_mapping = WeightMapping(
                target_path=jax_path,
                sharding=mapping.sharding,
                transpose=mapping.transpose,
                reshape=mapping.reshape,
                head_dim_padding=mapping.head_dim_padding,
                kv_head_padding=(
                    mapping.kv_head_padding
                    if ("k_proj" in jax_path or "v_proj" in jax_path)
                    else False
                ),
                concat_axis=mapping.concat_axis,
                is_eagle3=mapping.is_eagle3,
            )

            # Need to create a wrapper that handles the split offset
            split_lazy_loader = self._create_split_lazy_loader(
                lazy_loader, split_dim if not mapping.transpose else 1, split_offset, target_shape
            )

            # Load this split
            model_param = self._get_param(params, jax_path)
            sharded_weight = self._shard_weight_lazy(
                lazy_loader=split_lazy_loader,
                target_shape=target_shape,
                sharding_spec=split_mapping.sharding,
                mapping=split_mapping,
            )

            model_param.value = sharded_weight.astype(model_param.value.dtype)
            logger.debug(
                "Lazy loaded split %s -> %s, offset=%s, shape=%s",
                hf_key,
                jax_path,
                split_offset,
                target_shape,
            )

    def _create_split_lazy_loader(
        self,
        base_loader: LazyWeightLoader,
        split_dim: int,
        split_offset: int,
        target_shape: tuple,
    ) -> LazyWeightLoader:
        """Create a lazy loader that handles offset for split weights."""

        class SplitLazyLoader:
            def __init__(self, base, dim, offset, shape):
                self.base = base
                self.split_dim = dim
                self.split_offset = offset
                self._shape = shape
                self._dtype = base.dtype

            def load_slice(self, slice_spec: tuple) -> np.ndarray:
                # Adjust slice_spec to include the split offset
                adjusted_spec = list(slice_spec)
                dim_slice = adjusted_spec[self.split_dim]
                if isinstance(dim_slice, slice):
                    start = (dim_slice.start or 0) + self.split_offset
                    stop = (dim_slice.stop or self._shape[self.split_dim]) + self.split_offset
                    adjusted_spec[self.split_dim] = slice(start, stop, dim_slice.step)
                else:
                    adjusted_spec[self.split_dim] = dim_slice + self.split_offset

                return self.base.load_slice(tuple(adjusted_spec))

            @property
            def shape(self):
                return self._shape

            @property
            def dtype(self):
                return self._dtype

        return SplitLazyLoader(base_loader, split_dim, split_offset, target_shape)

    def _load_moe_weights_lazy(
        self,
        params: nnx.State,
        moe_mappings: dict[str, WeightMapping],
        safetensors_partition: int,
    ):
        """Load MoE expert weights using lazy loading.

        Each MoE group consists of multiple experts' weights that need to be stacked.
        With lazy loading, each process only loads the experts it needs based on expert parallelism.
        """
        for moe_key, mapping in tqdm(moe_mappings.items(), desc="[LAZY LOADING] MoE Experts"):
            target_path = mapping.target_path[0]
            expected_hf_keys = mapping.target_path[1:]  # List of HF keys for all experts

            # Create lazy loaders for all expert weights
            expert_lazy_loaders = {}
            for hf_key in expected_hf_keys:
                try:
                    expert_lazy_loaders[hf_key] = LazyWeightLoader(
                        st_files=self._st_files,
                        hf_key=hf_key,
                        model_path=self.model_config.model_path,
                    )
                except ValueError as e:
                    if not self._is_excluded_layer_weight(hf_key):
                        logger.warning("Could not find MoE expert weight %s: %s", hf_key, e)
                    continue

            if not expert_lazy_loaders:
                logger.warning("No expert weights found for MoE group %s", moe_key)
                continue

            # Get model parameter to determine target shape
            model_param = self._get_param(params, target_path)
            target_shape = model_param.value.shape  # (num_experts, ...)
            num_experts = target_shape[0]
            expert_weight_shape = target_shape[1:]

            # Create lazy loaded sharded array for MoE experts
            if "expert" in mapping.sharding:
                # Expert parallelism case
                ep_size = getattr(self.model_config.hf_config, "ep_size", 1)
                world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get("tensor", 1)
                tp_size = world_size // ep_size

                devices = self.mesh.devices.flatten()
                moe_mesh = jax.sharding.Mesh(
                    devices.reshape(ep_size, tp_size), axis_names=("expert", "tensor")
                )
                sharding_spec = mapping.sharding
                mesh_to_use = moe_mesh
            else:
                # No expert parallelism
                sharding_spec = mapping.sharding
                mesh_to_use = self.mesh

            # Ensure sharding_spec and target_shape are tuples
            if isinstance(sharding_spec, list):
                sharding_spec = tuple(sharding_spec)
            if isinstance(target_shape, list):
                target_shape = tuple(target_shape)
            target_sharding = jax.sharding.NamedSharding(mesh_to_use, P(*sharding_spec))

            # Create the sharded array using callback
            def make_moe_shard(
                indices,
                loaders=expert_lazy_loaders,
                hf_keys=expected_hf_keys,
                num_experts_val=num_experts,
                expert_weight_shape_val=expert_weight_shape,
                mapping_val=mapping,
                model_param_val=model_param,
            ):
                # indices is a tuple for the full (num_experts, ...) shape
                # indices[0] is the slice for expert dimension
                expert_slice = indices[0]

                # Determine which experts we need for this shard
                if isinstance(expert_slice, slice):
                    start_expert = expert_slice.start or 0
                    stop_expert = expert_slice.stop or num_experts_val
                    expert_indices = range(start_expert, stop_expert)
                else:
                    # Single expert index
                    expert_indices = [expert_slice]

                # Load weights for each expert in this shard
                expert_weights = []
                for expert_idx in expert_indices:
                    if expert_idx >= len(hf_keys):
                        logger.warning(
                            "Expert index %s out of range (total %s)", expert_idx, len(hf_keys)
                        )
                        continue

                    hf_key = hf_keys[expert_idx]
                    if hf_key not in loaders:
                        logger.warning("No loader for expert %s", hf_key)
                        continue

                    loader = loaders[hf_key]

                    # Compute slice for the non-expert dimensions
                    # indices[1:] are slices for the weight dimensions
                    weight_indices = indices[1:]

                    # Load the slice we need
                    source_slice = self._compute_moe_expert_source_slice(
                        weight_indices, expert_weight_shape_val, loader.shape, mapping_val
                    )
                    source_data = loader.load_slice(source_slice)

                    # Apply transformations
                    transformed = self._apply_moe_expert_transformations(
                        source_data,
                        weight_indices,
                        expert_weight_shape_val,
                        loader.shape,
                        mapping_val,
                    )

                    expert_weights.append(transformed)

                # Stack expert weights along axis 0
                if expert_weights:
                    stacked = jnp.stack(expert_weights, axis=0)
                    return stacked
                else:
                    # Fallback: return zeros if no weights found
                    shard_shape = tuple(
                        [len(expert_indices)]
                        + [
                            dim_slice.stop - dim_slice.start if isinstance(dim_slice, slice) else 1
                            for dim_slice in indices[1:]
                        ]
                    )
                    return jnp.zeros(shard_shape, dtype=model_param_val.value.dtype)

            if jax.process_count() == 1:
                # Single process: load all experts eagerly
                all_expert_weights = []
                for hf_key in expected_hf_keys:
                    if hf_key not in expert_lazy_loaders:
                        continue
                    loader = expert_lazy_loaders[hf_key]
                    full_weight = loader.load_slice(tuple(slice(None) for _ in loader.shape))
                    weight_jax = jnp.array(full_weight)

                    # Apply transformations
                    if mapping.transpose:
                        weight_jax = jnp.transpose(weight_jax, (1, 0))
                    all_expert_weights.append(weight_jax)

                stacked = jnp.stack(all_expert_weights, axis=0)
                sharded_weight = jax.device_put(stacked, target_sharding)
            else:
                # Multi-process: use lazy loading
                sharded_weight = jax.make_array_from_callback(
                    shape=target_shape, sharding=target_sharding, data_callback=make_moe_shard
                )

            model_param.value = sharded_weight.astype(model_param.value.dtype)
            logger.debug("Lazy loaded MoE group %s, shape: %s", moe_key, target_shape)

    def _compute_moe_expert_source_slice(
        self,
        weight_indices: tuple,
        target_shape: tuple,
        source_shape: tuple,
        mapping: WeightMapping,
    ) -> tuple:
        """Compute source slice for a single MoE expert weight."""
        # Similar to regular weight slice computation
        return ShardingHelper.compute_source_slice(
            target_indices=weight_indices,
            target_shape=target_shape,
            source_shape=source_shape,
            transpose=mapping.transpose,
            split_dim=None,
            split_offset=0,
        )

    def _apply_moe_expert_transformations(
        self,
        source_data: np.ndarray,
        weight_indices: tuple,
        target_shape: tuple,
        source_shape: tuple,
        mapping: WeightMapping,
    ) -> jax.Array:
        """Apply transformations to a single MoE expert weight."""
        data = jnp.array(source_data)

        # Apply transpose if needed
        if mapping.transpose:
            data = jnp.transpose(data, (1, 0))

        return data

    def _compute_target_shape(
        self, source_shape: tuple, mapping: WeightMapping, is_single: bool = True
    ) -> tuple:
        """Compute the target shape after applying transformations."""
        shape = source_shape

        # Apply transpose
        if mapping.transpose:
            shape = tuple(reversed(shape))

        # Apply KV head padding
        if mapping.kv_head_padding and self.model_config.needs_kv_head_replication(
            self.sharding_size
        ):
            # Expand KV head dimension
            head_dim_axis = 1 if not mapping.transpose else 0

            shape_list = list(shape)
            shape_list[head_dim_axis] = self.sharding_size * self.head_dim
            shape = tuple(shape_list)

        # Apply reshape
        if mapping.reshape is not None:
            shape = mapping.reshape
            # Ensure shape is a tuple
            if isinstance(shape, list):
                shape = tuple(shape)

        return shape

    def load_weights_from_safetensors(
        self,
        weight_mappings: dict[str, str | list[str] | WeightMapping],
        safetensors_partition=1,
        dummy=False,
        use_lazy_loading: bool | None = None,
    ):
        """Load weights from safetensors files or generate dummy weights.

        Args:
            weight_mappings: Mapping from HF keys to model paths with sharding info
            safetensors_partition: Number of safetensors partitions
            dummy: If True, generate random weights instead of loading from files
            use_lazy_loading: If True, use lazy loading (load only needed slices per process).
                            If None, auto-enable for multi-process (jax.process_count() > 1)
        """
        params = nnx.state(self.model)

        # Dummy mode: generate random weights using mapping's sharding info
        # Can be explicitly passed or set via model_config._dummy_mode
        if dummy or self.dummy_mode:
            self._load_dummy_weights(params, weight_mappings)
            return

        # Auto-enable lazy loading for multi-process
        if use_lazy_loading is None:
            use_lazy_loading = jax.process_count() > 1

        if use_lazy_loading:
            logger.info("Using lazy weight loading (each process loads only its needed slices)")
            self._load_weights_lazy(params, weight_mappings, safetensors_partition)
            return

        # Original eager loading path
        regular_mappings = {}
        moe_mappings = {}

        for key, mapping in weight_mappings.items():
            if key.startswith("__MOE_EXPERTS__"):
                moe_mappings[key] = mapping
            else:
                regular_mappings[key] = mapping

        moe_buffer = {}

        logger.info(
            "WeightLoader: Will load layers 0 to %s",
            self.model_config.num_hidden_layers - 1,
        )

        for hf_key, hf_weight in self._iterate_weights():
            if hf_key in regular_mappings:
                if hf_key == "d2t":
                    base = jnp.arange(hf_weight.shape[0], dtype=hf_weight.dtype)
                    hot_ids = (hf_weight + base).astype(jnp.int32)
                    params["hot_token_ids"].value = hot_ids
                    continue
                mapping = regular_mappings[hf_key]
                if isinstance(mapping, (str, list)):
                    mapping = WeightMapping(target_path=mapping)
                self._process_and_assign_weight(params, hf_key, hf_weight, mapping)
            elif (
                "mlp.experts." in hf_key or "block_sparse_moe.experts" in hf_key
            ) and hf_key.endswith(".weight"):
                if self._is_excluded_layer_weight(hf_key):
                    logger.debug("Skipping excluded MoE expert weight: %s", hf_key)
                    continue

                assigned = False
                for moe_key, mapping in moe_mappings.items():
                    expected_hf_keys = mapping.target_path[1:]  # list of expected HF keys
                    if hf_key in expected_hf_keys:
                        if moe_key not in moe_buffer:
                            moe_buffer[moe_key] = {}
                        if hf_key not in moe_buffer[moe_key]:
                            moe_buffer[moe_key][hf_key] = []
                        moe_buffer[moe_key][hf_key].append(hf_weight)
                        assigned = True

                        if len(moe_buffer[moe_key]) == len(expected_hf_keys):
                            shard_counts = [len(v) for v in moe_buffer[moe_key].values()]
                            # Validate all weights have consistent shard counts
                            if len(set(shard_counts)) != 1:
                                continue

                            # Auto-detect TP sharding:
                            # - Grok-2: concat_axis is set, needs multiple shards (e.g., 8)
                            if mapping.concat_axis is not None:
                                # TP-sharded weights: need to collect all TP shards
                                # Expected number of shards = total model files / experts per file
                                if shard_counts[0] < safetensors_partition:
                                    # Still collecting shards, wait for more
                                    continue
                            else:
                                # Non-TP-sharded weights: expect exactly 1 copy per expert
                                if shard_counts[0] != 1:
                                    continue

                            self._process_single_moe_group(
                                params, moe_key, mapping, moe_buffer[moe_key]
                            )
                            del moe_buffer[moe_key]  # free memory
                        break

                if not assigned:
                    logger.warning("MoE expert weight not assigned to any mapping: %s", hf_key)
            else:
                if self._is_excluded_layer_weight(hf_key):
                    logger.debug("Skipping excluded layer weight: %s", hf_key)
                else:
                    logger.warning("No mapping found for weight: %s", hf_key)

        if moe_buffer:
            for moe_key in moe_buffer:
                mapping = moe_mappings[moe_key]
                expected = len(mapping.target_path[1:])
                got = len(moe_buffer[moe_key])
                shard_counts = (
                    [len(v) for v in moe_buffer[moe_key].values()] if moe_buffer[moe_key] else []
                )
                logger.error(
                    "MoE group %s incomplete: %s/%s weights loaded, shard_counts=%s, concat_axis=%s",
                    moe_key,
                    got,
                    expected,
                    shard_counts,
                    mapping.concat_axis,
                )
            raise RuntimeError("Incomplete MoE expert weights detected.")

        nnx.update(self.model, params)

    def _process_single_moe_group(
        self,
        params: nnx.State,
        moe_key: str,
        mapping: WeightMapping,
        expert_weights_dict: dict[str, list[jax.Array]],
    ):
        target_path = mapping.target_path[0]
        expected_hf_keys = mapping.target_path[1:]

        collected_weights = []
        for hf_key in expected_hf_keys:
            weights = expert_weights_dict[hf_key]
            # If TP-sharded (e.g., Grok-2), concatenate shards along concat_axis
            if mapping.concat_axis is not None and len(weights) > 1:
                weight = jnp.concatenate(weights, axis=mapping.concat_axis)
            else:
                # Non-TP-sharded (e.g., Qwen3-MoE), expect single weight
                weight = weights[0]

            if mapping.transpose and not hf_key.endswith(".bias"):
                weight = jnp.transpose(weight, (1, 0))
            collected_weights.append(weight)

        stacked_weight = jnp.stack(collected_weights, axis=0)  # (num_experts, ...)

        if "expert" in mapping.sharding:
            ep_size = getattr(self.model_config.hf_config, "ep_size", 1)
            world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get("tensor", 1)
            tp_size = world_size // ep_size

            devices = self.mesh.devices.flatten()
            moe_mesh = jax.sharding.Mesh(
                devices.reshape(ep_size, tp_size), axis_names=("expert", "tensor")
            )

            sharded_weight = self._shard_weight(stacked_weight, mapping.sharding, mesh=moe_mesh)
        else:
            sharded_weight = self._shard_weight(stacked_weight, mapping.sharding)

        model_param = self._get_param(params, target_path)
        model_param.value = sharded_weight.astype(model_param.value.dtype)

        logger.debug("Assigned MoE group %s, shape: %s", moe_key, stacked_weight.shape)

    def _load_dummy_weights(
        self,
        params: nnx.State,
        weight_mappings: dict[str, str | list[str] | WeightMapping],
        seed: int = 1234,
    ):
        logger.info("Generating dummy weights with proper sharding from weight mappings")
        # Separate regular and MOE weights
        regular_mappings = {}
        moe_mappings = {}

        for hf_key, mapping in weight_mappings.items():
            if hf_key.startswith("__MOE_EXPERTS__"):
                moe_mappings[hf_key] = mapping
            else:
                regular_mappings[hf_key] = mapping

        # Process regular weights
        for hf_key, mapping in regular_mappings.items():

            if isinstance(mapping, (str, list)):
                mapping = WeightMapping(target_path=mapping)

            target_path = (
                mapping.target_path
                if isinstance(mapping.target_path, str)
                else mapping.target_path[0]
            )

            try:
                model_param = self._get_param(params, target_path)
            except (KeyError, AttributeError, ValueError):
                logger.debug("Skip dummy weight for %s (parameter not found)", target_path)
                continue

            shape = model_param.value.shape
            dtype = model_param.value.dtype

            # Generate dummy weight with correct sharding
            sharding_spec = P(*mapping.sharding) if mapping.sharding else P()
            sharding = jax.sharding.NamedSharding(self.mesh, sharding_spec)

            def make_shard(indices, shape=shape, dtype=dtype):
                # Compute shard shape from global shape and indices
                shard_shape = []
                for dim_size, idx in zip(shape, indices):
                    if isinstance(idx, slice):
                        start, stop, step = idx.indices(dim_size)
                        assert step == 1, f"Non-unit step not supported: {idx}"
                        shard_shape.append(stop - start)
                    else:
                        shard_shape.append(1)
                shard_shape = tuple(shard_shape)

                # Generate random data
                rng = np.random.default_rng(seed)
                if jnp.issubdtype(dtype, jnp.floating):
                    if dtype == jnp.bfloat16:
                        gen_dtype = np.float32
                    else:
                        gen_dtype = {
                            jnp.float16: np.float16,
                            jnp.float32: np.float32,
                            jnp.float64: np.float64,
                        }.get(dtype, np.float32)
                    arr_np = rng.uniform(-1e-3, 1e-3, size=shard_shape).astype(gen_dtype)
                    return jnp.asarray(arr_np, dtype=dtype)
                else:
                    # Non-floating types, just zeros
                    return jnp.zeros(shard_shape, dtype=dtype)

            dummy_weight = jax.make_array_from_callback(shape, sharding, make_shard)
            model_param.value = dummy_weight
            logger.debug(
                "Generated dummy weight for %s, shape=%s, sharding=%s",
                target_path,
                shape,
                sharding_spec,
            )

        # Process MOE weights
        for moe_key, mapping in moe_mappings.items():
            if isinstance(mapping, (str, list)):
                mapping = WeightMapping(target_path=mapping)

            target_path = mapping.target_path[0]

            try:
                model_param = self._get_param(params, target_path)
            except (KeyError, AttributeError, ValueError):
                logger.debug("Skip dummy MOE weight for %s (parameter not found)", target_path)
                continue

            # Expected shape: (num_experts, ...)
            full_shape = model_param.value.shape
            num_experts = full_shape[0]
            expert_weight_shape = full_shape[1:]
            dtype = model_param.value.dtype

            # Generate dummy weights for all experts
            collected_weights = []
            for expert_idx in range(num_experts):
                # For each expert weight, generate with appropriate sharding
                # Remove "expert" axis from sharding for individual expert weight generation
                if mapping.sharding and "expert" in mapping.sharding:
                    # Expert-parallel sharding: use tensor-only sharding for generation
                    expert_sharding_tuple = tuple(s for s in mapping.sharding if s != "expert")
                else:
                    expert_sharding_tuple = mapping.sharding

                expert_sharding_spec = P(*expert_sharding_tuple) if expert_sharding_tuple else P()
                expert_sharding = jax.sharding.NamedSharding(self.mesh, expert_sharding_spec)

                def make_expert_shard(
                    indices, weight_shape=expert_weight_shape, weight_dtype=dtype, idx=expert_idx
                ):
                    shard_shape = []
                    for dim_size, idx_val in zip(weight_shape, indices):
                        if isinstance(idx_val, slice):
                            start, stop, step = idx_val.indices(dim_size)
                            assert step == 1, f"Non-unit step not supported: {idx_val}"
                            shard_shape.append(stop - start)
                        else:
                            shard_shape.append(1)
                    shard_shape = tuple(shard_shape)

                    rng = np.random.default_rng(seed + idx)
                    if jnp.issubdtype(weight_dtype, jnp.floating):
                        gen_dtype = np.float32 if weight_dtype == jnp.bfloat16 else weight_dtype
                        arr_np = rng.uniform(-1e-3, 1e-3, size=shard_shape).astype(gen_dtype)
                        return jnp.asarray(arr_np, dtype=weight_dtype)
                    else:
                        return jnp.zeros(shard_shape, dtype=weight_dtype)

                expert_weight = jax.make_array_from_callback(
                    expert_weight_shape, expert_sharding, make_expert_shard
                )
                collected_weights.append(expert_weight)

            # Stack all expert weights: (num_experts, ...)
            stacked_weight = jnp.stack(collected_weights, axis=0)

            # Apply final sharding with expert axis if needed
            if mapping.sharding and "expert" in mapping.sharding:
                # Use MOE mesh with expert parallelism
                ep_size = getattr(self.model_config.hf_config, "ep_size", 1)
                if ep_size > 1:
                    world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get("tensor", 1)
                    tp_size = world_size // ep_size

                    devices = self.mesh.devices.flatten()
                    moe_mesh = jax.sharding.Mesh(
                        devices.reshape(ep_size, tp_size), axis_names=("expert", "tensor")
                    )
                    final_sharding_spec = P(*mapping.sharding)
                    final_sharding = jax.sharding.NamedSharding(moe_mesh, final_sharding_spec)
                else:
                    # No expert parallelism, use regular mesh
                    final_sharding_spec = P(*mapping.sharding)
                    final_sharding = jax.sharding.NamedSharding(self.mesh, final_sharding_spec)
            else:
                final_sharding_spec = P(*mapping.sharding) if mapping.sharding else P()
                final_sharding = jax.sharding.NamedSharding(self.mesh, final_sharding_spec)

            # Reshard to final sharding
            sharded_weight = jax.device_put(stacked_weight, final_sharding)
            model_param.value = sharded_weight.astype(dtype)

            logger.debug(
                "Generated dummy MOE weight for %s, shape=%s, num_experts=%s, sharding=%s",
                target_path,
                full_shape,
                num_experts,
                mapping.sharding,
            )

        nnx.update(self.model, params)
        logger.info("Dummy weights generated successfully!")

    def _iterate_weights(self):
        model_path = self.model_config.model_path
        weights_files = glob.glob(os.path.join(model_path, "*.safetensors"))

        if len(weights_files) == 0:
            raise RuntimeError(f"Cannot find any *.safetensors files in {model_path}")

        weights_files.sort()

        skipped_files = 0
        with tqdm(weights_files, desc="[LOADING] MODEL WEIGHTS", unit="file") as pbar:
            for st_file in pbar:
                filename = os.path.basename(st_file)
                pbar.set_postfix({"file": filename})

                with (
                    jax.default_device(jax.local_devices(backend="cpu")[0]),
                    safe_open(st_file, framework="flax") as f,
                ):
                    needed_keys = []
                    for name in f.keys():  # noqa: SIM118
                        if not name.startswith("model.layers."):
                            needed_keys.append(name)
                            continue

                        if not self._is_excluded_layer_weight(name):
                            needed_keys.append(name)

                    if not needed_keys:
                        skipped_files += 1
                        logger.debug(
                            "Skipping %s: 0/%s weights needed",
                            filename,
                            len(f.keys()),
                        )
                        continue

                    logger.debug(
                        "Loading %s: %s/%s weights needed",
                        filename,
                        len(needed_keys),
                        len(f.keys()),
                    )
                    for name in needed_keys:
                        weight_tensor = f.get_tensor(name)
                        yield name, weight_tensor

        if skipped_files > 0:
            logger.info(
                "Memory optimization: Skipped %s/%s files with no needed weights",
                skipped_files,
                len(weights_files),
            )

    def _process_and_assign_weight(
        self,
        params: nnx.State,
        hf_key: str,
        hf_weight: jax.Array,
        mapping: WeightMapping,
    ):
        processed_weight = hf_weight

        if mapping.transpose and not hf_key.endswith(".bias"):
            processed_weight = jnp.transpose(processed_weight, (1, 0))

        if isinstance(mapping.target_path, list):
            self._handle_split_weight(params, hf_key, processed_weight, mapping)
        else:
            self._handle_single_weight(params, hf_key, processed_weight, mapping)

    def _handle_single_weight(
        self, params: nnx.State, hf_key: str, weight: jax.Array, mapping: WeightMapping
    ):
        jax_path = mapping.target_path
        processed_weight = weight

        # Apply output_multiplier_scale to lm_head weights (matching PyTorch implementation)
        if "lm_head" in hf_key and hasattr(self.model_config.hf_config, "output_multiplier_scale"):
            logger.info(
                "Applying output_multiplier_scale (%.2f) to %s",
                self.model_config.hf_config.output_multiplier_scale,
                hf_key,
            )
            processed_weight = processed_weight.astype(jnp.float32)
            processed_weight = (
                processed_weight * self.model_config.hf_config.output_multiplier_scale
            )

        if mapping.reshape is not None:
            processed_weight = jnp.reshape(processed_weight, mapping.reshape)

        if mapping.kv_head_padding:
            processed_weight = self._apply_kv_head_padding(processed_weight, hf_key)

        sharded_weight = self._shard_weight(processed_weight, mapping.sharding)

        try:
            model_param = self._get_param(params, jax_path)
            logger.debug(
                "Loading %s -> %s, shape: %s, transpose: %s",
                hf_key,
                jax_path,
                processed_weight.shape,
                mapping.transpose,
            )
            model_param.value = sharded_weight.astype(model_param.value.dtype)
        except Exception as e:
            logger.error("Failed to load %s -> %s: %s", hf_key, jax_path, str(e))
            raise

    def _handle_split_weight(
        self, params: nnx.State, hf_key: str, weight: jax.Array, mapping: WeightMapping
    ):
        self._split_qkv_weight(params, hf_key, weight, mapping)

    def _split_qkv_weight(
        self, params: nnx.State, hf_key: str, weight: jax.Array, mapping: WeightMapping
    ):
        jax_paths = mapping.target_path

        if hf_key.endswith(".bias"):
            q_dim = self.num_heads * self.head_dim_original
            kv_dim = self.num_kv_heads * self.head_dim_original

            q_bias = weight[:q_dim]
            k_bias = weight[q_dim : q_dim + kv_dim]
            v_bias = weight[q_dim + kv_dim : q_dim + 2 * kv_dim]

            if mapping.head_dim_padding and self.head_dim_pad > 0:
                q_bias = jnp.reshape(q_bias, (self.num_heads, self.head_dim_original))
                q_bias = jnp.pad(q_bias, ((0, 0), (0, self.head_dim_pad)))
                q_bias = jnp.reshape(q_bias, (self.num_heads * self.head_dim,))

                k_bias = jnp.reshape(k_bias, (self.num_kv_heads, self.head_dim_original))
                k_bias = jnp.pad(k_bias, ((0, 0), (0, self.head_dim_pad)))
                k_bias = jnp.reshape(k_bias, (self.num_kv_heads * self.head_dim,))

                v_bias = jnp.reshape(v_bias, (self.num_kv_heads, self.head_dim_original))
                v_bias = jnp.pad(v_bias, ((0, 0), (0, self.head_dim_pad)))
                v_bias = jnp.reshape(v_bias, (self.num_kv_heads * self.head_dim,))

            splits = [q_bias, k_bias, v_bias]
        else:
            q_dim = self.num_heads * self.head_dim_original
            kv_dim = self.num_kv_heads * self.head_dim_original

            if mapping.transpose:
                q_weight = weight[:, :q_dim]
                k_weight = weight[:, q_dim : q_dim + kv_dim]
                v_weight = weight[:, q_dim + kv_dim : q_dim + 2 * kv_dim]
            else:
                q_weight = weight[:q_dim, :]
                k_weight = weight[q_dim : q_dim + kv_dim, :]
                v_weight = weight[q_dim + kv_dim : q_dim + 2 * kv_dim, :]

            if mapping.head_dim_padding and self.head_dim_pad > 0:
                if mapping.transpose:
                    q_weight = jnp.reshape(
                        q_weight,
                        (self.hidden_size, self.num_heads, self.head_dim_original),
                    )
                    q_weight = jnp.pad(q_weight, ((0, 0), (0, 0), (0, self.head_dim_pad)))
                    q_weight = jnp.reshape(
                        q_weight, (self.hidden_size, self.num_heads * self.head_dim)
                    )

                    k_weight = jnp.reshape(
                        k_weight,
                        (self.hidden_size, self.num_kv_heads, self.head_dim_original),
                    )
                    k_weight = jnp.pad(k_weight, ((0, 0), (0, 0), (0, self.head_dim_pad)))
                    k_weight = jnp.reshape(
                        k_weight, (self.hidden_size, self.num_kv_heads * self.head_dim)
                    )

                    v_weight = jnp.reshape(
                        v_weight,
                        (self.hidden_size, self.num_kv_heads, self.head_dim_original),
                    )
                    v_weight = jnp.pad(v_weight, ((0, 0), (0, 0), (0, self.head_dim_pad)))
                    v_weight = jnp.reshape(
                        v_weight, (self.hidden_size, self.num_kv_heads * self.head_dim)
                    )
                else:
                    q_weight = jnp.reshape(
                        q_weight,
                        (self.num_heads, self.head_dim_original, self.hidden_size),
                    )
                    q_weight = jnp.pad(q_weight, ((0, 0), (0, self.head_dim_pad), (0, 0)))
                    q_weight = jnp.reshape(
                        q_weight, (self.num_heads * self.head_dim, self.hidden_size)
                    )

                    k_weight = jnp.reshape(
                        k_weight,
                        (self.num_kv_heads, self.head_dim_original, self.hidden_size),
                    )
                    k_weight = jnp.pad(k_weight, ((0, 0), (0, self.head_dim_pad), (0, 0)))
                    k_weight = jnp.reshape(
                        k_weight, (self.num_kv_heads * self.head_dim, self.hidden_size)
                    )

                    v_weight = jnp.reshape(
                        v_weight,
                        (self.num_kv_heads, self.head_dim_original, self.hidden_size),
                    )
                    v_weight = jnp.pad(v_weight, ((0, 0), (0, self.head_dim_pad), (0, 0)))
                    v_weight = jnp.reshape(
                        v_weight, (self.num_kv_heads * self.head_dim, self.hidden_size)
                    )

            splits = [q_weight, k_weight, v_weight]

        for split_weight, jax_path in zip(splits, jax_paths):
            processed_weight = split_weight

            if mapping.kv_head_padding and ("k_proj" in jax_path or "v_proj" in jax_path):
                processed_weight = self._apply_kv_head_padding(processed_weight, jax_path)

            sharded_weight = self._shard_weight(processed_weight, mapping.sharding)

            model_param = self._get_param(params, jax_path)
            model_param.value = sharded_weight.astype(model_param.value.dtype)
            logger.debug("Split %s -> %s, shape: %s", hf_key, jax_path, processed_weight.shape)

    def _shard_weight(
        self, weight: jax.Array, sharding_spec: tuple, mesh: jax.sharding.Mesh = None
    ) -> jax.Array:
        if mesh is None:
            mesh = self.mesh
        target_sharding = jax.sharding.NamedSharding(mesh, P(*sharding_spec))

        if (
            getattr(weight, "_committed", False)
            and getattr(weight, "sharding", None) == target_sharding
        ):
            return weight

        if jax.process_count() > 1:

            def make_shard(indices):
                shard = weight[indices]
                return shard

            return jax.make_array_from_callback(
                shape=weight.shape, sharding=target_sharding, data_callback=make_shard
            )
        else:
            return jax.device_put(weight, target_sharding)

    def _get_param(self, params: nnx.State, path: str) -> nnx.State:
        keys = path.split(".")
        current_level = params

        for i, key in enumerate(keys):
            if key.isdigit():
                current_level = current_level[int(key)]
            else:
                if hasattr(current_level, "__contains__") and key in current_level:
                    current_level = current_level[key]
                elif hasattr(current_level, key):
                    current_level = getattr(current_level, key)
                else:
                    # Debug: print available keys at this level
                    partial_path = ".".join(keys[: i + 1])
                    if hasattr(current_level, "__contains__"):
                        available = (
                            list(current_level.keys()) if hasattr(current_level, "keys") else "N/A"
                        )
                    elif hasattr(current_level, "__dict__"):
                        available = list(current_level.__dict__.keys())
                    else:
                        available = dir(current_level)
                    logger.error(
                        "Path %s not found. At %s, available keys: %s",
                        path,
                        partial_path,
                        available[:10] if isinstance(available, list) else available,
                    )
                    raise ValueError(f"{path} is not a valid param path")

        return current_level

    def _apply_head_dim_padding(
        self, weight: jax.Array, hf_key: str, mapping: WeightMapping
    ) -> jax.Array:
        if hf_key.endswith(".bias"):
            if any(proj in hf_key for proj in ["q_proj", "k_proj", "v_proj"]):
                if "q_proj" in hf_key:
                    reshaped = jnp.reshape(weight, (self.num_heads, self.head_dim_original))
                    padded = jnp.pad(reshaped, ((0, 0), (0, self.head_dim_pad)))
                    return jnp.reshape(padded, (self.num_heads * self.head_dim,))
                else:  # k_proj or v_proj
                    reshaped = jnp.reshape(weight, (self.num_kv_heads, self.head_dim_original))
                    padded = jnp.pad(reshaped, ((0, 0), (0, self.head_dim_pad)))
                    return jnp.reshape(padded, (self.num_kv_heads * self.head_dim,))
        else:
            if mapping.reshape is not None:
                if "o_proj" in hf_key:
                    padded = jnp.pad(weight, ((0, 0), (0, 0), (0, self.head_dim_pad)))
                else:
                    padded = jnp.pad(weight, ((0, 0), (0, self.head_dim_pad), (0, 0)))
                return padded
            else:
                if mapping.transpose:
                    if "q_proj" in hf_key:
                        reshaped = jnp.reshape(
                            weight,
                            (
                                self.hidden_size if not mapping.is_eagle3 else 2 * self.hidden_size,
                                self.num_heads,
                                self.head_dim_original,
                            ),
                        )
                        padded = jnp.pad(reshaped, ((0, 0), (0, 0), (0, self.head_dim_pad)))
                        return jnp.reshape(
                            padded,
                            (
                                self.hidden_size if not mapping.is_eagle3 else 2 * self.hidden_size,
                                self.num_heads * self.head_dim,
                            ),
                        )
                    elif any(proj in hf_key for proj in ["k_proj", "v_proj"]):
                        reshaped = jnp.reshape(
                            weight,
                            (
                                self.hidden_size if not mapping.is_eagle3 else 2 * self.hidden_size,
                                self.num_kv_heads,
                                self.head_dim_original,
                            ),
                        )
                        padded = jnp.pad(reshaped, ((0, 0), (0, 0), (0, self.head_dim_pad)))
                        return jnp.reshape(
                            padded,
                            (
                                self.hidden_size if not mapping.is_eagle3 else 2 * self.hidden_size,
                                self.num_kv_heads * self.head_dim,
                            ),
                        )
                    elif "o_proj" in hf_key:
                        reshaped = jnp.reshape(
                            weight,
                            (self.num_heads * self.head_dim_original, self.hidden_size),
                        )
                        padded_reshaped = jnp.reshape(
                            reshaped,
                            (self.num_heads, self.head_dim_original, self.hidden_size),
                        )
                        padded = jnp.pad(padded_reshaped, ((0, 0), (0, self.head_dim_pad), (0, 0)))
                        return jnp.reshape(
                            padded, (self.num_heads * self.head_dim, self.hidden_size)
                        )

        return weight

    def _apply_kv_head_padding(self, weight: jax.Array, hf_key: str) -> jax.Array:
        """Apply KV head padding/replication when tp_size > total_kv_heads."""
        if any(
            proj in hf_key for proj in ["k_proj", "v_proj"]
        ) and self.model_config.needs_kv_head_replication(self.sharding_size):
            total_kv_heads = self.model_config.get_total_num_kv_heads()
            num_replicas = self.model_config.get_num_kv_head_replicas(self.sharding_size)
            padding_strategy = self.model_config.get_kv_padding_strategy()

            if padding_strategy == "replicate":
                if hf_key.endswith(".bias"):
                    replicated_bias_parts = []
                    for original_head_id in range(total_kv_heads):
                        start_idx = original_head_id * self.head_dim
                        end_idx = (original_head_id + 1) * self.head_dim
                        original_head_bias = weight[start_idx:end_idx]
                        for _ in range(num_replicas):
                            replicated_bias_parts.append(original_head_bias)
                    weight = jnp.concatenate(replicated_bias_parts, axis=0)
                else:
                    replicated_weight_parts = []
                    for original_head_id in range(total_kv_heads):
                        start_idx = original_head_id * self.head_dim
                        end_idx = (original_head_id + 1) * self.head_dim
                        original_head_weight = weight[:, start_idx:end_idx]
                        for _ in range(num_replicas):
                            replicated_weight_parts.append(original_head_weight)
                    weight = jnp.concatenate(replicated_weight_parts, axis=1)
            elif padding_strategy == "zero":
                target_heads = total_kv_heads * num_replicas
                target_size = target_heads * self.head_dim
                if hf_key.endswith(".bias"):
                    current_size = weight.shape[0]
                    padding_size = target_size - current_size
                    if padding_size > 0:
                        padding = jnp.zeros((padding_size,), dtype=weight.dtype)
                        weight = jnp.concatenate([weight, padding], axis=0)
                else:
                    current_size = weight.shape[1]
                    padding_size = target_size - current_size
                    if padding_size > 0:
                        padding = jnp.zeros((weight.shape[0], padding_size), dtype=weight.dtype)
                        weight = jnp.concatenate([weight, padding], axis=1)
        return weight

    def _is_excluded_layer_weight(self, hf_key: str) -> bool:
        if not hf_key.startswith("model.layers."):
            return False

        parts = hf_key.split(".")
        if len(parts) < 3 or not parts[2].isdigit():
            return False

        layer_num = int(parts[2])
        return layer_num >= self.model_config.num_hidden_layers
