# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""UMT5 Model Alignment Tests - Optimized Version"""

import argparse
import contextlib
import gc
import logging
import os
import sys
import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from transformers import (
    AutoTokenizer,
    UMT5Config,
    UMT5EncoderModel as HFUMt5EncoderModel,
    UMT5ForConditionalGeneration as HFUMt5ForConditionalGeneration,
    UMT5Model as HFUMt5Model,
)

# Ensure we can import the project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.models.umt5 import (
    UMT5EncoderModel as JAXUMt5EncoderModel,
    UMT5ForConditionalGeneration as JAXUMt5ForConditionalGeneration,
    UMT5Model as JAXUMt5Model,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_torch_dtype(precision: str):
    """Convert precision string to PyTorch dtype"""
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return dtype_map[precision]


def get_jax_dtype(precision: str):
    """Convert precision string to JAX dtype"""
    dtype_map = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }
    return dtype_map[precision]


class CompilationProgressTracker:
    """Track and display JAX compilation progress"""
    
    def __init__(self, description="Compiling JAX model"):
        self.description = description
        self.start_time = None
        self.stop_flag = False
        self.thread = None
    
    def __enter__(self):
        import threading
        import time
        
        self.start_time = time.time()
        self.stop_flag = False
        
        def progress_updater():
            elapsed = 0
            while not self.stop_flag:
                time.sleep(10)  # Update every 10 seconds
                if not self.stop_flag:
                    elapsed = int(time.time() - self.start_time)
                    logger.info(f"⏳ {self.description}... {elapsed}s elapsed")
        
        self.thread = threading.Thread(target=progress_updater, daemon=True)
        logger.info(f"🔧 {self.description}... (this may take 1-3 minutes for large models)")
        self.thread.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=1)
        
        elapsed = time.time() - self.start_time
        logger.info(f"✅ Compilation complete! Took {elapsed:.1f}s")
        return False


def compare_outputs(
    pt_output, jax_output, name="Output", threshold=None
) -> Tuple[float, float, float]:
    """
    Compare PyTorch and JAX outputs with detailed statistics.
    
    Args:
        pt_output: PyTorch tensor or numpy array
        jax_output: JAX array
        name: Name for logging
        threshold: Optional threshold for pass/fail (for logging)
        
    Returns:
        Tuple of (MAE, max_diff, relative_error)
    """
    # Convert PyTorch tensor to numpy (handle low-precision types by converting to float32 first)
    if torch.is_tensor(pt_output):
        if pt_output.dtype in (torch.bfloat16, torch.float16):
            # Convert to float32 for NumPy compatibility and better precision in comparison
            pt_np = pt_output.detach().cpu().float().numpy()
        else:
            pt_np = pt_output.detach().cpu().numpy()
    else:
        pt_np = pt_output
    
    # Convert JAX array to numpy (handle low-precision types similarly)
    if jax_output.dtype in (jnp.bfloat16, jnp.float16):
        jax_np = np.array(jax_output, dtype=np.float32)
    else:
        jax_np = np.array(jax_output)

    diff = np.abs(pt_np - jax_np)
    mae = np.mean(diff)
    max_diff = np.max(diff)
    relative_error = mae / (np.abs(pt_np).mean() + 1e-10)

    status = ""
    if threshold is not None:
        status = "✅" if mae < threshold else "❌"

    logger.info(f"\n{status} {name}:")
    logger.info(f"  Shape: PT={pt_np.shape}, JAX={jax_np.shape}")
    logger.info(f"  MAE: {mae:.10e}")
    logger.info(f"  Max Diff: {max_diff:.10e}")
    logger.info(f"  Relative Error: {relative_error:.10e}")

    return mae, max_diff, relative_error


def load_weights_from_hf(jax_model, hf_model, model_name, mesh=None, precision="float32"):
    """
    Load weights using the WeightLoader with enhanced error handling and mesh support.
    
    Args:
        jax_model: The JAX model to load weights into
        hf_model: The HuggingFace PyTorch model
        model_name: Model path/name
        mesh: JAX mesh for sharded weight loading (required for tensor parallelism)
        precision: Precision/dtype for the model ("float32", "bfloat16", "float16")
    
    Returns:
        bool: True if weights loaded successfully
    """
    logger.info(f"Loading weights from HuggingFace model (precision={precision})...")

    # Create ModelConfig
    model_config = ModelConfig(
        model_path=model_name, trust_remote_code=True, dtype=precision
    )

    # Try the built-in load_weights method first
    try:
        jax_model.load_weights(model_config=model_config)
        logger.info("✅ Weights loaded successfully via WeightLoader")
        return True
    except Exception as e:
        logger.warning(f"⚠️ load_weights failed: {e}")
        logger.info("Attempting manual weight copying from PyTorch model...")

        # Fallback: manually copy weights from the provided hf_model
        return _manual_weight_loading(jax_model, hf_model, mesh)


def _manual_weight_loading(jax_model, hf_model, mesh=None):
    """
    Manual weight loading fallback with memory optimization and mesh support.
    """
    pt_state = hf_model.state_dict()

    # Get weight mappings
    if hasattr(jax_model, "_create_umt5_weight_mappings"):
        mappings_dict = jax_model._create_umt5_weight_mappings()
    elif hasattr(jax_model, "_create_weight_mappings"):
        mappings_dict = jax_model._create_weight_mappings()
    else:
        logger.error("No weight mapping method found!")
        return False

    # Log mapping statistics
    logger.info(f"📊 Total JAX mappings created: {len(mappings_dict)}")
    logger.info(f"📊 Total PyTorch weights: {len(pt_state)}")

    # Check for missing keys (JAX expects but PyTorch doesn't have)
    missing_keys = [key for key in mappings_dict if key not in pt_state]
    if missing_keys:
        logger.warning(f"⚠️ Missing keys in PyTorch state_dict ({len(missing_keys)}):")
        for key in missing_keys[:5]:  # Show first 5
            logger.warning(f"  ❌ {key}")
        if len(missing_keys) > 5:
            logger.warning(f"  ... and {len(missing_keys) - 5} more")

    # Check for unmapped keys (PyTorch has but JAX doesn't map)
    unmapped_keys = [key for key in pt_state.keys() if key not in mappings_dict]
    if unmapped_keys:
        logger.warning(f"⚠️ Unmapped PyTorch weights (not used in JAX model): {len(unmapped_keys)}")
        for key in unmapped_keys[:10]:  # Show first 10
            logger.warning(f"  ⚙️ {key}")
        if len(unmapped_keys) > 10:
            logger.warning(f"  ... and {len(unmapped_keys) - 10} more")

    # Separate embeddings from other weights for memory efficiency
    embedding_keys = [k for k in mappings_dict.keys() if "shared.weight" in k or "embed_tokens.weight" in k]
    other_keys = [k for k in mappings_dict.keys() if k not in embedding_keys]

    logger.info(
        f"📦 Loading order: {len(embedding_keys)} embeddings first, then {len(other_keys)} other weights"
    )

    # Clean up memory before loading large embeddings
    logger.info("🧹 Cleaning up memory before loading embeddings...")
    for key in embedding_keys:
        if key in pt_state:
            del pt_state[key]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("  ✓ Memory cleaned")

    # Reload full state dict for embeddings
    pt_state_full = hf_model.state_dict()

    # IMPORTANT: Enter mesh context for sharded weight loading
    mesh_context = jax.set_mesh(mesh) if mesh is not None else contextlib.nullcontext()

    loaded_count = 0
    failed_keys = []

    with mesh_context:
        # Load embeddings first (when memory is fresh)
        for hf_key in embedding_keys:
            if hf_key not in pt_state_full:
                continue

            mapping = mappings_dict[hf_key]
            try:
                logger.info(f"🔄 Loading large embedding: {hf_key}")
                weight_tensor = pt_state_full[hf_key].detach().cpu()
                # Convert low-precision types (bfloat16, float16) to float32 for numpy compatibility
                if weight_tensor.dtype in (torch.bfloat16, torch.float16):
                    pt_weight = weight_tensor.float().numpy()
                else:
                    pt_weight = weight_tensor.numpy()
                if mapping.transpose:
                    pt_weight = pt_weight.T

                # Navigate to target parameter
                if _set_parameter(jax_model, mapping.target_path, pt_weight):
                    loaded_count += 1
                    logger.info(f"  ✓ Loaded successfully")
                else:
                    failed_keys.append((hf_key, "Navigation failed"))

            except Exception as e:
                logger.error(f"❌ CRITICAL: Failed to load embedding {hf_key}: {e}")
                failed_keys.append((hf_key, str(e)))

        # Load other weights
        for hf_key in other_keys:
            if hf_key not in pt_state:
                continue

            mapping = mappings_dict[hf_key]
            try:
                weight_tensor = pt_state[hf_key].detach().cpu()
                # Convert low-precision types (bfloat16, float16) to float32 for numpy compatibility
                if weight_tensor.dtype in (torch.bfloat16, torch.float16):
                    pt_weight = weight_tensor.float().numpy()
                else:
                    pt_weight = weight_tensor.numpy()
                if mapping.transpose:
                    pt_weight = pt_weight.T

                if _set_parameter(jax_model, mapping.target_path, pt_weight):
                    loaded_count += 1
                else:
                    failed_keys.append((hf_key, "Navigation failed"))

            except Exception as e:
                logger.debug(f"⚠️ Failed to load {hf_key}: {e}")
                failed_keys.append((hf_key, str(e)))

    logger.info(f"✅ Manually loaded {loaded_count}/{len(mappings_dict)} weights")

    if failed_keys:
        logger.error(f"❌ Failed to load {len(failed_keys)} weights:")
        for key, error in failed_keys[:5]:  # Show first 5
            logger.error(f"  - {key}: {error}")
        if len(failed_keys) > 5:
            logger.error(f"  ... and {len(failed_keys) - 5} more")

    # No need to share relative_attention_bias - each layer loads its own weights
    # (Unlike the old implementation where only layer 0 had the weights)

    return True


def _set_parameter(model, target_path: str, value: np.ndarray) -> bool:
    """
    Navigate to target parameter and set its value.
    
    Returns:
        bool: True if successful
    """
    try:
        target_path_parts = target_path.split(".")
        current = model

        # Navigate to parent
        for i, part in enumerate(target_path_parts[:-1]):
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)

        # Set the final attribute
        final_attr = target_path_parts[-1]
        if hasattr(current, final_attr):
            param = getattr(current, final_attr)
            if isinstance(param, nnx.Variable):
                # Use the parameter's existing dtype to avoid type mismatch warnings
                # Use param[...] instead of param.value (new nnx API)
                try:
                    target_dtype = param[...].dtype
                except (AttributeError, TypeError):
                    target_dtype = jnp.float32
                param[...] = jnp.array(value, dtype=target_dtype)
            else:
                # For new parameters, use float32 as default
                setattr(current, final_attr, nnx.Variable(jnp.array(value, dtype=jnp.float32)))
            return True
        return False
    except Exception as e:
        logger.debug(f"Failed to set parameter {target_path}: {e}")
        return False


def _share_relative_attention_bias(jax_model):
    """
    Share relative_attention_bias from layer 0 to other layers.
    This handles T5/UMT5's shared bias design where PyTorch only stores bias in block.0.
    """
    if hasattr(jax_model, "encoder") and hasattr(
        jax_model.encoder, "share_relative_attention_bias"
    ):
        logger.info(
            "🔄 Sharing encoder relative_attention_bias from layer 0 to other layers..."
        )

        # Debug: Check layer 0 bias before sharing
        if len(jax_model.encoder.block) > 1:
            layer0_bias = jax_model.encoder.block[0].layer0_SelfAttention.relative_attention_bias.embedding[...]
            layer1_bias_before = jax_model.encoder.block[1].layer0_SelfAttention.relative_attention_bias.embedding[...]
            logger.debug(
                f"  Layer 0 bias shape: {layer0_bias.shape}, mean: {jnp.mean(jnp.abs(layer0_bias)):.6f}"
            )
            logger.debug(
                f"  Layer 1 bias before (shape: {layer1_bias_before.shape}, mean: {jnp.mean(jnp.abs(layer1_bias_before)):.6f})"
            )

        jax_model.encoder.share_relative_attention_bias()

        # Debug: Check layer 1 bias after sharing
        if len(jax_model.encoder.block) > 1:
            layer1_bias_after = jax_model.encoder.block[1].layer0_SelfAttention.relative_attention_bias.embedding[...]
            logger.debug(
                f"  Layer 1 bias after  (shape: {layer1_bias_after.shape}, mean: {jnp.mean(jnp.abs(layer1_bias_after)):.6f})"
            )
            bias_match = jnp.allclose(layer0_bias, layer1_bias_after, atol=1e-6)
            logger.info(f"  ✓ Bias shared correctly: {bias_match}")

    if hasattr(jax_model, "decoder") and hasattr(
        jax_model.decoder, "share_relative_attention_bias"
    ):
        logger.info(
            "🔄 Sharing decoder relative_attention_bias from layer 0 to other layers..."
        )
        jax_model.decoder.share_relative_attention_bias()


def test_encoder_alignment(model_name, mesh, tokenizer, precision="float32"):
    """Test UMT5EncoderModel alignment"""
    logger.info("\n" + "=" * 80)
    logger.info("Testing UMT5EncoderModel Alignment")
    logger.info("=" * 80)

    # Load HF Encoder
    hf_config = UMT5Config.from_pretrained(model_name)
    hf_full_model = HFUMt5Model.from_pretrained(
        model_name, attn_implementation="eager", dtype=get_torch_dtype(precision)
    )
    hf_full_model.eval()

    # Create JAX Encoder
    with jax.set_mesh(mesh):
        jax_encoder = JAXUMt5EncoderModel(
            config=hf_config,
            mesh=mesh,
            dtype=get_jax_dtype(precision),
        )

    # Load weights
    load_weights_from_hf(jax_encoder, hf_full_model, model_name, mesh=mesh, precision=precision)

    # Create test input
    text = "Translate English to French: Hello, how are you today?"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    pt_input_ids = inputs.input_ids
    pt_attention_mask = inputs.attention_mask

    jax_input_ids = jnp.array(pt_input_ids.numpy())
    jax_attention_mask = jnp.array(pt_attention_mask.numpy())

    # Run inference
    with torch.no_grad():
        pt_encoder_output = hf_full_model.encoder(
            input_ids=pt_input_ids, attention_mask=pt_attention_mask
        ).last_hidden_state

    with CompilationProgressTracker("Compiling JAX encoder"):
        with jax.set_mesh(mesh):
            jax_encoder_output = jax_encoder(
                input_ids=jax_input_ids,
                attention_mask=jax_attention_mask,
                deterministic=True,
            )

    # Compare
    mae, max_diff, rel_err = compare_outputs(
        pt_encoder_output, jax_encoder_output, "Encoder Output", threshold=1e-4
    )

    passed = mae < 1e-4
    if passed:
        logger.info("✅ Encoder alignment PASSED (MAE < 1e-4)")
    else:
        logger.error(f"❌ Encoder alignment FAILED (MAE = {mae:.10e})")

    return passed, mae


def test_full_model_alignment(model_name, mesh, tokenizer, precision="float32"):
    """Test UMT5Model (Full Encoder-Decoder) alignment"""
    logger.info("\n" + "=" * 80)
    logger.info("Testing UMT5Model (Full Encoder-Decoder) Alignment")
    logger.info("=" * 80)

    # Load HF Model
    hf_config = UMT5Config.from_pretrained(model_name)
    hf_model = HFUMt5Model.from_pretrained(
        model_name, attn_implementation="eager", dtype=get_torch_dtype(precision)
    )
    hf_model.eval()

    # Create JAX Model
    with jax.set_mesh(mesh):
        jax_model = JAXUMt5Model(
            config=hf_config,
            mesh=mesh,
            dtype=get_jax_dtype(precision),
        )

    # Load weights
    load_weights_from_hf(jax_model, hf_model, model_name, mesh=mesh, precision=precision)

    # Create test inputs
    text = "Translate English to French: The weather is nice."
    target_text = "Le temps est agréable."

    inputs = tokenizer(text, return_tensors="pt")
    targets = tokenizer(target_text, return_tensors="pt")

    pt_input_ids = inputs.input_ids
    pt_attention_mask = inputs.attention_mask
    pt_decoder_input_ids = targets.input_ids
    pt_decoder_attention_mask = targets.attention_mask

    jax_input_ids = jnp.array(pt_input_ids.numpy())
    jax_attention_mask = jnp.array(pt_attention_mask.numpy())
    jax_decoder_input_ids = jnp.array(pt_decoder_input_ids.numpy())
    jax_decoder_attention_mask = jnp.array(pt_decoder_attention_mask.numpy())

    # Run inference
    with torch.no_grad():
        pt_outputs = hf_model(
            input_ids=pt_input_ids,
            attention_mask=pt_attention_mask,
            decoder_input_ids=pt_decoder_input_ids,
            decoder_attention_mask=pt_decoder_attention_mask,
        )
        pt_last_hidden = pt_outputs.last_hidden_state

    with CompilationProgressTracker("Compiling JAX encoder-decoder model"):
        with jax.set_mesh(mesh):
            jax_last_hidden = jax_model(
                input_ids=jax_input_ids,
                decoder_input_ids=jax_decoder_input_ids,
                attention_mask=jax_attention_mask,
                decoder_attention_mask=jax_decoder_attention_mask,
                deterministic=True,
            )

    # Compare final output
    mae, max_diff, rel_err = compare_outputs(
        pt_last_hidden, jax_last_hidden, "Full Model Output", threshold=1e-4
    )

    # Layer-by-layer debugging if needed
    if mae > 1e-5:
        logger.info("\n=== Layer-by-Layer Debugging ===")

        # Check embeddings
        with torch.no_grad():
            pt_enc_embeds = hf_model.shared(pt_input_ids)
        jax_enc_embeds = jax_model.shared(jax_input_ids)
        compare_outputs(pt_enc_embeds, jax_enc_embeds, "Encoder Embeddings")

        # Check encoder output
        with torch.no_grad():
            pt_enc_output = hf_model.encoder(
                pt_input_ids, attention_mask=pt_attention_mask
            ).last_hidden_state
        with jax.set_mesh(mesh):
            jax_enc_output = jax_model.encoder(
                jax_model.shared(jax_input_ids), mask=jax_attention_mask, deterministic=True
            )
        compare_outputs(pt_enc_output, jax_enc_output, "Encoder Final Output")

        # Check decoder embeddings
        with torch.no_grad():
            pt_dec_embeds = hf_model.shared(pt_decoder_input_ids)
        jax_dec_embeds = jax_model.shared(jax_decoder_input_ids)
        compare_outputs(pt_dec_embeds, jax_dec_embeds, "Decoder Embeddings")

    passed = mae < 1e-4
    if passed:
        logger.info("✅ Full Model alignment PASSED (MAE < 1e-4)")
    elif mae < 1e-3:
        logger.info("⚠️ Full Model alignment ACCEPTABLE (MAE < 1e-3)")
        passed = True
    else:
        logger.error(f"❌ Full Model alignment FAILED (MAE = {mae:.10e})")

    return passed, mae


def test_generation_model_alignment(model_name, mesh, tokenizer, precision="float32"):
    """Test UMT5ForConditionalGeneration alignment (with LM Head)"""
    logger.info("\n" + "=" * 80)
    logger.info("Testing UMT5ForConditionalGeneration Alignment")
    logger.info("=" * 80)

    # Load HF Model
    hf_config = UMT5Config.from_pretrained(model_name)
    hf_model = HFUMt5ForConditionalGeneration.from_pretrained(
        model_name, attn_implementation="eager", dtype=get_torch_dtype(precision)
    )
    hf_model.eval()

    # Create JAX Model
    with jax.set_mesh(mesh):
        jax_model = JAXUMt5ForConditionalGeneration(
            config=hf_config,
            mesh=mesh,
            dtype=get_jax_dtype(precision),
        )

    # Load weights
    load_weights_from_hf(jax_model, hf_model, model_name, mesh=mesh, precision=precision)

    # Create test inputs
    text = "Translate English to German: Thank you for your help."
    target_text = "Vielen Dank für Ihre Hilfe."

    inputs = tokenizer(text, return_tensors="pt")
    targets = tokenizer(target_text, return_tensors="pt")

    pt_input_ids = inputs.input_ids
    pt_attention_mask = inputs.attention_mask
    pt_decoder_input_ids = targets.input_ids
    pt_decoder_attention_mask = targets.attention_mask

    jax_input_ids = jnp.array(pt_input_ids.numpy())
    jax_attention_mask = jnp.array(pt_attention_mask.numpy())
    jax_decoder_input_ids = jnp.array(pt_decoder_input_ids.numpy())
    jax_decoder_attention_mask = jnp.array(pt_decoder_attention_mask.numpy())

    # Run inference
    with torch.no_grad():
        pt_outputs = hf_model(
            input_ids=pt_input_ids,
            attention_mask=pt_attention_mask,
            decoder_input_ids=pt_decoder_input_ids,
            decoder_attention_mask=pt_decoder_attention_mask,
        )
    pt_logits = pt_outputs.logits

    with CompilationProgressTracker("Compiling JAX generation model with LM head"):
        with jax.set_mesh(mesh):
            jax_logits = jax_model(
                input_ids=jax_input_ids,
                decoder_input_ids=jax_decoder_input_ids,
                attention_mask=jax_attention_mask,
                decoder_attention_mask=jax_decoder_attention_mask,
                deterministic=True,
            )

    # Compare logits
    mae, max_diff, rel_err = compare_outputs(
        pt_logits, jax_logits, "Generation Model Logits", threshold=1e-3
    )

    # Check top-k predictions
    logger.info("\n=== Token Prediction Comparison ===")
    pt_pred_ids = pt_logits.argmax(dim=-1)[0].cpu().numpy()
    jax_pred_ids = jax_logits.argmax(axis=-1)[0]

    matches = (pt_pred_ids == np.array(jax_pred_ids)).sum()
    total = len(pt_pred_ids)
    accuracy = matches / total

    logger.info(f"Token prediction accuracy: {matches}/{total} ({accuracy*100:.2f}%)")
    logger.info(f"PT predictions:  {pt_pred_ids[:10]}")
    logger.info(f"JAX predictions: {np.array(jax_pred_ids[:10])}")

    passed = mae < 1e-3 and accuracy > 0.95
    if passed:
        logger.info("✅ Generation Model alignment PASSED")
    else:
        logger.error(
            f"❌ Generation Model alignment FAILED (MAE={mae:.10e}, Acc={accuracy:.2%})"
        )

    return passed, mae


def test_end_to_end_translation(model_name: str, mesh: jax.sharding.Mesh, tokenizer, precision="float32"):
    """
    End-to-end translation test with actual text generation.
    Compares PyTorch and JAX generated translations.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Testing End-to-End Translation")
    logger.info("=" * 80)

    # Load models
    hf_config = UMT5Config.from_pretrained(model_name)
    hf_model = HFUMt5ForConditionalGeneration.from_pretrained(
        model_name, attn_implementation="eager", dtype=get_torch_dtype(precision)
    )
    hf_model.eval()

    with jax.set_mesh(mesh):
        jax_model = JAXUMt5ForConditionalGeneration(hf_config, mesh, dtype=get_jax_dtype(precision))

    # Load weights
    load_weights_from_hf(jax_model, hf_model, model_name, mesh=mesh, precision=precision)

    # Test cases
    test_cases = [
        "Translate English to French: Hello, how are you?",
        "Translate English to French: The weather is nice today.",
        "Translate English to German: I love machine learning.",
        "Translate French to English: Bonjour le monde.",
    ]

    logger.info(f"\nTesting {len(test_cases)} translation examples:")

    all_match = True

    for idx, source_text in enumerate(test_cases, 1):
        logger.info(f"\n{'-'*80}")
        logger.info(f"Test Case {idx}/{len(test_cases)}")
        logger.info(f'Input: "{source_text}"')

        # Tokenize input
        inputs = tokenizer(source_text, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # PyTorch generation
        with torch.no_grad():
            pt_outputs = hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=1,  # Greedy decoding for deterministic results
                do_sample=False,
                temperature=1.0,
            )
        pt_translation = tokenizer.decode(pt_outputs[0], skip_special_tokens=True)

        # JAX generation (manual greedy decoding)
        jax_input_ids = jnp.array(input_ids.numpy())
        jax_attention_mask = jnp.array(attention_mask.numpy())

        # Use progress tracker for first iteration only
        if idx == 1:
            with CompilationProgressTracker("Compiling JAX model for generation"):
                # Precompute encoder output
                with jax.set_mesh(mesh):
                    encoder_outputs = jax_model.encoder(
                        jax_model.shared(jax_input_ids),
                        mask=(jax_input_ids != 0).astype(jnp.int32),
                        deterministic=True,
                    )

                # Start decoding
                start_token_id = hf_config.decoder_start_token_id or 0
                decoder_input_ids = jnp.array([[start_token_id]])
                generated_tokens = [start_token_id]

                max_length = 50
                for step in range(max_length - 1):
                    with jax.set_mesh(mesh):
                        # Forward pass
                        logits = jax_model(
                            encoder_outputs=encoder_outputs,
                            decoder_input_ids=decoder_input_ids,
                            attention_mask=jax_attention_mask,
                            deterministic=True,
                        )

                        # Get last token logits
                        next_token_logits = logits[0, -1, :]

                        # Greedy decoding
                        next_token = jnp.argmax(next_token_logits).item()

                        # Check for EOS
                        if next_token == hf_config.eos_token_id:
                            break

                        generated_tokens.append(next_token)
                        decoder_input_ids = jnp.array([generated_tokens])
        else:
            # Subsequent iterations without progress tracker
            # Precompute encoder output
            with jax.set_mesh(mesh):
                encoder_outputs = jax_model.encoder(
                    jax_model.shared(jax_input_ids),
                    mask=(jax_input_ids != 0).astype(jnp.int32),
                    deterministic=True,
                )

            # Start decoding
            start_token_id = hf_config.decoder_start_token_id or 0
            decoder_input_ids = jnp.array([[start_token_id]])
            generated_tokens = [start_token_id]

            max_length = 50
            for step in range(max_length - 1):
                with jax.set_mesh(mesh):
                    # Forward pass
                    logits = jax_model(
                        encoder_outputs=encoder_outputs,
                        decoder_input_ids=decoder_input_ids,
                        attention_mask=jax_attention_mask,
                        deterministic=True,
                    )

                    # Get last token logits
                    next_token_logits = logits[0, -1, :]

                    # Greedy decoding
                    next_token = jnp.argmax(next_token_logits).item()

                    # Check for EOS
                    if next_token == hf_config.eos_token_id:
                        break

                    generated_tokens.append(next_token)
                    decoder_input_ids = jnp.array([generated_tokens])

        jax_translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Compare
        logger.info(f'\nPyTorch: "{pt_translation}"')
        logger.info(f'JAX:     "{jax_translation}"')

        # Token-level comparison
        pt_tokens = tokenizer.encode(pt_translation)
        jax_tokens = generated_tokens[1:]  # Skip decoder start token

        # Calculate match percentage
        min_len = min(len(pt_tokens), len(jax_tokens))
        matches = sum(
            1
            for i in range(min_len)
            if i < len(pt_tokens) and i < len(jax_tokens) and pt_tokens[i] == jax_tokens[i]
        )
        match_rate = (
            matches / max(len(pt_tokens), len(jax_tokens))
            if max(len(pt_tokens), len(jax_tokens)) > 0
            else 0
        )

        if pt_translation == jax_translation:
            logger.info(f"✅ Translations MATCH (100%)")
        else:
            logger.info(f"⚠️  Translations DIFFER (Match rate: {match_rate:.1%})")
            logger.info(f"   PT tokens: {pt_tokens[:10]}...")
            logger.info(f"   JAX tokens: {jax_tokens[:10]}...")
            all_match = False

    logger.info("\n" + "=" * 80)
    if all_match:
        logger.info("✅ All translations match!")
    else:
        logger.warning("⚠️  Some translations differ")
    logger.info("=" * 80)

    return all_match, 0.0


def test_end_to_end_generation(model_name: str, mesh: jax.sharding.Mesh, tokenizer, precision="float32"):
    """
    End-to-end generation alignment.
    Compares the generated token sequence from PyTorch generate() vs JAX autoregressive loop.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Testing End-to-End Generation Alignment")
    logger.info("=" * 80)

    # Load Models
    hf_config = UMT5Config.from_pretrained(model_name)
    hf_model = HFUMt5ForConditionalGeneration.from_pretrained(
        model_name, dtype=get_torch_dtype(precision)
    )
    hf_model.eval()

    with jax.set_mesh(mesh):
        jax_model = JAXUMt5ForConditionalGeneration(hf_config, mesh, dtype=get_jax_dtype(precision))
    load_weights_from_hf(jax_model, hf_model, model_name, mesh=mesh, precision=precision)

    # Input prompts
    prompts = [
        "Translate English to German: Good morning, how are you?",
        "Translate English to French: The weather is beautiful today.",
        "Translate English to Chinese: Artificial Intelligence is changing the world.",
        "Translate English to Spanish: The quick brown fox jumps over the lazy dog.",
    ]

    for i, prompt in enumerate(prompts):
        logger.info(f"\nTest Case {i+1}: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids

        # PyTorch Generation (Greedy)
        logger.info("Running PyTorch generation...")
        with torch.no_grad():
            pt_outputs = hf_model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,  # Deterministic greedy search
                use_cache=True,
            )
        pt_generated_text = tokenizer.decode(pt_outputs[0], skip_special_tokens=True)
        logger.info(f"PyTorch Output IDs: {pt_outputs[0].tolist()}")
        logger.info(f"PyTorch Output Text: {pt_generated_text}")

        # JAX Generation (Manual Greedy Loop)
        import contextlib
        progress_ctx = CompilationProgressTracker("Compiling JAX model for generation") if i == 0 else contextlib.nullcontext()
        
        with progress_ctx:
            if i > 0:
                logger.info("Running JAX generation...")
            jax_input_ids = jnp.array(input_ids.numpy())

            # Precompute Encoder Output
            with jax.set_mesh(mesh):
                encoder_embeds = jax_model.shared(jax_input_ids)
                encoder_outputs = jax_model.encoder(
                    encoder_embeds,
                    mask=(jax_input_ids != 0).astype(jnp.int32),
                    deterministic=True,
                )

            # Start decoding
            start_token = (
                hf_config.decoder_start_token_id
                if hf_config.decoder_start_token_id is not None
                else 0
            )
            decoder_input_ids = jnp.array([[start_token]])
            generated_ids = [int(start_token)]

            for step in range(20):
                if step % 5 == 0 and i == 0:  # Only show progress for first iteration
                    logger.info(f"    Step {step+1}/20...")

                with jax.set_mesh(mesh):
                    outputs = jax_model(
                        encoder_outputs=encoder_outputs,
                        decoder_input_ids=decoder_input_ids,
                        attention_mask=(jax_input_ids != 0).astype(jnp.int32),
                        deterministic=True,
                    )

                    next_token_logits = outputs[:, -1, :]
                    next_token_id = int(jnp.argmax(next_token_logits, axis=-1)[0])

                # Stop if EOS
                if next_token_id == hf_config.eos_token_id:
                    generated_ids.append(next_token_id)
                    break

                generated_ids.append(next_token_id)
                decoder_input_ids = jnp.concatenate(
                    [decoder_input_ids, jnp.array([[next_token_id]])], axis=1
                )

            jax_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            logger.info(f"JAX Output IDs:     {generated_ids}")
            logger.info(f"JAX Output Text:    {jax_generated_text}")

        # Compare
        pt_ids = pt_outputs[0].tolist()

        if pt_ids == generated_ids:
            logger.info("✅ Token Sequence: MATCH")
        else:
            logger.error("❌ Token Sequence: MISMATCH")
            logger.error(f"  PT:  {pt_ids}")
            logger.error(f"  JAX: {generated_ids}")
            return False, 1.0

    return True, 0.0



def verify_alignment(model_name="google/umt5-base", test_type="all", tp_size=None, precision="float32"):
    """Main alignment verification function with configurable precision"""
    logger.info(f"Starting alignment verification for {model_name}")
    logger.info(f"Test type: {test_type}")
    logger.info(f"Precision: {precision}")
    start_time = time.time()

    # Setup
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    devices = jax.devices()
    num_devices = len(devices)
    logger.info(f"Found {num_devices} devices: {[d.device_kind for d in devices]}")

    # Determine parallelism strategy
    if tp_size is not None:
        logger.info(f"🎯 Using user-specified Tensor Parallelism: TP={tp_size}")
    elif num_devices >= 4:
        tp_size = 4
        logger.info(
            f"🚀 Auto-detected: Using 4-way Tensor Parallelism (TP=4) for memory efficiency"
        )
    elif num_devices >= 2:
        tp_size = 2
        logger.info(
            f"🚀 Auto-detected: Using 2-way Tensor Parallelism (TP=2) for memory efficiency"
        )
    else:
        tp_size = 1
        logger.info(f"Auto-detected: Using single device (no tensor parallelism)")

    mesh = create_device_mesh(
        ici_parallelism=[1, tp_size],
        dcn_parallelism=[1, 1],
        devices=devices[:tp_size] if num_devices > 0 else None,
        use_explicit_sharding=True,
    )
    logger.info(f"Mesh shape: {mesh.shape}, axis_names: {mesh.axis_names}")

    # Run tests
    results = []

    test_functions = {
        "encoder": test_encoder_alignment,
        "encoder_decoder": test_full_model_alignment,
        "logits": test_generation_model_alignment,
        "generation": test_end_to_end_translation,
        "tokens": test_end_to_end_generation,
    }

    if test_type == "all":
        # Run core alignment tests
        for test_name in ["encoder", "encoder_decoder", "logits", "generation"]:
            try:
                passed, mae = test_functions[test_name](model_name, mesh, tokenizer, precision)
                results.append((test_name.replace("_", " ").title(), passed, mae))
            except Exception as e:
                logger.error(f"{test_name} test failed with exception: {e}")
                import traceback
                traceback.print_exc()
                results.append((test_name.replace("_", " ").title(), False, float("inf")))
    elif test_type in test_functions:
        try:
            passed, mae = test_functions[test_type](model_name, mesh, tokenizer, precision)
            results.append((test_type.replace("_", " ").title(), passed, mae))
        except Exception as e:
            logger.error(f"{test_type} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_type.replace("_", " ").title(), False, float("inf")))
    else:
        logger.error(f"Unknown test type: {test_type}")
        return False

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("ALIGNMENT TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Precision: PyTorch={precision}, JAX={precision}")
    logger.info("-" * 80)
    for name, passed, mae in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{name:30s} {status:15s} (MAE: {mae:.10e})")

    all_passed = all(r[1] for r in results)
    logger.info("=" * 80)
    if all_passed:
        logger.info("✅ All alignment tests PASSED!")
    else:
        logger.error("❌ Some alignment tests FAILED!")

    logger.info(f"\nTotal time: {time.time() - start_time:.2f}s")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test UMT5 model alignment between PyTorch and JAX"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="google/umt5-base",
        help="Model path or identifier (e.g., google/umt5-base, google/umt5-small)",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        default="all",
        choices=[
            "all",
            "encoder",
            "encoder_decoder",
            "logits",
            "generation",
            "tokens",
        ],
        help="Which test to run (encoder: encoder only, encoder_decoder: full model, logits: precision test, generation: text generation, tokens: exact token alignment)",
    )
    parser.add_argument(
        "--log_file", type=str, default=None, help="Path to save the log file"
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="Tensor parallelism size (1, 2, or 4). If None, auto-detect",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help=(
            "Precision/dtype for both PyTorch and JAX models. "
            "float32: Highest precision, best for alignment tests (default). "
            "bfloat16: Lower precision, faster on TPU, may cause alignment failures. "
            "float16: Half precision, may cause numerical instability."
        ),
    )

    args = parser.parse_args()

    # Configure logging
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        log_handlers.append(logging.FileHandler(args.log_file, mode="w"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=log_handlers,
        force=True,
    )

    # Configure precision for both JAX and PyTorch
    precision_map = {
        "float32": ("highest", "float32"),  # (jax_matmul_precision, dtype_str)
        "bfloat16": ("default", "bfloat16"),
        "float16": ("high", "float16"),
    }
    
    jax_precision, dtype_str = precision_map[args.precision]
    
    # Set JAX matmul precision
    jax.config.update("jax_default_matmul_precision", jax_precision)
    logger.info(f"🎯 Precision Configuration:")
    logger.info(f"   PyTorch dtype: {args.precision}")
    logger.info(f"   JAX dtype: {dtype_str}")
    logger.info(f"   JAX matmul precision: {jax_precision}")
    
    if args.precision != "float32":
        logger.warning("⚠️  WARNING: Using non-float32 precision may cause alignment failures!")
        logger.warning("   For accurate alignment testing, use --precision float32 (default)")

    # Store precision in args for use in verify_alignment
    args.dtype_str = dtype_str

    verify_alignment(args.model_path, args.test_type, tp_size=args.tp_size, precision=args.precision)

