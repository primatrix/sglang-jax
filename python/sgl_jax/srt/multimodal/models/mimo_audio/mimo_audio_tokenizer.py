import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from transformers import PretrainedConfig

from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

@dataclass
class EncoderOutput:
    hidden_states: jnp.ndarray
    packed_states: jnp.ndarray
    output_lengths: jnp.ndarray
    codes: Optional[jnp.ndarray] = None

@dataclass
class VocoderOutput:
    wav: jnp.ndarray
    wav_lengths: jnp.ndarray

def make_sequence_mask(lengths: Array, max_length: Optional[int] = None) -> Array:
    """Create a boolean mask for variable-length sequences."""
    max_len = max_length or int(jnp.max(lengths))
    base = jnp.arange(max_len)[None, :]
    return base < lengths[:, None]


def get_position_ids(lengths: Array, max_length: Optional[int] = None) -> Array:
    """Generate position IDs for sequences."""
    max_len = max_length or int(jnp.max(lengths))
    base = jnp.arange(max_len)[None, :]
    return jnp.broadcast_to(base, (lengths.shape[0], max_len))


def rotate_half(x: Array) -> Array:
    """Rotate half the hidden dims of the input."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary(x: Array, cos: Array, sin: Array) -> Array:
    """Apply rotary positional embeddings to input tensor."""
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return (x * cos) + (rotate_half(x) * sin)


def to_mappings(config: PretrainedConfig) -> dict[str, WeightMapping]:
    mappings = {
        "encoder.conv1.weight": WeightMapping(target_path="encoder.conv1.kernel", transpose_axes=(2, 1, 0)),
        "encoder.conv1.bias": WeightMapping(target_path="encoder.conv1.bias"),
        "encoder.conv2.weight": WeightMapping(target_path="encoder.conv2.kernel", transpose_axes=(2, 1, 0)),
        "encoder.conv2.bias": WeightMapping(target_path="encoder.conv2.bias"),
        "encoder.layer_norm.weight": WeightMapping(target_path="encoder.layer_norm.scale"),
        "encoder.layer_norm.bias": WeightMapping(target_path="encoder.layer_norm.bias"),
        "encoder.layers.*.self_attn.q_proj.weight": WeightMapping(target_path="encoder.layers.*.self_attn.q_proj.weight", transpose=True),
        "encoder.layers.*.self_attn.q_proj.bias": WeightMapping(target_path="encoder.layers.*.self_attn.q_proj.bias"),
        "encoder.layers.*.self_attn.k_proj.weight": WeightMapping(target_path="encoder.layers.*.self_attn.k_proj.weight", transpose=True),
        "encoder.layers.*.self_attn.v_proj.weight": WeightMapping(target_path="encoder.layers.*.self_attn.v_proj.weight", transpose=True),
        "encoder.layers.*.self_attn.v_proj.bias": WeightMapping(target_path="encoder.layers.*.self_attn.v_proj.bias"),
        "encoder.layers.*.self_attn.out_proj.weight": WeightMapping(target_path="encoder.layers.*.self_attn.out_proj.weight", transpose=True),
        "encoder.layers.*.self_attn.out_proj.bias": WeightMapping(target_path="encoder.layers.*.self_attn.out_proj.bias"),
        "encoder.layers.*.self_attn_layer_norm.weight": WeightMapping(target_path="encoder.layers.*.self_attn_layer_norm.scale"),
        "encoder.layers.*.self_attn_layer_norm.bias": WeightMapping(target_path="encoder.layers.*.self_attn_layer_norm.bias"),
        "encoder.layers.*.final_layer_norm.weight": WeightMapping(target_path="encoder.layers.*.final_layer_norm.scale"),
        "encoder.layers.*.final_layer_norm.bias": WeightMapping(target_path="encoder.layers.*.final_layer_norm.bias"),
        "encoder.layers.*.fc1.weight": WeightMapping(target_path="encoder.layers.*.fc1.weight", transpose=True),
        "encoder.layers.*.fc1.bias": WeightMapping(target_path="encoder.layers.*.fc1.bias"),
        "encoder.layers.*.fc2.weight": WeightMapping(target_path="encoder.layers.*.fc2.weight", transpose=True),
        "encoder.layers.*.fc2.bias": WeightMapping(target_path="encoder.layers.*.fc2.bias"),
        "encoder.down_sample_layer.0.weight": WeightMapping(target_path="encoder.down_sample_layer.kernel", transpose_axes=(2, 1, 0)),
        "encoder.down_sample_norm.weight": WeightMapping(target_path="encoder.down_norm.scale"),
        "encoder.down_sample_norm.bias": WeightMapping(target_path="encoder.down_norm.bias"),
        "encoder.quantizer.vq.layers.*._codebook.embed": WeightMapping(target_path="encoder.quantizer.codebooks.*.embedding.embedding"),
        "decoder.dconv1.conv.weight": WeightMapping(target_path="decoder.dconv1.conv.kernel"),
        "decoder.dconv1.conv.bias": WeightMapping(target_path="decoder.dconv1.conv.bias"),
        "decoder.dconv1.norm.weight": WeightMapping(target_path="decoder.dconv1.norm.scale"),
        "decoder.dconv1.norm.bias": WeightMapping(target_path="decoder.dconv1.norm.bias"),
        "decoder.layer_norm.weight": WeightMapping(target_path="decoder.layer_norm.scale"),
        "decoder.layer_norm.bias": WeightMapping(target_path="decoder.layer_norm.bias"),
        "decoder.dconv2.conv.weight": WeightMapping(target_path="decoder.dconv2.conv.kernel"),
        "decoder.dconv2.conv.bias": WeightMapping(target_path="decoder.dconv2.conv.bias"),
        "decoder.dconv2.norm.weight": WeightMapping(target_path="decoder.dconv2.norm.scale"),
        "decoder.dconv2.norm.bias": WeightMapping(target_path="decoder.dconv2.norm.bias"),
        "decoder.layers.*.self_attn.q_proj.weight": WeightMapping(target_path="decoder.layers.*.self_attn.q_proj.weight", transpose=True),
        "decoder.layers.*.self_attn.q_proj.bias": WeightMapping(target_path="decoder.layers.*.self_attn.q_proj.bias"),
        "decoder.layers.*.self_attn.k_proj.weight": WeightMapping(target_path="decoder.layers.*.self_attn.k_proj.weight", transpose=True),
        "decoder.layers.*.self_attn.v_proj.weight": WeightMapping(target_path="decoder.layers.*.self_attn.v_proj.weight", transpose=True),
        "decoder.layers.*.self_attn.v_proj.bias": WeightMapping(target_path="decoder.layers.*.self_attn.v_proj.bias"),
        "decoder.layers.*.self_attn.out_proj.weight": WeightMapping(target_path="decoder.layers.*.self_attn.out_proj.weight", transpose=True),
        "decoder.layers.*.self_attn.out_proj.bias": WeightMapping(target_path="decoder.layers.*.self_attn.out_proj.bias"),
        "decoder.layers.*.self_attn_layer_norm.weight": WeightMapping(target_path="decoder.layers.*.self_attn_layer_norm.scale"),
        "decoder.layers.*.self_attn_layer_norm.bias": WeightMapping(target_path="decoder.layers.*.self_attn_layer_norm.bias"),
        "decoder.layers.*.final_layer_norm.weight": WeightMapping(target_path="decoder.layers.*.final_layer_norm.scale"),
        "decoder.layers.*.final_layer_norm.bias": WeightMapping(target_path="decoder.layers.*.final_layer_norm.bias"),
        "decoder.layers.*.fc1.weight": WeightMapping(target_path="decoder.layers.*.fc1.weight", transpose=True),
        "decoder.layers.*.fc1.bias": WeightMapping(target_path="decoder.layers.*.fc1.bias"),
        "decoder.layers.*.fc2.weight": WeightMapping(target_path="decoder.layers.*.fc2.weight", transpose=True),
        "decoder.layers.*.fc2.bias": WeightMapping(target_path="decoder.layers.*.fc2.bias"),
        "decoder.vocoder.embeddings.weight": WeightMapping(target_path="decoder.vocoder.embeddings.weight", transpose=True),
        "decoder.vocoder.layer_norm.weight": WeightMapping(target_path="decoder.vocoder.layer_norm.scale"),
        "decoder.vocoder.layer_norm.bias": WeightMapping(target_path="decoder.vocoder.layer_norm.bias"),
        "decoder.vocoder.head.out.weight": WeightMapping(target_path="decoder.vocoder.head.linear.weight", transpose=True),
        "decoder.vocoder.head.out.bias": WeightMapping(target_path="decoder.vocoder.head.linear.bias"),
        "decoder.vocoder.head.istft.window": WeightMapping(target_path="decoder.vocoder.head.istft.window"),
        "decoder.vocoder.layers.*.self_attn.q_proj.weight": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.q_proj.weight", transpose=True),
        "decoder.vocoder.layers.*.self_attn.q_proj.bias": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.q_proj.bias"),
        "decoder.vocoder.layers.*.self_attn.k_proj.weight": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.k_proj.weight", transpose=True),
        "decoder.vocoder.layers.*.self_attn.v_proj.weight": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.v_proj.weight", transpose=True),
        "decoder.vocoder.layers.*.self_attn.v_proj.bias": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.v_proj.bias"),
        "decoder.vocoder.layers.*.self_attn.out_proj.weight": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.out_proj.weight", transpose=True),
        "decoder.vocoder.layers.*.self_attn.out_proj.bias": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.out_proj.bias"),
        "decoder.vocoder.layers.*.self_attn_layer_norm.weight": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn_layer_norm.scale"),
        "decoder.vocoder.layers.*.self_attn_layer_norm.bias": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn_layer_norm.bias"),
        "decoder.vocoder.layers.*.final_layer_norm.weight": WeightMapping(target_path="decoder.vocoder.layers.*.final_layer_norm.scale"),
        "decoder.vocoder.layers.*.final_layer_norm.bias": WeightMapping(target_path="decoder.vocoder.layers.*.final_layer_norm.bias"),
        "decoder.vocoder.layers.*.fc1.weight": WeightMapping(target_path="decoder.vocoder.layers.*.fc1.weight", transpose=True),
        "decoder.vocoder.layers.*.fc1.bias": WeightMapping(target_path="decoder.vocoder.layers.*.fc1.bias"),
        "decoder.vocoder.layers.*.fc2.weight": WeightMapping(target_path="decoder.vocoder.layers.*.fc2.weight", transpose=True),
        "decoder.vocoder.layers.*.fc2.bias": WeightMapping(target_path="decoder.vocoder.layers.*.fc2.bias"),
    }
    return mappings


class MelSpectrumExtractor:
    """Extract mel spectrogram from audio waveforms."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        f_min: float,
        f_max: float,
        n_mels: int,
        power: float = 1.0,
        center: bool = True,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.n_mels = int(n_mels)
        self.power = float(power)
        self.center = center

        self._window = self._build_window()
        self._mel_filterbank = self._build_mel_filterbank()

    def _build_window(self) -> jnp.ndarray:
        if self.win_length <= 1:
            return jnp.ones((self.win_length,), dtype=jnp.float32)
        n = jnp.arange(self.win_length, dtype=jnp.float32)
        return 0.5 - 0.5 * jnp.cos(2 * jnp.pi * n / (self.win_length - 1))

    def _hz_to_mel(self, freq: jnp.ndarray) -> jnp.ndarray:
        return 2595.0 * jnp.log10(1.0 + freq / 700.0)

    def _mel_to_hz(self, mel: jnp.ndarray) -> jnp.ndarray:
        return 700.0 * (jnp.power(10.0, mel / 2595.0) - 1.0)

    def _build_mel_filterbank(self) -> jnp.ndarray:
        freq_bins = jnp.linspace(
            0.0,
            self.sample_rate / 2,
            self.n_fft // 2 + 1,
            dtype=jnp.float32,
        )
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)
        mel_points = jnp.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        filterbanks = []
        for i in range(self.n_mels):
            lower = hz_points[i]
            center = hz_points[i + 1]
            upper = hz_points[i + 2]
            denom_left = jnp.maximum(center - lower, 1e-10)
            denom_right = jnp.maximum(upper - center, 1e-10)
            left_slope = (freq_bins - lower) / denom_left
            right_slope = (upper - freq_bins) / denom_right
            filterbanks.append(jnp.maximum(0.0, jnp.minimum(left_slope, right_slope)))

        return jnp.stack(filterbanks, axis=0)

    def _frame_signal(self, waveform: jnp.ndarray) -> jnp.ndarray:
        frame_length = self.n_fft

        if self.center:
            pad = self.n_fft // 2
            if waveform.shape[0] > 1:
                waveform = jnp.pad(waveform, (pad, pad), mode="reflect")
            else:
                waveform = jnp.pad(waveform, (pad, pad))

        total_length = int(waveform.shape[0])
        if total_length < frame_length:
            pad_amount = frame_length - total_length
            waveform = jnp.pad(waveform, (0, pad_amount))
            total_length = int(waveform.shape[0])

        num_frames = 1 + max(0, (total_length - frame_length) // self.hop_length)
        if num_frames <= 0:
            num_frames = 1

        starts = [idx * self.hop_length for idx in range(num_frames)]
        frames = jnp.stack([waveform[start : start + frame_length] for start in starts], axis=0)
        return frames

    def _mel_spectrogram(self, waveform: jnp.ndarray) -> jnp.ndarray:
        waveform = jnp.asarray(waveform, dtype=jnp.float32)
        frames = self._frame_signal(waveform)
        if self.win_length < self.n_fft:
            total_pad = self.n_fft - self.win_length
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            window = jnp.pad(self._window, (pad_left, pad_right))
        else:
            window = self._window[: self.n_fft]
        windowed = frames * window
        stft = jnp.fft.rfft(windowed, n=self.n_fft, axis=1)
        magnitude = jnp.abs(stft) ** self.power
        mel_spec = magnitude @ self._mel_filterbank.T
        return mel_spec.T

    def __call__(self, waveform: jnp.ndarray) -> jnp.ndarray:
        """Compute mel spectrogram from waveform.

        Args:
            waveform: JAX array of shape (samples,) or (batch, samples)

        Returns:
            Mel spectrogram of shape (n_mels, time) or (batch, n_mels, time)
        """
        waveform = jnp.asarray(waveform, dtype=jnp.float32)

        squeeze_dim = False
        if waveform.ndim == 1:
            waveform = waveform[jnp.newaxis, :]
            squeeze_dim = True

        mel_outputs = []
        for sample in waveform:
            mel_outputs.append(self._mel_spectrogram(sample))

        mel_stack = jnp.stack(mel_outputs, axis=0)
        if squeeze_dim:
            mel_stack = jnp.squeeze(mel_stack, axis=0)
        return mel_stack



class MiMoRotaryEmbedding(nnx.Module):
    """Rotary Position Embedding for MiMo Audio.

    This adapter returns (cos, sin) tuples to match the expected interface
    in MiMo's Attention layer, unlike sglang-jax's RotaryEmbedding which
    directly applies the rotation to query/key.
    """

    def __init__(
        self,
        base: float,
        dim: int,
        max_seq_len: int,
        rope_type: str = "default",
        dtype: jnp.dtype = jnp.float32,
    ):
        self.base = base
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_type = rope_type
        self.dtype = dtype

        half_dim = dim // 2
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, half_dim, dtype=jnp.float32) / float(half_dim)))
        self.inv_freq = nnx.Param(inv_freq)
        self.attention_scaling = 1.0

    def __call__(self, hidden_states: Array, position_ids: Array) -> Tuple[Array, Array]:
        """Compute cos and sin for rotary embeddings.

        Args:
            hidden_states: Input tensor (used only for dtype).
            position_ids: Position indices [batch, seq_len].

        Returns:
            Tuple of (cos, sin) tensors.
        """
        freq = position_ids[..., None] * self.inv_freq[None, None, :]
        emb = jnp.concatenate([freq, freq], axis=-1)
        cos = jnp.cos(emb) * self.attention_scaling
        sin = jnp.sin(emb) * self.attention_scaling
        return cos.astype(hidden_states.dtype), sin.astype(hidden_states.dtype)

class ResidualVectorQuantizer(nnx.Module):
    """Residual Vector Quantizer using multiple codebooks."""

    def __init__(
        self,
        dimension: int,
        n_q: int,
        bins: Sequence[int],
        dtype: jnp.dtype = jnp.float32,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.dimension = dimension
        self.n_q = n_q

        self.codebooks = nnx.List(
            [
                Embed(
                    num_embeddings=bins[min(i, len(bins) - 1)],
                    features=dimension,
                    dtype=dtype,
                    param_dtype=dtype,
                    kernel_axes=(None, None),
                    mesh=mesh,
                )
                for i in range(n_q)
            ]
        )

    def encode(
        self, hidden_states: Array, mask: Optional[Array] = None, n_q: Optional[int] = None
    ) -> Tuple[Array, Array]:
        """Encode hidden states into discrete codes."""
        num_levels = n_q or self.n_q
        residual = hidden_states
        quantized = jnp.zeros_like(hidden_states)
        codes = []
        mask_expanded = None if mask is None else mask[..., None]

        for i in range(num_levels):
            codebook = self.codebooks[i].embedding.value
            dist = jnp.sum((residual[:, None, :] - codebook[None, :, :]) ** 2, axis=-1)
            idx = jnp.argmin(dist, axis=-1)
            chosen = codebook[idx]
            if mask_expanded is not None:
                chosen = chosen * mask_expanded
            quantized = quantized + chosen
            residual = residual - chosen
            codes.append(idx)

        return jnp.stack(codes, axis=0), quantized

    def decode(self, codes: Array) -> Array:
        """Decode discrete codes back to continuous representations."""
        num_levels = codes.shape[0]
        flat = codes.reshape(num_levels, -1)
        decoded = jnp.zeros((flat.shape[1], self.dimension), dtype=jnp.float32)

        for i in range(num_levels):
            codebook = self.codebooks[i].embedding.value
            decoded = decoded + codebook[flat[i]]

        return decoded.reshape(*codes.shape[1:], self.dimension)


class ConvTranspose1d(nnx.Module):
    """1D Transposed Convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.stride = stride
        self.kernel_size = kernel_size

        kshape = (in_channels, out_channels, kernel_size)
        self.kernel = nnx.Param(jnp.zeros(kshape, dtype=dtype))
        self.bias = nnx.Param(jnp.zeros((out_channels,), dtype=dtype))

    def __call__(self, x: Array) -> Array:
        batch, length, channels = x.shape
        kernel = self.kernel.value
        kernel_size = kernel.shape[-1]

        # Upsample by inserting zeros
        up_len = (length - 1) * self.stride + 1
        idx = jnp.arange(length) * self.stride
        upsampled = jnp.zeros((batch, up_len, channels), dtype=x.dtype)
        upsampled = upsampled.at[:, idx, :].set(x)
        upsampled = jnp.pad(upsampled, ((0, 0), (kernel_size - 1, kernel_size - 1), (0, 0)))

        # Convolution
        lhs = jnp.swapaxes(upsampled, 1, 2)
        rhs = jnp.flip(kernel, axis=-1).transpose(1, 0, 2)
        y = jax.lax.conv_general_dilated(
            lhs=lhs,
            rhs=rhs,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCH", "OIH", "NCH"),
        )
        y = y + self.bias.value[None, :, None]
        y = jnp.swapaxes(y, 1, 2)
        return y

class ISTFT(nnx.Module):
    """Inverse Short-Time Fourier Transform for audio synthesis."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        padding: str = "same",
        dtype: jnp.dtype = jnp.float32,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.padding = padding

        self.window = nnx.Param(jnp.hanning(win_length).astype(dtype))
        self.pad = (self.win_length - self.hop_length) // 2 if padding == "same" else 0

    def __call__(self, spec: Array) -> Array:
        """Convert spectrogram to audio waveform."""
        frames = jnp.fft.irfft(spec, n=self.n_fft, axis=1, norm="backward")
        frames = frames * self.window.value[None, :, None]
        frames = jnp.swapaxes(frames, 1, 2)

        batch, num_frames, _ = frames.shape
        output_size = (num_frames - 1) * self.hop_length + self.win_length
        audio = jnp.zeros((batch, output_size), dtype=frames.dtype)
        env = jnp.zeros_like(audio)
        window_sq = jnp.square(self.window.value)

        def body(i, carry):
            audio_acc, env_acc = carry
            start = i * self.hop_length
            frame = frames[:, i, :]
            current_audio = jax.lax.dynamic_slice(audio_acc, (0, start), (batch, self.win_length))
            current_env = jax.lax.dynamic_slice(env_acc, (0, start), (batch, self.win_length))
            updated_audio = current_audio + frame
            updated_env = current_env + window_sq
            audio_acc = jax.lax.dynamic_update_slice(audio_acc, updated_audio, (0, start))
            env_acc = jax.lax.dynamic_update_slice(env_acc, updated_env, (0, start))
            return audio_acc, env_acc

        audio, env = jax.lax.fori_loop(0, num_frames, body, (audio, env))

        if self.pad > 0:
            audio = audio[:, self.pad : -self.pad]
            env = env[:, self.pad : -self.pad]

        env = jnp.maximum(env, 1e-11)
        audio = audio / env
        return audio

class ISTFTHead(nnx.Module):
    """ISTFT head that projects hidden states to magnitude and phase."""

    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.linear = LinearBase(
            dim,
            n_fft + 2,
            mesh=mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )
        self.istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            padding=padding,
            dtype=dtype,
        )

    def __call__(self, hidden_states: Array) -> Array:
        x, _ = self.linear(hidden_states)
        x = jnp.swapaxes(x, 1, 2)
        mag, phase = jnp.split(x, 2, axis=1)

        original_dtype = hidden_states.dtype
        mag = mag.astype(jnp.float32)
        phase = phase.astype(jnp.float32)

        mag = jnp.clip(jnp.exp(mag), a_max=1e2)
        real = jnp.cos(phase)
        imag = jnp.sin(phase)
        spec = mag * (real + 1j * imag)

        audio = self.istft(spec)
        audio = audio.astype(original_dtype)
        return audio

class Attention(nnx.Module):
    """Multi-head attention with optional window masking and causal masking."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
        causal: bool,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.window_size = window_size
        self.causal = causal

        self.q_proj = LinearBase(
            embed_dim, embed_dim, mesh=mesh, use_bias=True, params_dtype=dtype, kernel_axes=(None, None)
        )
        self.k_proj = LinearBase(
            embed_dim, embed_dim, mesh=mesh, use_bias=False, params_dtype=dtype, kernel_axes=(None, None)
        )
        self.v_proj = LinearBase(
            embed_dim, embed_dim, mesh=mesh, use_bias=True, params_dtype=dtype, kernel_axes=(None, None)
        )
        self.out_proj = LinearBase(
            embed_dim, embed_dim, mesh=mesh, use_bias=True, params_dtype=dtype, kernel_axes=(None, None)
        )

    def _window_mask(self, seq_len: int) -> Optional[Array]:
        """Create window mask for local attention."""
        left, right = self.window_size
        if left < 0 and right < 0:
            return None
        pos = jnp.arange(seq_len)
        rel = pos[None, :] - pos[:, None]
        mask = jnp.ones((seq_len, seq_len), dtype=bool)
        if left >= 0:
            mask &= rel >= -left
        if right >= 0:
            mask &= rel <= right
        return mask

    def __call__(
        self, x: Array, mask: Optional[Array], rope: Optional[Tuple[Array, Array]]
    ) -> Array:
        batch, seq_len, _ = x.shape

        q, _ = self.q_proj(x)
        k, _ = self.k_proj(x)
        v, _ = self.v_proj(x)

        def reshape(t):
            t = t.reshape(batch, seq_len, self.num_heads, self.head_dim)
            return jnp.swapaxes(t, 1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        if rope is not None:
            cos, sin = rope
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale

        if mask is not None:
            scores = jnp.where(mask[:, None, None, :], scores, -1e9)

        if self.causal:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
            scores = jnp.where(causal_mask, scores, -1e9)

        wmask = self._window_mask(seq_len)
        if wmask is not None:
            scores = jnp.where(wmask, scores, -1e9)

        weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        context = jnp.swapaxes(context, 1, 2).reshape(batch, seq_len, self.embed_dim)

        out, _ = self.out_proj(context)

        if mask is not None:
            out = out * mask[..., None]

        return out

class TransformerLayer(nnx.Module):
    """Transformer layer with self-attention and feed-forward network."""

    def __init__(
        self,
        d_model: int,
        attention_heads: int,
        ffn_dim: int,
        causal: bool,
        attn_window_size: Tuple[int, int],
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.act = jax.nn.gelu

        self.self_attn = Attention(
            d_model, attention_heads, attn_window_size, causal, mesh=mesh, dtype=dtype, rngs=rngs
        )
        self.self_attn_layer_norm = nnx.LayerNorm(d_model, epsilon=1e-6, param_dtype=dtype, rngs=rngs)
        self.final_layer_norm = nnx.LayerNorm(d_model, epsilon=1e-6, param_dtype=dtype, rngs=rngs)

        self.fc1 = LinearBase(d_model, ffn_dim, mesh=mesh, params_dtype=dtype, kernel_axes=(None, None))
        self.fc2 = LinearBase(ffn_dim, d_model, mesh=mesh, params_dtype=dtype, kernel_axes=(None, None))

    def __call__(
        self, hidden_states: Array, mask: Optional[Array], rope: Optional[Tuple[Array, Array]]
    ) -> Array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask, rope)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return residual + hidden_states

class CausalConvTranspose1d(nnx.Module):
    """Causal transposed convolution with group normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.conv = ConvTranspose1d(in_channels, out_channels, kernel_size, stride, dtype=dtype, rngs=rngs)
        self.norm = nnx.GroupNorm(num_features=out_channels, num_groups=1, epsilon=1e-5, param_dtype=dtype, rngs=rngs)
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x: Array, input_length: Array) -> Tuple[Array, Array]:
        y = self.conv(x)
        y = self.norm(y)
        trim = max(0, self.kernel_size - self.stride)
        if trim > 0:
            y = y[:, :-trim, :]
        output_len = (input_length - 1) * self.stride + self.kernel_size - trim
        return y, output_len

class TransformerVocos(nnx.Module):
    """Transformer-based vocoder for audio synthesis."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.config = config

        self.embeddings = LinearBase(
            config.n_mels,
            config.vocoder_dim,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )

        self.position_embedding = MiMoRotaryEmbedding(
            config.rope_theta,
            config.vocoder_dim // config.vocoder_attention_heads,
            config.max_audio_seconds * config.sampling_rate // config.hop_length,
            config.rope_type,
            dtype=dtype,
        )

        self.layers = nnx.List(
            [
                TransformerLayer(
                    config.vocoder_dim,
                    config.vocoder_attention_heads,
                    config.vocoder_intermediate_dim,
                    False,
                    tuple(config.vocoder_attn_window_size),
                    mesh=mesh,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(config.vocoder_num_layers)
            ]
        )

        self.layer_norm = nnx.LayerNorm(config.vocoder_dim, epsilon=1e-6, param_dtype=dtype, rngs=rngs)

        self.head = ISTFTHead(
            config.vocoder_dim,
            config.nfft,
            config.hop_length,
            config.vocoder_padding,
            mesh=mesh,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, mels: Array, input_length: Array) -> VocoderOutput:
        x, _ = self.embeddings(mels)
        mask = make_sequence_mask(input_length, x.shape[1])
        pos = get_position_ids(input_length, x.shape[1])
        rope = self.position_embedding(x, pos)

        for layer in self.layers:
            x = layer(x, mask, rope)

        x = self.layer_norm(x)
        x = x * mask[..., None]
        wav = self.head(x)
        wav_len = input_length * self.config.hop_length
        wav = wav[:, None, :]

        return VocoderOutput(wav=wav, wav_lengths=wav_len)

class AudioEncoder(nnx.Module):
    """Audio encoder with convolutional downsampling and transformer layers."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.config = config

        self.conv1 = nnx.Conv(
            in_features=config.n_mels,
            out_features=config.d_model,
            kernel_size=config.kernel_size,
            padding=1,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=config.d_model,
            out_features=config.d_model,
            kernel_size=config.kernel_size,
            strides=config.stride_size,
            padding=1,
            param_dtype=dtype,
            rngs=rngs,
        )

        self.position_embedding = MiMoRotaryEmbedding(
            config.rope_theta,
            config.d_model // config.encoder_attention_heads,
            int(config.max_audio_seconds * config.sampling_rate // config.hop_length),
            config.rope_type,
            dtype=dtype,
        )

        self.layers = nnx.List(
            [
                TransformerLayer(
                    config.d_model,
                    config.encoder_attention_heads,
                    config.encoder_ffn_dim,
                    config.encoder_causal,
                    tuple(config.encoder_attn_window_size),
                    mesh=mesh,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(config.encoder_layers)
            ]
        )

        self.layer_norm = nnx.LayerNorm(config.d_model, epsilon=1e-6, param_dtype=dtype, rngs=rngs)

        if config.avg_pooler != 1:
            self.down_sample_layer = nnx.Conv(
                in_features=config.d_model,
                out_features=config.d_model,
                kernel_size=config.avg_pooler,
                strides=config.avg_pooler,
                padding="SAME",
                use_bias=False,
                param_dtype=dtype,
                rngs=rngs,
            )
            self.down_norm = nnx.LayerNorm(config.d_model, epsilon=1e-6, param_dtype=dtype, rngs=rngs)
        else:
            self.down_sample_layer = None
            self.down_norm = None

        if config.num_quantizers:
            bins = config.codebook_size or [1024]
            self.quantizer = ResidualVectorQuantizer(
                config.d_model, config.num_quantizers, bins, dtype=dtype, mesh=mesh
            )
        else:
            self.quantizer = None

    def get_output_length(self, mel_len: Array) -> Array:
        """Compute output length after convolutional downsampling."""
        tgt = mel_len + 3 - self.config.kernel_size
        return (tgt + 2 - self.config.kernel_size) // self.config.stride_size + 1

    def __call__(
        self,
        input_features: Array,
        input_lens: Array,
        use_quantizer: bool = True,
        n_q: Optional[int] = None,
    ) -> EncoderOutput:
        x = input_features
        x = jax.nn.gelu(self.conv1(x))
        x = jax.nn.gelu(self.conv2(x))

        lengths = self.get_output_length(input_lens)
        max_len = x.shape[1]
        mask = make_sequence_mask(lengths, max_len)
        pos = get_position_ids(lengths, max_len)
        rope = self.position_embedding(x, pos)

        skip = 0.0
        for idx, layer in enumerate(self.layers):
            x = layer(x, mask, rope)
            if self.config.encoder_skip_layer_id and idx == self.config.encoder_skip_layer_id - 1:
                skip = x

        x = x + skip
        x = self.layer_norm(x)

        if self.down_sample_layer is not None:
            x = jax.nn.gelu(self.down_sample_layer(x))
            lengths = (lengths // self.config.avg_pooler) + (
                (lengths % self.config.avg_pooler) != 0
            ).astype(lengths.dtype)
            max_len = x.shape[1]
            mask = make_sequence_mask(lengths, max_len)
            x = self.down_norm(x)

        x = x * mask[..., None]
        packed = x.reshape(-1, self.config.d_model)
        mask_flat = mask.reshape(-1)

        codes = None
        if self.quantizer is not None and use_quantizer:
            codes, quantized = self.quantizer.encode(packed, mask=mask_flat, n_q=n_q)
            packed = quantized

        packed = packed.reshape(x.shape)

        return EncoderOutput(
            hidden_states=packed, packed_states=packed, output_lengths=lengths, codes=codes
        )

    def decode_vq(self, codes: Array) -> Array:
        """Decode VQ codes to hidden states."""
        if self.quantizer is None:
            raise ValueError("Quantizer disabled")
        return self.quantizer.decode(codes)

class AudioDecoder(nnx.Module):
    """Audio decoder with transformer layers and vocoder."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.config = config

        if config.avg_pooler != 1:
            self.dconv1 = CausalConvTranspose1d(
                config.d_model,
                config.d_model,
                config.avg_pooler,
                config.avg_pooler,
                dtype=dtype,
                rngs=rngs,
            )
        else:
            self.dconv1 = None

        self.position_embedding = MiMoRotaryEmbedding(
            config.rope_theta,
            config.d_model // config.decoder_attention_heads,
            int(config.max_audio_seconds * config.sampling_rate // config.hop_length),
            config.rope_type,
            dtype=dtype,
        )

        self.layers = nnx.List(
            [
                TransformerLayer(
                    config.d_model,
                    config.decoder_attention_heads,
                    config.decoder_ffn_dim,
                    config.decoder_causal,
                    tuple(config.decoder_attn_window_size),
                    mesh=mesh,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(config.decoder_layers)
            ]
        )

        self.layer_norm = nnx.LayerNorm(config.d_model, epsilon=1e-6, param_dtype=dtype, rngs=rngs)

        self.dconv2 = CausalConvTranspose1d(
            config.d_model,
            config.n_mels,
            config.decoder_kernel_size,
            config.decoder_stride_size,
            dtype=dtype,
            rngs=rngs,
        )

        self.vocoder = TransformerVocos(config, mesh=mesh, dtype=dtype, rngs=rngs)

    def __call__(self, audio_embed: Array, input_length: Array) -> Array:
        x = audio_embed
        lengths = input_length

        if self.dconv1 is not None:
            x, lengths = self.dconv1(x, lengths)

        mask = make_sequence_mask(lengths, x.shape[1])
        pos = get_position_ids(lengths, x.shape[1])
        rope = self.position_embedding(x, pos)

        for layer in self.layers:
            x = layer(x, mask, rope)

        x = self.layer_norm(x)
        coarse, mel_lengths = self.dconv2(x, lengths)
        vocoder_out = self.vocoder(coarse, mel_lengths)

        return vocoder_out.wav

class FlaxMiMoAudioTokenizer(nnx.Module):
    """MiMo Audio Tokenizer combining encoder and decoder."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.encoder = AudioEncoder(config, mesh=mesh, dtype=dtype, rngs=rngs)
        self.decoder = AudioDecoder(config, mesh=mesh, dtype=dtype, rngs=rngs)
        self.downsample_rate = int(config.hop_length * 2 * config.avg_pooler)

    def __call__(
        self, mels: Array, input_lens: Array, use_quantizer: bool = True
    ) -> Array:
        """Forward pass: encode mel spectrograms and decode to audio."""
        enc = self.encoder(mels, input_lens, use_quantizer=use_quantizer)
        return self.decoder(enc.hidden_states, enc.output_lengths)

    def encode(
        self,
        mels: Array,
        input_lens: Array,
        use_quantizer: bool = True,
        n_q: Optional[int] = None,
    ) -> EncoderOutput:
        """Encode mel spectrograms to hidden states or codes."""
        return self.encoder(mels, input_lens, use_quantizer=use_quantizer, n_q=n_q)

    def decode(self, codes: Array) -> Array:
        """Decode VQ codes to audio waveform."""
        hidden = self.encoder.decode_vq(codes)
        hidden = hidden[None, ...]
        lengths = jnp.array([hidden.shape[1]])
        return self.decoder(hidden, lengths)

    def load_weights(self, model_config):
        loader = WeightLoader(self, model_config, self.mesh, self.dtype)
        loader.load_weights_from_safetensors(to_mappings(self.config))
        self._init_rope_inv_freq()

    def _init_rope_inv_freq(self):
        config = self.config
        encoder_dim = config.d_model // config.encoder_attention_heads
        encoder_half = encoder_dim // 2
        self.encoder.position_embedding.inv_freq.value = 1.0 / (
            config.rope_theta ** (jnp.arange(0, encoder_half, dtype=jnp.float32) / encoder_half)
        )
        decoder_dim = config.d_model // config.decoder_attention_heads
        decoder_half = decoder_dim // 2
        self.decoder.position_embedding.inv_freq.value = 1.0 / (
            config.rope_theta ** (jnp.arange(0, decoder_half, dtype=jnp.float32) / decoder_half)
        )
        vocoder_dim = config.vocoder_dim // config.vocoder_attention_heads
        vocoder_half = vocoder_dim // 2
        self.decoder.vocoder.position_embedding.inv_freq.value = 1.0 / (
            config.rope_theta ** (jnp.arange(0, vocoder_half, dtype=jnp.float32) / vocoder_half)
        )

EntryClass = FlaxMiMoAudioTokenizer
