from typing import Sequence, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from sgl_jax.srt.layers.embeddings import Embed


class MelSpectrumExtractor:
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

class ResidualVectorQuantizer(nnx.Module):
    def __init__(
            self,
            dimension: int,
            n_q: int,
            bins: Sequence[int],
            dtype: jnp.dtype | None = jnp.float32,
            mesh: jax.sharding.Mesh | None = None,
    ):
        self.dimension = dimension
        self.n_q = n_q

        codebooks_list = []
        for i in range(n_q):
            size = bins[min(i, len(bins) - 1)]
            codebooks_list.append(
                Embed(
                    num_embeddings=size,
                    features=dimension,
                    type=dtype,
                    kernel_axes=("tensor", None),
                    param_dtype=dtype,
                    mesh=mesh,
                )
            )
        self.codebooks = nnx.List(codebooks_list)

    def encode(
            self, hidden_states: Array, mask: Optional[Array] = None, n_q: Optional[int] = None
    ) -> Tuple[Array, Array]:
        num_levels = n_q or self.n_q
        residual = hidden_states
        quantized = jnp.zeros_like(hidden_states)
        codes = []
        mask = None if mask is None else mask[..., None]
        for i in range(num_levels):
            codebook = self.codebooks[i].value
            dist = jnp.sum((residual[:, None, :] - codebook[None, :, :]) ** 2, axis=-1)
            idx = jnp.argmin(dist, axis=-1)
            chosen = codebook[idx]
            if mask is not None:
                chosen = chosen * mask
            quantized = quantized + chosen
            residual = residual - chosen
            codes.append(idx)
        return jnp.stack(codes, axis=0), quantized

    def decode(self, codes: Array) -> Array:
        num_levels = codes.shape[0]
        flat = codes.reshape(num_levels, -1)
        decoded = jnp.zeros((flat.shape[1], self.dimension), dtype=jnp.float32)
        for i in range(num_levels):
            codebook = self.codebooks[i].value
            decoded = decoded + codebook[flat[i]]
        return decoded.reshape(*codes.shape[1:], self.dimension)