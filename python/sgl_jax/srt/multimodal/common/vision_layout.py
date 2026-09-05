"""Per-image Qwen2.5-VL layout, independent of batch lanes and padding."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisionLayout:
    indices: np.ndarray  # [merge units, 2]: window order and its inverse
    position_ids: np.ndarray  # [patches, 2], in window order
    window_ends: np.ndarray
    frame_ends: np.ndarray


def build_vision_layout(grid_thw, merge: int, window: int) -> VisionLayout:
    """Plan one image/video; ``window`` is measured in spatial merge units."""
    t, h, w = map(int, np.asarray(grid_thw).reshape(3))
    unit = merge**2
    grid_h, grid_w = h // merge, w // merge
    index = np.arange(t * grid_h * grid_w, dtype=np.int32).reshape(t, grid_h, grid_w)
    pad_h, pad_w = (-grid_h) % window, (-grid_w) % window
    windows_h, windows_w = (grid_h + pad_h) // window, (grid_w + pad_w) // window
    index = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-1)
    index = index.reshape(t, windows_h, window, windows_w, window)
    index = index.transpose(0, 1, 3, 2, 4).reshape(-1, window, window)
    window_lengths = (index != -1).sum(axis=(1, 2)).astype(np.int32) * unit
    index = index.reshape(-1)
    index = index[index != -1]
    indices = np.empty((len(index), 2), dtype=np.int32)
    indices[:, 0] = index
    indices[index, 1] = np.arange(len(index), dtype=np.int32)

    y, x = np.indices((h, w), dtype=np.int32)
    coords = np.stack((y, x), axis=-1)
    coords = coords.reshape(grid_h, merge, grid_w, merge, 2)
    coords = coords.transpose(0, 2, 1, 3, 4).reshape(h * w, 2)
    coords = np.tile(coords, (t, 1))
    coords = coords.reshape(-1, unit, 2)[index].reshape(t * h * w, 2)
    return VisionLayout(
        indices,
        coords,
        np.cumsum(window_lengths, dtype=np.int32),
        np.arange(1, t + 1, dtype=np.int32) * h * w,
    )
