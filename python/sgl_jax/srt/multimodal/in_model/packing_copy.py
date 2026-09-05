"""Host feature copy/cast used by encoder lane packing."""

import ml_dtypes
import numpy as np
from numba import njit, types


# An explicit signature compiles (or loads the cache) at import, before serving.
# Lengths are runtime values. Read-only inputs also accept writable arrays, so
# both processor views and immutable inputs use this one specialization.
@njit(
    types.void(types.uint16[::1], types.Array(types.uint32, 1, "C", readonly=True)),
    nogil=True,
    cache=True,
)
def _copy_f32_to_bf16_bits(destination, source):
    for index in range(source.size):
        bits = source[index]
        if (bits & 0x7FFFFFFF) > 0x7F800000:
            # Match ml_dtypes: canonical quiet NaN, preserving the sign.
            destination[index] = np.uint16(((bits >> 16) & 0x8000) | 0x7FC0)
        else:
            # Round to nearest, ties to even, including subnormals and overflow.
            rounded = np.uint32(bits + 0x7FFF + ((bits >> 16) & 1))
            destination[index] = np.uint16(rounded >> 16)


def copy_features(destination: np.ndarray, source: np.ndarray) -> None:
    """Copy into a freshly allocated packed slice, converting dtype as needed."""
    if (
        source.dtype == np.float32
        and destination.dtype == ml_dtypes.bfloat16
        and source.shape == destination.shape
        and source.flags.c_contiguous
        and source.flags.aligned
        and destination.flags.c_contiguous
        and destination.flags.aligned
    ):
        _copy_f32_to_bf16_bits(
            destination.view(np.uint16).reshape(-1), source.view(np.uint32).reshape(-1)
        )
    else:
        # Preserve NumPy's casting, striding and broadcasting for other inputs.
        destination[...] = source
