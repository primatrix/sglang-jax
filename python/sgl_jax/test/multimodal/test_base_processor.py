import binascii
import io

import numpy as np
import pybase64
import torch
from PIL import Image

from sgl_jax.srt.multimodal.processors.base_processor import _normalize_image_source
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor


def test_normalize_image_source_decodes_data_uri_and_bare_base64():
    payload = b"multimodal-image-payload\x00\xff"
    encoded = pybase64.b64encode(payload).decode("ascii")

    assert _normalize_image_source(encoded) == payload
    assert _normalize_image_source(f"data:image/jpeg;base64,{encoded}") == payload


def test_normalize_image_source_rejects_invalid_base64():
    try:
        _normalize_image_source("not valid base64!")
    except binascii.Error:
        pass
    else:
        raise AssertionError("invalid base64 input should be rejected")


def test_qwen_load_image_decodes_to_rgb_tensor():
    pixels = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="PNG")

    image = QwenVLProcessor.load_image(encoded.getvalue())

    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.uint8
    np.testing.assert_array_equal(image.permute(1, 2, 0).numpy(), pixels)
