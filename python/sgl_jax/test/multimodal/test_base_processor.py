import binascii

import pybase64

from sgl_jax.srt.multimodal.processors.base_processor import _normalize_image_source


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
