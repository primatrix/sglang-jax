import numpy as np
import xxhash

from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    hash_feature,
)


def test_hash_feature_numpy_uses_xxh3_64():
    feature = np.arange(24, dtype=np.float32).reshape(6, 4)

    expected = xxhash.xxh3_64_intdigest(memoryview(feature).cast("B"))

    assert hash_feature(feature) == expected


def test_hash_feature_numpy_is_layout_independent():
    feature = np.arange(24, dtype=np.float32).reshape(6, 4)
    backing = np.empty((6, 8), dtype=np.float32)
    backing[:, ::2] = feature
    noncontiguous_feature = backing[:, ::2]

    assert not noncontiguous_feature.flags.c_contiguous
    assert hash_feature(noncontiguous_feature) == hash_feature(feature)


def test_hash_feature_numpy_changes_with_content():
    first = np.arange(24, dtype=np.float32).reshape(6, 4)
    second = first.copy()
    second[-1, -1] += 1

    assert hash_feature(first) != hash_feature(second)


def test_feature_content_determines_item_identity():
    feature = np.arange(24, dtype=np.float32).reshape(6, 4)
    same_feature = feature.copy()
    different_feature = feature.copy()
    different_feature[-1, -1] += 1

    items = [
        MultimodalDataItem(modality=Modality.IMAGE, feature=value)
        for value in (feature, same_feature, different_feature)
    ]
    for item in items:
        item.set_pad_value()

    assert items[0].hash == items[1].hash
    assert items[0].pad_value == items[1].pad_value
    assert items[0].hash != items[2].hash
    assert items[0].pad_value != items[2].pad_value
