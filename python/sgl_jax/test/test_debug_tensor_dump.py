import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _enable_dump(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_JAX_DEBUG_DUMP", "1")
    monkeypatch.setenv("SGLANG_JAX_DEBUG_DUMP_DIR", str(tmp_path))


def test_debug_dump_is_disabled_by_default(monkeypatch):
    from sgl_jax.srt.utils.debug_utils import maybe_dump_jax_array

    calls = []
    monkeypatch.delenv("SGLANG_JAX_DEBUG_DUMP", raising=False)
    monkeypatch.setattr(jax.debug, "callback", lambda *args, **kwargs: calls.append(args))

    value = jnp.ones((2,), dtype=jnp.float32)
    returned = maybe_dump_jax_array(value, component="dsa", name="q", layer_id=3)

    assert returned is value
    assert calls == []


@pytest.mark.parametrize(
    ("environment", "value", "expected_calls"),
    [
        ("SGLANG_JAX_DEBUG_DUMP_COMPONENTS", "dsa,decoder_layer", 1),
        ("SGLANG_JAX_DEBUG_DUMP_COMPONENTS", "logits", 0),
        ("SGLANG_JAX_DEBUG_DUMP_LAYERS", "1,3,5", 1),
        ("SGLANG_JAX_DEBUG_DUMP_LAYERS", "1,5", 0),
        ("SGLANG_JAX_DEBUG_DUMP_PROCESSES", "2,7", 1),
        ("SGLANG_JAX_DEBUG_DUMP_PROCESSES", "2,6", 0),
    ],
)
def test_debug_dump_filters_component_layer_and_process(
    monkeypatch, tmp_path, environment, value, expected_calls
):
    from sgl_jax.srt.utils.debug_utils import maybe_dump_jax_array

    _enable_dump(monkeypatch, tmp_path)
    monkeypatch.setenv(environment, value)
    monkeypatch.setattr(jax, "process_index", lambda: 7)
    calls = []
    monkeypatch.setattr(jax.debug, "callback", lambda *args, **kwargs: calls.append(args))

    array = jnp.ones((2,), dtype=jnp.float32)
    returned = maybe_dump_jax_array(
        array,
        component="dsa",
        name="q_index",
        layer_id=3,
        forward_mode="decode",
    )

    assert returned is array
    assert len(calls) == expected_calls


def test_debug_dump_layer_filter_keeps_global_components(monkeypatch, tmp_path):
    from sgl_jax.srt.utils.debug_utils import maybe_dump_jax_array

    _enable_dump(monkeypatch, tmp_path)
    monkeypatch.setenv("SGLANG_JAX_DEBUG_DUMP_LAYERS", "3,39")
    calls = []
    monkeypatch.setattr(jax.debug, "callback", lambda *args, **kwargs: calls.append(args))

    maybe_dump_jax_array(
        jnp.ones((2,), dtype=jnp.float32),
        component="logits",
        name="next_token_logits",
        layer_id=None,
    )

    assert len(calls) == 1


def test_debug_dump_callback_writes_array_and_manifest(monkeypatch, tmp_path):
    from sgl_jax.srt.utils.debug_utils import maybe_dump_jax_array

    _enable_dump(monkeypatch, tmp_path)
    monkeypatch.setattr(jax, "process_index", lambda: 7)

    def run_callback(callback, value, **_kwargs):
        callback(np.asarray(value))

    monkeypatch.setattr(jax.debug, "callback", run_callback)
    value = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

    returned = maybe_dump_jax_array(
        value,
        component="dsa/selection",
        name="logical topk ids",
        layer_id=39,
        forward_mode="decode",
    )

    assert returned is value
    manifests = list(tmp_path.glob("manifest-p00007.jsonl"))
    arrays = list(tmp_path.glob("*.npy"))
    assert len(manifests) == 1
    assert len(arrays) == 1
    np.testing.assert_array_equal(np.load(arrays[0]), np.asarray(value))

    rows = [json.loads(line) for line in manifests[0].read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row == {
        "filename": arrays[0].name,
        "process": 7,
        "component": "dsa/selection",
        "layer": 39,
        "forward_mode": "decode",
        "name": "logical topk ids",
        "occurrence": row["occurrence"],
        "shape": [2, 3],
        "dtype": "float32",
    }
    assert isinstance(row["occurrence"], int)
    assert row["occurrence"] >= 0
    assert arrays[0].parent == tmp_path
    assert "/" not in arrays[0].name
    assert " " not in arrays[0].name


def test_debug_dump_executes_from_jitted_code(monkeypatch, tmp_path):
    from sgl_jax.srt.utils.debug_utils import maybe_dump_jax_array

    _enable_dump(monkeypatch, tmp_path)

    @jax.jit
    def dump_from_jit(value):
        return maybe_dump_jax_array(
            value,
            component="decoder_layer",
            name="hidden_states",
            layer_id=1,
            forward_mode="extend",
        )

    value = jnp.array([1.0, 2.0], dtype=jnp.float32)
    np.testing.assert_array_equal(dump_from_jit(value).block_until_ready(), value)

    arrays = list(tmp_path.glob("*.npy"))
    assert len(arrays) == 1
    np.testing.assert_array_equal(np.load(arrays[0]), np.asarray(value))
