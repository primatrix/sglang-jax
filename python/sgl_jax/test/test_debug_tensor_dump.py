import json
from types import SimpleNamespace

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


def test_debug_dump_forward_occurrence_is_independent_of_callback_order(
    monkeypatch, tmp_path
):
    from sgl_jax.srt.utils.debug_utils import maybe_dump_jax_array

    def run_callbacks(output_dir, reverse):
        _enable_dump(monkeypatch, output_dir)
        pending = []
        monkeypatch.setattr(
            jax.debug,
            "callback",
            lambda callback, *values, **_kwargs: pending.append((callback, values)),
        )
        for token in (11, 29):
            forward_batch = SimpleNamespace(
                input_ids=jnp.array([token], dtype=jnp.int32),
                positions=jnp.array([token - 1], dtype=jnp.int32),
                seq_lens=jnp.array([token], dtype=jnp.int32),
            )
            maybe_dump_jax_array(
                jnp.array([token], dtype=jnp.int32),
                component="logits",
                name="token",
                forward_batch=forward_batch,
            )
        for callback, values in pending[:: -1 if reverse else 1]:
            callback(*(np.asarray(value) for value in values))
        rows = [
            json.loads(line)
            for line in next(output_dir.glob("manifest-*.jsonl")).read_text().splitlines()
        ]
        return {
            int(np.load(output_dir / row["filename"])[0]): row["occurrence"]
            for row in rows
        }

    in_order = run_callbacks(tmp_path / "in-order", reverse=False)
    reverse_order = run_callbacks(tmp_path / "reverse-order", reverse=True)

    assert in_order == reverse_order
    assert len(set(in_order.values())) == 2


def test_debug_dump_forward_occurrence_includes_array_shape(monkeypatch, tmp_path):
    from sgl_jax.srt.utils.debug_utils import maybe_dump_jax_array

    _enable_dump(monkeypatch, tmp_path)
    pending = []
    monkeypatch.setattr(
        jax.debug,
        "callback",
        lambda callback, *values, **_kwargs: pending.append((callback, values)),
    )
    for shape in ((2,), (1, 2)):
        maybe_dump_jax_array(
            jnp.asarray([len(shape)], dtype=jnp.int32),
            component="logits",
            name="shape_probe",
            forward_batch=SimpleNamespace(
                input_ids=jnp.asarray([1, 2], dtype=jnp.int32).reshape(shape),
                positions=jnp.asarray([0, 1], dtype=jnp.int32),
                seq_lens=jnp.asarray([2], dtype=jnp.int32),
            ),
        )
    for callback, values in pending:
        callback(*(np.asarray(value) for value in values))

    rows = [
        json.loads(line)
        for line in next(tmp_path.glob("manifest-*.jsonl")).read_text().splitlines()
    ]
    assert len({row["occurrence"] for row in rows}) == 2


def test_debug_dump_sum_does_not_evaluate_when_disabled(monkeypatch):
    from sgl_jax.srt.utils.debug_utils import maybe_dump_jax_array_sum

    class AdditionMustNotRun:
        def __add__(self, _other):
            raise AssertionError("disabled debug dump evaluated an addition")

    monkeypatch.delenv("SGLANG_JAX_DEBUG_DUMP", raising=False)
    maybe_dump_jax_array_sum(
        AdditionMustNotRun(),
        AdditionMustNotRun(),
        component="decoder_layer",
        name="hidden_states_post_mlp",
    )


def test_debug_dump_rejects_reused_directory_after_process_restart(monkeypatch, tmp_path):
    from sgl_jax.srt.utils import debug_utils

    _enable_dump(monkeypatch, tmp_path)
    monkeypatch.setattr(
        jax.debug,
        "callback",
        lambda callback, value, **_kwargs: callback(np.asarray(value)),
    )
    debug_utils.maybe_dump_jax_array(
        jnp.array([1], dtype=jnp.int32), component="dsa", name="ids"
    )

    debug_utils._DEBUG_DUMP_INITIALIZED.clear()
    debug_utils._DEBUG_DUMP_OCCURRENCES.clear()
    with pytest.raises(RuntimeError, match="already contains debug dumps"):
        debug_utils.maybe_dump_jax_array(
            jnp.array([2], dtype=jnp.int32), component="dsa", name="ids"
        )


def test_debug_dump_sanitized_names_do_not_collide(monkeypatch, tmp_path):
    from sgl_jax.srt.utils.debug_utils import maybe_dump_jax_array

    _enable_dump(monkeypatch, tmp_path)
    monkeypatch.setattr(
        jax.debug,
        "callback",
        lambda callback, value, **_kwargs: callback(np.asarray(value)),
    )

    maybe_dump_jax_array(jnp.array([1]), component="dsa", name="a/b")
    maybe_dump_jax_array(jnp.array([2]), component="dsa", name="a b")

    arrays = list(tmp_path.glob("*.npy"))
    assert len(arrays) == 2
    assert {int(np.load(path)[0]) for path in arrays} == {1, 2}


def test_debug_dump_executes_from_jitted_code(monkeypatch, tmp_path):
    from sgl_jax.srt.utils.debug_utils import maybe_dump_jax_array

    _enable_dump(monkeypatch, tmp_path)

    @jax.jit
    def dump_from_jit(value):
        forward_batch = SimpleNamespace(
            input_ids=jnp.array([7], dtype=jnp.int32),
            positions=jnp.array([3], dtype=jnp.int32),
            seq_lens=jnp.array([4], dtype=jnp.int32),
        )
        return maybe_dump_jax_array(
            value,
            component="decoder_layer",
            name="hidden_states",
            layer_id=1,
            forward_mode="extend",
            forward_batch=forward_batch,
        )

    value = jnp.array([1.0, 2.0], dtype=jnp.float32)
    np.testing.assert_array_equal(dump_from_jit(value).block_until_ready(), value)

    arrays = list(tmp_path.glob("*.npy"))
    assert len(arrays) == 1
    np.testing.assert_array_equal(np.load(arrays[0]), np.asarray(value))
