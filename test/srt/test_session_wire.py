"""Wire validation without loading the inference engine.

io_struct imports BaseFinishReason for annotations only. Stub that annotation
dependency, not the request dataclasses or normalization code under test.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def wire(monkeypatch):
    stub = ModuleType("sgl_jax.srt.managers.schedule_batch")
    stub.BaseFinishReason = object
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    path = Path(__file__).parents[2] / "python/sgl_jax/srt/managers/io_struct.py"
    spec = importlib.util.spec_from_file_location("_session_wire_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_full_context_session_normalizes(wire):
    req = wire.GenerateReqInput(input_ids=[1, 2, 3], session_params={"id": "a"})
    req.normalize_batch_and_arguments()
    assert req.session_params == {"id": "a"}
    assert req.input_ids == [1, 2, 3]


@pytest.mark.parametrize("params", [{}, {"id": []}, {"id": ""}, {"id": "a", "offset": 1}])
def test_invalid_session_options(wire, params):
    with pytest.raises(ValueError, match="session_params"):
        wire.GenerateReqInput(input_ids=[1], session_params=params).normalize_batch_and_arguments()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_ids": [[1], [2]]},
        {"input_ids": [1], "sampling_params": {"n": 2}},
    ],
)
def test_batch_and_parallel_sampling_rejected(wire, kwargs):
    with pytest.raises(ValueError, match="one input"):
        wire.GenerateReqInput(**kwargs, session_params={"id": "a"}).normalize_batch_and_arguments()


def test_default_and_positional_compatibility(wire):
    assert wire.TokenizedGenerateReqInput("rid").rid == "rid"
    assert wire.GenerateReqInput(1, "rid").rid == "rid"
    req = wire.GenerateReqInput(input_ids=[[1], [2]])
    req.normalize_batch_and_arguments()
    assert req.session_params is None and req.batch_size == 2
    assert wire.OpenSessionReqInput(request_id="open").session_id is None
