import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_PATH = ROOT / "scripts/kernels/validate_glm52_dsa_boundary_dumps.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("boundary_validator", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_forward(directory, occurrence, positions, logical_ids, counts, slots):
    directory.mkdir(exist_ok=True)
    mask = np.ones(len(positions), dtype=np.bool_)
    tensors = (
        ("debug_context", None, "token_valid_mask", mask),
        ("debug_context", None, "token_positions", np.asarray(positions, dtype=np.int32)),
        ("dsa_selection", 0, "logical_topk_ids", np.asarray(logical_ids, dtype=np.int32)),
        ("dsa_selection", 0, "selected_counts", np.asarray(counts, dtype=np.int32)),
        ("dsa_selection", 0, "physical_slots", np.asarray(slots, dtype=np.int32)),
        ("debug_context", None, "forward_complete", np.asarray(1, dtype=np.int8)),
    )
    rows = []
    for index, (component, layer, name, array) in enumerate(tensors):
        filename = f"{occurrence}-{index}.npy"
        np.save(directory / filename, array, allow_pickle=False)
        rows.append(
            {
                "component": component,
                "layer": layer,
                "forward_mode": "extend",
                "name": name,
                "occurrence": occurrence,
                "process": 0,
                "filename": filename,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
    with (directory / "manifest-p00000.jsonl").open("a", encoding="utf-8") as manifest:
        for row in rows:
            manifest.write(json.dumps(row) + "\n")


def test_boundary_validator_accepts_causal_full_and_truncated_selections(tmp_path):
    validator = _load_validator()
    _write_forward(
        tmp_path,
        1,
        positions=[2, 3],
        logical_ids=[[2, 0, 1, -1], [3, 1, 0, 2]],
        counts=[3, 4],
        slots=[[12, 10, 11, 0], [13, 11, 10, 12]],
    )
    _write_forward(
        tmp_path,
        2,
        positions=[4],
        logical_ids=[[4, 1, 3, 0]],
        counts=[4],
        slots=[[24, 21, 23, 20]],
    )

    report = validator.validate_boundary_dumps(
        tmp_path,
        index_topk=4,
        selection_layer=0,
        required_positions={2, 3, 4},
    )

    assert report["passed"] is True
    assert report["active_query_count"] == 3
    assert report["clipped_query_count"] == 1
    assert report["observed_position_max"] == 4


def test_boundary_validator_rejects_future_logical_id(tmp_path):
    validator = _load_validator()
    _write_forward(
        tmp_path,
        3,
        positions=[4],
        logical_ids=[[5, 1, 3, 0]],
        counts=[4],
        slots=[[25, 21, 23, 20]],
    )

    report = validator.validate_boundary_dumps(
        tmp_path,
        index_topk=4,
        selection_layer=0,
        required_positions={4},
    )

    assert report["passed"] is False
    assert any("future token" in failure for failure in report["failures"])
