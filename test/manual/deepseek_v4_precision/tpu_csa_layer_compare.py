import json
from pathlib import Path
import sys
import numpy as np
from common import compare

root = Path(sys.argv[1])
b = json.loads((root / "baseline/result.json").read_text())
c = json.loads((root / "candidate/result.json").read_text())
assert b["checkpoint"] == c["checkpoint"]
assert json.loads((root / "baseline/weights.json").read_text()) == json.loads(
    (root / "candidate/weights.json").read_text()
)
base = {r["name"]: r for r in b["cases"]}
assert base.keys() == {r["name"] for r in c["cases"]}
rows = []
for r in c["cases"]:
    name = r["name"]
    old = base[name]
    assert old["input_hashes"] == r["input_hashes"], name
    actual = np.load(root / f"candidate/{name}-output.npy")
    expected = np.load(root / f"baseline/{name}-output.npy")
    metric = compare(actual, expected)
    assert metric["nonfinite"] == 0, metric
    rows.append(
        dict(
            case=name,
            baseline_ms=old["median_ms"],
            candidate_ms=r["median_ms"],
            speedup=old["median_ms"] / r["median_ms"],
            baseline_capacity=old["compressed_capacity"],
            candidate_capacity=r["compressed_capacity"],
            cache_state_bitwise_equal=old["updated_pool_hashes"] == r["updated_pool_hashes"],
            output=metric,
        )
    )
result = dict(
    baseline_sha=b["source_sha"],
    candidate_sha=c["source_sha"],
    rows=rows,
    boundary="Real attention weights, TP8 production module, synthetic identical inputs/history. No model quality verdict.",
)
(root / "comparison.json").write_text(json.dumps(result, indent=2))
print("TPU_LAYER_COMPARISON", json.dumps(result), flush=True)
