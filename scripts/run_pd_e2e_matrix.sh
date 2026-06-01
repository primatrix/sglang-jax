#!/usr/bin/env bash
# PD e2e test matrix driver.
#
# Runs every script under
# python/sgl_jax/srt/disaggregation/tools/e2e/test_*.py
# against a deployed PD topology and prints a summary table.
#
# Topology is configured via environment variables (or repeat-flag
# CLI; this wrapper threads them through). Required:
#
#   export PD_P_URLS="http://10.0.0.1:30100,http://10.0.0.2:30100"
#   export PD_D_URLS="http://10.0.0.3:30200,http://10.0.0.4:30200"
#   export PD_BOOTSTRAP_URL="http://10.0.0.1:8998"
#   export SGL_JAX_PD_SHARED_SECRET=...           # optional
#   export PD_E2E_OUT_DIR=/tmp/pd_e2e              # optional
#
# Then:
#   bash scripts/run_pd_e2e_matrix.sh
# or to run a subset:
#   bash scripts/run_pd_e2e_matrix.sh topology correctness

set -uo pipefail

: "${PD_P_URLS?must set PD_P_URLS (comma-separated)}"
: "${PD_D_URLS?must set PD_D_URLS (comma-separated)}"
: "${PD_BOOTSTRAP_URL?must set PD_BOOTSTRAP_URL}"
OUT_DIR=${PD_E2E_OUT_DIR:-/tmp/pd_e2e}
mkdir -p "$OUT_DIR"

PY=${PYTHON:-python3}

build_args() {
    local args=()
    IFS=',' read -ra P <<< "$PD_P_URLS"
    for u in "${P[@]}"; do args+=(--p-url "$u"); done
    IFS=',' read -ra D <<< "$PD_D_URLS"
    for u in "${D[@]}"; do args+=(--d-url "$u"); done
    args+=(--bootstrap-url "$PD_BOOTSTRAP_URL")
    [[ -n "${SGL_JAX_PD_SHARED_SECRET:-}" ]] && \
        args+=(--shared-secret "$SGL_JAX_PD_SHARED_SECRET")
    printf '%q ' "${args[@]}"
}
COMMON_ARGS=$(build_args)

# Module paths in run order (P0 → I3 → orthogonal).
TESTS=(
    "test_topology_multi_pd"
    "test_correctness_byte_equal"
    "test_long_prompt"
    "test_concurrency"
    "test_chaos_wrapper"
    "test_orthogonal_dp"
)

# If positional args given, restrict to those (substring match).
FILTER=("$@")
match() {
    local name="$1"
    [[ ${#FILTER[@]} -eq 0 ]] && return 0
    for f in "${FILTER[@]}"; do [[ "$name" == *"$f"* ]] && return 0; done
    return 1
}

declare -a SUMMARY
START_TS=$(date +%s)

for t in "${TESTS[@]}"; do
    match "$t" || continue
    echo "=========================================="
    echo "[$(date +%T)] running $t"
    echo "=========================================="
    out_json="$OUT_DIR/$t.json"
    if eval "$PY -m sgl_jax.srt.disaggregation.tools.e2e.$t \
            $COMMON_ARGS --out $out_json"; then
        SUMMARY+=("PASS  $t")
    else
        SUMMARY+=("FAIL  $t")
    fi
done

END_TS=$(date +%s)
echo
echo "=========================================="
echo "PD e2e matrix summary  (duration $((END_TS-START_TS))s)"
echo "=========================================="
for line in "${SUMMARY[@]}"; do echo "$line"; done
echo
echo "JSON reports in: $OUT_DIR"

if printf '%s\n' "${SUMMARY[@]}" | grep -q "^FAIL"; then
    exit 1
fi
exit 0
