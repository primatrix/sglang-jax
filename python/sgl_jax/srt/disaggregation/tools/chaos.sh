#!/usr/bin/env bash
# Stage 4 H-F: PD chaos harness.
#
# Three scenarios, all designed to be re-run independently. Use a
# dedicated chaos cluster — do NOT run against a production pod.
#
#   ./chaos.sh kill_p      # SIGKILL one prefill pod, verify recovery
#   ./chaos.sh drop_dcn    # iptables drop cross-host traffic 30s
#   ./chaos.sh bootstrap   # restart bootstrap for 60s
#
# Each scenario logs to /tmp/pd_chaos_<scenario>.log and exits non-zero
# if recovery fails. Recovery criteria are documented inline and
# match the SLA window in docs/rfc/2026-05-25-pd-hardening.md.

set -euo pipefail

SCENARIO=${1:-help}
KUBECTL=${KUBECTL:-kubectl}
ROUTER_URL=${ROUTER_URL:-http://router:8001}
# Maximum tolerated extra failures (failures_after - failures_before).
MAX_FAIL_DELTA=${MAX_FAIL_DELTA:-5}

_failures_total() {
    # Sum pd_transfer_failures_total{...} across all label sets.
    curl -sf "$ROUTER_URL/metrics" \
        | awk '/^pd_transfer_failures_total\{/ { s += $NF } END { print (s == "" ? 0 : s) }'
}

_router_5xx_rate() {
    # Read 1-minute 5xx counter; require external scrape if absent.
    curl -sf "$ROUTER_URL/metrics" \
        | awk '/^http_responses_total\{.*status="5/ { s += $NF } END { print (s == "" ? 0 : s) }'
}

_assert_recovered() {
    local label="$1"
    local fail_before="$2"
    local fail_after
    fail_after=$(_failures_total)
    local delta
    delta=$(awk -v a="$fail_after" -v b="$fail_before" 'BEGIN { print a - b }')
    echo "[$label] failures delta: $delta (max $MAX_FAIL_DELTA)"
    awk -v d="$delta" -v m="$MAX_FAIL_DELTA" 'BEGIN { exit (d > m) }' || {
        echo "[$label] FAIL: recovery exceeded failure budget"
        exit 1
    }
}

case "$SCENARIO" in
    kill_p)
        # Pick one prefill pod, kill -9, wait 60s, verify failure
        # budget not exceeded.
        FAIL_BEFORE=$(_failures_total)
        POD=$($KUBECTL get pods -l pd-role=prefill -o name | head -1)
        echo "[kill_p] failures before: $FAIL_BEFORE, killing $POD"
        $KUBECTL exec "$POD" -- bash -c "kill -9 1" || true
        sleep 60
        _assert_recovered kill_p "$FAIL_BEFORE"
        ;;

    drop_dcn)
        # Pick a P pod and a D pod, iptables drop both directions for
        # 30s. Recovery: in-flight transfers go through pull_timeout →
        # FAILED → upstream retry; final consistency OK.
        FAIL_BEFORE=$(_failures_total)
        P=$($KUBECTL get pods -l pd-role=prefill -o name | head -1)
        D=$($KUBECTL get pods -l pd-role=decode -o name | head -1)
        P_IP=$($KUBECTL get $P -o jsonpath='{.status.podIP}')
        D_IP=$($KUBECTL get $D -o jsonpath='{.status.podIP}')
        echo "[drop_dcn] failures before: $FAIL_BEFORE, dropping $P_IP <-> $D_IP for 30s"
        $KUBECTL exec "$P" -- iptables -I INPUT  -s "$D_IP" -j DROP
        $KUBECTL exec "$D" -- iptables -I INPUT  -s "$P_IP" -j DROP
        sleep 30
        $KUBECTL exec "$P" -- iptables -D INPUT  -s "$D_IP" -j DROP
        $KUBECTL exec "$D" -- iptables -D INPUT  -s "$P_IP" -j DROP
        # Wait one full reap cycle so the timeout path completes.
        sleep 35
        _assert_recovered drop_dcn "$FAIL_BEFORE"
        ;;

    bootstrap)
        # Stop bootstrap deployment 60s, then bring it back. Router
        # should retry; new requests during the outage queue at the
        # router layer.
        FAIL_BEFORE=$(_failures_total)
        echo "[bootstrap] failures before: $FAIL_BEFORE"
        $KUBECTL scale deploy/pd-bootstrap --replicas=0
        sleep 60
        $KUBECTL scale deploy/pd-bootstrap --replicas=1
        # Allow bootstrap to come back up + heartbeats to re-register.
        sleep 30
        _assert_recovered bootstrap "$FAIL_BEFORE"
        ;;

    *)
        cat <<EOF
Usage: $0 {kill_p|drop_dcn|bootstrap}

Env knobs:
  KUBECTL=kubectl         kubectl binary
  ROUTER_URL=http://router:8001
  MAX_FAIL_DELTA=5        max acceptable new failures during recovery
EOF
        exit 2
        ;;
esac
