#!/usr/bin/env bash
# Start the same CPU topology as epd_sim.sh without running a benchmark.
set -euo pipefail
exec bash "$(dirname "$0")/epd_sim.sh" --serve-only
