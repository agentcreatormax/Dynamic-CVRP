#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../" && pwd)"
BENCH_DIR="${BENCH_DIR:-$REPO/medium1_benchmark}" CASE="${CASE:-A}" \
  exec "$HERE/run_pass_sweep.sh"
