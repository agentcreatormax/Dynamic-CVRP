#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BENCH_DIR="${BENCH_DIR:-$REPO/large1_benchmark}" \
  exec "$HERE/run_benchmark_runi.sh"
