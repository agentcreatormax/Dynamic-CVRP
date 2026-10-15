#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$HERE/.." && pwd)}"
CUDA_ARCH="${CUDA_ARCH:-120}"
REBUILD="${REBUILD:-0}"

# prashant-lab: CUDA 13.2 exists under /usr/local but may not be in PATH.
if ! command -v nvcc >/dev/null 2>&1; then
    if [[ -x /usr/local/cuda-13.2/bin/nvcc ]]; then
        export CUDA_HOME=/usr/local/cuda-13.2
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
        hash -r
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        export CUDA_HOME=/usr/local/cuda
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
        hash -r
    fi
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found."
    echo "On prashant-lab CUDA is normally /usr/local/cuda-13.2."
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/Makefile" ]]; then
    echo "ERROR: Makefile not found in $PROJECT_DIR"
    echo
    echo "Place dcvrp_visualizer directly inside Dynamic-CVRP-Runi:"
    echo "  Dynamic-CVRP-Runi/dcvrp_visualizer/start_visualizer.sh"
    exit 1
fi

if [[ "$REBUILD" == "1" || ! -x "$PROJECT_DIR/dcvrp.out" ]]; then
    echo "[visualizer] building Runi for sm_$CUDA_ARCH ..."
    (
        cd "$PROJECT_DIR"
        make clean
        make CUDA_ARCH="$CUDA_ARCH" PROFILE=1
    )
fi

echo "[visualizer] nvcc: $(command -v nvcc)"
echo "[visualizer] project: $PROJECT_DIR"
echo "[visualizer] binary: $PROJECT_DIR/dcvrp.out"
echo

export PROJECT_DIR
exec python3 "$HERE/app.py"
