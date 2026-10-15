#!/usr/bin/env bash
set -Eeuo pipefail
# Existing Runi engine, with archived CVRP final-state references (NOT certified BKS).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO/Dynamic-CVRP-Runi}"
GPU_BIN="${GPU_BIN:-$PROJECT_ROOT/dcvrp.out}"
CASE="${CASE:-all}" # x1001, medium1, medium2, medium3, all, or canonical name
CUDA_ARCH="${CUDA_ARCH:-120}"; BUILD="${BUILD:-0}"
PASSES="${PASSES:-1}"; HMAX="${HMAX:-5}"; REFINE_K="${REFINE_K:-5}"; QUEUE_CAP="${QUEUE_CAP:-5}"
OUT_ROOT="${OUT_ROOT:-$HERE/results_gpu}"

case "$CASE" in
    x1001|medium1|X-n1001-k43) tags=(X-n1001-k43);;
    medium2|XL-n10001-k1570) tags=(XL-n10001-k1570);;
    medium3|Flanders2) tags=(Flanders2);;
    both) tags=(XL-n10001-k1570 Flanders2);;  # legacy behavior
    all) tags=(X-n1001-k43 XL-n10001-k1570 Flanders2);;
    *) echo "ERROR: unknown CASE=$CASE; use x1001|medium1|medium2|medium3|all" >&2; exit 2;;
esac

if [[ "$BUILD" == 1 ]]; then
   echo "[build] $PROJECT_ROOT CUDA_ARCH=$CUDA_ARCH"
   make -C "$PROJECT_ROOT" CUDA_ARCH="$CUDA_ARCH" PROFILE=1
fi
[[ -x "$GPU_BIN" ]] || { echo "ERROR: GPU executable missing: $GPU_BIN (set BUILD=1)" >&2; exit 2; }
mkdir -p "$OUT_ROOT"
for tag in "${tags[@]}"; do
    case_dir="$HERE/$tag/case"
    meta="$HERE/$tag/metadata.json"
    reference="$HERE/$tag/reference/full.vrp"
    for path in "$case_dir/initial.vrp" "$case_dir/initial.sol" "$case_dir/events_add_to_full.txt" "$meta" "$reference"; do
        [[ -s "$path" ]] || { echo "ERROR missing $path" >&2; exit 2; }
    done
    out="$OUT_ROOT/$tag"; mkdir -p "$out"
    echo
    echo "[GPU] $tag | initial 80% + remaining 20% ADD | output=$out"
    echo "[start] $(date -Is)"
    start=$SECONDS
    "$GPU_BIN" \
        --vrp "$case_dir/initial.vrp" \
        --sol "$case_dir/initial.sol" \
        --events "$case_dir/events_add_to_full.txt" \
        --engine gpu \
        --passes "$PASSES" \
        --hmax "$HMAX" \
        --refine-k "$REFINE_K" \
        --queue-cap "$QUEUE_CAP" \
        --out-dir "$out" \
        --snap-vrp \
        2>&1 | tee "$out/run.log"
    echo "[finish] $(date -Is) elapsed=$((SECONDS-start))s"
    python3 "$HERE/show_gpu_results.py" "$meta" "$out" "$reference" \
        | tee "$out/reference_comparison.txt"
done
