#!/usr/bin/env bash
set -Eeuo pipefail

# Run the same 80/20 snapshot benchmark repeatedly. Every repetition reads the
# same initial VRP/SOL and event list, but writes to its own output directory.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO/Dynamic-CVRP-Runi}"
GPU_BIN="${GPU_BIN:-$PROJECT_ROOT/dcvrp.out}"
CASE="${CASE:-all}" # x1001, medium1, medium2, medium3, all, or canonical name
CUDA_ARCH="${CUDA_ARCH:-120}"
BUILD="${BUILD:-0}"
PASSES="${PASSES:-15}"
REPEATS="${REPEATS:-5}"
HMAX="${HMAX:-5}"
REFINE_K="${REFINE_K:-5}"
QUEUE_CAP="${QUEUE_CAP:-5}"
OUT_ROOT="${OUT_ROOT:-$HERE/results_gpu_15passes_5runs}"

case "$CASE" in
    x1001|medium1|X-n1001-k43) tags=(X-n1001-k43);;
    medium2|XL-n10001-k1570) tags=(XL-n10001-k1570);;
    medium3|Flanders2) tags=(Flanders2);;
    both) tags=(XL-n10001-k1570 Flanders2);;
    all) tags=(X-n1001-k43 XL-n10001-k1570 Flanders2);;
    *) echo "ERROR: unknown CASE=$CASE; use x1001|medium1|medium2|medium3|all" >&2; exit 2;;
esac

[[ "$PASSES" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: PASSES must be a positive integer" >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: REPEATS must be a positive integer" >&2; exit 2; }

if [[ "$BUILD" == 1 ]]; then
   echo "[build] $PROJECT_ROOT CUDA_ARCH=$CUDA_ARCH"
   make -C "$PROJECT_ROOT" CUDA_ARCH="$CUDA_ARCH" PROFILE=1
fi
[[ -x "$GPU_BIN" ]] || { echo "ERROR: GPU executable missing or not executable: $GPU_BIN (set BUILD=1 or GPU_BIN=...)" >&2; exit 2; }
mkdir -p "$OUT_ROOT"

for tag in "${tags[@]}"; do
    case_dir="$HERE/$tag/case"
    meta="$HERE/$tag/metadata.json"
    reference="$HERE/$tag/reference/full.vrp"
    for path in "$case_dir/initial.vrp" "$case_dir/initial.sol" "$case_dir/events_add_to_full.txt" "$meta" "$reference"; do
        [[ -s "$path" ]] || { echo "ERROR: missing or empty input $path" >&2; exit 2; }
    done

    case_out="$OUT_ROOT/$tag"
    mkdir -p "$case_out"
    echo
    echo "============================================================"
    echo "[GPU 80/20] $tag | passes=$PASSES | repetitions=$REPEATS"
    echo "[inputs] same initial VRP/SOL and ADD event list in all repetitions"
    echo "[output] $case_out"
    echo "============================================================"

    # Refuse to mix new data with old results. Choose a fresh OUT_ROOT to rerun,
    # or explicitly set OVERWRITE=1 to replace this case's run directories.
    if [[ "${OVERWRITE:-0}" == 1 ]]; then
        for ((run=1; run<=REPEATS; run++)); do
            run_id="$(printf 'run_%02d' "$run")"
            rm -rf -- "$case_out/$run_id"
        done
    else
        for ((run=1; run<=REPEATS; run++)); do
            run_id="$(printf 'run_%02d' "$run")"
            if [[ -e "$case_out/$run_id" ]]; then
                echo "ERROR: $case_out/$run_id already exists. Use a fresh OUT_ROOT or set OVERWRITE=1." >&2
                exit 2
            fi
        done
    fi

    for ((run=1; run<=REPEATS; run++)); do
        run_id="$(printf 'run_%02d' "$run")"
        run_out="$case_out/$run_id"
        mkdir -p "$run_out"
        echo
        echo "[RUN $run/$REPEATS] $tag/$run_id | $(date -Is)"
        start_time="$(date +%s.%N)"

        # Runs are deliberately sequential: no overlapping Runi invocations.
        "$GPU_BIN" \
            --vrp "$case_dir/initial.vrp" \
            --sol "$case_dir/initial.sol" \
            --events "$case_dir/events_add_to_full.txt" \
            --engine gpu \
            --passes "$PASSES" \
            --hmax "$HMAX" \
            --refine-k "$REFINE_K" \
            --queue-cap "$QUEUE_CAP" \
            --out-dir "$run_out" \
            --snap-vrp \
            2>&1 | tee "$run_out/run.log"

        end_time="$(date +%s.%N)"
        elapsed="$(awk -v a="$start_time" -v b="$end_time" 'BEGIN { printf "%.3f", b-a }')"
        printf '%s\n' "$elapsed" > "$run_out/wall_time_seconds.txt"
        echo "[RUN COMPLETE] $run_id wall_time=${elapsed}s"

        # This validates the final customer set and both GPU audits, then writes
        # machine-readable run_metrics.json for the five-run aggregation.
        python3 "$HERE/show_gpu_results.py" "$meta" "$run_out" "$reference" \
            | tee "$run_out/reference_comparison.txt"
    done

    python3 "$HERE/aggregate_gpu_runs.py" "$case_out" --expected-runs "$REPEATS"
done

# A compact root-level CSV is handy to send back after all requested cases run.
python3 "$HERE/aggregate_gpu_runs.py" "$OUT_ROOT" --combine-cases --expected-runs "$REPEATS"

echo
echo "All requested ${REPEATS}-run benchmarks completed and validated."
echo "Aggregates: $OUT_ROOT/*/five_run_summary.csv"
echo "Combined:   $OUT_ROOT/all_cases_summary.csv"
