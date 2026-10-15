#!/usr/bin/env bash
set -uo pipefail

# ============================================================================
# Large5 Dynamic-CVRP load test
#
# Workloads:
#   50 ... 409600 total events
#   exactly 50% DELETE + 50% ADD
#
# Each run starts from the SAME Large5 start/5.vrp + start/5.sol.
#
# Outputs:
#   large5_load_results/benchmark_summary.csv        old benchmark format
#   large5_load_results/load_test_results.csv        plotting-friendly format
#   large5_load_results/results.txt                  human-readable rows
#   large5_load_results/events_vs_algorithm_time.png log2 x-axis
#   large5_load_results/events_vs_algorithm_time_linear.png
#   large5_load_results/EVENTS_N/summary.csv          detailed Runi profiler
# ============================================================================

set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$HERE/.." && pwd)}"
BENCH_DIR="${BENCH_DIR:-$(cd "$PROJECT_DIR/.." && pwd)/large5_benchmark}"

CUDA_ARCH="${CUDA_ARCH:-120}"
PASSES="${PASSES:-1}"
HMAX="${HMAX:-5}"
REFINE_K="${REFINE_K:-5}"
QUEUE_CAP="${QUEUE_CAP:-5}"

# Default to the tested practical range; override with SIZES="..." for any subset.
SIZES_TEXT="${SIZES:-50 100 200 400 800 1600 3200 6400 12800}"
read -r -a SIZES <<< "${SIZES_TEXT//,/ }"
BUILD="${BUILD:-1}"
OUT="${OUT_DIR:-$HERE/large5_load_results_$(date +%Y%m%d_%H%M%S)}"

START_VRP="$BENCH_DIR/start/5.vrp"
START_SOL="$BENCH_DIR/start/5.sol"

SUMMARY="$OUT/benchmark_summary.csv"
LOADCSV="$OUT/load_test_results.csv"
RESULTS="$OUT/results.txt"

# CUDA 13.2 convenience on prashant-lab.
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
    exit 1
fi

[[ -f "$PROJECT_DIR/Makefile" ]] || {
    echo "ERROR: Makefile not found: $PROJECT_DIR/Makefile"
    echo "Place large5_load_test directly inside Dynamic-CVRP-Runi."
    exit 1
}
[[ -f "$START_VRP" ]] || { echo "ERROR: missing $START_VRP"; exit 1; }
[[ -f "$START_SOL" ]] || { echo "ERROR: missing $START_SOL"; exit 1; }

for n in "${SIZES[@]}"; do
    [[ -f "$HERE/events/large5_mixed_${n}.txt" ]] || {
        echo "ERROR: missing event file large5_mixed_${n}.txt"
        exit 1
    }
done

mkdir -p "$OUT"
# New timestamped default prevents accidental removal of previous expensive runs.

# Same top-level format used by the previous benchmark runners.
echo "implementation,case,updates,algorithm_time_sec,wall_time_sec,final_cost,final_routes,valid" > "$SUMMARY"

# More useful load-test table, including failures.
echo "events,delete_events,add_events,algorithm_time_sec,wall_time_sec,final_cost,final_routes,valid,status" > "$LOADCSV"
: > "$RESULTS"

echo "============================================================"
echo "Large5 Dynamic-CVRP load test"
echo "============================================================"
echo "Project   : $PROJECT_DIR"
echo "Benchmark : $BENCH_DIR"
echo "CUDA arch : sm_$CUDA_ARCH"
echo "PASSES    : $PASSES"
echo "HMAX      : $HMAX"
echo "REFINE_K  : $REFINE_K"
echo "Sizes     : ${SIZES[*]}"
echo

# Compile exactly once. PASSES/HMAX/etc. are runtime parameters.
if [[ "$BUILD" == 1 ]]; then
    make -C "$PROJECT_DIR" CUDA_ARCH="$CUDA_ARCH" PROFILE=1 || {
        echo "ERROR: build failed"; exit 1;
    }
fi
[[ -x "$PROJECT_DIR/dcvrp.out" ]] || { echo "ERROR: missing $PROJECT_DIR/dcvrp.out" >&2; exit 1; }

run_one() {
    local n="$1"
    local half=$((n / 2))
    local tag="EVENTS_${n}"
    local dir="$OUT/$tag"
    local event_file="$HERE/events/large5_mixed_${n}.txt"

    rm -rf "$dir"
    mkdir -p "$dir"

    echo
    echo "============================================================"
    echo "$tag  ($half DELETE + $half ADD)"
    echo "============================================================"

    local start_ns end_ns rc
    start_ns="$(date +%s%N)"

    set +e
    (
      cd "$PROJECT_DIR"
      ./dcvrp.out \
        --vrp "$START_VRP" \
        --sol "$START_SOL" \
        --events "$event_file" \
        --out-dir "$dir" \
        --engine gpu \
        --passes "$PASSES" \
        --hmax "$HMAX" \
        --refine-k "$REFINE_K" \
        --queue-cap "$QUEUE_CAP" \
        --snap-vrp \
        --verbose
    ) >"$dir/run.log" 2>&1
    rc=$?
    set -e

    end_ns="$(date +%s%N)"

    python3 - \
      "$n" "$half" "$rc" "$start_ns" "$end_ns" \
      "$dir" "$SUMMARY" "$LOADCSV" "$RESULTS" <<'PY'
import csv
import os
import re
import sys

n = int(sys.argv[1])
half = int(sys.argv[2])
rc = int(sys.argv[3])
start_ns = int(sys.argv[4])
end_ns = int(sys.argv[5])
run_dir, summary_out, load_out, txt_out = sys.argv[6:10]

wall = (end_ns - start_ns) / 1e9
summary = os.path.join(run_dir, "summary.csv")
logfile = os.path.join(run_dir, "run.log")
final_vrp = os.path.join(run_dir, "final.vrp")

algorithm = ""
cost = ""
routes = ""
valid = "NO"
status = "FAILED"

# Exit 0 already means the controller's internal route validation passed.
if rc == 0 and os.path.isfile(summary):
    with open(summary, newline="") as f:
        rows = list(csv.reader(f))
    m = {r[0]: r[1] for r in rows[1:] if len(r) >= 2}

    needed = ["delete_total_ms", "add_total_ms", "refine_total_ms"]
    if all(k in m for k in needed):
        algorithm = (
            float(m["delete_total_ms"])
            + float(m["add_total_ms"])
            + float(m["refine_total_ms"])
        ) / 1000.0

if os.path.isfile(logfile):
    log = open(logfile, errors="replace").read()
    matches = re.findall(
        r'\[transaction\].*?cost\s+[-+0-9.eE]+->([-+0-9.eE]+)\s+routes=(\d+)',
        log
    )
    if matches:
        c, r = matches[-1]
        cost = int(round(float(c)))
        routes = int(r)

# Equal DELETE + ADD means final customer count should remain 500000.
# final.vrp uses DIMENSION = customers + depot = 500001.
dimension_ok = False
if rc == 0 and os.path.isfile(final_vrp):
    with open(final_vrp, errors="replace") as f:
        for line in f:
            if line.upper().startswith("DIMENSION"):
                try:
                    dim = int(line.replace(":", " ").split()[-1])
                    dimension_ok = (dim == 500001)
                except Exception:
                    pass
                break

if rc == 0 and algorithm != "" and cost != "" and routes != "" and dimension_ok:
    valid = "YES"
    status = "OK"

with open(summary_out, "a", newline="") as f:
    csv.writer(f).writerow([
        "P0-Runi",
        f"LOAD_{n}_MIXED",
        n,
        "" if algorithm == "" else f"{algorithm:.9f}",
        f"{wall:.9f}",
        cost,
        routes,
        valid,
    ])

with open(load_out, "a", newline="") as f:
    csv.writer(f).writerow([
        n, half, half,
        "" if algorithm == "" else f"{algorithm:.9f}",
        f"{wall:.9f}",
        cost, routes, valid, status
    ])

line = (
    f"events={n} delete={half} add={half} "
    f"alg={'FAILED' if algorithm == '' else f'{algorithm:.6f}s'} "
    f"wall={wall:.6f}s cost={cost or '-'} routes={routes or '-'} "
    f"valid={valid} status={status}"
)
print(line)
with open(txt_out, "a") as f:
    f.write(line + "\n")

if status != "OK":
    print(f"  -> run failed or output was incomplete; inspect {logfile}")
PY
}

for n in "${SIZES[@]}"; do
    run_one "$n"
done

if python3 -c 'import matplotlib' >/dev/null 2>&1; then
    python3 "$HERE/plot_large5_load.py" "$LOADCSV" "$OUT"
else
    echo "[plot] matplotlib missing; CSV data remains available."
fi

echo
echo "============================================================"
echo "LOAD TEST RESULTS"
echo "============================================================"
column -s, -t "$LOADCSV" 2>/dev/null || cat "$LOADCSV"
echo
echo "Top-level benchmark summary : $SUMMARY"
echo "Load-test CSV              : $LOADCSV"
echo "Human-readable results     : $RESULTS"
echo "Log2 plot                  : $OUT/events_vs_algorithm_time.png"
echo "Linear plot                : $OUT/events_vs_algorithm_time_linear.png"
