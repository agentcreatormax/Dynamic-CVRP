#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# toy.sh
# Tiny Runi benchmark:
#   - 10 starting vertices total (1 depot + 9 customers)
#   - toy_2_ADD:    2 additions
#   - toy_2_DELETE: 2 deletions
#
# Put toy_benchmark/ directly inside your Dynamic-CVRP-Runi project and run:
#   cd toy_benchmark
#   CUDA_ARCH=120 ./toy.sh
# ============================================================================

TOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$TOY_DIR/.." && pwd)}"

CUDA_ARCH="${CUDA_ARCH:-120}"
PASSES="${PASSES:-1}"
HMAX="${HMAX:-5}"
REFINE_K="${REFINE_K:-5}"
QUEUE_CAP="${QUEUE_CAP:-5}"

# prashant-lab convenience: CUDA 13.2 may be installed but absent from PATH.
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
    echo "Set CUDA_HOME/PATH before running toy.sh."
    exit 1
fi

[[ -f "$PROJECT_DIR/Makefile" ]] || {
    echo "ERROR: Makefile not found in project directory: $PROJECT_DIR"
    echo "Expected layout: Dynamic-CVRP-Runi/toy_benchmark/toy.sh"
    exit 1
}

START_VRP="$TOY_DIR/start/toy_start.vrp"
START_SOL="$TOY_DIR/start/toy_start.sol"

OUT="$TOY_DIR/toy_results"
rm -rf "$OUT"
mkdir -p "$OUT"

SUMMARY="$OUT/toy_benchmark_summary.csv"
RESULTS_TXT="$OUT/toy_results.txt"

# Same top-level benchmark summary format as before.
echo "implementation,case,updates,algorithm_time_sec,wall_time_sec,final_cost,final_routes,valid" > "$SUMMARY"
: > "$RESULTS_TXT"

echo "============================================================"
echo "TOY Dynamic-CVRP benchmark"
echo "============================================================"
echo "Project    : $PROJECT_DIR"
echo "Benchmark  : $TOY_DIR"
echo "CUDA       : $(command -v nvcc)"
echo "CUDA arch  : sm_$CUDA_ARCH"
echo "PASSES     : $PASSES"
echo "HMAX       : $HMAX"
echo "REFINE_K   : $REFINE_K"
echo

cd "$PROJECT_DIR"

make clean
make CUDA_ARCH="$CUDA_ARCH" PROFILE=1

run_case() {
    local tag="$1"
    local events="$2"
    local expected="$3"
    local updates="$4"

    local dir="$OUT/$tag"
    rm -rf "$dir"
    mkdir -p "$dir"

    echo
    echo "========== $tag =========="

    local start_ns end_ns
    start_ns="$(date +%s%N)"

    ./dcvrp.out \
        --vrp "$START_VRP" \
        --sol "$START_SOL" \
        --events "$TOY_DIR/events/$events" \
        --out-dir "$dir" \
        --engine gpu \
        --passes "$PASSES" \
        --hmax "$HMAX" \
        --refine-k "$REFINE_K" \
        --queue-cap "$QUEUE_CAP" \
        --snap-vrp \
        --verbose \
        >"$dir/run.log" 2>&1

    end_ns="$(date +%s%N)"

    # Keep the raw detailed profiler and also give it a toy-specific name.
    if [[ -f "$dir/summary.csv" ]]; then
        cp "$dir/summary.csv" "$dir/toy_summary.csv"
    fi

    python3 - \
        "$dir/summary.csv" \
        "$dir/run.log" \
        "$dir/final.vrp" \
        "$TOY_DIR/static_inputs/$expected" \
        "$tag" "$updates" "$start_ns" "$end_ns" \
        "$SUMMARY" "$RESULTS_TXT" <<'PY'
import csv
import os
import re
import sys

(summary, runlog, got_vrp, expected_vrp, tag, updates,
 start_ns, end_ns, out_csv, out_txt) = sys.argv[1:]

updates = int(updates)
wall_sec = (int(end_ns) - int(start_ns)) / 1e9

def vrp_signature(path):
    coords = {}
    dem = {}
    mode = None
    cap = None

    with open(path) as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue

            if s.startswith("CAPACITY"):
                cap = float(s.replace(":", " ").split()[-1])
            elif s == "NODE_COORD_SECTION":
                mode = "coord"
            elif s == "DEMAND_SECTION":
                mode = "demand"
            elif s == "DEPOT_SECTION":
                mode = None
            elif s == "EOF":
                break
            elif mode == "coord":
                p = s.split()
                if len(p) >= 3:
                    coords[int(p[0])] = (float(p[1]), float(p[2]))
            elif mode == "demand":
                p = s.split()
                if len(p) >= 2:
                    dem[int(p[0])] = float(p[1])

    items = []
    for node, xy in coords.items():
        if node == 1:
            continue
        items.append((
            round(xy[0], 9),
            round(xy[1], 9),
            round(dem.get(node, 0.0), 9)
        ))

    return cap, sorted(items)

# --------------------------------------------------------------
# Detailed event profiler.
# We only require the timing fields needed to reconstruct algo time.
# Other requested profiler fields remain untouched in summary.csv:
#
#   scheduler_task_avg_us
#   worker_task_avg_us
#   avg_active_route_worker_warps
#   max_active_route_worker_warps
#   avg_active_scheduler_warps
#   max_active_scheduler_warps
#   refine_total_ms
#   move89_ms
#   move1swap_ms
#   move7_ms
# --------------------------------------------------------------
with open(summary, newline="") as f:
    rows = list(csv.reader(f))

metrics = {r[0]: r[1] for r in rows[1:] if len(r) >= 2}

needed = ["delete_total_ms", "add_total_ms", "refine_total_ms"]
missing = [k for k in needed if k not in metrics]
if missing:
    raise SystemExit(
        f"{tag}: missing required metrics in {summary}: {missing}"
    )

algorithm_time_sec = (
    float(metrics["delete_total_ms"]) +
    float(metrics["add_total_ms"]) +
    float(metrics["refine_total_ms"])
) / 1000.0

with open(runlog, errors="replace") as f:
    log = f.read()

matches = re.findall(
    r'\[transaction\].*?cost\s+[-+0-9.eE]+->([-+0-9.eE]+)\s+routes=(\d+)',
    log
)

if not matches:
    raise SystemExit(
        f"{tag}: could not parse final cost/routes from {runlog}"
    )

final_cost_s, final_routes_s = matches[-1]
final_cost = int(round(float(final_cost_s)))
final_routes = int(final_routes_s)

valid = (
    os.path.isfile(got_vrp)
    and vrp_signature(got_vrp) == vrp_signature(expected_vrp)
)
valid_s = "YES" if valid else "NO"

row = [
    "P0-Runi",
    tag,
    updates,
    f"{algorithm_time_sec:.9f}",
    f"{wall_sec:.9f}",
    final_cost,
    final_routes,
    valid_s,
]

with open(out_csv, "a", newline="") as f:
    csv.writer(f).writerow(row)

line = (
    f"{tag}: updates={updates} "
    f"alg={algorithm_time_sec:.6f}s "
    f"wall={wall_sec:.6f}s "
    f"cost={final_cost} "
    f"routes={final_routes} "
    f"valid={valid_s}"
)

print(line)
with open(out_txt, "a") as f:
    f.write(line + "\n")

if not valid:
    raise SystemExit(
        f"{tag}: final VRP does not match expected static input"
    )
PY
}

run_case "toy_2_ADD" \
    "toy_2_add.txt" \
    "toy_after_2_add.vrp" \
    2

run_case "toy_2_DELETE" \
    "toy_2_delete.txt" \
    "toy_after_2_delete.vrp" \
    2

echo
echo "============================================================"
echo "TOY RESULTS"
echo "============================================================"
column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo
echo "Detailed profiler files:"
echo "  $OUT/toy_2_ADD/toy_summary.csv"
echo "  $OUT/toy_2_DELETE/toy_summary.csv"
echo
echo "Benchmark summary:"
echo "  $SUMMARY"
echo
echo "Human-readable results:"
echo "  $RESULTS_TXT"
