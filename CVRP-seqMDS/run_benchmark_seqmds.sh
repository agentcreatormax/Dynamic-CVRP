#!/usr/bin/env bash
set -Eeuo pipefail
# Static final-state comparison; no dynamic ADD/DELETE timings.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
BENCH_DIR="${BENCH_DIR:-$REPO/medium1_benchmark}"
SOLVER="${SOLVER:-seq}"      # seq or par
BUILD="${BUILD:-1}"
CASES="${CASES:-all}"        # all, A, B, C or full tag
NTHREADS="${NTHREADS:-1}"; ROUND="${ROUND:-1}"
OUT_DIR="${OUT_DIR:-$ROOT/benchmark_${SOLVER}_$(basename "$BENCH_DIR")}" 
[[ -d "$BENCH_DIR/static_inputs" && -d "$BENCH_DIR/events" ]] || {
 echo "ERROR: BENCH_DIR missing events/static_inputs: $BENCH_DIR" >&2; exit 2;
}
BENCH_DIR="$(cd "$BENCH_DIR" && pwd)"
case "$SOLVER" in
  seq)
    BIN="${SEQ_BIN:-$ROOT/seqMDS.out}"
    if [[ "$BUILD" == 1 ]]; then
      command -v g++ >/dev/null || { echo 'ERROR g++ missing' >&2; exit 2; }
      g++ -O3 -std=c++14 "$ROOT/seqMDS.cpp" -o "$BIN"
    fi
    impl=seqMDS ;;
  par)
    BIN="${PAR_BIN:-$ROOT/parMDS.out}"
    if [[ "$BUILD" == 1 ]]; then
      command -v nvc++ >/dev/null || { echo 'ERROR nvc++ missing; set SOLVER=seq for g++' >&2; exit 2; }
      nvc++ -O3 -std=c++14 -acc=multicore "$ROOT/parMDS.cpp" -o "$BIN"
    fi
    impl=parMDS-ACC ;;
  *) echo 'ERROR: SOLVER=seq or SOLVER=par' >&2; exit 2 ;;
esac
[[ -x "$BIN" ]] || { echo "ERROR: binary not found: $BIN" >&2; exit 2; }
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/benchmark_summary.csv"
echo 'implementation,case,updates,algorithm_time_sec,wall_time_sec,final_cost,final_routes,valid' > "$SUMMARY"
case_selected(){
 [[ "$CASES" == all ]] && return 0
 local tag="$1" item
 IFS=',' read -r -a choices <<< "$CASES"
 for item in "${choices[@]}"; do [[ "$item" == "$tag" || "$item" == "${tag%%_*}" ]] && return 0; done
 return 1
}
found=0
for ev in "$BENCH_DIR"/events/[ABC]_*.txt; do
 [[ -f "$ev" ]] || continue
 base="$(basename "$ev" .txt)"; tag="$(echo "$base" | tr '[:lower:]' '[:upper:]')"
 case_selected "$tag" || continue
 vrp="$BENCH_DIR/static_inputs/${base:0:1}_after_${base:2}.vrp"
 [[ -f "$vrp" ]] || { echo "ERROR missing $vrp" >&2; exit 2; }
 updates=$(grep -Ec '^[[:space:]]*[AD][[:space:]]' "$ev" || true)
 dir="$OUT_DIR/$tag"; mkdir -p "$dir"
 echo "[$impl] $tag: static full reoptimization of $vrp"
 t0=$(date +%s%N)
 if [[ "$SOLVER" == par ]]; then
    "$BIN" "$vrp" -nthreads "$NTHREADS" -round "$ROUND" >"$dir/final.sol" 2>"$dir/run.stats"
 else
    "$BIN" "$vrp" -round "$ROUND" >"$dir/final.sol" 2>"$dir/run.stats"
 fi
 t1=$(date +%s%N)
 python3 - "$dir" "$tag" "$updates" "$t0" "$t1" "$impl" "$SUMMARY" <<'PY'
import csv,re,sys
from pathlib import Path
folder,tag,updates,t0,t1,impl,summary=sys.argv[1:]
s=(Path(folder)/'run.stats').read_text()
mc=re.search(r'\bCost\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)',s)
mt=re.search(r'\bTime\(seconds\)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)',s)
if not (mc and mt): raise SystemExit(f'ERROR: stats parsing failed: {folder}/run.stats')
if not re.search(r'\bVALID\b',s) or re.search(r'\bINVALID\b',s):
    raise SystemExit(f'ERROR: solver reported invalid solution for {tag}')
sol=(Path(folder)/'final.sol').read_text()
rr=len(re.findall(r'^Route\s+#\d+:',sol,re.M))
cost=mc.group(3);alg=mt.group(3);wall=(int(t1)-int(t0))/1e9
with open(summary,'a',newline='') as f:
    csv.writer(f).writerow([impl,tag,updates,alg,f'{wall:.9f}',cost,rr,'YES'])
print(f'{tag}: cost={cost} routes={rr} solver={alg}s wall={wall:.3f}s valid=YES')
PY
 found=1
done
(( found == 1 )) || { echo "ERROR: no cases selected: $CASES" >&2; exit 2; }
echo "[static summary] $SUMMARY"
column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
