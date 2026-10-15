#!/usr/bin/env bash
set -Eeuo pipefail
# Unified VND dynamic ADD/DELETE runner for medium* and large* benches.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
BENCH_DIR="${BENCH_DIR:-$REPO/medium1_benchmark}"
CUDA_ARCH="${CUDA_ARCH:-120}"
BIN="${VND_BIN:-$ROOT/d-cvrp.out}"
BUILD="${BUILD:-1}"
CASES="${CASES:-all}"  # all, A, B, C, or comma-separated case tags
OUT_DIR="${OUT_DIR:-$ROOT/benchmark_vnd_$(basename "$BENCH_DIR")}" 

[[ -d "$BENCH_DIR/start" && -d "$BENCH_DIR/events" && -d "$BENCH_DIR/static_inputs" ]] || {
  echo "ERROR: expected start/, events/, static_inputs/ in BENCH_DIR=$BENCH_DIR" >&2; exit 2;
}
BENCH_DIR="$(cd "$BENCH_DIR" && pwd)"
if [[ "$BUILD" == 1 ]]; then
  echo "[build] VND CUDA arch sm_$CUDA_ARCH"
  make -C "$ROOT" NVCCFLAGS="-std=c++17 -O2 -arch=sm_${CUDA_ARCH}" all
fi
[[ -x "$BIN" ]] || { echo "ERROR: VND binary missing: $BIN" >&2; exit 2; }

bench_name="$(basename "$BENCH_DIR")"
idx="$(echo "$bench_name" | sed -nE 's/^(medium|large)([0-9]+)_benchmark$/\2/p')"
if [[ -n "$idx" && -f "$BENCH_DIR/start/$idx.vrp" && -f "$BENCH_DIR/start/$idx.sol" ]]; then
  start_vrp="$BENCH_DIR/start/$idx.vrp"; start_sol="$BENCH_DIR/start/$idx.sol"
else
  start_vrp=''; start_sol=''
  for v in "$BENCH_DIR"/start/*.vrp; do
    [[ -f "$v" && -f "${v%.vrp}.sol" ]] || continue
    start_vrp="$v"; start_sol="${v%.vrp}.sol"; break
  done
fi
[[ -n "$start_vrp" ]] || { echo 'ERROR: no paired start .vrp/.sol found' >&2; exit 2; }
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/benchmark_summary.csv"
echo 'implementation,case,updates,algorithm_time_sec,wall_time_sec,final_cost,final_routes,valid' > "$SUMMARY"

# One case per unique original events file. Validation ignores renumbered customer
# identifiers, since the original VND program removes deleted nodes physically.
case_selected(){
  [[ "$CASES" == all ]] && return 0
  local name="$1" short="${1%%_*}" token
  IFS=',' read -r -a choice <<< "$CASES"
  for token in "${choice[@]}"; do
    [[ "$token" == "$short" || "$token" == "$name" ]] && return 0
  done
  return 1
}

any=0
for events in "$BENCH_DIR"/events/[ABC]_*.txt; do
  [[ -f "$events" ]] || continue
  base="$(basename "$events" .txt)"
  tag="$(echo "$base" | tr '[:lower:]' '[:upper:]')"
  case_selected "$tag" || continue
  expected="$BENCH_DIR/static_inputs/${base:0:1}_after_${base:2}.vrp"
  [[ -f "$expected" ]] || { echo "ERROR: expected snapshot missing: $expected" >&2; exit 2; }
  any=1
  dir="$OUT_DIR/$tag"; mkdir -p "$dir"
  cp "$start_vrp" "$dir/current.vrp"
  cp "$start_sol" "$dir/current.sol"
  : > "$dir/run.log"
  printf 'initial_nodes,mode,total_compute_excluding_parse,parse,add/delete_node,compute_correlated,allocate_ls,load_solution,run_vnd\n' > "$dir/timecheck_LARGE.csv"
  deleted_original=()
  updates=0
  echo "[VND] $tag: $events"
  t0=$(date +%s%N)
  while read -r op a b c _; do
    [[ -z "${op:-}" || "$op" == \#* ]] && continue
    if [[ "$op" == A ]]; then
      (cd "$dir" && "$BIN" -v -t a current.vrp current.sol next.sol next.vrp "$a" "$b" "$c") >>"$dir/run.log" 2>&1
    elif [[ "$op" == D ]]; then
      original=$a; current=$original
      for old in "${deleted_original[@]}"; do
        if (( old == original )); then echo "ERROR duplicate DELETE $original" >&2; exit 2; fi
        if (( old < original )); then current=$((current-1)); fi
      done
      echo "DELETE original=$original current_id=$current" >> "$dir/run.log"
      (cd "$dir" && "$BIN" -v -t d current.vrp current.sol next.sol next.vrp "$current") >>"$dir/run.log" 2>&1
      deleted_original+=("$original")
    else
      echo "ERROR: event opcode '$op' in $events" >&2; exit 2
    fi
    mv "$dir/next.vrp" "$dir/current.vrp"
    mv "$dir/next.sol" "$dir/current.sol"
    updates=$((updates+1))
  done < "$events"
  t1=$(date +%s%N)
  cp "$dir/current.vrp" "$dir/final.vrp"
  cp "$dir/current.sol" "$dir/final.sol"
  python3 - "$dir" "$expected" "$tag" "$updates" "$t0" "$t1" "$SUMMARY" <<'PY'
import csv,sys,re,collections
from pathlib import Path
folder,expected,tag,updates,t0,t1,summary=sys.argv[1:]
folder=Path(folder); expected=Path(expected)
def sig(path):
    cap=None; sec=None; xy={}; dem={}
    for line in path.read_text().splitlines():
        s=line.strip()
        if s.startswith('CAPACITY'): cap=float(s.replace(':',' ').split()[-1])
        elif s in ('NODE_COORD_SECTION','DEMAND_SECTION','DEPOT_SECTION','EOF'):
            sec={'NODE_COORD_SECTION':'xy','DEMAND_SECTION':'dem'}.get(s)
        elif sec=='xy':
            p=s.split()
            if len(p)>=3 and p[0].isdigit(): xy[int(p[0])]=(round(float(p[1]),9),round(float(p[2]),9))
        elif sec=='dem':
            p=s.split()
            if len(p)>=2 and p[0].isdigit(): dem[int(p[0])]=round(float(p[1]),9)
    return cap,collections.Counter((*v,dem[i]) for i,v in xy.items() if i!=1)
with (folder/'timecheck_LARGE.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
if len(rows)!=int(updates):
    raise SystemExit(f'ERROR expected {updates} VND timing rows, got {len(rows)}')
alg=sum(float(r['total_compute_excluding_parse']) for r in rows)
sol=(folder/'final.sol').read_text()
costs=re.findall(r'^Cost\s+([-+\d.eE]+)',sol,re.M)
if not costs: raise SystemExit('ERROR VND final cost absent')
route_count=len(re.findall(r'^Route\s+#\d+:',sol,re.M))
valid=(sig(folder/'final.vrp')==sig(expected) and 'Feasible: NO' not in (folder/'run.log').read_text())
wall=(int(t1)-int(t0))/1e9
record=['Senior-VND',tag,updates,f'{alg:.9f}',f'{wall:.9f}',costs[-1],route_count,'YES' if valid else 'NO']
with open(summary,'a',newline='') as f:csv.writer(f).writerow(record)
print(f'{tag}: updates={updates} alg={alg:.3f}s wall={wall:.3f}s cost={costs[-1]} routes={route_count} valid={record[-1]}')
if not valid: raise SystemExit('ERROR benchmark final-state validation failed')
PY
done
(( any == 1 )) || { echo "ERROR: no events selected (CASES=$CASES)" >&2; exit 2; }
echo "[VND summary] $SUMMARY"
column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
