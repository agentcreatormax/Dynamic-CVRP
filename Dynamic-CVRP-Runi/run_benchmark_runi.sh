#!/usr/bin/env bash
set -Eeuo pipefail
# Unified Runi batch transaction benchmark for medium and large instances.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BENCH_DIR="${BENCH_DIR:-$REPO/medium1_benchmark}"
GPU_BIN="${GPU_BIN:-$HERE/dcvrp.out}"
CUDA_ARCH="${CUDA_ARCH:-120}"
PASSES="${PASSES:-1}"; HMAX="${HMAX:-5}"; REFINE_K="${REFINE_K:-5}"; QUEUE_CAP="${QUEUE_CAP:-5}"
BUILD="${BUILD:-1}"; CASES="${CASES:-all}"
OUT_DIR="${OUT_DIR:-$HERE/benchmark_runi_$(basename "$BENCH_DIR")}" 
[[ -d "$BENCH_DIR/start" && -d "$BENCH_DIR/events" && -d "$BENCH_DIR/static_inputs" ]] || {
 echo "ERROR: invalid BENCH_DIR=$BENCH_DIR" >&2; exit 2;
}
BENCH_DIR="$(cd "$BENCH_DIR" && pwd)"
idx="$(basename "$BENCH_DIR" | sed -nE 's/^(medium|large)([0-9]+)_benchmark$/\2/p')"
if [[ -n "$idx" && -f "$BENCH_DIR/start/$idx.vrp" && -f "$BENCH_DIR/start/$idx.sol" ]]; then
 start_vrp="$BENCH_DIR/start/$idx.vrp";start_sol="$BENCH_DIR/start/$idx.sol"
else
 start_vrp='';start_sol=''
 for v in "$BENCH_DIR"/start/*.vrp; do
  [[ -f "$v" && -f "${v%.vrp}.sol" ]] || continue
  start_vrp="$v";start_sol="${v%.vrp}.sol";break
 done
fi
[[ -n "$start_vrp" ]] || { echo 'ERROR: no matched start VRP+SOL' >&2;exit 2; }
if [[ "$BUILD" == 1 ]]; then make -C "$HERE" CUDA_ARCH="$CUDA_ARCH" PROFILE=1; fi
[[ -x "$GPU_BIN" ]] || { echo "ERROR GPU binary missing $GPU_BIN" >&2;exit 2; }
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/benchmark_summary.csv"
echo 'implementation,case,updates,algorithm_time_sec,wall_time_sec,final_cost,final_routes,valid' > "$SUMMARY"
case_selected(){
 [[ "$CASES" == all ]] && return 0
 local tag="$1" item
 IFS=',' read -r -a selected <<< "$CASES"
 for item in "${selected[@]}"; do [[ "$item" == "$tag" || "$item" == "${tag%%_*}" ]] && return 0; done
 return 1
}
found=0
for ev in "$BENCH_DIR"/events/[ABC]_*.txt; do
 [[ -f "$ev" ]] || continue
 base="$(basename "$ev" .txt)";tag="$(echo "$base" | tr '[:lower:]' '[:upper:]')"
 case_selected "$tag" || continue
 expected="$BENCH_DIR/static_inputs/${base:0:1}_after_${base:2}.vrp"
 [[ -f "$expected" ]] || { echo "ERROR missing $expected" >&2;exit 2; }
 updates=$(grep -Ec '^[[:space:]]*[AD][[:space:]]' "$ev" || true)
 dir="$OUT_DIR/$tag";mkdir -p "$dir"
 echo "[GPU Runi] $tag from $start_vrp: $updates updates"
 t0=$(date +%s%N)
 "$GPU_BIN" --vrp "$start_vrp" --sol "$start_sol" --events "$ev" \
   --out-dir "$dir" --engine gpu --passes "$PASSES" --hmax "$HMAX" \
   --refine-k "$REFINE_K" --queue-cap "$QUEUE_CAP" --snap-vrp --verbose \
   >"$dir/run.log" 2>&1
 t1=$(date +%s%N)
 python3 - "$dir" "$expected" "$tag" "$updates" "$t0" "$t1" "$SUMMARY" <<'PY'
import csv,re,sys,collections
from pathlib import Path
folder,expected,tag,updates,t0,t1,summary=sys.argv[1:];folder=Path(folder)
with (folder/'summary.csv').open(newline='') as f:
 m={r[0]:r[1] for r in list(csv.reader(f))[1:] if len(r)>=2}
needed=('delete_total_ms','add_total_ms','refine_total_ms')
if not all(k in m for k in needed):raise SystemExit('ERROR GPU phase timing missing')
alg=sum(float(m[k]) for k in needed)/1000
log=(folder/'run.log').read_text(errors='replace')
match=re.findall(r'\[transaction\].*?cost\s+[-+0-9.eE]+->([-+0-9.eE]+)\s+routes=(\d+)',log)
if not match:raise SystemExit('ERROR GPU transaction result missing')
cost=int(round(float(match[-1][0])));routes=int(match[-1][1])
def sig(path):
 cap=None;xy={};dem={};mode=None
 for line in Path(path).read_text().splitlines():
  s=line.strip()
  if s.startswith('CAPACITY'):cap=float(s.replace(':',' ').split()[-1])
  elif s in ('NODE_COORD_SECTION','DEMAND_SECTION','DEPOT_SECTION','EOF'):mode={'NODE_COORD_SECTION':'xy','DEMAND_SECTION':'dem'}.get(s)
  elif mode=='xy':
   p=s.split()
   if len(p)>=3 and p[0].isdigit():xy[int(p[0])]=(round(float(p[1]),9),round(float(p[2]),9))
  elif mode=='dem':
   p=s.split()
   if len(p)>=2 and p[0].isdigit():dem[int(p[0])]=round(float(p[1]),9)
 return cap,collections.Counter((*v,dem[i]) for i,v in xy.items() if i!=1)
valid=sig(folder/'final.vrp')==sig(expected) and '[AUDIT AFTER-REPAIR] bad=0' in log and '[AUDIT AFTER-REFINE] bad=0' in log
wall=(int(t1)-int(t0))/1e9
with open(summary,'a',newline='') as f:
 csv.writer(f).writerow(['P0-Runi',tag,updates,f'{alg:.9f}',f'{wall:.9f}',cost,routes,'YES' if valid else 'NO'])
print(f'{tag}: alg={alg:.5f}s wall={wall:.5f}s cost={cost} routes={routes} valid={valid}')
if not valid:raise SystemExit('ERROR final-state validation failed')
PY
 found=1
done
(( found == 1 )) || { echo "ERROR no cases selected: $CASES" >&2;exit 2; }
echo "[GPU benchmark summary] $SUMMARY"
column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
