#!/usr/bin/env bash
set -Eeuo pipefail
# Works with medium1/2/3 or large1/2/3/4/5 benchmark folders.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$HERE/.." && pwd)}"
REPO="$(cd "$PROJECT_DIR/.." && pwd)"
BENCH_DIR="${BENCH_DIR:-$REPO/medium1_benchmark}"
CASE="${CASE:-A}"  # A, B, C or complete tag (A_50_ADD etc.)
CUDA_ARCH="${CUDA_ARCH:-120}"
HMAX="${HMAX:-5}"; REFINE_K="${REFINE_K:-5}"; QUEUE_CAP="${QUEUE_CAP:-5}"
PASSES_VALUES="${PASSES_VALUES:-1 5 10 15 20}" # space or comma separated
BUILD="${BUILD:-1}"
[[ -d "$BENCH_DIR/start" && -d "$BENCH_DIR/events" && -d "$BENCH_DIR/static_inputs" ]] || {
  echo "ERROR: invalid BENCH_DIR=$BENCH_DIR" >&2; exit 2;
}
BENCH_DIR="$(cd "$BENCH_DIR" && pwd)"
name="$(basename "$BENCH_DIR")"
idx="$(echo "$name" | sed -nE 's/^(medium|large)([0-9]+)_benchmark$/\2/p')"
if [[ -n "$idx" && -f "$BENCH_DIR/start/$idx.vrp" && -f "$BENCH_DIR/start/$idx.sol" ]]; then
 start_vrp="$BENCH_DIR/start/$idx.vrp"; start_sol="$BENCH_DIR/start/$idx.sol"
else
 start_vrp=''; start_sol=''
 for v in "$BENCH_DIR"/start/*.vrp; do
  [[ -f "$v" && -f "${v%.vrp}.sol" ]] || continue
  start_vrp="$v"; start_sol="${v%.vrp}.sol"; break
 done
fi
[[ -n "$start_vrp" ]] || { echo 'ERROR: no paired start VRP/SOL' >&2; exit 2; }

event=''; tag=''; expected=''
for path in "$BENCH_DIR"/events/[ABC]_*.txt; do
 [[ -f "$path" ]] || continue
 base="$(basename "$path" .txt)"
 title="$(echo "$base" | tr '[:lower:]' '[:upper:]')"
 if [[ "$CASE" == "$title" || "$CASE" == "${title%%_*}" ]]; then
   event="$path"; tag="$title"; expected="$BENCH_DIR/static_inputs/${base:0:1}_after_${base:2}.vrp"
   break
 fi
done
[[ -f "$event" && -f "$expected" ]] || { echo "ERROR: CASE=$CASE not found for $BENCH_DIR" >&2; exit 2; }

OUT_DIR="${OUT_DIR:-$HERE/passes_sweep_${name}_${tag}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/passes_sweep.csv"
echo 'passes,final_cost,algorithm_time_sec,final_routes,valid' > "$CSV"
PASSES_VALUES="${PASSES_VALUES//,/ }"
read -r -a passes <<< "$PASSES_VALUES"
[[ ${#passes[@]} -gt 0 ]] || { echo 'ERROR: empty PASSES_VALUES' >&2; exit 2; }
if [[ "$BUILD" == 1 ]]; then
 make -C "$PROJECT_DIR" CUDA_ARCH="$CUDA_ARCH" PROFILE=1
fi
GPU_BIN="${GPU_BIN:-$PROJECT_DIR/dcvrp.out}"
[[ -x "$GPU_BIN" ]] || { echo "ERROR: GPU binary missing $GPU_BIN" >&2; exit 2; }
echo "[passes] BENCH=$BENCH_DIR CASE=$tag PASSES=${passes[*]}"
for p in "${passes[@]}"; do
 [[ "$p" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid passes '$p'" >&2; exit 2; }
 run="$OUT_DIR/pass_$p"; mkdir -p "$run"
 echo "[passes] $tag $p"
 set +e
 "$GPU_BIN" \
   --vrp "$start_vrp" --sol "$start_sol" --events "$event" \
   --out-dir "$run" --engine gpu --passes "$p" --hmax "$HMAX" \
   --refine-k "$REFINE_K" --queue-cap "$QUEUE_CAP" --snap-vrp --verbose \
   >"$run/run.log" 2>&1
 rc=$?
 set -e
 python3 - "$p" "$rc" "$run" "$expected" "$CSV" <<'PY'
import csv,re,sys,collections
from pathlib import Path
p,rc,run,reference,out=sys.argv[1:]; run=Path(run)
rc=int(rc)
def sig(path):
 cap=None; mode=None;xy={};dem={}
 for line in path.read_text(errors='replace').splitlines():
  s=line.strip()
  if s.startswith('CAPACITY'):cap=float(s.replace(':',' ').split()[-1])
  elif s in ('NODE_COORD_SECTION','DEMAND_SECTION','DEPOT_SECTION','EOF'):
   mode={'NODE_COORD_SECTION':'xy','DEMAND_SECTION':'dem'}.get(s)
  elif mode=='xy':
   q=s.split()
   if len(q)>=3 and q[0].isdigit():xy[int(q[0])]=(round(float(q[1]),9),round(float(q[2]),9))
  elif mode=='dem':
   q=s.split()
   if len(q)>=2 and q[0].isdigit():dem[int(q[0])]=round(float(q[1]),9)
 return cap,collections.Counter((*coord,dem[i]) for i,coord in xy.items() if i!=1)
cost='';routes='';alg='';valid='NO'
log=(run/'run.log').read_text(errors='replace') if (run/'run.log').exists() else ''
if rc==0 and (run/'summary.csv').exists():
 with (run/'summary.csv').open(newline='') as f:metrics={r[0]:r[1] for r in list(csv.reader(f))[1:] if len(r)>=2}
 if all(k in metrics for k in ('delete_total_ms','add_total_ms','refine_total_ms')):
  alg=sum(float(metrics[k]) for k in ('delete_total_ms','add_total_ms','refine_total_ms'))/1000
 m=re.findall(r'\[transaction\].*?cost\s+[-+0-9.eE]+->([-+0-9.eE]+)\s+routes=(\d+)',log)
 if m:cost=int(round(float(m[-1][0])));routes=int(m[-1][1])
 try:
  valid='YES' if sig(run/'final.vrp')==sig(Path(reference)) and '[AUDIT AFTER-REFINE] bad=0' in log else 'NO'
 except (FileNotFoundError,KeyError,ValueError):pass
with open(out,'a',newline='') as f:csv.writer(f).writerow([p,cost,f'{alg:.9f}' if alg!='' else '',routes,valid])
print(f'passes={p} exit={rc} cost={cost or "-"} time={alg if alg!="" else "-"} routes={routes or "-"} valid={valid}')
PY
done
# Plot if matplotlib exists; preserve complete CSV even when not installed.
python3 - "$CSV" "$OUT_DIR/passes_vs_cost_time.png" "$tag" <<'PY'
import csv,sys
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
except ImportError:
 print('[plot] matplotlib not installed; CSV is still available')
 raise SystemExit(0)
csvfile,image,title=sys.argv[1:]
rows=[]
with open(csvfile,newline='') as f:
 for row in csv.DictReader(f):
  if row['valid']=='YES':
   try:rows.append((int(row['passes']),float(row['final_cost']),float(row['algorithm_time_sec'])))
   except ValueError:pass
if not rows:
 print('[plot] no valid passes, skipping')
 raise SystemExit(0)
x,y,t=zip(*rows)
fig,ax1=plt.subplots(figsize=(9,5));ax2=ax1.twinx()
a=ax1.plot(x,y,'o-',label='Cost');b=ax2.plot(x,t,'s-',label='Algorithm time')
ax1.set_xlabel('Max refinement passes');ax1.set_ylabel('Final cost');ax2.set_ylabel('Algorithm time (s)')
ax1.set_xticks(x);ax1.grid(True,alpha=.25);ax1.legend(a+b,[v.get_label() for v in a+b]);ax1.set_title(title)
fig.tight_layout();fig.savefig(image,dpi=150);print('[plot]',image)
PY
echo "[passes] $CSV"
