#!/usr/bin/env python3
"""Evaluate GPU final snapshot against archived complete-instance CVRP reference.

Not an online DVRP BKS comparison: service time and dispatch are absent.
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

if len(sys.argv) != 4:
    raise SystemExit("Usage: show_gpu_results.py metadata.json output_dir reference/full.vrp")
meta = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2]); reference = Path(sys.argv[3])
required = ('source_instance','initial_customers','num_adds','final_customers',
            'reference_full_cost_rounded_euc2d','reference_full_routes')
for key in required:
    if key not in meta: raise SystemExit(f'ERROR metadata missing {key}')
if meta['initial_customers'] + meta['num_adds'] != meta['final_customers']:
    raise SystemExit('ERROR inconsistent metadata customer counts')
with (out/'summary.csv').open(newline='') as f:
    rows=list(csv.reader(f))
metrics={r[0]:r[1] for r in rows if len(r)>=2 and r[0]!='metric'}
log=(out/'run.log').read_text(errors='replace')
ms=re.findall(r'\[transaction\].*?cost\s+[-+0-9.eE]+->([-+0-9.eE]+)\s+routes=(\d+)', log)
if not ms: raise SystemExit('ERROR: final transaction line absent')
cost=int(round(float(ms[-1][0]))); routes=int(ms[-1][1])
if not (out/'final.sol').is_file() or not (out/'final.vrp').is_file():
    raise SystemExit('ERROR: final.sol or final.vrp missing')

def sig(vrp):
    capacity=None; mode=None; coords={};demands={}; count=None
    for line in vrp.read_text(errors='replace').splitlines():
        s=line.strip()
        if s.startswith('DIMENSION'): count=int(s.replace(':',' ').split()[-1])
        elif s.startswith('CAPACITY'): capacity=float(s.replace(':',' ').split()[-1])
        elif s in ('NODE_COORD_SECTION','DEMAND_SECTION','DEPOT_SECTION','EOF'):
            mode={'NODE_COORD_SECTION':'coords','DEMAND_SECTION':'dem'}.get(s)
        elif mode=='coords':
            p=s.split()
            if len(p)>=3 and p[0].isdigit():coords[int(p[0])]=(round(float(p[1]),9),round(float(p[2]),9))
        elif mode=='dem':
            p=s.split()
            if len(p)>=2 and p[0].isdigit():demands[int(p[0])]=round(float(p[1]),9)
    if not (count and len(coords)==len(demands)==count):
        raise ValueError(f'VRP malformed or incomplete: {vrp}')
    return capacity,count,Counter((*coords[k],demands[k]) for k in coords if k!=1)

cap,dim,nodes=sig(out/'final.vrp')
ref_cap,ref_dim,ref_nodes=sig(reference)
if dim!=meta['final_customers']+1 or cap!=ref_cap or nodes!=ref_nodes or dim!=ref_dim:
    raise SystemExit('ERROR: final.vrp does not contain the identical full customer set/capacity')
if '[GPU-UNPLACED AFTER-REPAIR] count=0' not in log:
    raise SystemExit('ERROR: GPU unplaced customer count is nonzero or missing')
if '[AUDIT AFTER-REPAIR] bad=0' not in log or '[AUDIT AFTER-REFINE] bad=0' not in log:
    raise SystemExit('ERROR: GPU audit missing or invalid')
ref=meta['reference_full_cost_rounded_euc2d']
if ref<=0:raise SystemExit('ERROR: invalid reference cost')
print('\n========== Runi ADD 80/20 result ==========')
print(f"Instance:          {meta['source_instance']}")
print(f"Initial customers: {meta['initial_customers']}")
print(f"ADD customers:     {meta['num_adds']}")
print(f"Final customers:   {meta['final_customers']}")
print(f"GPU final cost:    {cost}")
print(f"GPU final routes:  {routes}")
print(f"Reference cost:    {ref} (archived full solution, NOT certified BKS)")
print(f"Gap vs reference:  {100.0*(cost-ref)/ref:+.3f}%")
for key in ('delete_total_ms','add_total_ms','scheduler_task_avg_us','worker_task_avg_us',
            'refine_total_ms','move89_ms','move1swap_ms','move7_ms'):
    if key in metrics: print(f'{key:26s}: {metrics[key]}')
print('Final snapshot:    same customer set and capacity as reference')
print('GPU audits:        both passed, no unplaced customers')
