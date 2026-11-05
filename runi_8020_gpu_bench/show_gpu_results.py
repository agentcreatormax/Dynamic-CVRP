#!/usr/bin/env python3
"""Validate one GPU 80/20 run and save machine-readable metrics.

The comparison target in this harness is the archived full-route solution in
reference/full.vrp. It is a feasible reference, not a certified BKS.
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

if len(sys.argv) != 4:
    raise SystemExit("Usage: show_gpu_results.py metadata.json output_dir reference/full.vrp")

meta_path = Path(sys.argv[1])
out = Path(sys.argv[2])
reference = Path(sys.argv[3])
meta = json.loads(meta_path.read_text())
required = (
    'source_instance', 'initial_customers', 'num_adds', 'final_customers',
    'reference_full_cost_rounded_euc2d', 'reference_full_routes',
)
for key in required:
    if key not in meta:
        raise SystemExit(f'ERROR: metadata missing {key}')
if meta['initial_customers'] + meta['num_adds'] != meta['final_customers']:
    raise SystemExit('ERROR: inconsistent metadata customer counts')

summary_path = out / 'summary.csv'
log_path = out / 'run.log'
if not summary_path.is_file() or not log_path.is_file():
    raise SystemExit('ERROR: summary.csv or run.log is missing')
with summary_path.open(newline='') as f:
    rows = list(csv.reader(f))
# The binary's summary.csv is written as (metric,value), one row per metric.
metrics = {row[0]: row[1] for row in rows if len(row) >= 2 and row[0] != 'metric'}
log = log_path.read_text(errors='replace')
transactions = re.findall(
    r'\[transaction\].*?cost\s+[-+0-9.eE]+->([-+0-9.eE]+)\s+routes=(\d+)', log
)
if not transactions:
    raise SystemExit('ERROR: final [transaction] cost/routes line is absent')
final_cost = int(round(float(transactions[-1][0])))
route_count = int(transactions[-1][1])

for required_output in ('final.sol', 'final.vrp'):
    if not (out / required_output).is_file():
        raise SystemExit(f'ERROR: {required_output} is missing')

# Check the final VRP describes exactly the complete customer set.
def parse_vrp_signature(path):
    capacity = None
    mode = None
    coords = {}
    demands = {}
    dimension = None
    for raw in path.read_text(errors='replace').splitlines():
        line = raw.strip()
        if line.startswith('DIMENSION'):
            dimension = int(line.replace(':', ' ').split()[-1])
        elif line.startswith('CAPACITY'):
            capacity = float(line.replace(':', ' ').split()[-1])
        elif line in ('NODE_COORD_SECTION', 'DEMAND_SECTION', 'DEPOT_SECTION', 'EOF'):
            mode = {'NODE_COORD_SECTION': 'coords', 'DEMAND_SECTION': 'dem'}.get(line)
        elif mode == 'coords':
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                coords[int(parts[0])] = (round(float(parts[1]), 9), round(float(parts[2]), 9))
        elif mode == 'dem':
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                demands[int(parts[0])] = round(float(parts[1]), 9)
    if not (dimension and capacity is not None and len(coords) == len(demands) == dimension):
        raise ValueError(f'VRP malformed or incomplete: {path}')
    nodes = Counter((coords[key][0], coords[key][1], demands[key]) for key in coords if key != 1)
    return capacity, dimension, nodes

capacity, dimension, nodes = parse_vrp_signature(out / 'final.vrp')
ref_capacity, ref_dimension, ref_nodes = parse_vrp_signature(reference)
if dimension != meta['final_customers'] + 1 or capacity != ref_capacity or nodes != ref_nodes or dimension != ref_dimension:
    raise SystemExit('ERROR: final.vrp does not contain the identical full customer set and capacity as the reference')

if '[GPU-UNPLACED AFTER-REPAIR] count=0' not in log:
    raise SystemExit('ERROR: GPU unplaced count is nonzero or missing')
if '[AUDIT AFTER-REPAIR] bad=0' not in log or '[AUDIT AFTER-REFINE] bad=0' not in log:
    raise SystemExit('ERROR: GPU audit is missing or invalid')

reference_cost = int(meta['reference_full_cost_rounded_euc2d'])
if reference_cost <= 0:
    raise SystemExit('ERROR: invalid archived reference cost')
gap_pct = 100.0 * (final_cost - reference_cost) / reference_cost

# The archive reference is explicitly not a certified BKS. Use the separate
# current CVRPLIB BKS value from metadata for the manuscript's BKS-gap range.
bks_cost_raw = meta.get('official_bks_cost_rounded_euc2d')
bks_cost = int(bks_cost_raw) if bks_cost_raw is not None else None
bks_gap_pct = (100.0 * (final_cost - bks_cost) / bks_cost) if bks_cost else None

# Parse all available kernel-stage metrics as numbers for safe aggregation.
metric_values = {}
for key, value in metrics.items():
    try:
        metric_values[key] = float(value)
    except (TypeError, ValueError):
        metric_values[key] = value

def metric_number(key):
    value = metric_values.get(key)
    return float(value) if isinstance(value, (int, float)) else None

parts = [metric_number(k) for k in ('delete_total_ms', 'add_total_ms', 'refine_total_ms')]
total_gpu_ms = sum(parts) if all(v is not None for v in parts) else None
wall_seconds_path = out / 'wall_time_seconds.txt'
wall_seconds = float(wall_seconds_path.read_text().strip()) if wall_seconds_path.is_file() else None

run_metrics = {
    'instance': meta['source_instance'],
    'run_id': out.name,
    'validated': True,
    'initial_customers': int(meta['initial_customers']),
    'add_customers': int(meta['num_adds']),
    'final_customers': int(meta['final_customers']),
    'final_cost': final_cost,
    'route_count': route_count,
    'archived_reference_cost': reference_cost,
    'archived_reference_is_certified_bks': bool(meta.get('reference_full_cost_is_verified_BKS', False)),
    'gap_vs_archived_reference_pct': gap_pct,
    'official_bks_cost': bks_cost,
    'official_bks_last_updated': meta.get('official_bks_last_updated'),
    'official_bks_source_url': meta.get('official_bks_source_url'),
    'gap_vs_official_bks_pct': bks_gap_pct,
    'metrics': metric_values,
    'total_gpu_ms': total_gpu_ms,
    'wall_time_seconds': wall_seconds,
}
(out / 'run_metrics.json').write_text(json.dumps(run_metrics, indent=2, sort_keys=True) + '\n')

print('\n========== Runi ADD 80/20 single-run result ==========')
print(f"Instance:                  {meta['source_instance']}")
print(f"Run:                       {out.name}")
print(f"Initial customers:         {meta['initial_customers']}")
print(f"ADD customers:             {meta['num_adds']}")
print(f"Final customers:           {meta['final_customers']}")
print(f"GPU final cost:            {final_cost}")
print(f"GPU final routes:          {route_count}")
print(f"Archived reference cost:   {reference_cost} (not certified BKS)")
print(f"Gap vs archived reference: {gap_pct:+.3f}%")
if bks_cost is not None:
    print(f"Official CVRPLIB BKS:      {bks_cost} (updated {meta.get('official_bks_last_updated')})")
    print(f"Gap vs official BKS:       {bks_gap_pct:+.3f}%")
if total_gpu_ms is not None:
    print(f"Total GPU stages:          {total_gpu_ms:.3f} ms")
if wall_seconds is not None:
    print(f"External wall time:        {wall_seconds:.0f} s")
for key in ('delete_total_ms', 'add_total_ms', 'scheduler_task_avg_us',
            'worker_task_avg_us', 'refine_total_ms', 'move89_ms',
            'move1swap_ms', 'move7_ms'):
    if key in metric_values:
        print(f'{key:26s}: {metric_values[key]}')
print('Final snapshot:            same customer set and capacity as reference')
print('GPU audits:                both passed; no unplaced customers')
print('Machine-readable output:   run_metrics.json')
