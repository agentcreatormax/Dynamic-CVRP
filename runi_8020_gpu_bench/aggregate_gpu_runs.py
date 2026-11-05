#!/usr/bin/env python3
"""Aggregate independent, validated Runi runs (mean, sample variance, SD, and gap ranges)."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def load_case(case_dir: Path, expected_runs: int):
    records = []
    for index in range(1, expected_runs + 1):
        run_dir = case_dir / f'run_{index:02d}'
        metrics_path = run_dir / 'run_metrics.json'
        if not metrics_path.is_file():
            raise SystemExit(f'ERROR: missing validated run metrics: {metrics_path}')
        record = json.loads(metrics_path.read_text())
        if record.get('validated') is not True:
            raise SystemExit(f'ERROR: run is not validated: {metrics_path}')
        if record.get('instance') is None or record.get('final_cost') is None:
            raise SystemExit(f'ERROR: incomplete metrics: {metrics_path}')
        records.append(record)

    instances = {r['instance'] for r in records}
    refs = {int(r['archived_reference_cost']) for r in records}
    bks_values = {int(r['official_bks_cost']) for r in records if r.get('official_bks_cost') is not None}
    customer_shapes = {(r['initial_customers'], r['add_customers'], r['final_customers']) for r in records}
    if len(instances) != 1 or len(refs) != 1 or len(customer_shapes) != 1:
        raise SystemExit(f'ERROR: runs in {case_dir} do not share the same benchmark definition/reference')
    if len(bks_values) > 1:
        raise SystemExit(f'ERROR: runs in {case_dir} use different BKS values')
    if bks_values and any(r.get('official_bks_cost') is None for r in records):
        raise SystemExit(f'ERROR: some runs in {case_dir} are missing the official BKS value')
    return records


def summarize_case(case_dir: Path, expected_runs: int):
    records = load_case(case_dir, expected_runs)
    costs = [int(r['final_cost']) for r in records]
    archive_gaps = [float(r['gap_vs_archived_reference_pct']) for r in records]
    reference_cost = int(records[0]['archived_reference_cost'])

    bks_cost = records[0].get('official_bks_cost')
    bks_gaps = [float(r['gap_vs_official_bks_pct']) for r in records] if bks_cost is not None else []

    mean_cost = statistics.mean(costs)
    # Unbiased sample variance (n-1 denominator) estimates run-to-run variability.
    sample_variance = statistics.variance(costs) if len(costs) > 1 else 0.0
    sample_sd = math.sqrt(sample_variance)

    def mean_metric(key):
        values = []
        for record in records:
            value = record.get('metrics', {}).get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return statistics.mean(values) if len(values) == len(records) else None

    mean_add_ms = mean_metric('add_total_ms')
    mean_refine_ms = mean_metric('refine_total_ms')
    mean_delete_ms = mean_metric('delete_total_ms')
    # Stage total is stored at the top level of each per-run JSON.
    total_values = [r.get('total_gpu_ms') for r in records]
    mean_gpu_total_ms = statistics.mean(total_values) if all(
        isinstance(v, (int, float)) for v in total_values
    ) else None
    wall_values = [r.get('wall_time_seconds') for r in records]
    mean_wall_seconds = statistics.mean(wall_values) if all(
        isinstance(v, (int, float)) for v in wall_values
    ) else None

    def signed_range(values):
        if not values:
            return None
        return f'{min(values):+.3f}% to {max(values):+.3f}%'

    first = records[0]
    return {
        'instance': first['instance'],
        'runs': len(records),
        'initial_customers': first['initial_customers'],
        'add_customers': first['add_customers'],
        'final_customers': first['final_customers'],
        'archived_reference_cost': reference_cost,
        'archived_reference_is_certified_bks': bool(first.get('archived_reference_is_certified_bks', False)),
        'official_bks_cost': bks_cost,
        'official_bks_last_updated': first.get('official_bks_last_updated'),
        'official_bks_source_url': first.get('official_bks_source_url'),
        'mean_final_cost': mean_cost,
        'sample_variance_cost_squared': sample_variance,
        'sample_stddev_cost': sample_sd,
        'min_final_cost': min(costs),
        'max_final_cost': max(costs),
        'mean_gap_vs_archived_reference_pct': statistics.mean(archive_gaps),
        'min_gap_vs_archived_reference_pct': min(archive_gaps),
        'max_gap_vs_archived_reference_pct': max(archive_gaps),
        'archived_reference_gap_range_pct': signed_range(archive_gaps),
        'mean_gap_vs_official_bks_pct': statistics.mean(bks_gaps) if bks_gaps else None,
        'min_gap_vs_official_bks_pct': min(bks_gaps) if bks_gaps else None,
        'max_gap_vs_official_bks_pct': max(bks_gaps) if bks_gaps else None,
        'official_bks_gap_range_pct': signed_range(bks_gaps),
        'mean_delete_total_ms': mean_delete_ms,
        'mean_add_total_ms': mean_add_ms,
        'mean_refine_total_ms': mean_refine_ms,
        'mean_total_gpu_ms': mean_gpu_total_ms,
        'mean_external_wall_seconds': mean_wall_seconds,
        'costs': costs,
        'archive_gaps': archive_gaps,
        'bks_gaps': bks_gaps,
    }


def write_case_outputs(case_dir: Path, summary: dict, records: list[dict]):
    per_run_path = case_dir / 'per_run_results.csv'
    with per_run_path.open('w', newline='') as f:
        fields = [
            'run_id', 'final_cost', 'route_count',
            'gap_vs_official_bks_pct', 'gap_vs_archived_reference_pct',
            'delete_total_ms', 'add_total_ms', 'refine_total_ms', 'total_gpu_ms',
            'wall_time_seconds',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metrics = record.get('metrics', {})
            bks_gap = record.get('gap_vs_official_bks_pct')
            writer.writerow({
                'run_id': record.get('run_id'),
                'final_cost': record.get('final_cost'),
                'route_count': record.get('route_count'),
                'gap_vs_official_bks_pct': '' if bks_gap is None else f'{bks_gap:.6f}',
                'gap_vs_archived_reference_pct': f"{record.get('gap_vs_archived_reference_pct', 0):.6f}",
                'delete_total_ms': metrics.get('delete_total_ms', ''),
                'add_total_ms': metrics.get('add_total_ms', ''),
                'refine_total_ms': metrics.get('refine_total_ms', ''),
                'total_gpu_ms': record.get('total_gpu_ms', ''),
                'wall_time_seconds': record.get('wall_time_seconds', ''),
            })

    summary_path = case_dir / 'five_run_summary.csv'
    csv_row = {k: v for k, v in summary.items() if k not in ('costs', 'archive_gaps', 'bks_gaps')}
    with summary_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        writer.writeheader()
        writer.writerow(csv_row)

    text_path = case_dir / 'five_run_report.txt'
    with text_path.open('w') as f:
        f.write(f"{summary['runs']}-run Runi 80/20 summary: {summary['instance']}\n")
        f.write(f"Validated repetitions: {summary['runs']}\n")
        f.write(f"Final costs: {', '.join(str(v) for v in summary['costs'])}\n")
        f.write(f"Mean final cost (Runi15): {summary['mean_final_cost']:.2f}\n")
        f.write(f"Sample variance (cost^2): {summary['sample_variance_cost_squared']:.2f}\n")
        f.write(f"Sample standard deviation: {summary['sample_stddev_cost']:.2f}\n")
        f.write(f"Min / max final cost: {summary['min_final_cost']} / {summary['max_final_cost']}\n")
        f.write(f"Archived reference cost (not certified BKS): {summary['archived_reference_cost']}\n")
        f.write(f"Mean gap vs archived reference: {summary['mean_gap_vs_archived_reference_pct']:+.3f}%\n")
        f.write(f"Archived-reference gap range: {summary['archived_reference_gap_range_pct']}\n")
        if summary['official_bks_cost'] is not None:
            f.write(f"Official CVRPLIB BKS: {summary['official_bks_cost']} (updated {summary['official_bks_last_updated']})\n")
            f.write(f"Mean gap vs official BKS: {summary['mean_gap_vs_official_bks_pct']:+.3f}%\n")
            f.write(f"Official-BKS gap range: {summary['official_bks_gap_range_pct']}\n")
            f.write(f"BKS source: {summary['official_bks_source_url']}\n")
        if summary['mean_add_total_ms'] is not None:
            f.write(f"Mean ADD time: {summary['mean_add_total_ms']:.3f} ms\n")
        if summary['mean_refine_total_ms'] is not None:
            f.write(f"Mean refinement time: {summary['mean_refine_total_ms']:.3f} ms\n")
        if summary['mean_total_gpu_ms'] is not None:
            f.write(f"Mean total GPU-stage time: {summary['mean_total_gpu_ms']:.3f} ms\n")
        if summary['mean_external_wall_seconds'] is not None:
            f.write(f"Mean external wall time: {summary['mean_external_wall_seconds']:.3f} s\n")
        f.write('\nInterpretation: variance and standard deviation describe run-to-run cost variation. '
                'The archived reference gap is distinct from the official CVRPLIB BKS gap.\n')
    return summary_path, per_run_path, text_path


def combine_case_summaries(root: Path):
    summaries = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        summary_path = case_dir / 'five_run_summary.csv'
        if not summary_path.is_file():
            continue
        with summary_path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        if rows:
            summaries.append(rows[0])
    if not summaries:
        raise SystemExit(f'ERROR: no five_run_summary.csv files found under {root}')
    out = root / 'all_cases_summary.csv'
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f'Combined case summary: {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='one case output directory, or output root with --combine-cases')
    parser.add_argument('--expected-runs', type=int, default=5)
    parser.add_argument('--combine-cases', action='store_true')
    args = parser.parse_args()
    if args.expected_runs < 1:
        parser.error('--expected-runs must be >= 1')
    if args.combine_cases:
        combine_case_summaries(args.path)
        return

    summary = summarize_case(args.path, args.expected_runs)
    records = load_case(args.path, args.expected_runs)
    summary_path, per_run_path, text_path = write_case_outputs(args.path, summary, records)
    print(f"\n========== {summary['instance']} ({summary['runs']} validated runs) ==========")
    print(f"Runi15 mean cost:       {summary['mean_final_cost']:.2f}")
    print(f"Sample variance:        {summary['sample_variance_cost_squared']:.2f} cost^2")
    print(f"Sample standard dev.:   {summary['sample_stddev_cost']:.2f}")
    print(f"Cost range:             {summary['min_final_cost']} .. {summary['max_final_cost']}")
    print(f"Mean archived-ref gap:  {summary['mean_gap_vs_archived_reference_pct']:+.3f}%")
    print(f"Archived-ref gap range: {summary['archived_reference_gap_range_pct']}")
    if summary['official_bks_cost'] is not None:
        print(f"Official BKS:            {summary['official_bks_cost']} ({summary['official_bks_last_updated']})")
        print(f"Mean BKS gap:            {summary['mean_gap_vs_official_bks_pct']:+.3f}%")
        print(f"BKS gap range:           {summary['official_bks_gap_range_pct']}")
    if summary['mean_add_total_ms'] is not None:
        print(f"Mean ADD time:           {summary['mean_add_total_ms']:.3f} ms")
    if summary['mean_refine_total_ms'] is not None:
        print(f"Mean refinement time:    {summary['mean_refine_total_ms']:.3f} ms")
    if summary['mean_total_gpu_ms'] is not None:
        print(f"Mean total GPU stages:   {summary['mean_total_gpu_ms']:.3f} ms")
    if summary['mean_external_wall_seconds'] is not None:
        print(f"Mean external wall time: {summary['mean_external_wall_seconds']:.3f} s")
    print(f"Summary CSV:             {summary_path}")
    print(f"Per-run CSV:             {per_run_path}")
    print(f"Text report:             {text_path}")


if __name__ == '__main__':
    main()
