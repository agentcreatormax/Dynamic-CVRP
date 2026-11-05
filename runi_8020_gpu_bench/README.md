# Runi 80/20 batch ADD benchmark — five-run evaluation

This is a **snapshot reoptimization** comparison. It has no time windows or vehicle-travel clock. It is not the same experiment as an online dynamic VRPTW benchmark.

Each test starts with approximately 80% of customers active, then uses the Runi CUDA engine to ADD the remaining 20% in one transaction and refine once with up to `PASSES` passes. The updated runner repeats each case `REPEATS` times (default: **5**), sequentially, using the same initial VRP, initial solution, event file, and algorithm parameters for every repetition. Each run has a separate output directory so one run cannot overwrite another.

| Case | Starting customers | ADD customers | Final customers | Archived feasible reference | CVRPLIB BKS used by runner (checked 2026-11-05) |
|---|---:|---:|---:|---:|---:|
| `X-n1001-k43` | 800 | 200 | 1,000 | 72,455 | 72,332 |
| `XL-n10001-k1570` | 8,000 | 2,000 | 10,000 | 2,333,757 | 2,329,417 |
| `Flanders2` | 24,000 | 6,000 | 30,000 | 4,373,244 | 4,364,450 |

**Important:** the archived full-route solutions are feasible references, **not certified BKSs**. The runner calculates and labels both gaps separately. The BKS values are stored separately in `metadata.json` and were checked against CVRPLIB bound histories on 2026-11-05. For the report's Table 5, use the `official_bks_gap_range_pct`, not the archived-reference gap range.

## Inputs

Each case folder contains:

- `case/initial.vrp`: the 80%-active instance;
- `case/initial.sol`: the initial feasible route set;
- `case/events_add_to_full.txt`: the fixed list of remaining 20% ADD requests;
- `reference/full.vrp` and `reference/full.sol`: archived full-instance reference;
- `metadata.json`: instance sizes and reference cost.

The X-n1001 starting solution is an independently prepared 800-customer solution, not necessarily formed by filtering the full reference routes. The XL/Flanders initial routes were produced by route filtering. The X-n1001 full VRP was reconstructed from the precise starting instance and its 200 ordered ADD events and verified against the archived full solution's cost and capacity. The other two cases' full inputs are unchanged.

## Run the five repetitions

Build Runi once from `../Dynamic-CVRP-Runi`, or set `BUILD=1` to build in the script:

```bash
PROJECT_ROOT="$(cd ../Dynamic-CVRP-Runi && pwd)" \
CASE=all PASSES=15 REPEATS=5 HMAX=5 REFINE_K=5 QUEUE_CAP=5 \
OUT_ROOT="$PWD/results_gpu_15passes_5runs" \
./run_gpu_8020.sh
```

Useful options:

- `CASE=x1001`, `CASE=medium2`, `CASE=medium3`, `CASE=both`, or `CASE=all`;
- `PASSES=15` is the default and matches the Runi15 experiment;
- `REPEATS=5` is the default; increase it for more repetitions;
- `GPU_BIN=/absolute/path/to/dcvrp.out` overrides the executable;
- `OUT_ROOT=...` changes the output directory;
- `CUDA_ARCH=120 BUILD=1` builds for RTX 5090; use the appropriate architecture for your GPU;
- `OVERWRITE=1` replaces only the selected case's `run_01`, `run_02`, ... directories. Otherwise the script refuses to mix old and new runs.

The repetitions run **one after another**, not concurrently. This avoids having this script compete with itself for the GPU. Other users on a shared GPU can still affect timing, so note substantial external GPU activity when interpreting runtime variation.

## Output layout

For each case, the runner creates:

```text
results_gpu_15passes_5runs/
  X-n1001-k43/
    run_01/ ... run_05/
      run.log
      summary.csv
      final.sol
      final.vrp
      run_metrics.json
      reference_comparison.txt
    per_run_results.csv
    five_run_summary.csv
    five_run_report.txt
  ...
  all_cases_summary.csv
```

Each run is validated for the full final customer set, capacity field, no unplaced customers, and both GPU audits before it is included in the statistics. If any repetition fails validation, the script stops instead of silently calculating statistics from fewer than five valid runs.

## Statistics

`aggregate_gpu_runs.py` calculates:

- arithmetic mean final cost (this is the value to put in the `Runi15` column);
- **sample variance**, using the `n-1` denominator, and sample standard deviation;
- minimum and maximum final cost;
- mean gap and gap range versus both the archived feasible reference and current CVRPLIB BKS;
- average ADD, refinement, total GPU-stage, and external wall times when available.

For five costs $C_1,\ldots,C_5$, the reported mean is $\bar C=\frac{1}{5}\sum_i C_i$ and sample variance is $s^2=\frac{1}{4}\sum_i(C_i-\bar C)^2$. The standard deviation is $s=\sqrt{s^2}$. `per_run_results.csv` retains the individual observations; `five_run_summary.csv` and `five_run_report.txt` hold the aggregates.

## Reproducibility notes

The runner does not change the instance, event order, or CLI parameters between repetitions. The repetitions are separate executions of the same inputs, so they can expose run-to-run variation from concurrent GPU scheduling / atomic work dispatch, if present in the engine. A zero variance is also a valid result. The current harness does not force an internal random seed because the benchmark executable's documented CLI in this package does not expose a seed option. BKS values and their source URLs are recorded per case in `metadata.json`; update them if CVRPLIB changes the BKS again before final submission.

The existing `results_gpu_15passes/` directory in this archive contains historical one-run outputs. It is preserved as an example, but it is **not** treated as one of the new five-run repetitions.

## Send results back for the report

After all cases complete, zip the generated `results_gpu_15passes_5runs/` directory and send it back. Keep the individual `run_01` through `run_05` folders; the raw per-run outputs are needed to compute mean, sample variance / standard deviation, and the BKS gap range correctly.
