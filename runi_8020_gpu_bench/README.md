# Runi 80/20 batch ADD benchmark

This is a **snapshot reoptimization** comparison. It has no time windows or vehicle travel clock. It is not the same experiment as an online dynamic VRPTW benchmark, and the archived full-route solutions are **feasible references, not independently verified best-known solutions**.

Each test begins with approximately 80% of customers active and uses the Runi CUDA engine to ADD the remaining 20% in one transaction, then refine once with up to `PASSES` passes.

| Case | Starting customers | ADD customers | Final customers | Reference cost (rounded EUC_2D) |
|---|---:|---:|---:|---:|
| `X-n1001-k43` | 800 | 200 | 1,000 | 72,455 |
| `XL-n10001-k1570` | 8,000 | 2,000 | 10,000 | 2,333,757 |
| `Flanders2` | 24,000 | 6,000 | 30,000 | 4,373,244 |

The `case/` folder contains `initial.vrp`, `initial.sol`, and `events_add_to_full.txt`. The `reference/` folder holds `full.sol` and `full.vrp`. The latter was reconstructed for X-n1001 from the precise starting instance and its 200 ordered ADD events and verified against the archived full solution's cost and capacity. The other two cases' full inputs are unchanged. `metadata.json` describes the case, and `results_gpu/<case>/` is created by the runner.

## Run

From this directory:

```bash
# Build CUDA once from ../Dynamic-CVRP-Runi first, or pass BUILD=1
PROJECT_ROOT="$(cd ../Dynamic-CVRP-Runi && pwd)" \
CASE=x1001 PASSES=1 HMAX=5 REFINE_K=5 QUEUE_CAP=5 \
./run_gpu_8020.sh

# All three cases (can be expensive on the largest case)
CASE=all BUILD=0 ./run_gpu_8020.sh
```

Also accepted: `CASE=medium1`, `CASE=medium2`, `CASE=medium3`, or `CASE=both` (medium2+medium3). `GPU_BIN=/absolute/path/to/dcvrp.out` overrides the executable. `OUT_ROOT=...` overrides outputs. `CUDA_ARCH=120 BUILD=1` builds for RTX 5090; use `CUDA_ARCH=80` for A100.

Each successful run writes `run.log`, `summary.csv`, `final.sol`, `final.vrp`, and `reference_comparison.txt`. It verifies the **full final customer set, capacity, no unplaced customers, and both GPU audits** before reporting the signed gap:

```text
Gap vs reference (%) = 100 * (GPU_final_cost - archived_reference_cost) / archived_reference_cost
```

A positive gap means higher cost than the archived full-route reference. Do **not** call this an optimality gap unless the reference is proven optimal. The X-n1001 starting solution in this archive is an independently prepared 800-customer solution; it is not necessarily formed by filtering the full reference routes. The XL/Flanders initial routes were produced by route filtering.
