# Runi — GPU batch Dynamic CVRP

The current implementation runs the customer-update transaction on CUDA: **DELETE all requested customers → ADD/repair all requested customers → one refinement phase**. The scheduler chooses centroid-nearest candidate routes (local cache `HMAX`, default 5); reusable route-worker warps perform insertion/ejection; sparse matching rounds perform interroute and intra-route local search.

## Folder map

```text
Dynamic-CVRP-Runi/
├── src/                 # CUDA/CPU source, VRP/solution parsing, repair/refinement
├── Makefile             # build dcvrp.out (CUDA) and dcvrp_cpu
├── run_benchmark_runi.sh
├── passes_validation/   # refinement-pass sensitivity
├── large5_load_test/    # mixed-update scaling and plots
├── toy_benchmark/       # toy input and smoke script
└── dcvrp_visualizer/    # interactive localhost browser server
```

## Build and run

```bash
make CUDA_ARCH=120 PROFILE=1   # RTX 5090
make CUDA_ARCH=80 PROFILE=1    # A100
make cpu                       # CPU-only build

./dcvrp.out \
  --vrp ../medium1_benchmark/start/1.vrp \
  --sol ../medium1_benchmark/start/1.sol \
  --events ../medium1_benchmark/events/A_50_add.txt \
  --out-dir example_out --engine gpu \
  --passes 1 --hmax 5 --refine-k 5 --queue-cap 5 --snap-vrp
```

`--passes` is an upper limit (can stop on convergence). For reproducible A/B/C benchmark output and static-set validation:

```bash
BENCH_DIR="$PWD/../medium1_benchmark" CASES=all PASSES=1 \
./run_benchmark_runi.sh

BENCH_DIR="$PWD/../large2_benchmark" CASES=C PASSES=1 \
./run_benchmark_runi.sh
```

The batch runner writes each case's `run.log`, `summary.csv`, `final.sol`, and `final.vrp`, plus `benchmark_summary.csv` for cost, routes, time and customer-set validation.

## Experiments

```bash
BENCH_DIR="$PWD/../medium2_benchmark" CASE=C PASSES_VALUES="1 5 10 15 20" \
./passes_validation/run_pass_sweep.sh

BENCH_DIR="$PWD/../large5_benchmark" SIZES="800 1600 3200 6400 12800" \
./large5_load_test/run_large5_load_test.sh
```

The load-test generator produces equal DELETE/ADD events for each size, and the runner records per-case failure status rather than treating a failed run time as a successful algorithm time. Pass-sweep and load-test default output directories include a timestamp so previously collected results are not deleted. Set `OUT_DIR=...` to choose a specific output folder. For the 80/20 addition benchmark and archived-reference cost gap, see `../runi_8020_gpu_bench/README.md`.

## Visualizer

```bash
cd dcvrp_visualizer
CUDA_ARCH=120 ./start_visualizer.sh
```

Open `http://127.0.0.1:8000`. To reset the interactive state, click **Reset toy benchmark**. `runtime/history/` keeps each operation's saved snapshots, profiler and logs.
