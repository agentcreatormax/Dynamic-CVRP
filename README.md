# Dynamic-CVRP — CUDA Routing and Benchmark Suite

Implementations and experimental workloads for **dynamic capacitated vehicle routing**. The main CUDA implementation applies a batch of DELETEs, then ADDs, then one refinement phase (up to the requested number of refinement passes). It maintains the vehicle capacity constraint while minimizing the final total route distance. This is **snapshot reoptimization**, not online vehicle dispatch with route travel clocks or time windows.

## Repository contents

```text
Dynamic-CVRP/
├── Makefile                  # one-command builds and experiments
├── README.md                 # this guide
├── Dynamic-CVRP-Runi/      # current GPU implementation (CUDA) + CPU reference
│   ├── src/                  # GPU scheduler/worker/refinement and input parser
│   ├── dcvrp_visualizer/     # local browser tool for interactive ADD/DELETE
│   ├── passes_validation/    # pass-count vs time/cost experiments
│   ├── large5_load_test/     # mixed DELETE/ADD scaling benchmark
│   └── toy_benchmark/        # small smoke test
├── Dynamic-CVRP-VND/        # earlier sequential-event VND algorithm
├── CVRP-seqMDS/             # static seqMDS; parMDS OpenACC/OpenMP baseline
├── runi_8020_gpu_bench/    # 80% initially active, 20% added: three testsets
├── medium1_benchmark/       # 50 events; X-n1001 type
├── medium2_benchmark/       # 50 events; XL-n10001 type
├── medium3_benchmark/       # 50 events; Flanders2 type
├── large1_benchmark/ ...    # 200-event A/B/C tests (large1 through large5)
└── References/              # related papers
```

Each standard benchmark folder contains `start/*.vrp` and `start/*.sol` (starting state), `events/*.txt` (A/B/C workloads), and `static_inputs/*.vrp` (expected final customer sets). The scripts verify the final set even if deleting customers renumbers node IDs.

## Requirements

Linux, GNU Make, Python 3, a recent `g++`, and an NVIDIA CUDA toolkit with `nvcc` for GPU/VND builds. `nvc++` from NVIDIA HPC SDK is needed **only** if building/running parMDS; seqMDS works with `g++`. Some plotting scripts use Matplotlib. The visualizer uses the Python standard library.

Example CUDA environment on RTX5090:

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

## Build

Run from the **Dynamic-CVRP root**:

```bash
make help
make gpu CUDA_ARCH=120      # RTX 5090
make gpu CUDA_ARCH=80       # A100
make cpu                     # Runi CPU fallback
make vnd CUDA_ARCH=120       # VND implementation
make seqMDS                  # g++ static baseline
make parMDS                  # requires nvc++
```

This invokes the existing subproject Makefiles; it does not replace them. For a standalone Runi build:

```bash
cd Dynamic-CVRP-Runi
make CUDA_ARCH=120 PROFILE=1
```

## Run a dynamic CUDA transaction

From `Dynamic-CVRP/`:

```bash
./Dynamic-CVRP-Runi/dcvrp.out \
  --vrp medium1_benchmark/start/1.vrp \
  --sol medium1_benchmark/start/1.sol \
  --events medium1_benchmark/events/C_25_add_25_delete.txt \
  --out-dir my_first_run \
  --engine gpu --passes 1 --hmax 5 --refine-k 5 --queue-cap 5 --snap-vrp
```

Event syntax is `A x y demand` or `D customer_id`. Regardless of input order, Runi applies all DELETEs first, then ADDs, then refinement. `--passes N` is a **maximum**; refinement may converge before completing all N passes.

Successful output resembles:

```text
[GPU-UNPLACED AFTER-REPAIR] count=0
[AUDIT AFTER-REPAIR] bad=0
[gpu-refine] passes=1/1 ...
[AUDIT AFTER-REFINE] bad=0
[transaction] deletes=25 adds=25 ... cost OLD->NEW routes=...
```

`my_first_run/summary.csv` reports phase times and scheduler/worker utilization. `final.vrp` and `final.sol` hold the new routes; `run.log` is produced by the supplied batch runners, not automatically by the executable when invoked directly. A `.sol` lists `Route #1: ...`, `Route #2: ...`, and `Cost <value>`.

## Benchmarks and comparison

```bash
# Standard medium or large A/B/C tests; replace benchmark path as required
make benchmark-gpu BENCH_DIR="$PWD/medium1_benchmark" CASES=all PASSES=1
make benchmark-vnd BENCH_DIR="$PWD/large1_benchmark" CASES=A,B,C
make benchmark-seq BENCH_DIR="$PWD/medium2_benchmark" CASES=A
make benchmark-par BENCH_DIR="$PWD/medium1_benchmark" CASES=C NTHREADS=1

# 80/20 batch ADD tests and reference gap, including X-n1001-k43
make benchmark-8020 CASE=x1001 PASSES=1
make benchmark-8020 CASE=medium2 PASSES=1
make benchmark-8020 CASE=medium3 PASSES=1

# Experiments
make passes BENCH_DIR="$PWD/medium1_benchmark" CASE=C PASSES_VALUES="1 5 10 15 20"
make loadtest BENCH_DIR="$PWD/large5_benchmark" SIZES="800 1600 3200 6400 12800"
```

`benchmark-vnd` processes events one at a time, with VND after **each** event. `benchmark-gpu` executes the complete batch transaction. `benchmark-seq` and `benchmark-par` solve the **final static VRP from scratch**, not a dynamic event stream. Report their time and quality as static reoptimization baselines, not equivalent dynamic-update runtimes. The 80/20 reference solutions are not certified BKS; the displayed gap is a **gap versus archived reference solution**.

## Visualizer

```bash
cd Dynamic-CVRP-Runi/dcvrp_visualizer
CUDA_ARCH=120 ./start_visualizer.sh
```

Open `http://127.0.0.1:8000`. Choose an ADD coordinate/demand or select a customer for DELETE, then inspect the updated routes/cost. To use another initial route set:

```bash
START_VRP="$PWD/../../medium1_benchmark/start/1.vrp" \
START_SOL="$PWD/../../medium1_benchmark/start/1.sol" \
CUDA_ARCH=120 ./start_visualizer.sh
```

The visualizer keeps operations under `runtime/history/`. If working on a remote lab GPU, forward the visualizer's port to your machine with SSH (`ssh -L 8000:127.0.0.1:8000 user@host`) and open the URL locally.
