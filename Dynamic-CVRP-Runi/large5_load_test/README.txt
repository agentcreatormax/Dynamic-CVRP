Large5 Dynamic-CVRP load test
================================

Goal
----
Measure how Runi algorithm time scales as the number of dynamic events grows.

Starting instance:
  Large5 / XML500000_2366_05
  500,000 customers
  capacity 1501

Workload sizes:
  50
  100
  200
  400
  800
  1,600
  3,200
  6,400
  12,800
  25,600
  51,200
  102,400
  204,800
  409,600

Every workload is exactly:
  50% unique DELETEs
  50% ADDs

The workloads are nested:
  the 50-event case is a prefix/subset of the 100-event case,
  the 100-event case is a prefix/subset of the 200-event case, etc.

Compatibility with your existing Large5 benchmark
--------------------------------------------------
The first 200 ADD entries are the exact existing A_200_ADD pool.
The first 200 DELETE entries are the exact existing B_200_DELETE pool.

Therefore:
  200 total events = first 100 DELETE + first 100 ADD
  400 total events = full existing 200 DELETE + full existing 200 ADD

Beyond those pools, additions are generated deterministically in the same
observed Large5 domain:
  x,y in [0,1000]
  demand in [1,100]

Extended deletes are deterministic, unique IDs in 1..500000.

Run
---
Place this folder directly inside Dynamic-CVRP-Runi:

  Dynamic-CVRP-Runi/
    large5_load_test/
    src/
    Makefile
    ...

Make sure the original Large5 benchmark is:
  $HOME/large5_benchmark

Then:

  cd ~/Dynamic-CVRP-Runi/large5_load_test
  CUDA_ARCH=120 PASSES=1 ./run_large5_load_test.sh

Or explicitly:

  BENCH_DIR="$HOME/large5_benchmark" \
  CUDA_ARCH=120 \
  PASSES=1 \
  HMAX=5 \
  REFINE_K=5 \
  ./run_large5_load_test.sh

Outputs
-------
large5_load_results/
  benchmark_summary.csv
      Same top-level format used in your earlier benchmark scripts.

  load_test_results.csv
      events, delete_events, add_events, algorithm_time_sec,
      wall_time_sec, final_cost, final_routes, valid, status

  results.txt
      Easy-to-read one-line result per load size.

  events_vs_algorithm_time.png
      Recommended plot. X axis is log2 because workload sizes double.

  events_vs_algorithm_time_linear.png
      Same measurements with a normal linear X axis.

  EVENTS_50/
  EVENTS_100/
  ...
  EVENTS_409600/
      Each successful folder keeps the raw detailed summary.csv requested
      earlier (ADD/DELETE/refine, scheduler/worker profiler, move timings),
      run.log, final.vrp and final.sol.

Important
---------
Very large event counts may exceed GPU memory or another fixed engine capacity.
That is useful information for this load test. The runner records a FAILED row
and continues to the next workload instead of deleting earlier successful data.

Every run starts from exactly the same start/5.vrp and start/5.sol.
It does NOT continue from the preceding load size.
