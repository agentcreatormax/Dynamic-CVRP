# Medium Instance 1 — Dynamic vs Static Benchmark

All three scenarios start from `start/1.vrp` + `start/1.sol`.

- A: 50 ADD as one bulk batch -> final dimension 1051
- B: 50 DELETE as one bulk batch -> final dimension 951
- C: 25 ADD then 25 DELETE -> two bulk batches -> final dimension 1001

## P0 dynamic runs
Run from the P0 project directory (the folder containing `dcvrp.out`). Copy this benchmark folder there or adjust the paths. Keep `--passes 0` for scheduler/repair timing.

```bash
mkdir -p results/bench_A results/bench_B results/bench_C

./dcvrp.out --vrp medium1_benchmark/start/1.vrp --sol medium1_benchmark/start/1.sol --events medium1_benchmark/events/A_50_add.txt --out-dir results/bench_A --engine gpu --passes 0 --queue-cap 5 --verbose

./dcvrp.out --vrp medium1_benchmark/start/1.vrp --sol medium1_benchmark/start/1.sol --events medium1_benchmark/events/B_50_delete.txt --out-dir results/bench_B --engine gpu --passes 0 --queue-cap 5 --verbose

./dcvrp.out --vrp medium1_benchmark/start/1.vrp --sol medium1_benchmark/start/1.sol --events medium1_benchmark/events/C_25_add_25_delete.txt --out-dir results/bench_C --engine gpu --passes 0 --queue-cap 5 --verbose
```

Read `results/bench_*/summary.csv`. Use `repair_us` for repair-only speed and `latency_us` for the event/batch end-to-end latency recorded by P0. `cost_after` is the dynamic solution cost.

## Static parMDS reruns
Copy the three files in `static_inputs/` into `parMDS-main/inputs/`, then from `parMDS-main`:

```bash
nvc++ -O3 -std=c++14 -acc=multicore parMDS.cpp -o parMDS.out
mkdir -p benchmark_results

./parMDS.out inputs/A_after_50_add.vrp -nthreads 20 -round 1 > benchmark_results/A.sol 2> benchmark_results/A.stats
./parMDS.out inputs/B_after_50_delete.vrp -nthreads 20 -round 1 > benchmark_results/B.sol 2> benchmark_results/B.stats
./parMDS.out inputs/C_after_25_add_25_delete.vrp -nthreads 20 -round 1 > benchmark_results/C.sol 2> benchmark_results/C.stats

cat benchmark_results/A.stats
cat benchmark_results/B.stats
cat benchmark_results/C.stats
```

Each `.stats` line contains `Cost ... Time(seconds) ...`; the last cost is parMDS's final cost and the third time value is its total algorithm time.

## Comparison
For each scenario compare P0 final `cost_after` and `repair_us`/`latency_us` against parMDS final cost and total time. Do not expect the costs to be identical: P0 is a dynamic repair from the existing solution, while parMDS solves the updated instance afresh. Correctness means the P0 validator passes and all active customers appear exactly once within capacity.
