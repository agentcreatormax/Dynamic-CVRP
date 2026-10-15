# Medium Instance 3 Benchmark

Original instance: `Flanders2` (30000 customers + depot).

Same workload structure as medium1:
- A: 50 ADD in one bulk batch
- B: 50 DELETE in one bulk batch
- C: 25 ADD then 25 DELETE (two batches)

The existing P0/Senior benchmark runners expect `start/1.vrp` and `start/1.sol`, so compatibility copies with those names are included. The canonical originals are also included as `start/3.vrp` and `start/3.sol`.

Run from the P0 project folder:
```bash
BENCH_DIR=$HOME/Desktop/UGRC/medium3_benchmark ./run_benchmark_p0.sh
```

Run from the senior project folder:
```bash
BENCH_DIR=$HOME/Desktop/UGRC/medium3_benchmark ./run_benchmark_senior.sh
```

Important: the current runner scripts reuse `benchmark_p0/` and `benchmark_senior/`; rename/save previous result folders before running another medium instance.
