TOY Dynamic-CVRP benchmark
==========================

Purpose
-------
A tiny correctness/profiling benchmark compatible with the Runi runner format.

Start instance
--------------
DIMENSION = 10 vertices total:
  1 depot
  9 customers

Capacity = 15

Files:
  start/toy_start.vrp
  start/toy_start.sol

Workloads
---------
toy_2_ADD
  events/toy_2_add.txt
  2 ADD events
  expected final customer set:
    static_inputs/toy_after_2_add.vrp

toy_2_DELETE
  events/toy_2_delete.txt
  2 DELETE events
  expected final customer set:
    static_inputs/toy_after_2_delete.vrp

DELETE convention
-----------------
DELETE IDs are stable internal customer IDs.
TSPLIB node number = internal customer ID + 1.

Therefore:
  D 2 deletes TSPLIB node 3  -> (25,25), demand 3
  D 7 deletes TSPLIB node 8  -> (80,80), demand 5

ADD events
----------
  A 50 80 4
  A 55 15 2

Running
-------
Place the toy_benchmark folder directly inside the Dynamic-CVRP-Runi project,
then run:

  cd toy_benchmark
  CUDA_ARCH=120 ./toy.sh

For a V100:
  CUDA_ARCH=70 ./toy.sh

Outputs
-------
toy_results/
  toy_benchmark_summary.csv
      Same top-level benchmark format used previously:
      implementation,case,updates,algorithm_time_sec,wall_time_sec,
      final_cost,final_routes,valid

  toy_results.txt
      Human-readable copy of the two result rows.

  toy_2_ADD/
      summary.csv       raw detailed profiler emitted by dcvrp.out
      toy_summary.csv   copy of the detailed profiler
      run.log
      final.vrp / final.sol

  toy_2_DELETE/
      same structure
