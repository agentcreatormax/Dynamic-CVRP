LARGE1 benchmark
=================
Original instance: XML100000_2222_01
Original customers: 100000
Capacity: 37
Deterministic seed: 42

Start:
  start/1.vrp
  start/1.sol

Workloads:
  A_200_ADD
    events/A_200_add.txt
    200 ADDs in one bulk batch
    static_inputs/A_after_200_add.vrp

  B_200_DELETE
    events/B_200_delete.txt
    200 DELETEs in one bulk batch
    static_inputs/B_after_200_delete.vrp

  C_100_ADD_100_DELETE
    events/C_100_add_100_delete.txt
    batch 1: 100 ADDs
    batch 2: 100 DELETEs
    static_inputs/C_after_100_add_100_delete.vrp

DELETE IDs are stable internal customer IDs 1..N (TSPLIB node = ID+1).
Static final VRPs renumber surviving/new customers contiguously.
A/B/C use the same deterministic generated pools: C uses the first 100 A-adds
and first 100 B-deletes.
