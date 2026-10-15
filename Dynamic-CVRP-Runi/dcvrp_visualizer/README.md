# Dynamic-CVRP Runi Visualizer

A local, interactive browser visualizer for the CUDA Runi project.

It starts from the included 10-vertex toy benchmark. You can:

- click the plot to choose X/Y coordinates;
- enter `x`, `y`, and `demand`, then ADD a customer;
- click an existing customer and DELETE it;
- run refinement without changing the customer set;
- change PASSES, nearest-K, HMAX, and queue capacity;
- see the new routes immediately;
- see final cost, algorithm time, wall time, routes, and passes run/requested;
- see the detailed scheduler/worker/refinement profiler values produced by
  `summary.csv`.

## Folder placement

Unzip this directly into the Runi project root:

```text
Dynamic-CVRP-Runi/
├── dcvrp.out
├── Makefile
├── src/
├── ...
└── dcvrp_visualizer/
    ├── app.py
    ├── start_visualizer.sh
    ├── sample_data/
    └── static/
```

## Run on  RTX 5090

```bash
cd ~/Dynamic-CVRP-Runi/dcvrp_visualizer
CUDA_ARCH=120 ./start_visualizer.sh
```

Open:

```text
http://127.0.0.1:8000
```

The launcher automatically adds `/usr/local/cuda-13.2/bin` to PATH when it
exists.

If `dcvrp.out` does not exist, it builds it automatically. To force a rebuild:

```bash
CUDA_ARCH=120 REBUILD=1 ./start_visualizer.sh
```

No Flask, npm, React, or sudo installation is required. The backend uses only
Python's standard library.

## What happens when ADD is clicked?

For example, entering:

```text
x      = 50
y      = 80
demand = 4
passes = 15
```

writes a one-line event:

```text
A 50 80 4
```

and invokes roughly:

```bash
./dcvrp.out \
  --vrp dcvrp_visualizer/runtime/current.vrp \
  --sol dcvrp_visualizer/runtime/current.sol \
  --events <generated-event-file> \
  --out-dir <history-run-directory> \
  --engine gpu \
  --passes 15 \
  --hmax 5 \
  --refine-k 5 \
  --queue-cap 5 \
  --snap-vrp \
  --verbose
```

After a successful run, `final.vrp` and `final.sol` become the next interactive
state. Therefore you can add/delete repeatedly and the browser keeps evolving
the solution.

## DELETE

The plot labels customers with the current internal customer IDs expected by
the Dynamic-CVRP controller. Click a point, then click **Delete selected +
re-optimize**.

After deletion, the snapshot is renumbered by the controller. The browser
reloads those current IDs automatically.

## Refinement passes

The visualizer shows both:

```text
passes requested
passes actually run
```

because Runi can stop early when a refinement pass commits no moves.

## Runtime files

All interactive runs are preserved under:

```text
dcvrp_visualizer/runtime/history/
```

Each operation contains:

```text
event.txt
run.log
summary.csv
final.vrp
final.sol
```

The active state is:

```text
runtime/current.vrp
runtime/current.sol
runtime/last_run.json
```

Click **Reset toy benchmark** in the UI to restore the included initial state.

## Use another starting instance

You can point the visualizer at another `.vrp` and `.sol` pair:

```bash
START_VRP="$HOME/some_benchmark/start/1.vrp" \
START_SOL="$HOME/some_benchmark/start/1.sol" \
CUDA_ARCH=120 \
./start_visualizer.sh
```

Delete `runtime/current.vrp` and `runtime/current.sol` first, or use the Reset
button after launching, so the new start files are copied into the runtime
state.

## Port

If port 8000 is occupied:

```bash
PORT=8080 CUDA_ARCH=120 ./start_visualizer.sh
```

then open `http://127.0.0.1:8080`.
