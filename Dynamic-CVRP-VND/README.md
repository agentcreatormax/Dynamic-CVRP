# Dynamic CVRP Solver & Visualizer

A GPU-accelerated and CPU-based standalone implementation for dynamic customer node insertion and deletion in the Capacitated Vehicle Routing Problem (CVRP). The solver uses a granular Variable Neighbourhood Descent (VND) local search (adapted from Vidal 2020) to rapidly re-optimize routes. 

For large-scale routing, the GPU version utilizes a custom CUDA kernel with per-thread partial selection to compute neighborhoods on-the-fly, bypassing dense $O(N^2)$ distance matrices to scale past 1M+ nodes.

---

## Prerequisites (Linux)

Before building or running the project on Linux, ensure your system has the following installed:

### 1. Compiler and Toolchains
*   **CUDA Toolkit** (includes `nvcc`) — required to compile GPU code (`.cu` files). Needs C++17 compatibility.
*   **GCC / G++ Compiler** (supporting C++17) — required to compile the C++ CPU code (`.cpp` files).
*   **GNU Make** — required to run the build commands defined in the `Makefile`.

### 2. Scripting & Execution Environments
*   **Bash Shell** — required to run `run_tests.sh`.

### 3. Python and Visualization Libraries
*   **Python 3.6+**
*   **Dash, Plotly, and Watchdog** libraries for the visualization server. Install them using pip:
    ```bash
    pip install dash plotly watchdog
    ```

---

## 1. How to Compile (Using the `Makefile`)

The [Makefile](Makefile) in the root directory manages compiling the different implementations of the project.

### Available Targets:
*   **Compile the Default GPU Version** (Compiles [d-cvrp.cu](d-cvrp.cu) to `d-cvrp.out`):
    ```bash
    make
    # or
    make all
    ```
*   **Compile the C++ CPU Version** (Compiles `All Versions/dynamic_cvrp.cpp` to `dynamic_cvrp.out`):
    ```bash
    make cpp
    ```
*   **Compile CUDA Version 1** (Compiles `All Versions/dynamic_cvrp_cuda.cu` to `dynamic_cvrp_cuda.out`):
    ```bash
    make v1
    ```
*   **Compile CUDA Version 2 (Optimized)** (Compiles `All Versions/dynamic_cvrp_cuda-v2.cu` to `dynamic_cvrp_cuda-v2.out`):
    ```bash
    make v2
    ```
*   **Clean Output Executables**:
    ```bash
    make clean
    ```
*   **Clean Temp Folders**:
    ```bash
    make cleanFolders
    ```

---

## 2. Running Automated Batch Tests (`run_tests.sh`)

An automated Bash runner script is provided to compile the selected engine and evaluate it on a target instance dataset (`medium` or `large`).

### CLI Signature:
```bash
./run_tests.sh <version> <size> [add_x] [add_y] [add_demand] [del_node_id]
```

### Parameter Details:
1.  **`<version>`** *(Mandatory)*: Choose compile version to run. Options:
    *   `cpp`: CPU C++ version
    *   `v1`: CUDA v1 GPU version
    *   `v2`: CUDA v2 Optimized GPU version
    *   `default`: Base CUDA version (`d-cvrp.cu`)
2.  **`<size>`** *(Mandatory)*: The dataset size to run on. Options:
    *   `medium`: Runs instances in `medium-instances/`
    *   `large`: Runs instances in `LARGE-instances/`
3.  **`[add_x] [add_y] [add_demand]`** *(Optional)*: Insertion coordinates and demand for addition mode. Defaults: `130 90 1`.
4.  **`[del_node_id]`** *(Optional)*: Node ID (1-based index) to remove during deletion mode. Default: `130`.

### Examples:
*   Run the CPU version on medium instances with default values:
    ```bash
    ./run_tests.sh cpp medium
    ```
*   Run the optimized GPU Version 2 on large instances:
    ```bash
    ./run_tests.sh v2 large
    ```
*   Run the default version on large instances with custom node insertion parameters (X=150, Y=100, demand=3):
    ```bash
    ./run_tests.sh default large 150 100 3
    ```

Outputs will be saved in either `medium-instances/medium-instances-out/` or `LARGE-instances/LARGE-instances-out/` respectively, labeled as `<instance_name>_add.sol`, `<instance_name>_add.vrp`, `<instance_name>_del.sol`, and `<instance_name>_del.vrp`.

---

## 3. Interactive Web Visualizer (`Visual graph generator`)

The [Visual graph generator](Visual%20graph%20generator/) folder contains a Python Dash application that plots routing schedules, customer locations, demands, and route flows dynamically.

### Key Feature: Real-Time File Watching
It leverages the `watchdog` library to monitor the folder containing the `.vrp` and `.sol` files. When the solver writes an updated output, the web visualizer automatically reloads and redraws the routes on the screen without requiring a page refresh.

### Running the Visualizer:
1.  Navigate to the visualizer directory:
    ```bash
    cd "Visual graph generator"
    ```
2.  Run the launcher script pointing to a VRP instance and a solution file:
    ```bash
    python main.py <path_to_cvrp_file.vrp> <path_to_solution_file.sol>
    ```
    *Example:*
    ```bash
    python main.py ../medium-instances/medium-instances-vrp/1.vrp ../medium-instances/medium-instances-sol/1.sol
    ```
3.  The browser will automatically open, or you can navigate to `http://127.0.0.1:8050` to view the interactive canvas.
