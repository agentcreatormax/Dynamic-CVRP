#!/bin/bash

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <version> <size> [add_x] [add_y] [add_demand] [del_node_id]"
    echo "  version: cpp | v1 | v2 | default"
    echo "  size:    medium | large"
    exit 1
fi

VERSION=$(echo "$1" | tr '[:upper:]' '[:lower:]')
SIZE=$(echo "$2" | tr '[:upper:]' '[:lower:]')

# Default parameters
ADD_X=${3:-130}
ADD_Y=${4:-90}
ADD_DEMAND=${5:-1}
DEL_NODE_ID=${6:-130}

# Determine target and binary name
case "$VERSION" in
    cpp)
        TARGET="cpp"
        BIN="./dynamic_cvrp.out"
        ;;
    v1)
        TARGET="v1"
        BIN="./dynamic_cvrp_cuda.out"
        ;;
    v2)
        TARGET="v2"
        BIN="./dynamic_cvrp_cuda-v2.out"
        ;;
    default)
        TARGET="all"
        BIN="./d-cvrp.out"
        ;;
    *)
        echo "Error: Unknown version '$VERSION'. Must be one of: cpp, v1, v2, default."
        exit 1
        ;;
esac

# Compile using the Makefile
echo "Compiling version '$VERSION' using Makefile target '$TARGET'..."
if hash make 2>/dev/null; then
    make "$TARGET"
elif hash mingw32-make 2>/dev/null; then
    mingw32-make "$TARGET"
else
    echo "WARNING: Neither 'make' nor 'mingw32-make' was found in PATH."
fi

if [ ! -f "$BIN" ]; then
    echo "ERROR: Executable '$BIN' not found and could not be built."
    exit 1
fi

# Determine folders based on size
case "$SIZE" in
    medium)
        VRP_DIR="medium-instances/medium-instances-vrp"
        SOL_DIR="medium-instances/medium-instances-sol"
        OUT_DIR="medium-instances/medium-instances-out"
        ;;
    large)
        VRP_DIR="LARGE-instances/LARGE-instances-vrp"
        SOL_DIR="LARGE-instances/LARGE-instances-sol"
        OUT_DIR="LARGE-instances/LARGE-instances-out"
        ;;
    *)
        echo "Error: Unknown size '$SIZE'. Must be one of: medium, large."
        exit 1
        ;;
esac

if [ ! -d "$VRP_DIR" ]; then
    echo "ERROR: VRP directory '$VRP_DIR' does not exist."
    exit 1
fi
if [ ! -d "$SOL_DIR" ]; then
    echo "ERROR: Solution directory '$SOL_DIR' does not exist."
    exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"

# Find all .vrp files in the directory
VRP_FILES=("$VRP_DIR"/*.vrp)

if [ ! -e "${VRP_FILES[0]}" ]; then
    echo "ERROR: No .vrp files found in '$VRP_DIR'."
    exit 1
fi

echo "================================================================="
echo "Running dynamic CVRP ($VERSION) tests on $SIZE instances..."
echo "Addition settings: X=$ADD_X, Y=$ADD_Y, demand=$ADD_DEMAND"
echo "Deletion settings: Node ID=$DEL_NODE_ID"
echo "Output directory:  $OUT_DIR"
echo "================================================================="

for VRP_FILE in "${VRP_FILES[@]}"; do
    BASE_NAME=$(basename "$VRP_FILE" .vrp)
    SOL_FILE="$SOL_DIR/$BASE_NAME.sol"
    
    if [ ! -f "$SOL_FILE" ]; then
        echo "Warning: Solution file '$SOL_FILE' not found. Skipping."
        continue
    fi
    
    echo "-----------------------------------------------------------------"
    echo "Processing Instance: $BASE_NAME"
    
    # 1. Addition Mode
    ADD_OUT_SOL="$OUT_DIR/${BASE_NAME}_add.sol"
    ADD_OUT_VRP="$OUT_DIR/${BASE_NAME}_add.vrp"
    echo "Running Addition Mode..."
    "$BIN" -t a "$VRP_FILE" "$SOL_FILE" "$ADD_OUT_SOL" "$ADD_OUT_VRP" "$ADD_X" "$ADD_Y" "$ADD_DEMAND"
    
    # 2. Deletion Mode
    DEL_OUT_SOL="$OUT_DIR/${BASE_NAME}_del.sol"
    DEL_OUT_VRP="$OUT_DIR/${BASE_NAME}_del.vrp"
    echo "Running Deletion Mode..."
    "$BIN" -t d "$VRP_FILE" "$SOL_FILE" "$DEL_OUT_SOL" "$DEL_OUT_VRP" "$DEL_NODE_ID"
done

echo "================================================================="
echo "All tests completed successfully!"
echo "Outputs generated in: $OUT_DIR/"
echo "================================================================="
