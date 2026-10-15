# Compilers and flags
CXX = g++
CXXFLAGS = -std=c++17 -O2
NVCC = nvcc
NVCCFLAGS = -std=c++17 -O2

.PHONY: all cpp v1 v2 clean cleanFolders

# Default target compiles d-cvrp.cu
all: d-cvrp.out

d-cvrp.out: d-cvrp.cu
	$(NVCC) $(NVCCFLAGS) d-cvrp.cu -o d-cvrp.out

# Compile the standalone C++ CPU version
cpp: All\ Versions/dynamic_cvrp.cpp
	$(CXX) $(CXXFLAGS) "All Versions/dynamic_cvrp.cpp" -o dynamic_cvrp.out

# Compile the CUDA version 1
v1: All\ Versions/dynamic_cvrp_cuda.cu
	$(NVCC) $(NVCCFLAGS) "All Versions/dynamic_cvrp_cuda.cu" -o dynamic_cvrp_cuda.out

# Compile the CUDA version 2
v2: All\ Versions/dynamic_cvrp_cuda-v2.cu
	$(NVCC) $(NVCCFLAGS) "All Versions/dynamic_cvrp_cuda-v2.cu" -o dynamic_cvrp_cuda-v2.out

clean:
	rm -f *.out

cleanFolders:
	rm -rf outinputs*
