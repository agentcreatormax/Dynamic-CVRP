# Top-level convenience build/benchmark targets for Dynamic-CVRP/
# Does not alter the individual subproject Makefiles.
SHELL := /bin/bash
CUDA_ARCH ?= 120
PROFILE ?= 1
PASSES ?= 1
HMAX ?= 5
REFINE_K ?= 5
QUEUE_CAP ?= 5
BENCH_DIR ?= $(CURDIR)/medium1_benchmark
CASES ?= all
SOLVER ?= seq
NTHREADS ?= 1
CASE ?= x1001
SIZES ?= 50 100 200 400 800 1600 3200 6400 12800
PASSES_VALUES ?= 1 5 10 15 20

.PHONY: all help gpu cpu vnd seqMDS seqmds parMDS parmds \
 benchmark-gpu benchmark-vnd benchmark-seq benchmark-par benchmark-8020 \
 passes loadtest visualizer clean

all: gpu
help:
	@echo 'make gpu CUDA_ARCH=120         # build GPU Runi'
	@echo 'make cpu                       # build Runi CPU-only'
	@echo 'make vnd CUDA_ARCH=120         # build senior VND CUDA'
	@echo 'make seqMDS                    # build seqMDS with g++'
	@echo 'make parMDS                    # build parMDS with nvc++'
	@echo 'make benchmark-gpu BENCH_DIR=... [CASES=A,B,C]'
	@echo 'make benchmark-vnd BENCH_DIR=... [CASES=A,B,C]'
	@echo 'make benchmark-seq BENCH_DIR=... [CASES=A,B,C]'
	@echo 'make benchmark-par BENCH_DIR=... [NTHREADS=1]'
	@echo 'make benchmark-8020 CASE=x1001|medium2|medium3|all'
	@echo 'make passes BENCH_DIR=... CASE=C PASSES_VALUES="1 5 10 15 20"'
	@echo 'make loadtest BENCH_DIR=... SIZES="800 1600 3200"'
	@echo 'make visualizer [CUDA_ARCH=120]'

gpu:
	$(MAKE) -C Dynamic-CVRP-Runi CUDA_ARCH=$(CUDA_ARCH) PROFILE=$(PROFILE)
cpu:
	$(MAKE) -C Dynamic-CVRP-Runi cpu
vnd:
	$(MAKE) -C Dynamic-CVRP-VND NVCCFLAGS='-std=c++17 -O2 -arch=sm_$(CUDA_ARCH)' all
seqMDS seqmds:
	g++ -O3 -std=c++14 CVRP-seqMDS/seqMDS.cpp -o CVRP-seqMDS/seqMDS.out
parMDS parmds:
	nvc++ -O3 -std=c++14 -acc=multicore CVRP-seqMDS/parMDS.cpp -o CVRP-seqMDS/parMDS.out

benchmark-gpu: gpu
	BENCH_DIR='$(BENCH_DIR)' CASES='$(CASES)' BUILD=0 PASSES='$(PASSES)' HMAX='$(HMAX)' REFINE_K='$(REFINE_K)' QUEUE_CAP='$(QUEUE_CAP)' ./Dynamic-CVRP-Runi/run_benchmark_runi.sh
benchmark-vnd: vnd
	BENCH_DIR='$(BENCH_DIR)' CASES='$(CASES)' BUILD=0 ./Dynamic-CVRP-VND/run_benchmark_vnd.sh
benchmark-seq: seqMDS
	BENCH_DIR='$(BENCH_DIR)' CASES='$(CASES)' BUILD=0 SOLVER=seq ./CVRP-seqMDS/run_benchmark_seqmds.sh
benchmark-par: parMDS
	BENCH_DIR='$(BENCH_DIR)' CASES='$(CASES)' BUILD=0 SOLVER=par NTHREADS='$(NTHREADS)' ./CVRP-seqMDS/run_benchmark_seqmds.sh
benchmark-8020: gpu
	CASE='$(CASE)' BUILD=0 PROJECT_ROOT='$(CURDIR)/Dynamic-CVRP-Runi' PASSES='$(PASSES)' HMAX='$(HMAX)' REFINE_K='$(REFINE_K)' QUEUE_CAP='$(QUEUE_CAP)' ./runi_8020_gpu_bench/run_gpu_8020.sh
passes: gpu
	BENCH_DIR='$(BENCH_DIR)' CASE='$(CASE)' PASSES_VALUES='$(PASSES_VALUES)' BUILD=0 HMAX='$(HMAX)' REFINE_K='$(REFINE_K)' QUEUE_CAP='$(QUEUE_CAP)' ./Dynamic-CVRP-Runi/passes_validation/run_pass_sweep.sh
loadtest: gpu
	BENCH_DIR='$(BENCH_DIR)' SIZES='$(SIZES)' BUILD=0 PASSES='$(PASSES)' HMAX='$(HMAX)' REFINE_K='$(REFINE_K)' QUEUE_CAP='$(QUEUE_CAP)' ./Dynamic-CVRP-Runi/large5_load_test/run_large5_load_test.sh
visualizer: gpu
	CUDA_ARCH='$(CUDA_ARCH)' REBUILD=0 ./Dynamic-CVRP-Runi/dcvrp_visualizer/start_visualizer.sh
clean:
	$(MAKE) -C Dynamic-CVRP-Runi clean
	$(MAKE) -C Dynamic-CVRP-VND clean
	$(MAKE) -C CVRP-seqMDS clean
