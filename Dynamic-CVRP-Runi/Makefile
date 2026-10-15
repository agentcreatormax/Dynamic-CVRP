# ============================================================================
# UGRC Dynamic-CVRP — Runi CUDA build
# ============================================================================

SHELL := /bin/bash

# ---------------- User-configurable options ----------------

CUDA_ARCH ?= 70
DEBUG     ?= 0
PROFILE   ?= 1
CXX_STD   ?= c++17

# If CUDA_HOME is supplied by the CUDA module, prefer its nvcc.
ifneq ($(strip $(CUDA_HOME)),)
NVCC ?= $(CUDA_HOME)/bin/nvcc
else
NVCC ?= nvcc
endif

HOST_CXX ?= g++
CXX      ?= g++

# ---------------- Internal flags ----------------
#
# Use project-specific flag variables instead of NVFLAGS so an exported
# environment variable cannot unexpectedly replace -std=c++17.

DCVRP_NVFLAGS := -O2
DCVRP_NVFLAGS += -std=$(CXX_STD)
DCVRP_NVFLAGS += --expt-relaxed-constexpr
DCVRP_NVFLAGS += -DDCVRP_SCHED_DEBUG=$(DEBUG)
DCVRP_NVFLAGS += -DDCVRP_PROFILE=$(PROFILE)

ifneq ($(strip $(CUDA_ARCH)),)
DCVRP_NVFLAGS += -arch=sm_$(CUDA_ARCH)
endif

# Explicitly tell nvcc which host C++ compiler to use.
DCVRP_NVFLAGS += -ccbin $(HOST_CXX)

CPU_FLAGS := -O2 -std=$(CXX_STD) -Isrc

TARGET     := dcvrp.out
CPU_TARGET := dcvrp_cpu

CUDA_SOURCES := \
	src/dynamic_cvrp.cu \
	src/gpu_warp.cuh \
	src/instance.hpp \
	src/routes.hpp \
	src/static_init.hpp \
	src/repair.hpp \
	src/refine.hpp

.PHONY: all clean cpu check-env info

all: $(TARGET)

# ============================================================================
# Environment sanity check
# ============================================================================

check-env:
	@set -e; \
	NVCC_BIN="$$(command -v "$(NVCC)" 2>/dev/null || true)"; \
	if [[ -z "$$NVCC_BIN" ]]; then \
		echo "ERROR: nvcc not found."; \
		echo "Load CUDA first, e.g. on AQUA:"; \
		echo "  module purge"; \
		echo "  module load gcc10.3.0"; \
		echo "  module load cuda12.2"; \
		exit 1; \
	fi; \
	\
	HOST_BIN="$$(command -v "$(HOST_CXX)" 2>/dev/null || true)"; \
	if [[ -z "$$HOST_BIN" ]]; then \
		echo "ERROR: host compiler '$(HOST_CXX)' not found."; \
		exit 1; \
	fi; \
	\
	CUDA_MAJOR="$$("$$NVCC_BIN" --version | \
		sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n1)"; \
	GCC_MAJOR="$$("$$HOST_BIN" -dumpversion | cut -d. -f1)"; \
	\
	echo "===== BUILD ENVIRONMENT ====="; \
	echo "nvcc       : $$NVCC_BIN"; \
	"$$NVCC_BIN" --version | grep -E "release|Cuda compilation" || true; \
	echo "host C++   : $$HOST_BIN"; \
	"$$HOST_BIN" --version | head -n1; \
	echo "CUDA arch  : sm_$(CUDA_ARCH)"; \
	echo "C++ std    : $(CXX_STD)"; \
	echo "DEBUG      : $(DEBUG)"; \
	echo "PROFILE    : $(PROFILE)"; \
	echo "============================="; \
	\
	if [[ -n "$$CUDA_MAJOR" && -n "$$GCC_MAJOR" ]]; then \
		if (( CUDA_MAJOR <= 10 && GCC_MAJOR > 8 )); then \
			echo; \
			echo "ERROR: incompatible CUDA/GCC combination."; \
			echo "CUDA $$CUDA_MAJOR does not support GCC $$GCC_MAJOR."; \
			echo; \
			echo "On AQUA use:"; \
			echo "  module purge"; \
			echo "  module load gcc10.3.0"; \
			echo "  module load cuda12.2"; \
			echo "  hash -r"; \
			exit 1; \
		fi; \
	fi

# ============================================================================
# CUDA build
# ============================================================================

$(TARGET): $(CUDA_SOURCES) | check-env
	$(NVCC) $(DCVRP_NVFLAGS) src/dynamic_cvrp.cu -o $(TARGET)

# ============================================================================
# CPU-only build
# ============================================================================

cpu: $(CPU_TARGET)

$(CPU_TARGET): src/dynamic_cvrp.cu $(wildcard src/*.hpp)
	$(CXX) $(CPU_FLAGS) \
		-DDCVRP_CPU_ONLY \
		-x c++ src/dynamic_cvrp.cu \
		-o $(CPU_TARGET)

# ============================================================================
# Utility targets
# ============================================================================

info: check-env
	@echo
	@echo "CUDA flags:"
	@echo "  $(DCVRP_NVFLAGS)"

clean:
	rm -f $(TARGET) $(CPU_TARGET)