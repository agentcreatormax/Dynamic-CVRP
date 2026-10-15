// ============================================================================
// gpu_warp.cuh - Opti-3 GPU dynamic-CVRP execution engine.
//
// DELETE
//   * mark the whole delete batch, then one warp per route compacts survivors.
//
// ADD
//   * a global frontier holds active/ejected customers waiting for placement;
//   * one logical adaptive MULTI-BLOCK scheduler handles ONE frontier customer
//     at a time; scheduler warps are striped across enough blocks to keep the
//     target routes/thread instead of being capped by block 0;
//   * first encounter builds a cached K-nearest-centroid route list (default 5);
//   * route centroids include the depot once; cached distances are refreshed
//     only when that candidate route version changes;
//   * if a route centroid beats the depot, only K cached routes may be tried;
//     after those attempts the customer becomes Depot-X-Depot singleton;
//   * if the depot beats every centroid, exhaustive visited-route fallback is
//     retained;
//   * scheduler enqueue publishes the exact non-empty route id to a ready-route
//     queue, so workers no longer poll every route queue;
//   * one full warp owns/mutates one route at a time.
//
// REFINEMENT
//   * every live route nominates only its K nearest route-centroid partners;
//   * sparse pairs are split into conflict-free matching rounds;
//   * one full warp per pair evaluates Move8/9, then Move1/SWAP*;
//   * centroids/nearest partners are rebuilt between the two inter-route stages;
//   * one full warp per route performs Move7 intra-route 2-opt.
// ============================================================================
#pragma once

#ifndef DCVRP_CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <unordered_set>
#include <vector>
#include "instance.hpp"
#include "routes.hpp"

#ifdef DCVRP_CPU_ONLY
#define DCVRP_GPU_UNAVAILABLE
#else

#define CUDA_OK(call) do {                                                     \
    cudaError_t e_ = (call);                                                  \
    if (e_ != cudaSuccess) {                                                  \
        std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                     cudaGetErrorString(e_), __FILE__, __LINE__);             \
        std::exit(EXIT_FAILURE);                                              \
    }                                                                          \
} while (0)

namespace dcvrp_cuda {

static constexpr int MAX_QUEUE_CAP = 16;
static constexpr int MAX_LOCAL_CANDIDATES = 8;
// Multi-block scheduler configuration.  The scheduler target is preserved
// across blocks instead of forcing all scheduler warps into block 0.
static constexpr int SCHED_ROUTES_PER_THREAD_TARGET = 8;
static constexpr int SCHED_WARPS_PER_BLOCK = 8;
static constexpr int MAX_ROUTE_WORKER_WARPS = 256;
static constexpr double INF = 1.0e30;
static constexpr double EPS = 1.0e-9;

__device__ __forceinline__ unsigned fullMask() { return 0xffffffffu; }

// All control counters below are modified concurrently by scheduler/worker warps.
// In a persistent kernel, plain loads may be cached/hoisted by the optimizer.
// Read synchronization state through an atomic operation so release builds do
// not spin forever on stale values.
__device__ __forceinline__ int atomicLoadInt(int* p) {
    return atomicAdd(p, 0);
}

// Scheduler debug is opt-in (build with make DEBUG=1).
#ifndef DCVRP_SCHED_DEBUG
#define DCVRP_SCHED_DEBUG 0
#endif
#if DCVRP_SCHED_DEBUG
__device__ __forceinline__ bool dbgTake(int *counter, int limit=200) {
    return atomicAdd(counter, 1) < limit;
}
#endif
template <typename T>
__device__ __forceinline__ T shfl(T x, int lane) {
    return __shfl_sync(fullMask(), x, lane);
}

struct WorkItem {
    int cust;
    int route;
    int depth;
    int banned;
};

#ifndef DCVRP_PROFILE
#define DCVRP_PROFILE 1
#endif

// Device-side counters for the scheduler/worker pipeline. The counters use
// clock64() durations so profiling stays on-device and avoids CUDA printf
// perturbing the scheduling behavior.
struct RepairProfileCounters {
    unsigned long long schedulerCycles;
    unsigned long long schedulerScanCycles;
    unsigned long long schedulerEnqueueCycles;
    // Warp-weighted useful scheduler work. If dt cycles were spent with N
    // scheduler warps doing useful work, add dt*N. This lets us compute the
    // average number of concurrently active scheduler warps over repair time.
    unsigned long long schedulerWarpCycles;
    unsigned long long maxActiveSchedulerWarps;
    unsigned long long schedulerScans;
    unsigned long long schedulerAssignments;
    unsigned long long schedulerSingletons;
    unsigned long long schedulerBlocked;
    unsigned long long schedulerFrontierItems;
    unsigned long long schedulerRouteChecks;
    unsigned long long candidateBuilds;
    unsigned long long candidateCacheSelections;
    unsigned long long depotModeScans;
    unsigned long long readyRouteSignals;
    unsigned long long workerCycles;
    unsigned long long workerTasks;
    unsigned long long workerDirect;
    unsigned long long workerEject;
    unsigned long long workerRetry;
    unsigned long long maxActiveWorkers;
    unsigned long long maxDepth;
    unsigned long long frontierPeak;
    unsigned long long queuePeak;
    unsigned long long repairStart;
    unsigned long long repairEnd;
};

struct RepairProfileReport {
    double schedulerTotalUs = 0.0;       // accumulated scheduler scan + enqueue work
    double schedulerUsPerAssignment = 0.0;
    double schedulerScanUsPerAssignment = 0.0;
    double schedulerEnqueueUsPerAssignment = 0.0;
    double avgActiveSchedulerWarps = 0.0;
    unsigned long long maxActiveSchedulerWarps = 0;
    double workerUsPerTask = 0.0;
    double workerCapacityUs = 0.0;     // Tw / W_resident
    double schedulerToCapacity = 0.0;  // Ts / (Tw/W_resident)
    double avgActiveWorkers = 0.0;     // sum(worker busy cycles) / repair elapsed cycles
    double workerUtilPct = 0.0;        // avgActiveWorkers / W_resident
    double repairKernelUs = 0.0;
    unsigned long long schedulerScans = 0;
    unsigned long long schedulerAssignments = 0;
    unsigned long long schedulerSingletons = 0;
    unsigned long long schedulerBlocked = 0;
    unsigned long long schedulerFrontierItems = 0;
    unsigned long long schedulerRouteChecks = 0;
    unsigned long long candidateBuilds = 0;
    unsigned long long candidateCacheSelections = 0;
    unsigned long long depotModeScans = 0;
    unsigned long long readyRouteSignals = 0;
    unsigned long long workerTasks = 0;
    unsigned long long workerDirect = 0;
    unsigned long long workerEject = 0;
    unsigned long long workerRetry = 0;
    unsigned long long maxActiveWorkers = 0;
    unsigned long long maxDepth = 0;
    unsigned long long frontierPeak = 0;
    unsigned long long queuePeak = 0;
    int schedulerWarps = 1;            // total cooperative scheduler warps across scheduler blocks
    int schedulerThreads = 32;         // total schedulerWarps * 32
    int schedulerBlocks = 1;
    int schedulerWarpsPerBlock = SCHED_WARPS_PER_BLOCK;
    int routesPerSchedulerThreadTarget = 8;
    int workerWarps = 0;               // useful worker warps launched
    int residentWorkerWarps = 0;       // occupancy-based concurrent capacity estimate
};

#if DCVRP_PROFILE
__device__ __forceinline__ void profAdd(unsigned long long* p, unsigned long long v) { atomicAdd(p, v); }
__device__ __forceinline__ void profMax(unsigned long long* p, unsigned long long v) { atomicMax(p, v); }
#endif

struct GpuStateView {
    double* xs;
    double* ys;
    double* dem;
    int* routeNodes;
    int* routeLen;
    double* routeLoad;
    double* cenX;
    double* cenY;
    uint32_t* routeVersion;
    uint32_t* cenVersion;   // seqlock for published centroid only
    int* routeOf;
    int* posOf;
    int* active;
    int* routeCount;
    int routeStride;
    int maxRoutes;
    int maxCustomers;
    double capacity;
};

struct WarpBest {
    double delta;
    int a;
    int b;
    int kind;
};

__device__ __forceinline__ WarpBest betterBest(WarpBest x, WarpBest y) {
    return (y.delta < x.delta) ? y : x;
}

__device__ __forceinline__ WarpBest warpReduceBest(WarpBest x) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        WarpBest y;
        y.delta = __shfl_down_sync(fullMask(), x.delta, off);
        y.a     = __shfl_down_sync(fullMask(), x.a, off);
        y.b     = __shfl_down_sync(fullMask(), x.b, off);
        y.kind  = __shfl_down_sync(fullMask(), x.kind, off);
        x = betterBest(x, y);
    }
    return x;
}

struct PairEval {
    double delta;
    int kind;       // 8,9 = 2-opt* tails; 1 = Move1; 10 = SWAP*
    int i;
    int j;
    int auxI;       // Move1: direction; SWAP*: insertion pos in R
    int auxJ;       // Move1: unused/optional; SWAP*: insertion pos in S
};

struct WarpPair {
    double delta;
    int kind, i, j, auxI, auxJ;
};

__device__ __forceinline__ WarpPair warpReducePair(WarpPair x) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        WarpPair y;
        y.delta = __shfl_down_sync(fullMask(), x.delta, off);
        y.kind  = __shfl_down_sync(fullMask(), x.kind, off);
        y.i     = __shfl_down_sync(fullMask(), x.i, off);
        y.j     = __shfl_down_sync(fullMask(), x.j, off);
        y.auxI  = __shfl_down_sync(fullMask(), x.auxI, off);
        y.auxJ  = __shfl_down_sync(fullMask(), x.auxJ, off);
        if (y.delta < x.delta) x = y;
    }
    return x;
}

__device__ __forceinline__ double dist(const GpuStateView& s, int a, int b) {
    const double dx = s.xs[a] - s.xs[b];
    const double dy = s.ys[a] - s.ys[b];
    return round(sqrt(dx * dx + dy * dy));
}

__device__ __forceinline__ int at(const GpuStateView& s, int r, int p) {
    const int L = s.routeLen[r];
    return (p >= 0 && p < L) ? s.routeNodes[r * s.routeStride + p] : 0;
}

__device__ __forceinline__ double edgeDelta(const GpuStateView& s, int a, int x, int b) {
    return dist(s, a, x) + dist(s, x, b) - dist(s, a, b);
}

// Customer at position p in the route after removing yPos.
__device__ __forceinline__ int reducedNode(const GpuStateView& s,
                                           int r, int yPos, int p) {
    return (p < yPos) ? at(s, r, p) : at(s, r, p + 1);
}

// Gap p in the reduced route. For an original route of length L, p is in
// [0,L-1] and there are L gaps after one customer is removed.
__device__ __forceinline__ void reducedGap(const GpuStateView& s,
                                           int r, int yPos, int p,
                                           int& prev, int& next) {
    const int L = s.routeLen[r];
    if (p == 0) prev = 0;
    else        prev = reducedNode(s, r, yPos, p - 1);

    if (p == L - 1) next = 0;
    else             next = reducedNode(s, r, yPos, p);
}

__device__ __forceinline__ void recomputeRouteCentroidWarp(const GpuStateView& s, int r) {
    const int lane = threadIdx.x & 31;
    const int L = s.routeLen[r];
    const int base = r * s.routeStride;
    // Route centroid includes the depot ONCE. Lane 0 seeds the depot and all
    // lanes then add their strided customer subset.
    double sx = (lane == 0) ? s.xs[0] : 0.0;
    double sy = (lane == 0) ? s.ys[0] : 0.0;
    for (int p = lane; p < L; p += 32) {
        const int v = s.routeNodes[base + p];
        sx += s.xs[v];
        sy += s.ys[v];
    }
    for (int off = 16; off > 0; off >>= 1) {
        sx += __shfl_down_sync(fullMask(), sx, off);
        sy += __shfl_down_sync(fullMask(), sy, off);
    }
    if (lane == 0) {
        const double cx = sx / (double)(L + 1);
        const double cy = sy / (double)(L + 1);
        // Publish the centroid with a tiny seqlock window. During the larger
        // route mutation the previously committed centroid remains readable,
        // allowing the scheduler to rank even a currently busy route.
        atomicAdd(&s.cenVersion[r], 1u); // odd = centroid publication in progress
        s.cenX[r] = cx;
        s.cenY[r] = cy;
        __threadfence();
        atomicAdd(&s.cenVersion[r], 1u); // even = stable published centroid
    }
}

__device__ __forceinline__ bool readPublishedCentroid(
        const GpuStateView& s, int r, double& cx, double& cy) {
    for (int attempt = 0; attempt < 8; ++attempt) {
        const uint32_t v1 = atomicAdd(&s.cenVersion[r], 0u);
        if (v1 & 1u) continue;
        cx = s.cenX[r];
        cy = s.cenY[r];
        __threadfence();
        const uint32_t v2 = atomicAdd(&s.cenVersion[r], 0u);
        if (v1 == v2 && !(v2 & 1u)) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// DELETE: mark every deleted customer first, then one warp per route removes
// all marked nodes and reconstructs the route in one pass.
// ---------------------------------------------------------------------------
__global__ void markDeletedKernel(GpuStateView s, const int* ids, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int v = ids[i];
    if (v < 0 || v >= s.maxCustomers) return;
    s.active[v] = 0;
    s.routeOf[v] = -1;
    s.posOf[v] = -1;
}

__global__ void compactDeletedRoutesKernel(GpuStateView s) {
    const int globalWarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int m = atomicLoadInt(s.routeCount);
    if (globalWarp >= m) return;
    const int r = globalWarp;
    const int L = s.routeLen[r];
    const int base = r * s.routeStride;

    // Compaction is already lane-0 serial in the original implementation, so
    // a routeStride-sized shared-memory scratch array is unnecessary.  Compact
    // in-place with a standard two-pointer pass (n <= p, therefore writes can
    // never overwrite a future unread element).  This removes the DELETE-side
    // shared-memory cliff for large ADD batches / large routeStride values.
    if (lane == 0) {
        int n = 0;
        for (int p = 0; p < L; ++p) {
            const int v = s.routeNodes[base + p];
            if (v > 0 && s.active[v]) s.routeNodes[base + n++] = v;
        }
        for (int p = n; p < L; ++p) s.routeNodes[base + p] = 0;
        s.routeLen[r] = n;
        double load = 0.0;
        for (int p = 0; p < n; ++p) load += s.dem[s.routeNodes[base + p]];
        s.routeLoad[r] = load;
    }
    __syncwarp();
    recomputeRouteCentroidWarp(s, r);
    __syncwarp();
    if (lane == 0) atomicAdd(&s.routeVersion[r], 1u);
    __syncwarp();

    const int n = s.routeLen[r];
    for (int p = lane; p < n; p += 32) {
        const int v = s.routeNodes[base + p];
        s.routeOf[v] = r;
        s.posOf[v] = p;
    }
}

// ---------------------------------------------------------------------------
// ADD scheduler + reusable route-worker warps.
//
// Policy:
//   1) One complete device warp (global warp 0) schedules the global frontier.
//      Its 32 lanes cooperatively scan route centroids for each READY customer,
//      then lane 0 selects the globally smallest customer->route centroid key.
//      The preference is the smallest vertex-to-route-centroid distance among
//      routes satisfying the promising criterion. If that route is busy or its
//      queue is full, the vertex remains at the frontier until the scheduler
//      can reconsider the same preferred route.
//   2) All remaining warps are a reusable pool. An idle warp scans route queues
//      and atomically claims one non-empty FREE route. A route therefore has at
//      most one mutating warp at a time.
//   3) The worker always consumes the TOP queue entry; queue priority is never
//      changed by warp election. __shfl_sync is only used to broadcast the
//      selected customer and for candidate reductions.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool wasRouteVisited(
        const uint32_t* visited, int wordsPerCustomer, int x, int r) {
    const size_t idx = (size_t)x * wordsPerCustomer + (r >> 5);
    return (visited[idx] & (1u << (r & 31))) != 0;
}

__device__ __forceinline__ void markRouteVisited(
        uint32_t* visited, int wordsPerCustomer, int x, int r) {
    const size_t idx = (size_t)x * wordsPerCustomer + (r >> 5);
    atomicOr(&visited[idx], 1u << (r & 31));
}

// ---------------------------------------------------------------------------
// Opti-3 scheduler locality helpers.
//
// First encounter of a customer:
//   * scheduler threads scan route centroids ONCE and build the K nearest route
//     cache (K = min(--hmax, MAX_LOCAL_CANDIDATES), default 5);
//   * if the nearest route centroid is closer than the depot, LOCAL mode uses
//     only those K cached routes;
//   * if the depot itself is closer than every route centroid, DEPOT mode keeps
//     the exhaustive visited-route fallback requested by the experiment.
//
// Cached route distances are versioned with routeVersion.  If a candidate route
// changes length/load/membership, routeVersion changes and only that cached
// candidate distance is refreshed.  We intentionally do NOT rescan every route
// on each retry in LOCAL mode.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void insertTopKLocal(
        double p2, int r, double* bestP2, int* bestR, int k) {
    if (r < 0 || k <= 0) return;
    int pos = k;
    for (int i = 0; i < k; ++i) {
        if (p2 < bestP2[i] ||
            (p2 == bestP2[i] && (bestR[i] < 0 || r < bestR[i]))) {
            pos = i; break;
        }
    }
    if (pos >= k) return;
    for (int i = k - 1; i > pos; --i) {
        bestP2[i] = bestP2[i - 1];
        bestR[i] = bestR[i - 1];
    }
    bestP2[pos] = p2;
    bestR[pos] = r;
}

// Build K nearest route-centroid candidates for x with one full route scan.
// Per lane we keep a register-resident top-K.  Each warp merges its 32 lane
// lists, then thread 0 merges at most schedulerWarps*K warp winners.
__device__ __forceinline__ bool blockBuildCandidateCache(
        const GpuStateView& s, int x, int schedulerWarps, int localK,
        double* shWarpTopP2, int* shWarpTopR,
        unsigned long long* shWarpChecks, int* shWarpFlags,
        int* candRoutes, double* candPriority2, uint32_t* candVersion,
        int* candCount, int* candMode,
        unsigned long long& routeChecks) {
    const int lane = threadIdx.x & 31;
    const int warpInBlock = threadIdx.x >> 5;
    const int tid = threadIdx.x;
    const int schedulerThreads = schedulerWarps * 32;
    const int m = atomicLoadInt(s.routeCount);

    double localP2[MAX_LOCAL_CANDIDATES];
    int localR[MAX_LOCAL_CANDIDATES];
#pragma unroll
    for (int i = 0; i < MAX_LOCAL_CANDIDATES; ++i) {
        localP2[i] = INF; localR[i] = -1;
    }

    unsigned long long checks = 0;
    int incomplete = 0;
    if (warpInBlock < schedulerWarps) {
        for (int r = tid; r < m; r += schedulerThreads) {
            if (atomicLoadInt(&s.routeLen[r]) <= 0) continue;
            double cx = 0.0, cy = 0.0;
            if (!readPublishedCentroid(s, r, cx, cy)) {
                incomplete = 1;
                continue;
            }
            ++checks;
            const double dx = s.xs[x] - cx;
            const double dy = s.ys[x] - cy;
            insertTopKLocal(dx * dx + dy * dy, r, localP2, localR, localK);
        }
    }

    // Merge each warp's 32 local top-K lists without spilling all lane lists
    // into shared memory.  Repeatedly reduce the current head of every lane.
    int cursor = 0;
    for (int k = 0; k < localK; ++k) {
        double p2 = (cursor < localK) ? localP2[cursor] : INF;
        int r = (cursor < localK) ? localR[cursor] : -1;
        for (int off = 16; off > 0; off >>= 1) {
            const double op = __shfl_down_sync(fullMask(), p2, off);
            const int oroute = __shfl_down_sync(fullMask(), r, off);
            if (op < p2 || (op == p2 && oroute >= 0 && (r < 0 || oroute < r))) {
                p2 = op; r = oroute;
            }
        }
        const double winP = shfl(p2, 0);
        const int winR = shfl(r, 0);
        if (lane == 0) {
            const int base = warpInBlock * MAX_LOCAL_CANDIDATES;
            shWarpTopP2[base + k] = winP;
            shWarpTopR[base + k] = winR;
        }
        if (cursor < localK && localR[cursor] == winR && winR >= 0) ++cursor;
    }

    // Sum route checks / incomplete flags per warp.
    unsigned long long sumChecks = checks;
    int flags = incomplete;
    for (int off = 16; off > 0; off >>= 1) {
        sumChecks += __shfl_down_sync(fullMask(), sumChecks, off);
        flags |= __shfl_down_sync(fullMask(), flags, off);
    }
    if (lane == 0) {
        shWarpChecks[warpInBlock] = sumChecks;
        shWarpFlags[warpInBlock] = flags;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double bestP[MAX_LOCAL_CANDIDATES];
        int bestR[MAX_LOCAL_CANDIDATES];
        for (int i = 0; i < MAX_LOCAL_CANDIDATES; ++i) {
            bestP[i] = INF; bestR[i] = -1;
        }
        unsigned long long totalChecks = 0;
        int anyIncomplete = 0;
        for (int w = 0; w < schedulerWarps; ++w) {
            totalChecks += shWarpChecks[w];
            anyIncomplete |= shWarpFlags[w];
            const int base = w * MAX_LOCAL_CANDIDATES;
            for (int k = 0; k < localK; ++k)
                insertTopKLocal(shWarpTopP2[base + k], shWarpTopR[base + k],
                                bestP, bestR, localK);
        }
        routeChecks = totalChecks;

        // A centroid was being published while we scanned. Rebuild next turn
        // rather than freezing a candidate list that might miss the real nearest
        // route. The seqlock window is tiny, so this should be rare.
        if (anyIncomplete) {
            candMode[x] = 0;
            candCount[x] = 0;
        } else {
            int count = 0;
            const size_t base = (size_t)x * MAX_LOCAL_CANDIDATES;
            for (int k = 0; k < localK; ++k) {
                if (bestR[k] < 0) break;
                candRoutes[base + count] = bestR[k];
                candPriority2[base + count] = bestP[k];
                candVersion[base + count] = atomicAdd(&s.routeVersion[bestR[k]], 0u);
                ++count;
            }
            candCount[x] = count;

            const double ddx = s.xs[x] - s.xs[0];
            const double ddy = s.ys[x] - s.ys[0];
            const double depot2 = ddx * ddx + ddy * ddy;
            const double tol = EPS * (1.0 + depot2);
            // 1=LOCAL(K-nearest only), 2=DEPOT(exhaust all unvisited routes).
            candMode[x] = (count > 0 && bestP[0] <= depot2 + tol) ? 1 : 2;
        }
    }
    __syncthreads();
    return candMode[x] != 0;
}

// LOCAL mode: select the best currently usable route from the cached K.
// A changed candidate route refreshes only its own centroid distance using the
// routeVersion stamp.  We do not rescan all routes.
__device__ __forceinline__ int chooseCachedLocalRoute(
        const GpuStateView& s, int x, int banned,
        const uint32_t* visited, int visitedWordsPerCustomer,
        int* routeOwner, int* qLen, int queueCap,
        int* candRoutes, double* candPriority2, uint32_t* candVersion,
        const int* candCount,
        bool& anyUnvisited, bool& sawBusy) {
    anyUnvisited = false; sawBusy = false;
    int bestR = -1; double bestP = INF;
    const size_t base = (size_t)x * MAX_LOCAL_CANDIDATES;
    const int count = candCount[x];
    for (int k = 0; k < count; ++k) {
        const size_t z = base + k;
        const int r = candRoutes[z];
        if (r < 0 || r == banned) continue;
        if (wasRouteVisited(visited, visitedWordsPerCustomer, x, r)) continue;
        if (atomicLoadInt(&s.routeLen[r]) <= 0) continue;
        anyUnvisited = true;

        const uint32_t v = atomicAdd(&s.routeVersion[r], 0u);
        if (candVersion[z] != v) {
            double cx=0.0, cy=0.0;
            if (!readPublishedCentroid(s, r, cx, cy)) {
                sawBusy = true; continue;
            }
            const double dx=s.xs[x]-cx, dy=s.ys[x]-cy;
            candPriority2[z] = dx*dx + dy*dy;
            candVersion[z] = v;
        }
        if (atomicLoadInt(&routeOwner[r]) != -1 || atomicLoadInt(&qLen[r]) >= queueCap) {
            sawBusy = true; continue;
        }
        const double p2 = candPriority2[z];
        if (p2 < bestP || (p2 == bestP && (bestR < 0 || r < bestR))) {
            bestP = p2; bestR = r;
        }
    }
    return bestR;
}

// DEPOT mode: the depot was closer than every centroid when the cache was
// built.  Honor the requested exhaustive policy: all unvisited live routes are
// eligible before singleton fallback. Scheduler lanes still split this scan.
__device__ __forceinline__ int blockChooseDepotRoute(
        const GpuStateView& s, int x, int banned,
        const uint32_t* visited, int visitedWordsPerCustomer,
        int schedulerWarps, int* routeOwner, int* qLen, int queueCap,
        double* shWarpBestPriority2, int* shWarpBestRoute,
        unsigned long long* shWarpChecks, int* shWarpFlags,
        bool& anyUnvisited, bool& sawBusy,
        unsigned long long& routeChecks) {
    const int lane = threadIdx.x & 31;
    const int warpInBlock = threadIdx.x >> 5;
    const int tid = threadIdx.x;
    const int schedulerThreads = schedulerWarps * 32;
    const int m = atomicLoadInt(s.routeCount);

    double localBest = INF; int localRoute = -1;
    unsigned long long checks = 0; int flags = 0; // bit0 any, bit1 busy
    if (warpInBlock < schedulerWarps) {
        for (int r = tid; r < m; r += schedulerThreads) {
            if (r == banned || atomicLoadInt(&s.routeLen[r]) <= 0) continue;
            if (wasRouteVisited(visited, visitedWordsPerCustomer, x, r)) continue;
            flags |= 1;
            if (atomicLoadInt(&routeOwner[r]) != -1 || atomicLoadInt(&qLen[r]) >= queueCap) {
                flags |= 2; continue;
            }
            double cx=0.0, cy=0.0;
            if (!readPublishedCentroid(s, r, cx, cy)) { flags |= 2; continue; }
            ++checks;
            const double dx=s.xs[x]-cx, dy=s.ys[x]-cy;
            const double p2=dx*dx+dy*dy;
            if (p2 < localBest || (p2 == localBest && (localRoute < 0 || r < localRoute))) {
                localBest=p2; localRoute=r;
            }
        }
    }

    for (int off=16; off>0; off>>=1) {
        const double op=__shfl_down_sync(fullMask(),localBest,off);
        const int oroute=__shfl_down_sync(fullMask(),localRoute,off);
        const unsigned long long oc=__shfl_down_sync(fullMask(),checks,off);
        const int of=__shfl_down_sync(fullMask(),flags,off);
        if (op < localBest || (op == localBest && oroute >= 0 && (localRoute < 0 || oroute < localRoute))) {
            localBest=op; localRoute=oroute;
        }
        checks += oc; flags |= of;
    }
    if (lane==0) {
        shWarpBestPriority2[warpInBlock]=localBest;
        shWarpBestRoute[warpInBlock]=localRoute;
        shWarpChecks[warpInBlock]=checks;
        shWarpFlags[warpInBlock]=flags;
    }
    __syncthreads();

    if (warpInBlock==0) {
        double bp=(lane<schedulerWarps)?shWarpBestPriority2[lane]:INF;
        int br=(lane<schedulerWarps)?shWarpBestRoute[lane]:-1;
        unsigned long long bc=(lane<schedulerWarps)?shWarpChecks[lane]:0ULL;
        int bf=(lane<schedulerWarps)?shWarpFlags[lane]:0;
        for(int off=16;off>0;off>>=1){
            const double op=__shfl_down_sync(fullMask(),bp,off);
            const int oroute=__shfl_down_sync(fullMask(),br,off);
            if(op<bp || (op==bp && oroute>=0 && (br<0 || oroute<br))){bp=op;br=oroute;}
            bc += __shfl_down_sync(fullMask(),bc,off);
            bf |= __shfl_down_sync(fullMask(),bf,off);
        }
        if(lane==0){
            shWarpBestPriority2[0]=bp; shWarpBestRoute[0]=br;
            shWarpChecks[0]=bc; shWarpFlags[0]=bf;
        }
    }
    __syncthreads();
    routeChecks=shWarpChecks[0];
    anyUnvisited=(shWarpFlags[0]&1)!=0;
    sawBusy=(shWarpFlags[0]&2)!=0;
    return shWarpBestRoute[0];
}

__device__ __forceinline__ int createSingletonRouteSafe(
        const GpuStateView& s, int x, int* routeOwner) {
    // Reuse an empty route only after reserving it. This path is rare and is
    // executed only by the single scheduler thread.
    int m = atomicLoadInt(s.routeCount);
    int r = -1;
    for (int k = 0; k < m; ++k) {
        if (atomicCAS(&routeOwner[k], -1, -2) != -1) continue;
        if (s.routeLen[k] == 0) {
            r = k;
            break; // keep reservation
        }
        atomicExch(&routeOwner[k], -1);
    }

    if (r < 0) {
        r = atomicAdd(s.routeCount, 1);
        if (r >= s.maxRoutes) {
            atomicSub(s.routeCount, 1);
            return -1;
        }
        // New slots are zero-length and routeOwner[] was initialized to FREE.
        atomicExch(&routeOwner[r], -2);
    }

    const int base = r * s.routeStride;
    s.routeNodes[base] = x;
    s.routeLen[r] = 1;
    s.routeLoad[r] = s.dem[x];
    atomicAdd(&s.cenVersion[r], 1u);
    s.cenX[r] = 0.5 * (s.xs[0] + s.xs[x]);
    s.cenY[r] = 0.5 * (s.ys[0] + s.ys[x]);
    __threadfence();
    atomicAdd(&s.cenVersion[r], 1u);
    atomicAdd(&s.routeVersion[r], 1u);
    s.routeOf[x] = r;
    s.posOf[x] = 0;
    __threadfence();
    atomicExch(&routeOwner[r], -1);
    return r;
}

// Append-only device work log. ready[] has three states:
//   0 = reserved/unpublished, 1 = ready, 2 = consumed.
// Producers publish the WorkItem before incrementing frontierPending. This lets
// the scheduler choose arbitrary priority entries without relying on FIFO head
// order, while termination uses the exact pending count.
__device__ __forceinline__ bool pushFrontier(
        WorkItem w, WorkItem* frontier, int* frontierReady,
        int* tail, int* pending, int cap, int* repairError,
        RepairProfileCounters* profile) {
    const int p = atomicAdd(tail, 1);
    if (p >= cap) {
        atomicCAS(repairError, 0, 1);
        return false;
    }
    frontier[p] = w;
    __threadfence();
    atomicExch(&frontierReady[p], 1);
    const int nowPending = atomicAdd(pending, 1) + 1;
#if DCVRP_PROFILE
    if (profile) profMax(&profile->frontierPeak, (unsigned long long)nowPending);
#else
    (void)profile;
#endif
    return true;
}

__device__ __forceinline__ void sortQueueForCurrentCentroid(
        const GpuStateView& s, int r, int* qLen,
        WorkItem* qItems, double* qPriority) {
    const int b = r * MAX_QUEUE_CAP;
    const int n = qLen[r];
    const double cx = s.cenX[r], cy = s.cenY[r];
    for (int k = 0; k < n; ++k) {
        const int x = qItems[b + k].cust;
        const double dx = s.xs[x] - cx, dy = s.ys[x] - cy;
        qPriority[b + k] = dx * dx + dy * dy;
    }
    for (int i = 1; i < n; ++i) {
        WorkItem w = qItems[b + i];
        double p = qPriority[b + i];
        int j = i - 1;
        while (j >= 0 && p < qPriority[b + j]) {
            qItems[b + j + 1] = qItems[b + j];
            qPriority[b + j + 1] = qPriority[b + j];
            --j;
        }
        qItems[b + j + 1] = w;
        qPriority[b + j + 1] = p;
    }
}

__device__ __forceinline__ bool signalReadyRoute(
        int r, int* readyRoutes, int* readyRouteTail, int readyRouteCap,
        int* repairError, RepairProfileCounters* profile) {
    // Scheduler thread 0 is the only producer of route tokens. Publish the
    // route id before advancing tail so workers never observe an unwritten slot.
    const int p = atomicLoadInt(readyRouteTail);
    if (p >= readyRouteCap) {
        atomicCAS(repairError, 0, 4);
        return false;
    }
    readyRoutes[p] = r;
    __threadfence();
    atomicExch(readyRouteTail, p + 1);
#if DCVRP_PROFILE
    if (profile) profAdd(&profile->readyRouteSignals, 1);
#else
    (void)profile;
#endif
    return true;
}

// ---------------------------------------------------------------------------
// Multi-block scheduler helpers.
//
// All scheduler blocks belong to the SAME persistent kernel launch.  Host-side
// launch sizing below guarantees that the entire grid fits concurrently, so a
// small scheduler-only global barrier is safe: no scheduler block waits for a
// block that cannot become resident.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void schedulerBlocksBarrier(
        int* barrierCount, int* barrierSense, int schedulerBlocks,
        int* blockLocalSense) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const int target = 1 - *blockLocalSense;
        // Publish this block's global writes before announcing arrival.
        __threadfence();
        const int ticket = atomicAdd(barrierCount, 1);
        if (ticket == schedulerBlocks - 1) {
            atomicExch(barrierCount, 0);
            __threadfence();
            atomicExch(barrierSense, target);
        } else {
            while (atomicLoadInt(barrierSense) != target) {}
        }
        *blockLocalSense = target;
    }
    __syncthreads();
}

// Each scheduler block scans a DISJOINT slice of all routes for the same
// customer.  The block writes one top-K summary; block 0 merges summaries after
// a schedulerBlocksBarrier().
__device__ __forceinline__ void schedulerCandidatePartial(
        const GpuStateView& s, int x, int totalSchedulerWarps, int localK,
        double* shWarpTopP2, int* shWarpTopR,
        unsigned long long* shWarpChecks, int* shWarpFlags,
        double* blockTopP2, int* blockTopR,
        unsigned long long* blockChecks, int* blockFlags) {
    const int lane = threadIdx.x & 31;
    const int warpInBlock = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;
    const int blockWarpBase = (int)blockIdx.x * warpsPerBlock;
    const int globalSchedWarp = blockWarpBase + warpInBlock;
    const bool activeWarp = globalSchedWarp < totalSchedulerWarps;
    const int totalSchedulerThreads = totalSchedulerWarps * 32;
    const int globalTid = globalSchedWarp * 32 + lane;
    const int m = atomicLoadInt(s.routeCount);

    double localP2[MAX_LOCAL_CANDIDATES];
    int localR[MAX_LOCAL_CANDIDATES];
#pragma unroll
    for (int i = 0; i < MAX_LOCAL_CANDIDATES; ++i) {
        localP2[i] = INF;
        localR[i] = -1;
    }

    unsigned long long checks = 0;
    int incomplete = 0;
    if (activeWarp) {
        for (int r = globalTid; r < m; r += totalSchedulerThreads) {
            if (atomicLoadInt(&s.routeLen[r]) <= 0) continue;
            double cx = 0.0, cy = 0.0;
            if (!readPublishedCentroid(s, r, cx, cy)) {
                incomplete = 1;
                continue;
            }
            ++checks;
            const double dx = s.xs[x] - cx;
            const double dy = s.ys[x] - cy;
            insertTopKLocal(dx * dx + dy * dy, r, localP2, localR, localK);
        }
    }

    int cursor = 0;
    if (activeWarp) {
        for (int k = 0; k < localK; ++k) {
            double p2 = (cursor < localK) ? localP2[cursor] : INF;
            int r = (cursor < localK) ? localR[cursor] : -1;
            for (int off = 16; off > 0; off >>= 1) {
                const double op = __shfl_down_sync(fullMask(), p2, off);
                const int oroute = __shfl_down_sync(fullMask(), r, off);
                if (op < p2 || (op == p2 && oroute >= 0 && (r < 0 || oroute < r))) {
                    p2 = op;
                    r = oroute;
                }
            }
            const double winP = shfl(p2, 0);
            const int winR = shfl(r, 0);
            if (lane == 0) {
                const int base = warpInBlock * MAX_LOCAL_CANDIDATES;
                shWarpTopP2[base + k] = winP;
                shWarpTopR[base + k] = winR;
            }
            if (cursor < localK && localR[cursor] == winR && winR >= 0) ++cursor;
        }
    }

    unsigned long long sumChecks = checks;
    int flags = incomplete;
    if (activeWarp) {
        for (int off = 16; off > 0; off >>= 1) {
            sumChecks += __shfl_down_sync(fullMask(), sumChecks, off);
            flags |= __shfl_down_sync(fullMask(), flags, off);
        }
        if (lane == 0) {
            shWarpChecks[warpInBlock] = sumChecks;
            shWarpFlags[warpInBlock] = flags;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double bestP[MAX_LOCAL_CANDIDATES];
        int bestR[MAX_LOCAL_CANDIDATES];
        for (int i = 0; i < MAX_LOCAL_CANDIDATES; ++i) {
            bestP[i] = INF;
            bestR[i] = -1;
        }
        const int remaining = totalSchedulerWarps - blockWarpBase;
        const int activeWarps = remaining <= 0 ? 0 : (remaining < warpsPerBlock ? remaining : warpsPerBlock);
        unsigned long long totalChecks = 0;
        int anyIncomplete = 0;
        for (int w = 0; w < activeWarps; ++w) {
            totalChecks += shWarpChecks[w];
            anyIncomplete |= shWarpFlags[w];
            const int base = w * MAX_LOCAL_CANDIDATES;
            for (int k = 0; k < localK; ++k)
                insertTopKLocal(shWarpTopP2[base + k], shWarpTopR[base + k],
                                bestP, bestR, localK);
        }
        const size_t outBase = (size_t)blockIdx.x * MAX_LOCAL_CANDIDATES;
        for (int k = 0; k < localK; ++k) {
            blockTopP2[outBase + k] = bestP[k];
            blockTopR[outBase + k] = bestR[k];
        }
        blockChecks[blockIdx.x] = totalChecks;
        blockFlags[blockIdx.x] = anyIncomplete;
        __threadfence();
    }
    __syncthreads();
}

// Same split for DEPOT mode: each block returns one best route plus flags.
__device__ __forceinline__ void schedulerDepotPartial(
        const GpuStateView& s, int x, int banned,
        const uint32_t* visited, int visitedWordsPerCustomer,
        int totalSchedulerWarps, int* routeOwner, int* qLen, int queueCap,
        double* shWarpBestPriority2, int* shWarpBestRoute,
        unsigned long long* shWarpChecks, int* shWarpFlags,
        double* blockTopP2, int* blockTopR,
        unsigned long long* blockChecks, int* blockFlags) {
    const int lane = threadIdx.x & 31;
    const int warpInBlock = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;
    const int blockWarpBase = (int)blockIdx.x * warpsPerBlock;
    const int globalSchedWarp = blockWarpBase + warpInBlock;
    const bool activeWarp = globalSchedWarp < totalSchedulerWarps;
    const int totalSchedulerThreads = totalSchedulerWarps * 32;
    const int globalTid = globalSchedWarp * 32 + lane;
    const int m = atomicLoadInt(s.routeCount);

    double localBest = INF;
    int localRoute = -1;
    unsigned long long checks = 0;
    int flags = 0; // bit0 any unvisited, bit1 saw busy

    if (activeWarp) {
        for (int r = globalTid; r < m; r += totalSchedulerThreads) {
            if (r == banned || atomicLoadInt(&s.routeLen[r]) <= 0) continue;
            if (wasRouteVisited(visited, visitedWordsPerCustomer, x, r)) continue;
            flags |= 1;
            if (atomicLoadInt(&routeOwner[r]) != -1 || atomicLoadInt(&qLen[r]) >= queueCap) {
                flags |= 2;
                continue;
            }
            double cx = 0.0, cy = 0.0;
            if (!readPublishedCentroid(s, r, cx, cy)) {
                flags |= 2;
                continue;
            }
            ++checks;
            const double dx = s.xs[x] - cx;
            const double dy = s.ys[x] - cy;
            const double p2 = dx * dx + dy * dy;
            if (p2 < localBest || (p2 == localBest && (localRoute < 0 || r < localRoute))) {
                localBest = p2;
                localRoute = r;
            }
        }

        for (int off = 16; off > 0; off >>= 1) {
            const double op = __shfl_down_sync(fullMask(), localBest, off);
            const int oroute = __shfl_down_sync(fullMask(), localRoute, off);
            const unsigned long long oc = __shfl_down_sync(fullMask(), checks, off);
            const int of = __shfl_down_sync(fullMask(), flags, off);
            if (op < localBest || (op == localBest && oroute >= 0 && (localRoute < 0 || oroute < localRoute))) {
                localBest = op;
                localRoute = oroute;
            }
            checks += oc;
            flags |= of;
        }
        if (lane == 0) {
            shWarpBestPriority2[warpInBlock] = localBest;
            shWarpBestRoute[warpInBlock] = localRoute;
            shWarpChecks[warpInBlock] = checks;
            shWarpFlags[warpInBlock] = flags;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const int remaining = totalSchedulerWarps - blockWarpBase;
        const int activeWarps = remaining <= 0 ? 0 : (remaining < warpsPerBlock ? remaining : warpsPerBlock);
        double bestP = INF;
        int bestR = -1;
        unsigned long long totalChecks = 0;
        int allFlags = 0;
        for (int w = 0; w < activeWarps; ++w) {
            const double p2 = shWarpBestPriority2[w];
            const int r = shWarpBestRoute[w];
            if (p2 < bestP || (p2 == bestP && r >= 0 && (bestR < 0 || r < bestR))) {
                bestP = p2;
                bestR = r;
            }
            totalChecks += shWarpChecks[w];
            allFlags |= shWarpFlags[w];
        }
        const size_t outBase = (size_t)blockIdx.x * MAX_LOCAL_CANDIDATES;
        blockTopP2[outBase] = bestP;
        blockTopR[outBase] = bestR;
        blockChecks[blockIdx.x] = totalChecks;
        blockFlags[blockIdx.x] = allFlags;
        __threadfence();
    }
    __syncthreads();
}

__global__ void addWorkerPoolKernel(
        GpuStateView s,
        WorkItem* frontier, int* frontierReady,
        int* frontierHead, int* frontierTail, int* frontierPending,
        int frontierCap, int* repairError,
        int* qLen, WorkItem* qItems, double* qPriority, int* qLock, int* qTotal,
        WorkItem* ejected, int* ejectedCount,
        uint32_t* visited, int visitedWordsPerCustomer,
        int* candRoutes, double* candPriority2, uint32_t* candVersion,
        int* candCount, int* candMode,
        int* routeOwner,
        int* readyRoutes, int* readyRouteHead, int* readyRouteTail,
        int* routeSignaled, int readyRouteCap,
        int queueCap, int hmax, int schedulerWarps, int schedulerBlocks,
        int workerWarps, int* workerScratch,
        int* schedCtl, WorkItem* schedWork,
        int* schedBarrierCount, int* schedBarrierSense,
        double* schedBlockTopP2, int* schedBlockTopR,
        unsigned long long* schedBlockChecks, int* schedBlockFlags,
        int* activeWorkers, int* done, int* dbgCounter,
        RepairProfileCounters* profile) {

    const int warpsPerBlock = blockDim.x >> 5;
    const int warpInBlock = threadIdx.x >> 5;
    const int globalWarp = blockIdx.x * warpsPerBlock + warpInBlock;
    const int lane = threadIdx.x & 31;
    const int localK = (hmax < 1) ? 1 : ((hmax > MAX_LOCAL_CANDIDATES) ? MAX_LOCAL_CANDIDATES : hmax);

    // Per-block scheduler reduction scratch.  The configured block size is
    // SCHED_WARPS_PER_BLOCK (8 warps), but keeping 32 slots makes the helper
    // robust if that compile-time choice is raised later.
    __shared__ double shSchedBestPriority2[32];
    __shared__ int shSchedBestRoute[32];
    __shared__ unsigned long long shSchedChecks[32];
    __shared__ int shSchedFlags[32];
    __shared__ double shWarpTopPriority2[32 * MAX_LOCAL_CANDIDATES];
    __shared__ int shWarpTopRoute[32 * MAX_LOCAL_CANDIDATES];
    __shared__ int shSchedBarrierSense;

    // ------------------------------------------------------------------
    // Opti-3 MULTI-BLOCK scheduler.
    //
    // The first schedulerBlocks blocks cooperate on ONE frontier customer at
    // a time.  Route scans are striped across ALL scheduler warps globally, so
    // the original target (~8 routes / scheduler thread) is preserved even
    // when more than one block is required.  Block 0 remains only the control /
    // enqueue coordinator; it is no longer the sole provider of scheduler
    // threads.
    // ------------------------------------------------------------------
    if ((int)blockIdx.x < schedulerBlocks) {
        if (threadIdx.x == 0) shSchedBarrierSense = 0;
        __syncthreads();

        int skipOnce = -1;                 // meaningful only in block0/thread0
        unsigned long long scanStart = 0;  // meaningful only in block0/thread0

#if DCVRP_PROFILE
        if (blockIdx.x == 0 && threadIdx.x == 0 && profile)
            profile->repairStart = clock64();
#endif
        // Make sure every scheduler block is resident and has initialized its
        // local barrier sense before the first cooperative scan.
        schedulerBlocksBarrier(
            schedBarrierCount, schedBarrierSense, schedulerBlocks,
            &shSchedBarrierSense);

        for (;;) {
            // Block 0 is the coordinator.  All scheduler blocks consume the
            // same control record after the barrier below.
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                schedCtl[0] = atomicLoadInt(repairError);
                schedCtl[1] = atomicLoadInt(frontierPending);
            }
            schedulerBlocksBarrier(
                schedBarrierCount, schedBarrierSense, schedulerBlocks,
                &shSchedBarrierSense);

            if (schedCtl[0] != 0) {
                if (blockIdx.x == 0 && threadIdx.x == 0) {
                    atomicExch(done, 1);
#if DCVRP_PROFILE
                    if (profile) profile->repairEnd = clock64();
#endif
                }
                schedulerBlocksBarrier(
                    schedBarrierCount, schedBarrierSense, schedulerBlocks,
                    &shSchedBarrierSense);
                break;
            }

            const int pending = schedCtl[1];
            if (pending > 0) {
                // Pick one READY frontier item exactly as before.  The route
                // scan for this one customer is what becomes multi-block.
                if (blockIdx.x == 0 && threadIdx.x == 0) {
                    const int h = atomicLoadInt(frontierHead);
                    const int t = atomicLoadInt(frontierTail);
                    int idx = -1;
                    for (int j = h; j < t; ++j) {
                        if (j == skipOnce) continue;
                        if (atomicLoadInt(&frontierReady[j]) == 1) {
                            idx = j;
                            break;
                        }
                    }
                    if (idx < 0 && skipOnce >= h && skipOnce < t &&
                        atomicLoadInt(&frontierReady[skipOnce]) == 1)
                        idx = skipOnce;

                    schedCtl[2] = idx;
                    schedCtl[3] = 0; // valid/built flag
                    if (idx >= 0) {
                        *schedWork = frontier[idx];
                        const int x = schedWork->cust;
                        if (x <= 0 || x >= s.maxCustomers || !s.active[x] ||
                            atomicLoadInt(&s.routeOf[x]) >= 0) {
                            atomicExch(&frontierReady[idx], 2);
                            atomicAdd(frontierPending, -1);
                        } else {
                            schedCtl[3] = 1;
                        }
                    }
                }
                schedulerBlocksBarrier(
                    schedBarrierCount, schedBarrierSense, schedulerBlocks,
                    &shSchedBarrierSense);

                const int idx = schedCtl[2];
                if (idx < 0 || !schedCtl[3]) {
                    if (blockIdx.x == 0 && threadIdx.x == 0) {
                        int nh = atomicLoadInt(frontierHead);
                        const int nt = atomicLoadInt(frontierTail);
                        while (nh < nt && atomicLoadInt(&frontierReady[nh]) == 2) ++nh;
                        atomicExch(frontierHead, nh);
                        skipOnce = -1;
                    }
                    schedulerBlocksBarrier(
                        schedBarrierCount, schedBarrierSense, schedulerBlocks,
                        &shSchedBarrierSense);
                    continue;
                }

                const int x = schedWork->cust;
                if (blockIdx.x == 0 && threadIdx.x == 0) {
                    scanStart = clock64();
                    schedCtl[7] = 1; // cached LOCAL control is one-warp work
                }
                schedulerBlocksBarrier(
                    schedBarrierCount, schedBarrierSense, schedulerBlocks,
                    &shSchedBarrierSense);

                // First encounter: ALL scheduler blocks cooperatively build the
                // K-nearest centroid cache.  Each route is checked by exactly
                // one scheduler thread globally.
                if (atomicLoadInt(&candMode[x]) == 0) {
                    if (blockIdx.x == 0 && threadIdx.x == 0)
                        schedCtl[7] = schedulerWarps;
                    schedulerBlocksBarrier(
                        schedBarrierCount, schedBarrierSense, schedulerBlocks,
                        &shSchedBarrierSense);

                    schedulerCandidatePartial(
                        s, x, schedulerWarps, localK,
                        shWarpTopPriority2, shWarpTopRoute,
                        shSchedChecks, shSchedFlags,
                        schedBlockTopP2, schedBlockTopR,
                        schedBlockChecks, schedBlockFlags);

                    schedulerBlocksBarrier(
                        schedBarrierCount, schedBarrierSense, schedulerBlocks,
                        &shSchedBarrierSense);

                    if (blockIdx.x == 0 && threadIdx.x == 0) {
                        double bestP[MAX_LOCAL_CANDIDATES];
                        int bestR[MAX_LOCAL_CANDIDATES];
                        for (int k = 0; k < MAX_LOCAL_CANDIDATES; ++k) {
                            bestP[k] = INF;
                            bestR[k] = -1;
                        }
                        unsigned long long routeChecks = 0;
                        int anyIncomplete = 0;
                        for (int b = 0; b < schedulerBlocks; ++b) {
                            routeChecks += schedBlockChecks[b];
                            anyIncomplete |= schedBlockFlags[b];
                            const size_t base = (size_t)b * MAX_LOCAL_CANDIDATES;
                            for (int k = 0; k < localK; ++k)
                                insertTopKLocal(
                                    schedBlockTopP2[base + k],
                                    schedBlockTopR[base + k],
                                    bestP, bestR, localK);
                        }

#if DCVRP_PROFILE
                        if (profile) {
                            profAdd(&profile->candidateBuilds, 1);
                            profAdd(&profile->schedulerRouteChecks, routeChecks);
                        }
#endif

                        if (anyIncomplete) {
                            candMode[x] = 0;
                            candCount[x] = 0;
                            schedCtl[3] = 0;
                        } else {
                            int count = 0;
                            const size_t base = (size_t)x * MAX_LOCAL_CANDIDATES;
                            for (int k = 0; k < localK; ++k) {
                                if (bestR[k] < 0) break;
                                candRoutes[base + count] = bestR[k];
                                candPriority2[base + count] = bestP[k];
                                candVersion[base + count] =
                                    atomicAdd(&s.routeVersion[bestR[k]], 0u);
                                ++count;
                            }
                            candCount[x] = count;
                            const double ddx = s.xs[x] - s.xs[0];
                            const double ddy = s.ys[x] - s.ys[0];
                            const double depot2 = ddx * ddx + ddy * ddy;
                            const double tol = EPS * (1.0 + depot2);
                            candMode[x] =
                                (count > 0 && bestP[0] <= depot2 + tol) ? 1 : 2;
                            schedCtl[3] = 1;
                        }
                        __threadfence();
                    }

                    schedulerBlocksBarrier(
                        schedBarrierCount, schedBarrierSense, schedulerBlocks,
                        &shSchedBarrierSense);

                    if (!schedCtl[3]) {
                        if (blockIdx.x == 0 && threadIdx.x == 0) {
                            skipOnce = idx;
#if DCVRP_PROFILE
                            if (profile) profAdd(&profile->schedulerBlocked, 1);
#endif
                        }
                        schedulerBlocksBarrier(
                            schedBarrierCount, schedBarrierSense, schedulerBlocks,
                            &shSchedBarrierSense);
                        continue;
                    }
                }

                if (blockIdx.x == 0 && threadIdx.x == 0) {
                    schedCtl[4] = -1; // selected route
                    schedCtl[5] = 0;  // any unvisited
                    schedCtl[6] = 0;  // saw busy
                }
                schedulerBlocksBarrier(
                    schedBarrierCount, schedBarrierSense, schedulerBlocks,
                    &shSchedBarrierSense);

                const int mode = atomicLoadInt(&candMode[x]);
                if (mode == 1) {
                    // Cached LOCAL mode is intentionally small (K<=8), so the
                    // coordinator handles it directly.
                    if (blockIdx.x == 0 && threadIdx.x == 0) {
                        bool any = false, busy = false;
                        const int r = chooseCachedLocalRoute(
                            s, x, schedWork->banned,
                            visited, visitedWordsPerCustomer,
                            routeOwner, qLen, queueCap,
                            candRoutes, candPriority2, candVersion, candCount,
                            any, busy);
                        schedCtl[4] = r;
                        schedCtl[5] = any ? 1 : 0;
                        schedCtl[6] = busy ? 1 : 0;
#if DCVRP_PROFILE
                        if (profile) profAdd(&profile->candidateCacheSelections, 1);
#endif
                    }
                    schedulerBlocksBarrier(
                        schedBarrierCount, schedBarrierSense, schedulerBlocks,
                        &shSchedBarrierSense);
                } else {
                    // DEPOT mode remains exhaustive, but is now striped across
                    // all scheduler blocks instead of only block 0.
                    if (blockIdx.x == 0 && threadIdx.x == 0)
                        schedCtl[7] = schedulerWarps;
                    schedulerBlocksBarrier(
                        schedBarrierCount, schedBarrierSense, schedulerBlocks,
                        &shSchedBarrierSense);

                    schedulerDepotPartial(
                        s, x, schedWork->banned,
                        visited, visitedWordsPerCustomer,
                        schedulerWarps, routeOwner, qLen, queueCap,
                        shSchedBestPriority2, shSchedBestRoute,
                        shSchedChecks, shSchedFlags,
                        schedBlockTopP2, schedBlockTopR,
                        schedBlockChecks, schedBlockFlags);

                    schedulerBlocksBarrier(
                        schedBarrierCount, schedBarrierSense, schedulerBlocks,
                        &shSchedBarrierSense);

                    if (blockIdx.x == 0 && threadIdx.x == 0) {
                        double bestP = INF;
                        int bestR = -1;
                        unsigned long long checks = 0;
                        int flags = 0;
                        for (int b = 0; b < schedulerBlocks; ++b) {
                            const size_t base = (size_t)b * MAX_LOCAL_CANDIDATES;
                            const double p2 = schedBlockTopP2[base];
                            const int r = schedBlockTopR[base];
                            if (p2 < bestP ||
                                (p2 == bestP && r >= 0 && (bestR < 0 || r < bestR))) {
                                bestP = p2;
                                bestR = r;
                            }
                            checks += schedBlockChecks[b];
                            flags |= schedBlockFlags[b];
                        }
                        schedCtl[4] = bestR;
                        schedCtl[5] = (flags & 1) ? 1 : 0;
                        schedCtl[6] = (flags & 2) ? 1 : 0;
#if DCVRP_PROFILE
                        if (profile) {
                            profAdd(&profile->depotModeScans, 1);
                            profAdd(&profile->schedulerRouteChecks, checks);
                        }
#endif
                        __threadfence();
                    }
                    schedulerBlocksBarrier(
                        schedBarrierCount, schedBarrierSense, schedulerBlocks,
                        &shSchedBarrierSense);
                }

#if DCVRP_PROFILE
                if (blockIdx.x == 0 && threadIdx.x == 0 && profile) {
                    const unsigned long long dt = clock64() - scanStart;
                    profAdd(&profile->schedulerCycles, dt);
                    profAdd(&profile->schedulerScanCycles, dt);
                    profAdd(&profile->schedulerWarpCycles,
                            dt * (unsigned long long)schedCtl[7]);
                    profMax(&profile->maxActiveSchedulerWarps,
                            (unsigned long long)schedCtl[7]);
                    profAdd(&profile->schedulerScans, 1);
                    profAdd(&profile->schedulerFrontierItems, 1);
                }
#endif

                const int r = schedCtl[4];
                const bool anyUnvisited = schedCtl[5] != 0;
                const bool sawBusy = schedCtl[6] != 0;

                if (blockIdx.x == 0 && threadIdx.x == 0) {
                    if (r < 0) {
                        if (anyUnvisited && sawBusy) {
                            skipOnce = idx;
#if DCVRP_PROFILE
                            if (profile) profAdd(&profile->schedulerBlocked, 1);
#endif
                        } else {
                            if (s.dem[x] > s.capacity + EPS) {
                                atomicCAS(repairError, 0, 2);
                            } else if (createSingletonRouteSafe(s, x, routeOwner) < 0) {
                                atomicCAS(repairError, 0, 3);
                            } else {
                                atomicExch(&frontierReady[idx], 2);
                                atomicAdd(frontierPending, -1);
#if DCVRP_PROFILE
                                if (profile) profAdd(&profile->schedulerSingletons, 1);
#endif
                            }
                            skipOnce = -1;
                        }
                    } else {
                        // Reserve the chosen route only for the short queue
                        // insertion.  This is still a single producer, so the
                        // ready-route queue semantics are unchanged.
                        if (atomicCAS(&routeOwner[r], -1, -2) != -1) {
                            skipOnce = idx;
#if DCVRP_PROFILE
                            if (profile) profAdd(&profile->schedulerBlocked, 1);
#endif
                        } else {
#if DCVRP_PROFILE
                            const unsigned long long enqueueStart = clock64();
#endif
                            bool queued = false;
                            if (atomicCAS(&qLock[r], 0, 1) == 0) {
                                const int n = atomicLoadInt(&qLen[r]);
                                if (n < queueCap) {
                                    const int b = r * MAX_QUEUE_CAP;
                                    const double cx = s.cenX[r];
                                    const double cy = s.cenY[r];
                                    const double dx = s.xs[x] - cx;
                                    const double dy = s.ys[x] - cy;
                                    const double pr2 = dx * dx + dy * dy;
                                    int ins = n;
                                    for (int k = 0; k < n; ++k)
                                        if (pr2 < qPriority[b + k]) { ins = k; break; }
                                    for (int k = n; k > ins; --k) {
                                        qItems[b + k] = qItems[b + k - 1];
                                        qPriority[b + k] = qPriority[b + k - 1];
                                    }
                                    qItems[b + ins] = WorkItem{
                                        x, r, schedWork->depth, schedWork->banned};
                                    qPriority[b + ins] = pr2;
                                    atomicExch(&qLen[r], n + 1);
                                    const int qt = atomicAdd(qTotal, 1) + 1;
                                    queued = true;
#if DCVRP_PROFILE
                                    if (profile)
                                        profMax(&profile->queuePeak,
                                                (unsigned long long)qt);
#endif
                                }
                                __threadfence();
                                atomicExch(&qLock[r], 0);
                            }
                            __threadfence();
                            atomicExch(&routeOwner[r], -1);

                            if (queued) {
                                if (atomicCAS(&routeSignaled[r], 0, 1) == 0)
                                    signalReadyRoute(
                                        r, readyRoutes, readyRouteTail,
                                        readyRouteCap, repairError, profile);
                                atomicExch(&frontierReady[idx], 2);
                                atomicAdd(frontierPending, -1);
#if DCVRP_PROFILE
                                if (profile) profAdd(&profile->schedulerAssignments, 1);
#endif
                                skipOnce = -1;
                            } else {
                                skipOnce = idx;
#if DCVRP_PROFILE
                                if (profile) profAdd(&profile->schedulerBlocked, 1);
#endif
                            }
#if DCVRP_PROFILE
                            if (profile) {
                                const unsigned long long dt = clock64() - enqueueStart;
                                profAdd(&profile->schedulerCycles, dt);
                                profAdd(&profile->schedulerEnqueueCycles, dt);
                                profAdd(&profile->schedulerWarpCycles, dt);
                                profMax(&profile->maxActiveSchedulerWarps, 1ULL);
                            }
#endif
                        }
                    }

                    int nh = atomicLoadInt(frontierHead);
                    const int nt = atomicLoadInt(frontierTail);
                    while (nh < nt && atomicLoadInt(&frontierReady[nh]) == 2) ++nh;
                    atomicExch(frontierHead, nh);
                }

                schedulerBlocksBarrier(
                    schedBarrierCount, schedBarrierSense, schedulerBlocks,
                    &shSchedBarrierSense);
                continue;
            }

            // No pending frontier item.  Block 0 owns termination, exactly as
            // before; the other scheduler blocks merely synchronize with it.
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                if (atomicLoadInt(qTotal) == 0 && atomicLoadInt(activeWorkers) == 0) {
                    __threadfence();
                    if (atomicLoadInt(frontierPending) == 0 &&
                        atomicLoadInt(qTotal) == 0 &&
                        atomicLoadInt(activeWorkers) == 0) {
                        atomicExch(done, 1);
#if DCVRP_PROFILE
                        if (profile) profile->repairEnd = clock64();
#endif
                    }
                }
            }
            schedulerBlocksBarrier(
                schedBarrierCount, schedBarrierSense, schedulerBlocks,
                &shSchedBarrierSense);
            if (atomicLoadInt(done)) break;
        }
        return;
    }

    // ------------------------------------------------------------------
    // Worker blocks: no route scan. A scheduler enqueue signals the exact route
    // number through readyRoutes[]. An idle worker pops a token and claims it.
    // ------------------------------------------------------------------
    const int workerId=((int)blockIdx.x-schedulerBlocks)*warpsPerBlock+warpInBlock;
    if(workerId>=workerWarps) return;
    const int workerToken=workerId+1;
    int ownedRoute=-1;
    for(;;){
        if(ownedRoute<0){
            int claimed=-1;
            if(lane==0){
                int tokenRoute=-1;
                for(;;){
                    const int h=atomicLoadInt(readyRouteHead);
                    const int t=atomicLoadInt(readyRouteTail);
                    if(h>=t) break;
                    if(atomicCAS(readyRouteHead,h,h+1)==h){tokenRoute=readyRoutes[h];break;}
                }
                if(tokenRoute>=0){
                    // The scheduler may be in a tiny -2 reservation window to
                    // append another task to the same queue. Wait for that short
                    // transaction instead of discarding/re-scanning the token.
                    while(atomicCAS(&routeOwner[tokenRoute],-1,workerToken)!=-1){
                        if(atomicLoadInt(done)) break;
                    }
                    if(atomicLoadInt(&routeOwner[tokenRoute])==workerToken){
                        __threadfence();
                        if(atomicLoadInt(&qLen[tokenRoute])>0 && atomicLoadInt(&s.routeLen[tokenRoute])>0)
                            claimed=tokenRoute;
                        else {
                            atomicExch(&routeSignaled[tokenRoute],0);
                            atomicExch(&routeOwner[tokenRoute],-1);
                        }
                    }
                }
            }
            ownedRoute=shfl(claimed,0);
            if(ownedRoute<0){
                if(atomicLoadInt(done)) break;
                continue;
            }
        }

        const int r = ownedRoute;
        WorkItem task{-1,r,0,-1};
        int hasTask = 0;

        if (lane == 0) {
            while (atomicCAS(&qLock[r], 0, 1) != 0) {}
            const int n = qLen[r];
            if (n > 0) {
                // TOP ALWAYS WINS. No ballot/ffs is used for queue ordering.
                task = qItems[r * MAX_QUEUE_CAP];
                for (int k = 1; k < n; ++k) {
                    qItems[r * MAX_QUEUE_CAP + k - 1] = qItems[r * MAX_QUEUE_CAP + k];
                    qPriority[r * MAX_QUEUE_CAP + k - 1] = qPriority[r * MAX_QUEUE_CAP + k];
                }
                // Mark the worker active BEFORE making qTotal smaller. This
                // ordering prevents the scheduler from observing both counters
                // as zero in the tiny hand-off window.
                const int aw = atomicAdd(activeWorkers, 1) + 1;
#if DCVRP_PROFILE
                if (profile) profMax(&profile->maxActiveWorkers, (unsigned long long)aw);
#endif
                atomicExch(&qLen[r], n - 1);
                atomicAdd(qTotal, -1);
                hasTask = 1;
            }
            __threadfence();
            atomicExch(&qLock[r], 0);
        }
        __syncwarp();
        hasTask = shfl(hasTask, 0);
        task.cust  = shfl(task.cust, 0);
        task.depth = shfl(task.depth, 0);
        task.banned= shfl(task.banned, 0);
#if DCVRP_SCHED_DEBUG
        if (lane == 0 && dbgTake(dbgCounter, 420)) printf("[DBG worker-pop] gw=%d r=%d has=%d x=%d qTotal=%d activeW=%d\n", globalWarp, r, hasTask, task.cust, *qTotal, *activeWorkers);
#endif

        if (!hasTask) {
            __syncwarp();
            if (lane == 0) {
                // This worker drained the only signalled work token for r.
                // Clear the signal before releasing ownership so a future
                // scheduler enqueue can publish a fresh ready-route token.
                atomicExch(&routeSignaled[r], 0);
                __threadfence();
                atomicExch(&routeOwner[r], -1);
            }
            __syncwarp();
            ownedRoute = -1;
            if (atomicLoadInt(done)) break;
            continue;
        }

        const int x = task.cust;
        const int depth = task.depth;
if (x <= 0 || x >= s.maxCustomers) {
    if (lane == 0) {
        atomicCAS(repairError, 0, 5);
        atomicSub(activeWorkers, 1);
    }
    continue;
}

if (!s.active[x]) {
    // Customer was genuinely deleted; nothing to repair.
    if (lane == 0)
        atomicSub(activeWorkers, 1);
    continue;
}

if (s.routeLen[r] <= 0) {
    // Never silently lose an ACTIVE customer.
    if (lane == 0) {
        pushFrontier(
            WorkItem{x,-1,depth,r},
            frontier,
            frontierReady,
            frontierTail,
            frontierPending,
            frontierCap,
            repairError,
            profile);

        atomicSub(activeWorkers, 1);
    }
    continue;
}
#if DCVRP_PROFILE
        const unsigned long long workerStart = (lane == 0) ? clock64() : 0ULL;
        if (lane == 0 && profile) profMax(&profile->maxDepth, (unsigned long long)depth);
#endif

        // Successful entry into a route records that route as visited for
        // this event. If the task later fails, the customer returns to the
        // frontier and the scheduler tries another promising route.
        if (lane == 0) {
            markRouteVisited(visited, visitedWordsPerCustomer, x, r);
        }
        __syncwarp();

        const int L = s.routeLen[r];
        const int base = r * s.routeStride;
        const bool fits = s.routeLoad[r] + s.dem[x] <= s.capacity + EPS;

        WarpBest direct{INF,-1,-1,0};
        if (fits) {
            for (int p = lane; p <= L; p += 32) {
                const int prev = (p == 0) ? 0 : s.routeNodes[base+p-1];
                const int next = (p == L) ? 0 : s.routeNodes[base+p];
                const double d = edgeDelta(s, prev, x, next);
                if (d < direct.delta) direct={d,p,-1,1};
            }
            direct=warpReduceBest(direct);
        }

        WarpBest eject{INF,-1,-1,0};
        // The scheduler enforces the Opti-3 attempt policy: LOCAL mode is
        // bounded by its cached K routes (default 5), while DEPOT mode may
        // exhaust all visited routes.  The worker only evaluates the route it
        // was assigned; depth remains ejection-chain telemetry.
        if (!fits) {
            for (int yPos=lane; yPos<L; yPos+=32) {
                const int y=s.routeNodes[base+yPos];
                if (s.routeLoad[r]-s.dem[y]+s.dem[x]>s.capacity+EPS) continue;
                const int py=(yPos==0)?0:s.routeNodes[base+yPos-1];
                const int ny=(yPos+1==L)?0:s.routeNodes[base+yPos+1];
                const double rem=edgeDelta(s,py,y,ny);
                double ib=INF; int ip=-1;
                for (int p=0;p<L;++p) {
                    int prev,next;
                    reducedGap(s,r,yPos,p,prev,next);
                    const double d=edgeDelta(s,prev,x,next);
                    if(d<ib){ib=d;ip=p;}
                }
                const double net=ib-rem;
                if(net<eject.delta)eject={net,yPos,ip,2};
            }
            eject=warpReduceBest(eject);
        }

        int kind=0,pos=-1,yPos=-1; double delta=INF;
        if(lane==0){
            if(fits){kind=1;pos=direct.a;delta=direct.delta;}
            else if(eject.delta < -EPS){kind=2;yPos=eject.a;pos=eject.b;delta=eject.delta;}
        }
        kind=shfl(kind,0); pos=shfl(pos,0); yPos=shfl(yPos,0); delta=shfl(delta,0);

#if DCVRP_SCHED_DEBUG
        if (lane == 0 && dbgTake(dbgCounter, 500)) printf("[DBG worker-eval] gw=%d r=%d x=%d fits=%d kind=%d delta=%.3f qlen=%d\n", globalWarp, r, x, (int)fits, kind, delta, qLen[r]);
#endif

        if(kind==0){
            if(lane==0){
                WorkItem retry{x,-1,depth,r};
                pushFrontier(retry,frontier,frontierReady,frontierTail,frontierPending,frontierCap,repairError,profile);
#if DCVRP_PROFILE
                if (profile) {
                    profAdd(&profile->workerCycles, clock64() - workerStart);
                    profAdd(&profile->workerTasks, 1);
                    profAdd(&profile->workerRetry, 1);
                }
#endif
                atomicSub(activeWorkers,1);
            }
            continue;
        }

        int ejectedNode=-1;
        const int newL=(kind==1)?L+1:L;
        // Normal path: one routeStride scratch segment per warp in dynamic
        // shared memory.  If even a one-warp block would exceed the conservative
        // shared-memory budget, host code passes workerScratch and we fall back
        // to a private global-memory segment instead.
        extern __shared__ int sh[];
        int* tmp = workerScratch
            ? workerScratch + (size_t)workerId * s.routeStride
            : sh + (size_t)warpInBlock * s.routeStride;

        if(kind==1){
            for(int out=lane;out<newL;out+=32){
                if(out==pos) tmp[out]=x;
                else {const int src=(out<pos)?out:out-1;tmp[out]=s.routeNodes[base+src];}
            }
        } else {
            ejectedNode=(lane==0)?s.routeNodes[base+yPos]:-1;
            ejectedNode=shfl(ejectedNode,0);
            for(int out=lane;out<newL;out+=32){
                if(out==pos) tmp[out]=x;
                else {const int red=(out<pos)?out:out-1;const int src=(red<yPos)?red:red+1;tmp[out]=s.routeNodes[base+src];}
            }
        }
        __syncwarp();
        for(int p=lane;p<newL;p+=32)s.routeNodes[base+p]=tmp[p];
        if(lane==0){
            for(int p=newL;p<L;++p)s.routeNodes[base+p]=0;
            s.routeLen[r]=newL;
            double load=0.0;
            for(int p=0;p<newL;++p)load+=s.dem[s.routeNodes[base+p]];
            s.routeLoad[r]=load;
            if(ejectedNode>0){
                atomicExch(&s.routeOf[ejectedNode],-1);s.posOf[ejectedNode]=-1;
                markRouteVisited(visited, visitedWordsPerCustomer, ejectedNode, r);
            }
        }
        __syncwarp();
        for(int p=lane;p<newL;p+=32){
            const int v=s.routeNodes[base+p]; s.routeOf[v]=r;s.posOf[v]=p;
        }
        __syncwarp();
        if(lane==0) atomicExch(&s.routeOf[x],r);
        recomputeRouteCentroidWarp(s,r);
        __syncwarp();
        if(lane==0) atomicAdd(&s.routeVersion[r], 1u);
        __syncwarp();

        if(lane==0){
            // Worker owns the route, so it may safely resort its waiting queue
            // against the new centroid before releasing ownership.
            while(atomicCAS(&qLock[r],0,1)!=0){}
            sortQueueForCurrentCentroid(s,r,qLen,qItems,qPriority);
            __threadfence();
            atomicExch(&qLock[r],0);
        }
        __syncwarp();

        if(ejectedNode>0 && lane==0){
            const int ep=atomicAdd(ejectedCount,1);
            if(ep<s.maxCustomers)ejected[ep]=WorkItem{ejectedNode,-1,depth+1,r};
            pushFrontier(WorkItem{ejectedNode,-1,depth+1,r},frontier,frontierReady,frontierTail,frontierPending,frontierCap,repairError,profile);
        }
        if(lane==0) {
#if DCVRP_PROFILE
            if (profile) {
                profAdd(&profile->workerCycles, clock64() - workerStart);
                profAdd(&profile->workerTasks, 1);
                if (kind == 1) profAdd(&profile->workerDirect, 1);
                else if (kind == 2) profAdd(&profile->workerEject, 1);
            }
#endif
            atomicSub(activeWorkers,1);
#if DCVRP_SCHED_DEBUG
            if (dbgTake(dbgCounter, 560)) printf("[DBG worker-done] gw=%d r=%d x=%d ejected=%d qlen=%d qTotal=%d activeW=%d\n", globalWarp, r, x, ejectedNode, qLen[r], *qTotal, *activeWorkers);
#endif
        }
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------
// Route-pair refinement. Host supplies sparse nearest-centroid pairs split
// into conflict-free matching rounds, so pairs in one launch are non-conflicting.
// This kernel evaluates and commits the best Move8/Move9 for each pair.
// ---------------------------------------------------------------------------
__device__ __forceinline__ PairEval bestMove89(const GpuStateView& s, int R, int S) {
    const int lane = threadIdx.x & 31;
    const int L1 = s.routeLen[R], L2 = s.routeLen[S];
    const int b1 = R * s.routeStride, b2 = S * s.routeStride;
    WarpPair best{0.0, 0, -1, -1, -1, -1};

    const long long cuts = (long long)(L1 + 1) * (L2 + 1);
    for (long long t = lane; t < cuts; t += 32) {
        const int i = (int)(t / (L2 + 1));
        const int j = (int)(t % (L2 + 1));
        if (i == L1 && j == L2) continue;

        double pref1 = 0.0, pref2 = 0.0;
        for (int k = 0; k < i; ++k) pref1 += s.dem[s.routeNodes[b1 + k]];
        for (int k = 0; k < j; ++k) pref2 += s.dem[s.routeNodes[b2 + k]];
        const double suf1 = s.routeLoad[R] - pref1;
        const double suf2 = s.routeLoad[S] - pref2;
        if (pref1 + suf2 > s.capacity + EPS || pref2 + suf1 > s.capacity + EPS) continue;

        const int prevA = (i == 0) ? 0 : s.routeNodes[b1 + i - 1];
        const int firstA = (i == L1) ? 0 : s.routeNodes[b1 + i];
        const int prevB = (j == 0) ? 0 : s.routeNodes[b2 + j - 1];
        const int firstB = (j == L2) ? 0 : s.routeNodes[b2 + j];

        const double oldBoundary = dist(s, prevA, firstA) + dist(s, prevB, firstB);

        // Move 9: exchange tails without reversing them.
        const double new9 = (j < L2 ? dist(s, prevA, firstB) : dist(s, prevA, 0))
                          + (i < L1 ? dist(s, prevB, firstA) : dist(s, prevB, 0));
        const double d9 = new9 - oldBoundary;
        if (d9 < best.delta) best = {d9, 9, i, j, -1, -1};

        // Move 8: exchange tails and reverse both tails.
        const int lastB = (j < L2) ? s.routeNodes[b2 + L2 - 1] : 0;
        const int firstB0 = (j < L2) ? s.routeNodes[b2 + j] : 0;
        const int lastA = (i < L1) ? s.routeNodes[b1 + L1 - 1] : 0;
        const int firstA0 = (i < L1) ? s.routeNodes[b1 + i] : 0;
        const double new8 = (j < L2 ? dist(s, prevA, lastB) + dist(s, firstB0, 0) : dist(s, prevA, 0))
                          + (i < L1 ? dist(s, prevB, lastA) + dist(s, firstA0, 0) : dist(s, prevB, 0));
        const double d8 = new8 - oldBoundary;
        if (d8 < best.delta) best = {d8, 8, i, j, -1, -1};
    }

    best = warpReducePair(best);
    PairEval out{best.delta, best.kind, best.i, best.j, best.auxI, best.auxJ};
    return out;
}

__device__ __forceinline__ PairEval bestMove1SwapStar(const GpuStateView& s, int R, int S) {
    const int lane = threadIdx.x & 31;
    const int L1 = s.routeLen[R], L2 = s.routeLen[S];
    const int b1 = R * s.routeStride, b2 = S * s.routeStride;
    WarpPair best{0.0, 0, -1, -1, -1, -1};

    // Move 1: relocate a node R -> S.
    for (int iu = lane; iu < L1; iu += 32) {
        const int u = s.routeNodes[b1 + iu];
        if (L2 >= s.routeStride) continue;
        if (s.routeLoad[S] + s.dem[u] > s.capacity + EPS) continue;
        const int up = (iu == 0) ? 0 : s.routeNodes[b1 + iu - 1];
        const int un = (iu + 1 == L1) ? 0 : s.routeNodes[b1 + iu + 1];
        const double remove = -edgeDelta(s, up, u, un);
        for (int p = 0; p <= L2; ++p) {
            const int prev = (p == 0) ? 0 : s.routeNodes[b2 + p - 1];
            const int next = (p == L2) ? 0 : s.routeNodes[b2 + p];
            const double ins = edgeDelta(s, prev, u, next);
            const double d = remove + ins;
            if (d < best.delta) best = {d, 1, iu, p, 0, -1};
        }
    }

    // Move 1: relocate S -> R.
    for (int jv = lane; jv < L2; jv += 32) {
        const int v = s.routeNodes[b2 + jv];
        if (L1 >= s.routeStride) continue;
        if (s.routeLoad[R] + s.dem[v] > s.capacity + EPS) continue;
        const int vp = (jv == 0) ? 0 : s.routeNodes[b2 + jv - 1];
        const int vn = (jv + 1 == L2) ? 0 : s.routeNodes[b2 + jv + 1];
        const double remove = -edgeDelta(s, vp, v, vn);
        for (int p = 0; p <= L1; ++p) {
            const int prev = (p == 0) ? 0 : s.routeNodes[b1 + p - 1];
            const int next = (p == L1) ? 0 : s.routeNodes[b1 + p];
            const double ins = edgeDelta(s, prev, v, next);
            const double d = remove + ins;
            if (d < best.delta) best = {d, 1, jv, p, 1, -1};
        }
    }

    // SWAP*: exchange one customer each way, with best insertion in each
    // reduced route.
    const long long pairs = (long long)L1 * L2;
    for (long long t = lane; t < pairs; t += 32) {
        const int iu = (int)(t / L2), jv = (int)(t % L2);
        const int u = s.routeNodes[b1 + iu], v = s.routeNodes[b2 + jv];
        if (s.routeLoad[R] - s.dem[u] + s.dem[v] > s.capacity + EPS) continue;
        if (s.routeLoad[S] - s.dem[v] + s.dem[u] > s.capacity + EPS) continue;

        const int up = (iu == 0) ? 0 : s.routeNodes[b1 + iu - 1];
        const int un = (iu + 1 == L1) ? 0 : s.routeNodes[b1 + iu + 1];
        const int vp = (jv == 0) ? 0 : s.routeNodes[b2 + jv - 1];
        const int vn = (jv + 1 == L2) ? 0 : s.routeNodes[b2 + jv + 1];
        const double remU = edgeDelta(s, up, u, un);
        const double remV = edgeDelta(s, vp, v, vn);

        double bestU = INF, bestV = INF;
        int posV = -1, posU = -1;
        for (int p = 0; p < L1; ++p) {
            int prev, next;
            reducedGap(s, R, iu, p, prev, next);
            const double d = edgeDelta(s, prev, v, next);
            if (d < bestV) { bestV = d; posV = p; }
        }
        for (int p = 0; p < L2; ++p) {
            int prev, next;
            reducedGap(s, S, jv, p, prev, next);
            const double d = edgeDelta(s, prev, u, next);
            if (d < bestU) { bestU = d; posU = p; }
        }
        const double d = bestU + bestV - remU - remV;
        if (d < best.delta) best = {d, 10, iu, jv, posV, posU};
    }

    best = warpReducePair(best);
    return PairEval{best.delta, best.kind, best.i, best.j, best.auxI, best.auxJ};
}

__device__ __forceinline__ void commitPairEval(const GpuStateView& s,
                                               int R, int S, const PairEval& c,
                                               int* shared) {
    const int lane = threadIdx.x & 31;
    const int warpLocal = threadIdx.x >> 5;
    int* A = shared + (size_t)warpLocal * 2 * s.routeStride;
    int* B = A + s.routeStride;
    const int bA = R * s.routeStride, bB = S * s.routeStride;
    const int L1 = s.routeLen[R], L2 = s.routeLen[S];
    int nA = L1, nB = L2;

    if (c.kind == 8 || c.kind == 9) {
        const int i = c.i, j = c.j;
        nA = i + (L2 - j);
        nB = j + (L1 - i);
        for (int p = lane; p < nA; p += 32) {
            if (p < i) {
                A[p] = s.routeNodes[bA + p];
            } else {
                const int off = p - i;
                const int src = (c.kind == 8) ? (L2 - 1 - off) : (j + off);
                A[p] = s.routeNodes[bB + src];
            }
        }
        for (int p = lane; p < nB; p += 32) {
            if (p < j) {
                B[p] = s.routeNodes[bB + p];
            } else {
                const int off = p - j;
                const int src = (c.kind == 8) ? (L1 - 1 - off) : (i + off);
                B[p] = s.routeNodes[bA + src];
            }
        }
    } else if (c.kind == 1) {
        const int dir = c.auxI;
        if (dir == 0) { // R -> S
            const int iu = c.i, p = c.j;
            const int u = s.routeNodes[bA + iu];
            nA = L1 - 1; nB = L2 + 1;
            for (int q = lane; q < nA; q += 32) {
                const int src = (q < iu) ? q : q + 1;
                A[q] = s.routeNodes[bA + src];
            }
            for (int q = lane; q < nB; q += 32) {
                if (q == p) B[q] = u;
                else {
                    const int src = (q < p) ? q : q - 1;
                    B[q] = s.routeNodes[bB + src];
                }
            }
        } else { // S -> R
            const int jv = c.i, p = c.j;
            const int v = s.routeNodes[bB + jv];
            nA = L1 + 1; nB = L2 - 1;
            for (int q = lane; q < nB; q += 32) {
                const int src = (q < jv) ? q : q + 1;
                B[q] = s.routeNodes[bB + src];
            }
            for (int q = lane; q < nA; q += 32) {
                if (q == p) A[q] = v;
                else {
                    const int src = (q < p) ? q : q - 1;
                    A[q] = s.routeNodes[bA + src];
                }
            }
        }
    } else if (c.kind == 10) { // SWAP*
        const int iu = c.i, jv = c.j, posV = c.auxI, posU = c.auxJ;
        const int u = s.routeNodes[bA + iu], v = s.routeNodes[bB + jv];
        nA = L1; nB = L2;
        for (int p = lane; p < nA; p += 32) {
            if (p == posV) A[p] = v;
            else {
                const int red = (p < posV) ? p : p - 1;
                const int src = (red < iu) ? red : red + 1;
                A[p] = s.routeNodes[bA + src];
            }
        }
        for (int p = lane; p < nB; p += 32) {
            if (p == posU) B[p] = u;
            else {
                const int red = (p < posU) ? p : p - 1;
                const int src = (red < jv) ? red : red + 1;
                B[p] = s.routeNodes[bB + src];
            }
        }
    }

    __syncwarp();
    for (int p = lane; p < nA; p += 32) s.routeNodes[bA + p] = A[p];
    for (int p = lane; p < nB; p += 32) s.routeNodes[bB + p] = B[p];
    __syncwarp();

    if (lane == 0) {
        for (int p = nA; p < L1; ++p) s.routeNodes[bA + p] = 0;
        for (int p = nB; p < L2; ++p) s.routeNodes[bB + p] = 0;
        s.routeLen[R] = nA;
        s.routeLen[S] = nB;
        double la = 0.0, lb = 0.0;
        for (int p = 0; p < nA; ++p) la += s.dem[s.routeNodes[bA + p]];
        for (int p = 0; p < nB; ++p) lb += s.dem[s.routeNodes[bB + p]];
        s.routeLoad[R] = la;
        s.routeLoad[S] = lb;
    }
    __syncwarp();
    recomputeRouteCentroidWarp(s, R);
    recomputeRouteCentroidWarp(s, S);
    __syncwarp();
    if (lane == 0) {
        atomicAdd(&s.routeVersion[R], 1u);
        atomicAdd(&s.routeVersion[S], 1u);
    }
    __syncwarp();
    for (int p = lane; p < nA; p += 32) {
        int v = s.routeNodes[bA + p];
        s.routeOf[v] = R;
        s.posOf[v] = p;
    }
    for (int p = lane; p < nB; p += 32) {
        int v = s.routeNodes[bB + p];
        s.routeOf[v] = S;
        s.posOf[v] = p;
    }
}

__global__ void pairMove89Kernel(GpuStateView s, const int2* pairs, int npairs, int* changed) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= npairs) return;
    const int R = pairs[warp].x, S = pairs[warp].y;
    if (s.routeLen[R] <= 0 || s.routeLen[S] <= 0) return;
    const PairEval c = bestMove89(s, R, S);
    const double d = shfl(c.delta, 0);
    if (d >= -EPS) return;
    const PairEval chosen{
        d,
        shfl(c.kind, 0), shfl(c.i, 0), shfl(c.j, 0),
        shfl(c.auxI, 0), shfl(c.auxJ, 0)
    };
    extern __shared__ int sh[];
    commitPairEval(s, R, S, chosen, sh);
    if (lane == 0) atomicAdd(changed, 1);
}

__global__ void pairMove1SwapKernel(GpuStateView s, const int2* pairs, int npairs, int* changed) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= npairs) return;
    const int R = pairs[warp].x, S = pairs[warp].y;
    if (s.routeLen[R] <= 0 || s.routeLen[S] <= 0) return;
    const PairEval c = bestMove1SwapStar(s, R, S);
    const double d = shfl(c.delta, 0);
    if (d >= -EPS) return;
    const PairEval chosen{
        d,
        shfl(c.kind, 0), shfl(c.i, 0), shfl(c.j, 0),
        shfl(c.auxI, 0), shfl(c.auxJ, 0)
    };
    extern __shared__ int sh[];
    commitPairEval(s, R, S, chosen, sh);
    if (lane == 0) atomicAdd(changed, 1);
}

// ---------------------------------------------------------------------------
// Final Move 7: one warp per route, one best intra-route 2-opt move.
// ---------------------------------------------------------------------------
__global__ void twoOptOnceKernel(GpuStateView s, int* changed) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int m = atomicLoadInt(s.routeCount);
    if (warp >= m) return;
    const int r = warp;
    const int L = s.routeLen[r];
    if (L < 3) return;
    const int base = r * s.routeStride;
    WarpBest best{0.0, -1, -1, 0};
    const long long total = (long long)L * (L - 1) / 2;
    for (long long t = lane; t < total; t += 32) {
        long long rem = t;
        int i = 0;
        while (rem >= L - i - 1) {
            rem -= (L - i - 1);
            ++i;
        }
        const int k = i + 1 + (int)rem;
        const int a = (i == 0) ? 0 : s.routeNodes[base + i - 1];
        const int b = s.routeNodes[base + i];
        const int c = s.routeNodes[base + k];
        const int d = (k + 1 == L) ? 0 : s.routeNodes[base + k + 1];
        const double delta = dist(s, a, c) + dist(s, b, d)
                           - dist(s, a, b) - dist(s, c, d);
        if (delta < best.delta) best = {delta, i, k, 7};
    }
    best = warpReduceBest(best);
    const double d = shfl(best.delta, 0);
    const int i = shfl(best.a, 0), k = shfl(best.b, 0);
    if (d >= -EPS || i < 0) return;

    for (int p = lane; p <= (k - i) / 2; p += 32) {
        const int x = i + p, y = k - p;
        const int vx = s.routeNodes[base + x], vy = s.routeNodes[base + y];
        s.routeNodes[base + x] = vy;
        s.routeNodes[base + y] = vx;
    }
    __syncwarp();
    for (int p = lane; p < L; p += 32) {
        const int v = s.routeNodes[base + p];
        s.routeOf[v] = r;
        s.posOf[v] = p;
    }
    __syncwarp();
    recomputeRouteCentroidWarp(s, r);
    __syncwarp();
    if (lane == 0) {
        atomicAdd(&s.routeVersion[r], 1u);
        atomicAdd(changed, 1);
    }
}

__global__ void activateCustomersKernel(GpuStateView s,const int*ids,const double*xs,
                                        const double*ys,const double*dem,int n){
    const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    const int v=ids[i]; s.xs[v]=xs[i];s.ys[v]=ys[i];s.dem[v]=dem[i];
    s.active[v]=1;s.routeOf[v]=-1;s.posOf[v]=-1;
}

// ---------------------------------------------------------------------------
// Host-side refinement telemetry. Times include CUDA launch/synchronization
// overhead for the corresponding stage; move counts are committed moves.
// ---------------------------------------------------------------------------
struct RefineProfileReport {
    double move89Us = 0.0;
    double move1SwapUs = 0.0;
    double move7Us = 0.0;
    unsigned long long move89Committed = 0;
    unsigned long long move1SwapCommitted = 0;
    unsigned long long move7Committed = 0;
    unsigned long long pairRounds = 0;
    unsigned long long pairCandidates89 = 0;
    unsigned long long pairCandidates1Swap = 0;
    int nearestPartnerK = 5;
    int passesRequested = 0;
    int passesRun = 0;
};

// ---------------------------------------------------------------------------
// Host-side device engine.
// ---------------------------------------------------------------------------
struct GpuDynamicEngine {
    const Instance* I = nullptr;
    int maxCustomers = 0, maxRoutes = 0, routeStride = 0;
    int frontierCap = 0, queueCap = 5;
    int visitedWordsPerCustomer = 0;

    double *d_x=nullptr,*d_y=nullptr,*d_dem=nullptr,*d_load=nullptr,*d_cenX=nullptr,*d_cenY=nullptr;
    int *d_nodes=nullptr,*d_len=nullptr,*d_active=nullptr,*d_routeOf=nullptr,*d_posOf=nullptr,*d_routeCount=nullptr;
    uint32_t *d_ver=nullptr,*d_cenVer=nullptr;

    int *d_qLen=nullptr,*d_qLock=nullptr,*d_qTotal=nullptr,*d_routeOwner=nullptr,*d_routeProbe=nullptr,*d_dbgCounter=nullptr;
    WorkItem *d_qItems=nullptr,*d_frontier=nullptr,*d_ejected=nullptr;
    double *d_qPriority=nullptr;
    int *d_candRoutes=nullptr,*d_candCount=nullptr,*d_candMode=nullptr;
    double *d_candPriority2=nullptr;
    uint32_t *d_candVersion=nullptr;
    int *d_readyRoutes=nullptr,*d_readyRouteHead=nullptr,*d_readyRouteTail=nullptr,*d_routeSignaled=nullptr;
    int *d_frontierReady=nullptr,*d_frontierHead=nullptr,*d_frontierTail=nullptr,*d_frontierPending=nullptr,*d_repairError=nullptr;
    int *d_ejectedCount=nullptr,*d_activeWorkers=nullptr,*d_done=nullptr;
    RepairProfileCounters *d_profile=nullptr;
    RepairProfileReport lastProfile{};
    RefineProfileReport lastRefine{};
    uint32_t *d_visited=nullptr;

    // Multi-block scheduler coordination + global route-worker scratch.
    int *d_workerScratch=nullptr;
    int *d_schedCtl=nullptr,*d_schedBarrierCount=nullptr,*d_schedBarrierSense=nullptr;
    WorkItem *d_schedWork=nullptr;
    double *d_schedBlockTopP2=nullptr;
    int *d_schedBlockTopR=nullptr,*d_schedBlockFlags=nullptr;
    unsigned long long *d_schedBlockChecks=nullptr;
    int maxSchedulerBlocks=1;

    int2 *d_pairs=nullptr; int *d_changed=nullptr;

    void freePtr(void* p){ if(p) CUDA_OK(cudaFree(p)); }
    ~GpuDynamicEngine(){ release(); }

    void release(){
        freePtr(d_x); freePtr(d_y); freePtr(d_dem); freePtr(d_load); freePtr(d_cenX); freePtr(d_cenY);
        freePtr(d_nodes); freePtr(d_len); freePtr(d_active); freePtr(d_routeOf); freePtr(d_posOf); freePtr(d_routeCount); freePtr(d_ver); freePtr(d_cenVer);
        freePtr(d_qLen); freePtr(d_qLock); freePtr(d_qTotal); freePtr(d_routeOwner); freePtr(d_routeProbe); freePtr(d_qItems); freePtr(d_qPriority); freePtr(d_dbgCounter);
        freePtr(d_candRoutes); freePtr(d_candCount); freePtr(d_candMode); freePtr(d_candPriority2); freePtr(d_candVersion);
        freePtr(d_readyRoutes); freePtr(d_readyRouteHead); freePtr(d_readyRouteTail); freePtr(d_routeSignaled);
        freePtr(d_frontier); freePtr(d_frontierReady); freePtr(d_frontierHead); freePtr(d_frontierTail); freePtr(d_frontierPending); freePtr(d_repairError);
        freePtr(d_ejected); freePtr(d_ejectedCount); freePtr(d_activeWorkers);
        freePtr(d_visited); freePtr(d_workerScratch);
        freePtr(d_schedCtl); freePtr(d_schedWork); freePtr(d_schedBarrierCount); freePtr(d_schedBarrierSense);
        freePtr(d_schedBlockTopP2); freePtr(d_schedBlockTopR); freePtr(d_schedBlockChecks); freePtr(d_schedBlockFlags);
        freePtr(d_pairs); freePtr(d_changed); freePtr(d_done); freePtr(d_profile);
        d_x=d_y=d_dem=d_load=d_cenX=d_cenY=nullptr; d_nodes=d_len=d_active=d_routeOf=d_posOf=d_routeCount=nullptr; d_ver=d_cenVer=nullptr;
        d_qLen=d_qLock=d_qTotal=d_routeOwner=d_routeProbe=d_dbgCounter=nullptr; d_qItems=nullptr; d_qPriority=nullptr; d_frontier=nullptr;
        d_candRoutes=d_candCount=d_candMode=nullptr; d_candPriority2=nullptr; d_candVersion=nullptr;
        d_readyRoutes=d_readyRouteHead=d_readyRouteTail=d_routeSignaled=nullptr;
        d_frontierReady=d_frontierHead=d_frontierTail=d_frontierPending=d_repairError=nullptr;
        d_ejected=nullptr; d_ejectedCount=d_activeWorkers=d_done=nullptr; d_visited=nullptr;
        d_workerScratch=nullptr;
        d_schedCtl=d_schedBarrierCount=d_schedBarrierSense=nullptr; d_schedWork=nullptr;
        d_schedBlockTopP2=nullptr; d_schedBlockTopR=d_schedBlockFlags=nullptr; d_schedBlockChecks=nullptr;
        maxSchedulerBlocks=1;
        d_pairs=nullptr; d_changed=nullptr; d_profile=nullptr; lastProfile={}; lastRefine={};
    }

    GpuStateView view() const {
        return {d_x,d_y,d_dem,d_nodes,d_len,d_load,d_cenX,d_cenY,d_ver,d_cenVer,d_routeOf,d_posOf,
                d_active,d_routeCount,routeStride,maxRoutes,maxCustomers,I->capacity};
    }

    int routes() const {
        int m=0; CUDA_OK(cudaMemcpy(&m,d_routeCount,sizeof(int),cudaMemcpyDeviceToHost)); return m;
    }

    void init(const Instance& inst, const RouteStore& S, int maxCust, int maxRt,
              int stride, int fcap, int qcap) {
        release();
        I=&inst; maxCustomers=maxCust; maxRoutes=maxRt; routeStride=stride;
        frontierCap=std::max(1024,fcap);
        queueCap=std::max(1,std::min(MAX_QUEUE_CAP,qcap));
        visitedWordsPerCustomer=(maxRoutes+31)/32;
        // Allocate scheduler partials for the worst case of ONE scheduler warp
        // per block. Runtime normally uses 8/4/2 warps per block.
        const int routesPerSchedulerWarp = SCHED_ROUTES_PER_THREAD_TARGET * 32;
        maxSchedulerBlocks = std::max(1,
            (maxRoutes + routesPerSchedulerWarp - 1) / routesPerSchedulerWarp);

        CUDA_OK(cudaMalloc(&d_x,maxCustomers*sizeof(double)));
        CUDA_OK(cudaMalloc(&d_y,maxCustomers*sizeof(double)));
        CUDA_OK(cudaMalloc(&d_dem,maxCustomers*sizeof(double)));
        CUDA_OK(cudaMalloc(&d_load,maxRoutes*sizeof(double)));
        CUDA_OK(cudaMalloc(&d_cenX,maxRoutes*sizeof(double)));
        CUDA_OK(cudaMalloc(&d_cenY,maxRoutes*sizeof(double)));
        CUDA_OK(cudaMalloc(&d_nodes,(size_t)maxRoutes*routeStride*sizeof(int)));
        CUDA_OK(cudaMalloc(&d_len,maxRoutes*sizeof(int)));
        CUDA_OK(cudaMalloc(&d_active,maxCustomers*sizeof(int)));
        CUDA_OK(cudaMalloc(&d_routeOf,maxCustomers*sizeof(int)));
        CUDA_OK(cudaMalloc(&d_posOf,maxCustomers*sizeof(int)));
        CUDA_OK(cudaMalloc(&d_routeCount,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_ver,maxRoutes*sizeof(uint32_t)));
        CUDA_OK(cudaMalloc(&d_cenVer,maxRoutes*sizeof(uint32_t)));

        CUDA_OK(cudaMalloc(&d_qLen,maxRoutes*sizeof(int)));
        CUDA_OK(cudaMalloc(&d_qLock,maxRoutes*sizeof(int)));
        CUDA_OK(cudaMalloc(&d_qTotal,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_routeOwner,maxRoutes*sizeof(int)));
        CUDA_OK(cudaMalloc(&d_routeProbe,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_dbgCounter,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_qItems,(size_t)maxRoutes*MAX_QUEUE_CAP*sizeof(WorkItem)));
        CUDA_OK(cudaMalloc(&d_qPriority,(size_t)maxRoutes*MAX_QUEUE_CAP*sizeof(double)));

        CUDA_OK(cudaMalloc(&d_candRoutes,(size_t)maxCustomers*MAX_LOCAL_CANDIDATES*sizeof(int)));
        CUDA_OK(cudaMalloc(&d_candPriority2,(size_t)maxCustomers*MAX_LOCAL_CANDIDATES*sizeof(double)));
        CUDA_OK(cudaMalloc(&d_candVersion,(size_t)maxCustomers*MAX_LOCAL_CANDIDATES*sizeof(uint32_t)));
        CUDA_OK(cudaMalloc(&d_candCount,maxCustomers*sizeof(int)));
        CUDA_OK(cudaMalloc(&d_candMode,maxCustomers*sizeof(int)));

        CUDA_OK(cudaMalloc(&d_readyRoutes,(size_t)frontierCap*sizeof(int)));
        CUDA_OK(cudaMalloc(&d_readyRouteHead,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_readyRouteTail,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_routeSignaled,maxRoutes*sizeof(int)));

        CUDA_OK(cudaMalloc(&d_frontier,(size_t)frontierCap*sizeof(WorkItem)));
        CUDA_OK(cudaMalloc(&d_frontierReady,(size_t)frontierCap*sizeof(int)));
        CUDA_OK(cudaMalloc(&d_frontierHead,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_frontierTail,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_frontierPending,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_repairError,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_ejected,maxCustomers*sizeof(WorkItem)));
        CUDA_OK(cudaMalloc(&d_ejectedCount,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_activeWorkers,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_done,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_profile,sizeof(RepairProfileCounters)));
        CUDA_OK(cudaMalloc(&d_visited,
                           (size_t)maxCustomers*visitedWordsPerCustomer*sizeof(uint32_t)));

        // Fallback route scratch: used only when routeStride is too large for
        // even a one-warp shared-memory block.
        CUDA_OK(cudaMalloc(&d_workerScratch,
                           (size_t)MAX_ROUTE_WORKER_WARPS*routeStride*sizeof(int)));

        // Multi-block scheduler control and one partial summary per scheduler
        // block.  These allocations are tiny compared with route storage.
        CUDA_OK(cudaMalloc(&d_schedCtl,16*sizeof(int)));
        CUDA_OK(cudaMalloc(&d_schedWork,sizeof(WorkItem)));
        CUDA_OK(cudaMalloc(&d_schedBarrierCount,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_schedBarrierSense,sizeof(int)));
        CUDA_OK(cudaMalloc(&d_schedBlockTopP2,
                           (size_t)maxSchedulerBlocks*MAX_LOCAL_CANDIDATES*sizeof(double)));
        CUDA_OK(cudaMalloc(&d_schedBlockTopR,
                           (size_t)maxSchedulerBlocks*MAX_LOCAL_CANDIDATES*sizeof(int)));
        CUDA_OK(cudaMalloc(&d_schedBlockChecks,
                           (size_t)maxSchedulerBlocks*sizeof(unsigned long long)));
        CUDA_OK(cudaMalloc(&d_schedBlockFlags,
                           (size_t)maxSchedulerBlocks*sizeof(int)));

        // d_pairs only stores ONE conflict-free round at a time (<= maxRoutes/2
        // pairs), not the whole O(R^2) complete graph.
        const int maxPairsPerRound=std::max(1,(maxRt+1)/2);
        CUDA_OK(cudaMalloc(&d_pairs,(size_t)maxPairsPerRound*sizeof(int2)));
        CUDA_OK(cudaMalloc(&d_changed,sizeof(int)));
        uploadHost(S);
    }

    void clearQueues(){
        int z=0;
        CUDA_OK(cudaMemset(d_qLen,0,maxRoutes*sizeof(int)));
        CUDA_OK(cudaMemset(d_qLock,0,maxRoutes*sizeof(int)));
        { std::vector<int> freeOwner(maxRoutes,-1); CUDA_OK(cudaMemcpy(d_routeOwner,freeOwner.data(),maxRoutes*sizeof(int),cudaMemcpyHostToDevice)); }
        CUDA_OK(cudaMemset(d_routeProbe,0,sizeof(int)));
        CUDA_OK(cudaMemset(d_routeSignaled,0,maxRoutes*sizeof(int)));
        CUDA_OK(cudaMemcpy(d_readyRouteHead,&z,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_readyRouteTail,&z,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_qTotal,&z,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(d_frontierReady,0,(size_t)frontierCap*sizeof(int)));
        CUDA_OK(cudaMemcpy(d_frontierHead,&z,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_frontierTail,&z,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_frontierPending,&z,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_repairError,&z,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_ejectedCount,&z,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_activeWorkers,&z,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_done,&z,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(d_schedCtl,0,16*sizeof(int)));
        CUDA_OK(cudaMemcpy(d_schedBarrierCount,&z,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_schedBarrierSense,&z,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(d_schedBlockTopP2,0,
                           (size_t)maxSchedulerBlocks*MAX_LOCAL_CANDIDATES*sizeof(double)));
        CUDA_OK(cudaMemset(d_schedBlockTopR,0xff,
                           (size_t)maxSchedulerBlocks*MAX_LOCAL_CANDIDATES*sizeof(int)));
        CUDA_OK(cudaMemset(d_schedBlockChecks,0,
                           (size_t)maxSchedulerBlocks*sizeof(unsigned long long)));
        CUDA_OK(cudaMemset(d_schedBlockFlags,0,
                           (size_t)maxSchedulerBlocks*sizeof(int)));
    }

    void uploadHost(const RouteStore& S){
        std::vector<double> x(maxCustomers,0),y(maxCustomers,0),dem(maxCustomers,0),load(maxRoutes,0),cx(maxRoutes,0),cy(maxRoutes,0);
        std::vector<int> active(maxCustomers,0),ro(maxCustomers,-1),po(maxCustomers,-1),len(maxRoutes,0);
        std::vector<uint32_t> ver(maxRoutes,0);
        std::vector<int> nodes((size_t)maxRoutes*routeStride,0);
        for(int v=0;v<(int)I->xs.size()&&v<maxCustomers;++v){
            x[v]=I->xs[v]; y[v]=I->ys[v]; dem[v]=I->demand[v];
            active[v]=(v<(int)S.activeCust.size()?S.activeCust[v]:0);
        }
        int m=std::min((int)S.meta.size(),maxRoutes);
        for(int r=0;r<m;++r){
            len[r]=S.meta[r].length; load[r]=S.meta[r].load; cx[r]=S.meta[r].cenX; cy[r]=S.meta[r].cenY; ver[r]=S.meta[r].version;
            if(len[r]>routeStride) std::abort();
            for(int k=0;k<len[r];++k){
                int v=S.seg[S.meta[r].offset+k]; nodes[(size_t)r*routeStride+k]=v; ro[v]=r; po[v]=k;
            }
        }
        CUDA_OK(cudaMemcpy(d_x,x.data(),maxCustomers*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_y,y.data(),maxCustomers*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_dem,dem.data(),maxCustomers*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_active,active.data(),maxCustomers*sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_routeOf,ro.data(),maxCustomers*sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_posOf,po.data(),maxCustomers*sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_nodes,nodes.data(),nodes.size()*sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_len,len.data(),maxRoutes*sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_load,load.data(),maxRoutes*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_cenX,cx.data(),maxRoutes*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_cenY,cy.data(),maxRoutes*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_ver,ver.data(),maxRoutes*sizeof(uint32_t),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(d_cenVer,0,maxRoutes*sizeof(uint32_t)));
        CUDA_OK(cudaMemcpy(d_routeCount,&m,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(d_visited,0,
                           (size_t)maxCustomers*visitedWordsPerCustomer*sizeof(uint32_t)));
        CUDA_OK(cudaMemset(d_qPriority,0,(size_t)maxRoutes*MAX_QUEUE_CAP*sizeof(double)));
        CUDA_OK(cudaMemset(d_candRoutes,0xff,(size_t)maxCustomers*MAX_LOCAL_CANDIDATES*sizeof(int)));
        CUDA_OK(cudaMemset(d_candPriority2,0,(size_t)maxCustomers*MAX_LOCAL_CANDIDATES*sizeof(double)));
        CUDA_OK(cudaMemset(d_candVersion,0,(size_t)maxCustomers*MAX_LOCAL_CANDIDATES*sizeof(uint32_t)));
        CUDA_OK(cudaMemset(d_candCount,0,maxCustomers*sizeof(int)));
        CUDA_OK(cudaMemset(d_candMode,0,maxCustomers*sizeof(int)));
        clearQueues();
    }

    void activate(const std::vector<int>& ids){
        if(ids.empty()) return;
        std::vector<double>x(ids.size()),y(ids.size()),q(ids.size());
        for(size_t i=0;i<ids.size();++i){int v=ids[i];x[i]=I->xs[v];y[i]=I->ys[v];q[i]=I->demand[v];}
        int* di=nullptr; double *dx=nullptr,*dy=nullptr,*dq=nullptr;
        const size_t n=ids.size();
        CUDA_OK(cudaMalloc(&di,n*sizeof(int))); CUDA_OK(cudaMalloc(&dx,n*sizeof(double))); CUDA_OK(cudaMalloc(&dy,n*sizeof(double))); CUDA_OK(cudaMalloc(&dq,n*sizeof(double)));
        CUDA_OK(cudaMemcpy(di,ids.data(),n*sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dx,x.data(),n*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dy,y.data(),n*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dq,q.data(),n*sizeof(double),cudaMemcpyHostToDevice));
        activateCustomersKernel<<<(int(n)+127)/128,128>>>(view(),di,dx,dy,dq,(int)n);
        CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaFree(di)); CUDA_OK(cudaFree(dx)); CUDA_OK(cudaFree(dy)); CUDA_OK(cudaFree(dq));
    }

    void eraseBatch(const std::vector<int>& ids){
        if(ids.empty()) return;
        int* di=nullptr;
        CUDA_OK(cudaMalloc(&di,ids.size()*sizeof(int)));
        CUDA_OK(cudaMemcpy(di,ids.data(),ids.size()*sizeof(int),cudaMemcpyHostToDevice));
        markDeletedKernel<<<((int)ids.size()+255)/256,256>>>(view(),di,(int)ids.size());
        CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize()); CUDA_OK(cudaFree(di));
        const int m=routes();
        if(m>0){
            const int threads=256;
            const int blocks=(m*32+threads-1)/threads;
            compactDeletedRoutesKernel<<<blocks,threads>>>(view());
            CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
        }
    }

    // Seed every new/ejected customer into the single device frontier. No
    // route is selected here; the scheduler chooses the route at turn time.
    void insertBatch(const std::vector<int>& batch,int hmax,double /*tau*/){
        if(batch.empty()) return;
        clearQueues();
        CUDA_OK(cudaMemset(d_visited,0,
                           (size_t)maxCustomers*visitedWordsPerCustomer*sizeof(uint32_t)));
        CUDA_OK(cudaMemset(d_candCount,0,maxCustomers*sizeof(int)));
        CUDA_OK(cudaMemset(d_candMode,0,maxCustomers*sizeof(int)));

        if (batch.size() > (size_t)frontierCap) {
            std::fprintf(stderr,
                         "[gpu] ADD batch (%zu) exceeds frontier capacity (%d)\n",
                         batch.size(), frontierCap);
            std::exit(EXIT_FAILURE);
        }

        std::vector<WorkItem> h(batch.size());
        std::vector<int> ready(batch.size(),1);
        for(size_t i=0;i<batch.size();++i) h[i]=WorkItem{batch[i],-1,0,-1};
        CUDA_OK(cudaMemcpy(d_frontier,h.data(),h.size()*sizeof(WorkItem),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_frontierReady,ready.data(),ready.size()*sizeof(int),cudaMemcpyHostToDevice));
        int head=0,tail=(int)h.size(),pending=(int)h.size(),zero=0;
        CUDA_OK(cudaMemcpy(d_frontierHead,&head,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_frontierTail,&tail,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_frontierPending,&pending,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(d_profile,0,sizeof(RepairProfileCounters)));
        { RepairProfileCounters hp{}; hp.frontierPeak=(unsigned long long)pending;
          CUDA_OK(cudaMemcpy(d_profile,&hp,sizeof(hp),cudaMemcpyHostToDevice)); }
        CUDA_OK(cudaMemcpy(d_ejectedCount,&zero,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_activeWorkers,&zero,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_done,&zero,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_dbgCounter,&zero,sizeof(int),cudaMemcpyHostToDevice));

        // Multi-block adaptive scheduler sizing.
        // Target: ~8 routes per scheduler thread over ALL scheduler blocks.
        const int routeSlotsBefore=routes();
        const int potentialRoutes=std::min(maxRoutes,
            std::max(1, routeSlotsBefore + (int)batch.size()));

        const int schedulerThreadsNeeded =
            (potentialRoutes + SCHED_ROUTES_PER_THREAD_TARGET - 1) /
            SCHED_ROUTES_PER_THREAD_TARGET;
        int schedulerWarps = std::max(1,
            (schedulerThreadsNeeded + 31) / 32);

        const int desiredWorkerWarps =
            std::max(1, std::min(potentialRoutes, MAX_ROUTE_WORKER_WARPS));

        int launchDev=0;
        CUDA_OK(cudaGetDevice(&launchDev));

        int smCount=0;
        CUDA_OK(cudaDeviceGetAttribute(
            &smCount, cudaDevAttrMultiProcessorCount, launchDev));

        // Keep route-worker scratch in fast shared memory while we can keep at
        // least TWO warps per block.  If shared-memory pressure would force the
        // launch down to one warp per block, switch back to the normal 8-warps
        // per block layout and use the preallocated per-worker global scratch.
        // This avoids exploding the persistent grid (e.g. 88 scheduler + 256
        // worker blocks at routeStride~=6483) just to preserve shared scratch.
        constexpr size_t SHARED_BUDGET = 48u * 1024u;
        int warpsPerBlock=SCHED_WARPS_PER_BLOCK;
        int threads=0;
        size_t shmem=0;
        int activeBlocksPerSM=0;
        bool useGlobalScratch=false;

        for(;;){
            threads=warpsPerBlock*32;
            shmem=(size_t)warpsPerBlock*routeStride*sizeof(int);
            activeBlocksPerSM=0;
            if(shmem<=SHARED_BUDGET){
                CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &activeBlocksPerSM,
                    addWorkerPoolKernel,
                    threads,
                    shmem));
                if(activeBlocksPerSM>0) break;
            }
            // Do not collapse to a one-warp block.  At that point the cost of
            // having hundreds of persistent blocks (and synchronizing all
            // scheduler blocks) is much worse than using global route scratch.
            // Keep the same TOTAL scheduler/worker warp counts; only repack them
            // into the normal 8-warps-per-block layout.
            if(warpsPerBlock<=2){
                useGlobalScratch=true;
                warpsPerBlock=SCHED_WARPS_PER_BLOCK;
                threads=warpsPerBlock*32;
                shmem=0;
                activeBlocksPerSM=0;
                CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &activeBlocksPerSM,
                    addWorkerPoolKernel,
                    threads,
                    shmem));
                break;
            }
            warpsPerBlock=std::max(2,warpsPerBlock/2);
        }

        const int maxResidentBlocks=activeBlocksPerSM*smCount;

        if(maxResidentBlocks<2){
            std::fprintf(stderr,
                "[gpu] multi-block scheduler needs at least two resident blocks; got %d\n",
                maxResidentBlocks);
            std::exit(EXIT_FAILURE);
        }

        int schedulerBlocks=
            (schedulerWarps+warpsPerBlock-1)/warpsPerBlock;

        // The scheduler uses a global barrier among its blocks.  Therefore the
        // complete persistent grid must fit concurrently.  Always leave at
        // least one resident block for route workers.
        const int maxSchedulerBlocksAllowed=std::max(1,maxResidentBlocks-1);
        if(schedulerBlocks>maxSchedulerBlocksAllowed){
            schedulerBlocks=maxSchedulerBlocksAllowed;
            schedulerWarps=std::min(schedulerWarps,
                                    schedulerBlocks*warpsPerBlock);
            std::fprintf(stderr,
                "[gpu] scheduler target limited by residency: using %d warps in %d blocks\n",
                schedulerWarps,schedulerBlocks);
        }

        if(schedulerBlocks>maxSchedulerBlocks){
            std::fprintf(stderr,
                "[gpu] internal scheduler partial-buffer capacity exceeded: need=%d allocated=%d\n",
                schedulerBlocks,maxSchedulerBlocks);
            std::exit(EXIT_FAILURE);
        }

        const int desiredWorkerBlocks=
            (desiredWorkerWarps+warpsPerBlock-1)/warpsPerBlock;
        const int workerBlocks=std::max(1,
            std::min(desiredWorkerBlocks,maxResidentBlocks-schedulerBlocks));
        const int workerWarps=std::max(1,
            std::min(desiredWorkerWarps,workerBlocks*warpsPerBlock));
        const int blocks=schedulerBlocks+workerBlocks;

        const int effectiveRoutesPerThread =
            (potentialRoutes + schedulerWarps*32 - 1)/(schedulerWarps*32);

        std::fprintf(stderr,
            "[gpu-occupancy] threads=%d shmem=%zu scratch=%s activeBlocksPerSM=%d residentBlocks=%d "
            "gridBlocks=%d routeStride=%d\n",
            threads,shmem,useGlobalScratch?"global":"shared",
            activeBlocksPerSM,maxResidentBlocks,blocks,routeStride);
        std::fprintf(stderr,
            "[gpu-scheduler] routes=%d potential=%d target=%d routes/thread "
            "schedulerWarps=%d schedulerBlocks=%d schedulerWarpsPerBlock=%d "
            "schedulerThreads=%d effectiveRPT=%d workerWarps=%d workerBlocks=%d blocks=%d\n",
            routeSlotsBefore,potentialRoutes,SCHED_ROUTES_PER_THREAD_TARGET,
            schedulerWarps,schedulerBlocks,warpsPerBlock,
            schedulerWarps*32,effectiveRoutesPerThread,
            workerWarps,workerBlocks,blocks);

        addWorkerPoolKernel<<<blocks,threads,shmem>>>(
            view(),d_frontier,d_frontierReady,d_frontierHead,d_frontierTail,d_frontierPending,frontierCap,d_repairError,
            d_qLen,d_qItems,d_qPriority,d_qLock,d_qTotal,
            d_ejected,d_ejectedCount,d_visited,visitedWordsPerCustomer,
            d_candRoutes,d_candPriority2,d_candVersion,d_candCount,d_candMode,
            d_routeOwner,
            d_readyRoutes,d_readyRouteHead,d_readyRouteTail,d_routeSignaled,frontierCap,
            queueCap,hmax,schedulerWarps,schedulerBlocks,workerWarps,
            useGlobalScratch?d_workerScratch:nullptr,
            d_schedCtl,d_schedWork,d_schedBarrierCount,d_schedBarrierSense,
            d_schedBlockTopP2,d_schedBlockTopR,d_schedBlockChecks,d_schedBlockFlags,
            d_activeWorkers,d_done,d_dbgCounter,d_profile);
        CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());

#if DCVRP_PROFILE
        {
            RepairProfileCounters hp{};
            CUDA_OK(cudaMemcpy(&hp,d_profile,sizeof(hp),cudaMemcpyDeviceToHost));
            int clockRateKHz=0;
            CUDA_OK(cudaDeviceGetAttribute(
                &clockRateKHz, cudaDevAttrClockRate, launchDev));
            // clock64() counts SM clock cycles. cudaDevAttrClockRate is in kHz,
            // therefore 1000 / kHz converts cycles to microseconds.
            const double cycToUs = (clockRateKHz > 0) ?
                (1000.0 / (double)clockRateKHz) : 0.0;
            lastProfile={};
            lastProfile.schedulerWarps=schedulerWarps;
            lastProfile.schedulerThreads=schedulerWarps*32;
            lastProfile.schedulerBlocks=schedulerBlocks;
            lastProfile.schedulerWarpsPerBlock=warpsPerBlock;
            lastProfile.routesPerSchedulerThreadTarget=SCHED_ROUTES_PER_THREAD_TARGET;
            lastProfile.workerWarps=workerWarps;
            // The grid itself was capped to the occupancy-resident block count,
            // so every useful launched worker warp belongs to the resident pool.
            const int residentWorkerWarps=workerWarps;
            lastProfile.residentWorkerWarps=residentWorkerWarps;

            lastProfile.schedulerScans=hp.schedulerScans;
            lastProfile.schedulerAssignments=hp.schedulerAssignments;
            lastProfile.schedulerSingletons=hp.schedulerSingletons;
            lastProfile.schedulerBlocked=hp.schedulerBlocked;
            lastProfile.schedulerFrontierItems=hp.schedulerFrontierItems;
            lastProfile.schedulerRouteChecks=hp.schedulerRouteChecks;
            lastProfile.candidateBuilds=hp.candidateBuilds;
            lastProfile.candidateCacheSelections=hp.candidateCacheSelections;
            lastProfile.depotModeScans=hp.depotModeScans;
            lastProfile.readyRouteSignals=hp.readyRouteSignals;
            lastProfile.workerTasks=hp.workerTasks;
            lastProfile.workerDirect=hp.workerDirect;
            lastProfile.workerEject=hp.workerEject;
            lastProfile.workerRetry=hp.workerRetry;
            lastProfile.maxActiveWorkers=hp.maxActiveWorkers;
            lastProfile.maxActiveSchedulerWarps=hp.maxActiveSchedulerWarps;
            lastProfile.maxDepth=hp.maxDepth;
            lastProfile.frontierPeak=hp.frontierPeak;
            lastProfile.queuePeak=hp.queuePeak;
            lastProfile.schedulerTotalUs = (double)hp.schedulerCycles * cycToUs;
            lastProfile.schedulerUsPerAssignment = hp.schedulerAssignments ?
                (lastProfile.schedulerTotalUs / (double)hp.schedulerAssignments) : 0.0;
            lastProfile.schedulerScanUsPerAssignment = hp.schedulerAssignments ?
                ((double)hp.schedulerScanCycles * cycToUs / (double)hp.schedulerAssignments) : 0.0;
            lastProfile.schedulerEnqueueUsPerAssignment = hp.schedulerAssignments ?
                ((double)hp.schedulerEnqueueCycles * cycToUs / (double)hp.schedulerAssignments) : 0.0;
            lastProfile.workerUsPerTask = hp.workerTasks ?
                ((double)hp.workerCycles * cycToUs / (double)hp.workerTasks) : 0.0;
            lastProfile.workerCapacityUs = residentWorkerWarps ?
                lastProfile.workerUsPerTask / (double)residentWorkerWarps : 0.0;
            lastProfile.schedulerToCapacity = lastProfile.workerCapacityUs > 0.0 ?
                lastProfile.schedulerUsPerAssignment / lastProfile.workerCapacityUs : 0.0;
            if (hp.repairEnd >= hp.repairStart)
                lastProfile.repairKernelUs = (double)(hp.repairEnd-hp.repairStart) * cycToUs;
            if (hp.repairEnd > hp.repairStart) {
                const double elapsed=(double)(hp.repairEnd-hp.repairStart);
                lastProfile.avgActiveWorkers=(double)hp.workerCycles/elapsed;
                lastProfile.avgActiveSchedulerWarps=(double)hp.schedulerWarpCycles/elapsed;
                if (residentWorkerWarps > 0)
                    lastProfile.workerUtilPct=100.0*lastProfile.avgActiveWorkers/(double)residentWorkerWarps;
            }
            std::fprintf(stderr,
                "[gpu-profile] Ts=%.3fus (scan=%.3f enqueue=%.3f) Tw=%.3fus "
                "Tw/Wres=%.3fus ratio=%.2f avgActive=%.2f util~=%.1f%% "
                "activeMax=%llu Savg=%.2f Smax=%llu SW=%d Sthreads=%d targetRPT=%d Wres=%d Wlaunch=%d "
                "scans=%llu assign=%llu singleton=%llu blocked=%llu frontierItems=%llu routeChecks=%llu "
                "candBuild=%llu cacheSelect=%llu depotScans=%llu readySignals=%llu "
                "tasks=%llu direct=%llu eject=%llu retry=%llu depthMax=%llu frontierPeak=%llu qPeak=%llu\n",
                lastProfile.schedulerUsPerAssignment,lastProfile.schedulerScanUsPerAssignment,
                lastProfile.schedulerEnqueueUsPerAssignment,lastProfile.workerUsPerTask,
                lastProfile.workerCapacityUs,lastProfile.schedulerToCapacity,
                lastProfile.avgActiveWorkers,lastProfile.workerUtilPct,lastProfile.maxActiveWorkers,
                lastProfile.avgActiveSchedulerWarps,lastProfile.maxActiveSchedulerWarps,
                schedulerWarps,schedulerWarps*32,SCHED_ROUTES_PER_THREAD_TARGET,
                residentWorkerWarps,workerWarps,lastProfile.schedulerScans,lastProfile.schedulerAssignments,lastProfile.schedulerSingletons,
                lastProfile.schedulerBlocked,lastProfile.schedulerFrontierItems,lastProfile.schedulerRouteChecks,
                lastProfile.candidateBuilds,lastProfile.candidateCacheSelections,lastProfile.depotModeScans,lastProfile.readyRouteSignals,
                lastProfile.workerTasks,lastProfile.workerDirect,
                lastProfile.workerEject,lastProfile.workerRetry,lastProfile.maxDepth,
                lastProfile.frontierPeak,lastProfile.queuePeak);
        }
#else
        lastProfile={}; lastProfile.schedulerWarps=schedulerWarps;
        lastProfile.schedulerThreads=schedulerWarps*32;
        lastProfile.schedulerBlocks=schedulerBlocks;
        lastProfile.schedulerWarpsPerBlock=warpsPerBlock;
        lastProfile.routesPerSchedulerThreadTarget=SCHED_ROUTES_PER_THREAD_TARGET;
        lastProfile.workerWarps=workerWarps; lastProfile.residentWorkerWarps=workerWarps;
#endif

        int repairError=0;
        CUDA_OK(cudaMemcpy(&repairError,d_repairError,sizeof(int),cudaMemcpyDeviceToHost));
        if(repairError!=0){
            const char* why =
                (repairError==1) ? "frontier capacity exhausted" :
                (repairError==2) ? "customer demand exceeds vehicle capacity" :
                (repairError==3) ? "no route slot available for singleton fallback" :
                (repairError==4) ? "ready-route signal capacity exhausted" :
                                   "unknown device repair error";
            std::fprintf(stderr,"[gpu] ADD repair failed: %s (code=%d)\n",why,repairError);
            std::exit(EXIT_FAILURE);
        }
    }

    const RepairProfileReport& repairProfile() const { return lastProfile; }

    std::vector<WorkItem> unplaced(int depth) const {
        std::vector<int> ro(maxCustomers),active(maxCustomers);
        CUDA_OK(cudaMemcpy(ro.data(),d_routeOf,maxCustomers*sizeof(int),cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(active.data(),d_active,maxCustomers*sizeof(int),cudaMemcpyDeviceToHost));
        std::vector<WorkItem> u;
        for(int v=1;v<(int)I->xs.size()&&v<maxCustomers;++v)
            if(active[v] && ro[v]<0) u.push_back({v,-1,depth,-1});
        return u;
    }

    std::vector<std::vector<int>> downloadRoutes() const {
        const int m=routes();
        std::vector<int> len(m),nodes((size_t)maxRoutes*routeStride);
        CUDA_OK(cudaMemcpy(len.data(),d_len,m*sizeof(int),cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(nodes.data(),d_nodes,nodes.size()*sizeof(int),cudaMemcpyDeviceToHost));
        std::vector<std::vector<int>> out;
        for(int r=0;r<m;++r) if(len[r]>0)
            out.emplace_back(nodes.begin()+r*routeStride,nodes.begin()+r*routeStride+len[r]);
        return out;
    }

    // Build a sparse refinement graph: each live route nominates only its K
    // nearest route centroids. The union is deduplicated, then greedily split
    // into conflict-free matching rounds so no route is mutated twice in one
    // kernel launch. This replaces the O(R^2) complete-graph round robin.
    std::vector<std::vector<int2>> makeNearestRounds(
            const std::vector<int>& live, int partnerK,
            unsigned long long& candidatePairs) const {
        candidatePairs = 0;
        if(live.size()<2) return {};
        partnerK=std::max(1,std::min(partnerK,(int)live.size()-1));
        const int m=routes();
        std::vector<double> cx(m),cy(m);
        CUDA_OK(cudaMemcpy(cx.data(),d_cenX,m*sizeof(double),cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(cy.data(),d_cenY,m*sizeof(double),cudaMemcpyDeviceToHost));

        struct Edge{int a,b;double d2;};
        std::vector<Edge> edges;
        edges.reserve((size_t)live.size()*partnerK);
        std::unordered_set<unsigned long long> seen;
        auto key=[](int a,int b){
            if(a>b) std::swap(a,b);
            return (static_cast<unsigned long long>(static_cast<unsigned int>(a))<<32) |
                   static_cast<unsigned int>(b);
        };

        for(int r:live){
            std::vector<std::pair<double,int>> best;
            best.reserve(partnerK+1);
            for(int q:live){
                if(q==r) continue;
                const double dx=cx[r]-cx[q],dy=cy[r]-cy[q];
                const double d2=dx*dx+dy*dy;
                auto it=std::lower_bound(best.begin(),best.end(),std::make_pair(d2,q),
                    [](const auto& A,const auto& B){
                        return A.first<B.first || (A.first==B.first && A.second<B.second);
                    });
                best.insert(it,{d2,q});
                if((int)best.size()>partnerK) best.pop_back();
            }
            for(const auto& [d2,q]:best){
                int a=std::min(r,q),b=std::max(r,q);
                const auto k=key(a,b);
                if(seen.insert(k).second) edges.push_back({a,b,d2});
            }
        }
        std::sort(edges.begin(),edges.end(),[](const Edge&A,const Edge&B){
            if(A.d2!=B.d2) return A.d2<B.d2;
            if(A.a!=B.a) return A.a<B.a;
            return A.b<B.b;
        });
        candidatePairs=edges.size();

        std::vector<std::vector<int2>> rounds;
        while(!edges.empty()){
            std::vector<char> used(m,0);
            std::vector<Edge> remain;
            std::vector<int2> round;
            round.reserve(live.size()/2);
            remain.reserve(edges.size());
            for(const auto&e:edges){
                if(!used[e.a]&&!used[e.b]){
                    used[e.a]=used[e.b]=1;
                    round.push_back(make_int2(e.a,e.b));
                }else remain.push_back(e);
            }
            if(!round.empty()) rounds.push_back(std::move(round));
            edges.swap(remain);
        }
        return rounds;
    }

    int runPairRounds(const std::vector<std::vector<int2>>& rounds,bool move89){
        // Preserve the previously stable <=48 KiB packing for normal cases.
        // Only when even ONE pair-processing warp needs more than 48 KiB do we
        // opt that refinement kernel into a bounded 60 KiB dynamic-shared path.
        // This keeps 800/1600/6400 behaviour unchanged while allowing the
        // ~51 KiB one-warp scratch needed by the 12800-event Large5 case.
        constexpr size_t DEFAULT_SHARED_BUDGET = 48u * 1024u;
        constexpr size_t OPTIN_SHARED_BUDGET   = 60u * 1024u;

        int warpsPerBlock=8;
        while(warpsPerBlock>1 &&
              (size_t)warpsPerBlock*2*routeStride*sizeof(int)>DEFAULT_SHARED_BUDGET)
            warpsPerBlock/=2;

        const int threads=warpsPerBlock*32;
        const size_t shmem=(size_t)warpsPerBlock*2*routeStride*sizeof(int);

        if(shmem>DEFAULT_SHARED_BUDGET){
            int dev=0, maxOptin=0;
            CUDA_OK(cudaGetDevice(&dev));
            CUDA_OK(cudaDeviceGetAttribute(
                &maxOptin,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));

            cudaFuncAttributes attr{};
            if(move89)
                CUDA_OK(cudaFuncGetAttributes(&attr,pairMove89Kernel));
            else
                CUDA_OK(cudaFuncGetAttributes(&attr,pairMove1SwapKernel));

            const size_t totalShared=shmem+(size_t)attr.sharedSizeBytes;
            if(shmem>OPTIN_SHARED_BUDGET || totalShared>(size_t)maxOptin){
                std::fprintf(stderr,
                    "[gpu-refine] routeStride=%d requires %zu bytes dynamic shared "
                    "(%zu total), above configured 60 KiB/device limit %d bytes\n",
                    routeStride,shmem,totalShared,maxOptin);
                std::exit(EXIT_FAILURE);
            }

            if(move89)
                CUDA_OK(cudaFuncSetAttribute(
                    pairMove89Kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)shmem));
            else
                CUDA_OK(cudaFuncSetAttribute(
                    pairMove1SwapKernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)shmem));
        }

        int zero=0;
        CUDA_OK(cudaMemcpy(d_changed,&zero,sizeof(int),cudaMemcpyHostToDevice));
        for(const auto& hp:rounds){
            if(hp.empty()) continue;
            CUDA_OK(cudaMemcpy(d_pairs,hp.data(),hp.size()*sizeof(int2),cudaMemcpyHostToDevice));
            const int blocks=((int)hp.size()+warpsPerBlock-1)/warpsPerBlock;
            if(move89)
                pairMove89Kernel<<<blocks,threads,shmem>>>(view(),d_pairs,(int)hp.size(),d_changed);
            else
                pairMove1SwapKernel<<<blocks,threads,shmem>>>(view(),d_pairs,(int)hp.size(),d_changed);
            CUDA_OK(cudaGetLastError());
            // A round is conflict-free internally, but the next round must see
            // this round's route mutations before it evaluates new candidates.
            CUDA_OK(cudaDeviceSynchronize());
        }
        int changed=0;
        CUDA_OK(cudaMemcpy(&changed,d_changed,sizeof(int),cudaMemcpyDeviceToHost));
        return changed;
    }

    const RefineProfileReport& refineProfile() const { return lastRefine; }

    void refineAll(int passes,int partnerK=5){
        lastRefine={};
        lastRefine.passesRequested=passes;
        lastRefine.nearestPartnerK=std::max(1,partnerK);
        if(passes<=0) return;

        using Clock=std::chrono::steady_clock;
        auto us=[](Clock::time_point a, Clock::time_point b){
            return std::chrono::duration<double,std::micro>(b-a).count();
        };
        auto currentLive=[&](){
            const int m=routes();
            std::vector<int> hLen(m);
            if(m>0) CUDA_OK(cudaMemcpy(hLen.data(),d_len,m*sizeof(int),cudaMemcpyDeviceToHost));
            std::vector<int> live; live.reserve(m);
            for(int r=0;r<m;++r) if(hLen[r]>0) live.push_back(r);
            return live;
        };

        const int maxPasses=std::max(1,passes);
        for(int pass=0;pass<maxPasses;++pass){
            const int m=routes();
            int c89=0,c1s=0,c7=0;

            // Stage 1: Move8/Move9 only against K nearest centroid partners.
            auto live=currentLive();
            if(live.size()>=2){
                unsigned long long pairs89=0;
                const auto rounds89=makeNearestRounds(live,lastRefine.nearestPartnerK,pairs89);
                lastRefine.pairCandidates89 += pairs89;
                lastRefine.pairRounds += rounds89.size();
                auto t0=Clock::now();
                c89=runPairRounds(rounds89,true);
                auto t1=Clock::now();
                lastRefine.move89Us += us(t0,t1);
                lastRefine.move89Committed += (unsigned long long)c89;
            }

            // Stage 2: Move1/SWAP*. Rebuild neighbours because Stage 1 may have
            // changed route centroids. Again only K-nearest route pairs enter.
            live=currentLive();
            if(live.size()>=2){
                unsigned long long pairs1s=0;
                const auto rounds1s=makeNearestRounds(live,lastRefine.nearestPartnerK,pairs1s);
                lastRefine.pairCandidates1Swap += pairs1s;
                lastRefine.pairRounds += rounds1s.size();
                auto t0=Clock::now();
                c1s=runPairRounds(rounds1s,false);
                auto t1=Clock::now();
                lastRefine.move1SwapUs += us(t0,t1);
                lastRefine.move1SwapCommitted += (unsigned long long)c1s;
            }

            // Stage 3: all live routes are independent. One FULL warp per route
            // evaluates the route's Move7 cuts; all available GPU warps are free
            // for this phase because repair has already completed.
            int zero=0; CUDA_OK(cudaMemcpy(d_changed,&zero,sizeof(int),cudaMemcpyHostToDevice));
            auto t0=Clock::now();
            if(m>0){
                const int threads=256;
                const int blocks=(m*32+threads-1)/threads;
                twoOptOnceKernel<<<blocks,threads>>>(view(),d_changed);
                CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(&c7,d_changed,sizeof(int),cudaMemcpyDeviceToHost));
            }
            auto t1=Clock::now();
            lastRefine.move7Us += us(t0,t1);
            lastRefine.move7Committed += (unsigned long long)c7;
            lastRefine.passesRun++;
            if(c89+c1s+c7==0) break;
        }

        std::fprintf(stderr,
            "[gpu-refine] passes=%d/%d nearestK=%d move89=%.3fms (%llu moves, %llu pairs) "
            "move1swap=%.3fms (%llu moves, %llu pairs) move7=%.3fms (%llu moves) pairRounds=%llu\n",
            lastRefine.passesRun,lastRefine.passesRequested,lastRefine.nearestPartnerK,
            lastRefine.move89Us/1000.0,lastRefine.move89Committed,lastRefine.pairCandidates89,
            lastRefine.move1SwapUs/1000.0,lastRefine.move1SwapCommitted,lastRefine.pairCandidates1Swap,
            lastRefine.move7Us/1000.0,lastRefine.move7Committed,lastRefine.pairRounds);
    }
};

} // namespace dcvrp_cuda
#endif // !DCVRP_CPU_ONLY
