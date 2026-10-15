// ═══════════════════════════════════════════════════════════════════════════════
// dynamic_cvrp_cuda.cu — GPU-Accelerated Dynamic CVRP Insertion & Deletion
//
// This tool handles modifications to an existing Capacitated Vehicle Routing
// Problem (CVRP) solution. It parses a problem file (.vrp) and its solution (.sol),
// adds or removes a customer node, and then runs a fast, granular Variable 
// Neighbourhood Descent (VND) local search to re-optimize and clean up the result.
//
// Usage:
//   Add Node:  dynamic_cvrp_cuda a <instance.vrp> <solution.sol> <out.sol> <out.vrp> <X> <Y> <demand>
//   Drop Node: dynamic_cvrp_cuda d <instance.vrp> <solution.sol> <out.sol> <out.vrp> <node_id>
//
// Compile:
//   nvcc -std=c++17 -O2 -o dynamic_cvrp_cuda dynamic_cvrp_cuda.cu
// ═══════════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// ─── CUDA & Thrust ───────────────────────────────────────────────────────────
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

// Helper to catch and report CUDA failures immediately
#define cudaCheckError(call)                                                   \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " — " << cudaGetErrorString(err) << "\n";            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ──────────────────────────── Core Constants ─────────────────────────────────
static constexpr double MY_EPSILON  = 0.00001;
static constexpr double PI_VAL      = 3.14159265359;
static constexpr int    NB_GRANULAR = 20;  // Neighborhood size for localized search filtering

// ══════════════════════════════════════════════════════════════════════════════
//  Data Structures
// ══════════════════════════════════════════════════════════════════════════════

// Tracks angular coverage of a route around the depot using integer mapping [0, 65535]
struct CircleSector {
    int start = 0, end = 0;
    static int positive_mod(int i) { return (i % 65536 + 65536) % 65536; }
    void initialize(int p) { start = end = p; }
    bool isEnclosed(int p) const {
        return positive_mod(p - start) <= positive_mod(end - start);
    }
    static bool overlap(const CircleSector& a, const CircleSector& b) {
        return positive_mod(b.start - a.start) <= positive_mod(a.end - a.start)
            || positive_mod(a.start - b.start) <= positive_mod(b.end - b.start);
    }
    void extend(int p) {
        if (!isEnclosed(p)) {
            if (positive_mod(p - end) <= positive_mod(start - p)) end = p;
            else start = p;
        }
    }
};

// Represents a standalone client profile.
// Coords are kept in separated global arrays (Structure of Arrays) inside the engine 
// to maximize cache hits during distance sweeps.
struct Client {
    double serviceDuration = 0.0;
    double demand         = 0.0;
    int    polarAngle     = 0;     // Angle relative to depot, normalized to 16-bit integer
};

struct Route;

// Doubly-linked list node representation for local search paths
struct Node {
    bool   isDepot   = false;
    int    cour      = 0;          // Client index (0 = depot, 1..n = customers)
    int    position  = 0;          // Index position inside its current route
    int    whenLastTestedRI = -1;  // Iteration tracking token for move filters
    Node*  next      = nullptr;
    Node*  prev      = nullptr;
    Route* route     = nullptr;
    double cumulatedLoad = 0.0;
    double cumulatedTime = 0.0;
    double cumulatedReversalDistance = 0.0;
    double deltaRemoval = 0.0;     // Precomputed savings cost used by SWAP*
};

// Route manager structure for tracking capacity and timing invariants
struct Route {
    int    cour       = 0;         // Route ID identifier
    int    nbCustomers = 0;
    int    whenLastModified = 0;
    int    whenLastTestedSWAPStar = -1;
    Node*  depot      = nullptr;   // Sentinel pointing to the route's starting depot
    double duration   = 0.0;
    double load       = 0.0;
    double reversalDistance = 0.0;
    double penalty    = 0.0;
    double polarAngleBarycenter = 0.0;
    CircleSector sector;
};

// Simple top-3 insertion cache tracking for the SWAP* operator
struct ThreeBestInsert {
    int    whenLastCalculated = -1;
    double bestCost[3]{};
    Node*  bestLocation[3]{};

    void compareAndAdd(double c, Node* p) {
        if (c >= bestCost[2]) return;
        if (c >= bestCost[1])      { bestCost[2]=c; bestLocation[2]=p; }
        else if (c >= bestCost[0]) { bestCost[2]=bestCost[1]; bestLocation[2]=bestLocation[1];
                                     bestCost[1]=c; bestLocation[1]=p; }
        else                       { bestCost[2]=bestCost[1]; bestLocation[2]=bestLocation[1];
                                     bestCost[1]=bestCost[0]; bestLocation[1]=bestLocation[0];
                                     bestCost[0]=c; bestLocation[0]=p; }
    }
    void reset() {
        bestCost[0] = bestCost[1] = bestCost[2] = 1.e30;
        bestLocation[0] = bestLocation[1] = bestLocation[2] = nullptr;
    }
    ThreeBestInsert() { reset(); }
};

// Container to retain data for the optimal target SWAP* change
struct SwapStarElement {
    double moveCost      = 1.e30;
    Node*  U             = nullptr;
    Node*  bestPositionU = nullptr;
    Node*  V             = nullptr;
    Node*  bestPositionV = nullptr;
};

// ══════════════════════════════════════════════════════════════════════════════
//  CVRPEngine Core
// ══════════════════════════════════════════════════════════════════════════════
class CVRPEngine {
public:
    bool verbose = false;
    bool timecheck = false;
    virtual ~CVRPEngine() = default;

protected:
    // Problem parameters
    int    nbClients  = 0;               // Total customers (excluding depot)
    int    nbVehicles = 0;               // Available route structural slots
    double vehicleCap = 0.0;
    double durationLimit = 1.e30;
    double maxDist = 0.0, maxDemand = 0.0;

    std::vector<Client>              cli;       // Client list: depot is at index 0
    std::vector<float>               coordX;    // SoA X coordinates
    std::vector<float>               coordY;    // SoA Y coordinates
    std::vector<std::vector<int>>    correlated; // Precomputed proximity neighbors

    // Compute integer-rounded Euclidean distance dynamically to save cache space
    inline float d(int i, int j) const {
        float dx = coordX[i] - coordX[j];
        float dy = coordY[i] - coordY[j];
        return roundf(sqrtf(dx * dx + dy * dy));
    }

    // Instance metadata strings
    std::string vrpName = "";
    std::string vrpComment = "";
    std::string vrpType = "CVRP";
    std::string edgeWeightType = "EUC_2D";

    // Initial parsed configuration 
    std::vector<std::vector<int>> solRoutes;    
    int solCost = 0;

    // Local Search constraint weight rules
    double penCapLS = 1.0;
    double penDurLS = 0.0;   // 0 indicates unconstrained route duration limits

    // Local Search memory layouts
    std::vector<Node>  nodesCli;          
    std::vector<Node>  lsDepots;          // Route start markers
    std::vector<Node>  lsDepotsEnd;       // Route end markers
    std::vector<Route> lsRoutes;

    std::set<int> emptyRoutes;
    std::vector<ThreeBestInsert> cacheUintoV;
    std::vector<ThreeBestInsert> cacheVintoU;

    int  nbMoves        = 0;
    bool searchCompleted = false;
    std::minstd_rand rng{42};

    // Shared state registers for localized structural modifications
    Node*  nodeU = nullptr;  Node*  nodeX = nullptr;
    Node*  nodeV = nullptr;  Node*  nodeY = nullptr;
    Route* routeU = nullptr; Route* routeV = nullptr;
    int nUPrev=0, nU=0, nX=0, nXNext=0;
    int nVPrev=0, nV=0, nY=0, nYNext=0;
    double ldU=0, ldX=0, ldV=0, ldY=0;
    double svU=0, svX=0, svV=0, svY=0;
    bool intraRoute = false;

    // Helpers
    void parseVRP(const std::string& path);
    void parseSolution(const std::string& path);
    void computeCorrelated();
    void allocateLS(int numRouteSlots);
    void loadSolutionRoutes();   
    void updateRouteData(Route* r);
    void insertNodeLS(Node* U, Node* V);   // Detach node U and relocate it directly after V
    void swapNodeLS(Node* U, Node* V);     // Directly swap graph locations of nodes U and V

    void setVarsU();       
    void setVarsV();       

    double penExLoad(double l) const { return std::max(0.0, l - vehicleCap) * penCapLS; }
    double penExDur (double d) const { return std::max(0.0, d - durationLimit) * penDurLS; }

    // Hard feasibility boundary check
    bool exceedsCap(double load) const { return load > vehicleCap + MY_EPSILON; }

    // Local Search neighborhood mutation operators
    bool move1();  // Relocate node U right after node V
    bool move2();  // Relocate adjacent pairs (U, X) right after V preserving order
    bool move3();  // Relocate adjacent pairs (U, X) right after V reversing their order
    bool move4();  // Swap single node locations of U and V
    bool move5();  // Swap adjacent pair (U, X) with a single node V
    bool move6();  // Swap adjacent pair (U, X) with adjacent pair (V, Y)
    bool move7();  // Localized 2-opt sequence reversal inside a single route
    bool move8();  // Inter-route 2-opt step involving sub-path inversion
    bool move9();  // Inter-route 2-opt cross step without tracking inversion

    // SWAP* Engine components
    bool doSwapStar();
    double cheapInsertSimultRemoval(Node* U, Node* V, Node*& bestPos,
                                    std::vector<ThreeBestInsert>& insCache);
    void preprocessInsertions(Route* R1, Route* R2,
                              std::vector<ThreeBestInsert>& insCache);

    void runVND();

    // Outputs
    struct SolOutput {
        std::vector<std::vector<int>> routes;
        int  totalDist = 0;
        bool feasible  = true;
        double capExcess = 0.0;
    };
    SolOutput exportSolution();
    void writeSolution(const std::string& path, const SolOutput& s);
    void writeVRP(const std::string& path, const std::string& opDesc);
    void printAndVerify(const SolOutput& sol);
};

// ══════════════════════════════════════════════════════════════════════════════
//  Inserter Engine (Dynamic Add Node)
// ══════════════════════════════════════════════════════════════════════════════
class Inserter : public CVRPEngine {
public:
    void run(const std::string& vrpFile,
             const std::string& solFile,
             const std::string& outSolFile,
             const std::string& outVrpFile,
             double newX, double newY, double newDemand);

private:
    int newNodeIdx = -1;
    void addNewNode(double x, double y, double demand);
    void loadSolutionWithSpare();
};

// ══════════════════════════════════════════════════════════════════════════════
//  Deleter Engine (Dynamic Remove Node)
// ══════════════════════════════════════════════════════════════════════════════
class Deleter : public CVRPEngine {
public:
    void run(const std::string& vrpFile,
             const std::string& solFile,
             const std::string& outSolFile,
             const std::string& outVrpFile,
             int deleteNodeId);

private:
    void removeNodeFromSolution(int nodeId);
    void removeNodeFromProblem(int nodeId);
};

// ══════════════════════════════════════════════════════════════════════════════
//  CVRPEngine Methods
// ══════════════════════════════════════════════════════════════════════════════

// Reads standard TSPLIB formatted layout strings
void CVRPEngine::parseVRP(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) throw std::string("Cannot open VRP file: " + path);

    std::string line, tok;
    int dimension = 0;

    while (std::getline(in, line)) {
        size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        size_t last = line.find_last_not_of(" \t\r\n");
        std::string cleaned = line.substr(first, (last - first + 1));

        if (cleaned == "NODE_COORD_SECTION") {
            break;
        }

        size_t colon = cleaned.find(':');
        if (colon != std::string::npos) {
            std::string key = cleaned.substr(0, colon);
            std::string val = cleaned.substr(colon + 1);

            size_t k_first = key.find_first_not_of(" \t");
            size_t k_last = key.find_last_not_of(" \t");
            if (k_first != std::string::npos) key = key.substr(k_first, (k_last - k_first + 1));

            size_t v_first = val.find_first_not_of(" \t");
            size_t v_last = val.find_last_not_of(" \t");
            if (v_first != std::string::npos) val = val.substr(v_first, (v_last - v_first + 1));

            if (key == "NAME") vrpName = val;
            else if (key == "COMMENT") vrpComment = val;
            else if (key == "TYPE") vrpType = val;
            else if (key == "DIMENSION") dimension = std::stoi(val);
            else if (key == "EDGE_WEIGHT_TYPE") edgeWeightType = val;
            else if (key == "CAPACITY") vehicleCap = std::stod(val);
            else if (key == "DISTANCE") durationLimit = std::stod(val);
        } else {
            std::istringstream ss(cleaned);
            std::string key;
            ss >> key;
            if (key == "DIMENSION") { ss >> dimension; }
            else if (key == "CAPACITY") { ss >> vehicleCap; }
            else if (key == "DISTANCE") { ss >> durationLimit; }
            else if (key == "EDGE_WEIGHT_TYPE") { ss >> edgeWeightType; }
            else if (key == "TYPE") { ss >> vrpType; }
        }
    }

    if (dimension <= 0)     throw std::string("DIMENSION not found in VRP file");
    if (vehicleCap <= 0)    throw std::string("CAPACITY not found in VRP file");

    nbClients = dimension - 1;  

    // Load geometry fields
    std::vector<double> xc(dimension), yc(dimension);
    for (int i = 0; i < dimension; i++) {
        int id; 
        in >> id >> xc[i] >> yc[i];
    }

    // Load client demand matrices
    in >> tok;
    if (tok != "DEMAND_SECTION") throw std::string("Expected DEMAND_SECTION, got: " + tok);
    std::vector<double> dem(dimension);
    for (int i = 0; i < dimension; i++) {
        int id; 
        in >> id >> dem[i];
    }

    // Initialize arrays (Index 0 is mapped for Depot tracking properties)
    cli.resize(dimension);
    coordX.resize(dimension);
    coordY.resize(dimension);
    for (int i = 0; i < dimension; i++) {
        coordX[i] = static_cast<float>(xc[i]);
        coordY[i] = static_cast<float>(yc[i]);
        cli[i].demand = dem[i];
        cli[i].serviceDuration = 0.0;
        cli[i].polarAngle = CircleSector::positive_mod(
            static_cast<int>(32768.0 * std::atan2(yc[i] - yc[0], xc[i] - xc[0]) / PI_VAL));
    }

    if (verbose) std::cerr << "[parse] VRP loaded: " << nbClients
              << " clients, capacity=" << vehicleCap << "\n";
}

// Read existing sequence patterns from solution output files
void CVRPEngine::parseSolution(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) throw std::string("Cannot open SOL file: " + path);

    solRoutes.clear();
    std::string tok;
    in >> tok; 

    while (tok == "Route") {
        in >> tok; 
        std::string line;
        std::getline(in, line);
        std::istringstream ss(line);
        std::vector<int> route;
        int cust;
        while (ss >> cust) route.push_back(cust);
        if (!route.empty()) solRoutes.push_back(std::move(route));
        in >> tok; 
    }
    if (tok == "Cost") in >> solCost;

    if (verbose) std::cerr << "[parse] Solution loaded: " << solRoutes.size()
              << " routes, cost=" << solCost << "\n";
}

// ══════════════════════════════════════════════════════════════════════════════
//  CUDA Kernels
// ══════════════════════════════════════════════════════════════════════════════

// Compute local neighborhoods per thread via on-the-fly calculations.
// This design lets us avoid computing a full dense NxN matrix in host memory,
// keeping the processing footprint highly lightweight.
__global__ void computeCorrelatedCostsKernel(
    const float* __restrict__ d_coordX,   
    const float* __restrict__ d_coordY,   
    int   N,                               
    int   nbClients,
    int   nbGran,                          
    int*  d_topk)                          
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nbClients) return;

    int i = idx + 1;   
    float xi = d_coordX[i];
    float yi = d_coordY[i];

    // Local thread scratchpads track nearest neighbors.
    // Neighbors with larger distances are pushed out as closer items arrive.
    float topDist[NB_GRANULAR];
    int   topIdx [NB_GRANULAR];

    for (int k = 0; k < nbGran; k++) {
        topDist[k] = 1.0e30f;
        topIdx[k]  = -1;
    }
    int maxPos = 0;  

    for (int j = 1; j <= nbClients; j++) {
        if (j == i) continue;

        float dx = xi - d_coordX[j];
        float dy = yi - d_coordY[j];
        float dist = roundf(sqrtf(dx * dx + dy * dy));  

        // Break ties deterministically using the node IDs
        if (dist < topDist[maxPos] || (dist == topDist[maxPos] && j < topIdx[maxPos])) {
            topDist[maxPos] = dist;
            topIdx[maxPos]  = j;

            maxPos = 0;
            for (int k = 1; k < nbGran; k++) {
                if (topDist[k] > topDist[maxPos] || 
                    (topDist[k] == topDist[maxPos] && topIdx[k] > topIdx[maxPos])) {
                    maxPos = k;
                }
            }
        }
    }

    int base = idx * nbGran;
    for (int k = 0; k < nbGran; k++) {
        d_topk[base + k] = topIdx[k];
    }
}

// Dynamic kernel tracking row-max values to establish bound thresholds
__global__ void computeRowMaxDistKernel(
    const float* __restrict__ d_coordX,
    const float* __restrict__ d_coordY,
    int   N,             
    float* d_rowMax)     
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float xi = d_coordX[i];
    float yi = d_coordY[i];
    float localMax = 0.0f;

    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        float dx = xi - d_coordX[j];
        float dy = yi - d_coordY[j];
        float dist = roundf(sqrtf(dx * dx + dy * dy));
        if (dist > localMax) localMax = dist;
    }
    d_rowMax[i] = localMax;
}

// Coordinates GPU worker threads to construct the proximity map
void CVRPEngine::computeCorrelated() {
    int nbGran = std::min(NB_GRANULAR, nbClients - 1);
    int N      = nbClients + 1;   

    if (nbGran <= 0 || nbClients <= 1) {
        correlated.assign(nbClients + 1, {});
        maxDist = 0; maxDemand = 0;
        if (nbClients >= 1) {
            maxDemand = cli[1].demand;
            for (int i = 0; i <= nbClients; i++)
                for (int j = 0; j <= nbClients; j++)
                    if (d(i, j) > maxDist) maxDist = d(i, j);
        }
        penCapLS = (maxDemand > 0) ? std::max(0.1, std::min(1000.0, maxDist / maxDemand)) : 1.0;
        penDurLS = 0.0;
        return;
    }

    float *d_coordX = nullptr, *d_coordY = nullptr;
    int    *d_topk   = nullptr;

    cudaCheckError(cudaMalloc(&d_coordX, N * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_coordY, N * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_topk,   nbClients * nbGran * sizeof(int)));

    cudaCheckError(cudaMemcpy(d_coordX, coordX.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_coordY, coordY.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = (nbClients + threadsPerBlock - 1) / threadsPerBlock;

    computeCorrelatedCostsKernel<<<blocks, threadsPerBlock>>>(
        d_coordX, d_coordY, N, nbClients, nbGran, d_topk);

    cudaCheckError(cudaGetLastError());         
    cudaCheckError(cudaDeviceSynchronize());    

    std::vector<int> h_topk(nbClients * nbGran);
    cudaCheckError(cudaMemcpy(h_topk.data(), d_topk, nbClients * nbGran * sizeof(int), cudaMemcpyDeviceToHost));

    // Reduce maximum spans inside GPU VRAM safely
    {
        thrust::device_vector<float> d_rowMax(N, 0.0f);
        int tpb  = 256;
        int blks = (N + tpb - 1) / tpb;
        computeRowMaxDistKernel<<<blks, tpb>>>(d_coordX, d_coordY, N, thrust::raw_pointer_cast(d_rowMax.data()));
        cudaCheckError(cudaGetLastError());
        cudaCheckError(cudaDeviceSynchronize());
        maxDist = *thrust::max_element(d_rowMax.begin(), d_rowMax.end());
    }

    {
        std::vector<double> h_demands(nbClients);
        for (int i = 0; i < nbClients; i++) h_demands[i] = cli[i + 1].demand;
        thrust::device_vector<double> d_demands(h_demands.begin(), h_demands.end());
        maxDemand = *thrust::max_element(d_demands.begin(), d_demands.end());
    }

    cudaCheckError(cudaFree(d_coordX));
    cudaCheckError(cudaFree(d_coordY));
    cudaCheckError(cudaFree(d_topk));

    // Construct the symmetric neighborhood links on the host
    correlated.assign(nbClients + 1, {});
    std::vector<std::set<int>> sets(nbClients + 1);

    for (int idx = 0; idx < nbClients; idx++) {
        int i    = idx + 1;
        int base = idx * nbGran;
        for (int k = 0; k < nbGran; k++) {
            int j = h_topk[base + k];
            if (j >= 1 && j <= nbClients) {    
                sets[i].insert(j);
                sets[j].insert(i);             
            }
        }
    }
    for (int i = 1; i <= nbClients; i++)
        correlated[i].assign(sets[i].begin(), sets[i].end());

    penCapLS = std::max(0.1, std::min(1000.0, maxDist / maxDemand));
    penDurLS = 0.0; 
}

// Memory instantiation initialization paths
void CVRPEngine::allocateLS(int numRouteSlots) {
    nbVehicles = numRouteSlots;

    nodesCli.resize(nbClients + 1);
    lsDepots.resize(nbVehicles);
    lsDepotsEnd.resize(nbVehicles);
    lsRoutes.resize(nbVehicles);

    cacheUintoV.assign(nbClients + 1, ThreeBestInsert());
    cacheVintoU.assign(nbClients + 1, ThreeBestInsert());

    for (int i = 0; i <= nbClients; i++) {
        nodesCli[i].cour    = i;
        nodesCli[i].isDepot = false;
    }
    for (int r = 0; r < nbVehicles; r++) {
        lsRoutes[r].cour  = r;
        lsRoutes[r].depot = &lsDepots[r];
        lsDepots[r].cour    = 0;
        lsDepots[r].isDepot = true;
        lsDepots[r].route   = &lsRoutes[r];
        lsDepotsEnd[r].cour    = 0;
        lsDepotsEnd[r].isDepot = true;
        lsDepotsEnd[r].route   = &lsRoutes[r];
    }
}

// Transform matrix array representations into connected pointer graphs
void CVRPEngine::loadSolutionRoutes() {
    emptyRoutes.clear();
    nbMoves = 0;

    for (int r = 0; r < static_cast<int>(solRoutes.size()); r++) {
        Node* dep    = &lsDepots[r];
        Node* depEnd = &lsDepotsEnd[r];
        dep->prev    = depEnd;
        depEnd->next = dep;

        if (!solRoutes[r].empty()) {
            Node* cur = &nodesCli[solRoutes[r][0]];
            cur->route = &lsRoutes[r];
            cur->prev  = dep;
            dep->next  = cur;
            for (int i = 1; i < static_cast<int>(solRoutes[r].size()); i++) {
                Node* pred = cur;
                cur = &nodesCli[solRoutes[r][i]];
                cur->prev  = pred;
                pred->next = cur;
                cur->route = &lsRoutes[r];
            }
            cur->next    = depEnd;
            depEnd->prev = cur;
        } else {
            dep->next    = depEnd;
            depEnd->prev = dep;
        }
        updateRouteData(&lsRoutes[r]);
        lsRoutes[r].whenLastTestedSWAPStar = -1;
    }

    // Explicitly process excess slots as clean empty paths
    for (int r = static_cast<int>(solRoutes.size()); r < nbVehicles; r++) {
        Node* dep    = &lsDepots[r];
        Node* depEnd = &lsDepotsEnd[r];
        dep->prev    = depEnd;
        depEnd->next = dep;
        dep->next    = depEnd;
        depEnd->prev = dep;
        updateRouteData(&lsRoutes[r]);
        lsRoutes[r].whenLastTestedSWAPStar = -1;
    }

    for (int i = 1; i <= nbClients; i++)
        nodesCli[i].whenLastTestedRI = -1;
}

// Recalculate cost, weight bounds, and geometry signatures for modified loops
void CVRPEngine::updateRouteData(Route* myRoute) {
    int    myplace = 0;
    double myload  = 0.0;
    double mytime  = 0.0;
    double myRev   = 0.0;
    double cumX    = 0.0, cumY = 0.0;

    Node* mynode = myRoute->depot;
    mynode->position = 0;
    mynode->cumulatedLoad = 0.0;
    mynode->cumulatedTime = 0.0;
    mynode->cumulatedReversalDistance = 0.0;

    bool first = true;
    while (!mynode->isDepot || first) {
        mynode = mynode->next;
        myplace++;
        mynode->position = myplace;
        myload += cli[mynode->cour].demand;
        mytime += d(mynode->prev->cour, mynode->cour) + cli[mynode->cour].serviceDuration;
        myRev  += d(mynode->cour, mynode->prev->cour) - d(mynode->prev->cour, mynode->cour);
        mynode->cumulatedLoad             = myload;
        mynode->cumulatedTime             = mytime;
        mynode->cumulatedReversalDistance  = myRev;
        if (!mynode->isDepot) {
            cumX += coordX[mynode->cour];
            cumY += coordY[mynode->cour];
            if (first) myRoute->sector.initialize(cli[mynode->cour].polarAngle);
            else       myRoute->sector.extend(cli[mynode->cour].polarAngle);
        }
        first = false;
    }

    myRoute->duration          = mytime;
    myRoute->load              = myload;
    myRoute->penalty           = penExDur(mytime) + penExLoad(myload);
    myRoute->nbCustomers       = myplace - 1;
    myRoute->reversalDistance   = myRev;
    myRoute->whenLastModified  = nbMoves;

    if (myRoute->nbCustomers == 0) {
        myRoute->polarAngleBarycenter = 1.e30;
        emptyRoutes.insert(myRoute->cour);
    } else {
        myRoute->polarAngleBarycenter = std::atan2(
            cumY / myRoute->nbCustomers - coordY[0],
            cumX / myRoute->nbCustomers - coordX[0]);
        emptyRoutes.erase(myRoute->cour);
    }
}

void CVRPEngine::insertNodeLS(Node* U, Node* V) {
    U->prev->next = U->next;
    U->next->prev = U->prev;
    V->next->prev = U;
    U->prev  = V;
    U->next  = V->next;
    V->next  = U;
    U->route = V->route;
}

void CVRPEngine::swapNodeLS(Node* U, Node* V) {
    Node*  vp = V->prev;  Node*  vs = V->next;
    Node*  up = U->prev;  Node*  us = U->next;
    Route* rU = U->route; Route* rV = V->route;

    up->next = V;  us->prev = V;
    vp->next = U;  vs->prev = U;

    U->prev = vp;  U->next = vs;
    V->prev = up;  V->next = us;

    U->route = rV;
    V->route = rU;
}

void CVRPEngine::setVarsU() {
    routeU   = nodeU->route;
    nodeX    = nodeU->next;
    nXNext   = nodeX->next->cour;
    nU       = nodeU->cour;
    nUPrev   = nodeU->prev->cour;
    nX       = nodeX->cour;
    ldU      = cli[nU].demand;
    svU      = cli[nU].serviceDuration;
    ldX      = cli[nX].demand;
    svX      = cli[nX].serviceDuration;
}

void CVRPEngine::setVarsV() {
    routeV    = nodeV->route;
    nodeY     = nodeV->next;
    nYNext    = nodeY->next->cour;
    nV        = nodeV->cour;
    nVPrev    = nodeV->prev->cour;
    nY        = nodeY->cour;
    ldV       = cli[nV].demand;
    svV       = cli[nV].serviceDuration;
    ldY       = cli[nY].demand;
    svY       = cli[nY].serviceDuration;
    intraRoute = (routeU == routeV);
}

// ══════════════════════════════════════════════════════════════════════════════
//  Local Search Operators
// ══════════════════════════════════════════════════════════════════════════════

// Move 1: Relocate node U right after V
bool CVRPEngine::move1() {
    double costU = d(nUPrev, nX) - d(nUPrev, nU) - d(nU, nX);
    double costV = d(nV, nU) + d(nU, nY) - d(nV, nY);

    if (!intraRoute) {
        if (exceedsCap(routeV->load + ldU)) return false;
        if (costU + costV >= routeU->penalty + routeV->penalty) return false;
        costU += penExDur(routeU->duration + costU - svU) + penExLoad(routeU->load - ldU) - routeU->penalty;
        costV += penExDur(routeV->duration + costV + svU) + penExLoad(routeV->load + ldU) - routeV->penalty;
    }
    if (costU + costV > -MY_EPSILON) return false;
    if (nU == nY) return false;

    insertNodeLS(nodeU, nodeV);
    nbMoves++;
    searchCompleted = false;
    updateRouteData(routeU);
    if (!intraRoute) updateRouteData(routeV);
    return true;
}

// Move 2: Relocate adjacent pair (U, X) right after V, preserving order
bool CVRPEngine::move2() {
    double costU = d(nUPrev, nXNext) - d(nUPrev, nU) - d(nX, nXNext);
    double costV = d(nV, nU) + d(nX, nY) - d(nV, nY);

    if (!intraRoute) {
        if (exceedsCap(routeV->load + ldU + ldX)) return false;
        if (costU + costV >= routeU->penalty + routeV->penalty) return false;
        costU += penExDur(routeU->duration + costU - d(nU, nX) - svU - svX) + penExLoad(routeU->load - ldU - ldX) - routeU->penalty;
        costV += penExDur(routeV->duration + costV + d(nU, nX) + svU + svX) + penExLoad(routeV->load + ldU + ldX) - routeV->penalty;
    }
    if (costU + costV > -MY_EPSILON) return false;
    if (nodeU == nodeY || nodeV == nodeX || nodeX->isDepot) return false;

    insertNodeLS(nodeU, nodeV);
    insertNodeLS(nodeX, nodeU);
    nbMoves++;
    searchCompleted = false;
    updateRouteData(routeU);
    if (!intraRoute) updateRouteData(routeV);
    return true;
}

// Move 3: Relocate adjacent pair (U, X) right after V, reversing their order
bool CVRPEngine::move3() {
    double costU = d(nUPrev, nXNext) - d(nUPrev, nU) - d(nU, nX) - d(nX, nXNext);
    double costV = d(nV, nX) + d(nX, nU) + d(nU, nY) - d(nV, nY);

    if (!intraRoute) {
        if (exceedsCap(routeV->load + ldU + ldX)) return false;
        if (costU + costV >= routeU->penalty + routeV->penalty) return false;
        costU += penExDur(routeU->duration + costU - svU - svX) + penExLoad(routeU->load - ldU - ldX) - routeU->penalty;
        costV += penExDur(routeV->duration + costV + svU + svX) + penExLoad(routeV->load + ldU + ldX) - routeV->penalty;
    }
    if (costU + costV > -MY_EPSILON) return false;
    if (nodeU == nodeY || nodeX == nodeV || nodeX->isDepot) return false;

    insertNodeLS(nodeX, nodeV);
    insertNodeLS(nodeU, nodeX);
    nbMoves++;
    searchCompleted = false;
    updateRouteData(routeU);
    if (!intraRoute) updateRouteData(routeV);
    return true;
}

// Move 4: Swap single node locations of U and V
bool CVRPEngine::move4() {
    double costU = d(nUPrev, nV) + d(nV, nX) - d(nUPrev, nU) - d(nU, nX);
    double costV = d(nVPrev, nU) + d(nU, nY) - d(nVPrev, nV) - d(nV, nY);

    if (!intraRoute) {
        if (exceedsCap(routeU->load + ldV - ldU) || exceedsCap(routeV->load + ldU - ldV)) return false;
        if (costU + costV >= routeU->penalty + routeV->penalty) return false;
        costU += penExDur(routeU->duration + costU + svV - svU) + penExLoad(routeU->load + ldV - ldU) - routeU->penalty;
        costV += penExDur(routeV->duration + costV - svV + svU) + penExLoad(routeV->load + ldU - ldV) - routeV->penalty;
    }
    if (costU + costV > -MY_EPSILON) return false;
    if (nU == nVPrev || nU == nY) return false;

    swapNodeLS(nodeU, nodeV);
    nbMoves++;
    searchCompleted = false;
    updateRouteData(routeU);
    if (!intraRoute) updateRouteData(routeV);
    return true;
}

// Move 5: Swap adjacent pair (U, X) with a single node V
bool CVRPEngine::move5() {
    double costU = d(nUPrev, nV) + d(nV, nXNext) - d(nUPrev, nU) - d(nX, nXNext);
    double costV = d(nVPrev, nU) + d(nX, nY) - d(nVPrev, nV) - d(nV, nY);

    if (!intraRoute) {
        if (exceedsCap(routeU->load + ldV - ldU - ldX) || exceedsCap(routeV->load + ldU + ldX - ldV)) return false;
        if (costU + costV >= routeU->penalty + routeV->penalty) return false;
        costU += penExDur(routeU->duration + costU - d(nU, nX) + svV - svU - svX) + penExLoad(routeU->load + ldV - ldU - ldX) - routeU->penalty;
        costV += penExDur(routeV->duration + costV + d(nU, nX) - svV + svU + svX) + penExLoad(routeV->load + ldU + ldX - ldV) - routeV->penalty;
    }
    if (costU + costV > -MY_EPSILON) return false;
    if (nodeU == nodeV->prev || nodeX == nodeV->prev || nodeU == nodeY || nodeX->isDepot) return false;

    swapNodeLS(nodeU, nodeV);
    insertNodeLS(nodeX, nodeU);
    nbMoves++;
    searchCompleted = false;
    updateRouteData(routeU);
    if (!intraRoute) updateRouteData(routeV);
    return true;
}

// Move 6: Swap adjacent pair (U, X) with adjacent pair (V, Y)
bool CVRPEngine::move6() {
    double costU = d(nUPrev, nV) + d(nY, nXNext) - d(nUPrev, nU) - d(nX, nXNext);
    double costV = d(nVPrev, nU) + d(nX, nYNext) - d(nVPrev, nV) - d(nY, nYNext);

    if (!intraRoute) {
        if (exceedsCap(routeU->load + ldV + ldY - ldU - ldX) || exceedsCap(routeV->load + ldU + ldX - ldV - ldY)) return false;
        if (costU + costV >= routeU->penalty + routeV->penalty) return false;
        costU += penExDur(routeU->duration + costU - d(nU, nX) + d(nV, nY) + svV + svY - svU - svX) + penExLoad(routeU->load + ldV + ldY - ldU - ldX) - routeU->penalty;
        costV += penExDur(routeV->duration + costV + d(nU, nX) - d(nV, nY) - svV - svY + svU + svX) + penExLoad(routeV->load + ldU + ldX - ldV - ldY) - routeV->penalty;
    }
    if (costU + costV > -MY_EPSILON) return false;
    if (nodeX->isDepot || nodeY->isDepot || nodeY == nodeU->prev || nodeU == nodeY || nodeX == nodeV || nodeV == nodeX->next) return false;

    swapNodeLS(nodeU, nodeV);
    swapNodeLS(nodeX, nodeY);
    nbMoves++;
    searchCompleted = false;
    updateRouteData(routeU);
    if (!intraRoute) updateRouteData(routeV);
    return true;
}

// Move 7: Localized 2-opt sequence reversal inside a single route
bool CVRPEngine::move7() {
    if (nodeU->position > nodeV->position) return false;

    double cost = d(nU, nV) + d(nX, nY) - d(nU, nX) - d(nV, nY) + nodeV->cumulatedReversalDistance - nodeX->cumulatedReversalDistance;

    if (cost > -MY_EPSILON) return false;
    if (nodeU->next == nodeV) return false;

    Node* num = nodeX->next;
    nodeX->prev = num;
    nodeX->next = nodeY;

    while (num != nodeV) {
        Node* tmp = num->next;
        num->next = num->prev;
        num->prev = tmp;
        num = tmp;
    }
    nodeV->next = nodeV->prev;
    nodeV->prev = nodeU;
    nodeU->next = nodeV;
    nodeY->prev = nodeX;

    nbMoves++;
    searchCompleted = false;
    updateRouteData(routeU);
    return true;
}

// Move 8: Inter-route 2-opt step involving sub-path inversion
bool CVRPEngine::move8() {
    double cost = d(nU, nV) + d(nX, nY) - d(nU, nX) - d(nV, nY) + nodeV->cumulatedReversalDistance + routeU->reversalDistance - nodeX->cumulatedReversalDistance - routeU->penalty - routeV->penalty;

    if (cost >= 0) return false;

    double newLoadA = nodeU->cumulatedLoad + nodeV->cumulatedLoad;
    double newLoadB = routeU->load + routeV->load - nodeU->cumulatedLoad - nodeV->cumulatedLoad;
    if (exceedsCap(newLoadA) || exceedsCap(newLoadB)) return false;

    cost += penExDur(nodeU->cumulatedTime + nodeV->cumulatedTime + nodeV->cumulatedReversalDistance + d(nU, nV))
          + penExDur(routeU->duration - nodeU->cumulatedTime - d(nU, nX) + routeU->reversalDistance - nodeX->cumulatedReversalDistance + routeV->duration - nodeV->cumulatedTime - d(nV, nY) + d(nX, nY))
          + penExLoad(newLoadA) + penExLoad(newLoadB);

    if (cost > -MY_EPSILON) return false;

    Node* depU    = routeU->depot;
    Node* depV    = routeV->depot;
    Node* depUFin = depU->prev;
    Node* depVFin = depV->prev;
    Node* depVS   = depV->next;

    Node* xx = nodeX;
    while (!xx->isDepot) {
        Node* tmp = xx->next;
        xx->next  = xx->prev;
        xx->prev  = tmp;
        xx->route = routeV;
        xx = tmp;
    }
    Node* vv = nodeV;
    while (!vv->isDepot) {
        Node* tmp = vv->prev;
        vv->prev  = vv->next;
        vv->next  = tmp;
        vv->route = routeU;
        vv = tmp;
    }

    nodeU->next = nodeV;  nodeV->prev = nodeU;
    nodeX->next = nodeY;  nodeY->prev = nodeX;

    if (nodeX->isDepot) {
        depUFin->next = depU;
        depUFin->prev = depVS;
        depUFin->prev->next = depUFin;
        depV->next = nodeY;
        nodeY->prev = depV;
    } else if (nodeV->isDepot) {
        depV->next = depUFin->prev;
        depV->next->prev = depV;
        depV->prev = depVFin;
        depUFin->prev = nodeU;
        nodeU->next = depUFin;
    } else {
        depV->next = depUFin->prev;
        depV->next->prev = depV;
        depUFin->prev = depVS;
        depUFin->prev->next = depUFin;
    }

    nbMoves++;
    searchCompleted = false;
    updateRouteData(routeU);
    updateRouteData(routeV);
    return true;
}

// Move 9: Inter-route 2-opt cross step without tracking inversion
bool CVRPEngine::move9() {
    double cost = d(nU, nY) + d(nV, nX) - d(nU, nX) - d(nV, nY) - routeU->penalty - routeV->penalty;

    if (cost >= 0) return false;

    double newLoadA = nodeU->cumulatedLoad + routeV->load - nodeV->cumulatedLoad;
    double newLoadB = nodeV->cumulatedLoad + routeU->load - nodeU->cumulatedLoad;
    if (exceedsCap(newLoadA) || exceedsCap(newLoadB)) return false;

    cost += penExDur(nodeU->cumulatedTime + routeV->duration - nodeV->cumulatedTime - d(nV, nY) + d(nU, nY))
          + penExDur(routeU->duration - nodeU->cumulatedTime - d(nU, nX) + nodeV->cumulatedTime + d(nV, nX))
          + penExLoad(newLoadA) + penExLoad(newLoadB);

    if (cost > -MY_EPSILON) return false;

    Node* depU    = routeU->depot;
    Node* depV    = routeV->depot;
    Node* depUFin = depU->prev;
    Node* depVFin = depV->prev;
    Node* depUpred = depUFin->prev;

    Node* count = nodeY;
    while (!count->isDepot) { count->route = routeU; count = count->next; }
    count = nodeX;
    while (!count->isDepot) { count->route = routeV; count = count->next; }

    nodeU->next = nodeY;  nodeY->prev = nodeU;
    nodeV->next = nodeX;  nodeX->prev = nodeV;

    if (nodeX->isDepot) {
        depUFin->prev = depVFin->prev;
        depUFin->prev->next = depUFin;
        nodeV->next = depVFin;
        depVFin->prev = nodeV;
    } else {
        depUFin->prev = depVFin->prev;
        depUFin->prev->next = depUFin;
        depVFin->prev = depUpred;
        depVFin->prev->next = depVFin;
    }

    nbMoves++;
    searchCompleted = false;
    updateRouteData(routeU);
    updateRouteData(routeV);
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
//  SWAP* Engine
// ══════════════════════════════════════════════════════════════════════════════

void CVRPEngine::preprocessInsertions(Route* R1, Route* R2, std::vector<ThreeBestInsert>& insCache) {
    for (Node* U = R1->depot->next; !U->isDepot; U = U->next) {
        U->deltaRemoval = d(U->prev->cour, U->next->cour) - d(U->prev->cour, U->cour) - d(U->cour, U->next->cour);
        insCache[U->cour].reset();
        insCache[U->cour].bestCost[0] = d(0, U->cour) + d(U->cour, R2->depot->next->cour) - d(0, R2->depot->next->cour);
        insCache[U->cour].bestLocation[0] = R2->depot;
        for (Node* V = R2->depot->next; !V->isDepot; V = V->next) {
            float dc = d(V->cour, U->cour) + d(U->cour, V->next->cour) - d(V->cour, V->next->cour);
            insCache[U->cour].compareAndAdd(dc, V);
        }
    }
}

double CVRPEngine::cheapInsertSimultRemoval(
        Node* U, Node* V, Node*& bestPos, std::vector<ThreeBestInsert>& insCache) {
    ThreeBestInsert* bi = &insCache[U->cour];
    bool found = false;

    bestPos = bi->bestLocation[0];
    double bestCost = bi->bestCost[0];
    found = (bestPos != V && bestPos->next != V);
    if (!found && bi->bestLocation[1] != nullptr) {
        bestPos  = bi->bestLocation[1];
        bestCost = bi->bestCost[1];
        found = (bestPos != V && bestPos->next != V);
        if (!found && bi->bestLocation[2] != nullptr) {
            bestPos  = bi->bestLocation[2];
            bestCost = bi->bestCost[2];
            found = true;
        }
    }

    double dc = d(V->prev->cour, U->cour) + d(U->cour, V->next->cour) - d(V->prev->cour, V->next->cour);
    if (!found || dc < bestCost) {
        bestPos  = V->prev;
        bestCost = dc;
    }
    return bestCost;
}

// Loops through prospective swap operations to trade node configurations across paths
bool CVRPEngine::doSwapStar() {
    SwapStarElement best;

    preprocessInsertions(routeU, routeV, cacheUintoV);
    preprocessInsertions(routeV, routeU, cacheVintoU);

    // Evaluate swaps
    for (nodeU = routeU->depot->next; !nodeU->isDepot; nodeU = nodeU->next) {
        for (nodeV = routeV->depot->next; !nodeV->isDepot; nodeV = nodeV->next) {
            double newLoadU = routeU->load + cli[nodeV->cour].demand - cli[nodeU->cour].demand;
            double newLoadV = routeV->load + cli[nodeU->cour].demand - cli[nodeV->cour].demand;
            if (exceedsCap(newLoadU) || exceedsCap(newLoadV)) continue;
            double dpU = penExLoad(newLoadU) - routeU->penalty;
            double dpV = penExLoad(newLoadV) - routeV->penalty;
            if (dpU + nodeU->deltaRemoval + dpV + nodeV->deltaRemoval <= 0) {
                SwapStarElement el;
                el.U = nodeU;  el.V = nodeV;

                double extraV = cheapInsertSimultRemoval(nodeU, nodeV, el.bestPositionU, cacheUintoV);
                double extraU = cheapInsertSimultRemoval(nodeV, nodeU, el.bestPositionV, cacheVintoU);
                el.moveCost = dpU + nodeU->deltaRemoval + extraU
                            + dpV + nodeV->deltaRemoval + extraV
                            + penExDur(routeU->duration + nodeU->deltaRemoval + extraU + cli[nodeV->cour].serviceDuration - cli[nodeU->cour].serviceDuration)
                            + penExDur(routeV->duration + nodeV->deltaRemoval + extraV - cli[nodeV->cour].serviceDuration + cli[nodeU->cour].serviceDuration);

                if (el.moveCost < best.moveCost) best = el;
            }
        }
    }

    // Evaluate standalone shifts for U
    for (nodeU = routeU->depot->next; !nodeU->isDepot; nodeU = nodeU->next) {
        if (exceedsCap(routeV->load + cli[nodeU->cour].demand)) continue;
        SwapStarElement el;
        el.U = nodeU;
        el.bestPositionU = cacheUintoV[nodeU->cour].bestLocation[0];
        double ddU = d(nodeU->prev->cour, nodeU->next->cour) - d(nodeU->prev->cour, nodeU->cour) - d(nodeU->cour, nodeU->next->cour);
        double ddV = cacheUintoV[nodeU->cour].bestCost[0];
        el.moveCost = ddU + ddV
            + penExLoad(routeU->load - cli[nodeU->cour].demand) - routeU->penalty
            + penExLoad(routeV->load + cli[nodeU->cour].demand) - routeV->penalty
            + penExDur(routeU->duration + ddU - cli[nodeU->cour].serviceDuration)
            + penExDur(routeV->duration + ddV + cli[nodeU->cour].serviceDuration);
        if (el.moveCost < best.moveCost) best = el;
    }

    // Evaluate standalone shifts for V
    for (nodeV = routeV->depot->next; !nodeV->isDepot; nodeV = nodeV->next) {
        if (exceedsCap(routeU->load + cli[nodeV->cour].demand)) continue;
        SwapStarElement el;
        el.V = nodeV;
        el.bestPositionV = cacheVintoU[nodeV->cour].bestLocation[0];
        double ddU = cacheVintoU[nodeV->cour].bestCost[0];
        double ddV = d(nodeV->prev->cour, nodeV->next->cour) - d(nodeV->prev->cour, nodeV->cour) - d(nodeV->cour, nodeV->next->cour);
        el.moveCost = ddU + ddV
            + penExLoad(routeU->load + cli[nodeV->cour].demand) - routeU->penalty
            + penExLoad(routeV->load - cli[nodeV->cour].demand) - routeV->penalty
            + penExDur(routeU->duration + ddU + cli[nodeV->cour].serviceDuration)
            + penExDur(routeV->duration + ddV - cli[nodeV->cour].serviceDuration);
        if (el.moveCost < best.moveCost) best = el;
    }

    if (best.moveCost > -MY_EPSILON) return false;

    if (best.bestPositionU != nullptr) insertNodeLS(best.U, best.bestPositionU);
    if (best.bestPositionV != nullptr) insertNodeLS(best.V, best.bestPositionV);
    nbMoves++;
    searchCompleted = false;
    updateRouteData(routeU);
    updateRouteData(routeV);
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
//  Variable Neighbourhood Descent (VND)
// ══════════════════════════════════════════════════════════════════════════════
void CVRPEngine::runVND() {
    std::vector<int> orderNodes(nbClients);
    std::iota(orderNodes.begin(), orderNodes.end(), 1); 
    std::vector<int> orderRoutes(nbVehicles);
    std::iota(orderRoutes.begin(), orderRoutes.end(), 0);

    std::shuffle(orderNodes.begin(),  orderNodes.end(),  rng);
    std::shuffle(orderRoutes.begin(), orderRoutes.end(), rng);
    for (int i = 1; i <= nbClients; i++)
        if (rng() % NB_GRANULAR == 0)
            std::shuffle(correlated[i].begin(), correlated[i].end(), rng);

    searchCompleted = false;
    int totalMovesBefore = nbMoves;

    for (int loopID = 0; !searchCompleted; loopID++) {
        if (loopID > 1) searchCompleted = true;

        // Phase 1: Try moves using localized neighbor pools
        for (int posU = 0; posU < nbClients; posU++) {
            nodeU = &nodesCli[orderNodes[posU]];
            int lastTest = nodeU->whenLastTestedRI;
            nodeU->whenLastTestedRI = nbMoves;

            for (int posV = 0; posV < static_cast<int>(correlated[nodeU->cour].size()); posV++) {
                nodeV = &nodesCli[correlated[nodeU->cour][posV]];
                if (loopID == 0 || std::max(nodeU->route->whenLastModified, nodeV->route->whenLastModified) > lastTest) {
                    setVarsU();
                    setVarsV();
                    if (move1()) continue;
                    if (move2()) continue;
                    if (move3()) continue;
                    if (nU <= nV && move4()) continue;
                    if (move5()) continue;
                    if (nU <= nV && move6()) continue;
                    if (intraRoute  && move7()) continue;
                    if (!intraRoute && move8()) continue;
                    if (!intraRoute && move9()) continue;

                    if (nodeV->prev->isDepot) {
                        nodeV = nodeV->prev;
                        setVarsV();
                        if (move1()) continue;
                        if (move2()) continue;
                        if (move3()) continue;
                        if (!intraRoute && move8()) continue;
                        if (!intraRoute && move9()) continue;
                    }
                }
            }

            if (loopID > 0 && !emptyRoutes.empty()) {
                nodeV = lsRoutes[*emptyRoutes.begin()].depot;
                setVarsU();
                setVarsV();
                if (move1()) continue;
                if (move2()) continue;
                if (move3()) continue;
                if (move9()) continue;
            }
        }

        // Phase 2: Run wider SWAP* checks across intersecting spatial sectors
        for (int rU = 0; rU < nbVehicles; rU++) {
            routeU = &lsRoutes[orderRoutes[rU]];
            int lastSS = routeU->whenLastTestedSWAPStar;
            routeU->whenLastTestedSWAPStar = nbMoves;
            for (int rV = 0; rV < nbVehicles; rV++) {
                routeV = &lsRoutes[orderRoutes[rV]];
                if (routeU->nbCustomers > 0 && routeV->nbCustomers > 0
                    && routeU->cour < routeV->cour
                    && (loopID == 0 || std::max(routeU->whenLastModified, routeV->whenLastModified) > lastSS))
                {
                    if (CircleSector::overlap(routeU->sector, routeV->sector))
                        doSwapStar();
                }
            }
        }
    }

    if (verbose) std::cerr << "[vnd]   Applied " << (nbMoves - totalMovesBefore) << " improving moves\n";
}

// Re-sort routes systematically using polar coordinates
CVRPEngine::SolOutput CVRPEngine::exportSolution() {
    SolOutput sol;

    std::vector<std::pair<double,int>> angles;
    for (int r = 0; r < nbVehicles; r++)
        angles.emplace_back(lsRoutes[r].polarAngleBarycenter, r);
    std::sort(angles.begin(), angles.end());

    double totalDist = 0.0;
    double totalCapEx = 0.0;
    for (int r = 0; r < nbVehicles; r++) {
        int ri = angles[r].second;
        std::vector<int> route;
        Node* nd = lsDepots[ri].next;
        while (!nd->isDepot) {
            route.push_back(nd->cour);
            nd = nd->next;
        }
        if (!route.empty()) {
            double rd = d(0, route[0]);
            double rl = cli[route[0]].demand;
            for (int i = 1; i < static_cast<int>(route.size()); i++) {
                rd += d(route[i-1], route[i]);
                rl += cli[route[i]].demand;
            }
            rd += d(route.back(), 0);
            totalDist += rd;
            if (rl > vehicleCap) totalCapEx += rl - vehicleCap;
            sol.routes.push_back(std::move(route));
        }
    }

    sol.totalDist = static_cast<int>(std::round(totalDist));
    sol.capExcess = totalCapEx;
    sol.feasible  = (totalCapEx < MY_EPSILON);
    return sol;
}

void CVRPEngine::writeSolution(const std::string& path, const SolOutput& sol) {
    std::ofstream out(path);
    if (!out.is_open()) throw std::string("Cannot open output file: " + path);

    for (int r = 0; r < static_cast<int>(sol.routes.size()); r++) {
        out << "Route #" << (r + 1) << ":";
        for (int c : sol.routes[r]) out << " " << c;
        out << "\n";
    }
    out << "Cost " << sol.totalDist << "\n";
    out.close();
}

void CVRPEngine::writeVRP(const std::string& path, const std::string& opDesc) {
    std::ofstream out(path);
    if (!out.is_open()) throw std::string("Cannot open output VRP file: " + path);

    out << "NAME : " << vrpName << "-updated\n";
    out << "COMMENT : " << vrpComment << " (" << opDesc << ")\n";
    out << "TYPE : " << vrpType << "\n";
    out << "DIMENSION : " << (nbClients + 1) << "\n";
    out << "EDGE_WEIGHT_TYPE : " << edgeWeightType << "\n";
    out << "CAPACITY : " << vehicleCap << "\n";
    if (durationLimit < 1.e29) {
        out << "DISTANCE : " << durationLimit << "\n";
    }
    out << "NODE_COORD_SECTION\n";
    for (int i = 0; i <= nbClients; i++) {
        out << " " << (i + 1) << " " << static_cast<int>(std::round(coordX[i]))
            << " " << static_cast<int>(std::round(coordY[i])) << "\n";
    }
    out << "DEMAND_SECTION\n";
    for (int i = 0; i <= nbClients; i++) {
        out << (i + 1) << " " << static_cast<int>(std::round(cli[i].demand)) << "\n";
    }
    out << "DEPOT_SECTION\n";
    out << " 1\n";
    out << " -1\n";
    out << "EOF\n";
    out.close();
}

void CVRPEngine::printAndVerify(const SolOutput& sol) {
    if (verbose) {
        std::cerr << "[done]  Routes: " << sol.routes.size()
                  << "  Cost: " << sol.totalDist
                  << "  Feasible: " << (sol.feasible ? "YES" : "NO");
        if (!sol.feasible)
            std::cerr << "  (cap excess=" << sol.capExcess << ")";
        std::cerr << "\n";
    }

    if (!sol.feasible) {
        if (verbose) std::cerr << "[WARN]  Solution has capacity violations (excess="
                  << sol.capExcess << "). This should not happen.\n";
    }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Inserter Engine (Implementation)
// ══════════════════════════════════════════════════════════════════════════════

void Inserter::addNewNode(double x, double y, double demand) {
    nbClients++;
    newNodeIdx = nbClients;

    Client c;
    c.demand = demand;
    c.serviceDuration = 0.0;
    c.polarAngle = CircleSector::positive_mod(static_cast<int>(32768.0 * std::atan2(y - coordY[0], x - coordX[0]) / PI_VAL));
    cli.push_back(c);

    coordX.push_back(x);
    coordY.push_back(y);

    if (verbose) std::cerr << "[node]  Added new customer " << newNodeIdx << " at (" << x << "," << y << ") demand=" << demand << "\n";
}

// Insert the newly added node into a clean spare path slot
void Inserter::loadSolutionWithSpare() {
    loadSolutionRoutes();  

    int spareR = static_cast<int>(solRoutes.size());
    Node* dep    = &lsDepots[spareR];
    Node* depEnd = &lsDepotsEnd[spareR];

    Node* nn = &nodesCli[newNodeIdx];
    nn->route  = &lsRoutes[spareR];
    nn->prev   = dep;
    dep->next  = nn;
    nn->next   = depEnd;
    depEnd->prev = nn;

    updateRouteData(&lsRoutes[spareR]);
}

void Inserter::run(const std::string& vrpFile,
                   const std::string& solFile,
                   const std::string& outSolFile,
                   const std::string& outVrpFile,
                   double newX, double newY, double newDemand) {

    auto time1 = std::chrono::high_resolution_clock::now();

    parseVRP(vrpFile);
    parseSolution(solFile);
    int initialNodes = nbClients + 1;  

    auto time2 = std::chrono::high_resolution_clock::now();

    addNewNode(newX, newY, newDemand);

    auto time3 = std::chrono::high_resolution_clock::now();

    computeCorrelated();

    auto time4 = std::chrono::high_resolution_clock::now();

    allocateLS(static_cast<int>(solRoutes.size()) + 1);

    auto time5 = std::chrono::high_resolution_clock::now();

    loadSolutionWithSpare();

    auto time6 = std::chrono::high_resolution_clock::now();

    if (verbose) std::cerr << "[vnd]   Starting Variable Neighbourhood Descent...\n";
    runVND();

    auto time7 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_compute = time7 - time2;
    std::chrono::duration<double> parse_time = time2 - time1;
    std::chrono::duration<double> add_node_time = time3 - time2;
    std::chrono::duration<double> compute_correlated_time = time4 - time3;
    std::chrono::duration<double> allocate_ls_time = time5 - time4;
    std::chrono::duration<double> load_solution_time = time6 - time5;
    std::chrono::duration<double> run_vnd_time = time7 - time6;

    if (timecheck) {
        bool file_exists = false;
        {
            std::ifstream check("timecheck_LARGE.csv");
            file_exists = check.good();
        }
        std::ofstream csv("timecheck_LARGE.csv", std::ios::app);
        if (!file_exists) {
            csv << "initial_nodes,mode,total_compute_excluding_parse,parse,add/delete_node,compute_correlated,allocate_ls,load_solution,run_vnd\n";
        }
        csv << initialNodes << ",add,"
            << total_compute.count() << ","
            << parse_time.count() << ","
            << add_node_time.count() << ","
            << compute_correlated_time.count() << ","
            << allocate_ls_time.count() << ","
            << load_solution_time.count() << ","
            << run_vnd_time.count() << "\n";
    }

    auto sol = exportSolution();
    printAndVerify(sol);

    writeSolution(outSolFile, sol);
    if (verbose) std::cerr << "[file]  Solution written to " << outSolFile << "\n";

    writeVRP(outVrpFile, "with 1 dynamic node added");
    if (verbose) std::cerr << "[file]  Updated VRP written to " << outVrpFile << "\n";
}

// ══════════════════════════════════════════════════════════════════════════════
//  Deleter Engine (Implementation)
// ══════════════════════════════════════════════════════════════════════════════

void Deleter::removeNodeFromSolution(int nodeId) {
    bool found = false;
    for (int r = 0; r < static_cast<int>(solRoutes.size()); r++) {
        auto& route = solRoutes[r];
        for (auto it = route.begin(); it != route.end(); ++it) {
            if (*it == nodeId) {
                route.erase(it);
                found = true;
                break;
            }
        }
        if (found) {
            if (route.empty()) {
                if (verbose) std::cerr << "[delete] Route #" << (r + 1) << " is now empty — removing it\n";
                solRoutes.erase(solRoutes.begin() + r);
            }
            break;
        }
    }
    if (!found) {
        throw std::string("Node " + std::to_string(nodeId) + " not found in any route of the solution");
    }

    if (solRoutes.empty()) {
        throw std::string("Solution is empty after deleting node " + std::to_string(nodeId) + " — no routes remain");
    }
}

// Clear out old nodes and compress indices to maintain dense contiguous storage
void Deleter::removeNodeFromProblem(int nodeId) {
    for (auto& route : solRoutes) {
        for (auto& c : route) {
            if (c > nodeId) c--;
        }
    }

    cli.erase(cli.begin() + nodeId);
    coordX.erase(coordX.begin() + nodeId);
    coordY.erase(coordY.begin() + nodeId);

    nbClients--;

    if (verbose) std::cerr << "[delete] Node " << nodeId << " removed from problem data. " << nbClients << " clients remain.\n";
}

void Deleter::run(const std::string& vrpFile,
                  const std::string& solFile,
                  const std::string& outSolFile,
                  const std::string& outVrpFile,
                  int deleteNodeId) {
    auto time1 = std::chrono::high_resolution_clock::now();
    
    parseVRP(vrpFile);
    parseSolution(solFile);
    int initialNodes = nbClients + 1;  

    auto time2 = std::chrono::high_resolution_clock::now();
    
    if (deleteNodeId < 1 || deleteNodeId > nbClients) {
        throw std::string("Invalid node ID " + std::to_string(deleteNodeId) + ". Must be in range [1, " + std::to_string(nbClients) + "]");
    }

    if (verbose) std::cerr << "[delete] Deleting customer " << deleteNodeId
              << " (coords: " << coordX[deleteNodeId] << "," << coordY[deleteNodeId]
              << ", demand=" << cli[deleteNodeId].demand << ")\n";

    removeNodeFromSolution(deleteNodeId);
    removeNodeFromProblem(deleteNodeId);

    auto time3 = std::chrono::high_resolution_clock::now();

    computeCorrelated();

    auto time4 = std::chrono::high_resolution_clock::now();
    
    allocateLS(static_cast<int>(solRoutes.size()));

    auto time5 = std::chrono::high_resolution_clock::now();
    
    loadSolutionRoutes();

    if (verbose) std::cerr << "[vnd]   Starting Variable Neighbourhood Descent...\n";

    auto time6 = std::chrono::high_resolution_clock::now();

    runVND();

    auto time7 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> parse_time = time2 - time1;
    std::chrono::duration<double> delete_node_time = time3 - time2;
    std::chrono::duration<double> compute_correlated_time = time4 - time3;
    std::chrono::duration<double> allocate_ls_time = time5 - time4;
    std::chrono::duration<double> load_solution_time = time6 - time5;
    std::chrono::duration<double> run_vnd_time = time7 - time6;
    std::chrono::duration<double> total_compute = time7 - time2;

    if (timecheck) {
        bool file_exists = false;
        {
            std::ifstream check("timecheck_LARGE.csv");
            file_exists = check.good();
        }
        std::ofstream csv("timecheck_LARGE.csv", std::ios::app);
        if (!file_exists) {
            csv << "initial_nodes,mode,total_compute_excluding_parse,parse,add/delete_node,compute_correlated,allocate_ls,load_solution,run_vnd\n";
        }
        csv << initialNodes << ",delete,"
            << total_compute.count() << ","
            << parse_time.count() << ","
            << delete_node_time.count() << ","
            << compute_correlated_time.count() << ","
            << allocate_ls_time.count() << ","
            << load_solution_time.count() << ","
            << run_vnd_time.count() << "\n";
    }

    auto sol = exportSolution();
    printAndVerify(sol);

    writeSolution(outSolFile, sol);
    if (verbose) std::cerr << "[file]  Solution written to " << outSolFile << "\n";

    writeVRP(outVrpFile, "with 1 node removed");
    if (verbose) std::cerr << "[file]  Updated VRP written to " << outVrpFile << "\n";
}

// ══════════════════════════════════════════════════════════════════════════════
//  Main Routing Entry Point
// ══════════════════════════════════════════════════════════════════════════════
static void printUsage() {
    std::cerr << "Usage:\n"
              << "  Addition:  dynamic_cvrp a <instance.vrp> <solution.sol> "
                 "<output.sol> <output.vrp> <X> <Y> <demand>\n"
              << "  Deletion:  dynamic_cvrp d <instance.vrp> <solution.sol> "
                 "<output.sol> <output.vrp> <node_id>\n"
              << "\n"
              << "  Inserts or removes a customer node from an existing CVRP\n"
              << "  solution and optimises via granular VND local search.\n"
              << "\n"
              << "  For addition, the new node is automatically assigned the\n"
              << "  next available customer ID (= original nbClients + 1).\n"
              << "\n"
              << "  For deletion, <node_id> is the customer ID as it appears\n"
              << "  in the solution file (1-based).\n";
}

int main(int argc, char* argv[]) {
    auto start_time = std::chrono::high_resolution_clock::now();
    bool verbose = false;
    bool timecheck = false;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-v") verbose = true;
        else if (arg == "-t") timecheck = true;
        else args.push_back(arg);
    }

    if (args.empty()) {
        printUsage();
        return 1;
    }

    std::string mode = args[0];

    try {
        if (mode == "a") {
            if (args.size() != 8) {
                std::cerr << "ERROR: Addition mode requires 7 arguments after 'a'.\n\n";
                printUsage();
                return 1;
            }
            Inserter ins;
            ins.verbose = verbose;
            ins.timecheck = timecheck;
            ins.run(args[1], args[2], args[3], args[4], std::stod(args[5]), std::stod(args[6]), std::stod(args[7]));
        }
        else if (mode == "d") {
            if (args.size() != 6) {
                std::cerr << "ERROR: Deletion mode requires 4 arguments after 'd'.\n\n";
                printUsage();
                return 1;
            }
            Deleter del;
            del.verbose = verbose;
            del.timecheck = timecheck;
            del.run(args[1], args[2], args[3], args[4], std::stoi(args[5]));
        }
        else {
            std::cerr << "ERROR: Unknown mode '" << mode << "'. Use 'a' for addition or 'd' for deletion.\n\n";
            printUsage();
            return 1;
        }
    }
    catch (const std::string& e) {
        std::cerr << "ERROR: " << e << "\n";
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    if (verbose) {
        std::cerr << "[time]  Total time: " << diff.count() << " seconds\n";
    }

    return 0;
}