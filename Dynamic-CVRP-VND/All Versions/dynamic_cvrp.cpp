// ═══════════════════════════════════════════════════════════════════════════════
// dynamic_cvrp.cpp — Standalone C++17 Dynamic CVRP Node Insertion & Deletion
//
// Reads a CVRP instance (.vrp) and an existing solution (.sol), adds or removes
// a single customer node, and optimises the updated solution via a granular
// Variable Neighbourhood Descent (VND) local search.
//
// All local-search algorithms are adapted from the HGS-CVRP codebase
// (Vidal, 2020) — this file carries NO dependency on HGS source files.
//
// Usage:
//   Addition:  dynamic_cvrp a <instance.vrp> <solution.sol> <output.sol> <output.vrp> <X> <Y> <demand>
//   Deletion:  dynamic_cvrp d <instance.vrp> <solution.sol> <output.sol> <output.vrp> <node_id>
//
// Compile:
//   g++ -std=c++17 -O2 -o dynamic_cvrp dynamic_cvrp.cpp
//   cl /std:c++17 /O2 /EHsc dynamic_cvrp.cpp          (MSVC)
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

// ──────────────────────────── Constants ───────────────────────────────────────
static constexpr double MY_EPSILON  = 0.00001;
static constexpr double PI_VAL      = 3.14159265359;
static constexpr int    NB_GRANULAR = 20;  // k-nearest for granular filtering

// ══════════════════════════════════════════════════════════════════════════════
//  Data Structures — lightweight copies of HGS types (no HGS #include needed)
// ══════════════════════════════════════════════════════════════════════════════

// ─── CircleSector (HGS-CVRP/Program/CircleSector.h) ─────────────────────────
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

// ─── Client (HGS-CVRP/Program/Params.h) ─────────────────────────────────────
struct Client {
    double coordX         = 0.0;
    double coordY         = 0.0;
    double serviceDuration = 0.0;
    double demand         = 0.0;
    int    polarAngle     = 0;     // angle around depot, mapped to [0,65535]
};

// Forward declaration
struct Route;

// ─── Node — linked-list element (HGS-CVRP/Program/LocalSearch.h) ────────────
struct Node {
    bool   isDepot   = false;
    int    cour      = 0;          // index: 0 = depot, 1..n = customers
    int    position  = 0;          // ordinal position inside the route
    int    whenLastTestedRI = -1;
    Node*  next      = nullptr;
    Node*  prev      = nullptr;
    Route* route     = nullptr;
    double cumulatedLoad = 0.0;
    double cumulatedTime = 0.0;
    double cumulatedReversalDistance = 0.0;
    double deltaRemoval = 0.0;     // used by SWAP*
};

// ─── Route (HGS-CVRP/Program/LocalSearch.h) ─────────────────────────────────
struct Route {
    int    cour       = 0;         // route index
    int    nbCustomers = 0;
    int    whenLastModified = 0;
    int    whenLastTestedSWAPStar = -1;
    Node*  depot      = nullptr;   // points to the depot sentinel for this route
    double duration   = 0.0;
    double load       = 0.0;
    double reversalDistance = 0.0;
    double penalty    = 0.0;
    double polarAngleBarycenter = 0.0;
    CircleSector sector;
};

// ─── ThreeBestInsert — top-3 insertion cache for SWAP* ──────────────────────
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

// ─── SwapStarElement — tracks the best SWAP* move ───────────────────────────
struct SwapStarElement {
    double moveCost      = 1.e30;
    Node*  U             = nullptr;
    Node*  bestPositionU = nullptr;
    Node*  V             = nullptr;
    Node*  bestPositionV = nullptr;
};


// ══════════════════════════════════════════════════════════════════════════════
//  CVRPEngine — shared base class holding problem data, local search,
//               parsing, and export logic used by both Inserter and Deleter
// ══════════════════════════════════════════════════════════════════════════════
class CVRPEngine {
public:
    bool verbose = false;
    bool timecheck = false;
    virtual ~CVRPEngine() = default;

protected:
    // ────── Problem data ────────────────────────────────────────────────────

    int    nbClients  = 0;               // customers, excl. depot
    int    nbVehicles = 0;               // total route slots
    double vehicleCap = 0.0;
    double durationLimit = 1.e30;
    double maxDist = 0.0, maxDemand = 0.0;

    std::vector<Client>              cli;       // [0] = depot, [1..n]
    std::vector<std::vector<double>> dist;      // dist[i][j], 0 = depot
    std::vector<std::vector<int>>    correlated; // granular neighbours

    // VRP file metadata for writing output VRP
    std::string vrpName = "";
    std::string vrpComment = "";
    std::string vrpType = "CVRP";
    std::string edgeWeightType = "EUC_2D";

    // ────── Parsed solution ─────────────────────────────────────────────────
    std::vector<std::vector<int>> solRoutes;    // solRoutes[r] = customer IDs
    int solCost = 0;

    // ────── Penalty coefficients ────────────────────────────────────────────
    double penCapLS = 1.0;
    double penDurLS = 0.0;   // 0 → no duration constraint

    // ────── Local-search linked-list structures ─────────────────────────────
    std::vector<Node>  nodesCli;          // [0..nbClients]
    std::vector<Node>  lsDepots;          // one per route
    std::vector<Node>  lsDepotsEnd;       // end-of-route sentinels
    std::vector<Route> lsRoutes;
    std::vector<std::vector<ThreeBestInsert>> bestIns; // [route][client]
    std::set<int> emptyRoutes;

    int  nbMoves        = 0;
    bool searchCompleted = false;
    std::minstd_rand rng{42};

    // ────── LS temporary variables (mirror HGS LocalSearch private state) ───
    Node*  nodeU = nullptr;  Node*  nodeX = nullptr;
    Node*  nodeV = nullptr;  Node*  nodeY = nullptr;
    Route* routeU = nullptr; Route* routeV = nullptr;
    int nUPrev=0, nU=0, nX=0, nXNext=0;
    int nVPrev=0, nV=0, nY=0, nYNext=0;
    double ldU=0, ldX=0, ldV=0, ldY=0;
    double svU=0, svX=0, svV=0, svY=0;
    bool intraRoute = false;

    // ────── Parsing ─────────────────────────────────────────────────────────
    void parseVRP(const std::string& path);
    void parseSolution(const std::string& path);

    // ────── Granular neighbourhood ──────────────────────────────────────────
    void computeCorrelated();

    // ────── LS setup ────────────────────────────────────────────────────────
    void allocateLS(int numRouteSlots);
    void loadSolutionRoutes();   // loads solRoutes into LS linked list

    // ────── LS core — adapted from HGS LocalSearch.cpp ──────────────────────
    void updateRouteData(Route* r);
    void insertNodeLS(Node* U, Node* V);   // remove U, insert after V
    void swapNodeLS(Node* U, Node* V);     // swap U and V in their routes

    void setVarsU();       // cache local vars for routeU
    void setVarsV();       // cache local vars for routeV

    // Penalty helpers (HGS LocalSearch.h L149-L150)
    double penExLoad(double l) const { return std::max(0.0, l - vehicleCap) * penCapLS; }
    double penExDur (double d) const { return std::max(0.0, d - durationLimit) * penDurLS; }

    // Hard capacity feasibility check — rejects any move that would overload a route
    bool exceedsCap(double load) const { return load > vehicleCap + MY_EPSILON; }

    // ────── LS moves — adapted from HGS move1..move9 ───────────────────────
    bool move1();  // Relocate: remove U, insert after V
    bool move2();  // Relocate (U,X) after V  (same order)
    bool move3();  // Relocate (U,X) after V  (reversed: X,U)
    bool move4();  // Swap U and V
    bool move5();  // Swap (U,X) and V
    bool move6();  // Swap (U,X) and (V,Y)
    bool move7();  // 2-opt intra-route
    bool move8();  // 2-opt* inter-route (with reversal)
    bool move9();  // 2-opt* inter-route (cross)

    // ────── SWAP* — adapted from HGS swapStar / preprocessInsertions ───────
    bool doSwapStar();
    double cheapInsertSimultRemoval(Node* U, Node* V, Node*& bestPos);
    void preprocessInsertions(Route* R1, Route* R2);

    // ────── VND main loop ───────────────────────────────────────────────────
    void runVND();

    // ────── Export ──────────────────────────────────────────────────────────
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
//  Inserter — adds a new customer node, then optimises via VND
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
//  Deleter — removes a customer node, then optimises via VND
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
//  CVRPEngine Implementation
// ══════════════════════════════════════════════════════════════════════════════

// ─────────────── parseVRP ────────────────────────────────────────────────────
// Adapted from HGS-CVRP/Program/InstanceCVRPLIB.cpp — reads the standard
// TSPLIB-style CVRP format with EUC_2D edge weights rounded to integers.
void CVRPEngine::parseVRP(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) throw std::string("Cannot open VRP file: " + path);

    std::string line, tok;
    int dimension = 0;

    while (std::getline(in, line)) {
        // Trim leading and trailing whitespace
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

            // Trim key
            size_t k_first = key.find_first_not_of(" \t");
            size_t k_last = key.find_last_not_of(" \t");
            if (k_first != std::string::npos) key = key.substr(k_first, (k_last - k_first + 1));

            // Trim val
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

    nbClients = dimension - 1;  // depot occupies one slot

    // ── Read coordinates ──
    std::vector<double> xc(dimension), yc(dimension);
    for (int i = 0; i < dimension; i++) {
        int id; in >> id >> xc[i] >> yc[i];
    }

    // ── Read demands ──
    in >> tok;
    if (tok != "DEMAND_SECTION") throw std::string("Expected DEMAND_SECTION, got: " + tok);
    std::vector<double> dem(dimension);
    for (int i = 0; i < dimension; i++) {
        int id; in >> id >> dem[i];
    }

    // ── Build Client array (index 0 = depot) ──
    cli.resize(dimension);
    for (int i = 0; i < dimension; i++) {
        cli[i].coordX = xc[i];
        cli[i].coordY = yc[i];
        cli[i].demand = dem[i];
        cli[i].serviceDuration = 0.0;
        cli[i].polarAngle = CircleSector::positive_mod(
            static_cast<int>(32768.0 * std::atan2(yc[i] - yc[0], xc[i] - xc[0]) / PI_VAL));
    }

    // ── Build distance matrix (EUC_2D, integer-rounded, matching HGS) ──
    dist.assign(dimension, std::vector<double>(dimension, 0.0));
    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++)
            dist[i][j] = std::round(std::sqrt(
                (xc[i]-xc[j])*(xc[i]-xc[j]) + (yc[i]-yc[j])*(yc[i]-yc[j])));

    if (verbose) std::cerr << "[parse] VRP loaded: " << nbClients
              << " clients, capacity=" << vehicleCap << "\n";
}

// ─────────────── parseSolution ───────────────────────────────────────────────
void CVRPEngine::parseSolution(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) throw std::string("Cannot open SOL file: " + path);

    solRoutes.clear();
    std::string tok;
    in >> tok; // first word should be "Route"

    while (tok == "Route") {
        in >> tok; // reads "#1:" — discard
        std::string line;
        std::getline(in, line);
        std::istringstream ss(line);
        std::vector<int> route;
        int cust;
        while (ss >> cust) route.push_back(cust);
        if (!route.empty()) solRoutes.push_back(std::move(route));
        in >> tok; // next "Route" or "Cost"
    }
    if (tok == "Cost") in >> solCost;

    if (verbose) std::cerr << "[parse] Solution loaded: " << solRoutes.size()
              << " routes, cost=" << solCost << "\n";
}

// ─────────────── computeCorrelated ───────────────────────────────────────────
void CVRPEngine::computeCorrelated() {
    int nbGran = std::min(NB_GRANULAR, nbClients - 1);
    correlated.assign(nbClients + 1, {});
    std::vector<std::set<int>> sets(nbClients + 1);

    for (int i = 1; i <= nbClients; i++) {
        std::vector<std::pair<double,int>> order;
        order.reserve(nbClients - 1);
        for (int j = 1; j <= nbClients; j++)
            if (i != j) order.emplace_back(dist[i][j], j);
        // std::sort(order.begin(), order.end());
        std::nth_element(order.begin(), order.begin() + nbGran, order.end());
        for (int k = 0; k < nbGran; k++) {
            sets[i].insert(order[k].second);
            sets[order[k].second].insert(i);  // symmetric
        }
    }
    for (int i = 1; i <= nbClients; i++)
        correlated[i].assign(sets[i].begin(), sets[i].end());

    // Compute global maxDist, maxDemand (for penalty initialisation)
    maxDist = 0; maxDemand = 0;
    for (int i = 0; i <= nbClients; i++)
        for (int j = 0; j <= nbClients; j++)
            if (dist[i][j] > maxDist) maxDist = dist[i][j];
    for (int i = 1; i <= nbClients; i++)
        if (cli[i].demand > maxDemand) maxDemand = cli[i].demand;

    // Penalty initialisation matching HGS-CVRP/Program/Params.cpp (L117)
    penCapLS = std::max(0.1, std::min(1000.0, maxDist / maxDemand));
    penDurLS = 0.0; // no duration constraint
}

// ─────────────── allocateLS ──────────────────────────────────────────────────
void CVRPEngine::allocateLS(int numRouteSlots) {
    nbVehicles = numRouteSlots;

    nodesCli.resize(nbClients + 1);
    lsDepots.resize(nbVehicles);
    lsDepotsEnd.resize(nbVehicles);
    lsRoutes.resize(nbVehicles);
    bestIns.assign(nbVehicles,
                   std::vector<ThreeBestInsert>(nbClients + 1));

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

// ─────────────── loadSolutionRoutes ──────────────────────────────────────────
// Loads solRoutes into the LS linked list. Any route slots beyond
// solRoutes.size() are set up as empty routes.
void CVRPEngine::loadSolutionRoutes() {
    emptyRoutes.clear();
    nbMoves = 0;

    // Load each existing route from solRoutes
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
        for (int i = 1; i <= nbClients; i++)
            bestIns[r][i].whenLastCalculated = -1;
    }

    // Set up any remaining route slots as empty
    for (int r = static_cast<int>(solRoutes.size()); r < nbVehicles; r++) {
        Node* dep    = &lsDepots[r];
        Node* depEnd = &lsDepotsEnd[r];
        dep->prev    = depEnd;
        depEnd->next = dep;
        dep->next    = depEnd;
        depEnd->prev = dep;
        updateRouteData(&lsRoutes[r]);
        lsRoutes[r].whenLastTestedSWAPStar = -1;
        for (int i = 1; i <= nbClients; i++)
            bestIns[r][i].whenLastCalculated = -1;
    }

    // Initialise per-node timestamps
    for (int i = 1; i <= nbClients; i++)
        nodesCli[i].whenLastTestedRI = -1;
}

// ─────────────── updateRouteData ─────────────────────────────────────────────
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
        mytime += dist[mynode->prev->cour][mynode->cour]
                + cli[mynode->cour].serviceDuration;
        myRev  += dist[mynode->cour][mynode->prev->cour]
                - dist[mynode->prev->cour][mynode->cour];
        mynode->cumulatedLoad             = myload;
        mynode->cumulatedTime             = mytime;
        mynode->cumulatedReversalDistance  = myRev;
        if (!mynode->isDepot) {
            cumX += cli[mynode->cour].coordX;
            cumY += cli[mynode->cour].coordY;
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
            cumY / myRoute->nbCustomers - cli[0].coordY,
            cumX / myRoute->nbCustomers - cli[0].coordX);
        emptyRoutes.erase(myRoute->cour);
    }
}

// ─────────────── insertNodeLS / swapNodeLS ────────────────────────────────────
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

// ─────────────── setVarsU / setVarsV ─────────────────────────────────────────
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
//  LS Moves — direct adaptations from HGS LocalSearch.cpp move1..move9
// ══════════════════════════════════════════════════════════════════════════════

// ─── move1: Relocate U after V ──────────────────────────────────────────────
bool CVRPEngine::move1() {
    double costU = dist[nUPrev][nX] - dist[nUPrev][nU] - dist[nU][nX];
    double costV = dist[nV][nU] + dist[nU][nY] - dist[nV][nY];

    if (!intraRoute) {
        if (exceedsCap(routeV->load + ldU)) return false;
        if (costU + costV >= routeU->penalty + routeV->penalty) return false;
        costU += penExDur(routeU->duration + costU - svU)
               + penExLoad(routeU->load - ldU) - routeU->penalty;
        costV += penExDur(routeV->duration + costV + svU)
               + penExLoad(routeV->load + ldU) - routeV->penalty;
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

// ─── move2: Relocate (U,X) after V, same order ─────────────────────────────
bool CVRPEngine::move2() {
    double costU = dist[nUPrev][nXNext] - dist[nUPrev][nU] - dist[nX][nXNext];
    double costV = dist[nV][nU] + dist[nX][nY] - dist[nV][nY];

    if (!intraRoute) {
        if (exceedsCap(routeV->load + ldU + ldX)) return false;
        if (costU + costV >= routeU->penalty + routeV->penalty) return false;
        costU += penExDur(routeU->duration + costU - dist[nU][nX] - svU - svX)
               + penExLoad(routeU->load - ldU - ldX) - routeU->penalty;
        costV += penExDur(routeV->duration + costV + dist[nU][nX] + svU + svX)
               + penExLoad(routeV->load + ldU + ldX) - routeV->penalty;
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

// ─── move3: Relocate (U,X) after V, reversed order ─────────────────────────
bool CVRPEngine::move3() {
    double costU = dist[nUPrev][nXNext] - dist[nUPrev][nU]
                 - dist[nU][nX] - dist[nX][nXNext];
    double costV = dist[nV][nX] + dist[nX][nU]
                 + dist[nU][nY] - dist[nV][nY];

    if (!intraRoute) {
        if (exceedsCap(routeV->load + ldU + ldX)) return false;
        if (costU + costV >= routeU->penalty + routeV->penalty) return false;
        costU += penExDur(routeU->duration + costU - svU - svX)
               + penExLoad(routeU->load - ldU - ldX) - routeU->penalty;
        costV += penExDur(routeV->duration + costV + svU + svX)
               + penExLoad(routeV->load + ldU + ldX) - routeV->penalty;
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

// ─── move4: Swap U and V ────────────────────────────────────────────────────
bool CVRPEngine::move4() {
    double costU = dist[nUPrev][nV] + dist[nV][nX]
                 - dist[nUPrev][nU] - dist[nU][nX];
    double costV = dist[nVPrev][nU] + dist[nU][nY]
                 - dist[nVPrev][nV] - dist[nV][nY];

    if (!intraRoute) {
        if (exceedsCap(routeU->load + ldV - ldU) || exceedsCap(routeV->load + ldU - ldV)) return false;
        if (costU + costV >= routeU->penalty + routeV->penalty) return false;
        costU += penExDur(routeU->duration + costU + svV - svU)
               + penExLoad(routeU->load + ldV - ldU) - routeU->penalty;
        costV += penExDur(routeV->duration + costV - svV + svU)
               + penExLoad(routeV->load + ldU - ldV) - routeV->penalty;
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

// ─── move5: Swap (U,X) and V ───────────────────────────────────────────────
bool CVRPEngine::move5() {
    double costU = dist[nUPrev][nV] + dist[nV][nXNext]
                 - dist[nUPrev][nU] - dist[nX][nXNext];
    double costV = dist[nVPrev][nU] + dist[nX][nY]
                 - dist[nVPrev][nV] - dist[nV][nY];

    if (!intraRoute) {
        if (exceedsCap(routeU->load + ldV - ldU - ldX) || exceedsCap(routeV->load + ldU + ldX - ldV)) return false;
        if (costU + costV >= routeU->penalty + routeV->penalty) return false;
        costU += penExDur(routeU->duration + costU - dist[nU][nX] + svV - svU - svX)
               + penExLoad(routeU->load + ldV - ldU - ldX) - routeU->penalty;
        costV += penExDur(routeV->duration + costV + dist[nU][nX] - svV + svU + svX)
               + penExLoad(routeV->load + ldU + ldX - ldV) - routeV->penalty;
    }
    if (costU + costV > -MY_EPSILON) return false;
    if (nodeU == nodeV->prev || nodeX == nodeV->prev
        || nodeU == nodeY || nodeX->isDepot) return false;

    swapNodeLS(nodeU, nodeV);
    insertNodeLS(nodeX, nodeU);
    nbMoves++;
    searchCompleted = false;
    updateRouteData(routeU);
    if (!intraRoute) updateRouteData(routeV);
    return true;
}

// ─── move6: Swap (U,X) and (V,Y) ──────────────────────────────────────────
bool CVRPEngine::move6() {
    double costU = dist[nUPrev][nV] + dist[nY][nXNext]
                 - dist[nUPrev][nU] - dist[nX][nXNext];
    double costV = dist[nVPrev][nU] + dist[nX][nYNext]
                 - dist[nVPrev][nV] - dist[nY][nYNext];

    if (!intraRoute) {
        if (exceedsCap(routeU->load + ldV + ldY - ldU - ldX) || exceedsCap(routeV->load + ldU + ldX - ldV - ldY)) return false;
        if (costU + costV >= routeU->penalty + routeV->penalty) return false;
        costU += penExDur(routeU->duration + costU - dist[nU][nX] + dist[nV][nY]
                   + svV + svY - svU - svX)
               + penExLoad(routeU->load + ldV + ldY - ldU - ldX) - routeU->penalty;
        costV += penExDur(routeV->duration + costV + dist[nU][nX] - dist[nV][nY]
                   - svV - svY + svU + svX)
               + penExLoad(routeV->load + ldU + ldX - ldV - ldY) - routeV->penalty;
    }
    if (costU + costV > -MY_EPSILON) return false;
    if (nodeX->isDepot || nodeY->isDepot || nodeY == nodeU->prev
        || nodeU == nodeY || nodeX == nodeV || nodeV == nodeX->next) return false;

    swapNodeLS(nodeU, nodeV);
    swapNodeLS(nodeX, nodeY);
    nbMoves++;
    searchCompleted = false;
    updateRouteData(routeU);
    if (!intraRoute) updateRouteData(routeV);
    return true;
}

// ─── move7: 2-opt intra-route ──────────────────────────────────────────────
bool CVRPEngine::move7() {
    if (nodeU->position > nodeV->position) return false;

    double cost = dist[nU][nV] + dist[nX][nY]
                - dist[nU][nX] - dist[nV][nY]
                + nodeV->cumulatedReversalDistance
                - nodeX->cumulatedReversalDistance;

    if (cost > -MY_EPSILON) return false;
    if (nodeU->next == nodeV) return false;

    // Reverse the segment between X and V
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

// ─── move8: 2-opt* inter-route with reversal ───────────────────────────────
bool CVRPEngine::move8() {
    double cost = dist[nU][nV] + dist[nX][nY]
                - dist[nU][nX] - dist[nV][nY]
                + nodeV->cumulatedReversalDistance
                + routeU->reversalDistance
                - nodeX->cumulatedReversalDistance
                - routeU->penalty - routeV->penalty;

    if (cost >= 0) return false;

    double newLoadA = nodeU->cumulatedLoad + nodeV->cumulatedLoad;
    double newLoadB = routeU->load + routeV->load - nodeU->cumulatedLoad - nodeV->cumulatedLoad;
    if (exceedsCap(newLoadA) || exceedsCap(newLoadB)) return false;

    cost += penExDur(nodeU->cumulatedTime + nodeV->cumulatedTime
                   + nodeV->cumulatedReversalDistance + dist[nU][nV])
          + penExDur(routeU->duration - nodeU->cumulatedTime - dist[nU][nX]
                   + routeU->reversalDistance - nodeX->cumulatedReversalDistance
                   + routeV->duration - nodeV->cumulatedTime - dist[nV][nY]
                   + dist[nX][nY])
          + penExLoad(newLoadA)
          + penExLoad(newLoadB);

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

// ─── move9: 2-opt* inter-route, cross ──────────────────────────────────────
bool CVRPEngine::move9() {
    double cost = dist[nU][nY] + dist[nV][nX]
                - dist[nU][nX] - dist[nV][nY]
                - routeU->penalty - routeV->penalty;

    if (cost >= 0) return false;

    double newLoadA = nodeU->cumulatedLoad + routeV->load - nodeV->cumulatedLoad;
    double newLoadB = nodeV->cumulatedLoad + routeU->load - nodeU->cumulatedLoad;
    if (exceedsCap(newLoadA) || exceedsCap(newLoadB)) return false;

    cost += penExDur(nodeU->cumulatedTime + routeV->duration - nodeV->cumulatedTime
                   - dist[nV][nY] + dist[nU][nY])
          + penExDur(routeU->duration - nodeU->cumulatedTime - dist[nU][nX]
                   + nodeV->cumulatedTime + dist[nV][nX])
          + penExLoad(newLoadA)
          + penExLoad(newLoadB);

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
//  SWAP* — adapted from HGS LocalSearch::swapStar (L485-L572)
// ══════════════════════════════════════════════════════════════════════════════

void CVRPEngine::preprocessInsertions(Route* R1, Route* R2) {
    for (Node* U = R1->depot->next; !U->isDepot; U = U->next) {
        U->deltaRemoval = dist[U->prev->cour][U->next->cour]
                        - dist[U->prev->cour][U->cour]
                        - dist[U->cour][U->next->cour];
        if (R2->whenLastModified > bestIns[R2->cour][U->cour].whenLastCalculated) {
            bestIns[R2->cour][U->cour].reset();
            bestIns[R2->cour][U->cour].whenLastCalculated = nbMoves;
            bestIns[R2->cour][U->cour].bestCost[0] =
                dist[0][U->cour] + dist[U->cour][R2->depot->next->cour]
              - dist[0][R2->depot->next->cour];
            bestIns[R2->cour][U->cour].bestLocation[0] = R2->depot;
            for (Node* V = R2->depot->next; !V->isDepot; V = V->next) {
                double dc = dist[V->cour][U->cour]
                          + dist[U->cour][V->next->cour]
                          - dist[V->cour][V->next->cour];
                bestIns[R2->cour][U->cour].compareAndAdd(dc, V);
            }
        }
    }
}

double CVRPEngine::cheapInsertSimultRemoval(
        Node* U, Node* V, Node*& bestPos) {
    ThreeBestInsert* bi = &bestIns[V->route->cour][U->cour];
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

    double dc = dist[V->prev->cour][U->cour]
              + dist[U->cour][V->next->cour]
              - dist[V->prev->cour][V->next->cour];
    if (!found || dc < bestCost) {
        bestPos  = V->prev;
        bestCost = dc;
    }
    return bestCost;
}

bool CVRPEngine::doSwapStar() {
    SwapStarElement best;

    preprocessInsertions(routeU, routeV);
    preprocessInsertions(routeV, routeU);

    // ── Evaluate all (U,V) SWAP* pairs ──
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

                double extraV = cheapInsertSimultRemoval(nodeU, nodeV,
                                                         el.bestPositionU);
                double extraU = cheapInsertSimultRemoval(nodeV, nodeU,
                                                         el.bestPositionV);
                el.moveCost = dpU + nodeU->deltaRemoval + extraU
                            + dpV + nodeV->deltaRemoval + extraV
                            + penExDur(routeU->duration + nodeU->deltaRemoval
                                     + extraU + cli[nodeV->cour].serviceDuration
                                     - cli[nodeU->cour].serviceDuration)
                            + penExDur(routeV->duration + nodeV->deltaRemoval
                                     + extraV - cli[nodeV->cour].serviceDuration
                                     + cli[nodeU->cour].serviceDuration);

                if (el.moveCost < best.moveCost) best = el;
            }
        }
    }

    // ── RELOCATE U → routeV ──
    for (nodeU = routeU->depot->next; !nodeU->isDepot; nodeU = nodeU->next) {
        if (exceedsCap(routeV->load + cli[nodeU->cour].demand)) continue;
        SwapStarElement el;
        el.U = nodeU;
        el.bestPositionU = bestIns[routeV->cour][nodeU->cour].bestLocation[0];
        double ddU = dist[nodeU->prev->cour][nodeU->next->cour]
                   - dist[nodeU->prev->cour][nodeU->cour]
                   - dist[nodeU->cour][nodeU->next->cour];
        double ddV = bestIns[routeV->cour][nodeU->cour].bestCost[0];
        el.moveCost = ddU + ddV
            + penExLoad(routeU->load - cli[nodeU->cour].demand) - routeU->penalty
            + penExLoad(routeV->load + cli[nodeU->cour].demand) - routeV->penalty
            + penExDur(routeU->duration + ddU - cli[nodeU->cour].serviceDuration)
            + penExDur(routeV->duration + ddV + cli[nodeU->cour].serviceDuration);
        if (el.moveCost < best.moveCost) best = el;
    }

    // ── RELOCATE V → routeU ──
    for (nodeV = routeV->depot->next; !nodeV->isDepot; nodeV = nodeV->next) {
        if (exceedsCap(routeU->load + cli[nodeV->cour].demand)) continue;
        SwapStarElement el;
        el.V = nodeV;
        el.bestPositionV = bestIns[routeU->cour][nodeV->cour].bestLocation[0];
        double ddU = bestIns[routeU->cour][nodeV->cour].bestCost[0];
        double ddV = dist[nodeV->prev->cour][nodeV->next->cour]
                   - dist[nodeV->prev->cour][nodeV->cour]
                   - dist[nodeV->cour][nodeV->next->cour];
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
//  VND main loop — adapted from HGS LocalSearch::run (L3-L96)
// ══════════════════════════════════════════════════════════════════════════════
void CVRPEngine::runVND() {
    std::vector<int> orderNodes(nbClients);
    std::iota(orderNodes.begin(), orderNodes.end(), 1); // 1..nbClients
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

        // ── Phase 1: Classical RI moves with granular restriction ──
        for (int posU = 0; posU < nbClients; posU++) {
            nodeU = &nodesCli[orderNodes[posU]];
            int lastTest = nodeU->whenLastTestedRI;
            nodeU->whenLastTestedRI = nbMoves;

            for (int posV = 0; posV < static_cast<int>(
                     correlated[nodeU->cour].size()); posV++) {
                nodeV = &nodesCli[correlated[nodeU->cour][posV]];
                if (loopID == 0
                    || std::max(nodeU->route->whenLastModified,
                                nodeV->route->whenLastModified) > lastTest) {
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

        // ── Phase 2: SWAP* over route pairs with sector overlap ──
        for (int rU = 0; rU < nbVehicles; rU++) {
            routeU = &lsRoutes[orderRoutes[rU]];
            int lastSS = routeU->whenLastTestedSWAPStar;
            routeU->whenLastTestedSWAPStar = nbMoves;
            for (int rV = 0; rV < nbVehicles; rV++) {
                routeV = &lsRoutes[orderRoutes[rV]];
                if (routeU->nbCustomers > 0 && routeV->nbCustomers > 0
                    && routeU->cour < routeV->cour
                    && (loopID == 0
                        || std::max(routeU->whenLastModified,
                                    routeV->whenLastModified) > lastSS))
                {
                    if (CircleSector::overlap(routeU->sector, routeV->sector))
                        doSwapStar();
                }
            }
        }
    }

    if (verbose) std::cerr << "[vnd]   Applied " << (nbMoves - totalMovesBefore)
              << " improving moves\n";
}

// ──────────────── exportSolution ─────────────────────────────────────────────
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
            double rd = dist[0][route[0]];
            double rl = cli[route[0]].demand;
            for (int i = 1; i < static_cast<int>(route.size()); i++) {
                rd += dist[route[i-1]][route[i]];
                rl += cli[route[i]].demand;
            }
            rd += dist[route.back()][0];
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

// ──────────────── writeSolution ──────────────────────────────────────────────
void CVRPEngine::writeSolution(const std::string& path,
                               const SolOutput& sol) {
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

// ──────────────── writeVRP ───────────────────────────────────────────────────
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
        out << " " << (i + 1) << " " << static_cast<int>(std::round(cli[i].coordX))
            << " " << static_cast<int>(std::round(cli[i].coordY)) << "\n";
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

// ──────────────── printAndVerify ─────────────────────────────────────────────
void CVRPEngine::printAndVerify(const SolOutput& sol) {
    if (verbose) {
        std::cerr << "[done]  Routes: " << sol.routes.size()
                  << "  Cost: " << sol.totalDist
                  << "  Feasible: " << (sol.feasible ? "YES" : "NO");
        if (!sol.feasible)
            std::cerr << "  (cap excess=" << sol.capExcess << ")";
        std::cerr << "\n";
    }

    // for (int r = 0; r < static_cast<int>(sol.routes.size()); r++) {
    //     double rl = 0;
    //     for (int c : sol.routes[r]) rl += cli[c].demand;
    //     bool overloaded = rl > vehicleCap + MY_EPSILON;
    //     if (verbose) {
    //         std::cerr << "        Route #" << (r+1) << ":";
    //         for (int c : sol.routes[r]) std::cerr << " " << c;
    //         std::cerr << "  | load=" << rl << "/" << vehicleCap
    //                   << (overloaded ? " *** OVER CAPACITY ***" : "")
    //                   << "\n";
    //     }
    // }

    if (!sol.feasible) {
        if (verbose) std::cerr << "[WARN]  Solution has capacity violations (excess="
                  << sol.capExcess << "). This should not happen.\n";
    }
}


// ══════════════════════════════════════════════════════════════════════════════
//  Inserter Implementation
// ══════════════════════════════════════════════════════════════════════════════

// ─────────────── addNewNode ──────────────────────────────────────────────────
void Inserter::addNewNode(double x, double y, double demand) {
    nbClients++;
    newNodeIdx = nbClients;

    Client c;
    c.coordX = x;
    c.coordY = y;
    c.demand = demand;
    c.serviceDuration = 0.0;
    c.polarAngle = CircleSector::positive_mod(
        static_cast<int>(32768.0 * std::atan2(y - cli[0].coordY, x - cli[0].coordX) / PI_VAL));
    cli.push_back(c);

    int N = nbClients + 1;
    for (int i = 0; i < N - 1; i++) {
        double d = std::round(std::sqrt(
            (cli[i].coordX - x) * (cli[i].coordX - x) +
            (cli[i].coordY - y) * (cli[i].coordY - y)));
        dist[i].push_back(d);
    }
    std::vector<double> newRow(N, 0.0);
    for (int j = 0; j < N - 1; j++)
        newRow[j] = dist[j].back();
    newRow[N - 1] = 0.0;
    dist.push_back(std::move(newRow));

    if (verbose) std::cerr << "[node]  Added new customer " << newNodeIdx
              << " at (" << x << "," << y << ") demand=" << demand << "\n";
}

// ─────────────── loadSolutionWithSpare ───────────────────────────────────────
void Inserter::loadSolutionWithSpare() {
    loadSolutionRoutes();  // loads solRoutes + sets up spare routes as empty

    // Put the new node into the spare route (first empty slot after solRoutes)
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

// ─────────────── Inserter::run ───────────────────────────────────────────────
void Inserter::run(const std::string& vrpFile,
                   const std::string& solFile,
                   const std::string& outSolFile,
                   const std::string& outVrpFile,
                   double newX, double newY, double newDemand) {

    auto time1 = std::chrono::high_resolution_clock::now();

    // 1. Parse
    parseVRP(vrpFile);
    parseSolution(solFile);
    int initialNodes = nbClients + 1;  // depot + customers before insertion

    auto time2 = std::chrono::high_resolution_clock::now();

    // 2. Add the new node
    addNewNode(newX, newY, newDemand);

    auto time3 = std::chrono::high_resolution_clock::now();

    // 3. Compute granular neighbourhoods for ALL nodes (including new)
    computeCorrelated();

    auto time4 = std::chrono::high_resolution_clock::now();

    // 4. Allocate LS structures (+1 spare route for new node)
    allocateLS(static_cast<int>(solRoutes.size()) + 1);

    auto time5 = std::chrono::high_resolution_clock::now();

    // 5. Load solution (new node in spare route)
    loadSolutionWithSpare();

    auto time6 = std::chrono::high_resolution_clock::now();

    // 6. Run VND
    if (verbose) std::cerr << "[vnd]   Starting Variable Neighbourhood Descent...\n";
    
    runVND();

    auto time7 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_compute = time7 - time1;
    std::chrono::duration<double> parse_time = time2 - time1;
    std::chrono::duration<double> add_node_time = time3 - time2;
    std::chrono::duration<double> compute_correlated_time = time4 - time3;
    std::chrono::duration<double> allocate_ls_time = time5 - time4;
    std::chrono::duration<double> load_solution_time = time6 - time5;
    std::chrono::duration<double> run_vnd_time = time7 - time6;

    if (timecheck) {
        bool file_exists = false;
        {
            std::ifstream check("timecheck.csv");
            file_exists = check.good();
        }
        std::ofstream csv("timecheck.csv", std::ios::app);
        if (!file_exists) {
            csv << "initial_nodes,mode,total_compute,parse,add/delete_node,compute_correlated,allocate_ls,load_solution,run_vnd\n";
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

    // 7. Export and write
    auto sol = exportSolution();
    printAndVerify(sol);

    writeSolution(outSolFile, sol);
    if (verbose) std::cerr << "[file]  Solution written to " << outSolFile << "\n";

    writeVRP(outVrpFile, "with 1 dynamic node added");
    if (verbose) std::cerr << "[file]  Updated VRP written to " << outVrpFile << "\n";
}


// ══════════════════════════════════════════════════════════════════════════════
//  Deleter Implementation
// ══════════════════════════════════════════════════════════════════════════════

// ─────────────── removeNodeFromSolution ──────────────────────────────────────
// Finds the target node in solRoutes, removes it. If the route becomes
// empty, removes that route entirely.
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
                if (verbose) std::cerr << "[delete] Route #" << (r + 1)
                          << " is now empty — removing it\n";
                solRoutes.erase(solRoutes.begin() + r);
            }
            break;
        }
    }
    if (!found) {
        throw std::string("Node " + std::to_string(nodeId)
                          + " not found in any route of the solution");
    }

    if (solRoutes.empty()) {
        throw std::string("Solution is empty after deleting node "
                          + std::to_string(nodeId)
                          + " — no routes remain");
    }
}

// ─────────────── removeNodeFromProblem ───────────────────────────────────────
// Removes the deleted node from cli[], dist[][], and re-indexes all customer
// IDs so the data structures remain dense and contiguous.
void Deleter::removeNodeFromProblem(int nodeId) {
    // Re-index solRoutes: any customer ID > nodeId is decremented
    for (auto& route : solRoutes) {
        for (auto& c : route) {
            if (c > nodeId) c--;
        }
    }

    // Remove from cli vector
    cli.erase(cli.begin() + nodeId);

    // Remove from distance matrix: row nodeId, then column nodeId
    dist.erase(dist.begin() + nodeId);
    for (auto& row : dist) {
        row.erase(row.begin() + nodeId);
    }

    nbClients--;

    if (verbose) std::cerr << "[delete] Node " << nodeId << " removed from problem data. "
              << nbClients << " clients remain.\n";
}

// ─────────────── Deleter::run ────────────────────────────────────────────────
void Deleter::run(const std::string& vrpFile,
                  const std::string& solFile,
                  const std::string& outSolFile,
                  const std::string& outVrpFile,
                  int deleteNodeId) {
    auto time1 = std::chrono::high_resolution_clock::now();
    
    // 1. Parse
    parseVRP(vrpFile);
    parseSolution(solFile);
    int initialNodes = nbClients + 1;  // depot + customers before deletion

    auto time2 = std::chrono::high_resolution_clock::now();
    
    // 2. Validate the node ID
    if (deleteNodeId < 1 || deleteNodeId > nbClients) {
        throw std::string("Invalid node ID " + std::to_string(deleteNodeId)
                          + ". Must be in range [1, "
                          + std::to_string(nbClients) + "]");
    }

    if (verbose) std::cerr << "[delete] Deleting customer " << deleteNodeId
              << " (coords: " << cli[deleteNodeId].coordX
              << "," << cli[deleteNodeId].coordY
              << ", demand=" << cli[deleteNodeId].demand << ")\n";

    auto start_compute = std::chrono::high_resolution_clock::now();

    // 3. Splice node out of solution routes
    removeNodeFromSolution(deleteNodeId);

    // 4. Shrink problem data and re-index
    removeNodeFromProblem(deleteNodeId);

    auto time3 = std::chrono::high_resolution_clock::now();

    // 5. Rebuild granular neighbourhoods
    computeCorrelated();

    auto time4 = std::chrono::high_resolution_clock::now();
    
    // 6. Allocate LS structures (no spare needed for deletion)
    allocateLS(static_cast<int>(solRoutes.size()));

    auto time5 = std::chrono::high_resolution_clock::now();
    
    // 7. Load the spliced solution
    loadSolutionRoutes();

    // 8. Run VND to heal gaps left by removal
    if (verbose) std::cerr << "[vnd]   Starting Variable Neighbourhood Descent...\n";

    auto time6 = std::chrono::high_resolution_clock::now();

    runVND();

    auto time7 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> total_compute = time7 - time1;
    std::chrono::duration<double> parse_time = time2 - time1;
    std::chrono::duration<double> delete_node_time = time3 - time2;
    std::chrono::duration<double> compute_correlated_time = time4 - time3;
    std::chrono::duration<double> allocate_ls_time = time5 - time4;
    std::chrono::duration<double> load_solution_time = time6 - time5;
    std::chrono::duration<double> run_vnd_time = time7 - time6;


    if (timecheck) {
        bool file_exists = false;
        {
            std::ifstream check("timecheck.csv");
            file_exists = check.good();
        }
        std::ofstream csv("timecheck.csv", std::ios::app);
        if (!file_exists) {
            csv << "initial_nodes,mode,total_compute,parse,add/delete_node,compute_correlated,allocate_ls,load_solution,run_vnd\n";
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

    // 9. Export and write
    auto sol = exportSolution();
    printAndVerify(sol);

    writeSolution(outSolFile, sol);
    if (verbose) std::cerr << "[file]  Solution written to " << outSolFile << "\n";

    writeVRP(outVrpFile, "with 1 node removed");
    if (verbose) std::cerr << "[file]  Updated VRP written to " << outVrpFile << "\n";
}


// ══════════════════════════════════════════════════════════════════════════════
//  main()
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
            // Addition: dynamic_cvrp a <vrp> <sol> <out.sol> <out.vrp> <X> <Y> <demand>
            if (args.size() != 8) {
                std::cerr << "ERROR: Addition mode requires 7 arguments after 'a'.\n\n";
                printUsage();
                return 1;
            }
            Inserter ins;
            ins.verbose = verbose;
            ins.timecheck = timecheck;
            ins.run(args[1], args[2], args[3], args[4],
                    std::stod(args[5]), std::stod(args[6]), std::stod(args[7]));
        }
        else if (mode == "d") {
            // Deletion: dynamic_cvrp d <vrp> <sol> <out.sol> <out.vrp> <node_id>
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
            std::cerr << "ERROR: Unknown mode '" << mode
                      << "'. Use 'a' for addition or 'd' for deletion.\n\n";
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
