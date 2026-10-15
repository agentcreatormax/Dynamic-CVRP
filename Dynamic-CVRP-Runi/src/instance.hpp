// ============================================================================
// instance.hpp — CVRP instance + solution I/O for the dynamic solver.
//
// UGRC Dynamic-CVRP (Nithin Hanumanthu, IIT Madras)
//
// Conventions (shared with parMDS and the d-cvrp baseline):
//   * Node id 0 is the DEPOT; customers are 1..n (n = DIMENSION-1).
//   * Route lines in .sol files list customer ids (depot never appears).
//   * Distances are integer-rounded EUC_2D Euclidean distances.
// ============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using node_t   = int;
using dist_t   = double;

struct Instance {
    int n = 0;                      // number of customers (excluding depot)
    double capacity = 0.0;
    std::string name;

    std::vector<double> xs, ys;     // size n+1, index 0 = depot
    std::vector<double> demand;     // size n+1, demand[0] = 0

    inline dist_t d(node_t i, node_t j) const {          // rounded euclid
        double dx = xs[i] - xs[j], dy = ys[i] - ys[j];
        return std::round(std::sqrt(dx * dx + dy * dy));
    }
    inline dist_t dDepot(node_t i) const { return d(0, i); }

    void reserveExtra(int extra) {                       // stable-id growth
        xs.reserve(n + 1 + extra);
        ys.reserve(n + 1 + extra);
        demand.reserve(n + 1 + extra);
    }
};

// ---------------------------------------------------------------------------
// TSPLIB .vrp reader (robust to tabs/spaces; ignores DEPOT_SECTION content,
// assumes row order == id order as produced by CVRPLIB / parMDS / d-cvrp).
// ---------------------------------------------------------------------------
inline void read_vrp(const std::string& path, Instance& inst) {
    std::ifstream in(path);
    if (!in.is_open()) throw std::runtime_error("cannot open vrp: " + path);

    int dimension = -1;
    inst.capacity = -1;
    std::string line;
    bool haveCapLine = false;

    while (std::getline(in, line)) {
        // trim
        auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        std::string clean = line.substr(first, line.find_last_not_of(" \t\r\n") - first + 1);
        if (clean == "NODE_COORD_SECTION") break;

        auto colon = clean.find(':');
        std::string key, val;
        if (colon != std::string::npos) {
            key = clean.substr(0, colon);
            val = clean.substr(colon + 1);
        } else {
            std::istringstream ss(clean);
            ss >> key >> val;
        }
        auto trim = [&](std::string s) {
            auto f = s.find_first_not_of(" \t");
            if (f == std::string::npos) return std::string();
            return s.substr(f, s.find_last_not_of(" \t") - f + 1);
        };
        key = trim(key); val = trim(val);
        if      (key == "DIMENSION") dimension = std::stoi(val);
        else if (key == "CAPACITY")  { inst.capacity = std::stod(val); haveCapLine = true; }
        else if (key == "NAME")      inst.name = val;
    }
    if (dimension <= 0) throw std::runtime_error("DIMENSION missing in " + path);
    if (!haveCapLine)   throw std::runtime_error("CAPACITY missing in " + path);

    inst.n = dimension - 1;
    inst.xs.assign(dimension, 0.0);
    inst.ys.assign(dimension, 0.0);
    inst.demand.assign(dimension, 0.0);

    for (int i = 0; i < dimension; ++i) {
        int id; in >> id >> inst.xs[i] >> inst.ys[i];
        (void)id; // row order == id order by construction of all our producers
    }
    std::string tok; in >> tok;
    if (tok != "DEMAND_SECTION") throw std::runtime_error("expected DEMAND_SECTION in " + path);
    for (int i = 0; i < dimension; ++i) {
        int id; in >> id >> inst.demand[i];
        (void)id;
    }
}

// ---------------------------------------------------------------------------
// DIMACS-style .sol reader: lines "Route #k: v1 v2 ..." + optional "Cost X".
// ---------------------------------------------------------------------------
inline void read_sol(const std::string& path, std::vector<std::vector<node_t>>& routes) {
    std::ifstream in(path);
    if (!in.is_open()) throw std::runtime_error("cannot open sol: " + path);
    routes.clear();
    std::string tok;
    while (in >> tok) {
        if (tok != "Route") continue;
        in >> tok;                       // "#k:"
        std::getline(in, tok);
        std::istringstream ss(tok);
        std::vector<node_t> r; int v;
        while (ss >> v) r.push_back(v);
        if (!r.empty()) routes.push_back(std::move(r));
    }
}

inline void write_sol(const std::string& path,
                      const std::vector<std::vector<node_t>>& routes,
                      const Instance& inst, double cost) {
    std::ofstream out(path);
    for (size_t r = 0; r < routes.size(); ++r) {
        out << "Route #" << (r + 1) << ":";
        for (node_t v : routes[r]) out << ' ' << v;
        out << '\n';
    }
    out << "Cost " << std::lround(cost) << '\n';
    (void)inst;
}

// Total rounded route cost (depot bookends each route).
template <typename Routes>
inline double solution_cost(const Routes& routes, const Instance& inst) {
    double total = 0.0;
    for (const auto& r : routes) {
        if (r.empty()) continue;
        node_t prev = 0;
        for (node_t v : r) { total += inst.d(prev, v); prev = v; }
        total += inst.d(prev, 0);
    }
    return total;
}
