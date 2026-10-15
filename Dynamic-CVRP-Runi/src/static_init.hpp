// ============================================================================
// static_init.hpp — static CVRP construction used for the initial solution.
//
// Compact port of the parMDS/seqMDS pipeline (Pandian et al., GECCO'23):
//   Prim MST -> randomized DFS shortcut tour -> capacity partition
//   -> per-route 2-opt polish.  Used only for --static-init; the dynamic
//   engine itself always warm-starts from an existing feasible solution.
// ============================================================================
#pragma once

#include "instance.hpp"
#include <random>
#include <set>

inline std::vector<std::vector<node_t>> static_mds_init(const Instance& inst, uint64_t seed) {
    const int n = inst.n;
    std::mt19937_64 rng(seed);

    // ---- Prim MST over complete graph, O(n^2) ----
    std::vector<double> key(n + 1, 1e30);
    std::vector<node_t> parent(n + 1, -1);
    std::vector<char>   inMST(n + 1, 0);
    key[0] = 0.0;
    for (int it = 0; it <= n; ++it) {
        int u = -1;
        for (int v = 0; v <= n; ++v)
            if (!inMST[v] && (u == -1 || key[v] < key[u])) u = v;
        inMST[u] = 1;
        for (int v = 0; v <= n; ++v)
            if (!inMST[v]) {
                double w = inst.d(u, v);
                if (w < key[v]) { key[v] = w; parent[v] = u; }
            }
    }
    std::vector<std::vector<node_t>> adj(n + 1);
    for (int v = 1; v <= n; ++v) {
        adj[v].push_back(parent[v]);
        adj[parent[v]].push_back(v);
    }

    // ---- randomized DFS preorder (shortcut tour) ----
    for (auto& lst : adj) std::shuffle(lst.begin(), lst.end(), rng);
    std::vector<node_t> order;
    order.reserve(2 * n + 1);
    std::vector<node_t> stack{0};
    std::vector<char> vis(n + 1, 0);
    while (!stack.empty()) {
        node_t u = stack.back(); stack.pop_back();
        if (vis[u]) continue;
        vis[u] = 1;
        if (u != 0) order.push_back(u);          // skip depot in partition step
        for (auto it = adj[u].rbegin(); it != adj[u].rend(); ++it)
            if (!vis[*it]) stack.push_back(*it);
    }

    // ---- capacity partition ----
    std::vector<std::vector<node_t>> routes;
    {
        std::vector<node_t> cur;
        double load = 0;
        for (node_t v : order) {
            if (load + inst.demand[v] <= inst.capacity + 1e-9) {
                cur.push_back(v); load += inst.demand[v];
            } else {
                routes.push_back(cur);
                cur.clear(); cur.push_back(v); load = inst.demand[v];
            }
        }
        if (!cur.empty()) routes.push_back(cur);
    }

    // ---- per-route 2-opt polish (segment reversal until no gain) ----
    auto routeCost = [&](const std::vector<node_t>& r) {
        double c = inst.d(0, r.front());
        for (size_t i = 1; i < r.size(); ++i) c += inst.d(r[i - 1], r[i]);
        return c + inst.d(r.back(), 0);
    };
    for (auto& r : routes) {
        if (r.size() < 3) continue;
        bool improved = true;
        while (improved) {
            improved = false;
            for (size_t i = 0; i + 1 < r.size(); ++i)
                for (size_t k = i + 1; k < r.size(); ++k) {
                    node_t a = (i == 0 ? 0 : r[i - 1]);
                    node_t b = (k + 1 == r.size() ? 0 : r[k + 1]);
                    double delta = -inst.d(a, r[i]) - inst.d(r[k], b)
                                   + inst.d(a, r[k]) + inst.d(r[i], b);
                    if (delta < -1e-9) {
                        std::reverse(r.begin() + i, r.begin() + k + 1);
                        improved = true;
                    }
                }
        }
        (void)routeCost;
    }
    return routes;
}
