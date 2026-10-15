// ============================================================================
// routes.hpp — dynamic route storage for the event-driven CVRP repair engine.
//
// Design (per ideation §8.2):
//   * Routes live in one contiguous segment buffer so that warp-cooperative
//     GPU kernels can stream over them CSR-style.
//   * Customer ids are STABLE across deletions (free-list recycling), unlike
//     the dense re-indexing used by the d-cvrp baseline (O(N) per delete).
//   * Per-route metadata: offset, length, load, centroid, version counter
//     (used by the conflict-resolution protocol during parallel commits).
// ============================================================================
#pragma once

#include <cstring>
#include "instance.hpp"

struct RouteMeta {
    int      offset;        // start inside segment buffer
    int      length;        // number of customers
    double   load;          // sum of demands
    double   cenX, cenY;    // route centroid INCLUDING depot once
    uint32_t version;       // bumped on every mutation
};

class RouteStore {
  public:
    std::vector<node_t>   seg;      // flattened routes (customers only)
    std::vector<RouteMeta> meta;    // one per live route
    std::vector<int>      routeOf;  // customer -> route index (-1 if unplaced)
    std::vector<char>     activeCust;
    std::vector<int>      freeIds;  // recycled customer slots
    int nextId = 1;                 // highest-ever id + 1

    void init(const Instance& inst, const std::vector<std::vector<node_t>>& routes) {
        size_t total = 0;
        for (auto& r : routes) total += r.size();
        seg.clear();  seg.reserve(total + 256);
        meta.clear(); meta.reserve(routes.size() + 16);
        routeOf.assign(inst.xs.size(), -1);
        activeCust.assign(inst.xs.size(), 0);
        freeIds.clear();
        nextId = (int)inst.xs.size();
        for (size_t r = 0; r < routes.size(); ++r) {
            RouteMeta m{};
            m.offset = (int)seg.size();
            m.length = (int)routes[r].size();
            m.version = 0;
            double cx = inst.xs[0], cy = inst.ys[0]; // include depot once
            for (node_t v : routes[r]) {
                seg.push_back(v);
                routeOf[v] = (int)r;
                activeCust[v] = 1;
                m.load += inst.demand[v];
                cx += inst.xs[v]; cy += inst.ys[v];
            }
            m.cenX = cx / (m.length + 1); m.cenY = cy / (m.length + 1);
            meta.push_back(m);
        }
        recomputeOffsets();
    }

    // Re-pack segments contiguously after arbitrary add/remove of customers.
    void recomputeOffsets() {
        int off = 0;
        for (auto& m : meta) {
            if (m.length == 0) continue;
            if (m.offset != off)
                std::memmove(&seg[off], &seg[m.offset], sizeof(node_t) * m.length);
            m.offset = off;
            off += m.length;
        }
        seg.resize(off);
    }

    void refreshRoute(const Instance& inst, int r) {
        auto& m = meta[r];
        m.load = 0; double cx = inst.xs[0], cy = inst.ys[0]; // include depot once
        for (int k = 0; k < m.length; ++k) {
            node_t v = seg[m.offset + k];
            m.load += inst.demand[v];
            cx += inst.xs[v]; cy += inst.ys[v];
        }
        m.cenX = cx / (m.length + 1); m.cenY = cy / (m.length + 1);
        m.version++;
    }

    inline node_t at(int r, int pos) const { return seg[meta[r].offset + pos]; }

    // Insert customer x at position `pos` of route r (0..length).
    void insertAt(int r, int pos, node_t x) {
        auto& m = meta[r];
        seg.insert(seg.begin() + m.offset + pos, x);
        for (auto& mm : meta)              // shift offsets after insertion point
            if (&mm != &m && mm.offset >= m.offset + pos) mm.offset++;
        m.length++;
        routeOf[x] = r;
    }

    // Remove customer at position `pos` of route r.
    void removeAt(int r, int pos) {
        auto& m = meta[r];
        routeOf[seg[m.offset + pos]] = -1;
        seg.erase(seg.begin() + m.offset + pos);
        for (auto& mm : meta)
            if (&mm != &m && mm.offset > m.offset + pos) mm.offset--;
        m.length--;
    }

    // Append x as a brand-new singleton route; returns new route index.
    int newSingleton(node_t x, const Instance& inst) {
        RouteMeta m{};
        m.offset = (int)seg.size();
        m.length = 1;
        m.load = inst.demand[x];
        m.cenX = 0.5 * (inst.xs[0] + inst.xs[x]);
        m.cenY = 0.5 * (inst.ys[0] + inst.ys[x]);
        m.version = 0;
        meta.push_back(m);
        seg.push_back(x);
        routeOf[x] = (int)meta.size() - 1;
        return (int)meta.size() - 1;
    }

    // Allocate a fresh/recycled customer slot for a new arrival.
    node_t acquireSlot(double x, double y, double q, Instance& inst) {
        node_t v;
        if (!freeIds.empty()) {
            v = freeIds.back(); freeIds.pop_back();
        } else {
            v = nextId++;
            inst.xs.push_back(0); inst.ys.push_back(0); inst.demand.push_back(0);
            routeOf.push_back(-1); activeCust.push_back(0);
        }
        inst.xs[v] = x; inst.ys[v] = y; inst.demand[v] = q;
        activeCust[v] = 1;
        inst.n++;
        return v;
    }

    void releaseSlot(node_t v, Instance& inst) {
        activeCust[v] = 0;
        inst.demand[v] = 0;
        freeIds.push_back(v);
        inst.n--;
    }


    // Rebuild host route storage from a GPU/parallel route snapshot without
    // changing stable customer-slot state (activeCust/freeIds/nextId).
    // Route indices may be compacted; all routeOf entries are rebuilt.
    void rebuildFromRoutes(const Instance& inst,
                           const std::vector<std::vector<node_t>>& routes) {
        seg.clear();
        meta.clear();
        routeOf.assign(routeOf.size(), -1);
        seg.reserve(seg.capacity() ? seg.capacity() : 1024);
        meta.reserve(routes.size());
        for (const auto& r : routes) {
            if (r.empty()) continue;
            RouteMeta m{};
            m.offset = (int)seg.size();
            m.length = (int)r.size();
            m.version = 0;
            m.cenX = inst.xs[0]; m.cenY = inst.ys[0]; // include depot once
            for (node_t v : r) {
                if (v <= 0 || v >= (int)inst.demand.size()) {
                    std::fprintf(stderr,
                        "[BUG] invalid customer %d returned by GPU\n", v);
                    std::abort();
                }

                if (v >= (int)activeCust.size() || !activeCust[v]) {
                    std::fprintf(stderr,
                        "[BUG] GPU route contains inactive/free customer %d\n", v);
                    std::abort();
                }

                seg.push_back(v);
                routeOf[v] = (int)meta.size();

                // DO NOT modify activeCust here.
                m.load += inst.demand[v];
                m.cenX += inst.xs[v];
                m.cenY += inst.ys[v];
            }
            if (m.length > 0) {
                m.cenX /= (m.length + 1);
                m.cenY /= (m.length + 1);
                meta.push_back(m);
            }
        }
        recomputeOffsets();
    }

    // Drop zero-length routes, compacting indices; remaps routeOf.
    void dropEmptyRoutes() {
        recomputeOffsets();                      // canonicalize live segments
        std::vector<RouteMeta> nm;
        std::vector<int> remap(meta.size(), -1);
        for (size_t r = 0; r < meta.size(); ++r)
            if (meta[r].length > 0) { remap[r] = (int)nm.size(); nm.push_back(meta[r]); }
        if (nm.size() == meta.size()) return;
        meta.swap(nm);
        for (size_t v = 0; v < routeOf.size(); ++v)
            if (routeOf[v] >= 0) routeOf[v] = remap[routeOf[v]];
    }
};
