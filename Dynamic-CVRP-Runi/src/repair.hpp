// ============================================================================
// repair.hpp — event-driven localized repair (ideation §4–§6).
//
//   * Promising-route filter:  R promising for x iff min_{v in R} d(x,v) <= d(x,depot)
//     Two tiers: tier-1 promises DIRECT insertion (load fits), tier-2 only
//     EJECTION-based repair.  Routes are prioritized by delta(x,R).
//   * Global work frontier W=(R,x,h,att,banned); items are re-evaluated at
//     dequeue time against the *current* state (no stale candidates).
//   * Bounded ejection chains: eject y, insert x, migrate (R',y,h+1).
//     Acceptance: net(y) = insCost(x | R-y) - remGain(y)  <= tau * d(x,depot).
//     Fallbacks: next promising attempt -> any-route placement -> singleton.
//   * Batch-first deletion: mark all, compact once per route, enqueue affected.
// ============================================================================
#pragma once

#include <algorithm>
#include "instance.hpp"
#include "routes.hpp"
#include <cmath>
#include <deque>
#include <unordered_map>

struct Params {
    double tau        = 1.0;   // ejection acceptance scale
    int    hmax       = 5;     // GPU local-centroid candidate budget (depot-nearest mode is exhaustive)
    int    amax       = 4;     // max route attempts in the CPU reference
    int    passes     = 6;     // refinement pass budget
    int    refineK    = 5;     // nearest-centroid partner routes per live route
    bool   extraOps   = false; // optional move1/move4 in refinement
    int    simWarps   = 32;    // logical warps for the trace timeline

};

// ---- instrumentation: logical warp schedule (for the Gantt visualizer) ----
struct TaskRec {
    int    eventId;  const char* kind;
    int    worker;   double t0, t1;
    int    routeA, routeB;   double units;
};
struct Tracer {
    bool on = false;
    bool decisions = false;   // human-readable decision log (stderr)
    std::vector<TaskRec> recs;
    std::vector<double>  wfree;
    int    simWarps = 32;
    int    curEvent = 0;
    void beginEvent(int ev) {
        curEvent = ev;
        if (!on) return;
        wfree.assign(simWarps, 0.0);
    }
    void log(const char* kind, double units, int ra, int rb = -1) {
        if (!on || units <= 0) return;
        int w = 0; for (int i = 1; i < simWarps; ++i) if (wfree[i] < wfree[w]) w = i;
        TaskRec t{curEvent, kind, w, wfree[w], wfree[w] + units, ra, rb, units};
        wfree[w] = t.t1;
        recs.push_back(t);
    }
};

// ------------------------------ work frontier ------------------------------
struct WItem {
    int    route;     // target route index (-1 => not yet chosen)
    node_t cust;      // customer awaiting placement
    int    depth;     // ejection-chain depth h
    int    attempts;  // remaining promising-route attempts
    int    banned;    // route to avoid (chain bounce-back guard)
};

class Repair {
  public:
    Repair(const Instance& inst, RouteStore& st, const Params& p, Tracer& tr)
        : I(inst), S(st), P(p), T(tr) {}

    // CPU reference: enumerate EVERY live route.  No geometric
    // no geometric route-pruning is performed. md is retained only
    // for compatibility with the old tracer interface.
    void allRoutes(node_t x, std::vector<int>& idx,
                   std::vector<double>& md, std::vector<char>& tier1) {
        idx.clear(); md.clear(); tier1.clear();
        for (int r = 0; r < (int)S.meta.size(); ++r) {
            const auto& m = S.meta[r];
            if (m.length == 0) continue;
            idx.push_back(r);
            md.push_back(0.0);
            tier1.push_back(m.load + I.demand[x] <= I.capacity + 1e-9);
        }
        (void)x;
    }

    // Best insertion position/cost of x into route r as-is. Returns false if
    // infeasible (capacity) — caller decides policy.
    bool bestInsert(node_t x, int r, int& pos, double& delta) const {
        const auto& m = S.meta[r];
        if (m.load + I.demand[x] > I.capacity + 1e-9) return false;
        delta = 1e30; pos = -1;
        node_t prev = 0;
        for (int k = 0; k <= m.length; ++k) {
            node_t nxt = (k == m.length ? 0 : S.seg[m.offset + k]);
            double dd = I.d(prev, x) + I.d(x, nxt) - I.d(prev, nxt);
            if (dd < delta) { delta = dd; pos = k; }
            prev = nxt;
        }
        return true;
    }

    // Best ejection: remove y from r, then insert x optimally in R\{y}.
    // Returns posRemove (of y), posX (insert slot in the reduced sequence),
    // and net cost change (insert delta minus removal gain).
    bool bestEject(node_t x, int r, int& posY, int& posX, double& net) const {
        const auto& m = S.meta[r];
        if (m.length == 0) return false;
        net = 1e30; posY = posX = -1;
        for (int yy = 0; yy < m.length; ++yy) {
            node_t y = S.seg[m.offset + yy];
            node_t py = (yy == 0 ? 0 : S.seg[m.offset + yy - 1]);
            node_t ny = (yy + 1 == m.length ? 0 : S.seg[m.offset + yy + 1]);
            double remGain = I.d(py, y) + I.d(y, ny) - I.d(py, ny);
            double restLoad = m.load - I.demand[y];
            if (restLoad + I.demand[x] > I.capacity + 1e-9) continue;
            // cheapest insertion of x in the sequence with y removed
            double insBest = 1e30; int insPos = -1;
            node_t prev = 0;
            for (int k = 0; k <= m.length; ++k) {
                if (k == yy) {           // skip gap occupied by y
                    prev = y; continue;
                }
                node_t nxt = (k == m.length ? 0 : S.seg[m.offset + k]);
                double dd = I.d(prev, x) + I.d(x, nxt) - I.d(prev, nxt);
                if (dd < insBest) { insBest = dd; insPos = k; }
                prev = nxt;
            }
            double n2 = insBest - remGain;
            if (n2 < net) { net = n2; posY = yy; posX = insPos; }
        }
        return posY >= 0;
    }

    // Cheapest feasible insertion delta of x across ALL live routes
    // (evaluation only — no mutation).
    bool bestInsertAnywhere(node_t x, int& br, int& bp, double& bd,
                            int avoidRoute = -1) const {
        bd = 1e30; br = -1; bp = -1;
        for (int r = 0; r < (int)S.meta.size(); ++r) {
            if (r == avoidRoute || S.meta[r].length == 0) continue;
            int p; double dl;
            if (bestInsert(x, r, p, dl) && dl < bd) { bd = dl; br = r; bp = p; }
        }
        return br >= 0;
    }

    // Safety net: cheapest feasible insertion across ALL live routes.
    bool placeAnywhere(node_t x, int avoidRoute = -1) {
        int br = -1, bp = -1; double bd = 1e30;
        for (int r = 0; r < (int)S.meta.size(); ++r) {
            if (r == avoidRoute || S.meta[r].length == 0) continue;
            int p; double dl;
            if (bestInsert(x, r, p, dl) && dl < bd) { bd = dl; br = r; bp = p; }
        }
        if (br < 0) return false;
        S.insertAt(br, bp, x);
        S.refreshRoute(I, br);
        T.log("PLACE_ANY", S.meta[br].length, br);
        return true;
    }

    // ------------------------- insertion event -------------------------
    // batch: newly acquired customer slots. Processes the global frontier.
    void insertBatch(const std::vector<node_t>& batch) {
        std::deque<WItem> fq;
        for (node_t x : batch) {
            std::vector<int> idx; std::vector<double> md; std::vector<char> t1;
            allRoutes(x, idx, md, t1);
            T.log("SCAN", (double)S.seg.size() * 32 / 32.0, -1);
            if (idx.empty()) {                       // nothing geometrically close
                if (!placeAnywhere(x))
                    S.newSingleton(x,I);
                continue;
            }
            fq.push_back({idx[0], x, 0, P.amax, -1});
            pendingOrder_[x] = idx;                  // cached priority order
        }
        while (!fq.empty()) {
            WItem it = fq.front(); fq.pop_front();
            if (S.routeOf[it.cust] >= 0) continue;   // placed meanwhile
            int r = it.route;
            if (r < 0 || r >= (int)S.meta.size() || S.meta[r].length == 0 ||
                r == it.banned) {
                if (--it.attempts > 0 && advance_(it)) fq.push_back(it);
                else fallbackPlace_(it.cust);
                continue;
            }
            T.log("OWN_ROUTE", 1, r);
            int ppos; double pdel;
            if (bestInsert(it.cust, r, ppos, pdel)) {         // direct hit
                S.insertAt(r, ppos, it.cust);
                S.refreshRoute(I, r);
                T.log("INSERT", S.meta[r].length, r);
                if (T.decisions)
                    fprintf(stderr, "    [x%3d] DIRECT-INSERT into R%d at pos %d (delta %.0f)\n",
                            it.cust, r, ppos, pdel);
                continue;
            }
            // capacity-infeasible: bounded ejection
            int py, px; double net;
            bool have = bestEject(it.cust, r, py, px, net);
            // acceptance scale = route-local proximity delta(x,R), NOT the
            // depot distance: delta0 can be huge for far arrivals and would
            // make ejection permissive relative to local geometry.
            double localScale = I.dDepot(it.cust);
            // [chain-vs-alternative policy] a greedy step can look free while
            // the ejected customer lands somewhere expensive downstream
            // (capacity-tight instances). Accept only if the WHOLE chain
            //   net(eject y, insert x) + bestReinsertion(y)
            // beats simply placing x at its best feasible spot anywhere.
            int ax, apx; double altX = 1e30;
            bool xHasAlt = bestInsertAnywhere(it.cust, ax, apx, altX, r);
            bool accept = false;
            if (have && it.depth < P.hmax) {
                node_t yCand = S.at(r, py);
                int by, bpy; double yIns = 1e30;
                bool yHasAlt = bestInsertAnywhere(yCand, by, bpy, yIns, r);
                double yReins = yHasAlt ? yIns
                                        : 2.0 * I.dDepot(yCand); // singleton cost
                double chainCost = net + yReins;
                double altCost = xHasAlt ? altX : 1e30;
                accept = xHasAlt ? (chainCost <= altCost + P.tau * std::max(localScale, 1.0))
                                 : true;   // no alternative: chain is the only option
            }
            if (have && accept) {
                node_t y = S.at(r, py);
                if (T.decisions)
                    fprintf(stderr, "    [x%3d] EJECT y=%d from R%d (net %.0f), insert x; chain y->next route (h=%d)\n",
                            it.cust, y, r, net, it.depth + 1);
                S.removeAt(r, py);
                if (px > py) px--;                     // shift after removal
                S.insertAt(r, px, it.cust);
                S.refreshRoute(I, r);
                T.log("EJECT", 2.0 * S.meta[r].length, r);
                // migrate ejected customer to its next route task
                std::vector<int> idx; std::vector<double> md; std::vector<char> t1;
                allRoutes(y, idx, md, t1);
                int nr = -1;
                for (size_t q = 0; q < idx.size(); ++q)
                    if (idx[q] != r) { nr = idx[q]; break; }
                if (nr >= 0) fq.push_back({nr, y, it.depth + 1, 1, r});
                else if (!placeAnywhere(y, r)) S.newSingleton(y,I);
                pendingOrder_.erase(y);
            } else {
                if (T.decisions)
                    fprintf(stderr, "    [x%3d] ejection on R%d rejected/depth-capped (attempts left %d)\n",
                            it.cust, r, it.attempts - 1);
                if (--it.attempts > 0 && advance_(it)) fq.push_back(it);
                else fallbackPlace_(it.cust);
            }
        }
    }

    // ------------------------- deletion event --------------------------
    // Batch-first: mark everything deleted, then compact each touched route
    // exactly once (ideation §6 — avoids sequential reconnect errors).
    void deleteBatch(const std::vector<node_t>& dels, std::vector<int>& affected) {
        std::vector<char> mark(S.routeOf.size(), 0);
        for (node_t v : dels) mark[v] = 1;
        affected.clear();
        for (int r = 0; r < (int)S.meta.size(); ++r) {
            auto& m = S.meta[r];
            bool touched = false;
            int w = 0;
            for (int k = 0; k < m.length; ++k) {
                node_t v = S.seg[m.offset + k];
                if (mark[v]) { S.routeOf[v] = -1; touched = true; }
                else S.seg[m.offset + w++] = v;
            }
            if (touched) {
                m.length = w;
                S.refreshRoute(I, r);
                affected.push_back(r);
                T.log("COMPACT", m.length, r);
            }
        }
        for (node_t v : dels) S.releaseSlot(v, const_cast<Instance&>(I));
        S.dropEmptyRoutes();
        // remap affected indices after drops
        for (auto& a : affected)
            while (a < (int)S.meta.size() && S.meta[a].length == 0) ++a;
        affected.erase(std::remove_if(affected.begin(), affected.end(),
                        [&](int a){ return a >= (int)S.meta.size() || S.meta[a].length == 0; }),
                       affected.end());
    }

  private:
    bool advance_(WItem& it) {                 // move to next route task
        auto f = pendingOrder_.find(it.cust);
        if (f == pendingOrder_.end()) return false;
        auto& lst = f->second;
        while (!lst.empty()) {
            int cand = lst.front();
            if (cand != it.banned &&
                cand < (int)S.meta.size() && S.meta[cand].length > 0) {
                it.route = cand;
                return true;
            }
            lst.erase(lst.begin());
        }
        return false;
    }
    void fallbackPlace_(node_t x) {
        pendingOrder_.erase(x);
        if (!placeAnywhere(x)) {
            S.newSingleton(x,I);
            if (T.decisions)
                fprintf(stderr, "    [x%3d] SINGLETON route created\n", x);
        }
    }

    const Instance& I; RouteStore& S; const Params& P; Tracer& T;
    std::unordered_map<node_t, std::vector<int>> pendingOrder_;
};
