// ============================================================================
// refine.hpp — targeted inter-route refinement (ideation §7).
//
// Only the 2-opt* family is mandatory:
//   move8: tail exchange with subpath reversal bookkeeping (Vidal's op. 8)
//   move9: cross-style tail exchange without inversion tracking   (Vidal's op. 9)
// Optional (-extra-ops): move1 (relocate) and move4 (swap) are evaluated
// on the same exhaustive route-pair frontier in the CPU reference.
//
// Parallel protocol (CPU twin of the GPU scheme):
//   each pair task is evaluated independently -> candidates sorted by delta ->
//   non-conflicting moves applied greedily under route ownership (a route is
//   touched at most once per pass) -> modified routes seed the next pass.
// ============================================================================
#pragma once

#include "repair.hpp"
#include "gpu_warp.cuh"
#include <cmath>
#include <cstring>
#include <unordered_set>

struct PairMove {
    double delta = 0;
    int    kind  = 0;      // 8, 9, 1, 4
    int    i = -1, j = -1;  // cut positions / node positions
};

class Refiner {
  public:
    Refiner(const Instance& inst, RouteStore& st, const Params& p, Tracer& tr)
        : I(inst), S(st), P(p), T(tr) {}

    inline node_t nodeAt(int r, int pos) const {   // depot-padded accessor
        const auto& m = S.meta[r];
        return pos < m.length ? S.seg[m.offset + pos] : 0;
    }
    inline double loadUpto(int r, int posIncl) const {
        const auto& m = S.meta[r];
        double l = 0;
        for (int k = 0; k <= posIncl && k < m.length; ++k)
            l += I.demand[S.seg[m.offset + k]];
        return l;
    }

    // Exhaustive reference evaluator for one unordered pair. Cut positions are
    // prefix lengths 0..route_length, so depot-adjacent edges are legal.
    bool evalPair(int R, int Sv, PairMove& out) {
        const auto& m1=S.meta[R]; const auto& m2=S.meta[Sv];
        bool found=false; out.delta=-1e-9;
        pre1_.assign(m1.length+1,0); pre2_.assign(m2.length+1,0);
        for(int k=0;k<m1.length;++k) pre1_[k+1]=pre1_[k]+I.demand[S.seg[m1.offset+k]];
        for(int k=0;k<m2.length;++k) pre2_[k+1]=pre2_[k]+I.demand[S.seg[m2.offset+k]];
        for(int i=0;i<=m1.length;++i){
            node_t a=(i==0?0:nodeAt(R,i-1));
            node_t an=(i==m1.length?0:nodeAt(R,i));
            node_t lastA=(m1.length?nodeAt(R,m1.length-1):0);
            double p1=pre1_[i], suf1=pre1_.back()-p1;
            for(int j=0;j<=m2.length;++j){
                node_t c=(j==0?0:nodeAt(Sv,j-1));
                node_t cn=(j==m2.length?0:nodeAt(Sv,j));
                node_t firstB=(j<m2.length?nodeAt(Sv,j):0);
                node_t lastB=(m2.length?nodeAt(Sv,m2.length-1):0);
                double p2=pre2_[j], suf2=pre2_.back()-p2;
                if(p1+suf2>I.capacity+1e-9 || p2+suf1>I.capacity+1e-9) continue;
                if(i==m1.length && j==m2.length) continue;

                // Move 9: simple tail exchange.
                double newSimpleA=(j<m2.length? I.d(a,firstB)+I.d(lastB,0):I.d(a,0));
                double newSimpleB=(i<m1.length? I.d(c,(i<m1.length?nodeAt(R,i):0))+I.d(lastA,0):I.d(c,0));
                double oldA=I.d(a,an)+(i<m1.length?I.d(lastA,0):0);
                double oldB=I.d(c,cn)+(j<m2.length?I.d(lastB,0):0);
                double d9=(newSimpleA+newSimpleB)-(oldA+oldB);
                if(d9<out.delta){out={d9,9,i,j};found=true;}

                // Move 8: reverse both exchanged tails.
                double newRevA=(j<m2.length? I.d(a,lastB)+I.d(firstB,0):I.d(a,0));
                double newRevB=(i<m1.length? I.d(c,lastA)+I.d(nodeAt(R,i),0):I.d(c,0));
                double d8=(newRevA+newRevB)-(oldA+oldB);
                if(d8<out.delta){out={d8,8,i,j};found=true;}
            }
        }
        return found;
    }

    void applyPairMove(int R, int Sv, const PairMove& mv) {
        if (mv.kind == 8)  tailExchange(R, Sv, mv.i, mv.j, /*reverse=*/true);
        else if (mv.kind == 9) tailExchange(R, Sv, mv.i, mv.j, /*reverse=*/false);
        else if (mv.kind == 1) relocate(R, Sv, mv.i, mv.j);
        else if (mv.kind == 4) swapNodes(R, Sv, mv.i, mv.j);
        T.log("COMMIT", 4, R, Sv);
    }

    // ------------------------- driver -------------------------
    // CPU twin: evaluate EVERY unordered live route pair.  No centroid, exact
    // distance, or k-partner pruning is performed.
    void run(const std::vector<int>& /*seeds*/) {
        for (int pass = 0; pass < P.passes; ++pass) {
            struct Cand { double delta; int kind, R, S_, i, j; };
            std::vector<Cand> cands;
            for (int r = 0; r < (int)S.meta.size(); ++r) {
                if (S.meta[r].length == 0) continue;
                for (int s = r + 1; s < (int)S.meta.size(); ++s) {
                    if (S.meta[s].length == 0) continue;
                    PairMove mv;
                    if (evalPair(r, s, mv))
                        cands.push_back({mv.delta, mv.kind, r, s, mv.i, mv.j});
                }
            }
            std::sort(cands.begin(), cands.end(),
                      [](const Cand& a, const Cand& b){ return a.delta < b.delta; });
            std::unordered_set<int> touched;
            bool improved = false;
            for (const auto& c : cands) {
                if (touched.count(c.R) || touched.count(c.S_)) continue;
                applyPairMove(c.R, c.S_, {c.delta, c.kind, c.i, c.j});
                touched.insert(c.R); touched.insert(c.S_);
                improved = true;
            }
            if (!improved) break;
        }
        // Final intra-route 2-opt is intentionally retained on the CPU fallback.
        for (int r = 0; r < (int)S.meta.size(); ++r) {
            bool improved = true;
            while (improved) {
                improved = false;
                const auto& m = S.meta[r];
                double best = -1e-9; int bi=-1,bk=-1;
                for (int i=0;i+1<m.length;++i) {
                    node_t a=nodeAt(r,i), b=nodeAt(r,i+1);
                    for (int k=i+1;k<m.length;++k) {
                        node_t c=nodeAt(r,k), d=nodeAt(r,k+1);
                        double dd=I.d(a,c)+I.d(b,d)-I.d(a,b)-I.d(c,d);
                        if (dd<best){best=dd;bi=i;bk=k;}
                    }
                }
                if (bi>=0) {
                    std::reverse(S.seg.begin()+m.offset+bi+1,
                                 S.seg.begin()+m.offset+bk+1);
                    S.refreshRoute(I,r);
                    improved=true;
                }
            }
        }
    }

  private:
    int findShifted_(int r) {                    // index may shift after compaction
        return std::min(r, (int)S.meta.size() - 1);
    }

    // Single source of truth for whole-route rewrites. Always correct: resize
    // the segment once, fix every offset, memcpy content, remap routeOf.
    void replaceRoute(int r, const std::vector<node_t>& content) {
        auto& m = S.meta[r];
        for (int k = 0; k < m.length; ++k) {
            node_t v = S.seg[m.offset + k];
            if (S.routeOf[v] == r)             // don't clobber newer owners
                S.routeOf[v] = -1;             // (paired rewrite ordering)
        }
        int oldLen = m.length, newLen = (int)content.size();
        int anchor = m.offset + oldLen;          // boundary in seg
        if (newLen > oldLen)
            S.seg.insert(S.seg.begin() + anchor, newLen - oldLen, 0);
        else if (newLen < oldLen)
            S.seg.erase(S.seg.begin() + m.offset + newLen,     // own extent ONLY
                        S.seg.begin() + m.offset + oldLen);
        for (auto& mm : S.meta)                  // shift later routes
            if (&mm != &m && mm.offset > m.offset)
                mm.offset += newLen - oldLen;
        m.length = newLen;
        ensureSeg_(m.offset + newLen);
        std::memcpy(&S.seg[m.offset], content.data(), sizeof(node_t) * newLen);
        for (int k = 0; k < newLen; ++k) S.routeOf[content[k]] = r;
    }
    // Exchange suffixes after prefix lengths i and j. reverse=true reverses
    // both exchanged tails. Supports i/j == 0 and i/j == route length.
    void tailExchange(int R,int Sv,int i,int j,bool reverse){
        auto oldR=std::vector<node_t>(S.seg.begin()+S.meta[R].offset,S.seg.begin()+S.meta[R].offset+S.meta[R].length);
        auto oldS=std::vector<node_t>(S.seg.begin()+S.meta[Sv].offset,S.seg.begin()+S.meta[Sv].offset+S.meta[Sv].length);
        std::vector<node_t> ar(oldR.begin(),oldR.begin()+i), as(oldS.begin(),oldS.begin()+j);
        std::vector<node_t> br(oldR.begin()+i,oldR.end()), bs(oldS.begin()+j,oldS.end());
        if(reverse){std::reverse(br.begin(),br.end());std::reverse(bs.begin(),bs.end());}
        ar.insert(ar.end(),bs.begin(),bs.end()); as.insert(as.end(),br.begin(),br.end());
        replaceRoute(R,ar); replaceRoute(Sv,as); S.refreshRoute(I,R);S.refreshRoute(I,Sv);S.dropEmptyRoutes();
    }

    void relocate(int R, int Sv, int iu, int jv) {
        node_t u = S.at(R, iu);
        S.removeAt(R, iu);
        S.refreshRoute(I, R);
        if (!S.meta[R].length) { S.dropEmptyRoutes(); Sv = findShifted_(Sv); }
        S.insertAt(Sv, std::min(jv + 1, S.meta[Sv].length), u);
        S.refreshRoute(I, Sv);
        S.dropEmptyRoutes();
    }

    void swapNodes(int R, int Sv, int iu, int jv) {
        node_t u = S.at(R, iu), v = S.at(Sv, jv);
        S.seg[S.meta[R].offset + iu] = v;
        S.seg[S.meta[Sv].offset + jv] = u;
        S.routeOf[v] = R; S.routeOf[u] = Sv;
        S.refreshRoute(I, R); S.refreshRoute(I, Sv);
    }

    void ensureSeg_(int sz) { if ((int)S.seg.size() < sz) S.seg.resize(sz, 0); }

    const Instance& I; RouteStore& S; const Params& P; Tracer& T;
    std::vector<double> pre1_, pre2_;
};
