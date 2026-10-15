// ============================================================================
// dynamic_cvrp.cu — event-driven Dynamic CVRP Opti-3 controller.
//
// Event policy: when an input contains both deletions and additions, all DELETE
// events are stably moved before all ADD events.  GPU repair uses the Opti-3
// locality/cache scheduler in gpu_warp.cuh; GPU refinement uses nearest-K route
// partners (default 5) for Move8/9 -> Move1/SWAP* -> Move7.
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "instance.hpp"
#include "routes.hpp"
#include "static_init.hpp"
#include "repair.hpp"
#include "refine.hpp"
#include "gpu_warp.cuh"

static void auditGpuRoutes(
    const char* where,
    const RouteStore& S,
    const std::vector<std::vector<int>>& routes)
{
    std::vector<int> seen(S.activeCust.size(), 0);

    for (const auto& r : routes) {
        for (int v : r) {
            if (v > 0 && v < (int)seen.size())
                seen[v]++;
        }
    }

    int bad = 0;

    for (int v = 1; v < (int)S.activeCust.size(); ++v) {
        if (S.activeCust[v] && seen[v] != 1) {
            std::cerr
                << "[AUDIT " << where << "] active customer "
                << v << " occurs " << seen[v] << " times\n";
            ++bad;
        }

        if (!S.activeCust[v] && seen[v] != 0) {
            std::cerr
                << "[AUDIT " << where << "] INACTIVE customer "
                << v << " still occurs " << seen[v] << " times\n";
            ++bad;
        }
    }

    std::cerr << "[AUDIT " << where << "] bad=" << bad << "\n";
}

static double nowUsSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count();
}

struct Event { char type='D'; double x=0, y=0, q=0; node_t id=-1; };

struct Options {
    std::string vrpFile, solFile, eventsFile, outDir="results/run";
    std::string staticInitOut, traceFile, genSpec;
    std::string engine="gpu";
    uint64_t seed=42;
    int snapshotEvery=0;
    int queueCap=5;
    bool verbose=false, wantSnapVrp=false;
};

static void usage(const char* a0) {
    std::cerr
      << "Usage: " << a0 << " --vrp f.vrp (--sol f.sol | --static-init-out o.sol)\n"
      << "                 [--events f.txt | --gen-events ADDS:DELS:SEED]\n"
      << "                 [--engine cpu|gpu] [--out-dir D]\n"
      << "                 [--snapshot-every k] [--snap-vrp] [--trace f.json]\n"
      << "                 [--tau 1.0] [--hmax 5] [--amax 4] [--passes 6] [--refine-k 5]\n"
      << "                 [--queue-cap 5] [--extra-ops] [--sim-warps 32]\n"
      << "                 [--seed s] [--verbose]\n";
    std::exit(1);
}

static Options parseArgs(int argc,char**argv,Params& P){
    Options o; auto need=[&](int i){if(i+1>=argc)usage(argv[0]);return argv[i+1];};
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a=="--vrp")o.vrpFile=need(i++); else if(a=="--sol")o.solFile=need(i++);
        else if(a=="--events")o.eventsFile=need(i++); else if(a=="--gen-events")o.genSpec=need(i++);
        else if(a=="--static-init-out")o.staticInitOut=need(i++); else if(a=="--engine")o.engine=need(i++);
        else if(a=="--out-dir")o.outDir=need(i++); else if(a=="--snapshot-every")o.snapshotEvery=std::atoi(need(i++));
        else if(a=="--snap-vrp")o.wantSnapVrp=true; else if(a=="--trace")o.traceFile=need(i++);
        else if(a=="--tau")P.tau=std::atof(need(i++)); else if(a=="--hmax")P.hmax=std::atoi(need(i++));
        else if(a=="--amax")P.amax=std::atoi(need(i++)); else if(a=="--passes")P.passes=std::atoi(need(i++));
        else if(a=="--refine-k")P.refineK=std::max(1,std::atoi(need(i++)));
        else if(a=="--queue-cap")o.queueCap=std::atoi(need(i++)); else if(a=="--extra-ops")P.extraOps=true;
        else if(a=="--sim-warps")P.simWarps=std::atoi(need(i++));
        else if(a=="--seed")o.seed=std::strtoull(need(i++),nullptr,10);
        else if(a=="--verbose")o.verbose=true; else usage(argv[0]);
    }
    if(o.vrpFile.empty())usage(argv[0]);
    o.queueCap=std::max(1,std::min(16,o.queueCap));
    return o;
}

static std::vector<Event> genEvents(const Instance&I,int adds,int dels,uint64_t seed){
    std::mt19937_64 rng(seed); double mnx=*std::min_element(I.xs.begin(),I.xs.end()),mxx=*std::max_element(I.xs.begin(),I.xs.end());
    double mny=*std::min_element(I.ys.begin(),I.ys.end()),mxy=*std::max_element(I.ys.begin(),I.ys.end());
    auto pad=[](double a,double b){return .15*(b-a)+1.;}; std::uniform_real_distribution<double>rx(mnx-pad(mnx,mxx),mxx+pad(mnx,mxx)),ry(mny-pad(mny,mxy),mxy+pad(mny,mxy));std::uniform_int_distribution<int>rq(1,12);
    std::vector<Event> ev; int a=0,d=0; while(a<adds||d<dels){if(a<adds){ev.push_back({'A',rx(rng),ry(rng),(double)rq(rng),-1});++a;} if(d<dels&&a>0){ev.push_back({'D',0,0,0,-2});++d;}}
    return ev;
}

static std::vector<Event> readEvents(const std::string&path){
    std::ifstream in(path); if(!in)throw std::runtime_error("cannot open events file "+path); std::vector<Event>ev;std::string line;
    while(std::getline(in,line)){auto f=line.find_first_not_of(" \t");if(f==std::string::npos||line[f]=='#')continue;std::istringstream ss(line.substr(f));Event e{};char t;ss>>t;if(t=='A'){e.type='A';ss>>e.x>>e.y>>e.q;}else{e.type='D';ss>>e.id;}ev.push_back(e);}return ev;
}

// Opti-3 batch policy: cancellations are applied before arrivals.  The final
// customer set is unchanged, but deletes get the cheapest direct-compaction
// path first and freed stable IDs may be recycled by the following ADD batch.
static void reorderDeletesBeforeAdds(std::vector<Event>& ev){
    std::stable_partition(ev.begin(), ev.end(), [](const Event& e){ return e.type=='D'; });
}

static double routeCost(const Instance&I,const std::vector<node_t>&r){if(r.empty())return 0;double c=I.d(0,r[0]);for(size_t i=1;i<r.size();++i)c+=I.d(r[i-1],r[i]);return c+I.d(r.back(),0);}
static double stateCost(const Instance&I,const RouteStore&S){double c=0;for(const auto&m:S.meta){std::vector<node_t>r(S.seg.begin()+m.offset,S.seg.begin()+m.offset+m.length);c+=routeCost(I,r);}return c;}

static void writeSnapshot(const std::string&base,const Instance&I,const RouteStore&S,bool withVrp){
    std::vector<node_t>ids{0};for(node_t v=1;v<(node_t)I.xs.size();++v)if(v<(node_t)S.activeCust.size()&&S.activeCust[v])ids.push_back(v);
    std::vector<int>map(I.xs.size(),-1);for(size_t k=0;k<ids.size();++k)map[ids[k]]=(int)k;
    std::vector<std::vector<node_t>>out;for(const auto&m:S.meta){std::vector<node_t>r;for(int k=0;k<m.length;++k)r.push_back(map[S.seg[m.offset+k]]);if(!r.empty())out.push_back(std::move(r));}
    Instance J;J.capacity=I.capacity;J.n=(int)ids.size()-1;J.name=I.name;J.xs.resize(ids.size());J.ys.resize(ids.size());J.demand.resize(ids.size());for(size_t k=0;k<ids.size();++k){J.xs[k]=I.xs[ids[k]];J.ys[k]=I.ys[ids[k]];J.demand[k]=I.demand[ids[k]];}
    double cost=solution_cost(out,J);write_sol(base+".sol",out,J,cost);if(!withVrp)return;std::ofstream v(base+".vrp");v<<"NAME : "<<I.name<<"\nCOMMENT : dynamic snapshot\nTYPE : CVRP\nDIMENSION : "<<ids.size()<<"\nEDGE_WEIGHT_TYPE : EUC_2D\nCAPACITY : "<<I.capacity<<"\nNODE_COORD_SECTION\n";for(size_t k=0;k<ids.size();++k)v<<(k+1)<<'\t'<<J.xs[k]<<'\t'<<J.ys[k]<<'\n';v<<"DEMAND_SECTION\n";for(size_t k=0;k<ids.size();++k)v<<(k+1)<<'\t'<<J.demand[k]<<'\n';v<<"DEPOT_SECTION\n1\n-1\nEOF\n";
}

static void validate(const Instance&I,const RouteStore&S,int ev){std::vector<int>seen(S.routeOf.size(),0);int placed=0;for(const auto&m:S.meta){double ld=0;for(int k=0;k<m.length;++k){int v=S.seg[m.offset+k];if(v<=0||v>=(int)seen.size()||++seen[v]>1){std::cerr<<"[INVALID] event "<<ev<<": duplicate/bad customer "<<v<<"\n";std::exit(2);}ld+=I.demand[v];if(I.demand[v]<0){std::cerr<<"[INVALID] negative demand\n";std::exit(2);}}if(ld>I.capacity+1e-9){std::cerr<<"[INVALID] event "<<ev<<": route load "<<ld<<" > Q\n";std::exit(2);}}
    for(int v=1;v<(int)S.activeCust.size();++v)if(S.activeCust[v]){if(S.routeOf[v]<0){std::cerr<<"[INVALID] event "<<ev<<": active customer "<<v<<" unplaced\n";std::exit(2);}++placed;}
    if((int)S.seg.size()!=placed){std::cerr<<"[INVALID] event "<<ev<<": seg size mismatch\n";std::exit(2);}
}

#ifndef DCVRP_CPU_ONLY
static std::vector<node_t> activeUnplaced(const RouteStore&S){std::vector<node_t>u;for(node_t v=1;v<(node_t)S.activeCust.size();++v)if(S.activeCust[v]&&S.routeOf[v]<0)u.push_back(v);return u;}
#endif

int main(int argc,char**argv){
    Params P;Options opt=parseArgs(argc,argv,P);Instance I;read_vrp(opt.vrpFile,I);
    std::cerr<<"[init] "<<I.name<<": "<<I.n<<" customers, Q="<<I.capacity<<"\n";
    if(!opt.staticInitOut.empty()){auto routes=static_mds_init(I,opt.seed);double c=solution_cost(routes,I);write_sol(opt.staticInitOut,routes,I,c);std::cerr<<"[static-init] routes="<<routes.size()<<" cost="<<c<<" -> "<<opt.staticInitOut<<"\n";return 0;}

    std::vector<std::vector<node_t>>initRoutes;read_sol(opt.solFile,initRoutes);RouteStore S;S.init(I,initRoutes);validate(I,S,0);
    std::vector<Event>events;if(!opt.genSpec.empty()){int a=0,d=0;sscanf(opt.genSpec.c_str(),"%d:%d:%llu",&a,&d,&opt.seed);events=genEvents(I,a,d,opt.seed);}else if(!opt.eventsFile.empty())events=readEvents(opt.eventsFile);
    reorderDeletesBeforeAdds(events);

#ifndef DCVRP_CPU_ONLY
    std::unique_ptr<dcvrp_cuda::GpuDynamicEngine>gpu;
    if(opt.engine=="gpu"){
        CUDA_OK(cudaFree(0)); // CUDA context warm-up is outside event timing.
        int adds=0,maxBatch=1;int run=0;for(auto&e:events){if(e.type=='A'){++adds;++run;maxBatch=std::max(maxBatch,run);}else run=0;}
        int maxCust=(int)I.xs.size()+adds+8;int maxRoutes=(int)S.meta.size()+adds+8;int maxLen=64;for(auto&m:S.meta)maxLen=std::max(maxLen,m.length);maxLen=std::min(std::max(maxLen+adds+4,64),std::max(64,maxCust));
        // Opti-3 bounds normal LOCAL-mode retries to hmax cached routes (5 by
        // default). DEPOT-nearest customers retain the exhaustive visited-route
        // fallback, so keep the conservative frontier allocation guard.
        size_t fcap=(size_t)std::max<long long>(4096LL,
            2LL * std::max(1,maxCust) * std::max(1,maxRoutes));
        gpu=std::make_unique<dcvrp_cuda::GpuDynamicEngine>();
        gpu->init(I,S,maxCust,maxRoutes,maxLen,(int)std::min<size_t>(fcap,1<<26),opt.queueCap);
        std::cerr<<"[gpu] scheduler=adaptive multi-warp centroid-priority"
                 <<" target=8 routes/thread local-hmax="<<P.hmax<<" queue-cap="<<opt.queueCap
                 <<"; ready-route dispatch; refine-nearest-k="<<P.refineK<<"\n";
    }
#else
    bool useGpu=false;
#endif

    std::filesystem::create_directories(opt.outDir);
    std::ofstream csv(opt.outDir+"/summary.csv");
    csv << "metric,value\n";

    // One benchmark invocation is one dynamic transaction:
    //   1) collect/apply ALL deletes, regardless of their input order;
    //   2) collect/apply ALL adds;
    //   3) run refinement ONCE on the resulting solution.
    // This avoids paying refinement once after DELETE and again after ADD in a
    // mixed workload.
    std::vector<Event> deleteEvents, addEvents;
    deleteEvents.reserve(events.size());
    addEvents.reserve(events.size());
    for (const auto& e : events) {
        if (e.type == 'D') deleteEvents.push_back(e);
        else if (e.type == 'A') addEvents.push_back(e);
    }

    std::mt19937_64 rng(opt.seed);
    const double costBefore = stateCost(I,S);

    // Resolve random delete requests (-2) once, without selecting the same
    // customer twice inside this transaction.
    std::vector<int> deleteIds;
    deleteIds.reserve(deleteEvents.size());
    std::unordered_set<int> selectedDeletes;
    for (const auto& e : deleteEvents) {
        int id = e.id;
        if (id == -2) {
            std::vector<int> available;
            for (int v=1; v<(int)S.activeCust.size(); ++v)
                if (S.activeCust[v] && !selectedDeletes.count(v))
                    available.push_back(v);
            if (available.empty()) continue;
            std::uniform_int_distribution<size_t> pick(0, available.size()-1);
            id = available[pick(rng)];
        }
        if (id>0 && id<(int)S.activeCust.size() && S.activeCust[id] &&
            selectedDeletes.insert(id).second)
            deleteIds.push_back(id);
    }

    double deleteUs=0.0, addUs=0.0, refineUs=0.0;
    double schedulerTaskAvgUs=0.0, workerTaskAvgUs=0.0, avgActiveWorkers=0.0;
    double avgActiveSchedulerWarps=0.0;
    unsigned long long maxActiveWorkers=0, maxActiveSchedulerWarps=0;
    double move89Us=0.0, move1SwapUs=0.0, move7Us=0.0;

    if (opt.engine=="cpu") {
        if (!deleteIds.empty()) {
            auto t0=std::chrono::steady_clock::now();
            std::vector<int> affected;
            Tracer tr;
            Repair rep(I,S,P,tr);
            rep.deleteBatch(deleteIds,affected);
            deleteUs=nowUsSince(t0);
        }

        if (!addEvents.empty()) {
            auto t0=std::chrono::steady_clock::now();
            std::vector<node_t> addIds;
            addIds.reserve(addEvents.size());
            for (const auto& e : addEvents)
                addIds.push_back(S.acquireSlot(e.x,e.y,e.q,I));
            Tracer tr;
            Repair rep(I,S,P,tr);
            rep.insertBatch(addIds);
            addUs=nowUsSince(t0);
        }

        auto t1=std::chrono::steady_clock::now();
        Tracer tr2;
        Refiner ref(I,S,P,tr2);
        ref.run({});
        refineUs=nowUsSince(t1);
    }
#ifndef DCVRP_CPU_ONLY
    else if (gpu) {
        // DELETE phase: all deletions first, no refinement here.
        if (!deleteIds.empty()) {
            auto t0=std::chrono::steady_clock::now();
            gpu->eraseBatch(deleteIds);
            for (int id : deleteIds) S.releaseSlot(id,I);
            deleteUs=nowUsSince(t0);
        }

        // ADD phase: all additions after deletes, still no refinement yet.
        if (!addEvents.empty()) {
            auto t0=std::chrono::steady_clock::now();
            std::vector<int> addIds;
            addIds.reserve(addEvents.size());
            for (const auto& e : addEvents)
                addIds.push_back(S.acquireSlot(e.x,e.y,e.q,I));

            gpu->activate(addIds);
            gpu->insertBatch(addIds,P.hmax,P.tau);
            addUs=nowUsSince(t0);

            auto gpuUnplaced = gpu->unplaced(0);
            std::cerr << "[GPU-UNPLACED AFTER-REPAIR] count=" << gpuUnplaced.size();
            for (const auto& w : gpuUnplaced) std::cerr << " " << w.cust;
            std::cerr << "\n";

            const auto& rp=gpu->repairProfile();
            // One scheduler work = one successful scheduler -> route-queue
            // assignment, with blocked/retry scheduler overhead amortized over
            // those successful assignments. This is directly comparable to
            // worker_task_avg_us for one route-worker task.
            schedulerTaskAvgUs=rp.schedulerUsPerAssignment;
            workerTaskAvgUs=rp.workerUsPerTask;
            avgActiveWorkers=rp.avgActiveWorkers;
            maxActiveWorkers=rp.maxActiveWorkers;
            avgActiveSchedulerWarps=rp.avgActiveSchedulerWarps;
            maxActiveSchedulerWarps=rp.maxActiveSchedulerWarps;
        }

        // Audit the combined DELETE -> ADD repaired state before the one refine.
        auto beforeRefine=gpu->downloadRoutes();
        auditGpuRoutes("AFTER-REPAIR",S,beforeRefine);

        // Exactly ONE refinement for the whole transaction.
        auto t1=std::chrono::steady_clock::now();
        gpu->refineAll(P.passes,P.refineK);
        refineUs=nowUsSince(t1); // excludes debug download/audit below

        const auto& rf=gpu->refineProfile();
        move89Us=rf.move89Us;
        move1SwapUs=rf.move1SwapUs;
        move7Us=rf.move7Us;

        auto afterRefine=gpu->downloadRoutes();
        auditGpuRoutes("AFTER-REFINE",S,afterRefine);
        S.rebuildFromRoutes(I,afterRefine);
    }
#endif

    validate(I,S,1);
    const double costAfter=stateCost(I,S);

    // Vertical profiler output: two columns only, and only the metrics requested.
    csv << std::fixed << std::setprecision(6);
    csv << "delete_total_ms," << deleteUs/1000.0 << '\n';
    csv << "add_total_ms," << addUs/1000.0 << '\n';
    csv << "scheduler_task_avg_us," << schedulerTaskAvgUs << '\n';
    csv << "worker_task_avg_us," << workerTaskAvgUs << '\n';
    csv << "avg_active_route_worker_warps," << avgActiveWorkers << '\n';
    csv << "max_active_route_worker_warps," << maxActiveWorkers << '\n';
    csv << "avg_active_scheduler_warps," << avgActiveSchedulerWarps << '\n';
    csv << "max_active_scheduler_warps," << maxActiveSchedulerWarps << '\n';
    csv << "refine_total_ms," << refineUs/1000.0 << '\n';
    csv << "move89_ms," << move89Us/1000.0 << '\n';
    csv << "move1swap_ms," << move1SwapUs/1000.0 << '\n';
    csv << "move7_ms," << move7Us/1000.0 << '\n';
    csv.close();

    if (opt.snapshotEvery)
        writeSnapshot(opt.outDir+"/snap_1",I,S,opt.wantSnapVrp);
    writeSnapshot(opt.outDir+"/final",I,S,opt.wantSnapVrp);

    std::cerr << "[transaction] deletes=" << deleteIds.size()
              << " adds=" << addEvents.size()
              << " delete=" << deleteUs/1000.0 << "ms"
              << " add=" << addUs/1000.0 << "ms"
              << " refine=" << refineUs/1000.0 << "ms"
              << " cost " << std::lround(costBefore) << "->" << std::lround(costAfter)
              << " routes=" << S.meta.size() << "\n";
    std::cerr << "[done] profile -> " << opt.outDir << "/summary.csv\n";
    return 0;
}
