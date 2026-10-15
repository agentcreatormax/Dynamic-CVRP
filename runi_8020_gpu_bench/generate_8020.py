#!/usr/bin/env python3
"""Make deterministic 80%-active/20%-ADD CVRP benchmarks from archived complete VRP/solution pairs.
No HGS, no optimization, and no new random customers. All original customer coordinates,
demands and distances are preserved. The first 80% of original internal customer IDs
remain active; the last 20% are reintroduced with A x y demand events.
"""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = {'XL-n10001-k1570': 10000, 'Flanders2': 30000}


def parse_vrp(path):
    values = {}
    coords = {}
    demands = {}
    mode = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line: continue
        if line in ('NODE_COORD_SECTION', 'DEMAND_SECTION', 'DEPOT_SECTION', 'EOF'):
            mode = {'NODE_COORD_SECTION':'coords', 'DEMAND_SECTION':'demands'}.get(line)
            continue
        if mode in ('coords', 'demands'):
            x = line.split()
            if mode == 'coords' and len(x)>=3:
                coords[int(x[0])] = (x[1],x[2])
            elif mode == 'demands' and len(x)>=2:
                demands[int(x[0])] = x[1]
        elif mode is None:
            for key in ('NAME','DIMENSION','CAPACITY'):
                m = re.match(r'^'+key+r'\s*:\s*(.*)$',line)
                if m: values[key]=m.group(1).strip()
    return values, coords, demands


def parse_sol(path):
    routes=[]
    for line in path.read_text().splitlines():
        m=re.match(r'^Route\s+#\d+:\s*(.*)$',line.strip())
        if m: routes.append([int(n) for n in m.group(1).split()])
    return routes


def rounded_cost(routes, coords):
    def d(a, b):
        dx = float(coords[a+1][0])-float(coords[b+1][0])
        dy = float(coords[a+1][1])-float(coords[b+1][1])
        return math.floor(math.hypot(dx,dy)+0.5)
    result=0
    for r in routes:
        if not r: continue
        for a,b in zip([0]+r, r+[0]): result+=d(a,b)
    return result


def create_case(name,n):
    case=ROOT/name
    reference=case/'reference'
    vdata,xy,dem=parse_vrp(reference/'full.vrp')
    routes=parse_sol(reference/'full.sol')
    assert int(vdata['DIMENSION'])==n+1, (name,vdata)
    assert set(xy)==set(range(1,n+2)), 'coordinates missing/duplicated or noncontiguous'
    assert set(dem)==set(range(1,n+2)), 'demands missing/duplicated or noncontiguous'
    all_cust=[c for r in routes for c in r]
    assert len(all_cust)==n and set(all_cust)==set(range(1,n+1)), (name,len(all_cust),len(set(all_cust)))
    capacity=float(vdata['CAPACITY'])
    assert all(sum(float(dem[c+1]) for c in r)<=capacity+1e-8 for r in routes)
    keep=int(n*.8)
    initial_routes=[[c for c in r if c<=keep] for r in routes]
    initial_routes=[r for r in initial_routes if r]
    initial_cust=[c for r in initial_routes for c in r]
    assert len(initial_cust)==keep and set(initial_cust)==set(range(1,keep+1))
    assert all(sum(float(dem[c+1]) for c in r)<=capacity+1e-8 for r in initial_routes)

    dest=case/'case'; dest.mkdir(exist_ok=True)
    # Initial VRP includes depot + first 80% of ORIGINAL customer ids. Since
    # their ids are contiguous, the solver's 0-based internal ids stay equal
    # to the original full-instance customer ids.
    with (dest/'initial.vrp').open('w') as f:
        f.write(f'NAME : {name}_initial80\nCOMMENT : prefix-first 80pct of source instance; full reference route-filtered; no HGS\nTYPE : CVRP\nDIMENSION : {keep+1}\nEDGE_WEIGHT_TYPE : EUC_2D\nCAPACITY : {vdata["CAPACITY"]}\nNODE_COORD_SECTION\n')
        for id in range(1,keep+2):
            a,b=xy[id]; f.write(f'{id} {a} {b}\n')
        f.write('DEMAND_SECTION\n')
        for id in range(1,keep+2): f.write(f'{id} {dem[id]}\n')
        f.write('DEPOT_SECTION\n1\n-1\nEOF\n')

    with (dest/'initial.sol').open('w') as f:
        for i,r in enumerate(initial_routes,1): f.write(f'Route #{i}: '+ ' '.join(map(str,r))+'\n')
        f.write(f'Cost {rounded_cost(initial_routes,xy)}\n')

    with (dest/'events_add_to_full.txt').open('w') as f:
        f.write(f'# Exact 20pct customer additions from {name}; original ids {keep+1}..{n}; no HGS\n')
        for cust in range(keep+1,n+1):
            ext=cust+1
            a,b=xy[ext]; f.write(f'A {a} {b} {dem[ext]}\n')

    # Outputs to facilitate fair reference cost comparison. The archived full
    # routes need not be a mathematical optimum or the published BKS.
    data={
        'source_instance':name,
        'initial_customers': keep,
        'num_adds':n-keep,
        'final_customers':n,
        'selection':'first 80 percent of original internal ids, add remaining 20 percent',
        'initial_solution':'original full routes, filtered to retained customers; same within-route order',
        'initial_routes':len(initial_routes),
        'initial_cost_rounded_euc2d':rounded_cost(initial_routes,xy),
        'reference_full_routes':len(routes),
        'reference_full_cost_rounded_euc2d':rounded_cost(routes,xy),
        'reference_full_cost_is_verified_BKS':False,
        'note':'This is a batch ADD CVRP test (snapshot reoptimization), not the online dynamic VRPTW literature protocol.'
    }
    (case/'metadata.json').write_text(json.dumps(data,indent=2)+'\n')
    print(f"{name}: 80%={keep}, ADD={n-keep}, routes={len(initial_routes)}, initial_cost={data['initial_cost_rounded_euc2d']}, reference_full_cost={data['reference_full_cost_rounded_euc2d']}")

def verify_x1001_case():
    # X-n1001 begins from an independently archived starting solution, not a
    # route-filtered version of the full solution. Regeneration here is therefore
    # verification-only: do not overwrite initial.sol or the prepared event trace.
    name = 'X-n1001-k43'
    path = ROOT / name
    info = json.loads((path / 'metadata.json').read_text())
    v, xy, demand = parse_vrp(path / 'reference' / 'full.vrp')
    routes = parse_sol(path / 'reference' / 'full.sol')
    initial, initxy, initdem = parse_vrp(path / 'case' / 'initial.vrp')
    events = [s.split() for s in (path/'case'/'events_add_to_full.txt').read_text().splitlines()
              if s.startswith('A ')]
    assert int(initial['DIMENSION']) == 801
    assert int(v['DIMENSION']) == 1001
    assert len(events) == 200
    for id in range(1,802):
        assert xy[id] == initxy[id] and demand[id] == initdem[id]
    for id, event in enumerate(events,802):
        assert event == ['A',xy[id][0],xy[id][1],demand[id]]
    assert len([c for route in routes for c in route])==1000
    cost = rounded_cost(routes,xy)
    assert cost == info['reference_full_cost_rounded_euc2d']
    assert info['source_instance']==name
    assert info['initial_customers']==800 and info['num_adds']==200
    print(f'{name}: preserved separate initial.sol, +200 exact events, full reference cost={cost} [OK]')

if __name__=='__main__':
    for name,n in CASES.items(): create_case(name,n)
    verify_x1001_case()
