"""
Hybrid Exact Solver for the MWIDSP.
Integrates the Competitive Matheuristic Ensemble to generate high-quality 
initial solutions (primal bounds) and injects them into Gurobi as Multiple MIP Starts 
to accelerate the branch-and-bound exploration.
"""

from argparse import ArgumentParser
import os
from time import perf_counter
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

from heuristic_competitive_ensemble import read_instance_with_pos, shift_full_heuristic, lp_rounding_heuristic

def solve_with_gurobi_hybrid(G, instance_name, time_limit=3600.0):
    start_time = perf_counter()
    
    print(f"\n[{instance_name}] Phase 1 & 2: Collecting Multiple MIP Warm Starts...")
    
    # 1. Collect Initial Solutions (Warm Starts)
    S_shift, inc_shift, cost_shift, time_shift = shift_full_heuristic(G)
    print(f"  - Shift_Full Finished: Cost {cost_shift:.1f} (Time: {time_shift:.2f}s)")
    
    S_lp, inc_lp, cost_lp, time_lp = lp_rounding_heuristic(G)
    print(f"  - LP_Rounding Finished: Cost {cost_lp:.1f} (Time: {time_lp:.2f}s)")
    
    starts_pool = []
    # Only inject strictly feasible solutions (0 incorrect nodes)
    if inc_shift == 0:
        starts_pool.append(('Shift_Full', S_shift, cost_shift))
    if inc_lp == 0:
        starts_pool.append(('LP_Rounding', S_lp, cost_lp))
        
    # 2. Initialize Gurobi Exact Solver Model
    print(f"\n[{instance_name}] Phase 3: Initializing Gurobi Exact Solver...")
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 1)
    env.start()
    
    m = gp.Model("MWIDSP_Hybrid", env=env)
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPFocus', 1)  # Focus on finding feasible incumbent solutions quickly
    
    # [Variables] Node selection (x) and Edge assignment (y)
    x = {}
    for i in G.nodes():
        x[i] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}")
        
    y = {}
    for u in G.nodes():
        for v in G.neighbors(u):
            y[u, v] = m.addVar(vtype=GRB.BINARY, name=f"y_{u}_{v}")
            
    # [Objective] Minimize sum of node weights and active routing edge weights
    obj_node = gp.quicksum(G.nodes[i]['weight'] * x[i] for i in G.nodes())
    obj_edge = gp.quicksum(G[u][v]['weight'] * y[u, v] for u in G.nodes() for v in G.neighbors(u))
    m.setObjective(obj_node + obj_edge, GRB.MINIMIZE)
    
    # [Constraints]
    for u in G.nodes():
        # (1) Coverage: Node must be selected or covered by at least one selected neighbor
        m.addConstr(x[u] + gp.quicksum(y[u, v] for v in G.neighbors(u)) >= 1, name=f"cov_{u}")
        
        # (2) Validity: Routing is only valid if the destination node is selected
        for v in G.neighbors(u):
            m.addConstr(y[u, v] <= x[v], name=f"val_{u}_{v}")
            
    # (3) Independence (Edge-packing): Adjacent nodes cannot be selected simultaneously
    covered_edges = set()
    for u, v in G.edges():
        if (u, v) not in covered_edges and (v, u) not in covered_edges:
            m.addConstr(x[u] + x[v] <= 1, name=f"ind_{u}_{v}")
            covered_edges.add((u, v))
            covered_edges.add((v, u))
        
    # 3. Core Logic: Inject Multiple MIP Starts from the heuristic pool
    if starts_pool:
        m.NumStart = len(starts_pool)
        
        for i, (name, S_init, cost) in enumerate(starts_pool):
            m.setParam('StartNumber', i)
            
            for v in G.nodes():
                x[v].Start = 1.0 if v in S_init else 0.0
                
            for u in G.nodes():
                if u not in S_init:
                    # If 'u' is not a dominator, connect it to the cheapest adjacent dominator in S_init
                    valid_neighbors = [v for v in G.neighbors(u) if v in S_init]
                    
                    if valid_neighbors:
                        best_v = min(valid_neighbors, key=lambda v: G[u][v]['weight'])
                        for v in G.neighbors(u):
                            y[u, v].Start = 1.0 if v == best_v else 0.0
                    else:
                        for v in G.neighbors(u):
                            y[u, v].Start = 0.0
                else:
                    # If 'u' is a dominator, it does not need to connect to anyone else
                    for v in G.neighbors(u):
                        y[u, v].Start = 0.0
                        
            print(f"  - [MIP Start {i}] Injected: {name} (Expected Cost: {cost:.1f})")
    
    # 4. Execute Gurobi Optimization
    print(f"  - Initiating Gurobi search (Time Limit: {time_limit}s)...\n")
    m.optimize()
    
    # 5. Result Extraction
    total_time = perf_counter() - start_time
    best_obj = float('inf')
    best_bound = float('inf')
    gap = float('inf')
    
    if m.SolCount > 0:
        best_obj = m.ObjVal
        best_bound = m.ObjBound
        gap = m.MIPGap * 100 
        print(f"\nFinal Optimization Complete: Best Objective {best_obj:.1f} (Gap: {gap:.2f}%)")
    else:
        print("\nFailed to find any feasible solution.")
        
    return best_obj, best_bound, gap, total_time

def run_gurobi_pipeline(in_dir_path, instances_subset, out_dir_path, time_limit):
    out_file_name = f"{instances_subset}_gurobi_hybrid_final_results.csv"
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    # Resume capability: write header if file is missing or empty
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write("instance,best_obj,lower_bound,gap_percent,total_time\n")
            
    # Read already processed instances to skip them
    processed = set()
    if os.path.exists(out_file_path):
        with open(out_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                if line.strip():
                    processed.add(line.split(',')[0].strip())
                
    for file_name in sorted(os.listdir(in_dir_path)):
        if file_name.startswith(instances_subset):
            if file_name in processed:
                print(f"[{file_name}] Already processed. Skipping.")
                continue
                
            file_path = os.path.join(in_dir_path, file_name)
            G = read_instance_with_pos(file_path)
            
            obj, bound, gap, total_time = solve_with_gurobi_hybrid(G, file_name, time_limit)
            
            # Immediately append result to prevent data loss
            with open(out_file_path, 'a') as f:
                f.write(f"{file_name},{obj},{bound},{gap},{total_time:.4f}\n")

if __name__ == '__main__':
    parser = ArgumentParser(description="Hybrid Exact Solver with Multiple MIP Starts")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Path to input instances")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Instance prefix to process")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Output directory path")
    parser.add_argument('-t', '--time_limit', type=float, default=1800.0, help="Gurobi Time Limit (seconds)")
    args = parser.parse_args()
    
    run_gurobi_pipeline(args.in_dir_path, args.instances_subset, args.out_dir_path, args.time_limit)