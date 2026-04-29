"""
Gurobi Warm-Start Testing Framework for the MWIDSP.
Evaluates the impact of different heuristic initialization strategies 
(Shift_Full, SA_smart, GRASP, LP_Rounding, and Ensemble) on the exact solver's 
convergence speed and dual bound progression.
"""

from argparse import ArgumentParser
import os
from time import perf_counter
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

# Ensure the newly renamed heuristic files are available in the same directory
from heuristic_competitive_ensemble import read_instance_with_pos, shift_full_heuristic, lp_rounding_heuristic_pool
from metaheuristic_sa_smart import simulated_annealing_smart
from metaheuristic_grasp import grasp_baseline_heuristic

def solve_with_gurobi_warm_tester(G, instance_name, warm_method, total_time_limit=1800.0):
    start_time = perf_counter()
    
    print(f"\n[{instance_name}] Phase 1: Collecting Initial Solutions via [{warm_method}]...")
    
    solution_pool = []
    
    # 1. Execute the selected heuristic warm-start method
    if warm_method == 'shift_full':
        S_init, inc, cost_init, _ = shift_full_heuristic(G)
        solution_pool = [S_init]
    elif warm_method == 'sa_smart':
        S_init, inc, cost_init, _, _ = simulated_annealing_smart(G)
        solution_pool = [S_init]
    elif warm_method == 'grasp':
        S_init, inc, cost_init, _, _ = grasp_baseline_heuristic(G, time_limit=1500.0)
        solution_pool = [S_init]
    elif warm_method == 'lp_rounding':
        S_init, inc, cost_init, _, lp_pool = lp_rounding_heuristic_pool(G, pool_size=5)
        solution_pool = lp_pool if lp_pool else [S_init]
    elif warm_method == 'ensemble':  
        print("  - [Ensemble Mode] Executing Shift_Full and LP_Rounding concurrently...")
        # Extract one geometric elite solution
        S_shift, inc_shift, cost_shift, _ = shift_full_heuristic(G)
        solution_pool = [S_shift]
        
        # Extract up to 5 topologically diverse solutions
        S_lp, inc_lp, cost_lp, _, lp_pool = lp_rounding_heuristic_pool(G, pool_size=5)
        
        # Merge pools while removing exact duplicates
        if lp_pool:
            for p in lp_pool:
                if p != S_shift:
                    solution_pool.append(p)
        elif S_lp != S_shift:
            solution_pool.append(S_lp)
            
        cost_init = min(cost_shift, cost_lp)
        inc = inc_shift + inc_lp 

    else:
        raise ValueError("Unknown warm-start method specified.")
        
    heuristic_time = perf_counter() - start_time
    print(f"  - {warm_method} Finished: Best Cost {cost_init:.1f} / Acquired {len(solution_pool)} solution(s) (Time: {heuristic_time:.2f}s)")
    
    if inc > 0:
        print("  - Warning: The best generated initial solution is invalid. Gurobi might reject it.")
        
    # 2. Calculate remaining time for Gurobi
    gurobi_time_limit = total_time_limit - heuristic_time
    if gurobi_time_limit <= 0:
        print("  - Warning: Heuristic consumed all allocated time. Granting minimal time to the solver.")
        gurobi_time_limit = 1.0

    # 3. Initialize Gurobi Model
    print(f"\n[{instance_name}] Phase 2: Initializing Gurobi Exact Solver...")
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 1)
    env.start()
    
    m = gp.Model(f"MWIDSP_{warm_method}", env=env)
    m.setParam('TimeLimit', gurobi_time_limit)
    m.setParam('MIPFocus', 1) 
    
    x = {}
    for i in G.nodes():
        x[i] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}")
        
    y = {}
    for u in G.nodes():
        for v in G.neighbors(u):
            y[u, v] = m.addVar(vtype=GRB.BINARY, name=f"y_{u}_{v}")
            
    obj_node = gp.quicksum(G.nodes[i]['weight'] * x[i] for i in G.nodes())
    obj_edge = gp.quicksum(G[u][v]['weight'] * y[u, v] for u in G.nodes() for v in G.neighbors(u))
    m.setObjective(obj_node + obj_edge, GRB.MINIMIZE)
    
    for u in G.nodes():
        m.addConstr(x[u] + gp.quicksum(y[u, v] for v in G.neighbors(u)) >= 1, name=f"cov_{u}")
        for v in G.neighbors(u):
            m.addConstr(y[u, v] <= x[v], name=f"val_{u}_{v}")
            
    covered_edges = set()
    for u, v in G.edges():
        if (u, v) not in covered_edges and (v, u) not in covered_edges:
            m.addConstr(x[u] + x[v] <= 1, name=f"ind_{u}_{v}")
            covered_edges.add((u, v))
            covered_edges.add((v, u))
        
    # 4. Inject Multiple MIP Starts
    m.NumStart = len(solution_pool)
    
    for idx, S_cand in enumerate(solution_pool):
        m.setParam('StartNumber', idx)
        
        for v in G.nodes():
            x[v].Start = 1.0 if v in S_cand else 0.0
            
        for u in G.nodes():
            if u not in S_cand:
                valid_neighbors = [v for v in G.neighbors(u) if v in S_cand]
                if valid_neighbors:
                    best_v = min(valid_neighbors, key=lambda v: G[u][v]['weight'])
                    for v in G.neighbors(u):
                        y[u, v].Start = 1.0 if v == best_v else 0.0
                else:
                    for v in G.neighbors(u):
                        y[u, v].Start = 0.0
            else:
                for v in G.neighbors(u):
                    y[u, v].Start = 0.0
                    
    print(f"  - [MIP Start] Injection Complete: {len(solution_pool)} diverse solution(s) injected.")
    
    print(f"  - Initiating Gurobi search (Solver Time Limit: {gurobi_time_limit:.2f}s)...\n")
    m.optimize()
    
    # 5. Extract results
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
        if m.status == GRB.TIME_LIMIT:
            best_bound = m.ObjBound
            print("\nTimeout: Failed to find any feasible solution.")
        else:
            print("\nOptimization Failed.")
            
    return best_obj, best_bound, gap, total_time

def run_warm_test_pipeline(in_dir_path, instances_subset, out_dir_path, warm_method, time_limit):
    out_file_name = f"{instances_subset}_gurobi_warm_{warm_method}_results.csv"
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    # Resume capability: write header if file is missing or empty
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write("instance,best_obj,lower_bound,gap_percent,total_time\n")
            
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
            
            obj, bound, gap, total_time = solve_with_gurobi_warm_tester(G, file_name, warm_method, time_limit)
            
            # Immediately append result to prevent data loss
            with open(out_file_path, 'a') as f:
                f.write(f"{file_name},{obj},{bound},{gap},{total_time:.4f}\n")

if __name__ == '__main__':
    parser = ArgumentParser(description="Gurobi Warm-Start Testing Framework")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Path to input instances")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Instance prefix to process")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Output directory path")
    parser.add_argument('-w', '--warm_method', type=str, choices=['shift_full', 'sa_smart', 'grasp', 'lp_rounding', 'ensemble'], required=True, help="Select heuristic for warm-start")
    parser.add_argument('-t', '--time_limit', type=float, default=1800.0, help="Total Time Limit (Heuristic + Gurobi)")
    args = parser.parse_args()
    
    run_warm_test_pipeline(args.in_dir_path, args.instances_subset, args.out_dir_path, args.warm_method, args.time_limit)