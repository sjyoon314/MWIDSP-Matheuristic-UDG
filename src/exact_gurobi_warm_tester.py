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
from heuristic_grid_ensemble import read_instance_with_pos, shift_full_heuristic, lp_rounding_heuristic_pool
from metaheuristic_sa_smart import simulated_annealing_smart
from metaheuristic_grasp import grasp_baseline_heuristic

def extract_weights(G):
    w_node = {u: G.nodes[u]['weight'] for u in G.nodes()}
    w_edge = {}
    for u, v in G.edges():
        weight = G[u][v]['weight']
        w_edge[u, v] = weight
        w_edge[v, u] = weight
    return w_node, w_edge

def solve_with_gurobi_warm_tester(G, instance_name, warm_method, total_time_limit=1500.0):
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

    
    w_node, w_edge = extract_weights(G)
    x = m.addVars(G.nodes(), vtype=GRB.BINARY, name="x")
    q = m.addVars(G.nodes(), vtype=GRB.CONTINUOUS, lb=0.0, name="q")
    
    m.setObjective(
        gp.quicksum(w_node[u] * x[u] + q[u] for u in G.nodes()),
        GRB.MINIMIZE
    )
    
    # 2. NEW2 제약식 (Edge-packing, Set Partitioning, Objective Cuts)
    for u, v in G.edges():
        m.addConstr(x[u] + x[v] <= 1, name=f"Indep_{u}_{v}")
        
    for u in G.nodes():
        neighbors = list(G.neighbors(u))
        m.addConstr(x[u] + gp.quicksum(x[v] for v in neighbors) >= 1, name=f"Dom_{u}")
        
        N_prime = sorted(neighbors, key=lambda v: w_edge[v, u])
        for s_idx, s_node in enumerate(N_prime):
            w_su = w_edge[s_node, u]
            sum_t = gp.quicksum(
                (w_su - w_edge[N_prime[t_idx], u]) * x[N_prime[t_idx]]
                for t_idx in range(s_idx)
            )
            m.addConstr(q[u] >= w_su - sum_t - w_su * x[u], name=f"Cut_{u}_{s_node}")

    m.update() # 변수와 제약식 동기화 필수

    # 3. 웜스타트 (Multiple MIP Starts) 주입 로직 변경
    m.NumStart = len(solution_pool)
    for idx, S_cand in enumerate(solution_pool):
        m.setParam('StartNumber', idx)
        
        for v in G.nodes():
            x[v].Start = 1.0 if v in S_cand else 0.0
            
        # NEW2에서는 y 변수가 없으므로, q 변수의 초기값을 계산해서 넣어줍니다.
        for u in G.nodes():
            if u in S_cand:
                q[u].Start = 0.0
            else:
                valid_neighbors = [v for v in G.neighbors(u) if v in S_cand]
                if valid_neighbors:
                    best_v = min(valid_neighbors, key=lambda v: w_edge[u, v])
                    q[u].Start = w_edge[u, best_v]
                else:
                    q[u].Start = float('inf') # 웜스타트가 Infeasible한 경우 대비 (이론상 발생 안함)

    print(f"  - [MIP Start] Injection Complete: {len(solution_pool)} diverse solution(s) injected into NEW2 base.")
    m.setParam('StartNumber', 0)
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
        if file_name.startswith(f"{instances_subset}_"):
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
    parser.add_argument('-t', '--time_limit', type=float, default=1500.0, help="Total Time Limit (Heuristic + Gurobi)")
    args = parser.parse_args()
    
    run_warm_test_pipeline(args.in_dir_path, args.instances_subset, args.out_dir_path, args.warm_method, args.time_limit)