import os
import time
from argparse import ArgumentParser
import gurobipy as gp
from gurobipy import GRB

from utils import read_instance, calc_initial_solution_cost_fast

def extract_weights(G):
    w_node = {u: G.nodes[u]['weight'] for u in G.nodes()}
    w_edge = {}
    for u, v in G.edges():
        weight = G[u][v]['weight']
        w_edge[u, v] = weight
        w_edge[v, u] = weight
    return w_node, w_edge

def solve_new1(G, time_limit):
    start_time = time.perf_counter()
    w_node, w_edge = extract_weights(G)
    
    model = gp.Model("MWIDSP_NEW_1")
    model.Params.OutputFlag = 1  # log
    model.Params.TimeLimit = time_limit
    
    x = model.addVars(G.nodes(), vtype=GRB.BINARY, name="x")
    q = model.addVars(G.nodes(), vtype=GRB.CONTINUOUS, lb=0.0, name="q") 
    
    model.setObjective(
        gp.quicksum(w_node[u] * x[u] + q[u] for u in G.nodes()),
        GRB.MINIMIZE
    )
    
    # 1. Edge-packing (Independence)
    for u, v in G.edges():
        model.addConstr(x[u] + x[v] <= 1, name=f"Indep_{u}_{v}")
        
    # 2. Set Partitioning (Domination)
    for u in G.nodes():
        neighbors = list(G.neighbors(u))
        model.addConstr(
            x[u] + gp.quicksum(x[v] for v in neighbors) >= 1,
            name=f"Dom_{u}"
        )
        
        # 3. Objective Cuts (NEW-1 Specific)
        for v in neighbors:
            N_v_u = [w for w in neighbors if w_edge[w, u] < w_edge[v, u]]
            model.addConstr(
                q[u] >= w_edge[v, u] * (1 - x[u] - gp.quicksum(x[w] for w in N_v_u)),
                name=f"Cut_{u}_{v}"
            )
            
    model.optimize()
    elapsed_time = time.perf_counter() - start_time
    
    if model.Status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
        best_obj, best_bound, gap_percent = float('inf'), float('-inf'), 100.0
    elif model.SolCount > 0:
        best_obj = model.ObjVal
        best_bound = model.ObjBound
        gap_percent = model.MIPGap * 100.0
    else:
        best_obj = float('inf')
        best_bound = model.ObjBound if hasattr(model, 'ObjBound') else float('-inf')
        gap_percent = 100.0
        
    return best_obj, best_bound, gap_percent, elapsed_time

def solve_all_new1(in_dir_path, instances_subset, out_dir_path):
    out_file_name = f"{instances_subset}_new1_results.csv"
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    # 1. Check file existence and write header for appending
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write("instance,best_obj,lower_bound,gap_percent,total_time\n")

    # 2. Extract already processed instances to avoid duplicate work
    processed = set()
    if os.path.exists(out_file_path):
        with open(out_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                if line.strip():
                    processed.add(line.split(',')[0].strip())

    # 3. Enforce underscore on prefix to prevent partial matching errors
    safe_prefix = instances_subset if instances_subset.endswith('_') else f"{instances_subset}_"
    
    for file_name in sorted(os.listdir(in_dir_path)):
        if not file_name.startswith(safe_prefix):    
            continue
            
        if file_name in processed:
            print(f"[Info] {file_name} already processed. Skipping.")
            continue
            
        file_path = os.path.join(in_dir_path, file_name)
        G = read_instance(file_path)
        
        current_time_limit = 3 * len(G.nodes())
        print(f"\n[Processing] {file_name} - Executing NEW1 Baseline...")
        best_obj, best_bound, gap, time_elapsed = solve_new1(G, current_time_limit)
            
        # 4. Append results immediately
        with open(out_file_path, 'a') as f:
            f.write(f"{file_name},{best_obj:.4f},{best_bound:.4f},{gap:.2f},{time_elapsed:.4f}\n")
        print(f"  -> [Success] NEW-1 Finished | Obj: {best_obj:.1f} | Bound: {best_bound:.1f} | Gap: {gap:.2f}% | Time: {time_elapsed:.4f}s")

def main():
    parser = ArgumentParser(description="Exact solver for MWIDSP using the NEW-1 MILP formulation.")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Path to the input instances directory")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Prefix of the instances to test")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    solve_all_new1(
        in_dir_path=args.in_dir_path,
        instances_subset=args.instances_subset,
        out_dir_path=args.out_dir_path
    )

if __name__ == '__main__':
    main()