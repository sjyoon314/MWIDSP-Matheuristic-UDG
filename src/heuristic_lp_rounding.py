"""
LP Relaxation and Randomized Rounding Matheuristic (LP_Rounding).
Specifically tailored for Node-oriented environments (VG), reducing the MWIDSP 
into a purely topological Packing-Constrained Set Cover problem by omitting routing variables.
"""

import os
import random
random.seed(42)
from time import perf_counter
from argparse import ArgumentParser
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

from utils import calc_initial_solution_cost

def read_instance(file_path):
    """Reads the instance file (spatial coordinates are ignored in this purely topological solver)."""
    G = nx.Graph()
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    num_nodes, num_edges = map(int, lines[0].split())
    G.add_nodes_from(range(num_nodes))
    
    idx = 1
    for u in range(num_nodes):
        G.nodes[u]['weight'] = float(lines[idx])
        idx += 1
        
    for _ in range(num_edges):
        u, v, w = map(float, lines[idx].split())
        u, v = int(u), int(v)
        G.add_edge(u, v, weight=w)
        idx += 1
        
    return G

def lp_rounding_heuristic(G, num_rounding_trials=100):
    """Executes the Continuous LP Relaxation followed by randomized rounding and deterministic repair."""
    start_time = perf_counter()
    
    # 1. Initialize Gurobi model (disable console output)
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model("MWIDSP_Relaxation", env=env)
    
    # 2. Variables: Continuous relaxation (0.0 <= x <= 1.0) instead of Binary
    x = m.addVars(G.nodes(), vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x")
    
    # 3. Objective: Minimize total node selection costs (ignoring edge routing costs)
    m.setObjective(gp.quicksum(G.nodes[v]['weight'] * x[v] for v in G.nodes()), GRB.MINIMIZE)
    
    # 4. Constraints
    # (1) Domination (Set Cover): v and its neighbors must sum to at least 1.0
    for v in G.nodes():
        m.addConstr(x[v] + gp.quicksum(x[u] for u in G.neighbors(v)) >= 1.0)
        
    # (2) Independence (Edge-packing): Adjacent nodes cannot sum to more than 1.0
    for u, v in G.edges():
        m.addConstr(x[u] + x[v] <= 1.0)
        
    # 5. Optimize the Continuous LP
    m.optimize()
    
    if m.status != GRB.OPTIMAL:
        print("LP Relaxation failed to find an optimal fractional solution.")
        return set(), 0, float('inf'), perf_counter() - start_time
        
    lp_probs = {v: x[v].X for v in G.nodes()}
    
    best_S = None
    best_cost = float('inf')
    best_incorrect = float('inf')
    
    # 6. Randomized Rounding & Correction Pipeline
    for _ in range(num_rounding_trials):
        # [Step A] Probabilistic Sampling: Treat fractional values as selection probabilities
        S = set()
        for v in G.nodes():
            if random.random() <= lp_probs[v]:
                S.add(v)
                
        # [Step B] Independence Correction: Deterministically evict higher-cost conflicting nodes
        S_list = list(S)
        for u in S_list:
            if u in S:
                for v in list(G.neighbors(u)):
                    if v in S:
                        if G.nodes[u]['weight'] >= G.nodes[v]['weight']:
                            S.remove(u)
                            break
                        else:
                            S.remove(v)
                            
        # [Step C] Greedy Domination Repair: Patch remaining blind spots
        covered = set()
        for v in S:
            covered.add(v)
            covered.update(G.neighbors(v))
            
        uncovered = set(G.nodes()) - covered
        
        while uncovered:
            best_cand = None
            best_ratio = float('inf')
            
            # Ensure candidate preserves strict independence
            valid_cands = [v for v in G.nodes() if v not in S and not any(u in S for u in G.neighbors(v))]
            
            if not valid_cands:
                break 
                
            # Select the most cost-effective candidate
            for v in valid_cands:
                covers_new = len((set(G.neighbors(v)) | {v}) & uncovered)
                if covers_new > 0:
                    ratio = G.nodes[v]['weight'] / covers_new
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_cand = v
                        
            if best_cand is not None:
                S.add(best_cand)
                uncovered -= (set(G.neighbors(best_cand)) | {best_cand})
            else:
                break
                
        # [Step D] Cost Evaluation
        num_incorrect, cost, _, _ = calc_initial_solution_cost(S, G)
        
        if num_incorrect == 0 and cost < best_cost:
            best_cost = cost
            best_S = S.copy()
            best_incorrect = num_incorrect
            
    # Fallback if no valid set was constructed perfectly
    if best_S is None:
        best_S = S
        best_cost = cost
        best_incorrect = num_incorrect

    time_elapsed = perf_counter() - start_time
    return best_S, best_incorrect, best_cost, time_elapsed

def solve_all_lp_rounding(in_dir_path, instances_subset, out_dir_path):
    out_file_name = f"{instances_subset}_lp_rounding_results.csv"
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    # Resume capability: write header if file is missing or empty
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write("instance,num_incorrect,cost,time\n")
            
    # Read already processed instances to skip them
    processed = set()
    if os.path.exists(out_file_path):
        with open(out_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]: # Skip header
                if line.strip():
                    processed.add(line.split(',')[0].strip())

    for file_name in sorted(os.listdir(in_dir_path)):
        if file_name.startswith(f"{instances_subset}_"):
            if file_name in processed:
                print(f"[{file_name}] Already processed. Skipping.")
                continue
                
            file_path = os.path.join(in_dir_path, file_name)
            G = read_instance(file_path)
            
            _, num_incorrect_nodes, cost, time_elapsed = lp_rounding_heuristic(G)
            
            with open(out_file_path, 'a') as f:
                f.write(f"{file_name},{num_incorrect_nodes},{cost},{time_elapsed:.4f}\n")
                
            print(f"[{file_name}] LP Rounding Finished | Cost: {cost} | Time: {time_elapsed:.4f}s")

if __name__ == '__main__':
    parser = ArgumentParser(description="LP Relaxation and Randomized Rounding Matheuristic")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Path to input instances")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Instance prefix to process")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Output directory path")
    args = parser.parse_args()
    
    solve_all_lp_rounding(args.in_dir_path, args.instances_subset, args.out_dir_path)