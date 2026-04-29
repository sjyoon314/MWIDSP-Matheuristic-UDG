"""
Baseline Exact Solver for the MWIDSP.
Executes the standard MILP formulation using Gurobi without any heuristic warm-starts.
Used to establish exact dual bounds and evaluate the baseline performance of the solver.
"""

from argparse import ArgumentParser
import os
from time import perf_counter
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

def read_instance_with_pos(file_path):
    """Reads the instance file including physical 2D coordinates."""
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
        
    if idx < len(lines) and lines[idx] == "POSITIONS":
        idx += 1
        for _ in range(num_nodes):
            parts = lines[idx].split()
            u = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            G.nodes[u]['pos'] = (x, y)
            idx += 1
            
    return G

def solve_with_new2_baseline(G, instance_name, time_limit=1800.0):
    """Executes the exact MILP formulation using Gurobi."""
    start_time = perf_counter()
    
    print(f"\n[{instance_name}] Starting Gurobi Exact Solver (Baseline)...")
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 1)  # Keep standard Gurobi logs visible
    env.start()
    
    m = gp.Model("MWIDSP_Baseline", env=env)
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPFocus', 1)  # Focus on finding feasible incumbent solutions
    
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
    
    # Note: No initial heuristic solution (Warm Start) is injected in this baseline.
    
    print(f"  - Initiating search (Time Limit: {time_limit}s)...\n")
    m.optimize()
    
    total_time = perf_counter() - start_time
    best_obj = float('inf')
    best_bound = float('inf')
    gap = float('inf')
    
    # Extract results
    if m.SolCount > 0:
        best_obj = m.ObjVal
        best_bound = m.ObjBound
        gap = m.MIPGap * 100 
        print(f"\nSearch Terminated: Best Objective {best_obj:.1f} (Gap: {gap:.2f}%)")
    else:
        if m.status == GRB.TIME_LIMIT:
            best_bound = m.ObjBound
            print("\nTimeout: Failed to find any feasible solution within the time limit.")
        else:
            print("\nOptimization Failed (Infeasible or Error).")
            
    return best_obj, best_bound, gap, total_time

def run_new2_baseline_pipeline(in_dir_path, instances_subset, out_dir_path, time_limit):
    # Note: Filename kept as '_gurobi_new2_baseline_results.csv' to ensure compatibility with comparison scripts.
    out_file_name = f"{instances_subset}_gurobi_new2_baseline_results.csv"
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
            
            obj, bound, gap, total_time = solve_with_new2_baseline(G, file_name, time_limit)
            
            # Immediately append result to prevent data loss
            with open(out_file_path, 'a') as f:
                f.write(f"{file_name},{obj},{bound},{gap},{total_time:.4f}\n")

if __name__ == '__main__':
    parser = ArgumentParser(description="Gurobi Exact Solver Baseline (No Warm-start)")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Path to input instances")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Instance prefix to process")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Output directory path")
    parser.add_argument('-t', '--time_limit', type=float, default=1800.0, help="Gurobi Time Limit (seconds)")
    args = parser.parse_args()
    
    run_new2_baseline_pipeline(args.in_dir_path, args.instances_subset, args.out_dir_path, args.time_limit)