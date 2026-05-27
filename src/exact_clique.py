import os
import time
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from argparse import ArgumentParser

def read_instance(file_path):
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
        parts = lines[idx].split()
        u, v = int(parts[0]), int(parts[1])
        w = float(parts[2]) if len(parts) > 2 else 1.0
        G.add_edge(u, v, weight=w)
        idx += 1
    return G

class MaxCliqueExactSolver:
    def __init__(self, G):
        self.n = len(G.nodes())
        self.G = G
        
        # Exact NEW2 Weights Precomputation
        self.w_node = {u: G.nodes[u]['weight'] for u in G.nodes()}
        self.w_edge = {}
        for u, v in G.edges():
            weight = G[u][v]['weight']
            self.w_edge[u, v] = weight
            self.w_edge[v, u] = weight

    def optimize(self, time_limit):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1) 
        env.start()
        model = gp.Model("MWIDSP_MaxClique_V4", env=env)
        
        # Variables
        x = model.addVars(self.G.nodes(), vtype=GRB.BINARY, name="x")
        q = model.addVars(self.G.nodes(), vtype=GRB.CONTINUOUS, lb=0.0, name="q")
        
        model.update() 
        
        # Objective
        model.setObjective(
            gp.quicksum(self.w_node[u] * x[u] + q[u] for u in self.G.nodes()),
            GRB.MINIMIZE
        )
        
        # --- 1. Domination & Routing Cuts (NEW2 Formulation) ---
        for u in self.G.nodes():
            neighbors = list(self.G.neighbors(u))
            model.addConstr(x[u] + gp.quicksum(x[v] for v in neighbors) >= 1, name=f"Dom_{u}")
            
            N_prime = sorted(neighbors, key=lambda v: self.w_edge[v, u])
            for s_idx, s_node in enumerate(N_prime):
                w_su = self.w_edge[s_node, u]
                sum_t = gp.quicksum(
                    (w_su - self.w_edge[N_prime[t_idx], u]) * x[N_prime[t_idx]]
                    for t_idx in range(s_idx)
                )
                model.addConstr(q[u] >= w_su - sum_t - w_su * x[u], name=f"Cut_{u}_{s_node}")

        # --- 2. EXTRACT ALL EXACT MAXIMAL CLIQUES UPFRONT ---
        print("  -> Extracting exact maximal cliques...")
        t_clq = time.time()
        maximal_cliques = list(nx.find_cliques(self.G))
        print(f"  -> Replaced {self.G.number_of_edges()} edges with {len(maximal_cliques)} exact cliques in {time.time()-t_clq:.3f}s")
        
        # Count participation for Strategy B (Hub-Based Priority)
        clique_counts = {u: 0 for u in self.G.nodes()}
        for idx, clq in enumerate(maximal_cliques):
            if len(clq) >= 2:
                model.addConstr(gp.quicksum(x[i] for i in clq) <= 1, name=f"MaxClq_{idx}")
                for u in clq:
                    clique_counts[u] += 1

        model.update()


        # --- 4. OPTIMIZED GUROBI PARAMETER LOADOUT ---
        model.Params.TimeLimit = time_limit
        model.Params.MIPGap = 0.0
        
        t0 = time.time()
        model.optimize()
        solve_time = time.time() - t0
        
        if model.Status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
            return float('inf'), float('-inf'), solve_time
            
        if model.SolCount > 0:
            return model.ObjVal, model.ObjBound, solve_time
        else:
            return float('inf'), float('-inf'), solve_time

def run_maxclique_pipeline(in_dir_path, instances_subset, out_dir_path, exp_name, time_limit=1500.0):
    if exp_name in ["", "fixed", "scaling_test"]:
        out_file_name = f'{instances_subset}_maxclique_v4_results_new.csv' 
    else:
        out_file_name = f'{instances_subset}_maxclique_v4_{exp_name}_results.csv'
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write('instance,best_obj,best_bound,gap_percent,total_time\n')
            
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
                continue
                
            file_path = os.path.join(in_dir_path, file_name)
            G = read_instance(file_path)
            
            print(f"\n[{file_name}] Executing Exact Max-Clique Solver (V4)...")
            
            solver = MaxCliqueExactSolver(G=G)
            best_obj, best_bound, solve_time = solver.optimize(time_limit=time_limit)
            
            gap = float('inf')
            if best_obj != float('inf') and best_obj > 0:
                gap = max(0.0, ((best_obj - best_bound) / best_obj) * 100)
                
            with open(out_file_path, 'a') as f:
                f.write(f'{file_name},{best_obj:.4f},{best_bound:.4f},{gap:.2f},{solve_time:.4f}\n')
                
            print(f"  -> Solver Done! Obj: {best_obj:.2f} | Bound: {best_bound:.2f} | Gap: {gap:.2f}% | Time: {solve_time:.4f}s")

if __name__ == "__main__":
    parser = ArgumentParser(description="Maximal Clique Exact Solver for MWIDSP")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True)
    parser.add_argument('-s', '--instances_subset', type=str, required=True)
    parser.add_argument('-o', '--out_dir_path', type=str, required=True)
    parser.add_argument('-e', '--exp_name', type=str, required=True, help="Experiment Name (e.g., fixed_radius or constant_density)")
    parser.add_argument('-t', '--time_limit', type=float, default=1500.0)
    parser.add_argument('--score_type', choices=['node_only', 'edge_aware_min', 'edge_aware_sum'], default='node_only', help="Dummy argument for pipeline compatibility") # dummy.

    
    args = parser.parse_args()
    run_maxclique_pipeline(args.in_dir_path, args.instances_subset, args.out_dir_path, args.exp_name, args.time_limit)