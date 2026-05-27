import os
import time
import math
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from argparse import ArgumentParser

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
        
    # 좌표 파싱 블록
    if idx < len(lines) and lines[idx] == "POSITIONS":
        idx += 1
        for _ in range(num_nodes):
            parts = lines[idx].split()
            u = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            G.nodes[u]['pos'] = (x, y)
            idx += 1
            
    return G

def get_cell(pos, cell_size):
    return (int(pos[0] // cell_size), int(pos[1] // cell_size))

class GridSOS1ExactSolver:
    def __init__(self, G, radius=0.14):
        self.n = len(G.nodes())
        self.G = G
        self.radius = radius
        
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
        model = gp.Model("MWIDSP_Grid_SOS1", env=env)
        
        x = model.addVars(self.G.nodes(), vtype=GRB.BINARY, name="x")
        q = model.addVars(self.G.nodes(), vtype=GRB.CONTINUOUS, lb=0.0, name="q")
        model.update() 
        
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

        # --- 2. GEOMETRIC GRID PARTITIONING & SOS1 BRANCHING ---
        print("  -> Constructing Geometric Grid Cells...")
        t_grid = time.time()
        cell_size = self.radius / math.sqrt(2)
        grid_cells = {}
        
        for v in self.G.nodes():
            pos = self.G.nodes[v]['pos'] 
            cell_id = get_cell(pos, cell_size)
            if cell_id not in grid_cells:
                grid_cells[cell_id] = []
            grid_cells[cell_id].append(v)


        cross_boundary_cuts = 0
        for u, v in self.G.edges():
            cell_u = get_cell(self.G.nodes[u]['pos'], cell_size)
            cell_v = get_cell(self.G.nodes[v]['pos'], cell_size)
            
            if cell_u != cell_v:
                model.addConstr(x[u] + x[v] <= 1, name=f"Indep_{u}_{v}")
                cross_boundary_cuts += 1
                
        print(f"  -> Added {cross_boundary_cuts} Cross-Boundary Independence constraints.")
        # ====================================================================

        print("  -> Applying GRB.SOS_TYPE1 & Branch Priority...")
        valid_sos1_count = 0
        for cell_id, nodes_in_cell in grid_cells.items():
            if len(nodes_in_cell) >= 2:
                model.addSOS(GRB.SOS_TYPE1, [x[v] for v in nodes_in_cell], list(range(1, len(nodes_in_cell) + 1)))

                valid_sos1_count += 1
                
                priority = len(nodes_in_cell)
                for v in nodes_in_cell:
                    x[v].BranchPriority = priority
                    
        print(f"  -> Added {valid_sos1_count} SOS1 Branching constraints.")
        model.update()

        # --- 3. OPTIMIZED GUROBI PARAMETER LOADOUT ---
        model.Params.TimeLimit = time_limit
        model.Params.MIPGap = 0.0
        
        t0 = time.time()
        model.optimize()
        solve_time = time.time() - t0
        
        explored_nodes = model.NodeCount if hasattr(model, 'NodeCount') else 0
        
        if model.Status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
            return float('inf'), float('-inf'), solve_time, explored_nodes
            
        if model.SolCount > 0:
            return model.ObjVal, model.ObjBound, solve_time, explored_nodes
        else:
            return float('inf'), float('-inf'), solve_time, explored_nodes

def run_grid_sos1_pipeline(in_dir_path, instances_subset, out_dir_path, exp_name, time_limit=1500.0, radius=0.14):
    out_file_name = f'{instances_subset}_grid_sos1_{exp_name}_results.csv'
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write('instance,best_obj,best_bound,gap_percent,total_time,explored_nodes\n')
            
    processed = set()
    if os.path.exists(out_file_path):
        with open(out_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                if line.strip():
                    processed.add(line.split(',')[0].strip())

    for file_name in sorted(os.listdir(in_dir_path)):
        if not file_name.startswith(f"{instances_subset}_") or not file_name.endswith(".rgg"):
            continue
            
        if file_name in processed:
            continue
            
        file_path = os.path.join(in_dir_path, file_name)
        
        G = read_instance_with_pos(file_path)
        
        print(f"\n[{file_name}] Executing Grid-SOS1 Exact Solver...")
        
        solver = GridSOS1ExactSolver(G=G, radius=radius)
        best_obj, best_bound, solve_time, explored_nodes = solver.optimize(time_limit=time_limit)
        
        gap = float('inf')
        if best_obj != float('inf') and best_obj > 0:
            gap = max(0.0, ((best_obj - best_bound) / best_obj) * 100)
            
        with open(out_file_path, 'a') as f:
            f.write(f'{file_name},{best_obj:.4f},{best_bound:.4f},{gap:.2f},{solve_time:.4f},{explored_nodes}\n')
            
        print(f"  -> Solver Done! Obj: {best_obj:.2f} | Bound: {best_bound:.2f} | Gap: {gap:.2f}% | Time: {solve_time:.4f}s | Explored Nodes: {explored_nodes}")

if __name__ == '__main__':
    parser = ArgumentParser(description="Geometric Grid SOS1 Branching for MWIDSP")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True)
    parser.add_argument('-s', '--instances_subset', type=str, required=True)
    parser.add_argument('-o', '--out_dir_path', type=str, required=True)
    parser.add_argument('-e', '--exp_name', type=str, required=True, help="Experiment Name")
    parser.add_argument('-t', '--time_limit', type=float, default=1500.0)
    parser.add_argument('-r', '--radius', type=float, default=0.14, help="Radius for the UDG grid")
    
    parser.add_argument('--score_type', choices=['node_only', 'edge_aware_min', 'edge_aware_sum'], default='node_only', help="Dummy argument")
    
    args = parser.parse_args()
    
    run_grid_sos1_pipeline(
        in_dir_path=args.in_dir_path, 
        instances_subset=args.instances_subset, 
        out_dir_path=args.out_dir_path, 
        exp_name=args.exp_name, 
        time_limit=args.time_limit,
        radius=args.radius
    )