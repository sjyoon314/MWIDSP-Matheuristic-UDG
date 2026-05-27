import os
import time
import math
from argparse import ArgumentParser
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

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

def extract_weights(G):
    w_node = {u: G.nodes[u]['weight'] for u in G.nodes()}
    w_edge = {}
    for u, v in G.edges():
        weight = G[u][v]['weight']
        w_edge[u, v] = weight
        w_edge[v, u] = weight
    return w_node, w_edge

def solve_new2_grid_cuts(G, time_limit, radius=0.14):
    start_time = time.perf_counter()
    w_node, w_edge = extract_weights(G)
    
    model = gp.Model("MWIDSP_NEW_2_GridCuts")
    model.Params.OutputFlag = 1  
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
        
        # 3. Objective Cuts (Routing Cost)
        N_prime = sorted(neighbors, key=lambda v: w_edge[v, u])
        for s_idx, s_node in enumerate(N_prime):
            w_su = w_edge[s_node, u]
            sum_t = gp.quicksum(
                (w_su - w_edge[N_prime[t_idx], u]) * x[N_prime[t_idx]]
                for t_idx in range(s_idx)
            )
            model.addConstr(
                q[u] >= w_su - sum_t - w_su * x[u],
                name=f"Cut_{u}_{s_node}"
            )

    cell_size = radius / math.sqrt(2)
    grid_cells = {}
    
    for u in G.nodes():
        if 'pos' in G.nodes[u]:
            x_pos, y_pos = G.nodes[u]['pos']
            grid_x, grid_y = int(x_pos // cell_size), int(y_pos // cell_size)
            if (grid_x, grid_y) not in grid_cells:
                grid_cells[(grid_x, grid_y)] = []
            grid_cells[(grid_x, grid_y)].append(u)
            
    grid_cut_count = 0
    for cell_nodes in grid_cells.values():
        if len(cell_nodes) >= 2:
            model.addConstr(gp.quicksum(x[i] for i in cell_nodes) <= 1, name=f"GridCut_{grid_cut_count}")
            grid_cut_count += 1
            
    print(f"  -> [Grid Cuts] Injected {grid_cut_count} lightweight clique cuts from spatial grids.")
    # --------------------------------------------------------
            
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

def solve_all_new2(in_dir_path, instances_subset, out_dir_path):
    out_file_name = f"{instances_subset}_grid_cut_results.csv"
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write("instance,best_obj,lower_bound,gap_percent,total_time\n")     

    # for continue
    processed = set()
    if os.path.exists(out_file_path):
        with open(out_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                if line.strip():
                    processed.add(line.split(',')[0].strip())

    for file_name in sorted(os.listdir(in_dir_path)):
        if not file_name.startswith(f"{instances_subset}_"):
            continue
            
        if file_name in processed:
            print(f"[{file_name}] Already processed. Skipping.")
            continue
            
        file_path = os.path.join(in_dir_path, file_name)
        
        G = read_instance_with_pos(file_path)
        
        radius = 0.14
        if 'r0c' in file_name:
            try:
                r_str = file_name.split('r0c')[1].split('_')[0]
                radius = float(f"0.{r_str}")
            except Exception:
                radius = 0.14
        
        current_time_limit = 3 * len(G.nodes())
        print(f"\n[{file_name}] Executing NEW2 + Grid Cuts...")
        best_obj, best_bound, gap, time_elapsed = solve_new2_grid_cuts(G, current_time_limit, radius)
            
        with open(out_file_path, 'a') as f:
            f.write(f"{file_name},{best_obj:.4f},{best_bound:.4f},{gap:.2f},{time_elapsed:.4f}\n")
        print(f" -> NEW-2+GridCuts Finished | Obj: {best_obj:.1f} | Bound: {best_bound:.1f} | Gap: {gap:.2f}% | Time: {time_elapsed:.4f}s")

def main():
    parser = ArgumentParser(description="Exact solver for MWIDSP using NEW-2 with Grid Cuts.")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True)
    parser.add_argument('-s', '--instances_subset', type=str, required=True)
    parser.add_argument('-o', '--out_dir_path', type=str, required=True)
    args = parser.parse_args()

    solve_all_new2(args.in_dir_path, args.instances_subset, args.out_dir_path)

if __name__ == '__main__':
    main()