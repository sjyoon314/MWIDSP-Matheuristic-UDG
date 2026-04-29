import os
import time
from argparse import ArgumentParser
import gurobipy as gp
from gurobipy import GRB

from utils import read_instance, calc_initial_solution_cost

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
    A = list(w_edge.keys())
    
    model = gp.Model("MWIDSP_NEW_1")
    model.Params.OutputFlag = 0  # Disable Gurobi console output
    model.Params.TimeLimit = time_limit
    
    x = model.addVars(G.nodes(), vtype=GRB.BINARY, name="x")
    y = model.addVars(A, vtype=GRB.BINARY, name="y")
    
    model.setObjective(
        gp.quicksum(w_node[u] * x[u] for u in G.nodes()) +
        gp.quicksum(w_edge[u, v] * y[u, v] for u, v in A),
        GRB.MINIMIZE
    )
    
    # Constraints
    # 1. Edge-packing (Independence)
    for u, v in G.edges():
        model.addConstr(x[u] + x[v] <= 1, name=f"Indep_{u}_{v}")
        
    # 2. Set Partitioning (Domination)
    for u in G.nodes():
        incoming_neighbors = list(G.neighbors(u))
        model.addConstr(
            x[u] + gp.quicksum(y[v, u] for v in incoming_neighbors) == 1,
            name=f"Dom_{u}"
        )
        
    # 3. Logical Valid Inequalities
    for u, v in A:
        model.addConstr(y[u, v] <= x[u], name=f"Logical_{u}_{v}")
        
    model.optimize()
    elapsed_time = time.perf_counter() - start_time
    
    if model.SolCount > 0:
        S = set(u for u in G.nodes() if x[u].x > 0.5)
        num_incorrect, cost, _, _ = calc_initial_solution_cost(S, G)
    else:
        S = set()
        num_incorrect = len(G.nodes())
        cost = float('inf')
        
    return S, num_incorrect, cost, elapsed_time

def solve_all_new1(in_dir_path, instances_subset, out_dir_path):
    out_file_name = f"{instances_subset}_new1_results.csv"
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    with open(out_file_path, 'w') as f:
        f.write("instance,num_incorrect,cost,time\n")
        
        for file_name in sorted(os.listdir(in_dir_path)):
            if not file_name.startswith(instances_subset):
                continue
                
            file_path = os.path.join(in_dir_path, file_name)
            G = read_instance(file_path)
            
            # Dynamic time limit: 3 * |V| seconds
            current_time_limit = 3 * len(G.nodes())
            _, num_incorrect_nodes, cost, time_elapsed = solve_new1(G, current_time_limit)
                
            f.write(f"{file_name},{num_incorrect_nodes},{cost},{time_elapsed:.4f}\n")
            print(f"[{file_name}] NEW-1 Finished | Cost: {cost} | Time: {time_elapsed:.4f}s")

def main():
    parser = ArgumentParser(description="Exact solver for MWIDSP using the NEW-1 MILP formulation.")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Path to the input instances directory")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Prefix of the instances to test (e.g., 100_r014)")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Path to the output directory for CSV results")
    args = parser.parse_args()

    solve_all_new1(
        in_dir_path=args.in_dir_path,
        instances_subset=args.instances_subset,
        out_dir_path=args.out_dir_path
    )

if __name__ == '__main__':
    main()