import os
from time import perf_counter
from argparse import ArgumentParser

from utils import read_instance, calc_initial_solution_cost

def greedy_new_baseline(G):
    start_time = perf_counter()
    S = set()
    G_copy = G.copy()

    # Optimization: Precompute and cache the sum of incident edge weights for each node
    edge_weights_cache = {u: 0 for u in G_copy.nodes}
    for u in G_copy.nodes:
        for v in G_copy[u]:
            edge_weights_cache[u] += G[u][v]['weight']

    while len(G_copy.nodes) > 0:
        v_star = None
        max_val = -1.0
        
        for v in G_copy.nodes:
            uv_edge_weights = 0
            other_edge_weights = 0
            node_weights_sum = 0
            
            for u in G_copy[v]:
                node_weights_sum += G_copy.nodes(data=True)[u]['weight']
                uv_edge_weights += G_copy[u][v]['weight']
                
                # O(1) deduction using the cached sum to simulate alternative routing
                other_edge_weights += edge_weights_cache[u] - G_copy[u][v]['weight']
                
            # Cost (Denominator): Node weight + routing costs to neighbors
            cost = G_copy.nodes(data=True)[v]['weight'] + uv_edge_weights
            
            # Benefit (Numerator): Savings from not activating neighbors + alternative routing costs
            benefit = node_weights_sum + other_edge_weights
            
            val = benefit / cost if cost > 0 else 0
                
            if val > max_val:
                max_val = val
                v_star = v
                
        # Add the most cost-effective node to the solution set
        S.add(v_star)
        
        # Update graph topology and cache (remove selected node and its neighbors)
        neighbors_of_v_star = list(G_copy[v_star])
        
        # Deduct weights from the cache of the neighbors' neighbors
        for v in neighbors_of_v_star:
            for u in G_copy[v]:
                edge_weights_cache[u] -= G_copy[v][u]['weight']
            G_copy.remove_node(v) 
            
        # Deduct weights from the cache of v*'s remaining neighbors
        if v_star in G_copy: 
            for u in G_copy[v_star]:
                edge_weights_cache[u] -= G_copy[v_star][u]['weight']
            G_copy.remove_node(v_star)

    # Final validation and cost calculation
    num_incorrect, cost, _, _ = calc_initial_solution_cost(S, G)
    time_elapsed = perf_counter() - start_time
    
    return S, num_incorrect, cost, time_elapsed

def solve_all_baseline(in_dir_path, instances_subset, out_dir_path):
    out_file_name = f"{instances_subset}_baseline_results.csv"
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    with open(out_file_path, 'w') as f:
        f.write("instance,num_incorrect,cost,time\n")
        
        for file_name in sorted(os.listdir(in_dir_path)):
            if not file_name.startswith(instances_subset):
                continue
                
            file_path = os.path.join(in_dir_path, file_name)
            G = read_instance(file_path)
            
            _, num_incorrect_nodes, cost, time_elapsed = greedy_new_baseline(G)
            
            f.write(f"{file_name},{num_incorrect_nodes},{cost},{time_elapsed:.4f}\n")
            print(f"[{file_name}] Baseline Finished | Cost: {cost} | Time: {time_elapsed:.4f}s")

def main():
    parser = ArgumentParser(description="Constructive heuristic baseline (GREEDY-NEW) for MWIDSP.")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Path to the input instances directory")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Prefix of the instances to test (e.g., 100_r014)")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Path to the output directory for CSV results")
    args = parser.parse_args()

    solve_all_baseline(
        in_dir_path=args.in_dir_path,
        instances_subset=args.instances_subset,
        out_dir_path=args.out_dir_path,
    )

if __name__ == '__main__':
    main()