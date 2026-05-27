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
            
            # Benefit (Numerator): Total weight of covered nodes + alternative routing saved
            benefit = node_weights_sum + other_edge_weights
            
            val = benefit / cost if cost > 0 else 0
            if val > max_val:
                max_val = val
                v_star = v
                
        if v_star is not None:
            S.add(v_star)
            
            neighbors = list(G_copy[v_star])
            G_copy.remove_node(v_star)
            for u in neighbors:
                if u in G_copy.nodes:
                    G_copy.remove_node(u)
                    
            # Recompute edge weights cache for remaining nodes
            edge_weights_cache = {u: 0 for u in G_copy.nodes}
            for u in G_copy.nodes:
                for v in G_copy[u]:
                    edge_weights_cache[u] += G[u][v]['weight']
        else:
            break
            
    num_incorrect, total_cost, _, _ = calc_initial_solution_cost(S, G)
    time_elapsed = perf_counter() - start_time
    
    return S, num_incorrect, total_cost, time_elapsed

def main():
    parser = ArgumentParser(description="Constructive heuristic baseline (GREEDY-NEW) for MWIDSP.")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Path to the input instances directory")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Prefix of the instances to test (e.g., 100_r014)")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    out_file_name = f"{args.instances_subset}_greedynew_results.csv"
    out_file_path = os.path.join(args.out_dir_path, out_file_name)
    os.makedirs(args.out_dir_path, exist_ok=True)
    
# 1. Initialize file and safely write header
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write("instance,num_incorrect,cost,time\n")
            
    # 2. Identify already processed instances to avoid duplicate work
    processed = set()
    if os.path.exists(out_file_path):
        with open(out_file_path, 'r') as f:
            for line in f.readlines()[1:]:
                if line.strip(): 
                    processed.add(line.split(',')[0].strip())

    # 3. Filter and execute only the exact target instances
    for file_name in sorted(os.listdir(args.in_dir_path)):
        if not file_name.startswith(f"{args.instances_subset}_") or not file_name.endswith(".rgg"):
            continue
            
        if file_name in processed:
            continue
            
        file_path = os.path.join(args.in_dir_path, file_name)
        G = read_instance(file_path)
        
        _, num_incorrect_nodes, cost, time_elapsed = greedy_new_baseline(G)
        
        # 4. Safely append results
        with open(out_file_path, 'a') as f:
            f.write(f"{file_name},{num_incorrect_nodes},{cost},{time_elapsed:.4f}\n")
        print(f"[Success] {file_name} - Baseline Finished | Cost: {cost} | Time: {time_elapsed:.4f}s")

if __name__ == '__main__':
    main()