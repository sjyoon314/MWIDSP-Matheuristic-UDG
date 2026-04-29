"""
Exploration-centric GRASP (Greedy Randomized Adaptive Search Procedure) Metaheuristic.
Acts as a computationally intensive baseline by continuously generating diverse 
initial skeletons using a Restricted Candidate List (RCL) without local search refinement.
"""

from argparse import ArgumentParser
import os
import random
from time import perf_counter
import networkx as nx

from utils import calc_initial_solution_cost

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

def grasp_baseline_heuristic(G, time_limit=1500.0, rcl_size=3):
    """Executes GRASP within the time limit and returns the best solution found."""
    start_time = perf_counter()
    
    best_overall_S = set()
    best_overall_cost = float('inf')
    best_overall_incorrect = 0
    iteration_count = 0
    
    # Cache optimization: Precompute sum of incident edge weights for baseline speed
    static_edge_weights = {u: sum(G[u][v]['weight'] for v in G[u]) for u in G.nodes}
    
    while perf_counter() - start_time < time_limit:
        iteration_count += 1
        S = set()
        G_copy = G.copy()
        
        dyn_edge_cache = static_edge_weights.copy()
        
        while len(G_copy.nodes) > 0:
            candidates_scores = []
            
            for v in G_copy.nodes:
                uv_w = 0
                other_w = 0
                nw_sum = 0
                neighbors_v = list(G_copy[v])
                
                for u in neighbors_v:
                    nw_sum += G_copy.nodes[u]['weight']
                    uv_w += G_copy[u][v]['weight']
                    other_w += dyn_edge_cache[u] - G_copy[u][v]['weight']
                    
                cost = G_copy.nodes[v]['weight'] + uv_w
                benefit = nw_sum + other_w
                score = benefit / cost if cost > 0 else 0
                
                candidates_scores.append((score, v))
            
            # Sort candidates descending by score
            candidates_scores.sort(key=lambda x: x[0], reverse=True)
            
            # [GRASP Core] Randomly pick from the Restricted Candidate List (RCL)
            top_k = candidates_scores[:rcl_size]
            selected_node = random.choice(top_k)[1]
            
            S.add(selected_node)
            
            # Dynamically update the graph and cache
            neighbors_of_selected = list(G_copy[selected_node])
            for v in neighbors_of_selected:
                for u in G_copy[v]:
                    dyn_edge_cache[u] -= G_copy[v][u]['weight']
                G_copy.remove_node(v) 
                
            if selected_node in G_copy: 
                for u in G_copy[selected_node]:
                    dyn_edge_cache[u] -= G_copy[selected_node][u]['weight']
                G_copy.remove_node(selected_node)
        
        # Evaluate the constructed solution at the end of each iteration
        num_incorrect, current_cost, _, _ = calc_initial_solution_cost(S, G)
        
        # Update the global best
        if current_cost < best_overall_cost:
            best_overall_cost = current_cost
            best_overall_S = S
            best_overall_incorrect = num_incorrect
            
    time_elapsed = perf_counter() - start_time
    
    return best_overall_S, best_overall_incorrect, best_overall_cost, time_elapsed, iteration_count

def solve_all_grasp(in_dir_path, instances_subset, out_dir_path):
    out_file_name = f"{instances_subset}_grasp_baseline_1500s_results.csv" 
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    # Resume capability: write header if file is missing or empty
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write("instance,num_incorrect,cost,time,iterations\n")
            
    # Read already processed instances to skip them
    processed = set()
    if os.path.exists(out_file_path):
        with open(out_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]: # Skip header
                if line.strip():
                    processed.add(line.split(',')[0].strip())
                
    for file_name in sorted(os.listdir(in_dir_path)):
        if file_name.startswith(instances_subset):
            if file_name in processed:
                print(f"[{file_name}] Already processed. Skipping.")
                continue
                
            file_path = os.path.join(in_dir_path, file_name)
            G = read_instance_with_pos(file_path)
            
            # Execute GRASP with a rigid 1,500s limit
            _, num_incorrect_nodes, cost, total_time, iters = grasp_baseline_heuristic(G, time_limit=1500.0, rcl_size=3)
            
            # Immediately append result to prevent data loss
            with open(out_file_path, 'a') as f:
                f.write(f"{file_name},{num_incorrect_nodes},{cost},{total_time:.2f},{iters}\n")
                
            print(f"[{file_name}] Saved | Cost: {cost} | Time: {total_time:.2f}s | Iterations: {iters}")

if __name__ == '__main__':
    parser = ArgumentParser(description="GRASP Metaheuristic Benchmark (1500s limit)")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Path to input instances")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Instance prefix to process")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Output directory path")
    args = parser.parse_args()
    
    solve_all_grasp(args.in_dir_path, args.instances_subset, args.out_dir_path)