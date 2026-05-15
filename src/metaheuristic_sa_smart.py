"""
Exploitation-centric Simulated Annealing (SA_smart) Metaheuristic.
Acts as a baseline that initializes with a cost-aware greedy skeleton and 
aggressively fine-tunes the local geometric topology using smart, cost-based repair logic.
"""

from argparse import ArgumentParser
import os
import random
random.seed(42)
import math
from time import perf_counter
import networkx as nx
import datetime

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

def smart_greedy_initial(G):
    """Generates a cost-aware initial skeleton to prioritize exploitation."""
    S = set()
    uncovered = set(G.nodes())
    forbidden = set()
    
    while uncovered:
        available = [v for v in G.nodes() if v not in forbidden]
        
        if not available:
            for v in list(uncovered):
                if v not in forbidden:
                    S.add(v)
                    uncovered.discard(v)
                    forbidden.add(v)
                    for u in G.neighbors(v):
                        uncovered.discard(u)
                        forbidden.add(u)
            break
            
        best_node = None
        best_score = -1.0
        
        for v in available:
            # Number of nodes covered (including itself)
            covers = len((set(G.neighbors(v)) | {v}) & uncovered)
            cost = G.nodes[v]['weight']
            
            # Efficiency score (avoid division by zero)
            score = covers / cost if cost > 0 else 0
            
            if score > best_score:
                best_score = score
                best_node = v
                
        S.add(best_node)
        uncovered.discard(best_node)
        forbidden.add(best_node)
        for u in G.neighbors(best_node):
            uncovered.discard(u)
            forbidden.add(u)
            
    return S

def simulated_annealing_smart(G):
    """Executes the SA_smart algorithm using subtle perturbation and cost-aware repair."""
    start_time = perf_counter()
    num_nodes = len(G.nodes)
    
    # Execution time limit scaling (e.g., 1500 seconds for 500 nodes)
    time_limit = 3.0 * num_nodes 
    print(f"  [Info] SA_smart optimization started. Time limit: {time_limit}s")
    
    # Initialize with a high-quality skeleton
    current_S = smart_greedy_initial(G)
    num_incorrect, current_cost, _, _ = calc_initial_solution_cost(current_S, G)
    
    best_S = current_S.copy()
    best_cost = current_cost
    
    T = 10000.0
    cooling_rate = 0.999 
    
    iterations = 0
    while perf_counter() - start_time < time_limit:
        iterations += 1
        
        new_S = current_S.copy()
        
        # Perturbation: Remove only a single node
        if new_S:
            node_to_remove = random.choice(list(new_S))
            new_S.remove(node_to_remove)
            
        uncovered = set()
        for v in G.nodes():
            if v not in new_S and not any(u in new_S for u in G.neighbors(v)):
                uncovered.add(v)
                
        # Cost-aware Repair Logic (Smart Patching)
        while uncovered:
            valid_cands = [v for v in G.nodes() if v not in new_S and not any(u in new_S for u in G.neighbors(v))]
            
            if not valid_cands:
                break
                
            scored_cands = []
            for v in valid_cands:
                covers_new = len((set(G.neighbors(v)) | {v}) & uncovered)
                if covers_new > 0:
                    cost = G.nodes[v]['weight']
                    score = covers_new / cost if cost > 0 else 0
                    scored_cands.append((score, v))
            
            if not scored_cands:
                break
                
            # Sort candidates by cost-efficiency and randomly select from top 3 (RCL)
            scored_cands.sort(key=lambda x: x[0], reverse=True)
            top_k = scored_cands[:min(3, len(scored_cands))]
            selected_v = random.choice(top_k)[1]
            
            new_S.add(selected_v)
            uncovered -= (set(G.neighbors(selected_v)) | {selected_v})
            
        num_incorrect, new_cost, _, _ = calc_initial_solution_cost(new_S, G)
        delta = new_cost - current_cost
        
        # SA Acceptance Criterion
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_S = new_S
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_S = current_S.copy()
                
        # Cool down and reheat if frozen
        T *= cooling_rate
        if T < 0.1:
            T = 10000.0 
            
    time_elapsed = perf_counter() - start_time
    num_incorrect, final_cost, _, _ = calc_initial_solution_cost(best_S, G)
    
    return best_S, num_incorrect, final_cost, time_elapsed, iterations

def solve_all_sa_smart(in_dir_path, instances_subset, out_dir_path):
    out_file_name = f"{instances_subset}_sa_smart_results.csv"
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    # Resume capability: write header if file is missing or empty
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write("instance,num_incorrect,cost,time,iterations\n")
            
    processed = set()
    if os.path.exists(out_file_path):
        with open(out_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]: 
                if line.strip():
                    processed.add(line.split(',')[0].strip())
    
    target_files = [f for f in sorted(os.listdir(in_dir_path)) 
                    if f.startswith(f"{instances_subset}_") and f not in processed]
    num_remaining = len(target_files)
    
    if num_remaining > 0:
        est_time_per_instance = 1500.0 
        total_est_seconds = num_remaining * est_time_per_instance
        est_end_time = datetime.datetime.now() + datetime.timedelta(seconds=total_est_seconds)
        
        print(f"\n=====================================================")
        print(f" Instances remaining in queue : {num_remaining}")
        print(f" Estimated total time         : {total_est_seconds / 60:.1f} minutes ({total_est_seconds / 3600:.1f} hours)")
        print(f" Estimated completion time    : {est_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=====================================================\n")
    else:
        print("All target instances have already been processed.")
        return 

    for file_name in target_files:
        file_path = os.path.join(in_dir_path, file_name)
        
        G = read_instance_with_pos(file_path)
        _, num_incorrect_nodes, cost, time_elapsed, iters = simulated_annealing_smart(G)
        
        # Immediately append result to prevent data loss
        with open(out_file_path, 'a') as f:
            f.write(f"{file_name},{num_incorrect_nodes},{cost},{time_elapsed:.2f},{iters}\n")
            
        print(f"[{file_name}] Saved | Cost: {cost} | Time: {time_elapsed:.2f}s | Iterations: {iters}")

if __name__ == '__main__':
    parser = ArgumentParser(description="SA_smart Metaheuristic Benchmark")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Path to input instances")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Instance prefix to process")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Output directory path")
    args = parser.parse_args()
    
    solve_all_sa_smart(args.in_dir_path, args.instances_subset, args.out_dir_path)