import os
import time
import argparse
import networkx as nx
import math
from itertools import combinations
from utils import calc_initial_solution_cost_fast

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

def get_cell(pos, cell_size):
    """Calculates grid cell coordinates for a given position."""
    return (int(pos[0] // cell_size), int(pos[1] // cell_size))

def greedy_set_cover(candidates, uncovered, G):
    """Standard greedy set cover algorithm based on weight/covers ratio."""
    S = set()
    while uncovered:
        best_cand = None
        best_ratio = float('inf')
        best_covers = set()
        
        for cand in candidates:
            covers_new = (set(G.neighbors(cand)) | {cand}) & uncovered
            if not covers_new:
                continue
            ratio = G.nodes[cand]['weight'] / len(covers_new)
            if ratio < best_ratio:
                best_ratio = ratio
                best_cand = cand
                best_covers = covers_new
                
        if best_cand is None:
            break
            
        S.add(best_cand)
        uncovered -= best_covers
    return S

def run_phase1_grid(G, radius):
    """Phase 1: Generates initial solution using geometric grid partitioning."""
    cell_size = radius / math.sqrt(2)
    cells = {}
    for v in G.nodes():
        cx, cy = get_cell(G.nodes[v]['pos'], cell_size)
        cells.setdefault((cx, cy), []).append(v)
        
    S_init = set()
    for cell_nodes in cells.values():
        cell_uncovered = set(cell_nodes)
        S_cell = greedy_set_cover(cell_nodes, cell_uncovered, G)
        S_init.update(S_cell)
        
    global_uncovered = set(G.nodes())
    for v in S_init:
        global_uncovered.discard(v)
        global_uncovered -= set(G.neighbors(v))
        
    if global_uncovered:
        S_repair = greedy_set_cover(G.nodes(), global_uncovered, G)
        S_init.update(S_repair)
        
    return S_init

def cross_boundary_ls_heuristic(G, radius=0.14):
    """
    Cross-boundary Local Search Heuristic.
    Uses calc_initial_solution_cost_fast for performance evaluation.
    """
    start_time = time.time()
    
    # 1. Generate initial solution
    S = run_phase1_grid(G, radius)
    _, current_cost, _, _ = calc_initial_solution_cost_fast(S, G)
    
    # 2. Identify boundary nodes
    cell_size = radius / math.sqrt(2)
    boundary_nodes = set()
    for v in G.nodes():
        pos_v = G.nodes[v]['pos']
        cell_v = get_cell(pos_v, cell_size)
        
        is_boundary = False
        for u in G.neighbors(v):
            if get_cell(G.nodes[u]['pos'], cell_size) != cell_v:
                is_boundary = True
                break
        if is_boundary:
            boundary_nodes.add(v)
            
    # 3. Perform Local Search exclusively on boundary targets
    S_boundary = S & boundary_nodes
    improved = True
    
    while improved:
        improved = False
        
        for v in list(S_boundary):
            if v not in S: continue
            
            S_temp = S.copy()
            S_temp.remove(v)
            
            uncovered_temp = set()
            for u in list(G.neighbors(v)) + [v]:
                if u not in S_temp and not any(t in S_temp for t in G.neighbors(u)):
                    uncovered_temp.add(u)
                    
            if not uncovered_temp:
                S = S_temp
                _, current_cost, _, _ = calc_initial_solution_cost_fast(S, G)
                improved = True
                continue
                
            valid_cands = set()
            for cand in G.nodes():
                if cand not in S_temp and not any(t in S_temp for t in G.neighbors(cand)):
                    valid_cands.add(cand)
                    
            relevant_cands = [c for c in valid_cands if len((set(G.neighbors(c)) | {c}) & uncovered_temp) > 0]
            
            best_swap_cost = current_cost
            best_swap = None
            
            # Evaluate 1-to-1 swaps
            for cand in relevant_cands:
                combined_covers = set(G.neighbors(cand)) | {cand}
                if uncovered_temp.issubset(combined_covers):
                    S_new = S_temp | {cand}
                    _, new_cost, _, _ = calc_initial_solution_cost_fast(S_new, G)
                    if new_cost < best_swap_cost:
                        best_swap_cost = new_cost
                        best_swap = {cand}
                        
            # Evaluate 1-to-2 swaps
            if best_swap is None:
                for c1, c2 in combinations(relevant_cands, 2):
                    if c2 not in G.neighbors(c1):
                        combined_covers = set(G.neighbors(c1)) | {c1} | set(G.neighbors(c2)) | {c2}
                        if uncovered_temp.issubset(combined_covers):
                            S_new = S_temp | {c1, c2}
                            _, new_cost, _, _ = calc_initial_solution_cost_fast(S_new, G)
                            if new_cost < best_swap_cost:
                                best_swap_cost = new_cost
                                best_swap = {c1, c2}
                                
            # Apply best swap found
            if best_swap is not None:
                S = S_temp | best_swap
                current_cost = best_swap_cost
                
                for new_node in best_swap:
                    if new_node in boundary_nodes:
                        S_boundary.add(new_node)
                S_boundary.discard(v)
                improved = True

    time_elapsed = time.time() - start_time
    
    # Final cost calculation
    num_incorrect_nodes, final_cost, _, _ = calc_initial_solution_cost_fast(S, G)
    return S, num_incorrect_nodes, final_cost, time_elapsed

def solve_all_cross_ls(in_dir_path, instances_subset, out_dir_path):
    """Iterates over instances and records the heuristic performance."""
    out_file_name = f"{instances_subset}_cross_ls_results.csv"
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write("instance,num_incorrect,cost,time\n")
            
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
                print(f"[{file_name}] Already processed. Skipping.")
                continue
                
            file_path = os.path.join(in_dir_path, file_name)
            G = read_instance_with_pos(file_path)
            
            _, num_incorrect_nodes, cost, time_elapsed = cross_boundary_ls_heuristic(G)
            
            with open(out_file_path, 'a') as f:
                f.write(f"{file_name},{num_incorrect_nodes},{cost:.4f},{time_elapsed:.4f}\n")
            print(f"[{file_name}] Cost: {cost:.4f}, Time: {time_elapsed:.4f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_dir_path", required=True, help="Input directory")
    parser.add_argument("-s", "--instances_subset", required=True, help="Subset prefix")
    parser.add_argument("-o", "--out_dir_path", required=True, help="Output directory")
    args = parser.parse_args()
    
    solve_all_cross_ls(args.in_dir_path, args.instances_subset, args.out_dir_path)