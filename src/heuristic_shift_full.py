"""
Geometric Shifting Heuristic (Shift_Full) for the MWIDSP.
Acts as a deterministic multi-start diversification mechanism by applying 
4 geometric offsets to the spatial decomposition grid to overcome the Grid Dilemma.
"""

import os
import time
import argparse
import networkx as nx
import math
from itertools import combinations
from utils import calc_initial_solution_cost_fast

def read_instance_with_pos(file_path):
    """Reads instance file including the POSITIONS block at the end."""
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

def get_cell(pos, cell_size, offset_x=0, offset_y=0):
    """Calculates cell coordinates with a given spatial offset."""
    return (int((pos[0] - offset_x) // cell_size), int((pos[1] - offset_y) // cell_size))

def greedy_set_cover(candidates, uncovered, G):
    """Greedy selection for dominating set within a grid cell."""
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

def run_phase1_shifted(G, cell_size, offset_x, offset_y):
    """Initial solution generation for a specific grid offset."""
    cells = {}
    for v in G.nodes():
        cx, cy = get_cell(G.nodes[v]['pos'], cell_size, offset_x, offset_y)
        cells.setdefault((cx, cy), []).append(v)
        
    S_init = set()
    for cell_nodes in cells.values():
        cell_uncovered = set(cell_nodes)
        S_init.update(greedy_set_cover(cell_nodes, cell_uncovered, G))
        
    global_uncovered = set(G.nodes())
    for v in S_init:
        global_uncovered.discard(v)
        global_uncovered -= set(G.neighbors(v))
        
    if global_uncovered:
        S_init.update(greedy_set_cover(G.nodes(), global_uncovered, G))
        
    return S_init

def shift_full_heuristic(G, radius=0.14):
    """
    Grid Shift-Full Heuristic using 4 spatial offsets.
    VNS logic perfectly aligned with cross_ls for fair comparison.
    """
    start_time = time.time()
    cell_size = radius / math.sqrt(2)
    offsets = [(0, 0), (cell_size/2, 0), (0, cell_size/2), (cell_size/2, cell_size/2)]
    
    best_overall_cost = float('inf')
    best_overall_S = set()

    for ox, oy in offsets:
        S = run_phase1_shifted(G, cell_size, ox, oy)
        _, current_cost, _, _ = calc_initial_solution_cost_fast(S, G)
        
        improved = True
        while improved:
            improved = False
            for v in list(S):
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

                # 1-to-1 swap
                for cand in relevant_cands:
                    combined_covers = set(G.neighbors(cand)) | {cand}
                    if uncovered_temp.issubset(combined_covers):
                        S_new = S_temp | {cand}
                        _, new_cost, _, _ = calc_initial_solution_cost_fast(S_new, G)
                        if new_cost < best_swap_cost:
                            best_swap_cost = new_cost
                            best_swap = {cand}

                # 1-to-2 swap
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

                if best_swap is not None:
                    S = S_temp | best_swap
                    current_cost = best_swap_cost
                    improved = True
                    # Do NOT break to outer loop, continue optimizing other nodes
        
        if current_cost < best_overall_cost:
            best_overall_cost = current_cost
            best_overall_S = S

    time_elapsed = time.time() - start_time
    num_incorrect, final_cost, _, _ = calc_initial_solution_cost_fast(best_overall_S, G)
    return best_overall_S, num_incorrect, final_cost, time_elapsed

def solve_all_shift_full(in_dir_path, instances_subset, out_dir_path):
    """Processes all matching instances and logs results."""
    out_file_path = os.path.join(out_dir_path, f"{instances_subset}_shift_full_results.csv")
    os.makedirs(out_dir_path, exist_ok=True)
    
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write("instance,num_incorrect,cost,time\n")
            
    processed = set()
    if os.path.exists(out_file_path):
        with open(out_file_path, 'r') as f:
            lines = f.readlines()[1:]
            processed = {line.split(',')[0].strip() for line in lines if line.strip()}

    for file_name in sorted(os.listdir(in_dir_path)):
        if file_name.startswith(f"{instances_subset}_") and file_name not in processed:
            G = read_instance_with_pos(os.path.join(in_dir_path, file_name))
            _, num_inc, cost, dt = shift_full_heuristic(G)
            with open(out_file_path, 'a') as f:
                f.write(f"{file_name},{num_inc},{cost:.4f},{dt:.4f}\n")
            print(f"[{file_name}] Cost: {cost:.4f}, Time: {dt:.4f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_dir_path", required=True)
    parser.add_argument("-s", "--instances_subset", required=True)
    parser.add_argument("-o", "--out_dir_path", required=True)
    args = parser.parse_args()
    solve_all_shift_full(args.in_dir_path, args.instances_subset, args.out_dir_path)