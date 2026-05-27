"""
Geometric Shifting Heuristic (Shift_Full) for the MWIDSP.
Acts as a deterministic multi-start diversification mechanism by applying 
4 geometric offsets to the spatial decomposition grid.
Utilizes an edge-aware sum scoring strategy and deterministic tie-breaking.
"""

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

def get_shifted_cell(pos, cell_size, offset_x=0.0, offset_y=0.0):
    """Calculates cell coordinates with a given spatial offset using floor division."""
    return (int((pos[0] - offset_x) // cell_size), int((pos[1] - offset_y) // cell_size))

# =====================================================================
# Phase 1: Edge-Aware Scoring Initialization
# =====================================================================
def run_phase1_edge_aware_sum(G, cell_size, offset_x, offset_y):
    """Initial solution generation for a specific grid offset (sum strategy)."""
    cells = {}
    for v in G.nodes():
        cx, cy = get_shifted_cell(G.nodes[v]['pos'], cell_size, offset_x, offset_y)
        if (cx, cy) not in cells:
            cells[(cx, cy)] = []
        cells[(cx, cy)].append(v)
        
    S = set()
    uncovered = set(G.nodes())
    forbidden_nodes = set()
    
    sorted_cells = sorted(cells.keys(), key=lambda c: len(cells[c]), reverse=True)

    for cell_id in sorted_cells:
        candidates = [v for v in cells[cell_id] if v not in forbidden_nodes]
        if not candidates:
            continue
            
        best_node = None
        best_score = -1.0
        best_fallback_node = None
        best_fallback_score = -1.0
        
        for v in candidates:
            cost = G.nodes[v]['weight']
            benefit_out = 0
            benefit_fallback = 0 
            
            for u in G.neighbors(v):
                if u in uncovered:
                    cost += G[v][u]['weight']
                    u_cell = get_shifted_cell(G.nodes[u]['pos'], cell_size, offset_x, offset_y)
                    
                    alt_edges = [G[u][t]['weight'] for t in G.neighbors(u) if t != v and t in uncovered]
                    alt_cost = sum(alt_edges) if alt_edges else 0
                    
                    if u_cell != cell_id:
                        benefit_out += G.nodes[u]['weight'] + alt_cost
                    
                    benefit_fallback += G.nodes[u]['weight'] + alt_cost
            
            score = benefit_out / cost if cost > 0 else 0
            fallback_score = benefit_fallback / cost if cost > 0 else 0
            
            if score > best_score:
                best_score = score
                best_node = v
                
            if fallback_score > best_fallback_score:
                best_fallback_score = fallback_score
                best_fallback_node = v
                
        selected = best_node if best_score > 0 else best_fallback_node
        
        if selected is not None:
            S.add(selected)
            uncovered.discard(selected)
            forbidden_nodes.add(selected)
            for u in G.neighbors(selected):
                uncovered.discard(u)
                forbidden_nodes.add(u)

    for v in list(uncovered):
        if v not in forbidden_nodes:
            S.add(v)
            uncovered.discard(v)
            forbidden_nodes.add(v)
            for u in G.neighbors(v):
                uncovered.discard(u)
                forbidden_nodes.add(u)

    return S

# =====================================================================
# Phase 2: Local Search (Shift-Full Integration)
# =====================================================================
def shift_full_heuristic(G, radius=0.14):
    """
    Grid Shift-Full Heuristic using 4 spatial offsets.
    Includes deterministic tie-breaking based on cost, efficiency, and node ID.
    """
    start_time = time.time()
    cell_size = radius / math.sqrt(2)
    offsets = [(0.0, 0.0), (cell_size/2, 0.0), (0.0, cell_size/2), (cell_size/2, cell_size/2)]
    
    best_overall_cost = float('inf')
    best_overall_S = set()

    for ox, oy in offsets:
        S = run_phase1_edge_aware_sum(G, cell_size, ox, oy)
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
                
                old_eff = len(set(G.neighbors(v)) | {v})
                
                best_swap_criteria = (current_cost, -old_eff, v)
                best_swap = None

                # 1-to-1 swap
                for cand in relevant_cands:
                    c1_covers = set(G.neighbors(cand)) | {cand}
                    if uncovered_temp.issubset(c1_covers):
                        S_new = S_temp | {cand}
                        _, new_cost, _, _ = calc_initial_solution_cost_fast(S_new, G)
                        
                        efficiency = len(c1_covers)
                        current_criteria = (new_cost, -efficiency, cand)
                        
                        if current_criteria < best_swap_criteria:
                            best_swap_criteria = current_criteria
                            best_swap = {cand}

                # 1-to-2 swap
                if best_swap is None:
                    for c1, c2 in combinations(relevant_cands, 2):
                        if c2 not in G.neighbors(c1):
                            combined_covers = set(G.neighbors(c1)) | {c1} | set(G.neighbors(c2)) | {c2}
                            if uncovered_temp.issubset(combined_covers):
                                S_new = S_temp | {c1, c2}
                                _, new_cost, _, _ = calc_initial_solution_cost_fast(S_new, G)
                                
                                efficiency = len(combined_covers)
                                current_criteria = (new_cost, -efficiency, min(c1, c2))
                                
                                if current_criteria < best_swap_criteria:
                                    best_swap_criteria = current_criteria
                                    best_swap = {c1, c2}

                if best_swap is not None:
                    S = S_temp | best_swap
                    current_cost = best_swap_criteria[0]
                    improved = True
        
        if current_cost < best_overall_cost:
            best_overall_cost = current_cost
            best_overall_S = S.copy()

    time_elapsed = time.time() - start_time
    num_incorrect, final_cost, _, _ = calc_initial_solution_cost_fast(best_overall_S, G)
    return best_overall_S, num_incorrect, final_cost, time_elapsed

def solve_all_shift_full(in_dir_path, instances_subset, out_dir_path):
    """Processes all matching instances and logs results."""
    out_file_name = f"{instances_subset}_shift_full_results.csv"
    out_file_path = os.path.join(out_dir_path, out_file_name)
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
        if file_name.startswith(f"{instances_subset}_"):
            if file_name in processed:
                print(f"[{file_name}] Already processed. Skipping.")
                continue
            
            file_path = os.path.join(in_dir_path, file_name)
            G = read_instance_with_pos(file_path)
            
            _, num_inc, cost, dt = shift_full_heuristic(G)
            
            with open(out_file_path, 'a') as f:
                f.write(f"{file_name},{num_inc},{cost:.4f},{dt:.4f}\n")
                
            print(f"[{file_name}] Cost: {cost:.4f} | Time: {dt:.4f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Shift-Full Heuristic")
    parser.add_argument("-i", "--in_dir_path", required=True, help="Input directory")
    parser.add_argument("-s", "--instances_subset", required=True, help="Instance prefix to process")
    parser.add_argument("-o", "--out_dir_path", required=True, help="Output directory")
    args = parser.parse_args()
    
    solve_all_shift_full(args.in_dir_path, args.instances_subset, args.out_dir_path)