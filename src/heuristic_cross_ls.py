"""
Cross-Boundary LS Heuristic for the MWIDSP.
Demonstrates the utility of spatial decomposition (Phase 1) and 
Geometrically-Bounded VNS (Phase 2) prior to applying geometric shifting.
"""

import os
import math
from time import perf_counter
from itertools import combinations
from argparse import ArgumentParser
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

def get_cell(pos, cell_size):
    """Assigns a node to a 2D grid cell based on coordinates."""
    x, y = pos
    return (int(x / cell_size), int(y / cell_size))

def cross_boundary_ls_heuristic(G, radius=0.14):
    """Executes the base spatial heuristic (Phase 1 + Phase 2) without offsets."""
    start_time = perf_counter()
    cell_size = radius / math.sqrt(2)
    
    # Assign nodes to grid cells
    cells = {}
    for v in G.nodes():
        cx, cy = get_cell(G.nodes[v]['pos'], cell_size)
        if (cx, cy) not in cells:
            cells[(cx, cy)] = []
        cells[(cx, cy)].append(v)
        
    S = set()
    uncovered = set(G.nodes())
    forbidden_nodes = set()
    
    # Visit cells in descending order of node density
    sorted_cells = sorted(cells.keys(), key=lambda c: len(cells[c]), reverse=True)

    # =======================================================
    # [Phase 1] Spatial Decomposition & Cross-Boundary Scoring
    # =======================================================
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
                    u_cell = get_cell(G.nodes[u]['pos'], cell_size)
                    
                    alt_edges = [G[u][t]['weight'] for t in G.neighbors(u) if t != v and t in uncovered]
                    min_alt = min(alt_edges) if alt_edges else 0
                    
                    # Reward coverage only if it spans across different cells
                    if u_cell != cell_id:
                        benefit_out += G.nodes[u]['weight'] + min_alt
                    
                    # Standard benefit metric for fallback
                    benefit_fallback += G.nodes[u]['weight'] + min_alt
            
            score = benefit_out / cost if cost > 0 else 0
            fallback_score = benefit_fallback / cost if cost > 0 else 0
            
            if score > best_score:
                best_score = score
                best_node = v
                
            if fallback_score > best_fallback_score:
                best_fallback_score = fallback_score
                best_fallback_node = v
                
        # Select the best cross-boundary node, or fallback to local efficiency
        selected = best_node if best_score > 0 else best_fallback_node
        
        if selected is not None:
            S.add(selected)
            uncovered.discard(selected)
            forbidden_nodes.add(selected)
            for u in G.neighbors(selected):
                uncovered.discard(u)
                forbidden_nodes.add(u)

    # Greedy repair for any remaining blind spots
    for v in list(uncovered):
        if v not in forbidden_nodes:
            S.add(v)
            uncovered.discard(v)
            forbidden_nodes.add(v)
            for u in G.neighbors(v):
                uncovered.discard(u)
                forbidden_nodes.add(u)

    # =======================================================
    # [Phase 2] Geometrically-Bounded VNS (GB-VNS)
    # =======================================================
    _, current_cost, _, _ = calc_initial_solution_cost(S, G)
    improved = True
    
    while improved:
        improved = False
        
        for v in list(S):
            S_temp = S.copy()
            S_temp.remove(v)
            
            # Identify nodes that lost coverage due to the removal of v
            uncovered_temp = set()
            for u in list(G.neighbors(v)) + [v]:
                if u not in S_temp and not any(t in S_temp for t in G.neighbors(u)):
                    uncovered_temp.add(u)
                    
            best_swap = None
            best_swap_cost = current_cost
            
            # [Type 1] 1-to-0 Exchange (Redundancy Removal)
            if len(uncovered_temp) == 0:
                _, new_cost, _, _ = calc_initial_solution_cost(S_temp, G)
                if new_cost < best_swap_cost:
                    best_swap_cost = new_cost
                    best_swap = set()
            
            else:
                # Find valid, independent replacement candidates
                valid_cands = set()
                for cand in G.nodes():
                    if cand not in S_temp and not any(t in S_temp for t in G.neighbors(cand)):
                        valid_cands.add(cand)
                
                relevant_cands = [c for c in valid_cands if len((set(G.neighbors(c)) | {c}) & uncovered_temp) > 0]
                
                # [Type 2] 1-to-1 Exchange
                for c1 in relevant_cands:
                    c1_covers = set(G.neighbors(c1)) | {c1}
                    if uncovered_temp.issubset(c1_covers): 
                        S_new = S_temp | {c1}
                        _, new_cost, _, _ = calc_initial_solution_cost(S_new, G)
                        if new_cost < best_swap_cost:
                            best_swap_cost = new_cost
                            best_swap = {c1}
                            
                # [Type 3] 1-to-2 Exchange (Boundary Stitching)
                if best_swap is None: 
                    for c1, c2 in combinations(relevant_cands, 2):
                        if c2 not in G.neighbors(c1): 
                            combined_covers = set(G.neighbors(c1)) | {c1} | set(G.neighbors(c2)) | {c2}
                            if uncovered_temp.issubset(combined_covers):
                                S_new = S_temp | {c1, c2}
                                _, new_cost, _, _ = calc_initial_solution_cost(S_new, G)
                                if new_cost < best_swap_cost:
                                    best_swap_cost = new_cost
                                    best_swap = {c1, c2}
            
            # Execute the best identified swap
            if best_swap is not None:
                S = S_temp | best_swap
                current_cost = best_swap_cost
                improved = True
                break

    num_incorrect, final_cost, _, _ = calc_initial_solution_cost(S, G)
    time_elapsed = perf_counter() - start_time
    
    return S, num_incorrect, final_cost, time_elapsed

def solve_all_cross_ls(in_dir_path, instances_subset, out_dir_path):
    out_file_name = f"{instances_subset}_cross_ls_results.csv"
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    # Resume capability: write header if file is missing or empty
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
        if file_name.startswith(instances_subset):
            if file_name in processed:
                print(f"[{file_name}] Already processed. Skipping.")
                continue
                
            file_path = os.path.join(in_dir_path, file_name)
            G = read_instance_with_pos(file_path)
            
            _, num_incorrect_nodes, cost, time_elapsed = cross_boundary_ls_heuristic(G)
            
            with open(out_file_path, 'a') as f:
                f.write(f"{file_name},{num_incorrect_nodes},{cost},{time_elapsed:.4f}\n")
                
            print(f"[{file_name}] Cross+LS Finished | Cost: {cost} | Time: {time_elapsed:.4f}s")

if __name__ == '__main__':
    parser = ArgumentParser(description="Cross-Boundary LS Heuristic (Base Spatial Decomposition)")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Path to input instances")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Instance prefix to process")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Output directory path")
    args = parser.parse_args()
    
    solve_all_cross_ls(args.in_dir_path, args.instances_subset, args.out_dir_path)