"""
Unified Sensitivity Analysis for Neighborhood Size (k) in GB-VNS.
Supports both 'Grid' and 'Clique' based initialization strategies.
Uses pure node-weight scoring without edge-weight noise.
"""

import os
import math
from time import perf_counter
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
from argparse import ArgumentParser

from utils import calc_initial_solution_cost, calc_initial_solution_cost_fast

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def read_instance_with_pos(file_path):
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
        parts = lines[idx].split()
        u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
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
    return (int(pos[0] / cell_size), int(pos[1] / cell_size))

def classify_environment(filename):
    if "nw10_ew1000" in filename: return "EG (Edge-oriented)"
    if "nw100_ew100" in filename: return "NG (Neutral)"
    if "nw1000_ew10" in filename: return "VG (Node-oriented)"
    return None

# ---------------------------------------------------------
# Phase 1 Initializers (Pure Node-Weight Scoring)
# ---------------------------------------------------------

def run_phase1_grid(G, radius=0.14):
    cell_size = radius / math.sqrt(2)
    cells = {}
    for v in G.nodes():
        cx, cy = get_cell(G.nodes[v]['pos'], cell_size)
        if (cx, cy) not in cells: cells[(cx, cy)] = []
        cells[(cx, cy)].append(v)
        
    S = set()
    uncovered = set(G.nodes())
    forbidden_nodes = set()
    sorted_cells = sorted(cells.keys(), key=lambda c: len(cells[c]), reverse=True)

    for cell_id in sorted_cells:
        candidates = [v for v in cells[cell_id] if v not in forbidden_nodes]
        if not candidates: continue
            
        best_node, best_score = None, -1.0
        best_fallback_node, best_fallback_score = None, -1.0
        
        for v in candidates:
            cost = G.nodes[v]['weight']
            # Core logic: Exclude edge weights, evaluate pure node coverage
            covers_out = 0
            covers_in = 0
            
            for u in G.neighbors(v):
                if u in uncovered:
                    u_cell = get_cell(G.nodes[u]['pos'], cell_size)
                    if u_cell != cell_id: covers_out += 1
                    else: covers_in += 1
            
            # Add bonus for cross-boundary coverage, normalized by cost
            score = (covers_out * 1.5 + covers_in) / cost if cost > 0 else 0
            fallback_score = (covers_in) / cost if cost > 0 else 0
            
            if score > best_score: best_score, best_node = score, v
            if fallback_score > best_fallback_score: best_fallback_score, best_fallback_node = fallback_score, v
                
        selected = best_node if best_score > 0 else best_fallback_node
        if selected is not None:
            S.add(selected)
            uncovered.discard(selected)
            forbidden_nodes.add(selected)
            for u in G.neighbors(selected):
                uncovered.discard(u)
                forbidden_nodes.add(u)

    # Greedy repair
    for v in list(uncovered):
        if v not in forbidden_nodes:
            S.add(v)
            uncovered.discard(v)
            forbidden_nodes.add(v)
            for u in G.neighbors(v):
                uncovered.discard(u)
                forbidden_nodes.add(u)
    return S

def run_phase1_clique(G):
    maximal_cliques = list(nx.find_cliques(G))
    sorted_cliques = sorted(maximal_cliques, key=len, reverse=True) # Size strategy
    
    S = set()
    uncovered = set(G.nodes())
    forbidden_nodes = set()
    
    for clique in sorted_cliques:
        candidates = [v for v in clique if v not in forbidden_nodes]
        if not candidates: continue
            
        best_node, best_score = None, -1.0
        best_fallback_node, best_fallback_score = None, -1.0
        
        for v in candidates:
            cost = G.nodes[v]['weight']
            covers_out = 0
            covers_in = 0
            
            for u in G.neighbors(v):
                if u in uncovered:
                    if u not in clique: covers_out += 1
                    else: covers_in += 1
            
            score = (covers_out * 1.5 + covers_in) / cost if cost > 0 else 0
            fallback_score = (covers_in) / cost if cost > 0 else 0
            
            if score > best_score: best_score, best_node = score, v
            if fallback_score > best_fallback_score: best_fallback_score, best_fallback_node = fallback_score, v
                
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

# ---------------------------------------------------------
# Dynamic VNS Core
# ---------------------------------------------------------
def run_dynamic_vns(G, S_init, k_max):
    start_time = perf_counter()
    S = S_init.copy()
    
    _, current_cost, _, _ = calc_initial_solution_cost_fast(S, G) 
    improved = True
    
    while improved:
        improved = False
        for v in list(S):
            S_temp = S.copy()
            S_temp.remove(v)
            
            uncovered_temp = set()
            for u in list(G.neighbors(v)) + [v]:
                if u not in S_temp and not any(t in S_temp for t in G.neighbors(u)):
                    uncovered_temp.add(u)
                    
            best_swap = None
            best_swap_cost = current_cost
            
            if len(uncovered_temp) == 0:
                _, new_cost, _, _ = calc_initial_solution_cost_fast(S_temp, G)
                if new_cost < best_swap_cost:
                    best_swap_cost, best_swap = new_cost, set()
            else:
                valid_cands = set(cand for cand in G.nodes() if cand not in S_temp and not any(t in S_temp for t in G.neighbors(cand)))
                relevant_cands = [c for c in valid_cands if len((set(G.neighbors(c)) | {c}) & uncovered_temp) > 0]
                
                for k in range(1, k_max + 1):
                    for combo in combinations(relevant_cands, k):
                        is_independent = True
                        if k > 1:
                            for c1, c2 in combinations(combo, 2):
                                if c2 in G.neighbors(c1):
                                    is_independent = False
                                    break
                        if not is_independent: continue
                        
                        combined_covers = set()
                        for c in combo:
                            combined_covers |= set(G.neighbors(c)) | {c}
                            
                        if uncovered_temp.issubset(combined_covers):
                            S_new = S_temp | set(combo)
                            _, new_cost, _, _ = calc_initial_solution_cost_fast(S_new, G)
                            if new_cost < best_swap_cost:
                                best_swap_cost = new_cost
                                best_swap = set(combo)

            if best_swap is not None:
                S = S_temp | best_swap
                current_cost = best_swap_cost
                improved = True
                break

    _, final_cost, _, _ = calc_initial_solution_cost_fast(S, G)
    time_elapsed = perf_counter() - start_time
    
    return final_cost, time_elapsed

# ---------------------------------------------------------
# Execution & Plotting Wrapper
# ---------------------------------------------------------
def run_batch_sensitivity_analysis(in_dir_path, out_dir_path, init_strategy):
    k_values = [1, 2, 3, 4, 5]
    
    env_results = {
        "EG (Edge-oriented)": {"costs": {k: [] for k in k_values}, "times": {k: [] for k in k_values}},
        "NG (Neutral)": {"costs": {k: [] for k in k_values}, "times": {k: [] for k in k_values}},
        "VG (Node-oriented)": {"costs": {k: [] for k in k_values}, "times": {k: [] for k in k_values}}
    }
    
    os.makedirs(out_dir_path, exist_ok=True)
    
    for filename in os.listdir(in_dir_path):
        if not filename.endswith(".rgg"): continue 
        
        env_name = classify_environment(filename)
        if env_name is None: continue
            
        file_path = os.path.join(in_dir_path, filename)
        print(f"\n[Processing: {init_strategy.upper()}] {filename} -> {env_name}")
        G = read_instance_with_pos(file_path)
        
        # 1. Generate initial solution (executed once to ensure fairness)
        t_init_start = perf_counter()
        if init_strategy == 'grid':
            S_init = run_phase1_grid(G)
        elif init_strategy == 'clique':
            S_init = run_phase1_clique(G)
        else:
            raise ValueError("Invalid initialization strategy provided.")
        t_init = perf_counter() - t_init_start
        
        # 2. Execute VNS for each k value
        for k in k_values:
            cost, t_vns = run_dynamic_vns(G, S_init, k_max=k)
            t_total = t_init + t_vns
            
            env_results[env_name]["costs"][k].append(cost)
            env_results[env_name]["times"][k].append(t_total)
            print(f"  k={k} | Cost: {cost:.2f}, Time: {t_total:.4f}s")
            
    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Sensitivity Analysis of $k$ (Init: {init_strategy.upper()})", fontsize=16, fontweight='bold')
    
    env_keys = ["EG (Edge-oriented)", "NG (Neutral)", "VG (Node-oriented)"]
    
    for idx, env_name in enumerate(env_keys):
        results = env_results[env_name]
        
        # Handle missing data
        if not results["costs"][1]: 
            axes[idx].set_title(f"{env_name}\n(No data)")
            continue
            
        # Calculate average cost
        avg_costs = [sum(results["costs"][k])/len(results["costs"][k]) for k in k_values]
        
        ax1 = axes[idx]
        
        # Plot objective cost (using default styling)
        ax1.plot(k_values, avg_costs, color='#1f77b4', marker='o', linestyle='-', linewidth=2.5, markersize=8, label='Avg Objective Cost')
        
        ax1.set_xlabel('Maximum Swap Size ($k$)', fontsize=12)
        ax1.set_ylabel('Average Cost', fontsize=12)
        ax1.set_xticks(k_values)
        
        # Apply clean grid styling
        ax1.grid(True, which="major", linestyle='-', alpha=0.4)
        ax1.grid(True, which="minor", linestyle=':', alpha=0.2)
        
        ax1.legend(fontsize=11, loc='upper right')
        
        num_files = len(results["costs"][1])
        ax1.set_title(f"{env_name}\n(Averaged over {num_files} instances)", fontsize=14)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = os.path.join(out_dir_path, f'k_sensitivity_{init_strategy}.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Sensitivity plot successfully saved to: {save_path}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True, help="Input directory")
    parser.add_argument('--out_dir', type=str, required=True, help="Output directory")
    parser.add_argument('--strategy', type=str, choices=['grid', 'clique'], required=True, help="Initialization strategy")
    args = parser.parse_args()
    
    run_batch_sensitivity_analysis(args.in_dir, args.out_dir, args.strategy)