"""
Sensitivity Analysis for Neighborhood Size (k) in GB-VNS.
Evaluates the trade-off between optimization cost and computation time
across different maximum swap sizes (k=1 to 5) in EG, NG, and VG environments.
Generates a comparative plot saved in the results/compare directory.
"""

import os
import math
from time import perf_counter
from itertools import combinations
import matplotlib.pyplot as plt
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
    return (int(pos[0] / cell_size), int(pos[1] / cell_size))

def cross_boundary_ls_dynamic_k(G, k_max, radius=0.14):
    """Executes the geometric heuristic with a dynamically bounded maximum swap size (k)."""
    start_time = perf_counter()
    cell_size = radius / math.sqrt(2)
    
    # Phase 1: Spatial Decomposition
    cells = {}
    for v in G.nodes():
        cx, cy = get_cell(G.nodes[v]['pos'], cell_size)
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
            
        best_node, best_score = None, -1.0
        best_fallback_node, best_fallback_score = None, -1.0
        
        for v in candidates:
            cost = G.nodes[v]['weight']
            benefit_out, benefit_fallback = 0, 0
            for u in G.neighbors(v):
                if u in uncovered:
                    cost += G[v][u]['weight']
                    u_cell = get_cell(G.nodes[u]['pos'], cell_size)
                    alt_edges = [G[u][t]['weight'] for t in G.neighbors(u) if t != v and t in uncovered]
                    min_alt = min(alt_edges) if alt_edges else 0
                    
                    if u_cell != cell_id:
                        benefit_out += G.nodes[u]['weight'] + min_alt
                    benefit_fallback += G.nodes[u]['weight'] + min_alt
            
            score = benefit_out / cost if cost > 0 else 0
            fallback_score = benefit_fallback / cost if cost > 0 else 0
            if score > best_score: 
                best_score, best_node = score, v
            if fallback_score > best_fallback_score: 
                best_fallback_score, best_fallback_node = fallback_score, v
                
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

    _, current_cost, _, _ = calc_initial_solution_cost(S, G)
    improved = True
    
    # Phase 2: Dynamic Geometrically-Bounded VNS
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
                _, new_cost, _, _ = calc_initial_solution_cost(S_temp, G)
                if new_cost < best_swap_cost:
                    best_swap_cost, best_swap = new_cost, set()
            else:
                valid_cands = set(cand for cand in G.nodes() if cand not in S_temp and not any(t in S_temp for t in G.neighbors(cand)))
                relevant_cands = [c for c in valid_cands if len((set(G.neighbors(c)) | {c}) & uncovered_temp) > 0]
                
                # Expand search size up to k_max
                for k in range(1, k_max + 1):
                    for combo in combinations(relevant_cands, k):
                        # Ensure the candidate combination maintains mutual independence
                        is_independent = True
                        if k > 1:
                            for c1, c2 in combinations(combo, 2):
                                if c2 in G.neighbors(c1):
                                    is_independent = False
                                    break
                        if not is_independent: 
                            continue
                        
                        combined_covers = set()
                        for c in combo:
                            combined_covers |= set(G.neighbors(c)) | {c}
                            
                        # If the valid combination covers all dropped nodes, evaluate cost
                        if uncovered_temp.issubset(combined_covers):
                            S_new = S_temp | set(combo)
                            _, new_cost, _, _ = calc_initial_solution_cost(S_new, G)
                            if new_cost < best_swap_cost:
                                best_swap_cost = new_cost
                                best_swap = set(combo)

            if best_swap is not None:
                S = S_temp | best_swap
                current_cost = best_swap_cost
                improved = True
                break

    _, final_cost, _, _ = calc_initial_solution_cost(S, G)
    time_elapsed = perf_counter() - start_time
    
    return final_cost, time_elapsed

def classify_environment(filename):
    """Classifies the environment topology based on the instance file name."""
    if "nw10_ew1000" in filename: return "EG (Edge-oriented)"
    if "nw100_ew100" in filename: return "NG (Neutral)"
    if "nw1000_ew10" in filename: return "VG (Node-oriented)"
    return None

def run_batch_sensitivity_analysis(in_dir_path, out_dir_path):
    k_values = [1, 2, 3, 4, 5]
    
    # Initialize results dictionary
    env_results = {
        "EG (Edge-oriented)": {"costs": {k: [] for k in k_values}, "times": {k: [] for k in k_values}},
        "NG (Neutral)": {"costs": {k: [] for k in k_values}, "times": {k: [] for k in k_values}},
        "VG (Node-oriented)": {"costs": {k: [] for k in k_values}, "times": {k: [] for k in k_values}}
    }
    
    # Ensure output directory exists
    os.makedirs(out_dir_path, exist_ok=True)
    
    # Process all files in the target directory
    for filename in os.listdir(in_dir_path):
        if not filename.endswith(".rgg"): 
            continue 
        
        env_name = classify_environment(filename)
        if env_name is None: 
            continue
            
        file_path = os.path.join(in_dir_path, filename)
        print(f"\n[Processing] {filename} -> {env_name}")
        G = read_instance_with_pos(file_path)
        
        for k in k_values:
            cost, t_elapsed = cross_boundary_ls_dynamic_k(G, k_max=k)
            env_results[env_name]["costs"][k].append(cost)
            env_results[env_name]["times"][k].append(t_elapsed)
            print(f"  k={k} | Cost: {cost:.2f}, Time: {t_elapsed:.4f}s")
            
    # Calculate averages and plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Batch Average Sensitivity Analysis of Neighborhood Size ($k$)", fontsize=16, fontweight='bold')
    
    env_keys = ["EG (Edge-oriented)", "NG (Neutral)", "VG (Node-oriented)"]
    
    for idx, env_name in enumerate(env_keys):
        results = env_results[env_name]
        
        # Handle cases where no files matched the environment
        if not results["costs"][1]: 
            axes[idx].set_title(f"{env_name}\n(No data)")
            continue
            
        avg_costs = [sum(results["costs"][k])/len(results["costs"][k]) for k in k_values]
        avg_times = [sum(results["times"][k])/len(results["times"][k]) for k in k_values]
        
        ax1 = axes[idx]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(k_values, avg_costs, 'b-o', linewidth=2, label='Avg Objective Cost')
        ax1.set_xlabel('Maximum Swap Size ($k$)', fontsize=12)
        ax1.set_ylabel('Average Cost', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xticks(k_values)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        line2 = ax2.plot(k_values, avg_times, 'r--^', linewidth=2, label='Avg Time (s)')
        ax2.set_ylabel('Average Time (s) [Log Scale]', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_yscale('log')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center')
        
        num_files = len(results["costs"][1])
        ax1.set_title(f"{env_name}\n(Averaged over {num_files} instances)", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = os.path.join(out_dir_path, 'k_sensitivity_batch_analysis.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Batch average plot saved to: {save_path}")

if __name__ == '__main__':
    # SCRIPT_DIR is assumed to be 'src/'. Navigates to '../results/compare'
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INSTANCE_DIR = os.path.join(SCRIPT_DIR, '..', 'instances', 'random_udg_with_pos')
    RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'compare')
    
    run_batch_sensitivity_analysis(INSTANCE_DIR, RESULTS_DIR)