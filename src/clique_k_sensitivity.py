"""
Sensitivity Analysis for Neighborhood Size (k) in Clique-based GB-VNS.
Evaluates the trade-off between optimization cost and computation time
across different maximum swap sizes (k=1 to 5) using Topological Cliques.
"""

import os
from time import perf_counter
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx

from utils import calc_initial_solution_cost_fast

def read_instance(file_path):
    """Reads the instance file."""
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
        u, v, w = map(float, lines[idx].split()[:3])
        G.add_edge(int(u), int(v), weight=w)
        idx += 1
            
    return G

def clique_ls_dynamic_k(G, k_max, strategy='size'):
    """Executes the Clique heuristic with a dynamically bounded swap size (k_max)."""
    start_time = perf_counter()
    
    # 1. Extract Maximal Cliques
    maximal_cliques = list(nx.find_cliques(G))
    clique_counts = {u: 0 for u in G.nodes()}
    for clq in maximal_cliques:
        for u in clq: 
            clique_counts[u] += 1
            
    # 2. Sort Cliques based on strategy (Baseline construction)
    if strategy == 'size':
        sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)
    elif strategy == 'hub':
        sorted_cliques = sorted(maximal_cliques, key=lambda c: sum(clique_counts[v] for v in c), reverse=True)
    else:
        sorted_cliques = sorted(maximal_cliques, key=len, reverse=True) # Default fallback
        
    S = set()
    uncovered = set(G.nodes())
    forbidden_nodes = set()
    
    # Phase 1: Topological Clique Construction
    for clique in sorted_cliques:
        candidates = [v for v in clique if v not in forbidden_nodes]
        if not candidates: continue
            
        best_node, best_score = None, -1.0
        
        for v in candidates:
            cost = G.nodes[v]['weight']
            benefit_out = 0
            
            for u in G.neighbors(v):
                if u in uncovered:
                    cost += G[v][u]['weight']
                    alt_edges = [G[u][t]['weight'] for t in G.neighbors(u) if t != v and t in uncovered]
                    min_alt = min(alt_edges) if alt_edges else 0
                    
                    if u not in clique:
                        benefit_out += G.nodes[u]['weight'] + min_alt
            
            score = benefit_out / cost if cost > 0 else 0
            # Hub modifier
            score *= (1.0 + 0.05 * clique_counts[v])
            
            if score > best_score:
                best_score = score
                best_node = v
                
        if best_score > 0 and best_node is not None:
            S.add(best_node)
            uncovered.discard(best_node)
            forbidden_nodes.add(best_node)
            for u in G.neighbors(best_node):
                uncovered.discard(u)
                forbidden_nodes.add(u)

    # Greedy repair for remaining nodes
    for v in list(uncovered):
        if v not in forbidden_nodes:
            S.add(v)
            uncovered.discard(v)
            forbidden_nodes.add(v)
            for u in G.neighbors(v):
                uncovered.discard(u)
                forbidden_nodes.add(u)

    _, current_cost, _, _ = calc_initial_solution_cost_fast(S, G)
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
                _, new_cost, _, _ = calc_initial_solution_cost_fast(S_temp, G)
                if new_cost < best_swap_cost:
                    best_swap_cost = new_cost
                    best_swap = set()
            else:
                valid_cands = set(cand for cand in G.nodes() if cand not in S_temp and not any(t in S_temp for t in G.neighbors(cand)))
                relevant_cands = [c for c in valid_cands if len((set(G.neighbors(c)) | {c}) & uncovered_temp) > 0]
                
                # Expand search size up to k_max
                for k in range(1, k_max + 1):
                    for combo in combinations(relevant_cands, k):
                        # Ensure independence within the combination
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
                            
                        # If combination covers dropped nodes
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

def classify_environment(filename):
    if "nw10_ew1000" in filename: return "EG (Edge-oriented)"
    if "nw100_ew100" in filename: return "NG (Neutral)"
    if "nw1000_ew10" in filename: return "VG (Node-oriented)"
    return None

def run_clique_batch_sensitivity(in_dir_path, out_dir_path):
    k_values = [1, 2, 3, 4, 5]
    
    env_results = {
        "EG (Edge-oriented)": {"costs": {k: [] for k in k_values}, "times": {k: [] for k in k_values}},
        "NG (Neutral)": {"costs": {k: [] for k in k_values}, "times": {k: [] for k in k_values}},
        "VG (Node-oriented)": {"costs": {k: [] for k in k_values}, "times": {k: [] for k in k_values}}
    }
    
    os.makedirs(out_dir_path, exist_ok=True)
    
    for filename in os.listdir(in_dir_path):
        # Position is not strictly needed for clique logic, but we read standard files
        if not filename.endswith(".rgg"): 
            continue 
        
        env_name = classify_environment(filename)
        if env_name is None: 
            continue
            
        file_path = os.path.join(in_dir_path, filename)
        print(f"\n[Processing] {filename} -> {env_name}")
        G = read_instance(file_path) # Clique doesn't need pos
        
        for k in k_values:
            # Fixing strategy to 'size' for consistent baseline VNS evaluation
            cost, t_elapsed = clique_ls_dynamic_k(G, k_max=k, strategy='size')
            env_results[env_name]["costs"][k].append(cost)
            env_results[env_name]["times"][k].append(t_elapsed)
            print(f"  k={k} | Cost: {cost:.2f}, Time: {t_elapsed:.4f}s")
            
    # Calculate averages and plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Clique VNS: Batch Average Sensitivity Analysis of Neighborhood Size ($k$)", fontsize=16, fontweight='bold')
    
    env_keys = ["EG (Edge-oriented)", "NG (Neutral)", "VG (Node-oriented)"]
    
    for idx, env_name in enumerate(env_keys):
        results = env_results[env_name]
        
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
    
    save_path = os.path.join(out_dir_path, 'clique_k_sensitivity_batch_analysis.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Clique Batch average plot saved to: {save_path}")

if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Modify paths as per your local structure
    INSTANCE_DIR = os.path.join(SCRIPT_DIR, '..', 'instances', 'random_udg_with_pos')
    RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'compare')
    
    run_clique_batch_sensitivity(INSTANCE_DIR, RESULTS_DIR)