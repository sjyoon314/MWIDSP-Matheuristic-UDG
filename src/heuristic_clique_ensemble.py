"""
Competitive Matheuristic Ensemble for the MWIDSP.
Achieves theoretical consistency by replacing the spatial Grid Multi-Start engine 
with a purely topological Maximal Clique Constructive Engine.
Logs individual strategy performance and utilizes efficiency-aware tie-breaking.
"""
import os
import random
random.seed(42)
from time import perf_counter
from itertools import combinations
from argparse import ArgumentParser
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from utils import calc_initial_solution_cost_fast

def read_instance_with_pos(file_path):
    """Reads the instance file, focusing on topology and weights."""
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

# =====================================================================
# Phase 1: Node-Only Scoring Strategy
# =====================================================================
def run_phase1_clique_node_only(G, maximal_cliques, clique_counts, strategy='size'):
    """Phase 1 using pure Node-Only topological scoring (Aligned with Eq: argmin w_c / |N[c] ∩ U|)."""
    if strategy == 'size':
        sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)
    elif strategy == 'cost':
        sorted_cliques = sorted(maximal_cliques, key=lambda c: sum(G.nodes[v]['weight'] for v in c)/len(c))
    elif strategy == 'hub':
        sorted_cliques = sorted(maximal_cliques, key=lambda c: sum(clique_counts[v] for v in c), reverse=True)
    elif strategy == 'inverse_size':
        sorted_cliques = sorted(maximal_cliques, key=len)
        
    S = set()
    uncovered = set(G.nodes())
    forbidden_nodes = set()
    
    for clique in sorted_cliques:
        candidates = [v for v in clique if v not in forbidden_nodes]
        if not candidates:
            continue
            
        best_node = None
        best_ratio = float('inf')
        
        for cand in candidates:
            # Calculate structurally covered nodes that are currently uncovered
            covers_new = (set(G.neighbors(cand)) | {cand}) & uncovered
            if not covers_new:
                continue
                
            # Pure cost-to-benefit ratio
            ratio = G.nodes[cand]['weight'] / len(covers_new)
            
            if ratio < best_ratio:
                best_ratio = ratio
                best_node = cand
                
        if best_node is not None:
            S.add(best_node)
            uncovered -= (set(G.neighbors(best_node)) | {best_node})
            # Enforce strict independence (only the dominator itself is forbidden)
            forbidden_nodes.add(best_node)

    # Greedy repair for blind spots
    for v in list(uncovered):
        if v not in forbidden_nodes:
            S.add(v)
            uncovered -= (set(G.neighbors(v)) | {v})
            forbidden_nodes.add(v)
                
    return S

# =====================================================================
# Phase 1: Edge-Aware Scoring Strategy
# =====================================================================
def run_phase1_clique_edge_aware(G, maximal_cliques, clique_counts, score_type, strategy='size'):
    """Phase 1: Topological Decomposition with Edge-Aware cross-boundary penalties."""
    if strategy == 'size':
        sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)
    elif strategy == 'cost':
        sorted_cliques = sorted(maximal_cliques, key=lambda c: sum(G.nodes[v]['weight'] for v in c)/len(c))
    elif strategy == 'hub':
        sorted_cliques = sorted(maximal_cliques, key=lambda c: sum(clique_counts[v] for v in c), reverse=True)
    elif strategy == 'inverse_size':
        sorted_cliques = sorted(maximal_cliques, key=len)
        
    S = set()
    uncovered = set(G.nodes())
    forbidden_nodes = set()
    
    for clique in sorted_cliques:
        candidates = [v for v in clique if v not in forbidden_nodes]
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
                    
                    alt_edges = [G[u][t]['weight'] for t in G.neighbors(u) if t != v and t in uncovered]
                    if score_type == 'edge_aware_min':
                        alt_cost = min(alt_edges) if alt_edges else 0
                    elif score_type == 'edge_aware_sum':
                        alt_cost = sum(alt_edges) if alt_edges else 0
                    else:
                        alt_cost = 0
                    
                    if u not in clique:
                        benefit_out += G.nodes[u]['weight'] + alt_cost
                    
                    benefit_fallback += G.nodes[u]['weight'] + alt_cost
            
            score = benefit_out / cost if cost > 0 else 0
            fallback_score = benefit_fallback / cost if cost > 0 else 0
            
            hub_modifier = 1.0 + (0.05 * clique_counts[v])
            score *= hub_modifier
            fallback_score *= hub_modifier
            
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
                #forbidden_nodes.add(u)

    for v in list(uncovered):
        if v not in forbidden_nodes:
            S.add(v)
            uncovered.discard(v)
            forbidden_nodes.add(v)
            for u in G.neighbors(v):
                uncovered.discard(u)
                forbidden_nodes.add(u)
                
    return S

# =============================================================================
# ENGINE 1: TOPOLOGICAL CLIQUE HEURISTIC + VNS
# =============================================================================
def clique_full_heuristic(G, maximal_cliques, clique_counts, instance_name, log_file_path, score_type='node_only'):
    """Executes the Clique-Guided algorithm and logs results. Employs efficiency-aware VNS."""
    start_time = perf_counter()
    strategies = ['size', 'cost', 'hub', 'inverse_size']
    
    global_best_S = None
    global_best_cost = float('inf')
    global_best_incorrect = float('inf')
    global_best_strategy = None  
    
    with open(log_file_path, 'a') as log_f:
        log_f.write(f"\n--- Instance: {instance_name} (Score: {score_type}) ---\n")
    
    for strategy in strategies:
        if score_type in ['edge_aware_min', 'edge_aware_sum']:
            S_cand = run_phase1_clique_edge_aware(G, maximal_cliques, clique_counts, score_type, strategy)
        else:
            S_cand = run_phase1_clique_node_only(G, maximal_cliques, clique_counts, strategy)
            
        _, current_cost, _, _ = calc_initial_solution_cost_fast(S_cand, G)
        
        S = S_cand.copy()
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
                
                # Tie-breaker criteria: (Cost, -Efficiency, Deterministic_ID)
                best_swap_criteria = (current_cost, -old_eff, v)
                best_swap = None
                
                # Evaluate 1-to-1 swaps
                for c1 in relevant_cands:
                    c1_covers = set(G.neighbors(c1)) | {c1}
                    if uncovered_temp.issubset(c1_covers): 
                        S_new = S_temp | {c1}
                        _, new_cost, _, _ = calc_initial_solution_cost_fast(S_new, G)
                        
                        efficiency = len(c1_covers)
                        current_criteria = (new_cost, -efficiency, c1)
                        
                        if current_criteria < best_swap_criteria:
                            best_swap_criteria = current_criteria
                            best_swap = {c1}
                            
                # Evaluate 1-to-2 swaps
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
                
                # Apply best swap found
                if best_swap is not None:
                    S = S_temp | best_swap
                    current_cost = best_swap_criteria[0]
                    improved = True

        num_incorrect, final_cost, _, _ = calc_initial_solution_cost_fast(S, G)
        
        log_msg = f"  - Strategy [{strategy:12s}] -> Cost: {final_cost:.2f}"
        print(log_msg)
        with open(log_file_path, 'a') as log_f:
            log_f.write(log_msg + "\n")
        
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_S = S.copy()
            global_best_incorrect = num_incorrect
            global_best_strategy = strategy 

    time_elapsed = perf_counter() - start_time
    return global_best_S, global_best_incorrect, global_best_cost, time_elapsed, global_best_strategy

# =============================================================================
# ENGINE 2: LP ROUNDING 
# =============================================================================
def lp_rounding_heuristic(G, num_rounding_trials=100):
    start_time = perf_counter()
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model("MWIDSP_Relaxation", env=env)
    
    x = m.addVars(G.nodes(), vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x")
    m.setObjective(gp.quicksum(G.nodes[v]['weight'] * x[v] for v in G.nodes()), GRB.MINIMIZE)
    
    for v in G.nodes():
        m.addConstr(x[v] + gp.quicksum(x[u] for u in G.neighbors(v)) >= 1.0)
    for u, v in G.edges():
        m.addConstr(x[u] + x[v] <= 1.0)
        
    m.optimize()
    if m.status != GRB.OPTIMAL: return set(), 0, float('inf'), perf_counter() - start_time
        
    lp_probs = {v: x[v].X for v in G.nodes()}
    best_S, best_cost, best_incorrect = None, float('inf'), float('inf')
    
    for _ in range(num_rounding_trials):
        S = set()
        for v in G.nodes():
            if random.random() <= lp_probs[v]: S.add(v)
                
        S_list = list(S)
        for u in S_list:
            if u in S:
                for v in list(G.neighbors(u)):
                    if v in S:
                        if G.nodes[u]['weight'] >= G.nodes[v]['weight']: S.remove(u); break
                        else: S.remove(v)
                            
        covered = set(v for u in S for v in G.neighbors(u)) | S
        uncovered = set(G.nodes()) - covered
        
        while uncovered:
            best_cand, best_ratio = None, float('inf')
            valid_cands = [v for v in G.nodes() if v not in S and not any(u in S for u in G.neighbors(v))]
            if not valid_cands: break 
                
            for v in valid_cands:
                covers_new = len((set(G.neighbors(v)) | {v}) & uncovered)
                if covers_new > 0:
                    ratio = G.nodes[v]['weight'] / covers_new
                    if ratio < best_ratio:
                        best_ratio = ratio; best_cand = v
                        
            if best_cand is not None:
                S.add(best_cand)
                uncovered -= (set(G.neighbors(best_cand)) | {best_cand})
            else: break
                
        num_inc, cost, _, _ = calc_initial_solution_cost_fast(S, G)
        if num_inc == 0 and cost < best_cost:
            best_cost, best_S, best_incorrect = cost, S.copy(), num_inc
            
    if best_S is None: best_S, best_cost, best_incorrect = S, cost, num_inc
    return best_S, best_incorrect, best_cost, perf_counter() - start_time

# =============================================================================
# COMPETITIVE ENSEMBLE WRAPPER
# =============================================================================
def competitive_ensemble_heuristic(G, instance_name, log_file_path, score_type='node_only'):
    total_start_time = perf_counter()
    
    maximal_cliques = list(nx.find_cliques(G))
    clique_counts = {u: 0 for u in G.nodes()}
    for clq in maximal_cliques:
        for u in clq: clique_counts[u] += 1
    
    S_clq, inc_clq, cost_clq, _, clq_strat = clique_full_heuristic(G, maximal_cliques, clique_counts, instance_name, log_file_path, score_type=score_type)
    
    S_lp, inc_lp, cost_lp, _ = lp_rounding_heuristic(G)
    
    if inc_lp == 0 and cost_lp < cost_clq:
        best_S, best_inc, best_cost, winner = S_lp, inc_lp, cost_lp, "LP_Rounding"
    else:
        best_S, best_inc, best_cost, winner = S_clq, inc_clq, cost_clq, f"Clique_{clq_strat.capitalize()}"
        
    total_time = perf_counter() - total_start_time
    return best_S, best_inc, best_cost, total_time, winner

def solve_all_ensemble(in_dir_path, instances_subset, out_dir_path, exp_name, score_type):
    if score_type == 'node_only':
        base_name = "clique_heuristic"
    else:
        base_name = f"clique_heuristic_{score_type}"
        
    if exp_name in ["", "fixed", "scaling_test"]:
        out_file_name = f'{instances_subset}_{base_name}_results.csv'
    else:
        out_file_name = f'{instances_subset}_{base_name}_{exp_name}_results.csv'
        
    out_file_path = os.path.join(out_dir_path, out_file_name)
    
    log_file_name = f"{exp_name}_{instances_subset}_{base_name}_strategy_logs.txt"
    log_file_path = os.path.join(out_dir_path, log_file_name)
    
    os.makedirs(out_dir_path, exist_ok=True)
    
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write("instance,num_incorrect,cost,time,winner\n")
            
    processed = set()
    if os.path.exists(out_file_path):
        with open(out_file_path, 'r') as f:
            for line in f.readlines()[1:]:
                if line.strip(): processed.add(line.split(',')[0].strip())
    
    with open(log_file_path, 'a') as log_f:
        log_f.write("\n========================================\n")
        log_score_str = "Node Only (Default)" if score_type == 'node_only' else score_type
        log_f.write(f"NEW RUN STARTED: {exp_name} - {instances_subset} (Score: {log_score_str})\n")
        log_f.write("========================================\n")
    
    for file_name in sorted(os.listdir(in_dir_path)):
        if file_name.startswith(f"{instances_subset}_"):
            if file_name in processed: 
                score_msg = "Node Only" if score_type == 'node_only' else score_type
                print(f"[{file_name}] Already processed ({score_msg}). Skipping.")
                continue
                
            file_path = os.path.join(in_dir_path, file_name)
            G = read_instance_with_pos(file_path)
            
            print(f"\n[{file_name}] Processing strategies...")
            _, num_incorrect_nodes, cost, time_elapsed, winner = competitive_ensemble_heuristic(G, file_name, log_file_path, score_type=score_type)
            
            with open(out_file_path, 'a') as f:
                f.write(f"{file_name},{num_incorrect_nodes},{cost:.4f},{time_elapsed:.4f},{winner}\n")
                
            score_msg = "Node Only" if score_type == 'node_only' else score_type
            print(f"  => Ensemble Finished | Score: {score_msg} | Cost: {cost:.4f} | Time: {time_elapsed:.4f}s | Winner: {winner}")
            with open(log_file_path, 'a') as log_f:
                log_f.write(f"  => FINAL WINNER: {winner} (Cost: {cost:.2f})\n")

if __name__ == '__main__':
    parser = ArgumentParser(description="Competitive Matheuristic Ensemble (Clique_Full + LP_Rounding)")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Path to input instances")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Instance prefix to process")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Output directory path")
    parser.add_argument('-e', '--exp_name', type=str, required=True, help="Experiment Name")
    parser.add_argument('--score_type', choices=['node_only', 'edge_aware_min', 'edge_aware_sum'], default='node_only', help="Scoring strategy")
    args = parser.parse_args()
    
    solve_all_ensemble(args.in_dir_path, args.instances_subset, args.out_dir_path, args.exp_name, args.score_type)