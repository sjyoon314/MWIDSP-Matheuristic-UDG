"""
Competitive Matheuristic Ensemble for the MWIDSP.
Concurrently executes a geometric multi-start engine (Shift_Full) and 
a continuous relaxation engine (LP_Rounding), returning the Pareto-efficient winner.
Retains multiple scoring strategies for ablation studies.
"""

import os
import math
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
# Phase 1: Node-Weight Scoring (node_only)
# =====================================================================
def greedy_set_cover(candidates, uncovered, G, forbidden_nodes):
    """Greedy selection ensuring both domination and independence."""
    S = set()
    while uncovered:
        best_cand = None
        best_ratio = float('inf')
        best_covers = set()
        
        valid_cands = [c for c in candidates if c not in forbidden_nodes]
        if not valid_cands:
            break
            
        for cand in valid_cands:
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
        
        forbidden_nodes.add(best_cand)
        for u in G.neighbors(best_cand):
            forbidden_nodes.add(u)
            
    return S

def run_phase1_shifted_node_only(G, cell_size, offset_x, offset_y):
    """Initial solution generation focusing solely on node weights."""
    cells = {}
    for v in G.nodes():
        cx, cy = get_shifted_cell(G.nodes[v]['pos'], cell_size, offset_x, offset_y)
        cells.setdefault((cx, cy), []).append(v)
        
    S_init = set()
    global_forbidden = set() 
    
    for cell_nodes in cells.values():
        cell_uncovered = set(cell_nodes)
        S_cell = greedy_set_cover(cell_nodes, cell_uncovered, G, global_forbidden)
        S_init.update(S_cell)
        
    global_uncovered = set(G.nodes())
    for v in S_init:
        global_uncovered.discard(v)
        global_uncovered -= set(G.neighbors(v))
        
    if global_uncovered:
        S_repair = greedy_set_cover(G.nodes(), global_uncovered, G, global_forbidden)
        S_init.update(S_repair)
        
    return S_init

# =====================================================================
# Phase 1: Edge-Aware Scoring (edge_aware_min/sum)
# =====================================================================
def run_phase1_shifted_edge_aware(G, cell_size, offset_x, offset_y, score_type):
    """Initial solution generation incorporating cross-boundary edge penalties."""
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
                    
                    if score_type == 'edge_aware_min':
                        alt_cost = min(alt_edges) if alt_edges else 0
                    elif score_type == 'edge_aware_sum':
                        alt_cost = sum(alt_edges) if alt_edges else 0
                    else:
                        alt_cost = 0
                    
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
# ENGINE 1: SHIFT-FULL (GEOMETRIC MULTI-START)
# =====================================================================
def shift_full_heuristic(G, radius=0.14, score_type='node_only'):
    """Executes the Shift_Full algorithm (Phase 1 + VNS Phase 2) over 4 grid offsets."""
    start_time = perf_counter()
    cell_size = radius / math.sqrt(2)
    
    offsets = [
        (0.0, 0.0), 
        (cell_size / 2, 0.0), 
        (0.0, cell_size / 2), 
        (cell_size / 2, cell_size / 2)
    ]
    
    global_best_S = None
    global_best_cost = float('inf')
    global_best_incorrect = float('inf')
    
    for ox, oy in offsets:
        if score_type in ['edge_aware_min', 'edge_aware_sum']:
            S_cand = run_phase1_shifted_edge_aware(G, cell_size, ox, oy, score_type)
        else:
            S_cand = run_phase1_shifted_node_only(G, cell_size, ox, oy)
            
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
                
                best_swap_criteria = (current_cost, -old_eff, v)
                best_swap = None

                # 1-to-1 swap
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

        num_incorrect, final_cost, _, _ = calc_initial_solution_cost_fast(S, G)
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_S = S.copy()
            global_best_incorrect = num_incorrect

    time_elapsed = perf_counter() - start_time
    return global_best_S, global_best_incorrect, global_best_cost, time_elapsed

# =====================================================================
# ENGINE 2: LP ROUNDING 
# =====================================================================
def lp_rounding_heuristic(G, num_rounding_trials=100):
    """LP Relaxation and Randomized Rounding Matheuristic for Node-heavy environments."""
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
    
    if m.status != GRB.OPTIMAL:
        return set(), 0, float('inf'), perf_counter() - start_time
        
    lp_probs = {v: x[v].X for v in G.nodes()}
    
    best_S = None
    best_cost = float('inf')
    best_incorrect = float('inf')
    
    for _ in range(num_rounding_trials):
        S = set()
        for v in G.nodes():
            if random.random() <= lp_probs[v]:
                S.add(v)
                
        S_list = list(S)
        for u in S_list:
            if u in S:
                for v in list(G.neighbors(u)):
                    if v in S:
                        if G.nodes[u]['weight'] >= G.nodes[v]['weight']:
                            S.remove(u)
                            break
                        else:
                            S.remove(v)
                            
        covered = set()
        for v in S:
            covered.add(v)
            covered.update(G.neighbors(v))
            
        uncovered = set(G.nodes()) - covered
        
        while uncovered:
            best_cand = None
            best_ratio = float('inf')
            
            valid_cands = [v for v in G.nodes() if v not in S and not any(u in S for u in G.neighbors(v))]
            
            if not valid_cands:
                break 
                
            for v in valid_cands:
                covers_new = len((set(G.neighbors(v)) | {v}) & uncovered)
                if covers_new > 0:
                    ratio = G.nodes[v]['weight'] / covers_new
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_cand = v
                        
            if best_cand is not None:
                S.add(best_cand)
                uncovered -= (set(G.neighbors(best_cand)) | {best_cand})
            else:
                break
                
        num_incorrect, cost, _, _ = calc_initial_solution_cost_fast(S, G)
        
        if num_incorrect == 0 and cost < best_cost:
            best_cost = cost
            best_S = S.copy()
            best_incorrect = num_incorrect
            
    if best_S is None:
        best_S = S
        best_cost = cost
        best_incorrect = num_incorrect

    time_elapsed = perf_counter() - start_time
    return best_S, best_incorrect, best_cost, time_elapsed

# =====================================================================
# ENSEMBLE WRAPPER
# =====================================================================
def competitive_ensemble_heuristic(G, score_type='node_only'):
    """Executes both algorithms concurrently and returns the Pareto-efficient winner."""
    total_start_time = perf_counter()
    
    # 1. Execute Geometric Engine
    S_shift, inc_shift, cost_shift, t_shift = shift_full_heuristic(G, score_type=score_type)
    
    # 2. Execute Continuous Relaxation Engine
    S_lp, inc_lp, cost_lp, t_lp = lp_rounding_heuristic(G)
    
    # 3. Determine the optimal configuration
    if inc_lp == 0 and cost_lp < cost_shift:
        best_S = S_lp
        best_inc = inc_lp
        best_cost = cost_lp
        winner = "LP_Rounding"
    else:
        best_S = S_shift
        best_inc = inc_shift
        best_cost = cost_shift
        winner = "Shift_Full"
        
    total_time = perf_counter() - total_start_time
    return best_S, best_inc, best_cost, total_time, winner

def solve_all_ensemble(in_dir_path, instances_subset, out_dir_path, exp_name, score_type):
    """Processes instances and logs the winner between geometric and continuous approaches."""
    if score_type == 'node_only':
        base_name = "grid_heuristic"
    else:
        base_name = f"grid_heuristic_{score_type}"
        
    if exp_name in ["", "fixed", "scaling_test"]:
        out_file_name = f'{instances_subset}_{base_name}_results.csv'
    else:
        out_file_name = f'{instances_subset}_{base_name}_{exp_name}_results.csv'
        
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write("instance,num_incorrect,cost,time,winner\n")
            
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
                score_msg = "Node Only" if score_type == 'node_only' else score_type
                print(f"[{file_name}] Already processed ({score_msg}). Skipping.")
                continue
                
            file_path = os.path.join(in_dir_path, file_name)
            G = read_instance_with_pos(file_path)
            
            _, num_incorrect_nodes, cost, time_elapsed, winner = competitive_ensemble_heuristic(G, score_type=score_type)
            
            with open(out_file_path, 'a') as f:
                f.write(f"{file_name},{num_incorrect_nodes},{cost},{time_elapsed:.4f},{winner}\n")
                
            score_msg = "Node Only" if score_type == 'node_only' else score_type
            print(f"[{file_name}] Score: {score_msg} | Cost: {cost:.4f} | Time: {time_elapsed:.4f}s | Winner: {winner}")

if __name__ == '__main__':
    parser = ArgumentParser(description="Competitive Matheuristic Ensemble (Shift_Full + LP_Rounding)")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Path to input instances")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Instance prefix to process")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Output directory path")
    parser.add_argument('-e', '--exp_name', type=str, required=True, help="Experiment Name")
    parser.add_argument('--score_type', choices=['node_only', 'edge_aware_min', 'edge_aware_sum'], default='node_only', help="Scoring strategy for Phase 1")
    args = parser.parse_args()
    
    solve_all_ensemble(args.in_dir_path, args.instances_subset, args.out_dir_path, args.exp_name, args.score_type)