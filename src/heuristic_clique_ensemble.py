"""
Competitive Matheuristic Ensemble for the MWIDSP.
Achieves Theoretical Consistency by replacing the spatial Grid Multi-Start engine 
with a purely Topological Maximal Clique Constructive Engine.
Logs individual strategy performance to a text file for analysis.
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

# =============================================================================
# ENGINE 1: TOPOLOGICAL CLIQUE HEURISTIC + VNS
# =============================================================================
def run_phase1_clique(G, maximal_cliques, clique_counts, strategy='size'):
    """Phase 1: Topological Decomposition and Cross-Boundary Scoring."""
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
                    min_alt = min(alt_edges) if alt_edges else 0
                    
                    # Topological Cross-Boundary Check
                    if u not in clique:
                        benefit_out += G.nodes[u]['weight'] + min_alt
                    
                    benefit_fallback += G.nodes[u]['weight'] + min_alt
            
            score = benefit_out / cost if cost > 0 else 0
            fallback_score = benefit_fallback / cost if cost > 0 else 0
            
            # THE PRIMAL-DUAL ALIGNMENT MULTIPLIER
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

def clique_full_heuristic(G, maximal_cliques, clique_counts, instance_name, log_file_path):
    """Executes the Clique-Guided algorithm and logs results to a text file."""
    start_time = perf_counter()
    strategies = ['size', 'cost', 'hub', 'inverse_size']
    
    global_best_S = None
    global_best_cost = float('inf')
    global_best_incorrect = float('inf')
    global_best_strategy = None  # ★ 어떤 전략이 이겼는지 추적할 변수 추가
    
    with open(log_file_path, 'a') as log_f:
        log_f.write(f"\n--- Instance: {instance_name} ---\n")
    
    for strategy in strategies:
        S_cand = run_phase1_clique(G, maximal_cliques, clique_counts, strategy)
        _, current_cost, _, _ = calc_initial_solution_cost_fast(S_cand, G)
        
        S = S_cand.copy()
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
                
                valid_cands = set()
                for cand in G.nodes():
                    if cand not in S_temp and not any(t in S_temp for t in G.neighbors(cand)):
                        valid_cands.add(cand)
                
                relevant_cands = [c for c in valid_cands if len((set(G.neighbors(c)) | {c}) & uncovered_temp) > 0]
                
                # 1-to-1 Exchange
                for c1 in relevant_cands:
                    c1_covers = set(G.neighbors(c1)) | {c1}
                    if uncovered_temp.issubset(c1_covers): 
                        S_new = S_temp | {c1}
                        _, new_cost, _, _ = calc_initial_solution_cost_fast(S_new, G)
                        if new_cost < current_cost:
                            best_swap_cost = new_cost
                            best_swap = {c1}
                            break
                            
                # 1-to-2 Exchange
                if best_swap is None: 
                    for c1, c2 in combinations(relevant_cands, 2):
                        if c2 not in G.neighbors(c1): 
                            combined_covers = set(G.neighbors(c1)) | {c1} | set(G.neighbors(c2)) | {c2}
                            if uncovered_temp.issubset(combined_covers):
                                S_new = S_temp | {c1, c2}
                                _, new_cost, _, _ = calc_initial_solution_cost_fast(S_new, G)
                                if new_cost < current_cost:
                                    best_swap_cost = new_cost
                                    best_swap = {c1, c2}
                                    break
                
                if best_swap is not None:
                    S = S_temp | best_swap
                    current_cost = best_swap_cost
                    improved = True
                    break

        num_incorrect, final_cost, _, _ = calc_initial_solution_cost_fast(S, G)
        
        log_msg = f"  - Strategy [{strategy:12s}] -> Cost: {final_cost:.2f}"
        print(log_msg)
        with open(log_file_path, 'a') as log_f:
            log_f.write(log_msg + "\n")
        
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_S = S.copy()
            global_best_incorrect = num_incorrect
            global_best_strategy = strategy  # ★ 최고 점수를 낸 전략 저장

    time_elapsed = perf_counter() - start_time
    # ★ 반환값 맨 끝에 승리 전략 이름 추가
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
def competitive_ensemble_heuristic(G, instance_name, log_file_path):
    total_start_time = perf_counter()
    
    maximal_cliques = list(nx.find_cliques(G))
    clique_counts = {u: 0 for u in G.nodes()}
    for clq in maximal_cliques:
        for u in clq: clique_counts[u] += 1
    
    # ★ 반환받을 때 맨 끝의 clq_strat (클리크 승리 전략) 변수 추가
    S_clq, inc_clq, cost_clq, _, clq_strat = clique_full_heuristic(G, maximal_cliques, clique_counts, instance_name, log_file_path)
    
    S_lp, inc_lp, cost_lp, _ = lp_rounding_heuristic(G)
    
    if inc_lp == 0 and cost_lp < cost_clq:
        best_S, best_inc, best_cost, winner = S_lp, inc_lp, cost_lp, "LP_Rounding"
    else:
        # ★ 퉁치지 않고 'Clique_hub', 'Clique_cost' 등으로 명확히 출력
        best_S, best_inc, best_cost, winner = S_clq, inc_clq, cost_clq, f"Clique_{clq_strat.capitalize()}"
        
    total_time = perf_counter() - total_start_time
    return best_S, best_inc, best_cost, total_time, winner

def solve_all_ensemble(in_dir_path, instances_subset, out_dir_path, exp_name):
    if exp_name in ["", "fixed", "scaling_test"]:
        out_file_name = f'{instances_subset}_clique_heuristic_results.csv'
    else:
        out_file_name = f'{instances_subset}_clique_heuristic_{exp_name}_results.csv'
    out_file_path = os.path.join(out_dir_path, out_file_name)
    
    # 전략별 로그를 저장할 텍스트 파일 경로
    log_file_path = os.path.join(out_dir_path, f"{exp_name}_{instances_subset}_strategy_logs.txt")
    
    os.makedirs(out_dir_path, exist_ok=True)
    
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write("instance,num_incorrect,cost,time,winner\n")
            
    processed = set()
    if os.path.exists(out_file_path):
        with open(out_file_path, 'r') as f:
            for line in f.readlines()[1:]:
                if line.strip(): processed.add(line.split(',')[0].strip())
    
    # 텍스트 로그 파일 초기화
    with open(log_file_path, 'a') as log_f:
        log_f.write("\n========================================\n")
        log_f.write(f"NEW RUN STARTED: {exp_name} - {instances_subset}\n")
        log_f.write("========================================\n")
    
    for file_name in sorted(os.listdir(in_dir_path)):
        if file_name.startswith(f"{instances_subset}_"):
            if file_name in processed: continue
                
            file_path = os.path.join(in_dir_path, file_name)
            G = read_instance_with_pos(file_path)
            
            print(f"\n[{file_name}] Processing strategies...")
            _, num_incorrect_nodes, cost, time_elapsed, winner = competitive_ensemble_heuristic(G, file_name, log_file_path)
            
            with open(out_file_path, 'a') as f:
                f.write(f"{file_name},{num_incorrect_nodes},{cost:.4f},{time_elapsed:.4f},{winner}\n")
                
            # ★ 터미널 출력 및 텍스트 파일(txt) 맨 아래에 최종 승자 기록
            print(f"  => Ensemble Finished | Cost: {cost:.4f} | Time: {time_elapsed:.4f}s | Winner: {winner}")
            with open(log_file_path, 'a') as log_f:
                log_f.write(f"  => FINAL WINNER: {winner} (Cost: {cost:.2f})\n")

if __name__ == '__main__':
    parser = ArgumentParser(description="Competitive Matheuristic Ensemble (Clique_Full + LP_Rounding)")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Path to input instances")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Instance prefix to process")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Output directory path")
    parser.add_argument('-e', '--exp_name', type=str, required=True, help="Experiment Name (e.g., fixed_radius or constant_density)")
    args = parser.parse_args()
    
    solve_all_ensemble(args.in_dir_path, args.instances_subset, args.out_dir_path, args.exp_name)