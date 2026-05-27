"""
Advanced Lexicographic Simulated Annealing for MWIDSP.
Addresses landscape ruggedness through Lexicographic Acceptance (Violations -> Cost),
expanded neighborhood operators (Add, Remove, Swap), and localized exact delta evaluation.
"""

from argparse import ArgumentParser
import os
import random
import math
from time import perf_counter
import networkx as nx
import datetime

random.seed(42)

def read_instance(file_path):
    """Reads the instance file and builds the graph with node/edge weights."""
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
        G.add_edge(int(u), int(v), weight=w)
        idx += 1
            
    return G

def get_initial_mis(G, nodes, w_node):
    """Generates a guaranteed feasible Maximal Independent Set (MIS) greedily."""
    S = set()
    uncovered = set(nodes)
    sorted_nodes = sorted(nodes, key=lambda x: w_node[x])
    
    for v in sorted_nodes:
        if v in uncovered:
            S.add(v)
            uncovered.remove(v)
            for u in G.neighbors(v):
                if u in uncovered:
                    uncovered.remove(u)
    return S

def simulated_annealing(G):
    start_time = perf_counter()
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    time_limit = 3.0 * num_nodes
    
    adj = {u: list(G.neighbors(u)) for u in nodes}
    w_node = {u: G.nodes[u]['weight'] for u in nodes}
    w_edge = {}
    for u, v in G.edges():
        w = G[u][v]['weight']
        w_edge[(u, v)] = w
        w_edge[(v, u)] = w

    # Local metrics calculation mapped to strict NEW-2 objective
    def calc_metrics(x, in_S_x, S_neigh_x):
        """Returns (violations, cost) for a single node x."""
        if in_S_x:
            # Independence violation: 1 for each adjacent S-node
            # Cost: weight of the node itself
            return len(S_neigh_x), w_node[x]
        else:
            # Domination violation: 1 if no S-neighbors
            # Cost: minimum edge weight to an S-neighbor (if dominated)
            if not S_neigh_x:
                return 1, 0.0 
            else:
                return 0, min(w_edge[(x, y)] for y in S_neigh_x)

    # --- Initialize State with Feasible MIS ---
    initial_S_set = get_initial_mis(G, nodes, w_node)
    
    in_S = [False] * num_nodes
    S_nodes = []
    S_nodes_pos = {}
    out_S_nodes = []
    out_S_nodes_pos = {}
    
    for u in nodes:
        if u in initial_S_set:
            in_S[u] = True
            S_nodes_pos[u] = len(S_nodes)
            S_nodes.append(u)
        else:
            in_S[u] = False
            out_S_nodes_pos[u] = len(out_S_nodes)
            out_S_nodes.append(u)
            
    S_neighbors = [set() for _ in range(num_nodes)]
    for u in S_nodes:
        for v in adj[u]:
            S_neighbors[v].add(u)
            
    # Calculate initial global metrics
    global_v = 0
    global_c = 0.0
    for u in nodes:
        v, c = calc_metrics(u, in_S[u], S_neighbors[u])
        global_v += v
        global_c += c
        
    best_cost = global_c
    best_S = set(S_nodes)
    
    # Temperature config
    max_w = max(w_node.values()) if w_node else 1.0
    T_0 = max_w * 2.0 
    T_final = 0.01
    
    print(f"  [Info] Advanced SA started. Time limit: {time_limit:.1f}s | Initial Cost: {best_cost:.2f}")
    
    def move_node(v, src_list, src_pos, dst_list, dst_pos):
        idx = src_pos[v]
        last_val = src_list[-1]
        src_list[idx] = last_val
        src_pos[last_val] = idx
        src_list.pop()
        del src_pos[v]
        dst_pos[v] = len(dst_list)
        dst_list.append(v)

    iterations = 0
    while True:
        elapsed_time = perf_counter() - start_time
        if elapsed_time >= time_limit:
            break
            
        iterations += 1
        progress = elapsed_time / time_limit
        T = T_0 * math.pow(T_final / T_0, progress)
        
        # --- Expanded Move Operators: Add, Remove, Swap ---
        targets = []
        if not S_nodes:
            targets.append(random.choice(out_S_nodes))
        elif not out_S_nodes:
            targets.append(random.choice(S_nodes))
        else:
            op = random.random()
            if op < 0.33:   # Remove
                targets.append(random.choice(S_nodes))
            elif op < 0.66: # Add
                targets.append(random.choice(out_S_nodes))
            else:           # Swap (Removes 1, Adds 1)
                targets.append(random.choice(S_nodes))
                targets.append(random.choice(out_S_nodes))
                
        # --- Ultra-Fast Overlapping Delta Evaluation ---
        # 1. Identify all nodes whose local metrics will change
        affected_nodes = set(targets)
        for t in targets:
            affected_nodes.update(adj[t])
            
        # 2. Calculate old metrics for the affected subset
        old_local_v, old_local_c = 0, 0.0
        for n in affected_nodes:
            v, c = calc_metrics(n, in_S[n], S_neighbors[n])
            old_local_v += v; old_local_c += c
            
        # 3. Temporarily apply state changes
        for t in targets:
            in_S[t] = not in_S[t]
            if in_S[t]:
                for u in adj[t]: S_neighbors[u].add(t)
            else:
                for u in adj[t]: S_neighbors[u].remove(t)
                
        # 4. Calculate new metrics for the affected subset
        new_local_v, new_local_c = 0, 0.0
        for n in affected_nodes:
            v, c = calc_metrics(n, in_S[n], S_neighbors[n])
            new_local_v += v; new_local_c += c
            
        delta_v = new_local_v - old_local_v
        delta_c = new_local_c - old_local_c
        
        # --- Lexicographic Acceptance Rule ---
        accept = False
        if delta_v < 0:
            accept = True # Always accept moves that reduce violations
        elif delta_v > 0:
            # Heavily penalize increasing violations, but allow small jumps to escape deep traps
            accept = random.random() < math.exp(-(delta_v * max_w) / T)
        else:
            # Violations are identical (delta_v == 0), compare true cost
            if delta_c < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta_c / T)
                
        if accept:
            global_v += delta_v
            global_c += delta_c
            
            # Commit tracking lists
            for t in targets:
                if in_S[t]:
                    move_node(t, out_S_nodes, out_S_nodes_pos, S_nodes, S_nodes_pos)
                else:
                    move_node(t, S_nodes, S_nodes_pos, out_S_nodes, out_S_nodes_pos)
                    
            # Global Best Tracking (Only strictly feasible solutions)
            if global_v == 0 and global_c < best_cost:
                best_cost = global_c
                best_S = set(S_nodes)
        else:
            # Reject: Revert temporary state changes
            for t in reversed(targets):
                in_S[t] = not in_S[t]
                if in_S[t]:
                    for u in adj[t]: S_neighbors[u].add(t)
                else:
                    for u in adj[t]: S_neighbors[u].remove(t)
            
    time_elapsed = perf_counter() - start_time
    num_incorrect = global_v # Return final state's violations if no feasible best was found
    
    return best_S, num_incorrect, best_cost, time_elapsed, iterations

def solve_all_sa(in_dir_path, instances_subset, out_dir_path):
    out_file_name = f"{instances_subset}_sa_results.csv"
    out_file_path = os.path.join(out_dir_path, out_file_name)
    os.makedirs(out_dir_path, exist_ok=True)
    
    if not os.path.exists(out_file_path) or os.path.getsize(out_file_path) == 0:
        with open(out_file_path, 'w') as f:
            f.write("instance,num_incorrect,cost,time,iterations\n")
            
    processed = set()
    if os.path.exists(out_file_path):
        with open(out_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]: 
                if line.strip():
                    processed.add(line.split(',')[0].strip())
    
    target_files = [f for f in sorted(os.listdir(in_dir_path)) 
                    if f.startswith(f"{instances_subset}_") and f not in processed]
    num_remaining = len(target_files)
    
    if num_remaining > 0:
        print(f"\n=====================================================")
        print(f" Instances remaining in queue : {num_remaining}")
        print(f"=====================================================\n")
    else:
        print("All target instances have already been processed.")
        return 

    for file_name in target_files:
        file_path = os.path.join(in_dir_path, file_name)
        
        G = read_instance(file_path)
        _, num_incorrect, cost, time_elapsed, iters = simulated_annealing(G)
        
        with open(out_file_path, 'a') as f:
            f.write(f"{file_name},{num_incorrect},{cost:.4f},{time_elapsed:.4f},{iters}\n")
            
        print(f"[{file_name}] Saved | Obj: {cost:.2f} | Infeasible: {num_incorrect} | Time: {time_elapsed:.2f}s | Iters: {iters}")

if __name__ == '__main__':
    parser = ArgumentParser(description="Advanced Lexicographic SA Metaheuristic")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True)
    parser.add_argument('-s', '--instances_subset', type=str, required=True)
    parser.add_argument('-o', '--out_dir_path', type=str, required=True)
    args = parser.parse_args()
    
    solve_all_sa(args.in_dir_path, args.instances_subset, args.out_dir_path)