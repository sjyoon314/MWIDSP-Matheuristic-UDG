from argparse import ArgumentParser
import os
import random
import math
from time import perf_counter
import networkx as nx

random.seed(42)


# ============================================================
# Instance Reader
# ============================================================

def read_instance(file_path):

    G = nx.Graph()

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    n, m = map(int, lines[0].split())

    G.add_nodes_from(range(n))

    idx = 1

    for u in range(n):
        G.nodes[u]['weight'] = float(lines[idx])
        idx += 1

    for _ in range(m):

        u, v, w = map(float, lines[idx].split())

        u = int(u)
        v = int(v)

        G.add_edge(u, v, weight=w)

        idx += 1

    return G


# ============================================================
# Utilities
# ============================================================

def build_weight_tables(G):

    w_node = {
        u: G.nodes[u]['weight']
        for u in G.nodes()
    }

    w_edge = {}

    for u, v in G.edges():

        w = G[u][v]['weight']

        w_edge[(u, v)] = w
        w_edge[(v, u)] = w

    return w_node, w_edge


def is_feasible(S, G):

    # independence
    for u in S:
        for v in G.neighbors(u):
            if v in S:
                return False

    # domination
    for u in G.nodes():

        if u in S:
            continue

        dominated = False

        for v in G.neighbors(u):
            if v in S:
                dominated = True
                break

        if not dominated:
            return False

    return True


def calc_objective(S, G, w_node, w_edge):

    total = 0.0

    # node cost
    for u in S:
        total += w_node[u]

    # assignment edge cost
    for u in G.nodes():

        if u in S:
            continue

        dominators = [
            v for v in G.neighbors(u)
            if v in S
        ]

        if not dominators:
            return float('inf')

        total += min(
            w_edge[(u, v)]
            for v in dominators
        )

    return total


# ============================================================
# Construction Phase
# ============================================================

def greedy_randomized_construction(
    G,
    w_node,
    w_edge,
    alpha=0.3
):

    nodes = list(G.nodes())

    uncovered = set(nodes)

    S = set()

    adj = {
        u: list(G.neighbors(u))
        for u in nodes
    }

    while uncovered:

        candidate_scores = []

        best_score = -float('inf')
        worst_score = float('inf')

        for v in uncovered:

            # independence guaranteed
            if any(u in S for u in adj[v]):
                continue

            newly_covered = set([v])

            for u in adj[v]:
                if u in uncovered:
                    newly_covered.add(u)

            # better marginal estimation
            node_cost = w_node[v]

            edge_gain = 0.0

            for u in newly_covered:

                if u == v:
                    continue

                edge_gain += w_edge[(u, v)]

            estimated_cost = node_cost + edge_gain

            score = (
                len(newly_covered)
                /
                (estimated_cost + 1e-9)
            )

            candidate_scores.append((score, v))

            best_score = max(best_score, score)
            worst_score = min(worst_score, score)

        # Correct GRASP alpha semantics
        threshold = best_score - alpha * (
            best_score - worst_score
        )

        rcl = [
            v
            for score, v in candidate_scores
            if score >= threshold
        ]

        if not rcl:
            rcl = [
                max(candidate_scores)[1]
            ]

        selected = random.choice(rcl)

        S.add(selected)

        uncovered.discard(selected)

        for u in adj[selected]:
            uncovered.discard(u)

    return S


# ============================================================
# Local Search
# ============================================================

def remove_redundant_nodes(S, G):

    improved = True

    while improved:

        improved = False

        for u in list(S):

            candidate = S - {u}

            if is_feasible(candidate, G):
                S = candidate
                improved = True

    return S


def swap_search(
    S,
    G,
    w_node,
    w_edge
):

    current_cost = calc_objective(
        S,
        G,
        w_node,
        w_edge
    )

    improved = True

    nodes = list(G.nodes())

    while improved:

        improved = False

        outside = list(set(nodes) - S)

        random.shuffle(outside)

        for out_node in outside:

            neighbors_in_S = [
                u for u in G.neighbors(out_node)
                if u in S
            ]

            random.shuffle(neighbors_in_S)

            for in_node in neighbors_in_S:

                candidate = set(S)

                candidate.remove(in_node)

                # independence check
                if any(
                    v in candidate
                    for v in G.neighbors(out_node)
                ):
                    continue

                candidate.add(out_node)

                if not is_feasible(candidate, G):
                    continue

                candidate = remove_redundant_nodes(
                    candidate,
                    G
                )

                candidate_cost = calc_objective(
                    candidate,
                    G,
                    w_node,
                    w_edge
                )

                if candidate_cost < current_cost:

                    S = candidate
                    current_cost = candidate_cost

                    improved = True

                    break

            if improved:
                break

    return S, current_cost


# ============================================================
# GRASP Main
# ============================================================

def grasp_mwidsp(
    G,
    time_limit,
    alpha=0.3
):

    start_time = perf_counter()

    w_node, w_edge = build_weight_tables(G)

    best_S = set()

    best_cost = float('inf')

    iterations = 0

    while perf_counter() - start_time < time_limit:

        iterations += 1

        # ----------------------------------------------------
        # Construction
        # ----------------------------------------------------

        S = greedy_randomized_construction(
            G,
            w_node,
            w_edge,
            alpha
        )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        S = remove_redundant_nodes(S, G)

        # ----------------------------------------------------
        # Local Search
        # ----------------------------------------------------

        S, cost = swap_search(
            S,
            G,
            w_node,
            w_edge
        )

        # ----------------------------------------------------
        # Global Best
        # ----------------------------------------------------

        if cost < best_cost:

            best_cost = cost
            best_S = set(S)

    elapsed = perf_counter() - start_time

    return (
        best_S,
        best_cost,
        elapsed,
        iterations
    )


# ============================================================
# Batch Solver
# ============================================================

def solve_all_grasp(
    in_dir_path,
    subset,
    out_dir_path
):

    os.makedirs(out_dir_path, exist_ok=True)

    out_file = os.path.join(
        out_dir_path,
        f"{subset}_grasp_results.csv"
    )

    if not os.path.exists(out_file):

        with open(out_file, 'w') as f:
            f.write(
                "instance,cost,time,iterations\n"
            )

    target_files = [

        f for f in sorted(os.listdir(in_dir_path))

        if f.startswith(f"{subset}_")

    ]

    for file_name in target_files:

        print(f"\n[{file_name}] Processing...")

        path = os.path.join(
            in_dir_path,
            file_name
        )

        G = read_instance(path)

        time_limit = 3.0 * len(G.nodes())

        _, cost, elapsed, iters = grasp_mwidsp(
            G,
            time_limit=time_limit,
            alpha=0.3
        )

        with open(out_file, 'a') as f:

            f.write(
                f"{file_name},"
                f"{cost:.4f},"
                f"{elapsed:.4f},"
                f"{iters}\n"
            )

        print(
            f" -> Cost: {cost:.2f} | "
            f"Time: {elapsed:.2f}s | "
            f"Iters: {iters}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        '-i',
        '--in_dir_path',
        type=str,
        required=True
    )

    parser.add_argument(
        '-s',
        '--instances_subset',
        type=str,
        required=True
    )

    parser.add_argument(
        '-o',
        '--out_dir_path',
        type=str,
        required=True
    )

    args = parser.parse_args()

    solve_all_grasp(
        args.in_dir_path,
        args.instances_subset,
        args.out_dir_path
    )