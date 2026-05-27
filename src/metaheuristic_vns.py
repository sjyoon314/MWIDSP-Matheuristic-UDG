from argparse import ArgumentParser
import os
import random
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
# Utility
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

    for u in S:
        total += w_node[u]

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
# GRASP Initial Solution
# ============================================================

def construct_initial_solution(
    G,
    w_node,
    w_edge,
    alpha=0.3
):

    uncovered = set(G.nodes())

    S = set()

    while uncovered:

        scores = []

        best = -float('inf')
        worst = float('inf')

        for v in uncovered:

            if any(u in S for u in G.neighbors(v)):
                continue

            newly_covered = {v}

            for u in G.neighbors(v):
                if u in uncovered:
                    newly_covered.add(u)

            est_cost = (
                w_node[v]
                +
                sum(
                    w_edge[(u, v)]
                    for u in newly_covered
                    if u != v
                )
            )

            score = len(newly_covered) / (est_cost + 1e-9)

            scores.append((score, v))

            best = max(best, score)
            worst = min(worst, score)

        threshold = best - alpha * (best - worst)

        rcl = [
            v
            for score, v in scores
            if score >= threshold
        ]

        selected = random.choice(rcl)

        S.add(selected)

        uncovered.discard(selected)

        for u in G.neighbors(selected):
            uncovered.discard(u)

    return S


# ============================================================
# Local Search Neighborhoods
# ============================================================

def remove_redundant(S, G):

    improved = True

    while improved:

        improved = False

        for u in list(S):

            candidate = S - {u}

            if is_feasible(candidate, G):

                S = candidate
                improved = True

    return S


def one_swap(
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

    nodes = list(G.nodes())

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

            if any(
                v in candidate
                for v in G.neighbors(out_node)
            ):
                continue

            candidate.add(out_node)

            if not is_feasible(candidate, G):
                continue

            candidate = remove_redundant(
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
                return candidate, candidate_cost

    return S, current_cost


def two_swap(
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

    outside = list(set(G.nodes()) - S)

    random.shuffle(outside)

    for a in outside:

        for b in outside:

            if a >= b:
                continue

            if G.has_edge(a, b):
                continue

            candidate = set(S)

            # remove conflicting nodes
            for u in list(candidate):

                if (
                    G.has_edge(u, a)
                    or
                    G.has_edge(u, b)
                ):
                    candidate.remove(u)

            candidate.add(a)
            candidate.add(b)

            if not is_feasible(candidate, G):
                continue

            candidate = remove_redundant(
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
                return candidate, candidate_cost

    return S, current_cost


# ============================================================
# Destroy / Repair
# ============================================================

def destroy_repair(
    S,
    G,
    w_node,
    w_edge,
    destroy_fraction=0.2
):

    S = set(S)

    remove_count = max(
        1,
        int(len(S) * destroy_fraction)
    )

    removed = random.sample(
        list(S),
        remove_count
    )

    for u in removed:
        S.remove(u)

    # repair
    uncovered = set()

    for u in G.nodes():

        if u in S:
            continue

        dominated = any(
            v in S
            for v in G.neighbors(u)
        )

        if not dominated:
            uncovered.add(u)

    while uncovered:

        best = None
        best_score = -float('inf')

        for v in G.nodes():

            if v in S:
                continue

            if any(
                u in S
                for u in G.neighbors(v)
            ):
                continue

            gain = 0

            for u in uncovered:

                if u == v or G.has_edge(u, v):
                    gain += 1

            score = gain / (w_node[v] + 1e-9)

            if score > best_score:

                best_score = score
                best = v

        if best is None:
            break

        S.add(best)

        uncovered = {

            u for u in uncovered

            if not (
                u == best
                or
                G.has_edge(u, best)
            )
        }

    S = remove_redundant(S, G)

    return S


# ============================================================
# Variable Neighborhood Descent
# ============================================================

def VND(
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

    k = 1

    while k <= 2:

        if k == 1:

            candidate, cost = one_swap(
                S,
                G,
                w_node,
                w_edge
            )

        else:

            candidate, cost = two_swap(
                S,
                G,
                w_node,
                w_edge
            )

        if cost < current_cost:

            S = candidate
            current_cost = cost

            k = 1

        else:
            k += 1

    return S, current_cost


# ============================================================
# Main VNS
# ============================================================

def VNS(
    G,
    time_limit
):

    start_time = perf_counter()

    w_node, w_edge = build_weight_tables(G)

    current_S = construct_initial_solution(
        G,
        w_node,
        w_edge
    )

    current_S = remove_redundant(current_S, G)

    current_cost = calc_objective(
        current_S,
        G,
        w_node,
        w_edge
    )

    best_S = set(current_S)
    best_cost = current_cost

    iterations = 0

    while perf_counter() - start_time < time_limit:

        iterations += 1

        # ============================================
        # Shaking
        # ============================================

        shaken = destroy_repair(
            current_S,
            G,
            w_node,
            w_edge,
            destroy_fraction=0.2
        )

        # ============================================
        # VND
        # ============================================

        improved_S, improved_cost = VND(
            shaken,
            G,
            w_node,
            w_edge
        )

        # ============================================
        # Acceptance
        # ============================================

        if improved_cost < current_cost:

            current_S = improved_S
            current_cost = improved_cost

            if current_cost < best_cost:

                best_cost = current_cost
                best_S = set(current_S)

    elapsed = perf_counter() - start_time

    return (
        best_S,
        best_cost,
        elapsed,
        iterations
    )


# ============================================================
# Batch
# ============================================================

def solve_all_vns(
    in_dir_path,
    subset,
    out_dir_path
):

    os.makedirs(out_dir_path, exist_ok=True)

    out_file = os.path.join(
        out_dir_path,
        f"{subset}_vns_results.csv"
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

        _, cost, elapsed, iters = VNS(
            G,
            time_limit
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
        required=True
    )

    parser.add_argument(
        '-s',
        '--instances_subset',
        required=True
    )

    parser.add_argument(
        '-o',
        '--out_dir_path',
        required=True
    )

    args = parser.parse_args()

    solve_all_vns(
        args.in_dir_path,
        args.instances_subset,
        args.out_dir_path
    )