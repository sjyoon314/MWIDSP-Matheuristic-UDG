import networkx as nx
from sortedcontainers import SortedSet

def calc_initial_solution_cost(solution, g):
    sum_node_weights = 0
    external_edge_weights = 0

    sorted_external_edges = [None for _ in g.nodes]

    num_incorrect = 0
    all_nodes = set(g.nodes)
    out_solution = all_nodes.difference(solution)
    sorted_out_edges = [None for _ in g.nodes]

    for u, u_weight in g.nodes(data=True):
        sorted_external_edges[u] = SortedSet([(v, g[u][v]['weight']) for v in solution if g.has_edge(u,v)],
                                                key=lambda x: x[1])
        sorted_out_edges[u] = SortedSet([(v, g[u][v]['weight']) for v in out_solution if g.has_edge(u,v)],
                                        key=lambda x: x[1])
        if u in solution:
            sum_node_weights += u_weight['weight']
            if len(sorted_external_edges[u]) > 0:
                num_incorrect += 1
        else:
            if len(sorted_external_edges[u]) > 0:
                external_edge_weights += sorted_external_edges[u][0][1]
            else:
                num_incorrect += 1
            
    total_cost = sum_node_weights + external_edge_weights
    return num_incorrect, total_cost, sorted_external_edges, sorted_out_edges

def read_line_of_ints(f):
    xs = [int(x) for x in f.readline().split()]
    if len(xs) == 1:
        return xs[0]
    return xs

def read_instance(file_path):
    with open(file_path, 'r') as f:
        num_nodes, num_edges = read_line_of_ints(f)
        g = nx.Graph()

        for i in range(num_nodes):
            node_weight = read_line_of_ints(f)
            g.add_node(i, weight=node_weight)

        for i in range(num_edges):
            u, v, edge_weight = read_line_of_ints(f)
            g.add_edge(u, v, weight=edge_weight)

        return g