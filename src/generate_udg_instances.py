"""
UDG Instance Generator for the MWIDSP.
Generates random Unit Disk Graphs (UDGs) with 2D spatial coordinates 
and assigns weights based on the three environments specified in the paper:
NG (Neutral), VG (Node-oriented), and EG (Edge-oriented).
"""

import networkx as nx
import random
import os
from argparse import ArgumentParser

def generate_udg_files(num_nodes=500, radius=0.14, num_instances=10, out_dir=None):
    # If out_dir is not explicitly provided, fall back to a default relative path
    if out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, '..', 'instances', 'my_udg_with_pos')
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Three weight schemes specified in the paper
    # NG (Neutral): Nodes 1~100, Edges 1~100
    # VG (Node-oriented): Nodes 1~1000, Edges 1~10
    # EG (Edge-oriented): Nodes 1~10, Edges 1~1000
    schemes = [
        {"type": "NG", "nw_max": 100, "ew_max": 100},
        {"type": "VG", "nw_max": 1000, "ew_max": 10},
        {"type": "EG", "nw_max": 10, "ew_max": 1000}
    ]
    
    for scheme in schemes:
        nw_max = scheme["nw_max"]
        ew_max = scheme["ew_max"]
        scheme_type = scheme["type"]
        
        print(f"\n--- Generating {num_instances} instances for {scheme_type} (nw: 1~{nw_max}, ew: 1~{ew_max}) ---")
        
        for i in range(num_instances):
            # 1. Generate a UDG with 2D spatial coordinates
            G = nx.random_geometric_graph(num_nodes, radius)
            
            # 2. Define filename (e.g., 500_r0c14_nw100_ew100_0.rgg)
            # Replace decimal point in radius for safe filename formatting (0.14 -> 0c14)
            radius_str = str(radius).replace('.', 'c')
            file_name = f"{num_nodes}_r{radius_str}_nw{nw_max}_ew{ew_max}_{i}.rgg"
            file_path = os.path.join(out_dir, file_name)
            
            edges = list(G.edges())
            
            with open(file_path, 'w') as f:
                # Header: Number of nodes, Number of undirected edges
                f.write(f"{num_nodes}\t{len(edges)}\n")
                
                # Node weights (1 ~ nw_max)
                node_weights = {u: random.randint(1, nw_max) for u in G.nodes()}
                for u in range(num_nodes):
                    f.write(f"{node_weights[u]}\n")
                    
                # Edge weights (1 ~ ew_max)
                for u, v in edges:
                    ew = random.randint(1, ew_max)
                    f.write(f"{u}\t{v}\t{ew}\n")
                    
                # Append 2D spatial coordinates at the end of the file
                f.write("POSITIONS\n")
                for u in range(num_nodes):
                    x, y = G.nodes[u]['pos']
                    f.write(f"{u}\t{x:.6f}\t{y:.6f}\n")
                    
            print(f"[{file_name}] Generation complete.")

if __name__ == "__main__":
    # Dynamically resolve relative path assuming script is in 'src/'
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'instances', 'my_udg_with_pos')
    
    parser = ArgumentParser(description="Generate UDG instances for MWIDSP")
    parser.add_argument('-n', '--num_nodes', type=int, default=500, help="Number of nodes per instance")
    parser.add_argument('-r', '--radius', type=float, default=0.14, help="Communication radius")
    parser.add_argument('-i', '--num_instances', type=int, default=10, help="Number of instances per scheme")
    parser.add_argument('-o', '--out_dir', type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    args = parser.parse_args()
    
    generate_udg_files(
        num_nodes=args.num_nodes, 
        radius=args.radius, 
        num_instances=args.num_instances, 
        out_dir=args.out_dir
    )