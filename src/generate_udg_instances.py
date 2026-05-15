"""
UDG Instance Generator for the MWIDSP.
Generates random Unit Disk Graphs (UDGs) with 2D spatial coordinates.
Supports two scalability testing modes:
1. Fixed Radius (Density Explosion): Increases N while keeping r constant.
2. Constant Density (Fair Scaling): Increases N while automatically shrinking r to maintain expected degree.
"""

import networkx as nx
import random
import os
import math
from argparse import ArgumentParser

def generate_udg_files(num_nodes=500, radius=0.14, num_instances=10, out_dir=None, maintain_density=False):

    if maintain_density:
        N_base = 500
        r_base = 0.14
        # equation: r_new = r_base * sqrt(N_base / N)
        calculated_radius = r_base * math.sqrt(N_base / num_nodes)
        radius = round(calculated_radius, 4) 
        print(f"[*] Constant Density Mode ON: Radius automatically adjusted to r={radius} for N={num_nodes}")
    else:
        print(f"[*] Fixed Radius Mode: Radius remains fixed at r={radius} for N={num_nodes}")


    if out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder_name = 'udg_constant_density' if maintain_density else 'udg_fixed_radius'
        out_dir = os.path.join(script_dir, '..', 'instances', folder_name)
        
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Output Directory: {out_dir}")
    
    # Three weight schemes specified in the paper
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
            
            # 2. Define filename (e.g., 1000_r0c099_nw100_ew100_0.rgg)
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
                    
            print(f"[{file_name}] Generation complete. (Nodes: {num_nodes}, Edges: {len(edges)})")

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate UDG instances for MWIDSP Scalability Tests")
    parser.add_argument('-n', '--num_nodes', type=int, default=500, help="Number of nodes per instance")
    parser.add_argument('-r', '--radius', type=float, default=0.14, help="Base communication radius (if not using --maintain_density)")
    parser.add_argument('-i', '--num_instances', type=int, default=10, help="Number of instances per scheme")
    parser.add_argument('-o', '--out_dir', type=str, default=None, help="Specific output directory (optional)")
    parser.add_argument('-d', '--maintain_density', action='store_true', help="Flag to automatically calculate r to maintain expected degree")
    
    args = parser.parse_args()
    
    generate_udg_files(
        num_nodes=args.num_nodes, 
        radius=args.radius, 
        num_instances=args.num_instances, 
        out_dir=args.out_dir,
        maintain_density=args.maintain_density
    )