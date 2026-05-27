"""
Utility script to generate data for the Pareto Summary Table (N=500).
Aggregates Cost and Time across EG, NG, VG environments for multiple algorithms,
and automatically calculates the percentage gap relative to the best metaheuristic.
"""

import os
import pandas as pd

def categorize_environment(instance_name):
    """Categorize environment based on instance parameters."""
    name = str(instance_name).lower()
    if 'nw1000_ew10' in name: return 'VG (Node-oriented)'
    elif 'nw100_ew100' in name: return 'NG (Neutral)'
    elif 'nw10_ew1000' in name: return 'EG (Edge-oriented)'
    else: return 'UNKNOWN'

def generate_pareto_table_data(file_mapping):
    dfs = []
    
    # 1. Load and merge files
    for algo_name, file_path in file_mapping.items():
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path, skipinitialspace=True)
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Flexibly locate cost and time columns
            cost_col = 'cost' if 'cost' in df.columns else 'best_obj'
            time_col = None
            for t_name in ['total_time', 'time', 'cpu_time', 'runtime']:
                if t_name in df.columns:
                    time_col = t_name
                    break
                    
            if cost_col not in df.columns or time_col is None:
                print(f"[Warning] Missing required cost or time column in {algo_name}. Skipping.")
                continue
                
            subset = df[['instance', cost_col, time_col]].copy()
            subset.rename(columns={cost_col: 'cost', time_col: 'time'}, inplace=True)
            subset['Algorithm'] = algo_name
            dfs.append(subset)
            
        except Exception as e:
            print(f"[Error] Failed to load data for {algo_name}: {e}")
            
    if not dfs:
        print("[Error] No valid data loaded. Process terminated.")
        return
        
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['Environment'] = df_all['instance'].apply(categorize_environment)
    
    # 2. Calculate averages per environment and algorithm
    summary = df_all.groupby(['Environment', 'Algorithm'])[['cost', 'time']].mean().reset_index()
    
    # 3. Map baseline metaheuristics per environment
    meta_map = {
        'EG (Edge-oriented)': 'VNS',
        'NG (Neutral)': 'VNS',
        'VG (Node-oriented)': 'VNS'
    }
    
    # Define display order (matching the LaTeX table structure)
    display_order = ['GREEDY-NEW', 'Grid_Ensemble', 'Clique_Ensemble', 'GRASP', 'VNS']
    
    print("\n" + "="*85)
    print(" PARETO TABLE SUMMARY DATA (N=500)")
    print("="*85)
    
    for env in ['EG (Edge-oriented)', 'NG (Neutral)', 'VG (Node-oriented)']:
        print(f"\n[{env}]")
        env_data = summary[summary['Environment'] == env]
        
        if env_data.empty:
            print("  [Info] No data available for this environment.")
            continue
            
        # Extract baseline metaheuristic cost for gap calculation
        meta_algo = meta_map.get(env)
        meta_row = env_data[env_data['Algorithm'] == meta_algo]
        
        meta_cost = None
        if not meta_row.empty:
            meta_cost = meta_row.iloc[0]['cost']
        else:
            print(f"  [Warning] Baseline metaheuristic '{meta_algo}' data missing. Cannot calculate gap.")
            
        print(f"{'Algorithm':<20} | {'Avg. Cost':<12} | {'Avg. Time (s)':<15} | {'Gap to Meta (%)':<20}")
        print("-" * 75)
        
        for algo in display_order:
            row = env_data[env_data['Algorithm'] == algo]
            if row.empty:
                continue
                
            cost = row.iloc[0]['cost']
            time = row.iloc[0]['time']
            
            # Calculate Gap %
            gap_str = "N/A"
            if meta_cost is not None:
                if algo == meta_algo:
                    gap_str = "Baseline (0.00%)"
                else:
                    gap_pct = ((cost - meta_cost) / meta_cost) * 100
                    sign = "+" if gap_pct > 0 else ""
                    gap_str = f"{sign}{gap_pct:.2f}%"
                    
            print(f"{algo:<20} | {cost:<12.1f} | {time:<15.4f} | {gap_str:<20}")

if __name__ == '__main__':
    files = {
        'GREEDY-NEW': '../results/500_r0c14_greedynew_results.csv',
        'Grid_Ensemble': '../results/500_r0c14_grid_heuristic_results.csv',
        'Clique_Ensemble': '../results/500_clique_heuristic_results.csv',
        #SA_smart': '../results/500_r0c14_sa_smart_results.csv',
        'GRASP': '../results/500_r0c14_grasp_results.csv',
        'VNS': '../results/500_r0c14_vns_results.csv',

    }
    
    generate_pareto_table_data(files)