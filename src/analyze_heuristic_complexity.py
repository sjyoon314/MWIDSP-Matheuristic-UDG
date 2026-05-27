import os
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np

def categorize_environment(instance_name):
    """Categorize environment based on instance name parameters."""
    name = str(instance_name).lower()
    if 'nw1000_ew10' in name: return 'VG'
    elif 'nw100_ew100' in name: return 'NG'
    elif 'nw10_ew1000' in name: return 'EG'
    else: return 'UNKNOWN'

def compare_and_plot(results_dir, exp_name, sizes):
    # Dictionary structure to store data by environment
    envs = ['EG', 'NG', 'VG']
    data = {
        env: {
            'grid_costs': [], 'clique_costs': [],
            'grid_times': [], 'clique_times': [],
            'valid_sizes': []
        } for env in envs
    }

    print(f"\n==================================================================")
    print(f"[{exp_name.upper()}] Grid vs Clique Ensemble by Environment")
    print(f"==================================================================")

    for n in sizes:
        grid_file = os.path.join(results_dir, f"{exp_name}_{n}_grid_ensemble_results.csv")
        clique_file = os.path.join(results_dir, f"{exp_name}_{n}_clique_ensemble_results.csv")

        # Fallback for constant_density n=500 (uses fixed_radius files)
        if exp_name == "constant_density" and n == 500:
            g_file1 = os.path.join(results_dir, "fixed_radius_500_grid_ensemble_results.csv")
            g_file2 = os.path.join(results_dir, "fixed_radius_500__grid_ensemble_results.csv")
            grid_file = g_file1 if os.path.exists(g_file1) else g_file2
            
            c_file1 = os.path.join(results_dir, "fixed_radius_500_clique_ensemble_results.csv")
            c_file2 = os.path.join(results_dir, "fixed_radius_500__clique_ensemble_results.csv")
            clique_file = c_file1 if os.path.exists(c_file1) else c_file2
            
        if not os.path.exists(grid_file) or not os.path.exists(clique_file):
            print(f"[Warning] N={n:<5} | Missing data (file not found) - Skipped")
            continue

        try:
            df_grid = pd.read_csv(grid_file)
            df_clique = pd.read_csv(clique_file)

            df_grid.columns = df_grid.columns.str.strip()
            df_clique.columns = df_clique.columns.str.strip()

            df_grid['Environment'] = df_grid['instance'].apply(categorize_environment)
            df_clique['Environment'] = df_clique['instance'].apply(categorize_environment)

            for env in envs:
                env_grid = df_grid[df_grid['Environment'] == env]
                env_clique = df_clique[df_clique['Environment'] == env]

                if not env_grid.empty and not env_clique.empty:
                    data[env]['grid_costs'].append(env_grid['cost'].mean())
                    data[env]['clique_costs'].append(env_clique['cost'].mean())
                    data[env]['grid_times'].append(env_grid['time'].mean())
                    data[env]['clique_times'].append(env_clique['time'].mean())
                    data[env]['valid_sizes'].append(n)
                    
        except Exception as e:
            print(f"[Error] N={n:<5} | File read error: {e}")

    # Plotting configurations
    colors = {'EG': 'blue', 'NG': 'green', 'VG': 'red'}
    markers_grid = {'EG': 'o', 'NG': 's', 'VG': '^'}
    markers_clique = {'EG': 'o', 'NG': 's', 'VG': '^'}
    linestyles_grid = '-'
    linestyles_clique = '--'

    has_data = any(len(data[env]['valid_sizes']) > 0 for env in envs)
    if not has_data:
        print("[Info] No valid data available to generate plots.")
        return

    # --- 1. Plot Cost Comparison ---
    plt.figure(figsize=(10, 6))
    for env in envs:
        sz = data[env]['valid_sizes']
        if not sz: continue
        
        plt.plot(sz, data[env]['grid_costs'], marker=markers_grid[env], linestyle=linestyles_grid, 
                 label=f'{env} Grid', color=colors[env], markersize=8)
        plt.plot(sz, data[env]['clique_costs'], marker=markers_clique[env], linestyle=linestyles_clique, 
                 label=f'{env} Clique', color=colors[env], alpha=0.6, markersize=8)

    plt.title(f'Cost Comparison by Environment ({exp_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Nodes (N)', fontsize=12)
    plt.ylabel('Average Cost', fontsize=12)
    
    # Set primary X-axis ticks based on the longest valid_sizes list
    best_x = max([data[e]['valid_sizes'] for e in envs if data[e]['valid_sizes']], key=len)
    plt.xticks(best_x)  
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    cost_plot_path = os.path.join(results_dir, f"{exp_name}_cost_env_comparison.png")
    plt.savefig(cost_plot_path, bbox_inches='tight')
    print(f"[Success] Cost comparison plot saved to: {cost_plot_path}")

    # --- 2. Plot Time Comparison (Log-Log) ---
    plt.figure(figsize=(10, 6))
    for env in envs:
        sz = data[env]['valid_sizes']
        if not sz: continue
        
        plt.loglog(sz, data[env]['grid_times'], marker=markers_grid[env], linestyle=linestyles_grid, 
                   label=f'{env} Grid', color=colors[env], markersize=8, base=10)
        plt.loglog(sz, data[env]['clique_times'], marker=markers_clique[env], linestyle=linestyles_clique, 
                   label=f'{env} Clique', color=colors[env], alpha=0.6, markersize=8, base=10)
        
        # Calculate slope
        if len(sz) > 1:
            log_n = np.log10(sz)
            slope_grid = np.polyfit(log_n, np.log10(data[env]['grid_times']), 1)[0]
            slope_clique = np.polyfit(log_n, np.log10(data[env]['clique_times']), 1)[0]
            print(f"[{env}] Slope (k) -> Grid: {slope_grid:.2f} | Clique: {slope_clique:.2f}")

    plt.title(f'Time Complexity by Environment (Log-Log) - {exp_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Nodes (N) [Log Scale]', fontsize=12)
    plt.ylabel('Average Time (Seconds) [Log Scale]', fontsize=12)
    plt.xticks(best_x, labels=[str(n) for n in best_x]) 
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    time_plot_path = os.path.join(results_dir, f"{exp_name}_time_env_loglog_comparison.png")
    plt.savefig(time_plot_path, bbox_inches='tight')
    print(f"[Success] Log-Log Time comparison plot saved to: {time_plot_path}")

if __name__ == '__main__':
    parser = ArgumentParser(description="Compare Grid vs Clique Ensembles by Environment")
    parser.add_argument('-d', '--results_dir', type=str, required=True, help="Directory path containing result CSV files")
    parser.add_argument('-e', '--exp_name', type=str, required=True, help="Experiment name (e.g., fixed_radius or constant_density)")
    parser.add_argument('-s', '--sizes', nargs='+', type=int, required=True, help="List of N sizes (e.g., 500 1000 2000)")
    
    args = parser.parse_args()
    compare_and_plot(args.results_dir, args.exp_name, args.sizes)