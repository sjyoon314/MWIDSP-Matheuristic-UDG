import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def categorize_environment(instance_name):
    """Categorize the environment based on the instance name."""
    name = str(instance_name).lower()
    if 'nw1000_ew10' in name: return 'VG'
    elif 'nw100_ew100' in name: return 'NG'
    elif 'nw10_ew1000' in name: return 'EG'
    else: return 'UNKNOWN'

def load_heuristic_data(file_mapping):
    dfs = []
    for algo_name, file_path in file_mapping.items():
        try:
            df = pd.read_csv(file_path, skipinitialspace=True)
            df.columns = [c.strip().lower() for c in df.columns]
            
            cost_col = 'cost' if 'cost' in df.columns else 'best_obj'
            time_col = None
            for t_name in ['total_time', 'time', 'cpu_time', 'runtime']:
                if t_name in df.columns:
                    time_col = t_name
                    break
                    
            if cost_col not in df.columns or time_col is None:
                continue
                
            subset = df[['instance', cost_col, time_col]].copy()
            subset.rename(columns={cost_col: 'cost', time_col: 'time'}, inplace=True)
            subset['Algorithm'] = algo_name
            dfs.append(subset)
            
        except Exception as e:
            print(f"[Error] Failed to load {algo_name}: {e}")
            
    if not dfs: return pd.DataFrame()
    
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df['Environment'] = merged_df['instance'].apply(categorize_environment)
    return merged_df

def get_pareto_front(df):
    """Extract the Pareto front from data, minimizing both time and cost."""
    # Sort in ascending order by time
    sorted_df = df.sort_values('time')
    pareto_front = []
    min_cost = float('inf')
    
    for index, row in sorted_df.iterrows():
        # Add to Pareto front only if the cost strictly improves
        if row['cost'] < min_cost:
            pareto_front.append(row)
            min_cost = row['cost']
            
    return pd.DataFrame(pareto_front)

def plot_pareto_time_vs_cost(df):
    """Visualize the Pareto front across VG, NG, and EG environments using subplots."""
    df_filtered = df[df['Environment'].isin(['VG', 'NG', 'EG'])]
    summary = df_filtered.groupby(['Environment', 'Algorithm']).agg({'time': 'mean', 'cost': 'mean'}).reset_index()
    
    sns.set_theme(style="white") # Set style to white for custom grids
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    environments = ['VG', 'NG', 'EG']
    
    for i, env in enumerate(environments):
        env_data = summary[summary['Environment'] == env].copy()
        if env_data.empty:
            continue
            
        ax = axes[i]
        
        # 1. Add major and minor grids for log scale clarity
        ax.grid(True, which="major", ls="-", color='0.85', zorder=0)
        ax.grid(True, which="minor", ls=":", color='0.90', zorder=0)
        
        # 2. Plot scatter points
        sns.scatterplot(
            data=env_data, 
            x='time', 
            y='cost', 
            hue='Algorithm',
            style='Algorithm', 
            s=300, # Enlarge markers
            palette="tab10",
            edgecolor='black',
            linewidth=1.5,
            alpha=0.9,
            ax=ax,
            zorder=3
        )
        
        # 3. Plot Pareto front lines
        pareto_df = get_pareto_front(env_data)
        if not pareto_df.empty:
            ax.plot(
                pareto_df['time'], 
                pareto_df['cost'], 
                color='gray', 
                linestyle='--', 
                linewidth=2, 
                zorder=2,
                label='Pareto Front' if i == 0 else "" # Prevent duplicate legend entries
            )
        
        # Configure axes
        ax.set_xscale('log')
        ax.set_title(f'{env} Environment', fontsize=16, fontweight='bold')
        ax.set_xlabel('Avg Time (s) - Log Scale', fontsize=12)
        if i == 0:
            ax.set_ylabel('Avg Cost (Lower is better)', fontsize=12)
        else:
            ax.set_ylabel('')
            
        # Remove individual subplot legends to avoid clutter
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Place unified legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    # Filter duplicate labels (e.g., Pareto Front)
    unique_labels = dict(zip(labels, handles))
    fig.legend(
        unique_labels.values(), 
        unique_labels.keys(), 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.05), # Position centrally at the bottom to save space
        ncol=5, # Spread across 5 columns
        fontsize=12, 
        title="Algorithms", 
        title_fontsize=14,
        frameon=True
    )
    
    plt.tight_layout()
    plt.savefig("../results/compare/Pareto_Time_vs_Cost_All.png", dpi=300, bbox_inches='tight')
    print("Successfully saved Pareto plot to: Pareto_Time_vs_Cost_All.png")
    plt.show()

if __name__ == "__main__":
    files = {
        "GREEDY-NEW": "../results/500_r0c14_greedynew_results.csv",
        "Grid_Ensemble": "../results/500_r0c14_grid_heuristic_edge_aware_sum_results.csv",
        "Clique_Ensemble": "../results/500_r0c14_clique_heuristic_results.csv",
        #"SA": "../results/500_r0c14_sa_results.csv",
        "GRASP": "../results/500_r0c14_grasp_results.csv",
        "VNS": "../results/500_r0c14_vns_results.csv",
        "LP_Rounding": "../results/500_r0c14_lp_rounding_results.csv",
        "Cross_LS": "../results/500_r0c14_cross_ls_results.csv",
        "Shift_Full": "../results/500_r0c14_shift_full_results.csv"
    }

    merged_df = load_heuristic_data(files)
    if not merged_df.empty:
        plot_pareto_time_vs_cost(merged_df)