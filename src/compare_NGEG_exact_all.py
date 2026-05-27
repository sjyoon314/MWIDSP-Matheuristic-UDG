import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_merge_data(file_mapping):
    """Merge multiple CSV files into one for data preprocessing."""
    dfs = []
    for algo_name, file_path in file_mapping.items():
        try:
            df = pd.read_csv(file_path, skipinitialspace=True)
            # Standardize all column names to lowercase and remove padding
            df.columns = [c.strip().lower() for c in df.columns]
            
            # 1. Identify Lower Bound column
            lb_col = None
            if 'best_bound' in df.columns:
                lb_col = 'best_bound'
            elif 'lower_bound' in df.columns:
                lb_col = 'lower_bound'
                
            # 2. Identify Upper Bound (Objective) column
            ub_col = None
            if 'best_obj' in df.columns:
                ub_col = 'best_obj'
            elif 'cost' in df.columns:
                ub_col = 'cost'
                
            # 3. Check for required columns
            if lb_col is None or ub_col is None or 'gap_percent' not in df.columns:
                print(f"[Warning] Skipping {algo_name}: Missing required columns (bound/obj/gap).")
                continue

            # Extract only necessary columns
            subset = df[['instance', ub_col, lb_col, 'gap_percent']].copy()
            
            # Standardize names for analysis
            subset.rename(columns={
                ub_col: 'best_obj',
                lb_col: 'best_bound'
            }, inplace=True)
            
            subset['Algorithm'] = algo_name
            dfs.append(subset)
            
        except Exception as e:
            print(f"[Error] Failed to load data for {algo_name}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def categorize_environment(instance_name):
    """Categorize environment into VG, NG, or EG based on instance parameters (nw, ew)."""
    name = str(instance_name).lower()
    if 'nw1000_ew10' in name: 
        return 'VG'
    elif 'nw100_ew100' in name: 
        return 'NG'
    elif 'nw10_ew1000' in name: 
        return 'EG'
    else: 
        return 'UNKNOWN'

def plot_gap_boxplots(df):
    """Visualize Optimality Gap percentages for NG and EG environments in side-by-side subplots."""
    df_plot = df[df['Environment'].isin(['NG', 'EG'])].copy()
    
    if df_plot.empty:
        print("[Error] No NG or EG environment data found. Please check instance naming conventions.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False) 
    
    # 1. Boxplot for NG Environment
    df_ng = df_plot[df_plot['Environment'] == 'NG']
    sns.boxplot(data=df_ng, x='Algorithm', y='gap_percent', ax=axes[0], palette="Set2")
    axes[0].set_title('Optimality Gap Distribution - NG Environment', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Gap (%)', fontsize=12)
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=45)

    # 2. Boxplot for EG Environment
    df_eg = df_plot[df_plot['Environment'] == 'EG']
    sns.boxplot(data=df_eg, x='Algorithm', y='gap_percent', ax=axes[1], palette="Set2")
    axes[1].set_title('Optimality Gap Distribution - EG Environment', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Gap (%)', fontsize=12)
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("../results/compare/Gap_Distribution_NG_EG.png", dpi=300)
    print("[Success] Boxplot saved: Gap_Distribution_NG_EG.png")
    plt.show()

def analyze_relative_bounds(df, baseline_name='NEW-2'):
    """Calculate Relative LB and UB performance against the baseline (NEW-2)."""
    if baseline_name not in df['Algorithm'].values:
        print(f"[Warning] Baseline '{baseline_name}' not found in data. Skipping relative analysis.")
        return

    df_pivot = df.pivot(index=['instance', 'Environment'], columns='Algorithm', 
                        values=['best_obj', 'best_bound', 'gap_percent'])
    
    results = []
    algorithms = df['Algorithm'].unique()
    
    for algo in algorithms:
        if algo == baseline_name:
            continue
            
        rel_lb = df_pivot['best_bound'][algo] / df_pivot['best_bound'][baseline_name]
        rel_ub = df_pivot['best_obj'][algo] / df_pivot['best_obj'][baseline_name]
        
        for env in ['NG', 'EG']:
            # Create mask for the current environment
            env_mask = df_pivot.index.get_level_values('Environment') == env
            
            res = {
                'Environment': env,
                'Algorithm': algo,
                'Avg Gap (%)': df_pivot['gap_percent'][algo][env_mask].mean(),
                'Median Rel_LB (>1.0 is better)': rel_lb[env_mask].median(),
                'Median Rel_UB (<1.0 is better)': rel_ub[env_mask].median()
            }
            results.append(res)
            
    summary_df = pd.DataFrame(results).round(4).sort_values(by=['Environment', 'Algorithm'])
    print("\n========== Relative Bound Analysis Summary (Baseline: NEW-2) ==========")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    files = {
        "NEW-2": "../results/500_r0c14_new2_results.csv",
        "Exact_Clique_Cut": "../results/500_r0c14_maxclique_v4_results.csv",
        "Exact_Warm_Grid": "../results/500_r0c14_gurobi_warm_ensemble_results.csv",
        "Exact_Grid_Cut": "../results/500_r0c14_grid_cut_results.csv",
        "Exact_Grid_SOS1": "../results/500_r0c14_grid_sos1_scaling_test_results.csv"
    }

    print("Initializing data loading and analysis...")
    merged_df = load_and_merge_data(files)

    if not merged_df.empty:
        merged_df['Environment'] = merged_df['instance'].apply(categorize_environment)
        plot_gap_boxplots(merged_df)
        analyze_relative_bounds(merged_df, baseline_name='NEW-2')
    else:
        print("[Error] Data loading failed. No data to analyze.")