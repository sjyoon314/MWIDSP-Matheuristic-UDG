import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_strategy_logs(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract instance names and costs for the 4 strategies using regex
    pattern = re.compile(
        r"--- Instance:\s+(.+?)\s+---\n"
        r"\s+- Strategy \[size\s*\] -> Cost:\s+([\d.]+)\n"
        r"\s+- Strategy \[cost\s*\] -> Cost:\s+([\d.]+)\n"
        r"\s+- Strategy \[hub\s*\] -> Cost:\s+([\d.]+)\n"
        r"\s+- Strategy \[inverse_size\s*\] -> Cost:\s+([\d.]+)"
    )
    
    data = []
    for m in pattern.findall(content):
        inst = m[0]
        # Determine the weight environment (VG, EG, NG) based on the instance name
        env = 'VG (Node-Heavy)' if 'nw1000_ew10' in inst else \
              'EG (Edge-Heavy)' if 'nw10_ew1000' in inst else 'NG (Neutral)'
        
        costs = {'size': float(m[1]), 'cost': float(m[2]), 'hub': float(m[3]), 'inverse_size': float(m[4])}
        best_cost = min(costs.values())
        best_strat = min(costs, key=costs.get)
        
        row = {'instance': inst, 'env': env, 'winner': best_strat}
        
        # Calculate the Inefficiency Gap (%) relative to the best (minimum) cost
        for k, v in costs.items():
            row[f'{k}_gap'] = (v - best_cost) / best_cost * 100
        data.append(row)
        
    return pd.DataFrame(data)

if __name__ == '__main__':
    # =========================================================================
    # User Configuration
    # Specify the exact path to the target log file to parse.
    # Example: '../results/complexity/scaling_test_500_r0c14_clique_heuristic_edge_aware_sum_strategy_logs.txt'
    # =========================================================================
    target_log_file = '../results/complexity/scaling_test_500_r0c14_clique_heuristic_edge_aware_sum_strategy_logs.txt'
    
    if not os.path.exists(target_log_file):
        print(f"Error: Log file not found at [{target_log_file}]. Please check the path.")
        exit()
        
    print(f"Parsing log file: {target_log_file}")
    output_dir = os.path.dirname(target_log_file)
    
    # Parse the specific target file
    df = parse_strategy_logs(target_log_file)

    if df.empty:
        print("Warning: No matching data found in the log file.")
        exit()

    # ---------------------------------------------------------
    # 1. Bar Chart: Winning Strategy Count by Environment
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='env', hue='winner', palette='viridis')
    plt.title('Winning Strategy by Weight Environment', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Wins', fontsize=12)
    plt.xlabel('Environment', fontsize=12)
    plt.legend(title='Strategy', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    wins_plot_path = os.path.join(output_dir, 'strategy_wins_by_env.png')
    plt.savefig(wins_plot_path, bbox_inches='tight')
    print(f"Saved winning strategy bar chart to: {wins_plot_path}")

    # ---------------------------------------------------------
    # 2. Heatmap: Strategy Inefficiency (Gap %) by Environment
    # ---------------------------------------------------------
    gap_cols = ['size_gap', 'cost_gap', 'hub_gap', 'inverse_size_gap']
    heatmap_data = df.groupby('env')[gap_cols].mean()
    heatmap_data.columns = ['Size', 'Cost', 'Hub', 'Inverse Size'] # Rename for readability

    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='Reds', cbar_kws={'label': 'Average Gap to Best (%)'})
    plt.title('Strategy Inefficiency (Gap %) by Environment', fontsize=16, fontweight='bold')
    plt.ylabel('Environment', fontsize=12)
    plt.xlabel('Strategy', fontsize=12)
    
    heatmap_plot_path = os.path.join(output_dir, 'strategy_gap_heatmap.png')
    plt.savefig(heatmap_plot_path, bbox_inches='tight')
    print(f"Saved strategy inefficiency heatmap to: {heatmap_plot_path}")