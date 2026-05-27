import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def estimate_big_o(n_array, time_array):
    t_safe = np.maximum(time_array, 1e-5)
    log_n = np.log(n_array)
    log_t = np.log(t_safe)
    slope, _ = np.polyfit(log_n, log_t, 1)
    return slope

def redraw_graphs_with_improvements():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'complexity')
    csv_path = os.path.join(OUT_DIR, 'massive_scale_comparison_summary.csv')
    
    if not os.path.exists(csv_path):
        print(f"[Error] Summary file not found: {csv_path}")
        return

    df_res = pd.read_csv(csv_path)
    TARGET_SCALES = sorted(df_res['N'].unique())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Massive Scale Performance & Cost Reduction Analysis', fontsize=18, fontweight='bold')
    
    color_map = {
        'grid_heuristic': {'node_only': '#1f77b4', 'edge_aware_min': '#2ca02c', 'edge_aware_sum': '#d62728'},
        'clique_heuristic': {'node_only': '#aec7e8', 'edge_aware_min': '#98df8a', 'edge_aware_sum': '#ff9896'},
        'Baseline': {'GreedyNew': '#000000'} 
    }
    marker_map = {'node_only': 'o', 'edge_aware_min': '^', 'edge_aware_sum': 's', 'GreedyNew': 'D'}
    ls_map = {'grid_heuristic': '-', 'clique_heuristic': '--', 'Baseline': '-.'}

    # 1. Plot empirical data
    for (algo, score), group in df_res.groupby(['Algorithm', 'Score']):
        group = group.sort_values('N')
        n_vals = group['N'].values
        time_vals = group['Time'].values
        cost_vals = group['Cost'].values
        
        k_power = estimate_big_o(n_vals, time_vals)
        algo_display = "Baseline" if algo == "Baseline" else algo.split('_')[0].capitalize()
        label_str = f"{algo_display} ({score}) [O(N^{k_power:.2f})]"
        label_cost = f"{algo_display} ({score})"
        
        c = color_map[algo][score]
        m = marker_map[score]
        ls = ls_map[algo]
        lw = 3 if algo == 'Baseline' else 2 
        
        ax1.plot(n_vals, time_vals, marker=m, linestyle=ls, color=c, label=label_str, linewidth=lw, markersize=8)
        ax2.plot(n_vals, cost_vals, marker=m, linestyle=ls, color=c, label=label_cost, linewidth=lw, markersize=8)

    # 2. Add cost reduction percentage annotation (based on N=5000)
    n_max = 5000
    df_max = df_res[df_res['N'] == n_max]
    if not df_max.empty:
        baseline_cost = df_max[df_max['Algorithm'] == 'Baseline']['Cost'].values[0]
        # Identify the best (lowest) cost among proposed algorithms
        best_ours_cost = df_max[df_max['Algorithm'] != 'Baseline']['Cost'].min()
        
        reduction_pct = ((baseline_cost - best_ours_cost) / baseline_cost) * 100
        
        # Highlight reduction with arrow and text box
        ax2.annotate(f'-{reduction_pct:.1f}% Cost\nReduction!',
                     xy=(n_max, best_ours_cost), xycoords='data',
                     xytext=(n_max - 500, baseline_cost * 0.7), textcoords='data',
                     arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=10),
                     fontsize=14, fontweight='bold', color='red',
                     ha='right', va='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=2))

    # Configure Time chart (Log-Log)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Nodes (N) [Log Scale]', fontsize=12)
    ax1.set_ylabel('Execution Time (s) [Log Scale]', fontsize=12)
    ax1.set_title('Empirical Time Complexity (Log-Log Plot)', fontsize=14)
    ax1.set_xticks(TARGET_SCALES)
    ax1.set_xticklabels([str(n) for n in TARGET_SCALES])
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    # Configure Cost chart (Log X-axis to clarify scaling gaps)
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Nodes (N) [Log Scale]', fontsize=12)
    ax2.set_ylabel('Average Cost (Lower is Better)', fontsize=12)
    ax2.set_title('Cost Scaling Progression (Highlighting Cost Savings)', fontsize=14)
    ax2.set_xticks(TARGET_SCALES)
    ax2.set_xticklabels([str(n) for n in TARGET_SCALES])
    ax2.grid(True, ls="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(OUT_DIR, 'massive_comparison_annotated.png')
    plt.savefig(plot_path, dpi=300)
    print(f"[Success] Annotated chart saved successfully to: {plot_path}")

if __name__ == "__main__":
    redraw_graphs_with_improvements()