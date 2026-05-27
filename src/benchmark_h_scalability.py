"""
Benchmark Script for Massive Scale MWIDSP Instances.
Evaluates the scalability and time complexity of proposed heuristic ensembles
across different scoring strategies, comparing them against a baseline (GreedyNew).
"""

import os
import subprocess
import glob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def estimate_big_o(n_array, time_array):
    """Estimates the exponent k for time complexity O(N^k) using log-log linear regression."""
    t_safe = np.maximum(time_array, 1e-5)
    log_n = np.log(n_array)
    log_t = np.log(t_safe)
    slope, _ = np.polyfit(log_n, log_t, 1)
    return slope

def run_massive_comparison_benchmark():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IN_DIR = os.path.join(SCRIPT_DIR, '..', 'instances', 'udg_fixed_radius')
    OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'complexity')
    os.makedirs(OUT_DIR, exist_ok=True)
    
    TARGET_SCALES = [500, 1000, 2000, 5000]
    SCORE_TYPES = ["node_only", "edge_aware_min", "edge_aware_sum"]
    ENSEMBLES = [
        ("grid_heuristic", "heuristic_grid_ensemble.py"),
        ("clique_heuristic", "heuristic_clique_ensemble.py")
    ]
    BASELINE_FILE = "base_h_greedynew.py"
    EXP_NAME = "scaling_test"
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting massive scale benchmark (Ensembles vs Baseline)")
    
    # ==========================================
    # 1. Execute Algorithms
    # ==========================================
    for n in TARGET_SCALES:
        print(f"\n{'='*50}\nEvaluating Scale: N={n}\n{'='*50}")
        
        prefix = f"{n}_r0c14"
        # Assumes instances are already generated and present in the directory
        expected_count = len([f for f in os.listdir(IN_DIR) if f.startswith(f"{prefix}_") and f.endswith(".rgg")])
        
        if expected_count == 0:
            print(f"  [WARNING] No instances found for N={n}. Skipping.")
            continue

        # [A] Run Ensembles (Grid and Clique) across all scoring strategies
        for base_name, py_file in ENSEMBLES:
            for score in SCORE_TYPES:
                out_csv = f"{prefix}_{base_name}_results.csv" if score == 'node_only' else f"{prefix}_{base_name}_{score}_{EXP_NAME}_results.csv"
                out_path = os.path.join(OUT_DIR, out_csv)
                
                is_done = False
                if os.path.exists(out_path):
                    with open(out_path, 'r') as f:
                        lines = [l for l in f.readlines()[1:] if l.strip()]
                        if len(lines) >= expected_count:
                            is_done = True
                
                if is_done:
                    print(f"  [SKIP] {base_name} ({score}) already processed.")
                else:
                    print(f"  [RUN] Processing {base_name} ({score})...")
                    cmd = ["python", py_file, "-i", IN_DIR, "-s", prefix, "-o", OUT_DIR, "-e", EXP_NAME, "--score_type", score]
                    try: 
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e: 
                        print(f"  [ERROR] {base_name} failed: {e}")

        # [B] Run Baseline (GreedyNew)
        baseline_search = glob.glob(os.path.join(OUT_DIR, f"{prefix}*greedy*.csv"))
        is_base_done = False
        if baseline_search:
            with open(baseline_search[0], 'r') as f:
                lines = [l for l in f.readlines()[1:] if l.strip()]
                if len(lines) >= expected_count:
                    is_base_done = True
                    
        if is_base_done:
            print(f"  [SKIP] Baseline (GreedyNew) already processed.")
        else:
            print(f"  [RUN] Processing Baseline (GreedyNew)...")
            cmd_base = ["python", BASELINE_FILE, "-i", IN_DIR, "-s", prefix, "-o", OUT_DIR]
            try: 
                subprocess.run(cmd_base, check=True)
            except subprocess.CalledProcessError as e: 
                print(f"  [ERROR] Baseline failed: {e}")

    # ==========================================
    # 2. Data Aggregation & Visualization
    # ==========================================
    print("\nAggregating data and estimating time complexity...")
    results = []
    
    for n in TARGET_SCALES:
        prefix = f"{n}_r0c14"
        
        # Collect Ensemble Results
        for base_name, _ in ENSEMBLES:
            for score in SCORE_TYPES:
                csv_name = f"{prefix}_{base_name}_results.csv" if score == 'node_only' else f"{prefix}_{base_name}_{score}_results.csv"
                csv_path = os.path.join(OUT_DIR, csv_name)
                
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, skipinitialspace=True)
                    if not df.empty:
                        results.append({'N': n, 'Algorithm': base_name, 'Score': score, 'Time': df['time'].mean(), 'Cost': df['cost'].mean()})
                else:
                    print(f"  [WARNING] Aggregation failed - File not found: {csv_name}")
                        
        # Collect Baseline Results
        baseline_files = glob.glob(os.path.join(OUT_DIR, f"{prefix}*greedy*.csv"))
        if baseline_files:
            df_base = pd.read_csv(baseline_files[0], skipinitialspace=True)
            if not df_base.empty:
                results.append({'N': n, 'Algorithm': 'Baseline', 'Score': 'GreedyNew', 'Time': df_base['time'].mean(), 'Cost': df_base['cost'].mean()})

    if not results:
        print("No result data found. Visualization aborted.")
        return

    df_res = pd.DataFrame(results)
    
    summary_excel = os.path.join(OUT_DIR, 'massive_scale_comparison_summary.xlsx')
    summary_csv = os.path.join(OUT_DIR, 'massive_scale_comparison_summary.csv')

    try:
        df_res.to_excel(summary_excel, index=False)
        print(f"Excel summary saved: {summary_excel}")
    except Exception as e:
        print(f"Failed to save Excel (check openpyxl): {e}")

    df_res.to_csv(summary_csv, index=False)
    print(f"CSV summary saved: {summary_csv}")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Massive Scale Performance: Proposed Ensembles vs Baseline', fontsize=18, fontweight='bold')
    
    color_map = {
        'grid_heuristic': {'node_only': '#1f77b4', 'edge_aware_min': '#2ca02c', 'edge_aware_sum': '#d62728'},
        'clique_heuristic': {'node_only': '#aec7e8', 'edge_aware_min': '#98df8a', 'edge_aware_sum': '#ff9896'},
        'Baseline': {'GreedyNew': '#000000'} 
    }
    marker_map = {'node_only': 'o', 'edge_aware_min': '^', 'edge_aware_sum': 's', 'GreedyNew': 'D'} 
    ls_map = {'grid_heuristic': '-', 'clique_heuristic': '--', 'Baseline': '-.'} 

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

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Nodes (N) [Log Scale]', fontsize=12)
    ax1.set_ylabel('Execution Time (s) [Log Scale]', fontsize=12)
    ax1.set_title('Empirical Time Complexity (Log-Log Plot)', fontsize=14)
    ax1.set_xticks(TARGET_SCALES)
    ax1.set_xticklabels([str(n) for n in TARGET_SCALES])
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2.set_xlabel('Number of Nodes (N)', fontsize=12)
    ax2.set_ylabel('Average Cost (Lower is Better)', fontsize=12)
    ax2.set_title('Cost Scaling Progression', fontsize=14)
    ax2.set_xticks(TARGET_SCALES)
    ax2.grid(True, ls="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(OUT_DIR, 'massive_comparison_with_baseline.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved successfully: {plot_path}")

if __name__ == "__main__":
    run_massive_comparison_benchmark()