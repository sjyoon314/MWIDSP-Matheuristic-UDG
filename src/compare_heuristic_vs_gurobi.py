"""
Utility script to compare Heuristic pipeline results against Gurobi exact solver results.
It aligns column names (cost/best_obj) internally, calculates optimization gaps, 
and generates a comparative text report and a bar chart.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def classify_scheme(filename):
    """Classify the instance into 3 schemes based on node/edge weight bounds."""
    if 'nw100_ew100' in filename:
        return 'NG (Neutral)'
    elif 'nw1000_ew10' in filename:
        return 'VG (Node-oriented)'
    elif 'nw10_ew1000' in filename:
        return 'EG (Edge-oriented)'
    else:
        return 'Other'

def get_algo_name(filepath):
    """Extract the algorithm/model name from the file path."""
    filename = os.path.basename(filepath)
    if 'r0c14_' in filename:
        return filename.split('r0c14_')[-1].split('_results.csv')[0]
    else:
        return filename.replace('_results.csv', '')

def format_gap(gap_pct):
    """Convert percentage difference into a readable string."""
    if gap_pct < 0:
        return f"Reduced by {-gap_pct:.2f}% (Improved)"
    elif gap_pct > 0:
        return f"Increased by {gap_pct:.2f}% (Degraded)"
    else:
        return "No change (0.00%)"

def compare_and_plot():
    # 1. Set file paths to compare
    file_1 = './results/500_r0c14_ensemble_shift_full_results.csv' 
    file_2 = './results/500_r0c14_gurobi_new2_baseline_results.csv'

    # Set save directory
    save_dir = r'../results/compare'
    os.makedirs(save_dir, exist_ok=True)

    name1 = get_algo_name(file_1)
    name2 = get_algo_name(file_2)

    try:
        df_1 = pd.read_csv(file_1, skipinitialspace=True)
        df_2 = pd.read_csv(file_2, skipinitialspace=True)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return

    df_1.columns = df_1.columns.str.strip()
    df_2.columns = df_2.columns.str.strip()

    # Align Gurobi columns (best_obj, total_time) with heuristic columns (cost, time)
    if 'best_obj' in df_1.columns:
        df_1 = df_1.rename(columns={'best_obj': 'cost', 'total_time': 'time'})
    if 'best_obj' in df_2.columns:
        df_2 = df_2.rename(columns={'best_obj': 'cost', 'total_time': 'time'})

    # Merge dataframes on instance name
    df = pd.merge(df_1, df_2, on='instance', suffixes=('_1', '_2'))
    df['scheme'] = df['instance'].apply(classify_scheme)

    grouped = df.groupby('scheme')[['cost_1', 'cost_2', 'time_1', 'time_2']].mean()

    output_lines = []
    header = f"========== Heuristic vs Exact Solver Comparison ({name1} vs {name2}) =========="
    print(f"\n{header}")
    output_lines.append(header)
    
    for scheme in grouped.index:
        subset = df[df['scheme'] == scheme]
        wins = (subset['cost_2'] < subset['cost_1']).sum()
        losses = (subset['cost_2'] > subset['cost_1']).sum()
        ties = (subset['cost_2'] == subset['cost_1']).sum()
        
        c_1 = grouped.loc[scheme, 'cost_1']
        c_2 = grouped.loc[scheme, 'cost_2']
        t_1 = grouped.loc[scheme, 'time_1']
        t_2 = grouped.loc[scheme, 'time_2']
        
        gap_pct = ((c_2 - c_1) / c_1) * 100 if c_1 > 0 else 0
        
        scheme_header = f"\n[{scheme}]"
        print(scheme_header)
        output_lines.append(scheme_header)
        
        cost_str = f"  - Avg Cost: {name1} {c_1:.1f} vs {name2} {c_2:.1f}"
        if c_2 < c_1:
            win_str = f" ({name2} Wins)"
        elif c_1 < c_2:
            win_str = f" ({name1} Wins)"
        else:
            win_str = " (Tie)"
            
        print(cost_str + win_str)
        output_lines.append(cost_str + win_str)
            
        stats1 = f"  - Record ({name2} perspective): {wins} Wins, {ties} Ties, {losses} Losses"
        stats2 = f"  - Cost Diff: {format_gap(gap_pct)}"
        stats3 = f"  - Avg Time: {name1} {t_1:.1f}s vs {name2} {t_2:.1f}s"
        
        print(stats1)
        print(stats2)
        print(stats3)
        output_lines.extend([stats1, stats2, stats3])
        
    overall_header = f"\n[Overall Average]"
    print(overall_header)
    output_lines.append(overall_header)
    
    total_wins = (df['cost_2'] < df['cost_1']).sum()
    total_losses = (df['cost_2'] > df['cost_1']).sum()
    total_ties = (df['cost_2'] == df['cost_1']).sum()
    
    avg_c_1 = df['cost_1'].mean()
    avg_c_2 = df['cost_2'].mean()
    total_gap_pct = ((avg_c_2 - avg_c_1) / avg_c_1) * 100 if avg_c_1 > 0 else 0
    
    overall1 = f"  - Avg Cost: {name1} {avg_c_1:.1f} vs {name2} {avg_c_2:.1f}"
    overall2 = f"  - Total Record: {total_wins} Wins, {total_ties} Ties, {total_losses} Losses"
    overall3 = f"  - Total Cost Diff: {format_gap(total_gap_pct)}"
    overall4 = f"  - Total Avg Time: {name1} {df['time_1'].mean():.1f}s vs {name2} {df['time_2'].mean():.1f}s"
    
    print(overall1)
    print(overall2)
    print(overall3)
    print(overall4)
    output_lines.extend([overall1, overall2, overall3, overall4])
    
    footer = "================================================================="
    print(f"{footer}\n")
    output_lines.append(footer)

    txt_filename = f"comp_Heur_vs_Gurobi.txt"
    txt_save_path = os.path.join(save_dir, txt_filename)
    with open(txt_save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"Text report saved: {txt_save_path}")

    # --- Plotting ---
    schemes = grouped.index.tolist()
    cost_1_vals = grouped['cost_1'].tolist()
    cost_2_vals = grouped['cost_2'].tolist()

    x = np.arange(len(schemes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, cost_1_vals, width, label=name1, color='#1f77b4')
    rects2 = ax.bar(x + width/2, cost_2_vals, width, label=name2, color='#ff7f0e')

    ax.set_ylabel('Average Cost')
    ax.set_title(f'Heuristic vs Exact Solver Cost Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(schemes)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()

    img_filename = f"comp_Heur_vs_Gurobi.png"
    img_save_path = os.path.join(save_dir, img_filename)
    plt.savefig(img_save_path, dpi=300)
    print(f"Chart saved: {img_save_path}")

if __name__ == '__main__':
    compare_and_plot()