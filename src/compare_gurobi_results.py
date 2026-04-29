"""
Utility script to compare two Gurobi exact solver results.
It aggregates the Best Objective, MIP Gap, and computation time across different 
weight schemes (EG, NG, VG) and generates a comparative text report and a bar chart.
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

def format_gap(diff_pct):
    """Convert percentage difference into a readable string."""
    if diff_pct < 0:
        return f"Reduced by {-diff_pct:.2f}% (Improved)"
    elif diff_pct > 0:
        return f"Increased by {diff_pct:.2f}% (Degraded)"
    else:
        return "No change (0.00%)"

def compare_gurobi_and_plot():
    # 1. Set file paths to compare (Ensure filenames match the target results)
    file_1 = './results/500_r0c14_gurobi_new2_baseline_results.csv' 
    file_2 = './results/500_r0c14_gurobi_warm_ensemble_results.csv'

    # Set save directory
    save_dir = './results/compare'
    os.makedirs(save_dir, exist_ok=True)

    name1 = get_algo_name(file_1)
    name2 = get_algo_name(file_2)

    try:
        df_1 = pd.read_csv(file_1, skipinitialspace=True)
        df_2 = pd.read_csv(file_2, skipinitialspace=True)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return

    # Strip whitespaces from column names
    df_1.columns = df_1.columns.str.strip()
    df_2.columns = df_2.columns.str.strip()

    # Merge dataframes on instance name
    df = pd.merge(df_1, df_2, on='instance', suffixes=('_1', '_2'))
    df['scheme'] = df['instance'].apply(classify_scheme)

    # Group by scheme to calculate averages
    grouped = df.groupby('scheme')[['best_obj_1', 'best_obj_2', 'gap_percent_1', 'gap_percent_2', 'total_time_1', 'total_time_2']].mean()

    output_lines = []
    header = f"========== Gurobi Exact Solver Performance Comparison ({name1} vs {name2}) =========="
    print(f"\n{header}")
    output_lines.append(header)
    
    for scheme in grouped.index:
        subset = df[df['scheme'] == scheme]
        
        # Win/Loss logic: (1) lower objective wins, (2) if tied, shorter time wins
        wins = ((subset['best_obj_2'] < subset['best_obj_1']) | 
                ((subset['best_obj_2'] == subset['best_obj_1']) & (subset['total_time_2'] < subset['total_time_1']))).sum()
        losses = ((subset['best_obj_2'] > subset['best_obj_1']) | 
                  ((subset['best_obj_2'] == subset['best_obj_1']) & (subset['total_time_2'] > subset['total_time_1']))).sum()
        ties = ((subset['best_obj_2'] == subset['best_obj_1']) & (subset['total_time_2'] == subset['total_time_1'])).sum()
        
        obj_1 = grouped.loc[scheme, 'best_obj_1']
        obj_2 = grouped.loc[scheme, 'best_obj_2']
        gap_1 = grouped.loc[scheme, 'gap_percent_1']
        gap_2 = grouped.loc[scheme, 'gap_percent_2']
        t_1 = grouped.loc[scheme, 'total_time_1']
        t_2 = grouped.loc[scheme, 'total_time_2']
        
        # Calculate objective difference percentage
        obj_diff_pct = ((obj_2 - obj_1) / obj_1) * 100 if obj_1 > 0 else 0
        
        scheme_header = f"\n[{scheme}]"
        print(scheme_header)
        output_lines.append(scheme_header)
        
        obj_str = f"  - Avg Best Objective: {name1} {obj_1:.1f} vs {name2} {obj_2:.1f}"
        if obj_2 < obj_1:
            win_str = f" ({name2} Wins)"
        elif obj_1 < obj_2:
            win_str = f" ({name1} Wins)"
        else:
            win_str = f" (Tie in Objective, evaluated by time)"
            
        print(obj_str + win_str)
        output_lines.append(obj_str + win_str)
            
        stats1 = f"  - Record ({name2} perspective): {wins} Wins, {ties} Ties, {losses} Losses"
        stats2 = f"  - Objective Diff: {format_gap(obj_diff_pct)}"
        stats3 = f"  - Avg MIP Gap: {name1} {gap_1:.2f}% vs {name2} {gap_2:.2f}%"
        stats4 = f"  - Avg Time: {name1} {t_1:.1f}s vs {name2} {t_2:.1f}s"
        
        print(stats1)
        print(stats2)
        print(stats3)
        print(stats4)
        output_lines.extend([stats1, stats2, stats3, stats4])
        
    overall_header = f"\n[Overall Average]"
    print(overall_header)
    output_lines.append(overall_header)
    
    total_wins = ((df['best_obj_2'] < df['best_obj_1']) | ((df['best_obj_2'] == df['best_obj_1']) & (df['total_time_2'] < df['total_time_1']))).sum()
    total_losses = ((df['best_obj_2'] > df['best_obj_1']) | ((df['best_obj_2'] == df['best_obj_1']) & (df['total_time_2'] > df['total_time_1']))).sum()
    total_ties = ((df['best_obj_2'] == df['best_obj_1']) & (df['total_time_2'] == df['total_time_1'])).sum()
    
    avg_obj_1 = df['best_obj_1'].mean()
    avg_obj_2 = df['best_obj_2'].mean()
    avg_gap_1 = df['gap_percent_1'].mean()
    avg_gap_2 = df['gap_percent_2'].mean()
    
    total_diff_pct = ((avg_obj_2 - avg_obj_1) / avg_obj_1) * 100 if avg_obj_1 > 0 else 0
    
    overall1 = f"  - Avg Best Objective: {name1} {avg_obj_1:.1f} vs {name2} {avg_obj_2:.1f}"
    overall2 = f"  - Total Record: {total_wins} Wins, {total_ties} Ties, {total_losses} Losses"
    overall3 = f"  - Total Objective Diff: {format_gap(total_diff_pct)}"
    overall4 = f"  - Total Avg MIP Gap: {name1} {avg_gap_1:.2f}% vs {name2} {avg_gap_2:.2f}%"
    overall5 = f"  - Total Avg Time: {name1} {df['total_time_1'].mean():.1f}s vs {name2} {df['total_time_2'].mean():.1f}s"
    
    print(overall1)
    print(overall2)
    print(overall3)
    print(overall4)
    print(overall5)
    output_lines.extend([overall1, overall2, overall3, overall4, overall5])
    
    footer = "================================================================="
    print(f"{footer}\n")
    output_lines.append(footer)

    # Save text report
    txt_filename = f"comp_gurobi_{name1}_vs_{name2}.txt"
    txt_save_path = os.path.join(save_dir, txt_filename)
    with open(txt_save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"Text report saved: {txt_save_path}")

    # --- Plotting (Best Objective) ---
    schemes = grouped.index.tolist()
    obj_1_vals = grouped['best_obj_1'].tolist()
    obj_2_vals = grouped['best_obj_2'].tolist()

    x = np.arange(len(schemes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, obj_1_vals, width, label=name1, color='#1f77b4')
    rects2 = ax.bar(x + width/2, obj_2_vals, width, label=name2, color='#ff7f0e')

    ax.set_ylabel('Average Best Objective')
    ax.set_title(f'Gurobi Best Objective: {name1} vs {name2} (500 Nodes)')
    ax.set_xticks(x)
    ax.set_xticklabels(schemes)
    ax.legend()

    # Label bars with exact values
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

    img_filename = f"comp_gurobi_{name1}_vs_{name2}.png"
    img_save_path = os.path.join(save_dir, img_filename)
    plt.savefig(img_save_path, dpi=300)
    print(f"Chart saved: {img_save_path}")

if __name__ == '__main__':
    compare_gurobi_and_plot()