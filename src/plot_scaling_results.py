import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

def classify_scheme(filename):
    if 'nw100_ew100' in filename: return 'NG'
    elif 'nw1000_ew10' in filename: return 'VG'
    elif 'nw10_ew1000' in filename: return 'EG'
    return 'Other'

def plot_scaling(scales, results_dir='../results'):
    schemes = ['EG', 'NG', 'VG']
    
    # Map the 4 model filenames 
    models = {
        'NEW2 Baseline': '{N}_r0c14_new2_results.csv',
        'NEW2 + Grid Cuts': '{N}_r0c14_grid_cut_results.csv',
        'Grid Warm-Start': '{N}_r0c14_gurobi_warm_ensemble_results.csv', 
        'Max Clique V4': '{N}_r0c14_maxclique_v4_results.csv'
    }
    
    # Assign colors and markers for visual distinction
    colors = {
        'NEW2 Baseline': '#d62728',     # Red
        'NEW2 + Grid Cuts': '#ff7f0e',  # Orange 
        'Grid Warm-Start': '#1f77b4',   # Blue
        'Max Clique V4': '#2ca02c'      # Green
    }
    markers = {
        'NEW2 Baseline': 'o',           # Circle
        'NEW2 + Grid Cuts': 'v',        # Downward triangle
        'Grid Warm-Start': 's',         # Square
        'Max Clique V4': '^'            # Upward triangle
    }

    data = []
    for N in scales:
        time_limit = 3.0 * N
        for model_name, file_pattern in models.items():
            file_path = os.path.join(results_dir, file_pattern.format(N=N))
            if not os.path.exists(file_path):
                continue
                
            df = pd.read_csv(file_path, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            df['scheme'] = df['instance'].apply(classify_scheme)
            
            for idx, row in df.iterrows():
                t = row['total_time']
                gap = row.get('gap_percent', 0.0) 
                
                # Determine timeout (exceeded time limit or gap remains)
                is_timeout = (t >= time_limit - 1.0) or (gap > 0.1)
                final_time = time_limit if is_timeout else t
                
                data.append({'N': N, 'model': model_name, 'scheme': row['scheme'], 'time': final_time, 'timeout': is_timeout})

    full_df = pd.DataFrame(data)
    if full_df.empty:
        print("[Error] No data found to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Scalability Limit of Exact Formulations (Time Limit: 3|V| sec)', fontsize=18, fontweight='bold', y=0.95)

    for i, scheme in enumerate(schemes):
        ax = axes[i]
        scheme_df = full_df[full_df['scheme'] == scheme]
        
        for model_name in models.keys():
            m_df = scheme_df[scheme_df['model'] == model_name]
            if m_df.empty: continue
            
            avg_df = m_df.groupby('N').agg({'time': 'mean', 'timeout': 'mean'}).reset_index()
            avg_df['timeout'] = avg_df['timeout'] > 0.5 
            
            # Connect all points with a line
            ax.plot(avg_df['N'], avg_df['time'], color=colors[model_name], linewidth=2.5, alpha=0.7, label=model_name)
            
            # Marker for successful (solved) instances
            solved = avg_df[~avg_df['timeout']]
            if not solved.empty:
                ax.scatter(solved['N'], solved['time'], color=colors[model_name], marker=markers[model_name], s=120, zorder=5)
            
            # X marker for timeout instances
            timeout = avg_df[avg_df['timeout']]
            if not timeout.empty:
                ax.scatter(timeout['N'], timeout['time'], color=colors[model_name], marker='X', s=200, zorder=6, edgecolor='black', linewidth=1.5)

        # 3|V| limit line
        x_vals = np.array(scales)
        ax.plot(x_vals, 3.0 * x_vals, color='black', linestyle='--', linewidth=2, alpha=0.6, label='Time Limit (3|V|)')

        ax.set_title(f'{scheme} (Weight Distribution)', fontsize=14, pad=10)
        ax.set_xlabel('Number of Nodes (N)', fontsize=12)
        if i == 0: ax.set_ylabel('Computation Time (s) - Log Scale', fontsize=12)
        
        ax.set_xlim(min(scales) - 20, max(scales) + 20)
        ax.set_xticks(scales)
        ax.set_yscale('log')
        ax.grid(True, which="major", ls="-", alpha=0.4)
        ax.grid(True, which="minor", ls=":", alpha=0.2)
        
        if i == 0:
            ax.legend(fontsize=11, loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(results_dir, 'scaling_comparison_4way_log.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[Success] 4-way comparison log plot saved to: {plot_path}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scales', type=str, required=True, help="Comma separated list of N (e.g., 200,250,300)")
    args = parser.parse_args()
    
    scales_list = [int(x) for x in args.scales.split(',')]
    plot_scaling(scales_list)