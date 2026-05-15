import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def classify_scheme(filename):
    if 'nw100_ew100' in filename:
        return 'NG (Neutral)'
    elif 'nw1000_ew10' in filename:
        return 'VG (Node-oriented)'
    elif 'nw10_ew1000' in filename:
        return 'EG (Edge-oriented)'
    else:
        return 'Other'

def get_algo_name(filepath):
    filename = os.path.basename(filepath)
    if 'r0c14_' in filename:
        return filename.split('r0c14_')[-1].split('_results.csv')[0]
    else:
        return filename.replace('_results.csv', '')

def format_gap(diff_pct):
    if diff_pct < 0:
        return f"Reduced by {-diff_pct:.2f}% (Improved)"
    elif diff_pct > 0:
        return f"Increased by {diff_pct:.2f}% (Degraded)"
    else:
        return "No change (0.00%)"

def compare_gurobi_and_plot():
    # 1. 파일 경로 설정
    file_1 = '../results/500_r0c14_gurobi_new2_baseline_results.csv' 
    file_2 = '../results/500_r0c14_gurobi_warm_ensemble_results.csv'

    save_dir = '../results/compare'
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

    df = pd.merge(df_1, df_2, on='instance', suffixes=('_1', '_2'))
    df['scheme'] = df['instance'].apply(classify_scheme)

    grouped = df.groupby('scheme')[['best_obj_1', 'best_obj_2', 'gap_percent_1', 'gap_percent_2', 'total_time_1', 'total_time_2']].mean()

    output_lines = []
    header = f"========== Exact Solver Performance: {name1} vs {name2} =========="
    print(f"\n{header}")
    output_lines.append(header)
    
    tol = 1e-4 # 부동소수점 오차 방지

    for scheme in grouped.index:
        subset = df[df['scheme'] == scheme]
        
        # [핵심 수정] 새로운 Win/Loss 로직: 1순위 Best Obj -> 2순위 MIP Gap -> 3순위 Time
        obj_tie = np.abs(subset['best_obj_2'] - subset['best_obj_1']) < tol
        gap_tie = np.abs(subset['gap_percent_2'] - subset['gap_percent_1']) < tol
        
        wins = (
            (subset['best_obj_2'] < subset['best_obj_1'] - tol) | 
            (obj_tie & (subset['gap_percent_2'] < subset['gap_percent_1'] - tol)) |
            (obj_tie & gap_tie & (subset['total_time_2'] < subset['total_time_1'] - tol))
        ).sum()
        
        losses = (
            (subset['best_obj_1'] < subset['best_obj_2'] - tol) | 
            (obj_tie & (subset['gap_percent_1'] < subset['gap_percent_2'] - tol)) |
            (obj_tie & gap_tie & (subset['total_time_1'] < subset['total_time_2'] - tol))
        ).sum()
        
        ties = len(subset) - wins - losses
        
        obj_1 = grouped.loc[scheme, 'best_obj_1']
        obj_2 = grouped.loc[scheme, 'best_obj_2']
        gap_1 = grouped.loc[scheme, 'gap_percent_1']
        gap_2 = grouped.loc[scheme, 'gap_percent_2']
        t_1 = grouped.loc[scheme, 'total_time_1']
        t_2 = grouped.loc[scheme, 'total_time_2']
        
        obj_diff_pct = ((obj_2 - obj_1) / obj_1) * 100 if obj_1 > 0 else 0
        gap_diff_pct = gap_2 - gap_1 # 갭은 % 단위이므로 절대 %p 차이로 계산
        
        scheme_header = f"\n[{scheme}]"
        print(scheme_header)
        output_lines.append(scheme_header)
        
        print(f"  - Record ({name2} perspective): {wins} Wins, {ties} Ties, {losses} Losses")
        print(f"  - Avg Best Obj: {name1} {obj_1:.1f} vs {name2} {obj_2:.1f} ({format_gap(obj_diff_pct)})")
        
        # 갭이 크게 개선된 경우 하이라이트 표시
        gap_str = f"  - Avg MIP Gap : {name1} {gap_1:.2f}% vs {name2} {gap_2:.2f}%"
        if gap_2 < gap_1 - tol:
            gap_str += f" (Massive Improvement: {gap_diff_pct:.2f}%p)"
        print(gap_str)
        print(f"  - Avg Time    : {name1} {t_1:.1f}s vs {name2} {t_2:.1f}s")
        
        output_lines.extend([
            f"  - Record: {wins} Wins, {ties} Ties, {losses} Losses",
            f"  - Avg Best Obj: {obj_1:.1f} vs {obj_2:.1f} ({format_gap(obj_diff_pct)})",
            gap_str,
            f"  - Avg Time: {t_1:.1f}s vs {t_2:.1f}s"
        ])

    # 텍스트 저장 로직 생략 (기존 코드와 동일)
    txt_filename = f"comp_gurobi_gap_{name1}_vs_{name2}.txt"
    txt_save_path = os.path.join(save_dir, txt_filename)
    with open(txt_save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"\nText report saved: {txt_save_path}")

    # --- Plotting (1) Best Objective (기존과 동일) ---
    schemes = grouped.index.tolist()
    x = np.arange(len(schemes))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) # 1행 2열로 차트 2개 생성

    rects1 = ax1.bar(x - width/2, grouped['best_obj_1'], width, label=name1, color='#1f77b4')
    rects2 = ax1.bar(x + width/2, grouped['best_obj_2'], width, label=name2, color='#ff7f0e')
    ax1.set_ylabel('Average Best Objective')
    ax1.set_title('Best Objective')
    ax1.set_xticks(x)
    ax1.set_xticklabels(schemes)
    ax1.legend()

    # --- Plotting (2) MIP Gap (핵심 차트) ---
    rects3 = ax2.bar(x - width/2, grouped['gap_percent_1'], width, label=name1, color='#d62728')
    rects4 = ax2.bar(x + width/2, grouped['gap_percent_2'], width, label=name2, color='#2ca02c')
    ax2.set_ylabel('Average MIP Gap (%)')
    ax2.set_title('MIP Gap (Proof of Polyhedral Strengthening)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(schemes)
    ax2.legend()

    def autolabel(rects, ax, fmt='{:.0f}'):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(fmt.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    autolabel(rects3, ax2, fmt='{:.1f}%')
    autolabel(rects4, ax2, fmt='{:.1f}%')

    plt.tight_layout()
    img_filename = f"comp_gurobi_gap_{name1}_vs_{name2}.png"
    img_save_path = os.path.join(save_dir, img_filename)
    plt.savefig(img_save_path, dpi=300)
    print(f"Chart saved: {img_save_path}")

if __name__ == '__main__':
    compare_gurobi_and_plot()