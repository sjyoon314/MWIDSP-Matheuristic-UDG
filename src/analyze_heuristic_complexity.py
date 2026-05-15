import os
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np

def compare_and_plot(results_dir, exp_name, sizes):
    grid_costs, clique_costs = [], []
    grid_times, clique_times = [], []
    valid_sizes = []

    print(f"\n==================================================================")
    print(f"[{exp_name.upper()}] Grid vs Clique Ensemble Comparison")
    print(f"==================================================================")
    print(f"{'N':<6} | {'Grid Cost':<12} | {'Clique Cost':<12} | {'Grid Time(s)':<14} | {'Clique Time(s)':<14}")
    print(f"------------------------------------------------------------------")

    for n in sizes:
        grid_file = os.path.join(results_dir, f"{exp_name}_{n}_grid_ensemble_results.csv")
        clique_file = os.path.join(results_dir, f"{exp_name}_{n}_clique_ensemble_results.csv")

        if exp_name == "constant_density" and n == 500:
            g_file1 = os.path.join(results_dir, "fixed_radius_500_grid_ensemble_results.csv")
            g_file2 = os.path.join(results_dir, "fixed_radius_500__grid_ensemble_results.csv") # 예전 파일명
            grid_file = g_file1 if os.path.exists(g_file1) else g_file2
            
            c_file1 = os.path.join(results_dir, "fixed_radius_500_clique_ensemble_results.csv")
            c_file2 = os.path.join(results_dir, "fixed_radius_500__clique_ensemble_results.csv") # 예전 파일명
            clique_file = c_file1 if os.path.exists(c_file1) else c_file2
            
        # 두 파일 중 하나라도 없으면 건너뜀
        if not os.path.exists(grid_file) or not os.path.exists(clique_file):
            print(f"{n:<6} | 데이터 누락 (파일 없음) - 스킵됨")
            continue

        try:
            df_grid = pd.read_csv(grid_file)
            df_clique = pd.read_csv(clique_file)

            # 컬럼명 공백 제거 (안전장치)
            df_grid.columns = df_grid.columns.str.strip()
            df_clique.columns = df_clique.columns.str.strip()

            avg_grid_cost = df_grid['cost'].mean()
            avg_clique_cost = df_clique['cost'].mean()
            avg_grid_time = df_grid['time'].mean()
            avg_clique_time = df_clique['time'].mean()

            grid_costs.append(avg_grid_cost)
            clique_costs.append(avg_clique_cost)
            grid_times.append(avg_grid_time)
            clique_times.append(avg_clique_time)
            valid_sizes.append(n)

            print(f"{n:<6} | {avg_grid_cost:<12.2f} | {avg_clique_cost:<12.2f} | {avg_grid_time:<14.4f} | {avg_clique_time:<14.4f}")
        except Exception as e:
            print(f"{n:<6} | 파일 읽기 에러: {e}")
            
    print(f"==================================================================\n")

    if not valid_sizes:
        print("그래프를 그릴 유효한 데이터가 없습니다.")
        return

    # --- 1. Cost 비교 그래프 그리기 ---
    plt.figure(figsize=(8, 5))
    plt.plot(valid_sizes, grid_costs, marker='o', linestyle='-', label='Grid Ensemble', color='blue', markersize=8)
    plt.plot(valid_sizes, clique_costs, marker='s', linestyle='-', label='Clique Ensemble', color='red', markersize=8)
    plt.title(f'Cost Comparison ({exp_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Nodes (N)', fontsize=12)
    plt.ylabel('Average Cost', fontsize=12)
    plt.xticks(valid_sizes)  # X축에 우리가 입력한 N 값만 딱 보이게 고정
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    cost_plot_path = os.path.join(results_dir, f"{exp_name}_cost_comparison.png")
    plt.savefig(cost_plot_path, bbox_inches='tight')
    print(f"✅ Cost 그래프 저장 완료: {cost_plot_path}")

    # --- 2. Time 비교 그래프 그리기 ---
    plt.figure(figsize=(8, 5))
    
    # X축, Y축 모두 로그 스케일로 플로팅
    plt.loglog(valid_sizes, grid_times, marker='o', linestyle='-', label='Grid Ensemble', color='blue', markersize=8, base=10)
    plt.loglog(valid_sizes, clique_times, marker='s', linestyle='-', label='Clique Ensemble', color='red', markersize=8, base=10)
    
    # [핵심] 실제 시간 복잡도(기울기 k) 계산
    if len(valid_sizes) > 1:
        log_n = np.log10(valid_sizes)
        slope_grid = np.polyfit(log_n, np.log10(grid_times), 1)[0]
        slope_clique = np.polyfit(log_n, np.log10(clique_times), 1)[0]
        
        # 그래프에 기울기(Empirical O(N^k)) 텍스트 표시
        plt.text(valid_sizes[-1], grid_times[-1], f' Slope (k) ≈ {slope_grid:.2f}', color='blue', fontsize=12, va='top')
        plt.text(valid_sizes[-1], clique_times[-1], f' Slope (k) ≈ {slope_clique:.2f}', color='red', fontsize=12, va='bottom')
        
        print(f"👉 [실증적 시간 복잡도] Grid: O(N^{slope_grid:.2f}) / Clique: O(N^{slope_clique:.2f})")

    plt.title(f'Time Complexity (Log-Log Scale) - {exp_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Nodes (N) [Log Scale]', fontsize=12)
    plt.ylabel('Average Time (Seconds) [Log Scale]', fontsize=12)
    
    # X축 눈금을 우리가 테스트한 N 값으로 명확히 표시
    plt.xticks(valid_sizes, labels=[str(n) for n in valid_sizes]) 
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    
    time_plot_path = os.path.join(results_dir, f"{exp_name}_time_loglog_comparison.png")
    plt.savefig(time_plot_path, bbox_inches='tight')
    print(f"✅ Log-Log Time 그래프 저장 완료: {time_plot_path}")

if __name__ == '__main__':
    parser = ArgumentParser(description="Compare Grid vs Clique Ensembles")
    parser.add_argument('-d', '--results_dir', type=str, required=True, help="결과 CSV가 들어있는 폴더 경로")
    parser.add_argument('-e', '--exp_name', type=str, required=True, help="실험 이름 (예: fixed_radius 또는 constant_density)")
    parser.add_argument('-s', '--sizes', nargs='+', type=int, required=True, help="N 사이즈 목록 (예: 500 1000 2000)")
    
    args = parser.parse_args()
    compare_and_plot(args.results_dir, args.exp_name, args.sizes)