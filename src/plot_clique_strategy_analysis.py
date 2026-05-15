import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_strategy_logs(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 텍스트 파일에서 인스턴스 이름과 4개 전략의 Cost 추출
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
        # 인스턴스 이름으로 가중치 환경(VG, EG, NG) 판별
        env = 'VG (Node-Heavy)' if 'nw1000_ew10' in inst else \
              'EG (Edge-Heavy)' if 'nw10_ew1000' in inst else 'NG (Neutral)'
        
        costs = {'size': float(m[1]), 'cost': float(m[2]), 'hub': float(m[3]), 'inverse_size': float(m[4])}
        best_cost = min(costs.values())
        best_strat = min(costs, key=costs.get)
        
        row = {'instance': inst, 'env': env, 'winner': best_strat}
        # 1등 전략과의 오차율(Gap %) 계산
        for k, v in costs.items():
            row[f'{k}_gap'] = (v - best_cost) / best_cost * 100
        data.append(row)
        
    return pd.DataFrame(data)

if __name__ == '__main__':
    # 타겟 경로 하드코딩
    results_dir = '../results'
    
    # ../results 폴더 안의 모든 strategy_logs.txt 파일 찾기
    log_files = [f for f in os.listdir(results_dir) if f.endswith('strategy_logs.txt')]
    
    if not log_files:
        print(f"[{results_dir}] 폴더 안에 분석할 로그 파일이 없습니다.")
        exit()
        
    print(f"총 {len(log_files)}개의 로그 파일을 분석 중입니다...")
    
    # 전체 데이터 병합
    df = pd.concat([parse_strategy_logs(os.path.join(results_dir, f)) for f in log_files], ignore_index=True)

    # ---------------------------------------------------------
    # 1. 환경별 승률 카운트 바 차트
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='env', hue='winner', palette='viridis')
    plt.title('Winning Strategy by Weight Environment', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Wins', fontsize=12)
    plt.xlabel('Environment', fontsize=12)
    plt.legend(title='Strategy', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    wins_plot_path = os.path.join(results_dir, 'strategy_wins_by_env.png')
    plt.savefig(wins_plot_path, bbox_inches='tight')
    print(f"✅ 승률 바 차트 저장 완료: {wins_plot_path}")

    # ---------------------------------------------------------
    # 2. 환경별 전략 오차율(Gap %) 히트맵
    # ---------------------------------------------------------
    gap_cols = ['size_gap', 'cost_gap', 'hub_gap', 'inverse_size_gap']
    heatmap_data = df.groupby('env')[gap_cols].mean()
    heatmap_data.columns = ['Size', 'Cost', 'Hub', 'Inverse Size'] # 보기 좋게 이름 변경

    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='Reds', cbar_kws={'label': 'Average Gap to Best (%)'})
    plt.title('Strategy Inefficiency (Gap %) by Environment', fontsize=16, fontweight='bold')
    plt.ylabel('Environment', fontsize=12)
    plt.xlabel('Strategy', fontsize=12)
    
    heatmap_plot_path = os.path.join(results_dir, 'strategy_gap_heatmap.png')
    plt.savefig(heatmap_plot_path, bbox_inches='tight')
    print(f"✅ 오차율 히트맵 저장 완료: {heatmap_plot_path}")