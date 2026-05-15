import os
import subprocess

def generate_and_test(N, num_instances=10):
    radius = 0.14
    time_limit = 3.0 * N
    in_dir = "../instances/udg_fixed_radius"
    out_dir = "../results"
    instance_prefix = f"{N}_r0c14"
    exp_name = "scaling_test"
    
    print(f"\n{'='*60}")
    print(f"🚀 스케일링 4자 비교 테스트: N={N} (Time Limit: {time_limit}s)")
    print(f"{'='*60}")
    
    os.makedirs(in_dir, exist_ok=True)
    
    # 1. 인스턴스 존재 여부 체크 (언더바 포함)
    safe_prefix = f"{instance_prefix}_"
    files_exist = any(f.startswith(safe_prefix) for f in os.listdir(in_dir))
    
    if files_exist:
        print(f"\n[1/5] N={N} 인스턴스가 이미 존재합니다. 생성을 패스합니다.")
    else:
        print(f"\n[1/5] N={N} 인스턴스를 {num_instances}개씩 생성합니다...")
        subprocess.run(["python", "generate_instances.py", "-n", str(N), "-r", str(radius), "-i", str(num_instances)])
    
    # 2. NEW2 Baseline 실행
    print(f"\n[2/5] NEW2 Baseline (순정) 실행 중...")
    subprocess.run(["python", "base_e_new2.py", "-i", in_dir, "-s", instance_prefix, "-o", out_dir])

    # 3. NEW2 + Grid Cuts 실행 (새로 추가됨)
    print(f"\n[3/5] NEW2 + Grid Cuts (경량화 컷) 실행 중...")
    subprocess.run(["python", "exact_grid_cut.py", "-i", in_dir, "-s", instance_prefix, "-o", out_dir])
    
    # 4. 그리드 앙상블 웜스타트 Gurobi 실행
    print(f"\n[4/5] 앙상블 웜스타트 (Primal 컷) 실행 중...")
    subprocess.run(["python", "exact_gurobi_warm_tester.py", "-i", in_dir, "-s", instance_prefix, "-o", out_dir, "-w", "ensemble", "-t", str(time_limit)])

    # 5. Max Clique V4 Gurobi 실행
    print(f"\n[5/5] Max Clique V4 (끝판왕) 실행 중...")
    subprocess.run(["python", "p4_phase2_maxclique_v4.py", "-i", in_dir, "-s", instance_prefix, "-o", out_dir, "-e", exp_name, "-t", str(time_limit)])
    
    print(f"\n✅ N={N} 스케일 테스트 사이클 완료!")

if __name__ == "__main__":
    # N=200부터 400까지 논스톱으로 달립니다.
    test_scales = [200, 250, 300, 350, 400]
    
    for n in test_scales:
        generate_and_test(N=n, num_instances=10)
        
    print("\n🎉 모든 스케일링 테스트 완료! 시각화 코드를 실행합니다.")
    
    # 결과 시각화 코드 자동 호출
    scales_str = ",".join(map(str, test_scales))
    subprocess.run(["python", "plot_scaling_results.py", "--scales", scales_str])

    # 시각화 이후 500 스케일 단독 가동
    print("\n🔥 N=500 거대 스케일 테스트를 추가로 시작합니다...")
    generate_and_test(N=500, num_instances=10)