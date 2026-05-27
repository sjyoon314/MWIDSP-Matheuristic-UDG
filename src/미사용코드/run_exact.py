import os
import subprocess
import time
from datetime import datetime

def run_sequential_pipeline():
    # 경로 설정 (디렉토리 구조에 맞게 수정)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IN_DIR = os.path.join(SCRIPT_DIR, '..', 'instances', 'udg_fixed_radius')
    OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'results')

    # 공통 설정값 하드코딩
    PREFIX = "500_r0c14"
    EXP_NAME = "scaling_test"
    TIME_LIMIT = "1500.0"

    # 실행할 파이썬 파일명 (실제 프로젝트의 파일명과 다를 경우 이 부분만 수정)
    FILE_NEW2 = "base_e_new2.py"
    FILE_MAXCLIQUE = "exact_clique.py"
    FILE_SA_SMART = "metaheuristic_sa_smart.py"
    FILE_GRID_CUT = "exact_grid_cut.py"
    FILE_GRID = "exact_gurobi_warm_tester.py"
    FILE_GRID_SOS = "exact_grid_sos1.py"

    # 각 알고리즘별 맞춤형 인자 세팅 (더미 인자 없이 필요한 것만 주입)
    tasks = [
        
        {
            "name": "Grid_sos1 (Exact)",
            "cmd": [
                "python", FILE_GRID_SOS,
                "-i", IN_DIR,
                "-s", PREFIX,
                "-o", OUT_DIR,
                "-e", EXP_NAME,
                "-t", TIME_LIMIT
            ]
        },
        {
            "name": "Grid(Exact)",
            "cmd": [
                "python", FILE_GRID,
                "-i", IN_DIR,
                "-s", PREFIX,
                "-o", OUT_DIR,
                "-w", "ensemble",
                "-t", TIME_LIMIT

            ]
        },
        {
            "name": "NEW2 (Exact)",
            "cmd": [
                "python", FILE_NEW2,
                "-i", IN_DIR,
                "-s", PREFIX,
                "-o", OUT_DIR
            ]
        },
        {
            "name": "Max Clique V4 (Exact)",
            "cmd": [
                "python", FILE_MAXCLIQUE,
                "-i", IN_DIR,
                "-s", PREFIX,
                "-o", OUT_DIR,
                "-e", EXP_NAME,
                "-t", TIME_LIMIT
            ]
        },
        {
            "name": "SA Smart (Metaheuristic)",
            "cmd": [
                "python", FILE_SA_SMART,
                "-i", IN_DIR,       # SA 코드는 -e, -t를 받지 않으므로 제외
                "-s", PREFIX,
                "-o", OUT_DIR
            ]
        },
        { # 200
            "name": "new2 + Grid Cut (Exact)",
            "cmd": [
                "python", FILE_GRID_CUT,
                "-i", IN_DIR,
                "-s", "200_r0c14",
                "-o", OUT_DIR
            ]
        },
        {
            "name": "new2 + Grid Cut (Exact)",
            "cmd": [
                "python", FILE_GRID_CUT,
                "-i", IN_DIR,
                "-s", "250_r0c14",
                "-o", OUT_DIR
            ]
        },
        {
            "name": "new2 + Grid Cut (Exact)",
            "cmd": [
                "python", FILE_GRID_CUT,
                "-i", IN_DIR,
                "-s", "300_r0c14",
                "-o", OUT_DIR
            ]
        },
        {
            "name": "new2 + Grid Cut (Exact)",
            "cmd": [
                "python", FILE_GRID_CUT,
                "-i", IN_DIR,
                "-s", "350_r0c14",
                "-o", OUT_DIR
            ]
        },
        {
            "name": "new2 + Grid Cut (Exact)",
            "cmd": [
                "python", FILE_GRID_CUT,
                "-i", IN_DIR,
                "-s", "400_r0c14",
                "-o", OUT_DIR
            ]
        },
        {
            "name": "new2 + Grid Cut (Exact)",
            "cmd": [
                "python", FILE_GRID_CUT,
                "-i", IN_DIR,
                "-s", PREFIX,
                "-o", OUT_DIR
            ]
        }
    ]

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 N=500 순차 실행 파이프라인 시작\n" + "="*50)

    for task in tasks:
        script_name = task["cmd"][1]
        if not os.path.exists(script_name):
            print(f"\n⚠️ [경고] 파일이 존재하지 않습니다. 스킵합니다: {script_name}")
            continue

        print(f"\n▶️ 실행 중: {task['name']} ({script_name})")
        print(f"명령어: {' '.join(task['cmd'])}")
        
        start_time = time.time()
        try:
            # subprocess.run은 해당 프로세스가 완전히 종료될 때까지 대기(블로킹)합니다.
            subprocess.run(task["cmd"], check=True)
            elapsed = time.time() - start_time
            print(f"✅ 완료: {task['name']} (소요 시간: {elapsed:.2f}초)")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 에러 발생: {task['name']} 실행 중 문제 발생 (종료 코드: {e.returncode})")
            # 에러가 발생해도 파이프라인이 멈추지 않고 다음 스크립트로 넘어가도록 continue 처리
            continue

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🎉 모든 파이프라인 실행 종료")

if __name__ == "__main__":
    run_sequential_pipeline()