import os
import re

def clean_result_filenames(results_dir='../results'):
    print(f"🧹 파일명 일괄 정리를 시작합니다... (대상: {results_dir})\n")
    
    count = 0
    for filename in os.listdir(results_dir):
        if not filename.endswith('.csv'): 
            continue

        new_name = filename

        # 1. 쓸데없는 꼬리표 및 접두사 완전 제거
        new_name = new_name.replace('_optionRevisedTest_ver2', '')
        new_name = new_name.replace('scaling_test_', '')
        new_name = new_name.replace('fixed_radius_', '')

        # 2. 용어 통일: ensemble -> heuristic
        new_name = new_name.replace('clique_ensemble', 'clique_heuristic')
        new_name = new_name.replace('grid_ensemble', 'grid_heuristic')

        # 2. constant_density_ 위치 변경 (맨 앞 -> 맨 뒤)
        # 예: constant_density_2000_grid_results.csv -> 2000_grid_density_results.csv
        match = re.match(r'^constant_density_(\d+_.*?)_results\.csv$', new_name)
        if match:
            new_name = f"{match.group(1)}_density_results.csv"

        # 파일명이 변경된 경우에만 이름 바꾸기 실행
        if filename != new_name:
            old_path = os.path.join(results_dir, filename)
            new_path = os.path.join(results_dir, new_name)
            os.rename(old_path, new_path)
            print(f"✅ 변경됨: {filename}\n   -> {new_name}\n")
            count += 1
            
    print(f"🎉 총 {count}개의 파일명이 깔끔하게 세탁되었습니다!")

if __name__ == "__main__":
    clean_result_filenames()