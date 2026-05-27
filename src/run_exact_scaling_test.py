import os
import subprocess

def generate_and_test(N, num_instances=10):
    radius = 0.14
    time_limit = 3.0 * N
    in_dir = "../instances/udg_fixed_radius"
    out_dir = "../results"
    instance_prefix = f"{N}_r0c14"
    exp_name = "scaling_test"
    
    # Use only a single score type for exact model evaluations (based on ablation conclusions)
    default_score_type = "node_only" 
    
    print(f"\n{'='*60}")
    print(f"4-Way Scaling Comparison Test: N={N} (Time Limit: {time_limit}s)")
    print(f"{'='*60}")
    
    os.makedirs(in_dir, exist_ok=True)
    
    # 1. Check if instances already exist (enforce underscore to avoid partial matches)
    safe_prefix = f"{instance_prefix}_"
    files_exist = any(f.startswith(safe_prefix) for f in os.listdir(in_dir))
    
    if files_exist:
        print(f"\n[1/5] Instances for N={N} already exist. Skipping generation.")
    else:
        print(f"\n[1/5] Generating {num_instances} instances for N={N}...")
        subprocess.run(["python", "generate_instances.py", "-n", str(N), "-r", str(radius), "-i", str(num_instances)])
    
    # 2. Run NEW2 Baseline
    print(f"\n[2/5] Running NEW2 Baseline (Default)...")
    subprocess.run(["python", "base_e_new2.py", "-i", in_dir, "-s", instance_prefix, "-o", out_dir])

    # 3. Run NEW2 + Grid Cuts
    print(f"\n[3/5] Running NEW2 + Grid Cuts (Lightweight cuts)...")
    subprocess.run(["python", "exact_grid_cut.py", "-i", in_dir, "-s", instance_prefix, "-o", out_dir])
    
    # 4. Run Grid Ensemble Warm-start Gurobi
    print(f"\n[4/5] Running Ensemble Warm-start (Primal cuts)...")
    subprocess.run([
        "python", "exact_gurobi_warm_tester.py", 
        "-i", in_dir, 
        "-s", instance_prefix, 
        "-o", out_dir, 
        "-w", "ensemble", 
        "-t", str(time_limit),
        "--score_type", default_score_type  # Hardcoded score type
    ])

    # 5. Run Max Clique V4 Gurobi
    print(f"\n[5/5] Running Max Clique V4 (Advanced formulation)...")
    subprocess.run([
        "python", "p4_phase2_maxclique_v4.py", 
        "-i", in_dir, 
        "-s", instance_prefix, 
        "-o", out_dir, 
        "-e", exp_name, 
        "-t", str(time_limit),
        "--score_type", default_score_type  # Hardcoded score type
    ])
    
    print(f"\n✅ [Success] Scaling test cycle completed for N={N}!")

if __name__ == "__main__":
    # Run continuously for N=200 to 400
    test_scales = [200, 250, 300, 350, 400]
    
    for n in test_scales:
        generate_and_test(N=n, num_instances=10)
        
    print("\n🎉 [Success] All scaling tests completed! Launching visualization script.")
    
    # Automatically execute visualization code
    scales_str = ",".join(map(str, test_scales))
    subprocess.run(["python", "plot_scaling_results.py", "--scales", scales_str])

    # Run the N=500 massive scale test separately after visualization
    print("\n[Info] Starting additional massive scale test for N=500...")
    generate_and_test(N=500, num_instances=10)