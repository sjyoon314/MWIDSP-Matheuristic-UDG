import os
import pandas as pd

def merge_and_save(file_mapping, out_filename, base_dir="../results"):
    """
    Reads files based on the provided file_mapping, merges them into a single matrix, and saves the result.
    """
    out_file = os.path.join(base_dir, out_filename)
    merged_df = None

    for algo_name, file_name in file_mapping.items():
        file_path = os.path.join(base_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"  [Warning] File not found (skipping): {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path, skipinitialspace=True)
        except Exception as e:
            print(f"  [Error] Failed to read {file_name}: {e}")
            continue
        
        # Standardize column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Use 'cost' for Heuristic and 'best_obj' for Exact algorithms
        target_col = None
        if 'cost' in df.columns:
            target_col = 'cost'
        elif 'best_obj' in df.columns:
            target_col = 'best_obj'
            
        if 'instance' not in df.columns or target_col is None:
            print(f"  [Warning] '{file_name}' is missing 'instance' or the target column ('cost'/'best_obj').")
            continue
            
        # Extract and format data
        df_subset = df[['instance', target_col]].copy()
        df_subset.rename(columns={target_col: algo_name}, inplace=True)
        df_subset['instance'] = df_subset['instance'].astype(str).str.strip()
        
        # Merge logic (Outer Join)
        if merged_df is None:
            merged_df = df_subset
        else:
            merged_df = pd.merge(merged_df, df_subset, on='instance', how='outer')

    if merged_df is None or merged_df.empty:
        print(f"  [Error] No data to merge. ({out_filename})")
        return None
    
    # Save results
    merged_df.to_csv(out_file, index=False)
    print(f"  [Success] Matrix saved to: {out_file}")
    return merged_df

def prepare_statistical_matrix():
    BASE_DIR = "../results"
    
    # 1. Heuristic algorithms mapping
    heuristic_mapping = {
        "GREEDY-NEW": "500_r0c14_greedynew_results.csv",
        "Cross_LS": "500_r0c14_cross_ls_results.csv",
        "Shift_Full": "500_r0c14_shift_full_results.csv",
        "Grid_Ensemble": "500_r0c14_grid_heuristic_results.csv",
        "Clique_Ensemble": "500_r0c14_clique_heuristic_results.csv",
        "LP_Rounding": "500_r0c14_lp_rounding_results.csv",
        "GRASP": "500_r0c14_grasp_results.csv",
        #"SA_smart": "500_r0c14_sa_smart_results.csv",
        "VNS": "500_r0c14_vns_results.csv"
    }

    # 2. Exact algorithms mapping (Update with actual filenames if necessary)
    exact_mapping = {
        "Exact_Clique_Cut": "500_r0c14_maxclique_v4_results.csv",
        "NEW-2": "500_r0c14_new2_results.csv",
        "Exact_Warm_Grid": "500_r0c14_gurobi_warm_ensemble_results.csv",
        "Exact_Grid_Cut": "500_r0c14_grid_cut_results.csv",
        "Exact_Grid_SOS1": "500_r0c14_grid_sos1_scaling_test_results.csv"
    }

    # 3. Combined mapping (Heuristic + Exact)
    all_mapping = {**heuristic_mapping, **exact_mapping}

    # Generate and save each matrix
    print("\n[1/3] Generating Heuristic-only statistical matrix...")
    merge_and_save(heuristic_mapping, "merged_stats_matrix_heuristic.csv", BASE_DIR)

    print("\n[2/3] Generating Exact-only statistical matrix...")
    merge_and_save(exact_mapping, "merged_stats_matrix_exact.csv", BASE_DIR)

    print("\n[3/3] Generating Combined (Heuristic + Exact) statistical matrix...")
    merge_and_save(all_mapping, "merged_stats_matrix_all.csv", BASE_DIR)

if __name__ == "__main__":
    prepare_statistical_matrix()