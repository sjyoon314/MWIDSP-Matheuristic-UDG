"""
Data Preparation Script for Statistical Analysis.
Merges individual cost results from all heuristic algorithms into a single 
comprehensive matrix. This format is strictly required to execute the Friedman 
test and Nemenyi post-hoc analysis for the Critical Difference (CD) diagram.
"""

import os
import pandas as pd
from functools import reduce
from argparse import ArgumentParser

def merge_results_for_stats(in_dir_path, instances_subset, out_dir_path):
    # Map algorithm names (column headers) to their respective result file suffixes
    # These match the exact output filenames defined in the refactored scripts
    files_to_merge = {
        'GREEDY-NEW': f"{instances_subset}_baseline_results.csv",
        'Cross_LS': f"{instances_subset}_cross_ls_results.csv",
        'Shift_Full': f"{instances_subset}_shift_full_results.csv",
        'LP_Rounding': f"{instances_subset}_lp_rounding_results.csv",
        'SA_smart': f"{instances_subset}_sa_smart_results.csv",
        'GRASP': f"{instances_subset}_grasp_baseline_1500s_results.csv",
        'Competitive_Ensemble': f"{instances_subset}_competitive_ensemble_results.csv"
    }
    
    dataframes = []
    
    for algo_name, file_name in files_to_merge.items():
        file_path = os.path.join(in_dir_path, file_name)
        try:
            df = pd.read_csv(file_path, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            
            # Extract only 'instance' and 'cost' columns
            if 'cost' not in df.columns:
                print(f"Warning: 'cost' column missing in {file_name}. Skipping.")
                continue
                
            df_subset = df[['instance', 'cost']].copy()
            # Rename the cost column to the algorithm's name for clarity in the merged table
            df_subset.rename(columns={'cost': algo_name}, inplace=True)
            
            dataframes.append(df_subset)
            print(f"[{algo_name}] Data loaded successfully: {len(df_subset)} rows.")
            
        except FileNotFoundError:
            print(f"Warning: File {file_name} not found. Skipping this algorithm.")
        except Exception as e:
            print(f"Error processing {algo_name}: {e}")

    if not dataframes:
        print("No data available to merge. Please check the input directory.")
        return

    # Merge all dataframes on the 'instance' column (Inner Join)
    # This ensures only instances evaluated by ALL algorithms are included in the statistics
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='instance'), dataframes)
    
    os.makedirs(out_dir_path, exist_ok=True)
    out_file = os.path.join(out_dir_path, f"{instances_subset}_merged_results_for_stats.csv")
    
    merged_df.to_csv(out_file, index=False)
    print(f"\nSuccessfully merged data and saved to: {out_file}")
    print(f"Final matrix size: {merged_df.shape[0]} instances x {merged_df.shape[1]} columns")

if __name__ == "__main__":
    parser = ArgumentParser(description="Merge heuristic results for Critical Difference (CD) analysis.")
    parser.add_argument('-i', '--in_dir_path', type=str, required=True, help="Directory containing the individual CSV results")
    parser.add_argument('-s', '--instances_subset', type=str, required=True, help="Instance prefix (e.g., 500_r0c14)")
    parser.add_argument('-o', '--out_dir_path', type=str, required=True, help="Directory to save the merged matrix CSV")
    args = parser.parse_args()
    
    merge_results_for_stats(args.in_dir_path, args.instances_subset, args.out_dir_path)