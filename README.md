# Polyhedral Analysis and Competitive Matheuristic Framework for the MWIDSP on Unit Disk Graphs

This repository contains the source code, instances, and experimental results for the paper: **"Polyhedral Analysis and Competitive Matheuristic Framework for the MWIDSP on Unit Disk Graphs"**.

This project explicitly integrates computational geometry into an Operations Research framework to solve the Minimum Weight Independent Dominating Set Problem (MWIDSP) on Unit Disk Graphs (UDGs). By replacing dense algebraic constraints with a topology-aware **Competitive Matheuristic Pipeline**, this framework successfully circumvents the fractional degeneracy typically encountered by exact solvers.

##  Repository Contents

* `instances/` - Contains the UDG datasets generated with 2D spatial coordinates (EG, NG, and VG schemes).
* `src/` - Contains the Python source code for all geometric heuristics, metaheuristics, and Gurobi exact solver models.
* `results/` - Contains the raw CSV output data from the experiments.
* `results/compare/` - Contains the generated Critical Difference (CD) diagrams and sensitivity analysis plots.

##  Requirements

The code is written in Python 3.9+ and requires the following packages:

**Core Optimization:**
* `gurobipy` (Gurobi Optimizer with a valid license)
* `networkx` (Graph modeling)

**Data Analysis & Statistics:**
* `pandas`
* `scipy`
* `scikit-posthocs`
* `matplotlib`

You can install the open-source dependencies using:
```bash
pip install networkx pandas scipy scikit-posthocs matplotlib
```
## Usage
All scripts utilize argparse for flexible execution. Below are examples of how to run the key components of the pipeline. Use the -h flag with any script to see all available arguments.

1. **Instance Generation** <br>
Generate random UDG instances with specific weight schemes (EG, NG, VG) and 2D coordinates:

```bash
python src/generate_udg_instances.py -n 500 -r 0.14 -i 10 -o ./instances/my_udg_with_pos
```

2. **Executing the Competitive Matheuristic Ensemble** <br>
Run the dual-engine pipeline (Shift_Full + LP_Rounding) on the generated instances:

```Bash
python src/heuristic_competitive_ensemble.py -i ./instances/my_udg_with_pos -s 500_r0c14 -o ./results
```
3. **Executing Gurobi Exact Solvers** <br>
Run the Baseline Gurobi solver (No warm-start) with a 1500-second time limit:

```Bash
python src/exact_gurobi_baseline.py -i ./instances/my_udg_with_pos -s 500_r0c14 -o ./results -t 1500
```
Run the Hybrid Gurobi solver (injecting Multiple MIP Starts from the ensemble):

```Bash
python src/exact_gurobi_hybrid.py -i ./instances/my_udg_with_pos -s 500_r0c14 -o ./results -t 1500
```
4. **Statistical Analysis and Plotting** <br>
Merge the heuristic results and generate the Critical Difference (CD) diagram:

```Bash
# Step 1: Merge individual CSV results into a matrix
python src/analysis_prepare_stats.py -i ./results -s 500_r0c14 -o ./results

# Step 2: Run Friedman & Nemenyi tests and generate the CD plot
python src/analysis_critical_difference.py -i ./results/500_r0c14_merged_results_for_stats.csv -o ./results/compare
```