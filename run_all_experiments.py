# -*- coding: utf-8 -*-
"""
Run All Experiments
===================
One-click runner for all experiments with a summary report.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

EXPERIMENTS = [
    ('exp1_segmentation_ablation.py', 'Segmented Modeling Ablation'),
    ('exp2_tkg_ekg_ablation.py', 'TKG/EKG Ablation'),
    ('exp3_boundary_robustness.py', 'Boundary Region Robustness Test'),
    ('exp4_spatial_generalization.py', 'Spatial Generalization Test'),
    ('exp5_graph_structure_analysis.py', 'Graph Structure Analysis'),
]

def run_experiment(script_name, description):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print('='*80)

    script_path = os.path.join(os.path.dirname(__file__), 'experiments', script_name)

    start_time = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True
    )
    elapsed = time.time() - start_time

    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    print(f"\nStatus: {status}, Elapsed: {elapsed:.1f}s")

    return result.returncode == 0, elapsed


def generate_summary_report():
    """Generate a summary report."""
    import pandas as pd

    results_dir = os.path.join(os.path.dirname(__file__), 'results')

    report = []
    report.append("="*80)
    report.append("Experiment Results Summary Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*80)

    # Experiment 1
    try:
        df = pd.read_csv(os.path.join(results_dir, 'exp1_summary.csv'))
        report.append("\n[Experiment 1: Segmented Modeling Ablation]")
        report.append(f"  Non-segmented baseline R2: {df['Baseline_R2_mean'].values[0]:.4f} +/- {df['Baseline_R2_std'].values[0]:.4f}")
        report.append(f"  Segmented model R2:        {df['Segmented_R2_mean'].values[0]:.4f} +/- {df['Segmented_R2_std'].values[0]:.4f}")
        report.append(f"  Performance improvement:    +{df['Improvement_pct'].values[0]:.1f}%")
    except Exception as e:
        report.append(f"\n[Experiment 1] Failed to read: {e}")

    # Experiment 2
    try:
        df = pd.read_csv(os.path.join(results_dir, 'exp2_tkg_ekg_ablation.csv'))
        report.append("\n[Experiment 2: TKG/EKG Ablation]")
        report.append(f"  TKG-only R2:  {df['TKG_r2'].mean():.4f} +/- {df['TKG_r2'].std():.4f}")
        report.append(f"  EKG-only R2:  {df['EKG_r2'].mean():.4f} +/- {df['EKG_r2'].std():.4f}")
        report.append(f"  Fusion R2:    {df['Fusion_r2'].mean():.4f} +/- {df['Fusion_r2'].std():.4f}")
    except Exception as e:
        report.append(f"\n[Experiment 2] Failed to read: {e}")

    # Experiment 3
    try:
        df = pd.read_csv(os.path.join(results_dir, 'exp3_summary.csv'))
        report.append("\n[Experiment 3: Boundary Region Robustness]")
        for _, row in df.iterrows():
            report.append(f"  {row['Model']}: BoundaryMAE={row['Boundary_MAE_mean']:.4f}, ErrorStd={row['Boundary_ErrorStd_mean']:.4f}")
    except Exception as e:
        report.append(f"\n[Experiment 3] Failed to read: {e}")

    # Experiment 4
    try:
        df = pd.read_csv(os.path.join(results_dir, 'exp4_summary.csv'))
        report.append("\n[Experiment 4: Spatial Generalization]")
        for _, row in df.iterrows():
            report.append(f"  {row['Model']}: R2={row['R2_mean']:.4f} +/- {row['R2_std']:.4f}, CV={row['CV_pct']:.2f}%")
    except Exception as e:
        report.append(f"\n[Experiment 4] Failed to read: {e}")

    # Experiment 5
    try:
        df = pd.read_csv(os.path.join(results_dir, 'exp5_graph_structure.csv'))
        report.append("\n[Experiment 5: Graph Structure Analysis]")
        for _, row in df.iterrows():
            report.append(f"  {row['metric']}: {row['value']:.4f}")
    except Exception as e:
        report.append(f"\n[Experiment 5] Failed to read: {e}")

    report.append("\n" + "="*80)

    report_text = "\n".join(report)
    with open(os.path.join(results_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: results/summary_report.txt")


if __name__ == '__main__':
    print("="*80)
    print("Starting All Experiments")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    total_start = time.time()
    results = []

    for script, desc in EXPERIMENTS:
        success, elapsed = run_experiment(script, desc)
        results.append((desc, success, elapsed))

    total_elapsed = time.time() - total_start

    print("\n" + "="*80)
    print("Experiment Run Summary")
    print("="*80)

    for desc, success, elapsed in results:
        status = "PASS" if success else "FAIL"
        print(f"  {status} {desc}: {elapsed:.1f}s")

    print(f"\nTotal elapsed: {total_elapsed/60:.1f} min")

    print("\nGenerating summary report...")
    generate_summary_report()

    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)
