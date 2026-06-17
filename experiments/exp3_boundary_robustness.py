# -*- coding: utf-8 -*-
"""
Experiment 3: Boundary Region Robustness Test
===============================================
Tests model robustness in the mechanism boundary region (around tau, 5-7 kg/m2).
This is a key experiment for demonstrating the value of fusion.

Expected result: Fusion has lower error variance in boundary regions, making it more robust.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn_models import TKGOnlyModel, EKGOnlyModel, FusionModel
from models.data_utils import load_real_data, create_spatial_graph, create_ecological_graph
from models.trainer import train_segmented_model, compute_boundary_metrics

RANDOM_SEED = 42
N_RUNS = 10
EPOCHS = 300
HIDDEN_DIM = 64
DROPOUT = 0.3
LR = 0.005
WEIGHT_DECAY = 1e-4
K = 5
THRESHOLD = 6.0

BOUNDARY_LOW = 5.0
BOUNDARY_HIGH = 7.0

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    print("="*80)
    print("Experiment 3: Boundary Region Robustness Test")
    print(f"Boundary region: {BOUNDARY_LOW} - {BOUNDARY_HIGH} kg/m2")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("\n1. Loading real data...")
    features, labels, coords, metadata = load_real_data()
    print(f"   Samples: {len(labels)}")

    boundary_mask = (labels >= BOUNDARY_LOW) & (labels <= BOUNDARY_HIGH)
    n_boundary = boundary_mask.sum()
    print(f"   Boundary samples: {n_boundary} ({n_boundary/len(labels)*100:.1f}%)")

    print("\n2. Building graph structures...")
    edge_index_tkg = create_spatial_graph(coords, k=K)
    edge_index_ekg = create_ecological_graph(features, k=K)

    print(f"\n3. Running {N_RUNS} experiment trials...")

    results = []

    for run in range(N_RUNS):
        seed = RANDOM_SEED + run
        set_seed(seed)

        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)

        print(f"\n   Run {run+1}/{N_RUNS} (seed={seed})...")

        for model_name, model_class, is_fusion, edge_idx, model_kwargs in [
            ('TKG', TKGOnlyModel, False, edge_index_tkg,
             {'input_dim': features.shape[1], 'hidden_dim': HIDDEN_DIM, 'heads': 2, 'dropout': DROPOUT}),
            ('EKG', EKGOnlyModel, False, edge_index_ekg,
             {'input_dim': features.shape[1], 'hidden_dim': HIDDEN_DIM, 'dropout': DROPOUT}),
            ('Fusion', FusionModel, True, edge_index_tkg,
             {'input_dim': features.shape[1], 'hidden_dim': HIDDEN_DIM, 'heads': 2, 'dropout': DROPOUT})
        ]:
            edge_ekg = edge_index_ekg if is_fusion else None

            metrics, predictions = train_segmented_model(
                model_class, features, labels, edge_idx, train_idx, test_idx,
                device, threshold=THRESHOLD, is_fusion=is_fusion, edge_index_ekg=edge_ekg,
                epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, **model_kwargs
            )

            boundary_metrics = compute_boundary_metrics(
                predictions, labels, test_idx, BOUNDARY_LOW, BOUNDARY_HIGH
            )

            if boundary_metrics:
                results.append({
                    'run': run + 1,
                    'model': model_name,
                    'overall_r2': metrics['r2'],
                    'overall_mae': metrics['mae'],
                    'boundary_n': boundary_metrics['n_samples'],
                    'boundary_mae': boundary_metrics['mae'],
                    'boundary_rmse': boundary_metrics['rmse'],
                    'boundary_error_std': boundary_metrics['error_std'],
                })

                print(f"      {model_name}: R2={metrics['r2']:.4f}, BoundaryMAE={boundary_metrics['mae']:.4f}, ErrorStd={boundary_metrics['error_std']:.4f}")

    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)

    df = pd.DataFrame(results)

    print("\n[Overall Performance]")
    print("-"*60)
    for model in ['TKG', 'EKG', 'Fusion']:
        model_df = df[df['model'] == model]
        print(f"{model}: R2={model_df['overall_r2'].mean():.4f}+/-{model_df['overall_r2'].std():.4f}")

    print("\n[Boundary Robustness] (Key Metrics)")
    print("-"*60)
    print(f"{'Model':<10} {'BoundaryMAE':<15} {'BoundaryRMSE':<15} {'ErrorStd':<15}")
    print("-"*60)

    for model in ['TKG', 'EKG', 'Fusion']:
        model_df = df[df['model'] == model]
        mae = f"{model_df['boundary_mae'].mean():.4f}+/-{model_df['boundary_mae'].std():.4f}"
        rmse = f"{model_df['boundary_rmse'].mean():.4f}+/-{model_df['boundary_rmse'].std():.4f}"
        err_std = f"{model_df['boundary_error_std'].mean():.4f}+/-{model_df['boundary_error_std'].std():.4f}"
        print(f"{model:<10} {mae:<15} {rmse:<15} {err_std:<15}")

    print("\n[Statistical Tests]")
    print("-"*60)

    tkg_err_std = df[df['model'] == 'TKG']['boundary_error_std'].values
    ekg_err_std = df[df['model'] == 'EKG']['boundary_error_std'].values
    fusion_err_std = df[df['model'] == 'Fusion']['boundary_error_std'].values

    # Fusion vs TKG
    t_stat, p_val = stats.ttest_rel(fusion_err_std, tkg_err_std)
    print(f"Fusion vs TKG (boundary error std): t={t_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.05:
        winner = "Fusion is more robust" if fusion_err_std.mean() < tkg_err_std.mean() else "TKG is more robust"
        print(f"  -> Significant difference (p<0.05): {winner}")

    # Fusion vs EKG
    t_stat, p_val = stats.ttest_rel(fusion_err_std, ekg_err_std)
    print(f"Fusion vs EKG (boundary error std): t={t_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.05:
        winner = "Fusion is more robust" if fusion_err_std.mean() < ekg_err_std.mean() else "EKG is more robust"
        print(f"  -> Significant difference (p<0.05): {winner}")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    df.to_csv(os.path.join(results_dir, 'exp3_boundary_robustness.csv'), index=False)
    print(f"\nResults saved to: results/exp3_boundary_robustness.csv")

    summary_data = []
    for model in ['TKG', 'EKG', 'Fusion']:
        model_df = df[df['model'] == model]
        summary_data.append({
            'Model': model,
            'Overall_R2_mean': model_df['overall_r2'].mean(),
            'Overall_R2_std': model_df['overall_r2'].std(),
            'Boundary_MAE_mean': model_df['boundary_mae'].mean(),
            'Boundary_MAE_std': model_df['boundary_mae'].std(),
            'Boundary_ErrorStd_mean': model_df['boundary_error_std'].mean(),
            'Boundary_ErrorStd_std': model_df['boundary_error_std'].std(),
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_dir, 'exp3_summary.csv'), index=False)

    print("\n" + "="*80)
    print("Experiment complete!")
    print("="*80)
