# -*- coding: utf-8 -*-
"""
Experiment 2: TKG/EKG Ablation
======================
Compares performance of TKG-only, EKG-only, and Fusion models.
All models use segmented modeling.

Expected result: All three achieve similar performance (~0.72), confirming segmentation as the key factor.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn_models import TKGOnlyModel, EKGOnlyModel, FusionModel
from models.data_utils import load_real_data, create_spatial_graph, create_ecological_graph
from models.trainer import train_segmented_model

RANDOM_SEED = 42
N_RUNS = 10
EPOCHS = 300
HIDDEN_DIM = 64
DROPOUT = 0.3
LR = 0.005
WEIGHT_DECAY = 1e-4
K = 5
THRESHOLD = 6.0

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    print("="*80)
    print("Experiment 2: TKG/EKG Ablation")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("\n1. Loading real data...")
    features, labels, coords, metadata = load_real_data()
    print(f"   Samples: {len(labels)}")
    print(f"   Feature dim: {features.shape[1]}")

    print("\n2. Building graph structures...")
    edge_index_tkg = create_spatial_graph(coords, k=K)
    edge_index_ekg = create_ecological_graph(features, k=K)
    print(f"   TKG edges: {edge_index_tkg.shape[1]}")
    print(f"   EKG edges: {edge_index_ekg.shape[1]}")

    print(f"\n3. Running {N_RUNS} experiment trials...")

    results = []

    for run in range(N_RUNS):
        seed = RANDOM_SEED + run
        set_seed(seed)

        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)

        print(f"\n   Run {run+1}/{N_RUNS} (seed={seed})...")

        run_results = {'run': run + 1, 'seed': seed}

        # TKG-only
        tkg_metrics, _ = train_segmented_model(
            TKGOnlyModel, features, labels, edge_index_tkg, train_idx, test_idx,
            device, threshold=THRESHOLD, is_fusion=False,
            epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
            input_dim=features.shape[1], hidden_dim=HIDDEN_DIM, heads=2, dropout=DROPOUT
        )
        run_results['TKG_r2'] = tkg_metrics['r2']
        run_results['TKG_mae'] = tkg_metrics['mae']
        print(f"      TKG-only: R2={tkg_metrics['r2']:.4f}")

        # EKG-only
        ekg_metrics, _ = train_segmented_model(
            EKGOnlyModel, features, labels, edge_index_ekg, train_idx, test_idx,
            device, threshold=THRESHOLD, is_fusion=False,
            epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
            input_dim=features.shape[1], hidden_dim=HIDDEN_DIM, dropout=DROPOUT
        )
        run_results['EKG_r2'] = ekg_metrics['r2']
        run_results['EKG_mae'] = ekg_metrics['mae']
        print(f"      EKG-only: R2={ekg_metrics['r2']:.4f}")

        # Fusion
        fusion_metrics, _ = train_segmented_model(
            FusionModel, features, labels, edge_index_tkg, train_idx, test_idx,
            device, threshold=THRESHOLD, is_fusion=True, edge_index_ekg=edge_index_ekg,
            epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
            input_dim=features.shape[1], hidden_dim=HIDDEN_DIM, heads=2, dropout=DROPOUT
        )
        run_results['Fusion_r2'] = fusion_metrics['r2']
        run_results['Fusion_mae'] = fusion_metrics['mae']
        print(f"      Fusion:   R2={fusion_metrics['r2']:.4f}")

        results.append(run_results)

    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)

    df = pd.DataFrame(results)

    print("\n[Model Performance Comparison] (Segmented, tau=6)")
    print("-"*60)
    print(f"{'Model':<12} {'R2 Mean':<12} {'R2 Std':<12} {'MAE Mean':<12}")
    print("-"*60)

    for model in ['TKG', 'EKG', 'Fusion']:
        r2_mean = df[f'{model}_r2'].mean()
        r2_std = df[f'{model}_r2'].std()
        mae_mean = df[f'{model}_mae'].mean()
        print(f"{model:<12} {r2_mean:<12.4f} {r2_std:<12.4f} {mae_mean:<12.4f}")

    print("-"*60)

    best_single = max(df['TKG_r2'].mean(), df['EKG_r2'].mean())
    fusion_gain = (df['Fusion_r2'].mean() - best_single) / best_single * 100

    print(f"\nFusion gain over best single graph: {fusion_gain:+.2f}%")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    df.to_csv(os.path.join(results_dir, 'exp2_tkg_ekg_ablation.csv'), index=False)
    print(f"\nResults saved to: results/exp2_tkg_ekg_ablation.csv")

    print("\n" + "="*80)
    print("Experiment complete!")
    print("="*80)
