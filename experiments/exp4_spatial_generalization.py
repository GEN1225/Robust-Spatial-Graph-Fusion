# -*- coding: utf-8 -*-
"""
Experiment 4: Spatial Generalization Test
==========================================
Uses leave-one-region-out cross-validation to test spatial generalization.
Divides by latitude into 5 blocks and evaluates each as the held-out test set.

Expected result: Fusion shows smaller performance variance across regions.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn_models import TKGOnlyModel, EKGOnlyModel, FusionModel
from models.data_utils import load_real_data, create_spatial_graph, create_ecological_graph
from models.trainer import train_segmented_model

RANDOM_SEED = 42
EPOCHS = 300
HIDDEN_DIM = 64
DROPOUT = 0.3
LR = 0.005
WEIGHT_DECAY = 1e-4
K = 5
THRESHOLD = 6.0
N_BLOCKS = 5

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    print("="*80)
    print("Experiment 4: Spatial Generalization (Leave-One-Region-Out CV)")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    set_seed(RANDOM_SEED)

    print("\n1. Loading real data...")
    features, labels, coords, metadata = load_real_data()
    print(f"   Samples: {len(labels)}")

    print("\n2. Building graph structures...")
    edge_index_tkg = create_spatial_graph(coords, k=K)
    edge_index_ekg = create_ecological_graph(features, k=K)

    print(f"\n3. Dividing into {N_BLOCKS} spatial blocks by latitude...")
    lats = np.array([c[0] for c in coords])
    lat_bins = np.linspace(lats.min(), lats.max(), N_BLOCKS + 1)
    block_labels = np.digitize(lats, lat_bins[:-1]) - 1
    block_labels = np.clip(block_labels, 0, N_BLOCKS - 1)

    for b in range(N_BLOCKS):
        n_block = (block_labels == b).sum()
        lat_range = (lat_bins[b], lat_bins[b+1])
        print(f"   Block {b}: {n_block} samples, latitude [{lat_range[0]:.2f}, {lat_range[1]:.2f}]")

    print(f"\n4. Running leave-one-region-out cross-validation...")

    results = []

    for test_block in range(N_BLOCKS):
        test_mask = block_labels == test_block
        train_mask = ~test_mask

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        print(f"\n   Block {test_block} as test set (n={len(test_idx)})...")

        for model_name, model_class, is_fusion, edge_idx, model_kwargs in [
            ('TKG', TKGOnlyModel, False, edge_index_tkg,
             {'input_dim': features.shape[1], 'hidden_dim': HIDDEN_DIM, 'heads': 2, 'dropout': DROPOUT}),
            ('EKG', EKGOnlyModel, False, edge_index_ekg,
             {'input_dim': features.shape[1], 'hidden_dim': HIDDEN_DIM, 'dropout': DROPOUT}),
            ('Fusion', FusionModel, True, edge_index_tkg,
             {'input_dim': features.shape[1], 'hidden_dim': HIDDEN_DIM, 'heads': 2, 'dropout': DROPOUT})
        ]:
            edge_ekg = edge_index_ekg if is_fusion else None

            metrics, _ = train_segmented_model(
                model_class, features, labels, edge_idx, train_idx, test_idx,
                device, threshold=THRESHOLD, is_fusion=is_fusion, edge_index_ekg=edge_ekg,
                epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, **model_kwargs
            )

            results.append({
                'test_block': test_block,
                'model': model_name,
                'r2': metrics['r2'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'n_test': len(test_idx),
            })

            print(f"      {model_name}: R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")

    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)

    df = pd.DataFrame(results)

    print("\n[Block Performance]")
    print("-"*70)
    print(f"{'Block':<8} {'TKG R2':<12} {'EKG R2':<12} {'Fusion R2':<12} {'Best':<10}")
    print("-"*70)

    for block in range(N_BLOCKS):
        block_df = df[df['test_block'] == block]
        tkg_r2 = block_df[block_df['model'] == 'TKG']['r2'].values[0]
        ekg_r2 = block_df[block_df['model'] == 'EKG']['r2'].values[0]
        fusion_r2 = block_df[block_df['model'] == 'Fusion']['r2'].values[0]

        best = 'Fusion' if fusion_r2 >= max(tkg_r2, ekg_r2) else ('TKG' if tkg_r2 > ekg_r2 else 'EKG')
        print(f"{block:<8} {tkg_r2:<12.4f} {ekg_r2:<12.4f} {fusion_r2:<12.4f} {best:<10}")

    print("-"*70)

    print("\n[Spatial Generalization Stability]")
    print("-"*60)
    print(f"{'Model':<10} {'R2 Mean':<12} {'R2 Std':<12} {'CV (%)':<12}")
    print("-"*60)

    stability_data = []
    for model in ['TKG', 'EKG', 'Fusion']:
        model_df = df[df['model'] == model]
        r2_mean = model_df['r2'].mean()
        r2_std = model_df['r2'].std()
        cv = r2_std / r2_mean * 100 if r2_mean > 0 else float('inf')
        print(f"{model:<10} {r2_mean:<12.4f} {r2_std:<12.4f} {cv:<12.2f}")

        stability_data.append({
            'Model': model,
            'R2_mean': r2_mean,
            'R2_std': r2_std,
            'CV_pct': cv,
        })

    print("-"*60)

    stability_df = pd.DataFrame(stability_data)
    most_stable = stability_df.loc[stability_df['R2_std'].idxmin(), 'Model']
    print(f"\nMost stable model: {most_stable} (lowest R2 std)")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    df.to_csv(os.path.join(results_dir, 'exp4_spatial_generalization.csv'), index=False)
    stability_df.to_csv(os.path.join(results_dir, 'exp4_summary.csv'), index=False)

    print(f"\nResults saved to: results/exp4_spatial_generalization.csv")

    print("\n" + "="*80)
    print("Experiment complete!")
    print("="*80)
