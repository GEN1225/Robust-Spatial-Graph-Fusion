# -*- coding: utf-8 -*-
"""
Experiment 5: Graph Structure Analysis
=======================================
Analyzes structural differences between TKG and EKG to verify information complementarity.

Metrics:
- Jaccard similarity (edge overlap)
- Degree distribution differences
- Neighbor overlap
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_utils import load_real_data, create_spatial_graph, create_ecological_graph

K = 5


if __name__ == '__main__':
    print("="*80)
    print("Experiment 5: Graph Structure Analysis")
    print("="*80)

    print("\n1. Loading real data...")
    features, labels, coords, metadata = load_real_data()
    print(f"   Samples: {len(labels)}")

    print("\n2. Building graph structures...")
    edge_index_tkg = create_spatial_graph(coords, k=K)
    edge_index_ekg = create_ecological_graph(features, k=K)

    print(f"   TKG edges: {edge_index_tkg.shape[1]}")
    print(f"   EKG edges: {edge_index_ekg.shape[1]}")

    tkg_edges = set(tuple(e) for e in edge_index_tkg.t().numpy().tolist())
    ekg_edges = set(tuple(e) for e in edge_index_ekg.t().numpy().tolist())

    print("\n3. Computing graph structural differences...")

    intersection = len(tkg_edges & ekg_edges)
    union = len(tkg_edges | ekg_edges)
    jaccard = intersection / union if union > 0 else 0

    print(f"\n[Edge Overlap Analysis]")
    print(f"   TKG edges: {len(tkg_edges)}")
    print(f"   EKG edges: {len(ekg_edges)}")
    print(f"   Shared edges: {intersection}")
    print(f"   Jaccard similarity: {jaccard:.4f} ({jaccard*100:.2f}%)")
    print(f"   Structural difference: {1-jaccard:.4f} ({(1-jaccard)*100:.2f}%)")

    print("\n[Degree Distribution Analysis]")

    n_nodes = len(labels)
    tkg_degrees = np.zeros(n_nodes)
    ekg_degrees = np.zeros(n_nodes)

    for i, j in tkg_edges:
        tkg_degrees[i] += 1

    for i, j in ekg_edges:
        ekg_degrees[i] += 1

    print(f"   TKG degrees: mean={tkg_degrees.mean():.2f}, std={tkg_degrees.std():.2f}, range=[{tkg_degrees.min():.0f}, {tkg_degrees.max():.0f}]")
    print(f"   EKG degrees: mean={ekg_degrees.mean():.2f}, std={ekg_degrees.std():.2f}, range=[{ekg_degrees.min():.0f}, {ekg_degrees.max():.0f}]")

    degree_corr = np.corrcoef(tkg_degrees, ekg_degrees)[0, 1]
    print(f"   Degree correlation: {degree_corr:.4f}")

    print("\n[Neighbor Overlap Analysis]")

    tkg_neighbors = {i: set() for i in range(n_nodes)}
    ekg_neighbors = {i: set() for i in range(n_nodes)}

    for i, j in tkg_edges:
        tkg_neighbors[i].add(j)

    for i, j in ekg_edges:
        ekg_neighbors[i].add(j)

    neighbor_overlaps = []
    for i in range(n_nodes):
        tkg_n = tkg_neighbors[i]
        ekg_n = ekg_neighbors[i]
        if len(tkg_n) > 0 and len(ekg_n) > 0:
            overlap = len(tkg_n & ekg_n) / len(tkg_n | ekg_n)
            neighbor_overlaps.append(overlap)

    neighbor_overlaps = np.array(neighbor_overlaps)
    print(f"   Mean neighbor overlap: {neighbor_overlaps.mean():.4f} ({neighbor_overlaps.mean()*100:.2f}%)")
    print(f"   Neighbor overlap std: {neighbor_overlaps.std():.4f}")
    print(f"   Fully non-overlapping nodes: {(neighbor_overlaps == 0).mean()*100:.1f}%")

    print("\n[Neighbor Overlap by Biomass Range]")

    ranges = [(0, 3, 'Low (<3)'), (3, 6, 'Low-mid (3-6)'), (6, 9, 'Mid (6-9)'),
              (9, 12, 'Mid-high (9-12)'), (12, 20, 'High (>12)')]

    for low, high, name in ranges:
        mask = (labels >= low) & (labels < high)
        if mask.sum() < 10:
            continue

        range_overlaps = neighbor_overlaps[mask[:len(neighbor_overlaps)]]
        if len(range_overlaps) > 0:
            print(f"   {name}: neighbor overlap={range_overlaps.mean():.4f} (n={mask.sum()})")

    print("\n4. Saving results...")

    results = {
        'metric': [
            'TKG_edges', 'EKG_edges', 'Shared_edges',
            'Jaccard_similarity', 'Structure_difference',
            'TKG_degree_mean', 'EKG_degree_mean', 'Degree_correlation',
            'Neighbor_overlap_mean', 'Neighbor_overlap_std'
        ],
        'value': [
            len(tkg_edges), len(ekg_edges), intersection,
            jaccard, 1 - jaccard,
            tkg_degrees.mean(), ekg_degrees.mean(), degree_corr,
            neighbor_overlaps.mean(), neighbor_overlaps.std()
        ]
    }

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, 'exp5_graph_structure.csv'), index=False)

    print(f"\nResults saved to: results/exp5_graph_structure.csv")

    print("\n" + "="*80)
    print("Conclusion")
    print("="*80)
    print(f"""
The structural difference between TKG and EKG is {(1-jaccard)*100:.1f}%, indicating:
- The two graphs capture nearly completely different relationships
- TKG is based on spatial proximity, EKG on feature similarity
- This high divergence provides the theoretical basis for information fusion

Neighbor overlap is only {neighbor_overlaps.mean()*100:.1f}%, further confirming:
- The same node connects to different neighbors in the two graphs
- Fusion can integrate information from different perspectives
""")

    print("="*80)
    print("Experiment complete!")
    print("="*80)
