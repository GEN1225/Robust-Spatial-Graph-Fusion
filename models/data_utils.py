# -*- coding: utf-8 -*-
"""
Data loading and graph construction utilities.
Ensures consistent data loading across all experiments.

NOTE: This file provides public data utilities. The full graph
construction pipeline with proprietary optimization strategies
is available upon reasonable request for academic collaboration.
"""

import os
import numpy as np
import pickle
import torch
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def load_real_data():
    features = np.load(os.path.join(DATA_DIR, 'aligned_features_multiyear.npy'))
    labels = np.load(os.path.join(DATA_DIR, 'aligned_labels_multiyear.npy'))

    with open(os.path.join(DATA_DIR, 'aligned_metadata_multiyear.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    coords = metadata['coords']

    return features, labels, coords, metadata


def create_spatial_graph(coords, k=5):
    """Create kNN graph based on geographic coordinates (TKG)."""
    coords_array = np.array([[c[0], c[1]] for c in coords])
    distances = pairwise_distances(coords_array)

    edge_src = []
    edge_dst = []
    for i in range(len(coords)):
        neighbors = np.argsort(distances[i])[1:k+1]
        for j in neighbors:
            edge_src.append(i)
            edge_dst.append(j)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    return edge_index


def create_knowledge_graph(features, k=5, sim_threshold=0.7):
    """Create kNN graph based on feature similarity (EKG)."""
    sim_matrix = cosine_similarity(features)

    edge_src = []
    edge_dst = []
    for i in range(len(features)):
        sim_scores = sim_matrix[i].copy()
        sim_scores[i] = -1
        neighbors = np.argsort(sim_scores)[::-1][:k]

        for j in neighbors:
            if sim_scores[j] >= sim_threshold:
                edge_src.append(i)
                edge_dst.append(j)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    return edge_index


def create_ecological_graph(features, k=5, sim_threshold=0.7):
    """
    Create ecological knowledge graph (EKG) from feature similarity.

    Core innovation: proprietary feature selection and graph construction
    strategy for ecological knowledge embedding.
    Full implementation is proprietary — contact authors for collaboration.
    """
    raise NotImplementedError(
        "Ecological graph construction algorithm is proprietary. "
        "Please contact the authors for academic collaboration."
    )


def normalize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)
