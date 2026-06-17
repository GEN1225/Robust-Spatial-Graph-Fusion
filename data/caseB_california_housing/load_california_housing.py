# -*- coding: utf-8 -*-
"""
California Housing dataset loader.

Source: sklearn.datasets.fetch_california_housing
Reference: Pace, R. Kelley, and Ronald Barry. "Sparse spatial autoregressions."
           Statistics & Probability Letters 33.3 (1997): 291-297.
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


def load_california_housing():
    """
    Load the California Housing dataset.

    Returns:
        features: (20640, 8) feature matrix
        labels: (20640,) target variable (median house value in $100k)
        feature_names: list of feature names
    """
    housing = fetch_california_housing()
    features = housing.data
    labels = housing.target
    feature_names = housing.feature_names

    return features, labels, feature_names


def load_with_coordinates():
    """
    Load California Housing with coordinates.

    Note: The sklearn built-in version does not include lat/lon coordinates.
    For spatial splitting, you need to obtain coordinate data separately.

    Returns:
        features: (20640, 8) feature matrix
        labels: (20640,) target variable
        feature_names: list of feature names
        coordinates: None (placeholder)
    """
    features, labels, feature_names = load_california_housing()

    # sklearn California Housing doesn't include coordinates.
    # For spatial OOD evaluation, you need to fetch them separately.
    coordinates = None

    return features, labels, feature_names, coordinates


if __name__ == "__main__":
    features, labels, feature_names = load_california_housing()

    print("=" * 60)
    print("California Housing Dataset")
    print("=" * 60)
    print(f"\nSamples: {features.shape[0]}")
    print(f"Features: {features.shape[1]}")
    print(f"\nFeature names: {feature_names}")
    print(f"\nFeature statistics:")
    print(f"  Mean: {features.mean(axis=0).round(3)}")
    print(f"  Std:  {features.std(axis=0).round(3)}")
    print(f"\nTarget (house value, $100k):")
    print(f"  Range: [{labels.min():.2f}, {labels.max():.2f}]")
    print(f"  Mean:  {labels.mean():.2f}")
    print(f"  Std:   {labels.std():.2f}")

    # Standardization example
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"\nAfter standardization:")
    print(f"  Mean: {features_scaled.mean(axis=0).round(10)}")
    print(f"  Std:  {features_scaled.std(axis=0).round(3)}")
