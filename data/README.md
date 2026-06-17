# Data Directory

This directory contains datasets used in the experiments.

## Structure

```
data/
├── caseA_forest_biomass/        # Forest biomass prediction
│   ├── aligned_features_multiyear.npy    # Features (4000 samples, 12 dims)
│   ├── aligned_labels_multiyear.npy      # Labels
│   └── aligned_metadata_multiyear.pkl    # Metadata (coordinates, etc.)
│
├── caseB_california_housing/    # California housing prices
│   └── load_california_housing.py        # Data loader script
│
├── caseC_county_poverty/        # US county poverty
│   ├── county_poverty_features_final.csv  # Features (1749 samples, 9 dims)
│   └── county_poverty_final.csv           # Full dataset
│
└── caseD_eurosat/               # EuroSAT remote sensing
    ├── eurosat_features.npy     # Features (27000 samples, 512 dims)
    └── eurosat_labels.npy       # Labels
```

## Dataset Summary

| Dataset | Samples | Features | Task | Source |
|---------|---------|----------|------|--------|
| Case A: Forest Biomass | 4,000 | 12 | Regression | Remote sensing (NDVI + AGB + DEM) |
| Case B: California Housing | 20,640 | 8 | Regression | 1990 California Census |
| Case C: County Poverty | 1,749 | 9 | Regression | ACS American Community Survey |
| Case D: EuroSAT | 27,000 | 512 | Classification | Sentinel-2 satellite imagery |

## Usage

Scripts use relative paths to load data:

```python
# Case A
features = np.load('data/caseA_forest_biomass/aligned_features_multiyear.npy')
labels = np.load('data/caseA_forest_biomass/aligned_labels_multiyear.npy')

# Case B
from data.caseB_california_housing.load_california_housing import load_california_housing
features, labels, names = load_california_housing()

# Case C
import pandas as pd
df = pd.read_csv('data/caseC_county_poverty/county_poverty_features_final.csv')

# Case D
features = np.load('data/caseD_eurosat/eurosat_features.npy')
labels = np.load('data/caseD_eurosat/eurosat_labels.npy')
```
