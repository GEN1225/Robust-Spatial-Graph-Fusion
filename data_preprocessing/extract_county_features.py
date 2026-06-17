# -*- coding: utf-8 -*-
"""
Extract county-level features for GNN experiments.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("="*80)
print("Extract County-Level Features for GNN Experiment")
print("="*80)

data_dir = Path(__file__).parent
acs_file = data_dir / 'ACSST5Y2022_S1701.csv'
base_file = data_dir / 'county_poverty_final.csv'

# ============================================================================
# Load base data
# ============================================================================
print("\nLoading base data...")
df_base = pd.read_csv(base_file)
print(f"  Base data shape: {df_base.shape}")
print(f"  Columns: {df_base.columns.tolist()}")

# ============================================================================
# Load ACS data
# ============================================================================
print("\nLoading ACS data...")
df_acs = pd.read_csv(acs_file, low_memory=False)
print(f"  ACS data shape: {df_acs.shape}")

df_t = df_acs.set_index(df_acs.columns[0]).T
df_t.index.name = 'geo_column'
df_t = df_t.reset_index()

print(f"  Shape after transpose: {df_t.shape}")

# ============================================================================
# Extract county-level rows
# ============================================================================
print("\nExtracting county-level rows...")

county_mask = df_t['geo_column'].str.contains('County', na=False) & \
              df_t['geo_column'].str.contains('Estimate', na=False)
df_counties = df_t[county_mask].copy()

print(f"  County rows: {len(df_counties)}")

def parse_geo_column(col_name):
    """Parse county name, state, and indicator from the geo column string."""
    parts = col_name.split('!!')
    if len(parts) >= 3:
        county_state = parts[0].strip()
        indicator = parts[1].strip()
        return county_state, indicator
    return None, None

df_counties['county_state'], df_counties['indicator_type'] = \
    zip(*df_counties['geo_column'].apply(parse_geo_column))

df_counties = df_counties[df_counties['county_state'].notna()].copy()

print(f"  After parsing: {len(df_counties)}")
print(f"\nIndicator types:")
print(df_counties['indicator_type'].value_counts())

# ============================================================================
# Define feature mapping
# ============================================================================
print("\nDefining feature mapping...")

available_indicators = df_acs[df_acs.columns[0]].tolist()
print(f"\nAvailable indicators: {len(available_indicators)}")
print("\nFirst 30 indicators:")
for i, ind in enumerate(available_indicators[:30]):
    print(f"  {i}: {ind}")

feature_mapping = {
    'under_18_pct': 'Under 18 years',
    'age_65_over_pct': '65 years and over',
    'less_than_hs_pct': 'Less than high school graduate',
    'bachelor_or_higher_pct': "Bachelor's degree or higher",
    'unemployed_pct': 'Unemployed',
    'worked_full_time_pct': 'Worked full-time, year-round in the past 12 months',
    'white_alone_pct': 'White alone',
    'black_alone_pct': 'Black or African American alone',
    'hispanic_origin_pct': 'Hispanic or Latino origin (of any race)',
}

# ============================================================================
# Extract features from "Percent below poverty level" rows
# ============================================================================
print("\nExtracting features (from Percent below poverty level)...")

df_percent = df_counties[df_counties['indicator_type'] == 'Percent below poverty level'].copy()
print(f"  Percent rows: {len(df_percent)}")

features_extracted = {}
for feature_name, indicator_name in feature_mapping.items():
    if indicator_name in df_percent.columns:
        features_extracted[feature_name] = indicator_name
        print(f"  + {feature_name}")
    else:
        print(f"  - {feature_name} (column: {indicator_name})")

df_features = df_percent[['county_state'] + list(features_extracted.values())].copy()

rename_dict = {v: k for k, v in features_extracted.items()}
df_features = df_features.rename(columns=rename_dict)

print(f"\nFeature data shape: {df_features.shape}")
print(f"Extracted features: {list(features_extracted.keys())}")

# ============================================================================
# Clean data
# ============================================================================
print("\nCleaning data...")

for col in df_features.columns:
    if col != 'county_state':
        df_features[col] = df_features[col].astype(str).str.replace('%', '').str.replace(',', '').str.strip()
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

print(f"  Shape after cleaning: {df_features.shape}")
print(f"\nFirst 5 rows:")
print(df_features.head())

# ============================================================================
# Merge features into base data
# ============================================================================
print("\nMerging features...")

import geopandas as gpd

shapefile = data_dir / 'tl_2022_us_county' / 'tl_2022_us_county.shp'
gdf = gpd.read_file(shapefile)

state_fips_to_name = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas', '06': 'California',
    '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware', '11': 'District of Columbia',
    '12': 'Florida', '13': 'Georgia', '15': 'Hawaii', '16': 'Idaho', '17': 'Illinois',
    '18': 'Indiana', '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana',
    '23': 'Maine', '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
    '28': 'Mississippi', '29': 'Missouri', '30': 'Montana', '31': 'Nebraska', '32': 'Nevada',
    '33': 'New Hampshire', '34': 'New Jersey', '35': 'New Mexico', '36': 'New York',
    '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio', '40': 'Oklahoma', '41': 'Oregon',
    '42': 'Pennsylvania', '44': 'Rhode Island', '45': 'South Carolina', '46': 'South Dakota',
    '47': 'Tennessee', '48': 'Texas', '49': 'Utah', '50': 'Vermont', '51': 'Virginia',
    '53': 'Washington', '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming', '72': 'Puerto Rico'
}

gdf['STATE_NAME'] = gdf['STATEFP'].map(state_fips_to_name)
gdf['county_state'] = gdf['NAME'] + ' County, ' + gdf['STATE_NAME']

geoid_to_county_state = dict(zip(gdf['GEOID'], gdf['county_state']))

df_base['county_state'] = df_base['fips'].map(geoid_to_county_state)

print(f"  Base data with county_state: {df_base.shape}")
print(f"  Successfully matched: {df_base['county_state'].notna().sum()} / {len(df_base)}")

df_merged = df_base.merge(df_features, on='county_state', how='left')

print(f"  Shape after merge: {df_merged.shape}")
print(f"  Feature columns: {[col for col in df_merged.columns if col.endswith('_pct')]}")

# ============================================================================
# Standardize features (using TRAIN SET statistics)
# ============================================================================
print("\nStandardizing features (using TRAIN SET statistics)...")

feature_cols = [col for col in df_merged.columns if col.endswith('_pct')]
print(f"  Number of features: {len(feature_cols)}")

train_mask = df_merged['split'] == 'train'
train_data = df_merged[train_mask]

print(f"  Train set samples: {train_data.shape[0]}")

train_stats = {}
for col in feature_cols:
    mean = train_data[col].mean()
    std = train_data[col].std()
    train_stats[col] = {'mean': mean, 'std': std}
    print(f"  {col}: mean={mean:.3f}, std={std:.3f}")

for col in feature_cols:
    mean = train_stats[col]['mean']
    std = train_stats[col]['std']
    if std > 0:
        df_merged[col] = (df_merged[col] - mean) / std
    else:
        print(f"  Warning: {col} has zero std, skipping standardization")

print(f"\nStandardization complete")

# ============================================================================
# Prepare final output
# ============================================================================
print("\nPreparing final output...")

df_final = df_merged.rename(columns={
    'fips': 'GEOID',
    'poverty_rate': 'y'
})

feature_rename = {col: f'X{i+1}' for i, col in enumerate(feature_cols)}
df_final = df_final.rename(columns=feature_rename)

final_cols = ['GEOID', 'lat', 'lon', 'split', 'y'] + [f'X{i+1}' for i in range(len(feature_cols))]
final_cols = [col for col in final_cols if col in df_final.columns]

df_output = df_final[final_cols].copy()

print(f"  Final shape: {df_output.shape}")
print(f"  Columns: {df_output.columns.tolist()}")

initial_count = len(df_output)
df_output = df_output.dropna()
print(f"  After dropping NaN: {initial_count} -> {len(df_output)}")

# ============================================================================
# Save
# ============================================================================
print("\nSaving data...")

output_file = data_dir / 'county_poverty_features_final.csv'
df_output.to_csv(output_file, index=False)

print(f"\nData saved: {output_file}")
print(f"  Shape: {df_output.shape}")
print(f"  Number of features: {len(feature_cols)}")

feature_mapping_file = data_dir / 'feature_mapping.txt'
with open(feature_mapping_file, 'w', encoding='utf-8') as f:
    f.write("Feature Mapping\n")
    f.write("="*80 + "\n\n")
    for i, (old_name, new_name) in enumerate(feature_rename.items()):
        f.write(f"{new_name}: {old_name}\n")
    f.write("\n")
    f.write("Train Set Statistics (used for standardization):\n")
    f.write("-"*80 + "\n")
    for col in feature_cols:
        stats = train_stats[col]
        f.write(f"{col}: mean={stats['mean']:.3f}, std={stats['std']:.3f}\n")

print(f"Feature mapping saved: {feature_mapping_file}")

print("\n" + "="*80)
print("Summary")
print("="*80)
print(f"\nSplit distribution:")
print(df_output['split'].value_counts())
print(f"\nTarget variable (y) statistics:")
print(df_output['y'].describe())

print("\n" + "="*80)
print("Done!")
print("="*80)
