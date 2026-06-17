# -*- coding: utf-8 -*-
"""
County-Level Poverty Rate Spatial Regression Dataset

Data sources:
1. ACS 5-Year Poverty Subject Table (S1701), 2022
2. US County Shapefile (tl_2022_us_county.shp)

Outputs:
1. county_poverty_final.csv
2. split_summary.txt
3. delta_value.txt
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from math import radians, cos, sin, asin, sqrt
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("="*80)
print("County-Level Poverty Rate Spatial Regression Dataset")
print("="*80)
print()

data_dir = Path(__file__).parent
acs_file = data_dir / 'ACSST5Y2022_S1701.csv'
shapefile = data_dir / 'tl_2022_us_county' / 'tl_2022_us_county.shp'
output_dir = data_dir
output_dir.mkdir(exist_ok=True)

# ============================================================================
# STEP 1 — CLEAN ACS DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 1 — CLEAN ACS DATA")
print("="*80)

print("\nLoading ACS data...")
df_acs = pd.read_csv(acs_file, low_memory=False)
print(f"  Raw shape: {df_acs.shape}")
print(f"  Number of columns: {len(df_acs.columns)}")

print("\nFirst 5 columns:")
print(df_acs.iloc[:5, :5])

print("\nDetecting data structure...")

indicator_col = df_acs.columns[0]
print(f"  Indicator column: {indicator_col}")

county_cols = [col for col in df_acs.columns if col != indicator_col]
print(f"  Number of counties: {len(county_cols)}")

print("\nTransposing data...")
df_transposed = df_acs.set_index(indicator_col).T
df_transposed.index.name = 'GEOID'
df_transposed = df_transposed.reset_index()

print(f"  Shape after transpose: {df_transposed.shape}")
print(f"  Number of columns: {len(df_transposed.columns)}")

print("\nFirst 5 rows, first 5 columns after transpose:")
print(df_transposed.iloc[:5, :5])

print("\nExtracting target features...")

feature_mapping = {
    'poverty_rate': 'Estimate!!Below poverty level!!Percent below poverty level!!POVERTY RATE FOR FAMILIES AND PEOPLE FOR WHOM POVERTY STATUS IS DETERMINED',
    'under_18_pct': 'Estimate!!Below poverty level!!Percent below poverty level!!AGE!!Under 18 years',
    'age_18_64_pct': 'Estimate!!Below poverty level!!Percent below poverty level!!AGE!!18 to 64 years',
    'age_65_over_pct': 'Estimate!!Below poverty level!!Percent below poverty level!!AGE!!65 years and over',
    'less_than_hs_pct': 'Estimate!!Below poverty level!!Percent below poverty level!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Less than high school graduate',
    'bachelor_or_higher_pct': 'Estimate!!Below poverty level!!Percent below poverty level!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Bachelor\'s degree or higher',
    'employed_pct': 'Estimate!!Below poverty level!!Percent below poverty level!!EMPLOYMENT STATUS!!Civilian labor force 16 years and over!!Employed',
    'unemployed_pct': 'Estimate!!Below poverty level!!Percent below poverty level!!EMPLOYMENT STATUS!!Civilian labor force 16 years and over!!Unemployed',
    'worked_full_time_pct': 'Estimate!!Below poverty level!!Percent below poverty level!!WORK EXPERIENCE!!Population 16 years and over!!Worked full-time, year-round in the past 12 months',
    'did_not_work_pct': 'Estimate!!Below poverty level!!Percent below poverty level!!WORK EXPERIENCE!!Population 16 years and over!!Did not work',
    'white_alone_pct': 'Estimate!!Below poverty level!!Percent below poverty level!!RACE AND HISPANIC OR LATINO ORIGIN!!White alone',
    'black_alone_pct': 'Estimate!!Below poverty level!!Percent below poverty level!!RACE AND HISPANIC OR LATINO ORIGIN!!Black or African American alone',
    'hispanic_origin_pct': 'Estimate!!Below poverty level!!Percent below poverty level!!RACE AND HISPANIC OR LATINO ORIGIN!!Hispanic or Latino origin (of any race)',
    'mean_income_deficit': 'Estimate!!Below poverty level!!MEAN INCOME DEFICIT FOR FAMILIES AND PEOPLE FOR WHOM POVERTY STATUS IS DETERMINED (dollars)'
}

print("\nAttempting keyword-based column matching...")

available_cols = df_transposed.columns.tolist()
print(f"\nAvailable columns: {len(available_cols)}")

def find_column_by_keywords(df, keywords):
    """Find a column whose name contains all given keywords."""
    for col in df.columns:
        if all(kw.lower() in col.lower() for kw in keywords):
            return col
    return None

feature_keywords = {
    'poverty_rate': ['percent', 'poverty', 'level', 'population'],
    'under_18_pct': ['under 18', 'poverty'],
    'age_18_64_pct': ['18 to 64', 'poverty'],
    'age_65_over_pct': ['65 years and over', 'poverty'],
    'less_than_hs_pct': ['less than high school', 'poverty'],
    'bachelor_or_higher_pct': ['bachelor', 'poverty'],
    'employed_pct': ['employed', 'poverty', 'civilian'],
    'unemployed_pct': ['unemployed', 'poverty'],
    'worked_full_time_pct': ['worked full-time', 'poverty'],
    'did_not_work_pct': ['did not work', 'poverty'],
    'white_alone_pct': ['white alone', 'poverty'],
    'black_alone_pct': ['black', 'african american', 'poverty'],
    'hispanic_origin_pct': ['hispanic', 'latino', 'poverty'],
    'mean_income_deficit': ['mean income deficit', 'dollars']
}

matched_features = {}
for feature_name, keywords in feature_keywords.items():
    col = find_column_by_keywords(df_transposed, keywords)
    if col:
        matched_features[feature_name] = col
        print(f"  + {feature_name}: matched")
    else:
        print(f"  - {feature_name}: no match found")

if len(matched_features) < 5:
    print("\nWarning: too few features matched. Printing all column names for inspection...")
    print("\nAll column names (first 20):")
    for i, col in enumerate(available_cols[:20]):
        print(f"  {i}: {col}")

    col_names_file = output_dir / 'acs_column_names.txt'
    with open(col_names_file, 'w', encoding='utf-8') as f:
        for col in available_cols:
            f.write(f"{col}\n")
    print(f"\n  All column names saved to: {col_names_file}")

print("\nBuilding final dataframe...")
df_final = pd.DataFrame()
df_final['fips'] = df_transposed['GEOID']

for feature_name, col_name in matched_features.items():
    df_final[feature_name] = pd.to_numeric(df_transposed[col_name], errors='coerce')

print(f"  Final shape: {df_final.shape}")
print(f"  Number of features: {len(matched_features)}")

# ============================================================================
# STEP 2 — GEO MERGE
# ============================================================================
print("\n" + "="*80)
print("STEP 2 — GEO MERGE")
print("="*80)

print("\nLoading county shapefile...")
gdf = gpd.read_file(shapefile)
print(f"  Shapefile shape: {gdf.shape}")
print(f"  CRS: {gdf.crs}")

gdf = gdf[['GEOID', 'geometry']]

print("\nComputing centroids...")
gdf['centroid'] = gdf.geometry.centroid

gdf['lon'] = gdf.centroid.x
gdf['lat'] = gdf.centroid.y

print(f"  Longitude range: [{gdf['lon'].min():.2f}, {gdf['lon'].max():.2f}]")
print(f"  Latitude range: [{gdf['lat'].min():.2f}, {gdf['lat'].max():.2f}]")

print("\nMerging ACS data with geography...")
df_merged = df_final.merge(gdf[['GEOID', 'lat', 'lon']], left_on='fips', right_on='GEOID', how='inner')
df_merged = df_merged.drop(columns=['GEOID'])

print(f"  Shape after merge: {df_merged.shape}")

# ============================================================================
# STEP 3 — FINAL TABLE FORMAT
# ============================================================================
print("\n" + "="*80)
print("STEP 3 — FINAL TABLE FORMAT")
print("="*80)

print("\nRemoving Alaska, Hawaii, Puerto Rico...")

df_merged['state_fips'] = df_merged['fips'].astype(str).str[:2]

exclude_states = ['02', '15', '72']
df_contiguous = df_merged[~df_merged['state_fips'].isin(exclude_states)].copy()

print(f"  Before removal: {len(df_merged)} counties")
print(f"  After removal: {len(df_contiguous)} counties")

print("\nDropping missing values...")
df_contiguous = df_contiguous.dropna()
print(f"  After dropping NaN: {len(df_contiguous)} counties")

# ============================================================================
# STEP 4 — STRICT OOD SPLIT
# ============================================================================
print("\n" + "="*80)
print("STEP 4 — STRICT OOD SPLIT")
print("="*80)

train_states = [
    '09', '23', '25', '33', '44', '50',  # New England
    '34', '36', '42',  # Mid-Atlantic
    '17', '18', '26', '39', '55',  # Midwest (east of Mississippi)
    '01', '10', '11', '12', '13', '21', '24', '28', '37', '45', '47', '51', '54'  # South (east of Mississippi)
]

buffer_states = [
    '30', '56', '08', '35',  # MT, WY, CO, NM
    '38', '46', '31', '20', '40', '48'  # ND, SD, NE, KS, OK, TX
]

test_states = [
    '53', '41', '06', '32', '04', '16', '49'  # WA, OR, CA, NV, AZ, ID, UT
]

def assign_split(state_fips):
    if state_fips in train_states:
        return 'train'
    elif state_fips in buffer_states:
        return 'buffer'
    elif state_fips in test_states:
        return 'test'
    else:
        return 'unknown'

df_contiguous['split'] = df_contiguous['state_fips'].apply(assign_split)

print("\nSplit statistics:")
print(df_contiguous['split'].value_counts())

# ============================================================================
# STEP 5 — DISTANCE CHECK
# ============================================================================
print("\n" + "="*80)
print("STEP 5 — DISTANCE CHECK")
print("="*80)

def haversine(lon1, lat1, lon2, lat2):
    """Compute Haversine distance between two points (km)."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in km
    return c * r

train_counties = df_contiguous[df_contiguous['split'] == 'train'][['lat', 'lon']].values
test_counties = df_contiguous[df_contiguous['split'] == 'test'][['lat', 'lon']].values

print(f"\nTrain counties: {len(train_counties)}")
print(f"Test counties: {len(test_counties)}")

print("\nComputing minimum distance between train and test...")
min_distance = float('inf')

for i, (lat1, lon1) in enumerate(train_counties):
    if i % 100 == 0:
        print(f"  Progress: {i}/{len(train_counties)}")

    for lat2, lon2 in test_counties:
        dist = haversine(lon1, lat1, lon2, lat2)
        if dist < min_distance:
            min_distance = dist

print(f"\nMinimum distance delta = {min_distance:.2f} km")

if min_distance > 500:
    print(f"  Pass: delta > 500 km")
else:
    print(f"  Fail: delta <= 500 km")

# ============================================================================
# STEP 6 — OUTPUT
# ============================================================================
print("\n" + "="*80)
print("STEP 6 — OUTPUT")
print("="*80)

output_csv = output_dir / 'county_poverty_final.csv'
df_contiguous.to_csv(output_csv, index=False)
print(f"\nData saved: {output_csv}")

split_summary_file = output_dir / 'split_summary.txt'
with open(split_summary_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("Split Summary\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total counties: {len(df_contiguous)}\n\n")
    f.write("Split distribution:\n")
    f.write(df_contiguous['split'].value_counts().to_string())
    f.write("\n\n")
    f.write("Train states (FIPS):\n")
    f.write(", ".join(train_states))
    f.write("\n\n")
    f.write("Buffer states (FIPS):\n")
    f.write(", ".join(buffer_states))
    f.write("\n\n")
    f.write("Test states (FIPS):\n")
    f.write(", ".join(test_states))
    f.write("\n")

print(f"Split summary saved: {split_summary_file}")

delta_file = output_dir / 'delta_value.txt'
with open(delta_file, 'w', encoding='utf-8') as f:
    f.write(f"Minimum distance between train and test counties:\n")
    f.write(f"delta = {min_distance:.2f} km\n\n")
    if min_distance > 500:
        f.write("Pass: delta > 500 km\n")
    else:
        f.write("Fail: delta <= 500 km\n")

print(f"Delta value saved: {delta_file}")

print("\n" + "="*80)
print("Dataset build complete!")
print("="*80)
print(f"\nOutput files:")
print(f"  1. {output_csv}")
print(f"  2. {split_summary_file}")
print(f"  3. {delta_file}")
