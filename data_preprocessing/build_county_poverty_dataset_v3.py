# -*- coding: utf-8 -*-
"""
County-Level Poverty Rate Spatial Regression Dataset v3
Simplified version: directly extract needed columns.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from math import radians, cos, sin, asin, sqrt
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("="*80)
print("County-Level Poverty Rate Spatial Regression Dataset v3")
print("="*80)

data_dir = Path(__file__).parent
acs_file = data_dir / 'ACSST5Y2022_S1701.csv'
shapefile = data_dir / 'tl_2022_us_county' / 'tl_2022_us_county.shp'

# ============================================================================
# STEP 1 — CLEAN ACS DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 1 — CLEAN ACS DATA")
print("="*80)

print("\nLoading ACS data...")
df_raw = pd.read_csv(acs_file, low_memory=False)
print(f"  Raw shape: {df_raw.shape}")

df_t = df_raw.set_index(df_raw.columns[0]).T
df_t.index.name = 'geo_column'
df_t = df_t.reset_index()

print(f"  Shape after transpose: {df_t.shape}")

county_mask = df_t['geo_column'].str.contains('County', na=False) & df_t['geo_column'].str.contains('Estimate', na=False)
df_counties = df_t[county_mask].copy()

print(f"  County rows: {len(df_counties)}")

def parse_geo_column(col_name):
    """Extract county name and state from the geo column string."""
    parts = col_name.split('!!')
    if len(parts) >= 3:
        county_state = parts[0].strip()
        indicator = parts[1].strip()
        return county_state, indicator
    return None, None

df_counties['county_state'], df_counties['indicator_type'] = zip(*df_counties['geo_column'].apply(parse_geo_column))

df_counties = df_counties[df_counties['county_state'].notna()].copy()
print(f"  County rows after parsing: {len(df_counties)}")

print("\nIndicator types:")
print(df_counties['indicator_type'].value_counts().head(10))

df_poverty = df_counties[df_counties['indicator_type'] == 'Percent below poverty level'].copy()
print(f"\nPoverty rate rows: {len(df_poverty)}")

poverty_col = 'Population for whom poverty status is determined'
if poverty_col in df_poverty.columns:
    df_poverty['poverty_rate'] = df_poverty[poverty_col]
    print(f"  Poverty rate extracted")
else:
    print(f"  Poverty rate column not found")
    print(f"  Available columns: {df_poverty.columns.tolist()[:10]}")

feature_cols = {
    'under_18_pct': 'Under 18 years',
    'age_18_64_pct': '18 to 64 years',
    'age_65_over_pct': '65 years and over',
    'less_than_hs_pct': 'Less than high school graduate',
    'bachelor_or_higher_pct': "Bachelor's degree or higher",
    'employed_pct': 'Employed',
    'unemployed_pct': 'Unemployed',
    'worked_full_time_pct': 'Worked full-time, year-round in the past 12 months',
    'did_not_work_pct': 'Did not work',
    'white_alone_pct': 'White alone',
    'black_alone_pct': 'Black or African American alone',
    'hispanic_origin_pct': 'Hispanic or Latino origin (of any race)'
}

print("\nExtracting features:")
for new_col, orig_col in feature_cols.items():
    if orig_col in df_poverty.columns:
        df_poverty[new_col] = df_poverty[orig_col]
        print(f"  + {new_col}")
    else:
        print(f"  - {new_col} (column: {orig_col})")

keep_cols = ['county_state', 'poverty_rate'] + list(feature_cols.keys())
available_cols = [col for col in keep_cols if col in df_poverty.columns]
df_final = df_poverty[available_cols].copy()

print(f"\nFinal feature count: {len(available_cols) - 1}")

print("\nCleaning data...")
for col in df_final.columns:
    if col != 'county_state':
        df_final[col] = df_final[col].astype(str).str.replace('%', '').str.replace(',', '').str.strip()
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

print(f"  Shape after cleaning: {df_final.shape}")
print(f"\nFirst 5 rows:")
print(df_final.head())

# ============================================================================
# STEP 2 — GEO MERGE
# ============================================================================
print("\n" + "="*80)
print("STEP 2 — GEO MERGE")
print("="*80)

print("\nLoading shapefile...")
gdf = gpd.read_file(shapefile)
print(f"  Shape: {gdf.shape}")

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

if gdf.crs != 'EPSG:4326':
    gdf = gdf.to_crs('EPSG:4326')

gdf['centroid'] = gdf.geometry.centroid
gdf['lon'] = gdf.centroid.x
gdf['lat'] = gdf.centroid.y

print("\nMerging data...")
df_merged = df_final.merge(
    gdf[['GEOID', 'county_state', 'lat', 'lon']],
    on='county_state',
    how='inner'
)
print(f"  Counties after merge: {len(df_merged)}")

# ============================================================================
# STEP 3 — FINAL TABLE FORMAT
# ============================================================================
print("\n" + "="*80)
print("STEP 3 — FINAL TABLE FORMAT")
print("="*80)

df_merged = df_merged.rename(columns={'GEOID': 'fips'})

df_merged['state_fips'] = df_merged['fips'].astype(str).str[:2]
exclude_states = ['02', '15', '72']
df_contiguous = df_merged[~df_merged['state_fips'].isin(exclude_states)].copy()

print(f"After removing AK/HI/PR: {len(df_contiguous)} counties")

df_contiguous = df_contiguous.dropna(subset=['poverty_rate', 'lat', 'lon'])
print(f"After dropping NaN: {len(df_contiguous)} counties")

# ============================================================================
# STEP 4 — STRICT OOD SPLIT
# ============================================================================
print("\n" + "="*80)
print("STEP 4 — STRICT OOD SPLIT")
print("="*80)

train_states = ['09', '23', '25', '33', '44', '50', '34', '36', '42', '17', '18', '26', '39', '55',
                '01', '10', '11', '12', '13', '21', '24', '28', '37', '45', '47', '51', '54']
buffer_states = ['30', '56', '08', '35', '38', '46', '31', '20', '40', '48']
test_states = ['53', '41', '06', '32', '04', '16', '49']

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
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

train_counties = df_contiguous[df_contiguous['split'] == 'train'][['lat', 'lon']].values
test_counties = df_contiguous[df_contiguous['split'] == 'test'][['lat', 'lon']].values

print(f"Train: {len(train_counties)}, Test: {len(test_counties)}")

if len(train_counties) > 0 and len(test_counties) > 0:
    print("\nComputing minimum distance...")
    min_distance = float('inf')
    for i, (lat1, lon1) in enumerate(train_counties):
        if i % 100 == 0:
            print(f"  {i}/{len(train_counties)}")
        for lat2, lon2 in test_counties:
            dist = haversine(lon1, lat1, lon2, lat2)
            if dist < min_distance:
                min_distance = dist
    print(f"\ndelta = {min_distance:.2f} km")
    print(f"  {'PASS' if min_distance > 500 else 'FAIL'}: delta > 500 km")
else:
    min_distance = None

# ============================================================================
# STEP 6 — OUTPUT
# ============================================================================
print("\n" + "="*80)
print("STEP 6 — OUTPUT")
print("="*80)

final_cols = ['fips', 'poverty_rate'] + [c for c in df_contiguous.columns if c.endswith('_pct')] + ['lat', 'lon', 'split']
final_cols = [c for c in final_cols if c in df_contiguous.columns]
df_output = df_contiguous[final_cols].copy()

output_csv = data_dir / 'county_poverty_final.csv'
df_output.to_csv(output_csv, index=False)
print(f"\nData saved: {output_csv}")
print(f"  Shape: {df_output.shape}")
print(f"  Columns: {df_output.columns.tolist()}")

split_summary = data_dir / 'split_summary.txt'
with open(split_summary, 'w', encoding='utf-8') as f:
    f.write(f"Total: {len(df_output)}\n\n")
    f.write(df_output['split'].value_counts().to_string())
print(f"Split summary saved: {split_summary}")

if min_distance:
    delta_file = data_dir / 'delta_value.txt'
    with open(delta_file, 'w', encoding='utf-8') as f:
        f.write(f"delta = {min_distance:.2f} km\n")
        f.write(f"{'PASS' if min_distance > 500 else 'FAIL'}: delta > 500 km\n")
    print(f"Delta value saved: {delta_file}")

print("\n" + "="*80)
print("Done!")
print("="*80)
