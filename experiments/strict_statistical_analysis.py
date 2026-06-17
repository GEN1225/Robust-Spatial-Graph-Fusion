"""
Rigorous Statistical Analysis Script — Real Experimental Data Only
All statistics must be computed directly from the raw experimental outputs.
"""

import os
import sys
import numpy as np
import pickle
from scipy import stats
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_real_experimental_data():
    """Load real experimental data."""
    print("=" * 60)
    print("Loading real experimental data")
    print("=" * 60)

    with open(os.path.join(BASE_DIR, 's2_results_fixed.pkl'), 'rb') as f:
        s2_results = pickle.load(f)

    tkg_r2s = s2_results['TKG']['r2']
    ekg_r2s = s2_results['EKG']['r2']
    fusion_r2s = s2_results['Fusion']['r2']

    print(f"\nData source: s2_results_fixed.pkl")
    print(f"Data structure: 5 seeds, each with 5-fold averaged R^2")
    print(f"Total samples: {len(fusion_r2s)}")

    print(f"\nRaw data (5 seeds):")
    print(f"TKG R2: {tkg_r2s}")
    print(f"EKG R2: {ekg_r2s}")
    print(f"Fusion R2: {fusion_r2s}")

    return tkg_r2s, ekg_r2s, fusion_r2s

def compute_paired_deltas(tkg_r2s, ekg_r2s, fusion_r2s):
    """Compute paired deltas: Delta = Fusion - Best Single."""
    print("\n" + "=" * 60)
    print("Computing paired deltas: Delta = Fusion - Best Single")
    print("=" * 60)

    # Best Single = max(TKG, EKG) per seed
    best_single = [max(t, e) for t, e in zip(tkg_r2s, ekg_r2s)]

    deltas = [f - b for f, b in zip(fusion_r2s, best_single)]

    print(f"\nSeed-by-seed breakdown:")
    for i, (t, e, f, b, d) in enumerate(zip(tkg_r2s, ekg_r2s, fusion_r2s, best_single, deltas)):
        print(f"Seed {i+1}: TKG={t:.4f}, EKG={e:.4f}, Fusion={f:.4f}, Best={b:.4f}, Delta={d:.4f}")

    print(f"\nDelta array (for downstream statistics):")
    print(f"Delta = {deltas}")

    return np.array(deltas), np.array(best_single)

def compute_cliffs_delta_exact(fusion_r2s, best_single):
    """Compute Cliff's Delta exactly."""
    print("\n" + "=" * 60)
    print("Cliff's Delta Computation")
    print("=" * 60)

    print(f"\nFormula: delta = (#{fusion > best} - #{fusion < best}) / (n1 x n2)")
    print(f"where n1 = n2 = {len(fusion_r2s)}")

    n1 = len(fusion_r2s)
    n2 = len(best_single)

    greater = sum(1 for f in fusion_r2s for b in best_single if f > b)
    less = sum(1 for f in fusion_r2s for b in best_single if f < b)
    equal = sum(1 for f in fusion_r2s for b in best_single if f == b)

    print(f"\nPairwise comparison statistics:")
    print(f"  Fusion > Best: {greater}")
    print(f"  Fusion < Best: {less}")
    print(f"  Fusion = Best: {equal}")
    print(f"  Total pairs: {n1 * n2}")

    delta = (greater - less) / (n1 * n2)

    print(f"\nCliff's Delta: delta = ({greater} - {less}) / ({n1} x {n2}) = {delta:.6f}")

    if abs(delta) < 0.147:
        effect = "negligible"
    elif abs(delta) < 0.33:
        effect = "small"
    elif abs(delta) < 0.474:
        effect = "medium"
    else:
        effect = "large"

    print(f"Effect size: {effect}")
    print(f"Direction: {'Fusion worse' if delta < 0 else 'Fusion better'}")

    return delta, effect

def compute_wilcoxon_test_exact(fusion_r2s, best_single):
    """Compute the Wilcoxon Signed-Rank Test exactly."""
    print("\n" + "=" * 60)
    print("Wilcoxon Signed-Rank Test")
    print("=" * 60)

    print(f"\nTest type: Paired, two-sided")
    print(f"H0: Median difference between Fusion and Best Single is zero")
    print(f"H1: Median difference between Fusion and Best Single is not zero")

    if len(set(fusion_r2s)) <= 1 or len(set(best_single)) <= 1:
        print(f"\nWarning: Insufficient variance in data — cannot run Wilcoxon test")
        return None, None

    statistic, p_value = wilcoxon(fusion_r2s, best_single, alternative='two-sided')

    print(f"\nTest result:")
    print(f"  Test Statistic: {statistic:.6f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.001:
        sig = "*** (p < 0.001)"
    elif p_value < 0.01:
        sig = "** (p < 0.01)"
    elif p_value < 0.05:
        sig = "* (p < 0.05)"
    else:
        sig = "ns (p >= 0.05)"

    print(f"  Significance: {sig}")

    return statistic, p_value

def compute_distribution_statistics(deltas):
    """Compute distribution statistics (non-parametric)."""
    print("\n" + "=" * 60)
    print("Distribution Statistics (Non-parametric)")
    print("=" * 60)

    mean = np.mean(deltas)
    median = np.median(deltas)
    std = np.std(deltas, ddof=1)

    q1 = np.percentile(deltas, 25)
    q3 = np.percentile(deltas, 75)
    iqr = q3 - q1

    print(f"\nBasic statistics:")
    print(f"  Mean: {mean:.6f}")
    print(f"  Median: {median:.6f}")
    print(f"  Std (sample): {std:.6f}")

    print(f"\nQuartiles:")
    print(f"  Q1 (25%): {q1:.6f}")
    print(f"  Q3 (75%): {q3:.6f}")
    print(f"  IQR: {iqr:.6f}")

    print(f"\nBootstrap 95% CI:")
    print(f"  Method: Percentile bootstrap")
    print(f"  Bootstrap iterations: 10000")
    print(f"  Random seed: 42")

    np.random.seed(42)
    n_bootstrap = 10000
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(deltas, size=len(deltas), replace=True)
        bootstrap_means.append(np.mean(sample))

    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")

    return {
        'mean': mean,
        'median': median,
        'std': std,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def analyze_knn_sensitivity_exact():
    """Analyze k-NN sensitivity using real fold-level data."""
    print("\n" + "=" * 60)
    print("k-NN Sensitivity Analysis (fold-level data)")
    print("=" * 60)

    with open(os.path.join(BASE_DIR, 'knn_sensitivity_two_seeds_results.pkl'), 'rb') as f:
        knn_data = pickle.load(f)
        knn_results = knn_data['results']

    k_values = [3, 5, 8, 10, 15]
    seeds = [42, 123]

    print(f"\nData source: knn_sensitivity_two_seeds_results.pkl")
    print(f"k values: {k_values}")
    print(f"Seeds: {seeds}")

    print(f"\n" + "=" * 60)
    print("Fold-level Deltas (Fusion - Best Single)")
    print("=" * 60)

    for k in k_values:
        print(f"\nk = {k}:")
        for seed in seeds:
            tkg_folds = knn_results[seed][k]['TKG']['r2']
            ekg_folds = knn_results[seed][k]['EKG']['r2']
            fusion_folds = knn_results[seed][k]['Fusion']['r2']

            deltas_folds = []
            for i, (t, e, f) in enumerate(zip(tkg_folds, ekg_folds, fusion_folds)):
                best = max(t, e)
                delta = f - best
                deltas_folds.append(delta)

            mean_delta = np.mean(deltas_folds)
            std_delta = np.std(deltas_folds, ddof=1)

            print(f"  Seed {seed}:")
            print(f"    Fold-level Delta: {[f'{d:.4f}' for d in deltas_folds]}")
            print(f"    Mean Delta: {mean_delta:.6f}")
            print(f"    Std Delta: {std_delta:.6f}")

            if abs(mean_delta) < std_delta:
                print(f"    -> |Mean Delta| ({abs(mean_delta):.4f}) < Std ({std_delta:.4f}): within fold variability")
            else:
                print(f"    -> |Mean Delta| ({abs(mean_delta):.4f}) >= Std ({std_delta:.4f})")

    print(f"\n" + "=" * 60)
    print("Sign Consistency Analysis")
    print("=" * 60)

    consistency_count = 0
    for k in k_values:
        delta_42 = np.mean([
            knn_results[42][k]['Fusion']['r2'][i] - max(
                knn_results[42][k]['TKG']['r2'][i],
                knn_results[42][k]['EKG']['r2'][i]
            )
            for i in range(5)
        ])

        delta_123 = np.mean([
            knn_results[123][k]['Fusion']['r2'][i] - max(
                knn_results[123][k]['TKG']['r2'][i],
                knn_results[123][k]['EKG']['r2'][i]
            )
            for i in range(5)
        ])

        is_consistent = (delta_42 > 0 and delta_123 > 0) or (delta_42 < 0 and delta_123 < 0)
        if is_consistent:
            consistency_count += 1

        status = "[consistent]" if is_consistent else "[conflict]"
        print(f"k={k}: Seed42={delta_42:+.6f}, Seed123={delta_123:+.6f} {status}")

    consistency_rate = consistency_count / len(k_values)
    print(f"\nSign consistency: {consistency_rate*100:.1f}% ({consistency_count}/{len(k_values)})")
    print(f"Sign flip rate: {(1-consistency_rate)*100:.1f}%")

def generate_strict_statistical_report():
    """Generate a rigorous statistical report."""
    print("=" * 60)
    print("Rigorous Statistical Analysis Report")
    print("Based on real experimental data — no derived or assumed values")
    print("=" * 60)

    tkg_r2s, ekg_r2s, fusion_r2s = load_real_experimental_data()
    deltas, best_single = compute_paired_deltas(tkg_r2s, ekg_r2s, fusion_r2s)
    cliffs_delta, effect_size = compute_cliffs_delta_exact(fusion_r2s, best_single)
    wilcoxon_stat, wilcoxon_p = compute_wilcoxon_test_exact(fusion_r2s, best_single)
    dist_stats = compute_distribution_statistics(deltas)
    analyze_knn_sensitivity_exact()

    print("\n" + "=" * 60)
    print("Saving Statistical Report")
    print("=" * 60)

    report = {
        'data_source': 's2_results_fixed.pkl',
        'sample_size': len(deltas),
        'raw_data': {
            'tkg_r2': list(tkg_r2s),
            'ekg_r2': list(ekg_r2s),
            'fusion_r2': list(fusion_r2s),
            'best_single': list(best_single),
            'deltas': list(deltas)
        },
        'cliffs_delta': {
            'value': float(cliffs_delta),
            'effect_size': effect_size
        },
        'wilcoxon_test': {
            'statistic': float(wilcoxon_stat) if wilcoxon_stat is not None else None,
            'p_value': float(wilcoxon_p) if wilcoxon_p is not None else None
        },
        'distribution': dist_stats
    }

    report_file = os.path.join(BASE_DIR, 'strict_statistical_report.pkl')
    with open(report_file, 'wb') as f:
        pickle.dump(report, f)

    print(f"Report saved to: {report_file}")

    print("\n" + "=" * 60)
    print("Reproducible Statistical Summary")
    print("=" * 60)

    print(f"\nData source: s2_results_fixed.pkl")
    print(f"Sample size: n = {len(deltas)}")

    print(f"\nRaw Delta array:")
    print(f"Delta = {list(deltas)}")

    print(f"\nCliff's Delta:")
    print(f"  delta = {cliffs_delta:.6f}")
    print(f"  Effect size: {effect_size}")

    print(f"\nWilcoxon Signed-Rank Test:")
    print(f"  Statistic = {wilcoxon_stat:.6f}")
    print(f"  p-value = {wilcoxon_p:.6f}")

    print(f"\nDistribution statistics:")
    print(f"  Mean = {dist_stats['mean']:.6f}")
    print(f"  Median = {dist_stats['median']:.6f}")
    print(f"  Std = {dist_stats['std']:.6f}")
    print(f"  IQR = [{dist_stats['q1']:.6f}, {dist_stats['q3']:.6f}]")
    print(f"  95% CI (bootstrap, n=10000, seed=42) = [{dist_stats['ci_lower']:.6f}, {dist_stats['ci_upper']:.6f}]")

    print("\n" + "=" * 60)
    print("All statistics are reproducible from the raw data")
    print("=" * 60)

    return report

if __name__ == '__main__':
    try:
        report = generate_strict_statistical_report()
        print("\nRigorous statistical analysis complete!")
    except Exception as e:
        print("\nError during analysis:")
        import traceback
        traceback.print_exc()
