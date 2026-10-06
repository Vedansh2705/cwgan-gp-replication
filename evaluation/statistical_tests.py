# evaluation/statistical_tests.py
# =============================================================
# Step 11: Friedman Test + Nemenyi Post-hoc (Paper Section 3.4)
# =============================================================
# DEVIATION FROM PAPER: Paper Friedman test ko 15 DATASETS ko
# "blocks" ki tarah use karta hai. Humare paas sirf 1 dataset hai
# (Iris), isliye hum 5 REPEATS ko blocks ki tarah use kar rahe hain.
# Ye ek explicit, documented assumption hai - N=5 (paper mein N=15).

import numpy as np
from scipy.stats import friedmanchisquare

# Nemenyi critical value table (Demsar 2006), alpha=0.05
# k = number of methods being compared
NEMENYI_Q_ALPHA_005 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
}


def friedman_test_per_classifier(all_repeat_results, metric_name):
    """
    Har classifier ke liye ek Friedman test karta hai:
    kya 5 methods ka performance significantly different hai,
    5 repeats (blocks) ke across?

    Returns: {classifier: (statistic, p_value)}
    """
    method_names = list(all_repeat_results[0].keys())
    classifiers = list(all_repeat_results[0][method_names[0]].keys())

    results = {}
    for clf_name in classifiers:
        # Har method ke liye, 5 repeats ki values ek array mein
        method_arrays = []
        for method_name in method_names:
            values = [
                repeat[method_name][clf_name][metric_name]
                for repeat in all_repeat_results
            ]
            method_arrays.append(values)

        stat, p_value = friedmanchisquare(*method_arrays)
        results[clf_name] = (stat, p_value)

    return results


def compute_mean_ranks(all_repeat_results, metric_name):
    """
    Har method ka mean rank nikaalta hai (sab classifiers aur
    repeats ke across) - paper ke Table 4 jaisa.
    Paper convention: HIGHER rank number = BETTER performance.
    """
    method_names = list(all_repeat_results[0].keys())
    classifiers = list(all_repeat_results[0][method_names[0]].keys())
    n_methods = len(method_names)

    all_ranks = {m: [] for m in method_names}

    for repeat in all_repeat_results:
        for clf_name in classifiers:
            scores = {m: repeat[m][clf_name][metric_name] for m in method_names}
            # Sort ascending, phir rank 1 (worst) se n_methods (best) tak do
            sorted_methods = sorted(scores.items(), key=lambda x: x[1])
            for rank_val, (method_name, _) in enumerate(sorted_methods, start=1):
                all_ranks[method_name].append(rank_val)

    mean_ranks = {m: np.mean(ranks) for m, ranks in all_ranks.items()}
    return mean_ranks


def nemenyi_critical_distance(n_methods, n_blocks, alpha=0.05):
    """
    Paper Eq. (Section 3.4): Nemenyi critical distance.
    CD = q_alpha * sqrt( k(k+1) / (6N) )

    n_blocks = N = humare case mein 5 repeats (paper mein N=15 datasets)
    """
    q_alpha = NEMENYI_Q_ALPHA_005.get(n_methods)
    if q_alpha is None:
        raise ValueError(f"q_alpha table mein {n_methods} methods nahi hain")

    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_blocks))
    return cd


def run_statistical_analysis(all_repeat_results):
    """
    Poora Phase 11 analysis: Friedman test (har classifier ke liye,
    har metric ke liye) + mean ranks + Nemenyi critical distance.
    """
    method_names = list(all_repeat_results[0].keys())
    n_methods = len(method_names)
    n_blocks = len(all_repeat_results)  # 5 repeats

    for metric_name in ["f_measure", "g_mean", "auc"]:
        print(f"\n{'='*70}")
        print(f"STATISTICAL ANALYSIS: {metric_name.upper()}")
        print(f"{'='*70}")

        # ---- Friedman test per classifier ----
        friedman_results = friedman_test_per_classifier(all_repeat_results, metric_name)
        print(f"\nFriedman Test (per classifier, N={n_blocks} repeats as blocks):")
        print(f"{'Classifier':12s} {'Statistic':>12s} {'p-value':>12s} {'Significant?':>15s}")
        print("-" * 55)
        for clf_name, (stat, p_val) in friedman_results.items():
            significant = "YES (p<0.05)" if p_val < 0.05 else "no"
            print(f"{clf_name:12s} {stat:>12.4f} {p_val:>12.4f} {significant:>15s}")

        # ---- Mean ranks ----
        mean_ranks = compute_mean_ranks(all_repeat_results, metric_name)
        print(f"\nMean Ranks (higher = better, max={n_methods}):")
        sorted_ranks = sorted(mean_ranks.items(), key=lambda x: -x[1])
        for method_name, rank in sorted_ranks:
            print(f"  {method_name:20s}: {rank:.3f}")

        # ---- Nemenyi Critical Distance ----
        cd = nemenyi_critical_distance(n_methods, n_blocks, alpha=0.05)
        print(f"\nNemenyi Critical Distance (alpha=0.05): {cd:.3f}")
        print("  (Do methods ke beech rank-difference agar CD se zyada ho,")
        print("   toh woh significantly different maane jaate hain)")

        best_method, best_rank = sorted_ranks[0]
        print(f"\n  Best method ({best_method}) vs others:")
        for method_name, rank in sorted_ranks[1:]:
            diff = best_rank - rank
            sig = "SIGNIFICANT" if diff > cd else "not significant"
            print(f"    vs {method_name:20s}: rank diff = {diff:.3f}  -> {sig}")