# experiments/full_experiment.py
# =============================================================
# Step 9: Full Experiment - Paper ke SAARE 8 Methods (5 Repeats)
# =============================================================
# SMOTE, ADASYN, Borderline-SMOTE, ROS, GAN, CGAN, WGAN-GP, CWGAN-GP
# Command line se dataset choose kar sakte ho:
#   python experiments/full_experiment.py iris
#   python experiments/full_experiment.py heart
#   python experiments/full_experiment.py haberman
#   python experiments/full_experiment.py liver

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from collections import defaultdict

from experiments.train_test_split_demo import get_train_test_splits
from baselines.smote import apply_smote
from baselines.adasyn import apply_adasyn
from baselines.borderline_and_ros import apply_borderline_smote, apply_ros
from baselines.gan_and_cgan import apply_gan, apply_cgan
from baselines.wgan_gp import apply_wgan_gp
from baselines.cwgan_gp_oversample import apply_cwgan_gp
from classifiers.classifiers import get_classifiers
from evaluation.metrics import evaluate_all_metrics
from evaluation.save_results import save_averaged_results_csv
from evaluation.statistical_tests import run_statistical_analysis


def run_single_repeat(X_train, y_train, X_test, y_test, gan_epochs=5000, repeat_idx=0):
    """Ek split ke liye SAB 8 oversampling methods se sab classifiers evaluate karta hai."""
    results = {}

    X_smote, y_smote = apply_smote(X_train, y_train, k_neighbors=5, random_state=repeat_idx)
    X_adasyn, y_adasyn = apply_adasyn(X_train, y_train, n_neighbors=5, random_state=repeat_idx)
    X_bsmote, y_bsmote = apply_borderline_smote(X_train, y_train, random_state=repeat_idx)
    X_ros, y_ros = apply_ros(X_train, y_train, random_state=repeat_idx)

    print("    GAN training...")
    X_gan, y_gan = apply_gan(X_train, y_train, epochs=gan_epochs)
    print("    CGAN training...")
    X_cgan, y_cgan = apply_cgan(X_train, y_train, epochs=gan_epochs)
    print("    WGAN-GP training...")
    X_wgan, y_wgan = apply_wgan_gp(X_train, y_train, epochs=gan_epochs)
    print("    CWGAN-GP training...")
    X_cwgan, y_cwgan = apply_cwgan_gp(X_train, y_train, epochs=gan_epochs)

    methods = {
        "SMOTE": (X_smote, y_smote),
        "ADASYN": (X_adasyn, y_adasyn),
        "Borderline-SMOTE": (X_bsmote, y_bsmote),
        "ROS": (X_ros, y_ros),
        "GAN": (X_gan, y_gan),
        "CGAN": (X_cgan, y_cgan),
        "WGAN-GP": (X_wgan, y_wgan),
        "CWGAN-GP": (X_cwgan, y_cwgan)
    }

    for method_name, (X_res, y_res) in methods.items():
        results[method_name] = {}
        classifiers = get_classifiers()

        for clf_name, clf in classifiers.items():
            clf.fit(X_res, y_res)
            y_pred = clf.predict(X_test)
            y_scores = clf.predict_proba(X_test)[:, 1]
            metrics = evaluate_all_metrics(y_test, y_pred, y_scores)
            results[method_name][clf_name] = metrics

    return results


def average_results(all_repeat_results):
    collected = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for repeat_result in all_repeat_results:
        for method_name, clf_results in repeat_result.items():
            for clf_name, metrics in clf_results.items():
                for metric_name, value in metrics.items():
                    collected[method_name][clf_name][metric_name].append(value)

    averaged = {}
    for method_name, clf_dict in collected.items():
        averaged[method_name] = {}
        for clf_name, metric_dict in clf_dict.items():
            averaged[method_name][clf_name] = {}
            for metric_name, values in metric_dict.items():
                averaged[method_name][clf_name][metric_name] = (np.mean(values), np.std(values))
    return averaged


def compute_ranks(averaged, metric_name):
    method_names = list(averaged.keys())
    classifiers = list(next(iter(averaged.values())).keys())
    ranks = {}
    for clf_name in classifiers:
        scores = {m: averaged[m][clf_name][metric_name][0] for m in method_names}
        sorted_methods = sorted(scores.items(), key=lambda x: x[1])
        ranks[clf_name] = {}
        for rank_val, (method_name, _) in enumerate(sorted_methods, start=1):
            ranks[clf_name][method_name] = rank_val
    return ranks


def print_full_table(averaged):
    method_names = list(averaged.keys())
    classifiers = list(next(iter(averaged.values())).keys())

    for metric_name in ["f_measure", "g_mean", "auc"]:
        print(f"\n{'='*60}")
        print(f"Metric: {metric_name.upper()}  (mean over 5 repeats)")
        print(f"{'='*60}")

        for clf_name in classifiers:
            print(f"\n  {clf_name}:")
            for m in method_names:
                mean_val, std_val = averaged[m][clf_name][metric_name]
                print(f"    {m:20s}: {mean_val:.3f} ± {std_val:.3f}")

        ranks = compute_ranks(averaged, metric_name)
        print(f"\n  Mean Rank ({metric_name}, higher=better, max={len(method_names)}):")
        rank_summary = []
        for m in method_names:
            avg_rank = np.mean([ranks[clf][m] for clf in classifiers])
            rank_summary.append((m, avg_rank))
        rank_summary.sort(key=lambda x: -x[1])
        for m, avg_rank in rank_summary:
            print(f"    {m:20s}: {avg_rank:.2f}")


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "iris"

    if dataset_name == "iris":
        from data.load_iris import load_imbalanced_iris
        X, y = load_imbalanced_iris()
    elif dataset_name == "heart":
        from data.load_heart import load_imbalanced_heart
        X, y = load_imbalanced_heart()
    elif dataset_name == "haberman":
        from data.load_haberman import load_imbalanced_haberman
        X, y = load_imbalanced_haberman()
    elif dataset_name == "liver":
        from data.load_liver import load_imbalanced_liver
        X, y = load_imbalanced_liver()
    elif dataset_name == "lymphography":
        from data.load_lymphography import load_imbalanced_lymphography
        X, y = load_imbalanced_lymphography()
    elif dataset_name == "wine":
        from data.load_wine import load_imbalanced_wine
        X, y = load_imbalanced_wine()
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\n{'='*60}\nDATASET: {dataset_name.upper()}\n{'='*60}")

    splits = get_train_test_splits(X, y, n_repeats=5, test_size=0.2)

    all_repeat_results = []

    for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
        print(f"\n{'#'*60}")
        print(f"# REPEAT {i+1}/5")
        print(f"{'#'*60}")

        repeat_result = run_single_repeat(
            X_train, y_train, X_test, y_test,
            gan_epochs=5000, repeat_idx=i
        )
        all_repeat_results.append(repeat_result)

    print("\n\nSab 5 repeats complete. Ab average nikaal rahe hain...\n")

    averaged = average_results(all_repeat_results)
    print_full_table(averaged)

    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'results',
        f'full_comparison_8methods_{dataset_name.upper()}.csv'
    )
    save_averaged_results_csv(averaged, output_path)

    print("\n\n" + "#"*70)
    print("# PHASE 11: STATISTICAL ANALYSIS (Friedman + Nemenyi)")
    print("#"*70)
    run_statistical_analysis(all_repeat_results)