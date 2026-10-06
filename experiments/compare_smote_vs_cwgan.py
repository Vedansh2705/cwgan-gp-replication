# experiments/compare_smote_vs_cwgan.py
# =============================================================
# Final Comparison: SMOTE vs CWGAN-GP
# =============================================================
# Dono oversampling methods se data balance karke, sab 5
# classifiers par F-measure/G-mean/AUC calculate karte hain,
# aur side-by-side table mein dikhate hain.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.load_iris import load_imbalanced_iris
from experiments.train_test_split_demo import get_train_test_splits
from baselines.smote import apply_smote
from baselines.cwgan_gp_oversample import apply_cwgan_gp
from classifiers.classifiers import get_classifiers
from evaluation.metrics import evaluate_all_metrics


def run_comparison(X_train, y_train, X_test, y_test, cwgan_epochs=1000):
    """
    SMOTE aur CWGAN-GP dono se oversample karke, sab classifiers
    par evaluate karta hai. Results dictionary mein return karta hai.
    """
    results = {}

    # ---- Method 1: SMOTE ----
    X_smote, y_smote = apply_smote(X_train, y_train, k_neighbors=5, random_state=0)

    # ---- Method 2: CWGAN-GP ----
    print("CWGAN-GP training ho raha hai...")
    X_cwgan, y_cwgan = apply_cwgan_gp(X_train, y_train, epochs=cwgan_epochs)
    print("CWGAN-GP training complete.\n")

    methods = {
        "SMOTE": (X_smote, y_smote),
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


def print_comparison_table(results):
    """Results ko readable table mein print karta hai."""
    classifiers = list(next(iter(results.values())).keys())

    for metric_name in ["f_measure", "g_mean", "auc"]:
        print(f"\n{'='*55}")
        print(f"Metric: {metric_name.upper()}")
        print(f"{'='*55}")
        print(f"{'Classifier':10s} {'SMOTE':>10s} {'CWGAN-GP':>10s} {'Winner':>10s}")
        print("-" * 55)

        for clf_name in classifiers:
            smote_val = results["SMOTE"][clf_name][metric_name]
            cwgan_val = results["CWGAN-GP"][clf_name][metric_name]
            winner = "CWGAN-GP" if cwgan_val > smote_val else ("SMOTE" if smote_val > cwgan_val else "Tie")

            print(f"{clf_name:10s} {smote_val:>10.3f} {cwgan_val:>10.3f} {winner:>10s}")


if __name__ == "__main__":
    X, y = load_imbalanced_iris()
    splits = get_train_test_splits(X, y, n_repeats=5, test_size=0.2)

    # Abhi ke liye sirf pehla split use kar rahe hain
    # (Phase 9 mein hum sab 5 repeats ka average lenge)
    X_train, X_test, y_train, y_test = splits[0]

    results = run_comparison(X_train, y_train, X_test, y_test, cwgan_epochs=1000)
    print_comparison_table(results)