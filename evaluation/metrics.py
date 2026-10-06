# evaluation/metrics.py
# =============================================================
# Step 10: Evaluation Metrics (Paper Section 3.3, Eq. 12-13)
# =============================================================

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import numpy as np


def calculate_f_measure(y_true, y_pred, positive_label=1):
    return f1_score(y_true, y_pred, pos_label=positive_label)


def calculate_g_mean(y_true, y_pred, positive_label=1):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = np.sqrt(sensitivity * specificity)
    return g_mean


def calculate_auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)


def evaluate_all_metrics(y_true, y_pred, y_scores):
    return {
        "f_measure": calculate_f_measure(y_true, y_pred),
        "g_mean": calculate_g_mean(y_true, y_pred),
        "auc": calculate_auc(y_true, y_scores)
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from data.load_iris import load_imbalanced_iris
    from experiments.train_test_split_demo import get_train_test_splits
    from baselines.smote import apply_smote
    from classifiers.classifiers import get_classifiers

    X, y = load_imbalanced_iris()
    splits = get_train_test_splits(X, y, n_repeats=5, test_size=0.2)
    X_train, X_test, y_train, y_test = splits[0]

    X_resampled, y_resampled = apply_smote(X_train, y_train, k_neighbors=5, random_state=0)

    classifiers = get_classifiers()

    print("Classifier performance (SMOTE data par) - Paper ke metrics:\n")
    print(f"{'Classifier':10s} {'F-measure':>10s} {'G-mean':>10s} {'AUC':>10s}")
    print("-" * 45)

    for name, clf in classifiers.items():
        clf.fit(X_resampled, y_resampled)
        y_pred = clf.predict(X_test)
        y_scores = clf.predict_proba(X_test)[:, 1]

        metrics = evaluate_all_metrics(y_test, y_pred, y_scores)

        print(f"{name:10s} {metrics['f_measure']:>10.3f} "
              f"{metrics['g_mean']:>10.3f} {metrics['auc']:>10.3f}")