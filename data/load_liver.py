# data/load_liver.py
# =============================================================
# Dataset 4: Liver Disorders / BUPA (Paper Table 1, ID 2)
# =============================================================
# DOCUMENTED ISSUE: UCI ka original BUPA dataset mein "selector"
# column tha (7th column, train/test split marker), jo kai papers
# class label ki tarah reuse karte hain. ucimlrepo wrapper mein
# ye column missing hai - sirf 6 columns available hain (5 blood
# test features + drinks). Threshold search (neeche) ne confirm
# kiya ki koi bhi drinks-threshold paper ke exact 145/200 split
# se match nahi karta - sabse kareeb match hai threshold=3.0,
# jo 155/190 deta hai (IR=1.22 vs paper's 1.38). Ye ek
# UNAVOIDABLE, DOCUMENTED DEVIATION hai (Phase 1, gap #13) -
# paper apna binarization rule specify nahi karta.

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np


def load_imbalanced_liver_raw():
    """Raw features + drinks column (threshold lagane se pehle)."""
    liver = fetch_ucirepo(id=60)
    X_df = liver.data.features
    y_df = liver.data.targets

    combined = pd.concat([X_df, y_df], axis=1).dropna()
    feature_cols = ["mcv", "alkphos", "sgpt", "sgot", "gammagt"]

    X = combined[feature_cols].values.astype(np.float64)
    drinks = combined["drinks"].values.astype(np.float64)
    return X, drinks


def find_best_threshold():
    """
    Paper ka target: minority=145, majority=200.
    Different thresholds try karke dekhta hai kaunsa closest hai.
    """
    X, drinks = load_imbalanced_liver_raw()
    unique_vals = sorted(np.unique(drinks))

    print("Threshold search (target: minority=145, majority=200):\n")
    print(f"{'Threshold':>10s} {'Class>thresh':>14s} {'Class<=thresh':>15s}")
    print("-" * 45)

    best_threshold = None
    best_diff = float('inf')

    for t in unique_vals:
        n_above = (drinks > t).sum()
        n_below = (drinks <= t).sum()
        print(f"{t:>10.1f} {n_above:>14d} {n_below:>15d}")

        diff1 = abs(n_above - 145) + abs(n_below - 200)
        diff2 = abs(n_above - 200) + abs(n_below - 145)
        min_diff = min(diff1, diff2)

        if min_diff < best_diff:
            best_diff = min_diff
            best_threshold = t

    print(f"\nSabse kareeb threshold: {best_threshold} (total diff from paper's counts: {best_diff})")
    return best_threshold


def load_imbalanced_liver():
    """
    UCI Liver Disorders (ID=60), 'drinks' field ko threshold=3.0
    par binarize kiya (drinks > 3 -> class 1 / minority).

    DOCUMENTED DEVIATION: Paper ke exact 145/200 split se match
    nahi karta (sabse kareeb match: 155/190, IR=1.22 vs paper's
    IR=1.38). Paper apna exact binarization rule specify nahi
    karta, isliye ye gap unavoidable hai.

    Returns:
        X (np.ndarray): shape (345, 5)
        y (np.ndarray): shape (345,) - 1=minority(drinks>3), 0=majority
    """
    X, drinks = load_imbalanced_liver_raw()
    y = (drinks > 3.0).astype(int)
    return X, y


if __name__ == "__main__":
    X, y = load_imbalanced_liver()
    n_minority = (y == 1).sum()
    n_majority = (y == 0).sum()
    IR = n_majority / n_minority

    print("Total samples:", len(y))
    print("Feature count:", X.shape[1], "(paper says 6; selector column unavailable in ucimlrepo)")
    print(f"Minority: {n_minority} (paper says 145)")
    print(f"Majority: {n_majority} (paper says 200)")
    print(f"IR: {IR:.2f}   (paper's Table 1 value = 1.38)")